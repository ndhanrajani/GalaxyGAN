"""Train conditional WGAN-GP on Galaxy10 DECaLS (69×69)."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from galaxygan.data.galaxy10 import Galaxy10DECaLS
from galaxygan.models.cwgan import Critic, Generator, gradient_penalty


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def save_samples(
    G: Generator,
    path: Path,
    device: torch.device,
    z_dim: int,
    classes: list[int],
) -> None:
    from torchvision.utils import save_image

    G.eval()
    tiles: list[torch.Tensor] = []
    for c in classes:
        z = torch.randn(1, z_dim, device=device)
        y = torch.tensor([c], device=device, dtype=torch.long)
        tiles.append(G(z, y))
    G.train()
    grid = torch.cat(tiles, dim=0)
    save_image((grid + 1) * 0.5, path, nrow=len(classes))


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        # Fixed 69×69 conv sizes: autotune helps; avoids leaving GPU idle while NFS was the bottleneck.
        torch.backends.cudnn.benchmark = True

    if args.preload:
        print(
            "Preloading training images into RAM (one pass from HDF5; then epochs avoid NFS).",
            flush=True,
        )
    train_ds = Galaxy10DECaLS(split="train", target_size=args.image_size, preload=args.preload)
    if args.preload:
        print("Preload finished.", flush=True)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    G = Generator(z_dim=args.z_dim, out_size=args.image_size).to(device)
    D = Critic().to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        ckpt_obj = torch.load(resume_path, map_location=device, weights_only=False)
        G.load_state_dict(ckpt_obj["G"])
        D.load_state_dict(ckpt_obj["D"])
        done_epoch = int(ckpt_obj["epoch"])
        start_epoch = done_epoch + 1
        print(
            f"Resumed from {resume_path} (finished epoch {done_epoch}); "
            f"training epochs {start_epoch}..{args.epochs}",
            flush=True,
        )

    if start_epoch > args.epochs:
        print(f"Nothing to do: last checkpoint epoch {start_epoch - 1} >= --epochs {args.epochs}.", flush=True)
        train_ds.close()
        return

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    n_batches = len(loader)
    global_step = (start_epoch - 1) * n_batches

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch = time.perf_counter()
        for real, y in loader:
            real = real.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            b = real.size(0)

            # ----- Critic -----
            loss_d_acc = 0.0
            gp_acc = 0.0
            for _ in range(args.n_critic):
                z = torch.randn(b, args.z_dim, device=device)
                with torch.no_grad():
                    fake = G(z, y)
                d_real = D(real, y).mean()
                d_fake = D(fake, y).mean()
                gp = gradient_penalty(D, real, fake, y)
                loss_d = d_fake - d_real + args.lambda_gp * gp
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
                loss_d_acc += float(loss_d.detach())
                gp_acc += float(gp.detach())

            loss_d_mean = loss_d_acc / args.n_critic
            gp_mean = gp_acc / args.n_critic

            # ----- Generator -----
            z = torch.randn(b, args.z_dim, device=device)
            gen = G(z, y)
            loss_g = -D(gen, y).mean()
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            if wandb_run is not None and global_step % args.log_every == 0:
                wandb.log(
                    {
                        "loss/g": float(loss_g.detach()),
                        "loss/d": loss_d_mean,
                        "gp": gp_mean,
                        "step": global_step,
                        "epoch": epoch,
                    }
                )
            global_step += 1

        ckpt = {
            "G": G.state_dict(),
            "D": D.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }
        torch.save(ckpt, out / f"checkpoint_epoch{epoch:04d}.pt")
        save_samples(G, out / f"samples_epoch{epoch:04d}.png", device, args.z_dim, list(range(10)))
        epoch_sec = time.perf_counter() - t_epoch
        print(
            f"epoch {epoch}/{args.epochs} completed in {epoch_sec:.1f}s ({epoch_sec / 60.0:.2f} min)",
            flush=True,
        )

        if wandb_run is not None:
            wandb.log({"epoch": epoch}, step=global_step)

    train_ds.close()
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--lambda-gp", type=float, default=10.0)
    p.add_argument("--image-size", type=int, default=69)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="runs/cwgan_gp")
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint_epochXXXX.pt; loads G/D and continues from the next epoch.",
    )
    p.add_argument(
        "--preload",
        action="store_true",
        help="Load the full train split into RAM once (~1–2 GB at 69²); much faster on slow/shared storage.",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="galaxygan")
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
