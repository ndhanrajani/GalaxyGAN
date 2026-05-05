"""Minimal Gradio demo for class-conditional GalaxyGAN generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from galaxygan.data.galaxy10 import CLASS_NAMES
from galaxygan.models.cwgan import Generator


def load_generator(checkpoint_path: Path, device: torch.device) -> tuple[Generator, int]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    z_dim = int(train_args.get("z_dim", 128))
    image_size = int(train_args.get("image_size", 69))

    model = Generator(z_dim=z_dim, out_size=image_size).to(device)
    model.load_state_dict(ckpt["G"])
    model.eval()
    return model, z_dim


def make_app(generator: Generator, z_dim: int, device: torch.device) -> gr.Blocks:
    @torch.no_grad()
    def generate(class_name: str, seed: int) -> np.ndarray:
        class_idx = CLASS_NAMES.index(class_name)
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        z = torch.randn(1, z_dim, generator=g, device=device)
        y = torch.tensor([class_idx], dtype=torch.long, device=device)
        out = generator(z, y)[0]
        img = ((out.clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        return img

    with gr.Blocks(title="GalaxyGAN Demo") as demo:
        gr.Markdown("## GalaxyGAN Conditional Generator Demo")
        gr.Markdown("Select a galaxy morphology class and random seed.")
        with gr.Row():
            class_input = gr.Dropdown(choices=list(CLASS_NAMES), value=CLASS_NAMES[0], label="Morphology class")
            seed_input = gr.Number(value=42, precision=0, label="Seed")
        output = gr.Image(type="numpy", label="Generated image")
        run_btn = gr.Button("Generate")
        run_btn.click(generate, inputs=[class_input, seed_input], outputs=output)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint_epochXXXX.pt")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, z_dim = load_generator(checkpoint_path, device)
    app = make_app(generator, z_dim, device)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
