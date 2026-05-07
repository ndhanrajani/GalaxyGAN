"""Conditional DCGAN-style generator and critic for WGAN-GP (RGB 69×69)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Generator(nn.Module):
    """Maps noise + class embedding to RGB in ``[-1, 1]`` at ``out_size`` (default 69)."""

    def __init__(
        self,
        z_dim: int = 128,
        num_classes: int = 10,
        embed_dim: int = 32,
        ngf: int = 64,
        out_size: int = 69,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.out_size = out_size
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Linear(z_dim + embed_dim, ngf * 8 * 4 * 4)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z: Tensor, y: Tensor) -> Tensor:
        e = self.embed(y)
        x = torch.cat([z, e], dim=1)
        x = self.fc(x).view(z.size(0), -1, 4, 4)
        x = self.conv(x)
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return x


class Critic(nn.Module):
    """WGAN critic conditioned by broadcasting class embeddings on the image grid."""

    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 32,
        ndf: int = 64,
        img_channels: int = 3,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        ch = img_channels + embed_dim
        self.main = nn.Sequential(
            nn.Conv2d(ch, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        e = self.embed(y).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, e], dim=1)
        return self.main(x).view(x.size(0), 1)


def gradient_penalty(critic: Critic, real: Tensor, fake: Tensor, y: Tensor) -> Tensor:
    b = real.size(0)
    device = real.device
    eps = torch.rand(b, 1, 1, 1, device=device, dtype=real.dtype)
    interp = eps * real + (1.0 - eps) * fake
    interp = interp.detach().requires_grad_(True)
    out = critic(interp, y)
    grad = torch.autograd.grad(
        outputs=out,
        inputs=interp,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gn = grad.flatten(1).norm(2, dim=1)
    return ((gn - 1.0) ** 2).mean()
