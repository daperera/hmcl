from __future__ import annotations

import math

import torch
from torch import nn


class LogNoiseEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, sigma):
        sigma = sigma.reshape(-1, 1).clamp_min(1e-8)
        log_sigma = torch.log(sigma)

        return log_sigma.repeat(1, self.dim)


class SmoothNoiseEmbedding(nn.Module):
    """
    Smooth non-periodic features for continuation-style conditioning.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, sigma):
        sigma = sigma.reshape(-1, 1).clamp_min(1e-8)

        log_sigma = torch.log(sigma)

        features = torch.cat(
            [
                log_sigma,
                sigma,
                sigma**2,
                1.0 / sigma,
            ],
            dim=-1,
        )

        repeats = (self.dim + features.shape[1] - 1) // features.shape[1]
        features = features.repeat(1, repeats)

        return features[:, : self.dim]


class SinusoidalNoiseEmbedding(nn.Module):
    """Transformer/DDPM-style sinusoidal features of log sigma."""

    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("dim must be at least 2.")
        self.dim = dim
        self.max_period = max_period

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.reshape(-1).clamp_min(1e-8)
        half = self.dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            half, device=sigma.device, dtype=sigma.dtype
        )
        freqs = torch.exp(exponent / max(half - 1, 1))
        args = torch.log(sigma)[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class GaussianFourierNoiseEmbedding(nn.Module):
    """Random Fourier features of log sigma, common in score/diffusion models."""

    def __init__(self, dim: int, scale: float = 16.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("GaussianFourierNoiseEmbedding dim must be even.")
        weight = torch.randn(dim // 2) * scale
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.reshape(-1).clamp_min(1e-8)
        projected = torch.log(sigma)[:, None] * self.weight[None, :] * 2 * math.pi
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class NoiseConditioning(nn.Module):
    """Embed sigma and pass it through a small MLP."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        kind: str = "fourier",
        fourier_scale: float = 16.0,
    ) -> None:
        super().__init__()
        if kind == "fourier":
            base: nn.Module = GaussianFourierNoiseEmbedding(embedding_dim, scale=fourier_scale)
        elif kind == "sinusoidal":
            base = SinusoidalNoiseEmbedding(embedding_dim)
        elif kind == "log":
            base = LogNoiseEmbedding(embedding_dim)
        elif kind == "smooth":
            base = SmoothNoiseEmbedding(embedding_dim)
        else:
            raise ValueError("Noise embedding kind must be 'fourier' or 'sinusoidal'.")

        self.net = nn.Sequential(
            base,
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.net(sigma)
