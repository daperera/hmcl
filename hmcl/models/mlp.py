from __future__ import annotations

import torch
from torch import nn

from hmcl.models.embeddings import NoiseConditioning


class FiLMResidualBlock(nn.Module):
    """Residual MLP block modulated by a noise embedding."""

    def __init__(self, hidden_dim: int, conditioning_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.film = nn.Linear(conditioning_dim, 2 * hidden_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(conditioning).chunk(2, dim=-1)
        h = self.linear(self.norm(x))
        h = h * (1.0 + gamma) + beta
        h = self.dropout(self.activation(h))
        return x + h


class NoiseConditionedMLP(nn.Module):
    """Multi-hypothesis regressor conditioned on an explicit noise level."""

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        num_hypotheses: int = 8,
        hidden_dim: int = 128,
        depth: int = 4,
        noise_embedding_dim: int = 64,
        noise_hidden_dim: int = 128,
        noise_kind: str = "fourier",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.num_hypotheses = num_hypotheses

        self.noise = NoiseConditioning(
            embedding_dim=noise_embedding_dim,
            hidden_dim=noise_hidden_dim,
            output_dim=hidden_dim,
            kind=noise_kind,
        )
        self.stem = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.blocks = nn.ModuleList(
            FiLMResidualBlock(hidden_dim, hidden_dim, dropout=dropout) for _ in range(depth)
        )
        self.head = nn.Linear(hidden_dim, num_hypotheses * target_dim)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x[:, None]
        conditioning = self.noise(sigma)
        h = self.stem(x)
        for block in self.blocks:
            h = block(h, conditioning)
        out = self.head(h)
        return out.reshape(x.shape[0], self.num_hypotheses, self.target_dim)
