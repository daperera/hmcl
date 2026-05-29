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

    def forward(self, x: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 1:
            x = x[:, None]
        if sigma is None:
            sigma = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        conditioning = self.noise(sigma)
        h = self.stem(x)
        for block in self.blocks:
            h = block(h, conditioning)
        out = self.head(h)
        return out.reshape(x.shape[0], self.num_hypotheses, self.target_dim)


class BaseMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        num_hypotheses: int = 5,
        hidden_dim: int = 50,
        depth: int = 1,
        detach_scores: bool = True,
    ) -> None:
        super().__init__()

        self.num_hypotheses = num_hypotheses
        self.target_dim = target_dim
        self.detach_scores = detach_scores

        layers = [nn.Linear(input_dim + 1, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

        self.backbone = nn.Sequential(*layers)
        self.hypotheses = nn.Linear(hidden_dim, num_hypotheses * target_dim)
        self.scores = nn.Sequential(nn.Linear(hidden_dim, num_hypotheses), nn.Sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sigma is None:
            sigma = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        x = torch.cat([x, sigma[:, None]], dim=-1)
        h = self.backbone(x)

        hypotheses = self.hypotheses(h).view(
            x.shape[0], self.num_hypotheses, self.target_dim
        )

        scores = self.scores(h.detach()) if self.detach_scores else self.scores(h)

        return hypotheses, scores

class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        num_hypotheses: int = 5,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_hypotheses = num_hypotheses
        self.target_dim = target_dim

        self.stem = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            for _ in range(depth)
        )

        self.hypotheses = nn.ModuleList(
            [nn.Linear(hidden_dim, target_dim) for _ in range(num_hypotheses)]
        )

        self.scores = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_hypotheses)]
        )

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sigma is None:
            sigma = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        h = self.stem(torch.cat([x, sigma[:, None]], dim=-1))

        for block in self.blocks:
            h = h + block(h)

        predictions = torch.stack(
            [head(h) for head in self.hypotheses],
            dim=1,
        )

        scores = torch.stack(
            [torch.sigmoid(head(h.detach())) for head in self.scores],
            dim=1,
        ).squeeze(-1)

        return predictions, scores