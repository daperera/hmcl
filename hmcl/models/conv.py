from __future__ import annotations

import torch
from torch import nn

from hmcl.models.embeddings import NoiseConditioning
from hmcl.models.mlp import FiLMResidualBlock


class NoiseConditionedConvClassifier(nn.Module):
    """Small image classifier that emits K class-logit hypotheses."""

    def __init__(
        self,
        image_channels: int = 1,
        num_classes: int = 10,
        num_hypotheses: int = 8,
        hidden_dim: int = 128,
        depth: int = 3,
        noise_embedding_dim: int = 64,
        noise_hidden_dim: int = 128,
        noise_kind: str = "fourier",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_hypotheses = num_hypotheses
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, hidden_dim),
            nn.SiLU(),
        )
        self.noise = NoiseConditioning(
            embedding_dim=noise_embedding_dim,
            hidden_dim=noise_hidden_dim,
            output_dim=hidden_dim,
            kind=noise_kind,
        )
        self.blocks = nn.ModuleList(
            FiLMResidualBlock(hidden_dim, hidden_dim, dropout=dropout) for _ in range(depth)
        )
        self.head = nn.Linear(hidden_dim, num_hypotheses * num_classes)

    def forward(self, image: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        conditioning = self.noise(sigma)
        h = self.encoder(image)
        for block in self.blocks:
            h = block(h, conditioning)
        logits = self.head(h)
        return logits.reshape(image.shape[0], self.num_hypotheses, self.num_classes)
