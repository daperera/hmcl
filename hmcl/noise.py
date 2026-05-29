from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NoiseSchedule:
    """Sample and enumerate explicit noise levels."""

    sigma_min: float = 0.01
    sigma_max: float = 1.0
    sigma_init: float = 0.5
    sigma_decay: float = 0.95
    distribution: str = "log_uniform"
    num_discrete: int | None = None

    def __post_init__(self) -> None:
        if self.sigma_min <= 0:
            raise ValueError("sigma_min must be positive because noise embeddings use log(sigma).")
        if self.sigma_max < self.sigma_min:
            raise ValueError("sigma_max must be greater than or equal to sigma_min.")
        valid = {"log_uniform", "uniform", "discrete", "exponential_decay"}
        if self.distribution not in valid:
            raise ValueError(f"Unknown distribution {self.distribution!r}; expected one of {valid}.")
        if self.distribution == "discrete" and (self.num_discrete is None or self.num_discrete < 2):
            raise ValueError("discrete schedules need num_discrete >= 2.")

    def sample(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        """Sample sigma as a vector with shape ``[batch_size]``."""

        if self.distribution == "uniform":
            u = torch.rand(batch_size, device=device)
            return self.sigma_min + u * (self.sigma_max - self.sigma_min)

        if self.distribution == "log_uniform":
            log_min = torch.log(torch.tensor(self.sigma_min, device=device))
            log_max = torch.log(torch.tensor(self.sigma_max, device=device))
            u = torch.rand(batch_size, device=device)
            return torch.exp(log_min + u * (log_max - log_min))

        grid = self.grid(self.num_discrete, device=device)
        index = torch.randint(0, grid.numel(), (batch_size,), device=device)
        return grid[index]

    def grid(self, num: int | None = None, device: torch.device | str | None = None) -> torch.Tensor:
        """Return an ordered grid from low to high sigma."""

        count = num or self.num_discrete or 32
        if self.distribution in {"log_uniform", "discrete"}:
            return torch.logspace(
                torch.log10(torch.tensor(self.sigma_min)).item(),
                torch.log10(torch.tensor(self.sigma_max)).item(),
                steps=count,
                device=device,
            )
        if self.distribution == "exponential_decay":
            return self.sigma_init * self.sigma_decay ** torch.arange(count).to(device)
        return torch.linspace(self.sigma_min, self.sigma_max, steps=count, device=device)


def sigma_to_temperature(
    sigma: torch.Tensor,
    scale: float = 1.0,
    power: float = 2.0,
    floor: float = 1e-4,
) -> torch.Tensor:
    """Map a conditioning noise level to an MCL assignment temperature."""

    return floor + scale * sigma.clamp_min(1e-8).pow(power)
