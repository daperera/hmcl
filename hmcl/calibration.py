from __future__ import annotations

import torch

from hmcl.losses import squared_l2_distances


def min_distance(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Distance from each target to its nearest hypothesis."""

    return squared_l2_distances(predictions, target).sqrt().min(dim=1).values


def empirical_coverage(
    predictions: torch.Tensor,
    target: torch.Tensor,
    radius: float | torch.Tensor,
) -> torch.Tensor:
    """Fraction of targets within ``radius`` of any hypothesis."""

    nearest = min_distance(predictions, target)
    return (nearest <= radius).float().mean()


def min_distance_calibration_bins(
    predictions: torch.Tensor,
    target: torch.Tensor,
    sigmas: torch.Tensor,
    num_bins: int = 10,
) -> dict[str, torch.Tensor]:
    """Relate conditioning sigma to observed nearest-hypothesis error."""

    errors = min_distance(predictions, target)
    edges = torch.linspace(sigmas.min(), sigmas.max(), steps=num_bins + 1, device=sigmas.device)
    bin_error = torch.zeros(num_bins, device=sigmas.device)
    bin_count = torch.zeros(num_bins, device=sigmas.device)
    for index in range(num_bins):
        mask = (sigmas >= edges[index]) & (sigmas <= edges[index + 1])
        bin_count[index] = mask.float().sum()
        if mask.any():
            bin_error[index] = errors[mask].mean()
    return {"edges": edges, "mean_min_distance": bin_error, "count": bin_count}
