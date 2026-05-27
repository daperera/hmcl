from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from hmcl.noise import sigma_to_temperature


def _as_column_temperature(temperature: torch.Tensor | float, device: torch.device) -> torch.Tensor:
    if isinstance(temperature, torch.Tensor):
        return temperature.to(device=device).reshape(-1, 1).clamp_min(1e-8)
    return torch.tensor(float(temperature), device=device).reshape(1, 1).clamp_min(1e-8)


def squared_l2_distances(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return squared distances with shape ``[batch, hypotheses]``."""

    if target.ndim == 1:
        target = target[:, None]
    return (predictions - target[:, None, :]).pow(2).sum(dim=-1)


def annealed_mcl_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor | None = None,
    temperature: torch.Tensor | float | None = None,
    assignment: str = "softmin",
    temperature_scale: float = 1.0,
    temperature_power: float = 2.0,
    temperature_floor: float = 1e-4,
    detach_responsibilities: bool = True,
    epsilon: float = 0.05,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """Compute a noise-conditioned MCL regression loss."""

    distances = squared_l2_distances(predictions, target)

    if assignment == "wta":
        winner_distance, winner = distances.min(dim=1)
        loss = winner_distance.mean()
        responsibilities = F.one_hot(winner, num_classes=distances.shape[1]).to(distances.dtype)
    else:
        if temperature is None:
            if sigma is None:
                raise ValueError("Provide either sigma or temperature for annealed assignments.")
            temperature = sigma_to_temperature(
                sigma,
                scale=temperature_scale,
                power=temperature_power,
                floor=temperature_floor,
            )
        temp = _as_column_temperature(temperature, device=distances.device)
        soft = torch.softmax(-distances / temp, dim=1)

        if assignment == "softmin":
            responsibilities = soft.detach() if detach_responsibilities else soft
        elif assignment == "epsilon_wta":
            winner = distances.argmin(dim=1)
            hard = F.one_hot(winner, num_classes=distances.shape[1]).to(distances.dtype)
            responsibilities = (1.0 - epsilon) * hard + epsilon * soft
            responsibilities = responsibilities.detach() if detach_responsibilities else responsibilities
        else:
            raise ValueError("assignment must be one of 'softmin', 'epsilon_wta', or 'wta'.")

        loss = (responsibilities * distances).sum(dim=1).mean()

    min_distance, winner = distances.min(dim=1)
    counts = torch.bincount(winner, minlength=distances.shape[1]).to(predictions.dtype)
    counts = counts / counts.sum().clamp_min(1.0)
    entropy = -(responsibilities.clamp_min(1e-8) * responsibilities.clamp_min(1e-8).log()).sum(
        dim=1
    )
    metrics = {
        "loss": loss.detach(),
        "min_distance": min_distance.mean().detach(),
        "assignment_entropy": entropy.mean().detach(),
        "winner_usage_min": counts.min().detach(),
        "winner_usage_max": counts.max().detach(),
    }
    return loss, metrics


def mcl_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor | None = None,
    temperature: torch.Tensor | float | None = None,
    assignment: str = "softmin",
    temperature_scale: float = 1.0,
    temperature_power: float = 2.0,
    temperature_floor: float = 1e-4,
    detach_responsibilities: bool = True,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """MCL loss for K class-logit hypotheses with shape ``[batch, K, classes]``."""

    batch, hypotheses, classes = logits.shape
    flat_logits = logits.reshape(batch * hypotheses, classes)
    expanded_target = target[:, None].expand(batch, hypotheses).reshape(batch * hypotheses)
    ce = F.cross_entropy(flat_logits, expanded_target, reduction="none").reshape(batch, hypotheses)

    if assignment == "wta":
        winner_loss, winner = ce.min(dim=1)
        responsibilities = F.one_hot(winner, hypotheses).to(ce.dtype)
        loss = winner_loss.mean()
    else:
        if temperature is None:
            if sigma is None:
                raise ValueError("Provide either sigma or temperature for annealed assignments.")
            temperature = sigma_to_temperature(
                sigma,
                scale=temperature_scale,
                power=temperature_power,
                floor=temperature_floor,
            )
        temp = _as_column_temperature(temperature, device=logits.device)
        responsibilities = torch.softmax(-ce / temp, dim=1)
        if detach_responsibilities:
            responsibilities = responsibilities.detach()
        loss = (responsibilities * ce).sum(dim=1).mean()
        winner = ce.argmin(dim=1)

    entropy = -(responsibilities.clamp_min(1e-8) * responsibilities.clamp_min(1e-8).log()).sum(
        dim=1
    )
    usage = torch.bincount(winner, minlength=hypotheses).float() / max(float(winner.numel()), 1.0)
    metrics = {
        "loss": loss.detach(),
        "min_ce": ce.min(dim=1).values.mean().detach(),
        "assignment_entropy": entropy.mean().detach(),
        "winner_usage_max": usage.max().detach(),
    }
    return loss, metrics
