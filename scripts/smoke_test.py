from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hmcl.data import build_synthetic_dataset
from hmcl.losses import annealed_mcl_loss
from hmcl.models import NoiseConditionedMLP
from hmcl.noise import NoiseSchedule


def main() -> None:
    dataset = build_synthetic_dataset(
        "branching_curves",
        split="train",
        n_samples=64,
        seed=123,
        observation_noise=0.02,
        num_branches=4,
    )
    batch = torch.utils.data.default_collate([dataset[index] for index in range(32)])
    schedule = NoiseSchedule(sigma_min=0.02, sigma_max=1.0, distribution="log_uniform")
    model = NoiseConditionedMLP(
        input_dim=dataset.spec.input_dim,
        target_dim=dataset.spec.target_dim,
        num_hypotheses=6,
        hidden_dim=64,
        depth=2,
    )
    sigma = schedule.sample(batch["x"].shape[0])
    predictions = model(batch["x"], sigma)
    loss, metrics = annealed_mcl_loss(predictions, batch["y"], sigma=sigma)
    loss.backward()
    print(
        "ok",
        {
            "predictions": tuple(predictions.shape),
            "loss": round(float(loss), 4),
            "entropy": round(float(metrics["assignment_entropy"]), 4),
        },
    )


if __name__ == "__main__":
    main()
