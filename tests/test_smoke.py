import torch

from hmcl.data import build_synthetic_dataset
from hmcl.losses import annealed_mcl_loss
from hmcl.models import NoiseConditionedMLP
from hmcl.noise import NoiseSchedule


def test_noise_conditioned_mcl_backward() -> None:
    dataset = build_synthetic_dataset("branching_curves", split="train", n_samples=16, seed=3)
    batch = torch.utils.data.default_collate([dataset[index] for index in range(16)])
    schedule = NoiseSchedule(sigma_min=0.02, sigma_max=0.5)
    model = NoiseConditionedMLP(
        input_dim=dataset.spec.input_dim,
        target_dim=dataset.spec.target_dim,
        num_hypotheses=5,
        hidden_dim=32,
        depth=1,
    )
    sigma = schedule.sample(16)
    predictions = model(batch["x"], sigma)
    loss, metrics = annealed_mcl_loss(predictions, batch["y"], sigma=sigma)
    loss.backward()

    assert predictions.shape == (16, 5, 2)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["assignment_entropy"])
