from __future__ import annotations

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def predict_sigma_grid(
    model: torch.nn.Module,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Predict hypotheses for every sigma.

    Returns a tensor with shape ``[num_sigmas, batch, hypotheses, target_dim]``.
    """

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    x = x.to(device)
    outputs = []
    for sigma in sigmas.to(device):
        sigma_batch = torch.full((x.shape[0],), float(sigma), device=device)
        outputs.append(model(x, sigma_batch).detach().cpu())
    return torch.stack(outputs, dim=0)


def align_hypothesis_trajectories(predictions: torch.Tensor) -> torch.Tensor:
    """Greedily align hypothesis indices across adjacent sigma levels."""

    aligned = predictions.detach().cpu().clone()
    num_sigmas, batch_size, _, _ = aligned.shape
    for sigma_index in range(1, num_sigmas):
        for batch_index in range(batch_size):
            previous = aligned[sigma_index - 1, batch_index].numpy()
            current = aligned[sigma_index, batch_index].numpy()
            cost = ((previous[:, None, :] - current[None, :, :]) ** 2).sum(axis=-1)
            _, column_index = linear_sum_assignment(cost)
            aligned[sigma_index, batch_index] = aligned[sigma_index, batch_index, column_index]
    return aligned


def cluster_hypotheses(
    hypotheses: torch.Tensor | np.ndarray,
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
    method: str = "ward",
) -> dict[str, np.ndarray]:
    """Hierarchically cluster one set of hypotheses."""

    array = hypotheses.detach().cpu().numpy() if isinstance(hypotheses, torch.Tensor) else hypotheses
    if array.shape[0] < 2:
        return {"labels": np.ones(array.shape[0], dtype=np.int64), "linkage": np.empty((0, 4))}
    tree = linkage(array, method=method)
    if n_clusters is not None:
        labels = fcluster(tree, t=n_clusters, criterion="maxclust")
    elif distance_threshold is not None:
        labels = fcluster(tree, t=distance_threshold, criterion="distance")
    else:
        labels = np.arange(1, array.shape[0] + 1)
    return {"labels": labels.astype(np.int64), "linkage": tree}
