from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def _rng(seed: int, split: str) -> np.random.Generator:
    offset = {"train": 0, "val": 10_000, "test": 20_000}.get(split, 30_000)
    return np.random.default_rng(seed + offset)


@dataclass(frozen=True)
class SyntheticBatchSpec:
    input_dim: int
    target_dim: int
    num_modes: int


class BranchingCurveDataset(Dataset):
    """One-dimensional context with several plausible two-dimensional outcomes."""

    spec = SyntheticBatchSpec(input_dim=1, target_dim=2, num_modes=4)

    def __init__(
        self,
        n_samples: int = 10_000,
        split: str = "train",
        seed: int = 0,
        num_branches: int = 4,
        observation_noise: float = 0.03,
    ) -> None:
        super().__init__()
        generator = _rng(seed, split)
        x = generator.uniform(-1.0, 1.0, size=(n_samples, 1)).astype("float32")
        branch = generator.integers(0, num_branches, size=n_samples)
        offsets = np.linspace(-0.9, 0.9, num_branches).astype("float32")
        phases = np.linspace(0.0, np.pi, num_branches, endpoint=False).astype("float32")
        y0 = x[:, 0]
        y1 = 0.55 * np.sin(2.5 * np.pi * x[:, 0] + phases[branch]) + offsets[branch]
        y = np.stack([y0, y1], axis=-1).astype("float32")
        y += generator.normal(scale=observation_noise, size=y.shape).astype("float32")

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.mode = torch.from_numpy(branch.astype("int64"))
        self.spec = SyntheticBatchSpec(input_dim=1, target_dim=2, num_modes=num_branches)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[index], "y": self.y[index], "mode": self.mode[index]}


class ConcentricArcDataset(Dataset):
    """Same angle context, multiple plausible radii."""

    spec = SyntheticBatchSpec(input_dim=1, target_dim=2, num_modes=4)

    def __init__(
        self,
        n_samples: int = 10_000,
        split: str = "train",
        seed: int = 0,
        num_rings: int = 4,
        observation_noise: float = 0.025,
    ) -> None:
        super().__init__()
        generator = _rng(seed, split)
        theta = generator.uniform(-np.pi, np.pi, size=n_samples).astype("float32")
        mode = generator.integers(0, num_rings, size=n_samples)
        radii = np.linspace(0.5, 1.4, num_rings).astype("float32")
        radius = radii[mode]
        y = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=-1).astype("float32")
        y += generator.normal(scale=observation_noise, size=y.shape).astype("float32")

        self.x = torch.from_numpy((theta / np.pi).reshape(-1, 1).astype("float32"))
        self.y = torch.from_numpy(y)
        self.mode = torch.from_numpy(mode.astype("int64"))
        self.spec = SyntheticBatchSpec(input_dim=1, target_dim=2, num_modes=num_rings)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[index], "y": self.y[index], "mode": self.mode[index]}


class HierarchicalGaussianDataset(Dataset):
    """Unconditional mixture with explicit parent-child cluster structure."""

    spec = SyntheticBatchSpec(input_dim=1, target_dim=2, num_modes=8)

    def __init__(
        self,
        n_samples: int = 10_000,
        split: str = "train",
        seed: int = 0,
        observation_noise: float = 0.06,
        input_dim: int = 1,
        target_dim: int = 2,
        num_modes: int = 7,
        num_parents: int = 3,
        parent_scale: float = 1.4,
        child_scale: float = 0.28,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be at least 1.")
        if target_dim < 1:
            raise ValueError("target_dim must be at least 1.")
        if num_modes < 1:
            raise ValueError("num_modes must be at least 1.")
        if num_parents < 1:
            raise ValueError("num_parents must be at least 1.")

        generator = _rng(seed, split)
        structure_generator = np.random.default_rng(seed)
        num_parents = min(num_parents, num_modes)

        parent_centers = structure_generator.normal(
            loc=0.0,
            scale=parent_scale,
            size=(num_parents, target_dim),
        ).astype("float32")
        parent_ids = np.arange(num_modes, dtype="int64") % num_parents
        child_offsets = structure_generator.normal(
            loc=0.0,
            scale=child_scale,
            size=(num_modes, target_dim),
        ).astype("float32")
        centers = parent_centers[parent_ids] + child_offsets
        mode = generator.integers(0, centers.shape[0], size=n_samples)
        y = centers[mode] + generator.normal(scale=observation_noise, size=(n_samples, target_dim))

        self.x = torch.zeros(n_samples, input_dim, dtype=torch.float32)
        self.y = torch.from_numpy(y.astype("float32"))
        self.mode = torch.from_numpy(mode.astype("int64"))
        self.parent = torch.tensor(parent_ids, dtype=torch.long)[self.mode]
        self.centers = torch.from_numpy(centers)
        self.parent_centers = torch.from_numpy(parent_centers)
        self.spec = SyntheticBatchSpec(
            input_dim=input_dim,
            target_dim=target_dim,
            num_modes=centers.shape[0],
        )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "y": self.y[index],
            "mode": self.mode[index],
            "parent": self.parent[index],
        }

class OneGaussianDataset(Dataset):
    """Unconditional mixture with explicit parent-child cluster structure."""

    spec = SyntheticBatchSpec(input_dim=1, target_dim=2, num_modes=8)

    def __init__(
        self,
        n_samples: int = 10_000,
        split: str = "train",
        seed: int = 0,
        observation_noise: float = 0.06,

    ) -> None:
        super().__init__()
        input_dim, target_dim = 1, 2

        generator = _rng(seed, split)
        centers = np.array([[0.0, 0.0]])
        mode = generator.integers(0, centers.shape[0], size=n_samples)
        y = centers[mode] + generator.normal(scale=observation_noise, size=(n_samples, target_dim))

        self.x = torch.zeros(n_samples, input_dim, dtype=torch.float32)
        self.y = torch.from_numpy(y.astype("float32"))
        self.mode = torch.from_numpy(mode.astype("int64"))
        self.spec = SyntheticBatchSpec(
            input_dim=input_dim,
            target_dim=target_dim,
            num_modes=centers.shape[0],
        )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "y": self.y[index],
            "mode": self.mode[index],
        }

class RegularPolygonGaussianDataset(Dataset):
    """
    Unconditional mixture of N Gaussians placed on a regular polygon.

    The polygon radius is chosen such that the centroid distribution
    has variance 1:
        E[||c||^2] = 1

    In 2D this means radius = 1.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        split: str = "train",
        seed: int = 0,
        num_modes: int = 8,
        observation_noise: float = 0.06,
    ) -> None:
        super().__init__()

        input_dim = 1
        target_dim = 2

        generator = _rng(seed, split)

        # Regular polygon on the unit circle
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            num_modes,
            endpoint=False,
        )

        radius = 1.0

        centers = radius * np.stack(
            [
                np.cos(angles),
                np.sin(angles),
            ],
            axis=1,
        )

        mode = generator.integers(
            0,
            num_modes,
            size=n_samples,
        )

        y = (
            centers[mode]
            + generator.normal(
                scale=observation_noise,
                size=(n_samples, target_dim),
            )
        )

        self.x = torch.zeros(
            n_samples,
            input_dim,
            dtype=torch.float32,
        )

        self.y = torch.from_numpy(
            y.astype("float32")
        )

        self.mode = torch.from_numpy(
            mode.astype("int64")
        )

        self.centers = torch.from_numpy(
            centers.astype("float32")
        )

        self.spec = SyntheticBatchSpec(
            input_dim=input_dim,
            target_dim=target_dim,
            num_modes=num_modes,
        )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "y": self.y[index],
            "mode": self.mode[index],
        }

class HierarchicalRegularPolygonGaussianDataset(Dataset):
    """
    Hierarchical Gaussian mixture built recursively from regular polygons.

    Level 0:
        one centroid at the origin

    Level 1:
        regular polygon around the origin

    Level 2:
        regular polygon around each level-1 centroid

    ...

    This creates multiple spatial scales and therefore multiple
    annealing phase transitions.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        split: str = "train",
        seed: int = 0,
        branching_factor: int = 3,
        depth: int = 3,
        scale_decay: float = 0.35,
        observation_noise: float = 0.04,
    ) -> None:
        super().__init__()

        input_dim = 1
        target_dim = 2

        generator = _rng(seed, split)

        #
        # Build hierarchy of centroids
        #

        centers = [np.zeros(2, dtype=np.float32)]

        for level in range(depth):

            radius = scale_decay**level

            angles = np.linspace(
                0.0,
                2.0 * np.pi,
                branching_factor,
                endpoint=False,
            )

            polygon_offsets = radius * np.stack(
                [
                    np.cos(angles),
                    np.sin(angles),
                ],
                axis=1,
            )

            new_centers = []

            for parent in centers:
                for offset in polygon_offsets:
                    new_centers.append(parent + offset)

            centers = new_centers

        centers = np.asarray(centers, dtype=np.float32)

        num_modes = centers.shape[0]

        #
        # Sample modes
        #

        mode = generator.integers(
            0,
            num_modes,
            size=n_samples,
        )

        y = (
            centers[mode]
            + generator.normal(
                scale=observation_noise,
                size=(n_samples, target_dim),
            )
        )

        self.x = torch.zeros(
            n_samples,
            input_dim,
            dtype=torch.float32,
        )

        self.y = torch.from_numpy(
            y.astype("float32")
        )

        self.mode = torch.from_numpy(
            mode.astype("int64")
        )

        self.centers = torch.from_numpy(
            centers.astype("float32")
        )

        self.spec = SyntheticBatchSpec(
            input_dim=input_dim,
            target_dim=target_dim,
            num_modes=num_modes,
        )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "y": self.y[index],
            "mode": self.mode[index],
        }

def build_synthetic_dataset(name: str, split: str, **kwargs: object) -> Dataset:
    registry = {
        "branching_curves": BranchingCurveDataset,
        "concentric_arcs": ConcentricArcDataset,
        "hierarchical_gaussians": HierarchicalGaussianDataset,
        "one_gaussian": OneGaussianDataset,
        "regular_polygon_gaussians": RegularPolygonGaussianDataset,
        "hierarchical_polygon_gaussians": HierarchicalRegularPolygonGaussianDataset,
    }
    try:
        dataset_cls = registry[name]
    except KeyError as exc:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown synthetic dataset {name!r}. Available: {available}.") from exc
    return dataset_cls(split=split, **kwargs)
