from __future__ import annotations

from typing import Any

from hmcl.data.synthetic import (
    BranchingCurveDataset,
    ConcentricArcDataset,
    HierarchicalGaussianDataset,
    build_synthetic_dataset,
)
from hmcl.data.uci import (
    UCIRegressionDataset,
    available_uci_datasets,
    build_uci_dataset,
    canonical_uci_name,
    is_uci_dataset,
)


def build_dataset(name: str, split: str, source: str | None = None, **kwargs: Any):
    if source is None:
        source = "uci" if is_uci_dataset(name) else "synthetic"
    source = source.lower()
    if source == "uci":
        return build_uci_dataset(name, split=split, **kwargs)
    if source == "synthetic":
        return build_synthetic_dataset(name, split=split, **kwargs)
    raise ValueError("source must be 'synthetic', 'uci', or omitted for name-based inference.")


__all__ = [
    "BranchingCurveDataset",
    "ConcentricArcDataset",
    "HierarchicalGaussianDataset",
    "UCIRegressionDataset",
    "available_uci_datasets",
    "build_dataset",
    "build_synthetic_dataset",
    "build_uci_dataset",
    "canonical_uci_name",
    "is_uci_dataset",
]
