from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


UCI_DATASET_ALIASES = {
    "boston": "bostonHousing",
    "boston_housing": "bostonHousing",
    "bostonHousing": "bostonHousing",
    "concrete": "concrete",
    "energy": "energy",
    "kin8nm": "kin8nm",
    "naval": "naval-propulsion-plant",
    "naval_propulsion_plant": "naval-propulsion-plant",
    "naval-propulsion-plant": "naval-propulsion-plant",
    "power": "power-plant",
    "power_plant": "power-plant",
    "power-plant": "power-plant",
    "protein": "protein-tertiary-structure",
    "protein_tertiary_structure": "protein-tertiary-structure",
    "protein-tertiary-structure": "protein-tertiary-structure",
    "wine": "wine-quality-red",
    "wine_quality_red": "wine-quality-red",
    "wine-quality-red": "wine-quality-red",
    "yacht": "yacht",
    "year": "YearPredictionMSD",
    "year_prediction_msd": "YearPredictionMSD",
    "YearPredictionMSD": "YearPredictionMSD",
}


@dataclass(frozen=True)
class UCIBatchSpec:
    input_dim: int
    target_dim: int
    num_modes: int = 1


def canonical_uci_name(name: str) -> str:
    try:
        return UCI_DATASET_ALIASES[name]
    except KeyError as exc:
        available = ", ".join(sorted(UCI_DATASET_ALIASES))
        raise ValueError(f"Unknown UCI dataset {name!r}. Available aliases: {available}.") from exc


def is_uci_dataset(name: str) -> bool:
    return name in UCI_DATASET_ALIASES


def available_uci_datasets() -> list[str]:
    return sorted(set(UCI_DATASET_ALIASES.values()))


def _load_index(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64)
    return np.atleast_1d(values).astype(np.int64)


def _as_2d_target(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values.reshape(-1, 1)
    return values


class UCIRegressionDataset(Dataset):
    """UCI regression benchmark split used by PBP, deep ensembles, and aMCL codebases."""

    def __init__(
        self,
        name: str,
        split: str = "train",
        data_root: str | Path = "data/uci",
        split_num: int = 0,
        train_ratio: float = 0.8,
        seed: int = 1,
        normalize_x: bool = True,
        normalize_y: bool = True,
        one_hot_encoding: bool = False,
        y_dim: int = 1,
        **_: Any,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        self.requested_name = name
        self.name = canonical_uci_name(name)
        self.split = split
        self.split_num = int(split_num)
        self.train_ratio = float(train_ratio)
        self.normalize_x = bool(normalize_x)
        self.normalize_y = bool(normalize_y)
        self.one_hot_encoding = bool(one_hot_encoding)

        data_dir = Path(data_root) / self.name / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"UCI dataset directory not found: {data_dir}")
        self.data_dir = data_dir

        data = np.loadtxt(data_dir / "data.txt")
        feature_indices = _load_index(data_dir / "index_features.txt")
        target_indices = _load_index(data_dir / "index_target.txt")
        x_all = data[:, feature_indices].astype(np.float32)
        y_all = _as_2d_target(data[:, target_indices].astype(np.float32))

        x_all, self.dim_cat = self._preprocess_feature_set(
            x=x_all,
            dataset_name=self.name,
            one_hot_encoding=self.one_hot_encoding,
        )

        train_indices = _load_index(data_dir / f"index_train_{self.split_num}.txt")
        test_indices = _load_index(data_dir / f"index_test_{self.split_num}.txt")
        x_train = x_all[train_indices]
        y_train = y_all[train_indices]
        x_test = x_all[test_indices]
        y_test = y_all[test_indices]

        split_key = split.lower()
        if split_key in {"validation", "valid"}:
            cutoff = int(self.train_ratio * x_train.shape[0])
            x_fit = x_train[:cutoff]
            y_fit = y_train[:cutoff]
            x_selected = x_train[cutoff:]
            y_selected = y_train[cutoff:]
            selected_indices = train_indices[cutoff:]
        else:
            x_fit = x_train
            y_fit = y_train
            if split_key == "train":
                x_selected = x_train
                y_selected = y_train
                selected_indices = train_indices
            elif split_key in {"val", "test"}:
                x_selected = x_test
                y_selected = y_test
                selected_indices = test_indices
            else:
                raise ValueError("split must be 'train', 'val', 'test', or 'validation'.")

        self.scaler_x: StandardScaler | None = None
        self.scaler_y: StandardScaler | None = None
        if self.normalize_x:
            x_fit, x_selected = self._normalize_x(x_fit, x_selected)
        if self.normalize_y:
            y_fit, y_selected = self._normalize_y(y_fit, y_selected)

        self.x = torch.from_numpy(x_selected.astype(np.float32))
        self.y = torch.from_numpy(y_selected.astype(np.float32))
        if y_dim > 0 and self.y.shape[1] != y_dim:
            self.y = self.y[:, :y_dim]
        self.index = torch.from_numpy(selected_indices.astype(np.int64))
        self.spec = UCIBatchSpec(
            input_dim=int(self.x.shape[1]),
            target_dim=int(self.y.shape[1]),
            num_modes=1,
        )
        self.train_size = int(x_train.shape[0])
        self.test_size = int(x_test.shape[0])

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[index], "y": self.y[index], "index": self.index[index]}

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        if self.scaler_y is None:
            return y
        mean = torch.as_tensor(self.scaler_y.mean_, device=y.device, dtype=y.dtype)
        scale = torch.as_tensor(self.scaler_y.scale_, device=y.device, dtype=y.dtype)
        return y * scale + mean

    def _normalize_x(
        self,
        x_fit: np.ndarray,
        x_selected: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.scaler_x = StandardScaler(with_mean=True, with_std=True)
        if self.dim_cat == 0:
            return (
                self.scaler_x.fit_transform(x_fit).astype(np.float32),
                self.scaler_x.transform(x_selected).astype(np.float32),
            )

        x_fit_num, x_fit_cat = x_fit[:, :-self.dim_cat], x_fit[:, -self.dim_cat:]
        x_selected_num, x_selected_cat = (
            x_selected[:, :-self.dim_cat],
            x_selected[:, -self.dim_cat:],
        )
        x_fit_num = self.scaler_x.fit_transform(x_fit_num).astype(np.float32)
        x_selected_num = self.scaler_x.transform(x_selected_num).astype(np.float32)
        return (
            np.concatenate([x_fit_num, x_fit_cat], axis=1).astype(np.float32),
            np.concatenate([x_selected_num, x_selected_cat], axis=1).astype(np.float32),
        )

    def _normalize_y(
        self,
        y_fit: np.ndarray,
        y_selected: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        return (
            self.scaler_y.fit_transform(y_fit).astype(np.float32),
            self.scaler_y.transform(y_selected).astype(np.float32),
        )

    def _onehot_encode_cat_feature(
        self,
        x: np.ndarray,
        cat_var_idx_list: list[int],
    ) -> tuple[np.ndarray, int]:
        x_num = np.delete(arr=x, obj=cat_var_idx_list, axis=1)
        x_cat = x[:, cat_var_idx_list]
        x_onehot_cat = []
        for col in range(x_cat.shape[1]):
            x_onehot_cat.append(np.asarray(pd.get_dummies(x_cat[:, col], drop_first=True)))
        cat = np.concatenate(x_onehot_cat, axis=1).astype(np.float32)
        return np.concatenate([x_num, cat], axis=1).astype(np.float32), int(cat.shape[1])

    def _preprocess_feature_set(
        self,
        x: np.ndarray,
        dataset_name: str,
        one_hot_encoding: bool,
    ) -> tuple[np.ndarray, int]:
        if not one_hot_encoding:
            return x, 0
        if dataset_name == "bostonHousing":
            return self._onehot_encode_cat_feature(x, [3])
        if dataset_name == "energy":
            return self._onehot_encode_cat_feature(x, [4, 6, 7])
        if dataset_name == "naval-propulsion-plant":
            return self._onehot_encode_cat_feature(x, [0, 1, 8, 11])
        return x, 0


def build_uci_dataset(name: str, split: str, **kwargs: Any) -> UCIRegressionDataset:
    return UCIRegressionDataset(name=name, split=split, **kwargs)
