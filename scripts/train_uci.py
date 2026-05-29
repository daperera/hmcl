from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hmcl.utils import load_yaml, save_yaml
from scripts.train_synthetic import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def mean_std(metrics_list: list[dict]) -> dict:
    result = {}
    keys = metrics_list[0].keys()

    for key in keys:
        values = np.array([m[key] for m in metrics_list])
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_std"] = float(values.std())

    return result


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    max_split = config["data"].pop("max_split")
    root_output_dir = Path(config["run"]["output_dir"])

    if root_output_dir.exists():
        shutil.rmtree(root_output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    for split_num in range(max_split):
        split_config = deepcopy(config)
        split_config["data"]["split_num"] = split_num
        split_config["run"]["output_dir"] = str(
            root_output_dir / f"split_{split_num:02d}"
        )
        metrics.append(train(split_config))

    summary = mean_std(metrics)
    save_yaml(summary, root_output_dir / "summary.yaml")

    print(summary)


if __name__ == "__main__":
    main()