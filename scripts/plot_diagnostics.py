from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hmcl.data import build_dataset
from hmcl.losses import annealed_mcl_loss
from hmcl.models import build_model
from hmcl.noise import NoiseSchedule, sigma_to_temperature
from hmcl.utils import resolve_device, save_jsonl, method_model_sigma


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _latest_metric_segment(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return records
    start = 0
    previous_step: int | None = None
    for index, record in enumerate(records):
        step = record.get("step")
        if not isinstance(step, int):
            continue
        if previous_step is not None and step < previous_step:
            start = index
        previous_step = step
    return records[start:]


def generate_loss_plot(metrics_path: Path, output: Path | None = None) -> Path:
    """Plot the latest training/validation loss segment from ``metrics.jsonl``."""

    records = _latest_metric_segment(_read_metrics(metrics_path))
    train_steps = [record["step"] for record in records if "train/loss" in record]
    train_losses = [record["train/loss"] for record in records if "train/loss" in record]
    val_steps = [record["step"] for record in records if "val/loss" in record]
    val_losses = [record["val/loss"] for record in records if "val/loss" in record]

    if not train_losses and not val_losses:
        raise ValueError(f"No train/loss or val/loss records found in {metrics_path}.")

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if train_losses:
        axis.plot(train_steps, train_losses, label="train", color="tab:blue", linewidth=1.8)
    if val_losses:
        axis.plot(
            val_steps,
            val_losses,
            label="validation",
            color="tab:orange",
            marker="o",
            markersize=4,
            linewidth=1.8,
        )

    positive_losses = [value for value in train_losses + val_losses if value > 0]
    if positive_losses and len(positive_losses) == len(train_losses) + len(val_losses):
        axis.set_yscale("log")
    axis.set_xlabel("training step")
    axis.set_ylabel("MCL loss")
    axis.set_title("Training and validation loss")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

    output = output or metrics_path.with_name("loss_curves.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"wrote {output}")
    plt.close(fig)
    return output


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    spec = checkpoint["spec"]
    model_config = dict(config["model"])
    model_name = model_config.pop("name")
    model = build_model(
        model_name,
        input_dim=spec["input_dim"],
        target_dim=spec["target_dim"],
        **model_config,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def _build_eval_loader(config: dict, batch_size: int | None = None) -> DataLoader:
    data_config = dict(config["data"])
    name = data_config.pop("name")
    source = data_config.pop("source", None)
    n_val = data_config.pop("n_val", None)
    data_config.pop("n_train", None)
    split = data_config.pop("val_split", "val")
    configured_batch_size = int(data_config.pop("batch_size", 1024))
    if n_val is not None:
        data_config["n_samples"] = int(n_val)
    dataset = build_dataset(name, split=split, source=source, **data_config)
    return DataLoader(dataset, batch_size=batch_size or configured_batch_size, shuffle=False)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def generate_sigma_wta_sweep_plot(
    checkpoint: Path,
    output: Path | None = None,
    data_output: Path | None = None,
    num_sigmas: int = 80,
    max_batches: int | None = None,
    device_name: str = "auto",
) -> Path:
    """Evaluate hard-WTA MCL loss while sweeping the model conditioning sigma."""

    device = resolve_device(device_name)
    model, config = _load_model(checkpoint, device)
    method = config.get("run", {}).get("method", "hmcl")
    noise_config = config.get("plot", {}).get("noise", config["noise"])
    schedule = NoiseSchedule(**noise_config)
    loss_config = dict(config["loss"])
    loader = _build_eval_loader(config)
    sigmas = schedule.grid(num_sigmas, device=device)
    scheduled_temperatures = sigma_to_temperature(
        sigmas,
        scale=float(loss_config.get("temperature_scale", 1.0)),
        power=float(loss_config.get("temperature_power", 2.0)),
        floor=float(loss_config.get("temperature_floor", 1e-4)),
    )

    records: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for sigma, scheduled_temperature in zip(sigmas, scheduled_temperatures):
            sigma_float = float(sigma.item())
            loss_sum = 0.0
            min_distance_sum = 0.0
            winner_usage_min_sum = 0.0
            winner_usage_max_sum = 0.0
            sample_count = 0

            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                batch = _move_batch(batch, device)
                batch_size = batch["x"].shape[0]
                sigma_batch = torch.full((batch_size,), sigma_float, device=device)
                model_sigma = method_model_sigma(method, sigma_batch)
                predictions, scores = model(batch["x"], model_sigma)
                loss, metrics = annealed_mcl_loss(
                    predictions,
                    scores,
                    batch["y"],
                    assignment="wta",
                )
                loss_sum += float(loss.detach()) * batch_size
                min_distance_sum += float(metrics["min_distance"]) * batch_size
                winner_usage_min_sum += float(metrics["winner_usage_min"]) * batch_size
                winner_usage_max_sum += float(metrics["winner_usage_max"]) * batch_size
                sample_count += batch_size

            denom = max(sample_count, 1)
            records.append(
                {
                    "conditioning_sigma": sigma_float,
                    "scheduled_temperature": float(scheduled_temperature.item()),
                    "wta_loss": loss_sum / denom,
                    "min_distance": min_distance_sum / denom,
                    "winner_usage_min": winner_usage_min_sum / denom,
                    "winner_usage_max": winner_usage_max_sum / denom,
                }
            )

    data_output = data_output or checkpoint.with_name("sigma_wta_sweep.jsonl")
    if data_output.exists():
        data_output.unlink()
    for record in records:
        save_jsonl(record, data_output)

    sigma_values = [record["conditioning_sigma"] for record in records]
    losses = [record["wta_loss"] for record in records]
    usage_min = [record["winner_usage_min"] for record in records]
    usage_max = [record["winner_usage_max"] for record in records]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    axes[0].plot(sigma_values, losses, color="tab:blue", linewidth=1.9)
    axes[0].set_ylabel("hard-WTA MCL loss")
    axes[0].set_title("Model sigma sweep with hard-WTA assignment")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        sigma_values,
        usage_min,
        color="tab:green",
        linewidth=1.6,
        label="least-used head",
    )
    axes[1].plot(
        sigma_values,
        usage_max,
        color="tab:red",
        linewidth=1.6,
        label="most-used head",
    )
    axes[1].set_xlabel("model conditioning sigma")
    axes[1].set_ylabel("winner usage fraction")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    if schedule.distribution in {"log_uniform", "discrete"}:
        axes[0].set_xscale("log")
        axes[1].set_xscale("log")

    output = output or checkpoint.with_name("sigma_wta_sweep.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"wrote {output}")
    print(f"wrote {data_output}")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--loss-output", type=Path, default=None)
    parser.add_argument("--sigma-output", type=Path, default=None)
    parser.add_argument("--sigma-data-output", type=Path, default=None)
    parser.add_argument("--num-sigmas", type=int, default=80)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = args.metrics or args.checkpoint.with_name("metrics.jsonl")
    generate_loss_plot(metrics, output=args.loss_output)
    generate_sigma_wta_sweep_plot(
        checkpoint=args.checkpoint,
        output=args.sigma_output,
        data_output=args.sigma_data_output,
        num_sigmas=args.num_sigmas,
        max_batches=args.max_batches,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
