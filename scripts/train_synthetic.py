from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import trange

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hmcl.data import build_synthetic_dataset
from hmcl.losses import annealed_mcl_loss
from hmcl.models import NoiseConditionedMLP
from hmcl.noise import NoiseSchedule, sigma_to_temperature
from hmcl.utils import cycle, load_yaml, resolve_device, save_jsonl, save_yaml, set_seed
from scripts.plot_diagnostics import generate_loss_plot, generate_sigma_wta_sweep_plot
from scripts.plot_trajectories import generate_trajectory_plot, generate_trajectory_gif

VALID_METHODS = {"hmcl", "amcl", "mcl"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/branching_curves.yaml"))
    parser.add_argument("--steps", type=int, default=None, help="Override training steps.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override run output directory.",
    )
    return parser.parse_args()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(
    model: NoiseConditionedMLP,
    loader: DataLoader,
    schedule: NoiseSchedule,
    loss_config: dict,
    device: torch.device,
    max_batches: int = 8,
    eval_sigma: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        if eval_sigma is None:
            sigma = schedule.sample(batch["x"].shape[0], device=device)
        else:
            sigma = eval_sigma.to(device=device).expand(batch["x"].shape[0])
        predictions = model(batch["x"], sigma)
        _, metrics = annealed_mcl_loss(predictions, batch["y"], sigma=sigma, **loss_config)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
    model.train()
    return {f"val/{key}": value / max(count, 1) for key, value in totals.items()}


def get_method(config: dict) -> str:
    method = str(config.get("run", {}).get("method", "hmcl")).lower()
    if method not in VALID_METHODS:
        raise ValueError(f"run.method must be one of {sorted(VALID_METHODS)}, got {method!r}.")
    return method


def step_sigma(schedule: NoiseSchedule, step: int, steps: int, device: torch.device) -> torch.Tensor:
    grid = schedule.grid(steps, device=device)
    return grid[steps - step]


def method_batch_sigma(
    method: str,
    schedule: NoiseSchedule,
    batch_size: int,
    step: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    if method == "hmcl":
        return schedule.sample(batch_size, device=device)
    if method == "amcl":
        return step_sigma(schedule, step=step, steps=steps, device=device).expand(batch_size)
    return torch.full((batch_size,), float(schedule.sigma_min), device=device)


def method_loss_config(method: str, config: dict) -> dict:
    loss_config = dict(config["loss"])
    if method == "mcl":
        loss_config["assignment"] = "wta"
    return loss_config


def save_checkpoint(
    path: Path,
    model: NoiseConditionedMLP,
    config: dict,
    step: int,
    train_data,
    method: str,
    sigma: torch.Tensor,
) -> None:
    loss_config = method_loss_config(method, config)
    temperature = sigma_to_temperature(
        sigma.detach().cpu(),
        scale=float(loss_config.get("temperature_scale", 1.0)),
        power=float(loss_config.get("temperature_power", 2.0)),
        floor=float(loss_config.get("temperature_floor", 1e-4)),
    )
    checkpoint = {
        "model": model.state_dict(),
        "config": config,
        "step": step,
        "method": method,
        "training_sigma": float(sigma.detach().mean().cpu()),
        "training_temperature": float(temperature.mean()),
        "spec": {
            "input_dim": train_data.spec.input_dim,
            "target_dim": train_data.spec.target_dim,
            "num_modes": train_data.spec.num_modes,
        },
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.steps is not None:
        config["optim"]["steps"] = args.steps
    if args.output_dir is not None:
        config["run"]["output_dir"] = str(args.output_dir)

    set_seed(int(config["run"]["seed"]))
    method = get_method(config)
    device = resolve_device(config["run"].get("device", "auto"))
    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output_dir / "config.yaml")

    data_config = dict(config["data"])
    name = data_config.pop("name")
    n_train = data_config.pop("n_train")
    n_val = data_config.pop("n_val")
    batch_size = data_config.pop("batch_size")
    train_data = build_synthetic_dataset(name, split="train", n_samples=n_train, **data_config)
    val_data = build_synthetic_dataset(name, split="val", n_samples=n_val, **data_config)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    train_iter = cycle(train_loader)

    schedule = NoiseSchedule(**config["noise"])
    model = NoiseConditionedMLP(
        input_dim=train_data.spec.input_dim,
        target_dim=train_data.spec.target_dim,
        **config["model"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optim"]["lr"]),
        weight_decay=float(config["optim"]["weight_decay"]),
    )

    steps = int(config["optim"]["steps"])
    log_every = int(config["optim"].get("log_every", 100))
    val_every = int(config["optim"].get("val_every", 500))
    metrics_path = output_dir / "metrics.jsonl"
    start = time.time()

    progress = trange(1, steps + 1, desc=config["run"]["name"])
    checkpoint_path = output_dir / "last.pt"
    for step in progress:
        batch = move_batch(next(train_iter), device)
        sigma = method_batch_sigma(
            method=method,
            schedule=schedule,
            batch_size=batch["x"].shape[0],
            step=step,
            steps=steps,
            device=device,
        )
        predictions = model(batch["x"], sigma)
        loss, metrics = annealed_mcl_loss(
            predictions,
            batch["y"],
            sigma=sigma,
            **method_loss_config(method, config),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if step % log_every == 0 or step == 1:
            record = {
                "step": step,
                "elapsed_sec": time.time() - start,
                "sigma_mean": float(sigma.mean()),
                **{f"train/{key}": float(value) for key, value in metrics.items()},
            }
            progress.set_postfix(
                loss=record["train/loss"],
                min_dist=record["train/min_distance"],
            )
            save_jsonl(record, metrics_path)

        if step % val_every == 0 or step == steps:
            val_metrics = evaluate(
                model,
                val_loader,
                schedule,
                method_loss_config(method, config),
                device=device,
                eval_sigma=sigma[:1] if method != "hmcl" else None,
            )
            save_jsonl({"step": step, **val_metrics}, metrics_path)
            save_checkpoint(
                checkpoint_path,
                model=model,
                config=config,
                step=step,
                train_data=train_data,
                method=method,
                sigma=sigma,
            )
            if method == "amcl":
                save_checkpoint(
                    output_dir / f"checkpoint_{step:06d}.pt",
                    model=model,
                    config=config,
                    step=step,
                    train_data=train_data,
                    method=method,
                    sigma=sigma,
                )

    if method != "mcl":
        generate_trajectory_plot(
            checkpoint=checkpoint_path,
            device_name=str(device),
            num_inputs=int(config["plot"]["num_inputs"]),
            num_sigmas=int(config["plot"]["num_sigmas"]),
            method=method,
            plot_mode="trajectory",
        )
        generate_trajectory_gif(
            checkpoint=checkpoint_path,
            device_name=str(device),
            num_inputs=int(config["plot"]["num_inputs"]),
            num_sigmas=int(config["plot"]["num_sigmas"]),
            method=method,
        )
    generate_trajectory_plot(
        checkpoint=checkpoint_path,
        output=checkpoint_path.with_name("final_configuration.png"),
        device_name=str(device),
        num_inputs=int(config["plot"]["num_inputs"]),
        num_sigmas=int(config["plot"]["num_sigmas"]),
        method=method,
        plot_mode="last",
    )
    generate_loss_plot(metrics_path)
    if method != "mcl":
        generate_sigma_wta_sweep_plot(checkpoint=checkpoint_path, device_name=str(device))


if __name__ == "__main__":
    main()
