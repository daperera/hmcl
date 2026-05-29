from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hmcl.data import build_dataset
from hmcl.inference import align_hypothesis_trajectories, cluster_hypotheses, predict_sigma_grid
from hmcl.models import build_model
from hmcl.noise import NoiseSchedule, sigma_to_temperature
from hmcl.utils import resolve_device, method_model_sigma


class TrajectoryData(TypedDict):
    sigmas: torch.Tensor
    x: torch.Tensor
    predictions: torch.Tensor
    display_predictions: torch.Tensor
    projected_predictions: torch.Tensor
    projected_high_sigma: torch.Tensor
    projected_target: torch.Tensor
    projected_background: torch.Tensor
    align: bool
    method: str
    level_label: str
    final_only: bool


def _project_last_dim_to_2d(array: torch.Tensor) -> torch.Tensor:
    if array.shape[-1] == 2:
        return array
    if array.shape[-1] == 1:
        zeros = torch.zeros_like(array)
        return torch.cat([array, zeros], dim=-1)

    flat = array.reshape(-1, array.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    _, _, v = torch.linalg.svd(centered, full_matrices=False)
    basis = v[:2].T
    projected = centered @ basis
    return projected.reshape(*array.shape[:-1], 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Also write an animated GIF next to the checkpoint.",
    )
    parser.add_argument(
        "--gif-output",
        type=Path,
        default=None,
        help="Optional path for the animated GIF showing trajectories as sigma decreases.",
    )
    parser.add_argument("--num-sigmas", type=int, default=16)
    parser.add_argument("--num-inputs", type=int, default=4)
    parser.add_argument("--num-background", type=int, default=1024)
    parser.add_argument("--gif-fps", type=int, default=3)
    parser.add_argument(
        "--method",
        choices=("hmcl", "amcl", "mcl"),
        default=None,
        help="Override run.method from the checkpoint config.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=("trajectory", "last"),
        default="trajectory",
        help="Static PNG mode. Defaults to the full trajectory when available.",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Apply nearest-neighbor head alignment across adjacent sigma values.",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def model_from_checkpoint(checkpoint: dict, device: torch.device) -> torch.nn.Module:
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
    return model


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    return model_from_checkpoint(checkpoint, device), checkpoint["config"]


def _checkpoint_method(checkpoint_data: dict, override: str | None = None) -> str:
    method = override or checkpoint_data.get("method") or checkpoint_data["config"].get("run", {}).get("method", "hmcl")
    method = str(method).lower()
    if method not in {"hmcl", "amcl", "mcl"}:
        raise ValueError(f"method must be one of 'hmcl', 'amcl', or 'mcl', got {method!r}.")
    return method


def _build_plot_tensors(
    config: dict,
    num_inputs: int,
    num_background: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data_config = dict(config["data"])
    name = data_config.pop("name")
    source = data_config.pop("source", None)
    n_val = data_config.pop("n_val", None)
    data_config.pop("n_train", None)
    data_config.pop("batch_size", None)
    data_config.pop("val_split", None)
    if n_val is not None:
        data_config["n_samples"] = max(num_inputs, num_background, 256)
    dataset = build_dataset(name, split="test", source=source, **data_config)
    count = min(num_inputs, len(dataset))
    x = torch.stack([dataset[index]["x"] for index in range(count)], dim=0)
    target = torch.stack([dataset[index]["y"] for index in range(count)], dim=0)
    background = torch.stack(
        [dataset[index]["y"] for index in range(min(num_background, len(dataset)))],
        dim=0,
    )
    return x, target, background


def _amcl_checkpoint_paths(checkpoint: Path, num_checkpoints: int) -> list[Path]:
    paths = sorted(checkpoint.parent.glob("checkpoint_*.pt"))
    if not paths:
        return [checkpoint]
    if len(paths) <= num_checkpoints:
        return paths
    indices = torch.linspace(0, len(paths) - 1, steps=num_checkpoints).round().long().tolist()
    return [paths[index] for index in indices]


def _predict_single_checkpoint(
    checkpoint_data: dict,
    x: torch.Tensor,
    sigma_value: float,
    method: str,
    device: torch.device,
) -> torch.Tensor:
    model = model_from_checkpoint(checkpoint_data, device)
    sigma = torch.full((x.shape[0],), float(sigma_value), device=device)
    with torch.no_grad():
        model_sigma = method_model_sigma(method, sigma_value)
        predictions, scores = model(x.to(device), model_sigma)
        return predictions.cpu()


def _prepare_trajectory_data(
    checkpoint: Path,
    num_sigmas: int = 16,
    num_inputs: int = 4,
    num_background: int = 1024,
    align: bool = False,
    device_name: str = "auto",
    method: Literal["hmcl", "amcl", "mcl"] | None = None,
) -> TrajectoryData:
    device = resolve_device(device_name)
    checkpoint_data = load_checkpoint(checkpoint, device)
    config = checkpoint_data["config"]
    resolved_method = _checkpoint_method(checkpoint_data, method)
    noise_config = config.get("plot", {}).get("noise", config["noise"])
    schedule = NoiseSchedule(**noise_config)
    x, target, background = _build_plot_tensors(config, num_inputs, num_background)

    if resolved_method == "amcl":
        checkpoint_paths = _amcl_checkpoint_paths(checkpoint, num_sigmas)
        level_values = []
        prediction_steps = []
        for path in checkpoint_paths:
            step_checkpoint = load_checkpoint(path, device)
            sigma_value = float(step_checkpoint.get("training_sigma", schedule.sigma_min))
            if "training_temperature" in step_checkpoint:
                level_value = float(step_checkpoint["training_temperature"])
            else:
                loss_config = dict(step_checkpoint["config"].get("loss", {}))
                level_value = float(
                    sigma_to_temperature(
                        torch.tensor([sigma_value]),
                        scale=float(loss_config.get("temperature_scale", 1.0)),
                        power=float(loss_config.get("temperature_power", 2.0)),
                        floor=float(loss_config.get("temperature_floor", 1e-4)),
                    ).item()
                )
            level_values.append(level_value)
            prediction_steps.append(_predict_single_checkpoint(step_checkpoint, x, sigma_value, method, device))
        sigmas = torch.tensor(level_values)
        predictions = torch.stack(prediction_steps, dim=0)
        level_label = "checkpoint training temperature"
    elif resolved_method == "mcl":
        sigma_value = float(checkpoint_data.get("training_sigma", schedule.sigma_min))
        sigmas = torch.tensor([sigma_value])
        predictions = _predict_single_checkpoint(checkpoint_data, x, sigma_value, resolved_method, device).unsqueeze(0)
        level_label = "final checkpoint"
    else:
        model = model_from_checkpoint(checkpoint_data, device)
        sigmas = schedule.grid(num_sigmas)
        predictions = predict_sigma_grid(model, x, sigmas, resolved_method, device=device)
        level_label = "sigma"

    display_predictions = align_hypothesis_trajectories(predictions) if align and len(sigmas) > 1 else predictions
    high_index = int(torch.argmax(sigmas).item()) if len(sigmas) > 1 else -1
    return {
        "sigmas": sigmas,
        "x": x,
        "predictions": predictions,
        "display_predictions": display_predictions,
        "projected_predictions": _project_last_dim_to_2d(display_predictions),
        "projected_high_sigma": _project_last_dim_to_2d(predictions[high_index]),
        "projected_target": _project_last_dim_to_2d(target),
        "projected_background": _project_last_dim_to_2d(background),
        "align": align,
        "method": resolved_method,
        "level_label": level_label,
        "final_only": resolved_method == "mcl",
    }


def _trajectory_limits(
    projected_predictions: torch.Tensor,
    projected_target: torch.Tensor,
    projected_background: torch.Tensor,
) -> tuple[float, float, float, float]:
    points = torch.cat(
        [
            projected_predictions.reshape(-1, 2),
            projected_target.reshape(-1, 2),
            projected_background.reshape(-1, 2),
        ],
        dim=0,
    )
    x_min, y_min = points.min(dim=0).values.tolist()
    x_max, y_max = points.max(dim=0).values.tolist()
    x_span = max(x_max - x_min, 1e-3)
    y_span = max(y_max - y_min, 1e-3)
    padding = 0.08 * max(x_span, y_span)
    return x_min - padding, x_max + padding, y_min - padding, y_max + padding


def _final_configuration_index(data: TrajectoryData) -> int:
    if data["method"] == "hmcl":
        return int(torch.argmin(data["sigmas"]).item())
    return len(data["sigmas"]) - 1


def generate_trajectory_plot(
    checkpoint: Path,
    output: Path | None = None,
    num_sigmas: int = 16,
    num_inputs: int = 4,
    num_background: int = 1024,
    align: bool = False,
    device_name: str = "auto",
    method: Literal["hmcl", "amcl", "mcl"] | None = None,
    plot_mode: Literal["trajectory", "last"] = "trajectory",
) -> Path:
    data = _prepare_trajectory_data(
        checkpoint=checkpoint,
        num_sigmas=num_sigmas,
        num_inputs=num_inputs,
        num_background=num_background,
        align=align,
        device_name=device_name,
        method=method,
    )
    if plot_mode not in {"last", "trajectory"}:
        raise ValueError("plot_mode must be either 'last' or 'trajectory'.")
    plot_last_only = plot_mode == "last"
    sigmas = data["sigmas"]
    x = data["x"]
    projected_predictions = data["projected_predictions"]
    if plot_last_only:
        final_index = _final_configuration_index(data)
        sigmas = sigmas[final_index : final_index + 1]
        projected_predictions = projected_predictions[final_index : final_index + 1]
        projected_group_predictions = projected_predictions[0]
    else:
        projected_group_predictions = data["projected_high_sigma"]
    projected_target = data["projected_target"]
    projected_background = data["projected_background"]
    final_only = data["final_only"] or plot_last_only

    fig, axes = plt.subplots(
        1,
        num_inputs,
        figsize=(5 * num_inputs, 5),
        squeeze=False,
        constrained_layout=True,
    )
    sigma_min = float(sigmas.min())
    sigma_max = float(sigmas.max())
    if sigma_min == sigma_max:
        sigma_min -= 1e-6
        sigma_max += 1e-6
    sigma_norm = Normalize(vmin=sigma_min, vmax=sigma_max)
    sigma_cmap = plt.cm.viridis
    head_cmap = plt.cm.get_cmap("tab10", projected_predictions.shape[2])
    for input_index, axis in enumerate(axes[0]):
        axis.scatter(
            projected_background[:, 0],
            projected_background[:, 1],
            color="0.83",
            s=12,
            alpha=0.3,
            zorder=0,
            label="data distribution" if input_index == 0 else None,
        )
        all_points = projected_predictions[:, input_index].reshape(
            -1,
            projected_predictions.shape[-1],
        )
        axis.scatter(
            all_points[:, 0],
            all_points[:, 1],
            color="0.9",
            s=10,
            alpha=0.4,
            zorder=0,
        )
        for head in range(projected_predictions.shape[2]):
            trajectory = projected_predictions[:, input_index, head]
            head_color = head_cmap(head)
            if not final_only:
                axis.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    color=head_color,
                    linewidth=1.8,
                    alpha=0.95,
                    zorder=1,
                )
            axis.scatter(
                trajectory[:, 0],
                trajectory[:, 1],
                c=sigmas.numpy(),
                cmap=sigma_cmap,
                norm=sigma_norm,
                s=30,
                edgecolors=[head_color],
                linewidths=0.9,
                zorder=2,
            )
            axis.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                color=head_color,
                s=70,
                marker="o",
                edgecolors="black",
                linewidths=1.0,
                zorder=3,
            )
            if not final_only:
                axis.scatter(
                    trajectory[-1, 0],
                    trajectory[-1, 1],
                    color=head_color,
                    s=80,
                    marker="X",
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=3,
                )
        high_sigma_clusters = cluster_hypotheses(projected_group_predictions[input_index], n_clusters=3)
        group_label = "final groups" if final_only else "high-sigma groups"
        axis.set_title(
            "x="
            f"{float(x[input_index, 0]):.2f}, "
            f"{group_label}={high_sigma_clusters['labels'].max()}"
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.set_xlabel("y0")
        axis.set_ylabel("y1")

    sm = plt.cm.ScalarMappable(cmap=sigma_cmap, norm=sigma_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
    cbar.set_label(data["level_label"])

    handles = [
        plt.Line2D([0], [0], color=head_cmap(head), lw=2, label=f"head {head}")
        for head in range(projected_predictions.shape[2])
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(4, len(handles)),
        frameon=False,
    )
    mode = "aligned heads" if align else "raw head identities"
    if final_only:
        fig.suptitle(
            f"Final prediction positions ({data['level_label']}={float(sigmas[0]):.4g}); "
            "circle=prediction, star=true target"
        )
    else:
        fig.suptitle(
            f"Hypothesis trajectories across {data['level_label']} ({mode}); "
            "circle=first level, X=last level, star=true target"
        )
    output = output or checkpoint.with_name("trajectories.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"wrote {output}")
    plt.close(fig)
    return output


def generate_trajectory_gif(
    checkpoint: Path,
    output: Path | None = None,
    num_sigmas: int = 16,
    num_inputs: int = 4,
    num_background: int = 1024,
    align: bool = False,
    device_name: str = "auto",
    fps: int = 3,
    method: Literal["hmcl", "amcl", "mcl"] | None = None,
) -> Path:
    data = _prepare_trajectory_data(
        checkpoint=checkpoint,
        num_sigmas=num_sigmas,
        num_inputs=num_inputs,
        num_background=num_background,
        align=align,
        device_name=device_name,
        method=method,
    )
    if data["final_only"]:
        raise ValueError("MCL has only final prediction positions; GIF trajectories are not generated.")
    sigmas = data["sigmas"]
    x = data["x"]
    projected_predictions = data["projected_predictions"]
    projected_target = data["projected_target"]
    projected_background = data["projected_background"]
    head_cmap = plt.cm.get_cmap("tab10", projected_predictions.shape[2])
    if data["method"] == "amcl":
        frame_indices = list(range(len(sigmas)))
    else:
        frame_indices = list(range(len(sigmas) - 1, -1, -1))
    x_min, x_max, y_min, y_max = _trajectory_limits(
        projected_predictions,
        projected_target,
        projected_background,
    )

    fig, axes = plt.subplots(
        1,
        num_inputs,
        figsize=(5 * num_inputs, 5),
        squeeze=False,
        constrained_layout=True,
    )
    axes_list = axes[0].tolist()
    for input_index, axis in enumerate(axes_list):
        axis.scatter(
            projected_background[:, 0],
            projected_background[:, 1],
            color="0.86",
            s=12,
            alpha=0.24,
            zorder=0,
            label="data distribution" if input_index == 0 else None,
        )
        axis.set_title(f"x={float(x[input_index, 0]):.2f}")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.grid(alpha=0.2)
        axis.set_xlabel("y0")
        axis.set_ylabel("y1")

    trajectory_lines: list[list[plt.Line2D]] = []
    current_points: list[list[plt.Line2D]] = []
    for input_index, axis in enumerate(axes_list):
        input_lines = []
        input_points = []
        for head in range(projected_predictions.shape[2]):
            head_color = head_cmap(head)
            line = axis.plot(
                [],
                [],
                color=head_color,
                linewidth=1.8,
                alpha=0.95,
                zorder=1,
            )[0]
            point = axis.plot(
                [],
                [],
                marker="o",
                markersize=7,
                markerfacecolor=head_color,
                markeredgecolor="black",
                markeredgewidth=0.9,
                linestyle="None",
                zorder=3,
            )[0]
            input_lines.append(line)
            input_points.append(point)
        trajectory_lines.append(input_lines)
        current_points.append(input_points)

    handles = [
        plt.Line2D([0], [0], color=head_cmap(head), lw=2, label=f"head {head}")
        for head in range(projected_predictions.shape[2])
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(4, len(handles)),
        frameon=False,
    )
    mode = "aligned heads" if align else "raw head identities"

    def update(frame_position: int) -> list[plt.Artist]:
        sigma_index = frame_indices[frame_position]
        visited = frame_indices[: frame_position + 1]
        artists: list[plt.Artist] = []
        for input_index, _axis in enumerate(axes_list):
            for head in range(projected_predictions.shape[2]):
                trajectory = projected_predictions[visited, input_index, head]
                current = projected_predictions[sigma_index, input_index, head]
                trajectory_lines[input_index][head].set_data(
                    trajectory[:, 0].numpy(),
                    trajectory[:, 1].numpy(),
                )
                current_points[input_index][head].set_data(
                    [float(current[0])],
                    [float(current[1])],
                )
                artists.extend(
                    [
                        trajectory_lines[input_index][head],
                        current_points[input_index][head],
                    ]
                )
        fig.suptitle(
            f"Hypothesis trajectories across {data['level_label']} ({mode}); "
            f"{data['level_label']}={float(sigmas[sigma_index]):.4g}, star=true target"
        )
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / max(fps, 1),
        blit=False,
        repeat=True,
    )
    output = output or checkpoint.with_name("trajectories.gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=max(fps, 1)), dpi=120)
    print(f"wrote {output}")
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    generate_trajectory_plot(
        checkpoint=args.checkpoint,
        output=args.output,
        num_sigmas=args.num_sigmas,
        num_inputs=args.num_inputs,
        num_background=args.num_background,
        align=args.align,
        method=args.method,
        plot_mode=args.plot_mode,
    )
    if args.plot_mode != "last":
        generate_trajectory_plot(
            checkpoint=args.checkpoint,
            output=args.checkpoint.with_name("final_configuration.png"),
            num_sigmas=args.num_sigmas,
            num_inputs=args.num_inputs,
            num_background=args.num_background,
            align=args.align,
            method=args.method,
            plot_mode="last",
        )
    if args.gif or args.gif_output is not None:
        try:
            generate_trajectory_gif(
                checkpoint=args.checkpoint,
                output=args.gif_output,
                num_sigmas=args.num_sigmas,
                num_inputs=args.num_inputs,
                num_background=args.num_background,
                align=args.align,
                fps=args.gif_fps,
                method=args.method,
            )
        except ValueError as error:
            print(f"skipping GIF: {error}")


if __name__ == "__main__":
    main()
