# HMCL: Noise-Conditioned Annealed MCL

This workspace is set up for experiments with a variant of [Annealed Multiple Choice Learning](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1456560769bbc38e4f8c5055048ea712-Abstract-Conference.html)
(aMCL) where the noise level is an explicit model input. The same scalar
noise level also controls the hypothesis assignment temperature in the loss, so
large noise levels encourage coarse/shared hypotheses and small noise levels move
toward winner-takes-all specialization.

The first target is multimodal regression on meaningful synthetic data. The code is
kept plain PyTorch so it is easy to inspect, change, and move into notebooks.

## Setup

CPU environment:

```bash
conda env create -f environment.yml
conda activate hmcl
```

CUDA environment:

```bash
conda env create -f environment-cuda.yml
conda activate hmcl
```

The conda files install this workspace in editable mode with `pip -e .`.

## Quick Smoke Test

```bash
python scripts/smoke_test.py
```

## Train A Starter Experiment

```bash
python scripts/train_synthetic.py --config configs/one_gaussian.yaml
python scripts/train_synthetic.py --config configs/regular_polygon_gaussians.yaml
python scripts/train_synthetic.py --config configs/hierarchical_polygon_gaussians.yaml
```

Each run writes checkpoints, the resolved config, trajectory plots, loss curves, and a
hard-WTA sigma-sweep diagnostic under `runs/`.

Set `run.method` in a config to choose the training/plotting mode:

- `hmcl`: the noise-conditioned method in this repo; plots sweep one checkpoint over sigma.
- `amcl`: anneals temperature over training and plots saved checkpoints at their recorded temperatures.
- `mcl`: trains with hard winner-takes-all assignment and plots only the final prediction positions.

For `hmcl` and `amcl`, training writes `trajectories.png` for the full static trajectory and `trajectories.gif` for the animated trajectory. All methods also write `final_configuration.png` for the last/final prediction configuration.

## Plot Prediction Trajectories

```bash
python scripts/plot_trajectories.py --checkpoint runs/branching_curves/last.pt
python scripts/plot_trajectories.py --checkpoint runs/branching_curves/last.pt --gif
```

By default, the static PNG shows the full trajectory when the method has one, and the script also writes `final_configuration.png` for the last/final prediction configuration. Use `--plot-mode last` when you only want the final static view. With `--gif`, the script also writes `trajectories.gif`, where animation time runs from high sigma to low sigma for `hmcl` and through saved checkpoints for `amcl`.

## Plot Training Diagnostics

```bash
python scripts/plot_diagnostics.py --checkpoint runs/branching_curves/last.pt
```

Training runs call this automatically after the final checkpoint. It writes
`loss_curves.png` from the latest `metrics.jsonl` run segment, plus
`sigma_wta_sweep.png` and `sigma_wta_sweep.jsonl` to show how the trained
model's hard-WTA MCL loss changes as the model conditioning sigma is swept.
The sweep records the scheduled temperature for each sigma as metadata, but the
loss itself uses hard winner-takes-all assignment.

## Core Pieces

- `hmcl.models.NoiseConditionedMLP`: diffusion-style Fourier or sinusoidal noise embeddings with FiLM-conditioned residual MLP blocks.
- `hmcl.losses.annealed_mcl_loss`: soft, hard, and epsilon-WTA MCL objectives.
- `hmcl.noise.NoiseSchedule`: log-uniform, uniform, or discrete sampling over sigma values.
- `hmcl.inference`: utilities for sigma-grid prediction, head trajectory matching, and hierarchical clustering over hypotheses.
- `hmcl.calibration`: starter metrics for coverage and min-distance calibration.

See `docs/research_plan.md` for the modeling rationale and `docs/datasets.md` for dataset ideas that should make hierarchical or uncertainty-aware behavior visible.
