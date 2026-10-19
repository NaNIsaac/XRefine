# XRefine

[![Tests](https://github.com/NaNIsaac/XRefine/actions/workflows/ci.yml/badge.svg)](https://github.com/NaNIsaac/XRefine/actions/workflows/ci.yml)

Superseded by [ReflectoLearn](https://github.com/NaNIsaac/ReflectoLearn).

A neural network that predicts thin-film layer parameters (thickness, roughness, real and imaginary SLD) from X-ray reflectivity (XRR) curves, with a differentiable Abeles simulator in PyTorch used both in the training loss and for gradient-based refinement afterwards.

Training data is synthesized on the fly from the parameter ranges in `config.yaml` (substrate plus one or more layers, and instrument parameters: sample length, beam width, resolution `dq`, intensity scale, background, 2θ offset), with augmentations for noise, background, resolution and q-shift. The model outputs normalized parameters that are mapped back to physical ranges; the curve reconstructed by the Abeles engine is compared to the input (linear, gradient and log terms), together with a supervised parameter loss whose weight decays over training (`sup_w_start` → `sup_w_end`) and a physics-constraint penalty. Three architectures are registered: `fourier_conv` (Fourier features + 1D CNN, the default), `vit` and `latent_trans`.

Refinement takes the network prediction as the starting point and fits a measured curve in stages (I0 pre-fit, scale and footprint, film structure, fine tuning) by gradient descent through the same simulator.

Personal research code; several modules are experimental or superseded.

## Install

Python 3.13 and [uv](https://docs.astral.sh/uv/). `torch==2.9.0` is pulled from the PyTorch CUDA 12.6 index.

```bash
git clone https://github.com/NaNIsaac/XRefine.git
cd XRefine
uv sync
```

`pyproject.toml` has no build-system, so the `xrefine` package is not installed; run everything from the repository root.

## Usage

Set `save_root` and `experiment_name` in `config.yaml` first (the committed `save_root` is a local Windows path).

```bash
uv run python main.py train --cfg config.yaml
uv run python main.py train --cfg config.yaml --resume
uv run python main.py train --cfg config.yaml --train.lr 1e-3 --model.model_type vit   # any config key as --section.key value
uv run python main.py eval --checkpoint path/to/best.pt
uv run python main.py config                                                           # print the default config as JSON
```

Training writes `config.yaml`, `best.pt`, `last.pt` and TensorBoard logs (`logs/`) to `save_root/experiment_name`. `eval` simulates a synthetic test set and writes `eval_metrics_full.csv`, `eval_*.png` and `report_corr_*.png` next to the checkpoint; without `--checkpoint` it uses `save_root/experiment_name/best.pt`.

`main_enhanced.py` has the same commands but trains a `fourier_conv` variant with soft clamping at the parameter bounds and curriculum learning (`models/common_enhanced.py`).

Refine a measured curve (two whitespace-separated columns: 2θ in degrees, intensity). Normalization is set by `data.normalization_mode` in the config (`plateau`, `max`, `anchor`, `none`) and must match training:

```bash
uv run python -m xrefine.refine.pipeline_next sample.dat --checkpoint path/to/best.pt --out ./results
```

This writes `<name>_<timestamp>_result.png` and `<name>_<timestamp>_params.csv`. `xrefine.refine.pipeline` is the earlier version with the same arguments.

Other scripts:

```bash
uv run python -m xrefine.refine.benchmark -c path/to/best.pt -n 3 -o ./benchmark_results   # refine one fixed + n random synthetic cases
uv run python -m xrefine.hpo --db hpo_fourier.db --trials 100                               # Optuna search for fourier_conv
uv run python convert_checkpoint.py path/to/best.pt                                         # Linux checkpoint -> best_win.pt loadable on Windows
```

## Structure

```
main.py, main_enhanced.py   Typer CLI: train / eval / config
config.yaml                 parameter ranges, augmentation, training and model settings
xrefine/
  config.py                 pydantic config (XRefineConfig)
  core/                     Abeles matrix, smearing, kinematical approximation, ParamSet structures
  data/                     synthetic batch generator, augmentations, intensity normalization
  models/                   model registry; fourier_model.py, vit.py, transformer.py
  train/                    trainer (AMP, decaying supervision weight, checkpoints), loss
  refine/                   staged refinement pipelines, optimizers, losses, benchmark
  evaluate.py               synthetic-set evaluation and plots
  hpo.py                    Optuna hyperparameter search
debug/                      one-off debugging scripts
scripts/                    baseline vs enhanced comparison
tests/                      gradient tests for the enhanced model
```
