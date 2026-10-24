# XRefine

[![Tests](https://github.com/NaNIsaac/XRefine/actions/workflows/ci.yml/badge.svg)](https://github.com/NaNIsaac/XRefine/actions/workflows/ci.yml)

A neural network that predicts thin-film layer parameters (thickness, roughness, real and imaginary SLD) from X-ray reflectivity (XRR) curves, with a differentiable Abeles simulator in PyTorch inside the model: the curve rebuilt from the predicted parameters is part of the training loss, and the same simulator is the forward model of the refinement afterwards.

Training data is synthesized on the fly from the parameter ranges in `config.yaml` (substrate plus one or more layers, and instrument parameters: sample length, beam width, resolution `dq`, intensity scale, background, 2θ offset). Since c1bc382 (2026-10-18) the sampled footprint, resolution and 2θ offset are applied to the training curves through the same `instrument_reflectivity` the model uses; before that the training curves were bare Abeles curves. The augmentor (`data/augmentations.py`) then resamples the 2θ grid, applies intensity scale and background, adds Poisson noise and masks the low-q and high-q ends. The `res_sigma_range` and `q_shift_sigma` settings in `config.py` are defined but not used. The model outputs normalized parameters that are mapped back to physical ranges; the curve reconstructed by the Abeles engine is compared to the input (linear, gradient and log terms), together with a supervised parameter loss whose weight decays over training (`sup_w_start` → `sup_w_end`) and a physics-constraint penalty. Three architectures are registered: `fourier_conv` (Fourier features + 1D CNN, the default), `vit` and `latent_trans`.

Refinement takes the network prediction as the starting point and fits a measured curve in stages (I0 pre-fit, scale and footprint, film structure, fine tuning) through the same simulator. In `refine.pipeline` every stage is a SciPy differential evolution (gradient-free), the film stage after a correlation grid search over thickness, and only the I0 pre-fit uses Adam. `refine.pipeline_next` is a hybrid: differential evolution for the early stages, Adam (L-BFGS for a final stage) for the later ones.

Personal research code; several modules are experimental.

XRefine is one of the XRR research branches; its approach (a physics-informed loss through the simulator, multilayer and complex-SLD outputs, instrument parameters predicted by the network, staged DE refinement) is described in the ReflectoLearn master index `docs/xrr-research-index.md`. [ReflectoLearn](https://github.com/NaNIsaac/ReflectoLearn) took the Abeles engine, the unit helpers and the refinement losses from here (ReflectoLearn f35ab56, e3b26d6) but trains with a supervised parameter loss only; the two training approaches have not been compared. No training or evaluation metrics of XRefine are recorded in this repository.

## Install

Python 3.13 and [uv](https://docs.astral.sh/uv/). `torch==2.9.0` is pulled from the PyTorch CUDA 12.6 index.

```bash
git clone https://github.com/NaNIsaac/XRefine.git
cd XRefine
uv sync
```

`pyproject.toml` has no build-system, so the `xrefine` package is not installed; run everything from the repository root.

## Usage

Set `save_root` and `experiment_name` in `config.yaml` first (the committed `save_root` is `$DATA/XRR_AI/xrefine`).

```bash
uv run python main.py train --cfg config.yaml
uv run python main.py train --cfg config.yaml --resume
uv run python main.py train --cfg config.yaml --train.lr 1e-3 --model.model_type vit   # any config key as --section.key value
uv run python main.py eval --checkpoint path/to/best.pt
uv run python main.py config                                                           # print the default config as JSON
```

Training writes `config.yaml`, `best.pt`, `last.pt` and TensorBoard logs (`logs/`) to `save_root/experiment_name`. `eval` simulates a synthetic test set and writes `eval_metrics_full.csv`, `eval_*.png` and `report_corr_*.png` next to the checkpoint; without `--checkpoint` it uses `save_root/experiment_name/best.pt`.

`main_enhanced.py` has the same commands but trains a `fourier_conv` variant with soft clamping inside the simulator (the angle term and the lower bound of the reflectivity) and a physics-penalty weight that ramps up over training (`models/common_enhanced.py`, `physics_utils_enhanced.py`).

Refine a measured curve (two whitespace-separated columns: 2θ in degrees, intensity). Normalization is set by `data.normalization_mode` in the config (`plateau`, `max`, `anchor`, `none`) and must match training:

```bash
uv run python -m xrefine.refine.pipeline_next sample.dat --checkpoint path/to/best.pt --out ./results
```

This writes `<name>_<timestamp>_result.png` and `<name>_<timestamp>_params.csv`. `xrefine.refine.pipeline` takes the same arguments; it imports the I0 pre-fit from `pipeline_next`, and the October 2026 fixes went mostly into `pipeline`.

Other scripts:

```bash
uv run python -m xrefine.refine.benchmark -c path/to/best.pt -n 3 -o ./benchmark_results   # refine one fixed + n random synthetic cases
uv run python -m xrefine.hpo --db hpo_fourier.db --trials 100                               # Optuna search for fourier_conv
uv run python convert_checkpoint.py path/to/best.pt                                         # older or Linux checkpoint -> best_win.pt
```

Checkpoints are loaded with `torch.load(weights_only=True)`, so a file holding pickled Python objects is refused instead of run. Those are checkpoints from before the config was stored as JSON, or with Linux paths in them; convert one you made yourself with `convert_checkpoint.py`.

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
tests/                      gradient tests for the enhanced model, forward-model consistency
```

## License

MIT, see [LICENSE](LICENSE).
