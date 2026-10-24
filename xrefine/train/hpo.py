from pathlib import Path

import numpy as np
import optuna
import optuna.logging
import torch
import yaml
from tqdm import tqdm

from xrefine.config import XRefineConfig
from xrefine.logger import logger
from xrefine.models.fourier_model import FourierConvPINN
from xrefine.train import XRefineTrainer

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. HPO 상태 저장용 설정
DB_PATH = "hpo_study.db"
STUDY_NAME = "xrefine_hpo_expanded_v1"
YAML_PATH = Path("config.yaml")


def objective(trial: optuna.Trial):
    # 설정 로드 및 파라미터 샘플링
    if YAML_PATH.exists():
        base_cfg = XRefineConfig.load_yaml(YAML_PATH)
    else:
        from xrefine.config import CONFIG

        base_cfg = CONFIG.model_copy(deep=True)

    # --- Hyperparameter Search Space ---

    # 1. New Exploration: Model Architecture (Concentrate here)
    enc_depth = trial.suggest_int("enc_depth", 4, 8, step=2)
    n_channels = trial.suggest_categorical("n_channels", [64, 128, 256])
    kernel_size = trial.suggest_int("kernel_size", 15, 31, step=2)
    hidden_dim = trial.suggest_categorical("hidden_dim", [512, 1024])

    # [LOCKED] Previously Optimized Params (Using values from config.yaml)
    # f_scale = 43.0, lr = 5e-4, phys_penalty_w = 0.86
    # We trust these values and focus computing power on architecture.

    base_cfg.model.encoder_depth = enc_depth
    base_cfg.model.n_channels = n_channels
    base_cfg.model.kernel_size = kernel_size
    base_cfg.model.hidden_dim = hidden_dim

    # 2. Training Dynamics
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    sup_w_end = trial.suggest_float("sup_w_end", 1000.0, 5000.0, log=True)

    base_cfg.train.batch_size = batch_size
    base_cfg.train.sup_w_end = sup_w_end

    # 3. Augmentation
    intensity_noise = trial.suggest_float("intensity_noise", 0.05, 0.4)
    base_cfg.augmentations.intensity_noise_scale = intensity_noise

    # Experiment Setup
    base_cfg.experiment_name = f"hpo_{trial.number}"
    base_cfg.train.epochs = 15  # 에폭 약간 증가

    # 모델 및 트레이너 초기화
    model = FourierConvPINN(base_cfg)
    model.to(base_cfg.device)

    trainer = XRefineTrainer(model, config=base_cfg)
    best_metric = float("inf")

    desc = f"Trial {trial.number}"
    epoch_pbar = tqdm(range(base_cfg.train.epochs), desc=desc, leave=False)

    for epoch in epoch_pbar:
        # 학습
        curr_w = trainer._get_curriculum_weight(epoch)
        train_loss = trainer.train_epoch(epoch, curr_w)

        if np.isnan(train_loss):
            return float("inf")

        # 검증 (Parameter Loss를 메인 메트릭으로 사용)
        metrics, _, _ = trainer.validate(curr_w)
        val_param_error = metrics["loss_param"]

        # Pruning check
        trial.report(val_param_error, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        best_metric = min(best_metric, val_param_error)
        epoch_pbar.set_postfix(mae=f"{val_param_error:.4f}", best=f"{best_metric:.4f}")

    return best_metric


if __name__ == "__main__":
    storage_url = f"sqlite:///{DB_PATH}"

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_url,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info(f"🟢 HPO Session: {STUDY_NAME} | Already Completed: {completed_trials}")

    total_target_trials = 100
    remaining_trials = max(0, total_target_trials - completed_trials)

    if remaining_trials > 0:
        logger.info(f"🔥 Running {remaining_trials} more trials...")
        try:
            study.optimize(objective, n_trials=remaining_trials)
        except KeyboardInterrupt:
            logger.info("🛑 Optimization paused by user.")

    print("\n" + "=" * 60)
    try:
        print(f"🏆 BEST (Param MAE): {study.best_value:.6f}")
        print("Params:", study.best_params)

        # Best Config 저장
        best_params = study.best_params
        print("💾 Saving best configuration to 'best_hpo_config.yaml'...")

        if YAML_PATH.exists():
            best_cfg = XRefineConfig.load_yaml(YAML_PATH)
        else:
            from xrefine.config import CONFIG

            best_cfg = CONFIG.model_copy(deep=True)

        # Apply Best Params (Only Optimized Ones)
        best_cfg.model.encoder_depth = best_params.get("enc_depth", best_cfg.model.encoder_depth)
        best_cfg.model.n_channels = best_params.get("n_channels", best_cfg.model.n_channels)
        best_cfg.model.kernel_size = best_params.get("kernel_size", best_cfg.model.kernel_size)
        best_cfg.model.hidden_dim = best_params.get("hidden_dim", best_cfg.model.hidden_dim)

        best_cfg.train.batch_size = best_params.get("batch_size", best_cfg.train.batch_size)
        best_cfg.train.sup_w_end = best_params.get("sup_w_end", best_cfg.train.sup_w_end)

        best_cfg.augmentations.intensity_noise_scale = best_params.get(
            "intensity_noise", best_cfg.augmentations.intensity_noise_scale
        )

        # Save
        with open("best_hpo_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(best_cfg.model_dump(mode="json"), f, sort_keys=False)

    except ValueError:
        logger.warning("No completed trials found in the database yet.")

    print("=" * 60)
