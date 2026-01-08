"""
Hyperparameter Optimization (HPO) for XRefine
=============================================
Transformer-VAE 아키텍처 최적화를 위한 Optuna 스크립트.
"""
import optuna
import numpy as np
from tqdm import tqdm
from pathlib import Path

from xrefine.config import XRefineConfig
from xrefine.train import XRefineTrainer
from xrefine.models import XRR_Transformer_PINN, XRR_Fourier_PINN
from xrefine.logger import logger

import optuna.logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. HPO 상태 저장용 설정
DB_PATH = "hpo_study_transformer.db"
STUDY_NAME = "xrefine_hpo_transformer_v1"
YAML_PATH = Path("config.yaml")

def objective(trial: optuna.Trial):
    # 설정 로드
    if YAML_PATH.exists():
        base_cfg = XRefineConfig.load_yaml(YAML_PATH)
    else:
        from xrefine.config import CONFIG
        base_cfg = CONFIG.model_copy(deep=True)

    # --- Hyperparameter Search Space (Transformer Focus) ---
    
    # 1. Model Architecture
    model_type = base_cfg.model.model_type
    
    if model_type == 'transformer':
        # Transformer Parameters
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        enc_depth = trial.suggest_int("enc_depth", 2, 6) # Layer 개수
        n_heads = trial.suggest_categorical("n_heads", [4, 8])
        dim_ff = trial.suggest_int("dim_feedforward", 256, 1024, step=128)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        
        # Fourier Feature (매우 중요)
        fourier_scale = trial.suggest_float("fourier_scale", 20.0, 60.0)
        
        base_cfg.model.hidden_dim = hidden_dim
        base_cfg.model.encoder_depth = enc_depth
        base_cfg.model.n_heads = n_heads
        base_cfg.model.dim_feedforward = dim_ff
        base_cfg.model.dropout = dropout
        base_cfg.model.fourier_scale = fourier_scale
        
    else:
        # CNN Parameters (Legacy Support)
        enc_depth = trial.suggest_int("enc_depth", 4, 8, step=2)
        n_channels = trial.suggest_categorical("n_channels", [64, 128, 256])
        kernel_size = trial.suggest_int("kernel_size", 15, 31, step=2)
        
        base_cfg.model.encoder_depth = enc_depth
        base_cfg.model.n_channels = n_channels
        base_cfg.model.kernel_size = kernel_size

    # 2. Training Dynamics
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    phys_penalty_w = trial.suggest_float("phys_penalty_w", 1.0, 50.0, log=True)
    
    base_cfg.train.batch_size = batch_size
    base_cfg.train.lr = lr
    base_cfg.train.phys_penalty_w = phys_penalty_w
    
    # 3. Augmentation
    intensity_noise = trial.suggest_float("intensity_noise", 0.05, 0.3)
    base_cfg.augmentations.intensity_noise_scale = intensity_noise

    # Experiment Setup
    base_cfg.experiment_name = f"hpo_{trial.number}"
    base_cfg.train.epochs = 15  # 빠른 탐색을 위해 짧게 설정 (15 Epochs)
    
    # 모델 초기화
    if model_type == 'transformer':
        model = XRR_Transformer_PINN(base_cfg).to(base_cfg.device)
    else:
        model = XRR_Fourier_PINN(base_cfg, device=base_cfg.device).to(base_cfg.device)
    
    trainer = XRefineTrainer(model, config=base_cfg)
    best_metric = float('inf')
    
    desc = f"Trial {trial.number}"
    epoch_pbar = tqdm(range(base_cfg.train.epochs), desc=desc, leave=False)
    
    for epoch in epoch_pbar:
        # 학습
        curr_w = trainer._get_curriculum_weight(epoch)
        train_loss = trainer.train_epoch(epoch, curr_w)
        
        if np.isnan(train_loss):
            return float('inf')
            
        # 검증
        # [FIX] validate가 4개의 값을 반환함 (metrics, p_pred, r_recon, log_r_obs)
        val_metrics, _, _, _ = trainer.validate(curr_w)
        
        # Optimize Target: Param Loss + KL Loss (Total Validation Loss)
        # 물리적 적합도도 중요하지만, 파라미터 정확도가 HPO에서는 더 직관적일 수 있음
        target_metric = val_metrics["loss"] 
        
        # Pruning check
        trial.report(target_metric, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        best_metric = min(best_metric, target_metric)
        epoch_pbar.set_postfix(loss=f"{target_metric:.4f}", best=f"{best_metric:.4f}")

    return best_metric

if __name__ == "__main__":
    storage_url = f"sqlite:///{DB_PATH}"
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_url,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
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

    print("\n" + "="*60)
    try:
        print(f"🏆 BEST (Val Loss): {study.best_value:.6f}")
        print("Params:", study.best_params)
        
        # Best Config 저장
        best_params = study.best_params
        print("💾 Saving best configuration to 'best_hpo_config.yaml'...")
        
        if YAML_PATH.exists():
            best_cfg = XRefineConfig.load_yaml(YAML_PATH)
        else:
            from xrefine.config import CONFIG
            best_cfg = CONFIG.model_copy(deep=True)
            
        # Apply Best Params (Generic Mapping)
        # Config 구조가 복잡하므로 키 이름 매핑 필요
        mapping = {
            'hidden_dim': ('model', 'hidden_dim'),
            'enc_depth': ('model', 'encoder_depth'),
            'n_heads': ('model', 'n_heads'),
            'dim_feedforward': ('model', 'dim_feedforward'),
            'dropout': ('model', 'dropout'),
            'fourier_scale': ('model', 'fourier_scale'),
            'batch_size': ('train', 'batch_size'),
            'lr': ('train', 'lr'),
            'phys_penalty_w': ('train', 'phys_penalty_w'),
            'intensity_noise': ('augmentations', 'intensity_noise_scale')
        }

        for param_name, (section, key) in mapping.items():
            if param_name in best_params:
                # getattr(best_cfg, section) -> best_cfg.model
                # setattr(..., key, value)
                setattr(getattr(best_cfg, section), key, best_params[param_name])

        # Save
        best_cfg.save_yaml("best_hpo_config.yaml")
            
    except ValueError:
        logger.warning("No completed trials found in the database yet.")
    
    print("="*60)