"""
Hyperparameter Optimization (HPO) for Fourier-Conv Model
======================================================
Focus: Optimizing Fourier-Convolutional Neural Network Architecture.
Refactored to use Pydantic config structure cleanly without warnings.
Now supports custom DB path via CLI arguments.
"""
import optuna
import torch
import gc
import numpy as np
import argparse
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict
from pydantic import BaseModel

from xrefine.config import XRefineConfig, CONFIG as DEFAULT_CONFIG
from xrefine.train import XRefineTrainer
from xrefine.models import get_model 
from xrefine.logger import logger

# Optuna Logging Level
import optuna.logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -----------------------------------------------------------------------------
# Global Settings
# -----------------------------------------------------------------------------
STUDY_NAME = "xrefine_fourier_conv_opt_v1"
YAML_PATH = Path("config.yaml")

def update_config_recursively(config_obj: Any, params: Dict[str, Any]):
    """
    Optuna params(flat dict)를 Pydantic Config(nested object)에 매핑하는 스마트 함수.
    [FIX] Pydantic V2 Warning 해결: .model_fields 접근 시 인스턴스가 아닌 클래스 사용.
    """
    for param_name, param_value in params.items():
        found = False
        
        # 1. Root Level (e.g. experiment_name)
        if hasattr(config_obj, param_name):
            setattr(config_obj, param_name, param_value)
            found = True
        
        # 2. Nested Level (model, train, augmentations)
        elif isinstance(config_obj, BaseModel):
            for field_name in type(config_obj).model_fields:
                sub_obj = getattr(config_obj, field_name)
                
                if isinstance(sub_obj, BaseModel) and hasattr(sub_obj, param_name):
                    setattr(sub_obj, param_name, param_value)
                    found = True
                    break
        
        if not found and "w_" not in param_name: 
            pass

def objective(trial: optuna.Trial):
    # 1. Config 로드 (기본값 복사)
    base_cfg = XRefineConfig.load_yaml(YAML_PATH) if YAML_PATH.exists() else deepcopy(DEFAULT_CONFIG)

    # 2. Fourier-Conv Dedicated Search Space
    # -----------------------------------------------------
    hparams = {}
    
    # [A] Model Architecture
    hparams['hidden_dim'] = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    hparams['encoder_depth'] = trial.suggest_int("encoder_depth", 4, 8)
    hparams['kernel_size'] = trial.suggest_categorical("kernel_size", [11, 15, 21, 31])
    hparams['dilation'] = trial.suggest_categorical("dilation", [1, 2])
    
    # Fourier Feature
    hparams['fourier_scale'] = trial.suggest_float("fourier_scale", 20.0, 80.0)
    hparams['mapping_size'] = trial.suggest_categorical("mapping_size", [32, 64, 128])

    # [B] Training Dynamics
    hparams['lr'] = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    hparams['batch_size'] = trial.suggest_categorical("batch_size", [128, 256, 512])
    hparams['phys_penalty_w'] = trial.suggest_float("phys_penalty_w", 0.5, 10.0)
    
    # [C] Loss Weights
    hparams['w_gradient'] = trial.suggest_float("w_gradient", 10.0, 100.0)

    # [D] Augmentation
    hparams['intensity_noise_scale'] = trial.suggest_float("intensity_noise_scale", 0.05, 0.2)
    
    # -----------------------------------------------------
    
    # 3. Apply Params
    update_config_recursively(base_cfg, hparams)
    
    # HPO용 강제 설정
    base_cfg.model.model_type = 'fourier_conv'
    base_cfg.experiment_name = f"optuna_fconv_{trial.number}"
    base_cfg.train.epochs = 15
    base_cfg.save_root = Path("./hpo_temp") 
    
    # 4. Init & Run
    try:
        model = get_model(base_cfg).to(base_cfg.device)
        trainer = XRefineTrainer(model, config=base_cfg)
    except Exception as e:
        logger.error(f"Trial {trial.number} Init Failed: {e}")
        return float('inf')

    best_val_loss = float('inf')
    
    try:
        for epoch in range(1, base_cfg.train.epochs + 1):
            curr_w = trainer._get_curriculum_weight(epoch)
            
            # Train
            train_loss = trainer.train_epoch(epoch, curr_w)
            if np.isnan(train_loss):
                raise optuna.exceptions.TrialPruned("NaN Loss")

            # Validate
            val_metrics, _, _, _ = trainer.validate(curr_w)
            val_loss = val_metrics['loss']
            
            # Report & Prune
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            best_val_loss = min(best_val_loss, val_loss)

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {e}")
        return float('inf')
    finally:
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    return best_val_loss

def run_hpo(n_trials=50, db_path="hpo_fourier.db"):
    # [NEW] DB 경로 처리
    db_file = Path(db_path)
    # 부모 디렉토리가 없으면 생성 (경로 에러 방지)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    storage_url = f"sqlite:///{db_file.resolve()}"
    logger.info(f"🚀 Starting Fourier-Conv HPO: {STUDY_NAME}")
    logger.info(f"💾 Database Path: {db_file.resolve()}")
    
    pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=15, reduction_factor=3)
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_url,
        direction="minimize",
        load_if_exists=True,
        pruner=pruner
    )

    logger.info(f"📊 Previous Trials: {len(study.trials)}")

    try:
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user.")

    print("\n" + "="*60)
    try:
        print(f"🏆 Best Loss: {study.best_value:.6f}")
        print("✨ Best Params:")
        for k, v in study.best_params.items():
            print(f"  - {k}: {v}")
        
        save_best_config(study.best_params)
    except ValueError:
        logger.warning("No trials completed.")
    print("="*60)

def save_best_config(best_params):
    if YAML_PATH.exists():
        final_cfg = XRefineConfig.load_yaml(YAML_PATH)
    else:
        final_cfg = deepcopy(DEFAULT_CONFIG)
    
    update_config_recursively(final_cfg, best_params)
    final_cfg.model.model_type = 'fourier_conv'
    final_cfg.experiment_name += "_optimized"
    
    out_path = Path("best_fourier_config.yaml")
    final_cfg.save_yaml(out_path)
    print(f"💾 Saved to: {out_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="XRefine Hyperparameter Optimization")
    parser.add_argument("--db", type=str, default="hpo_fourier.db", help="Path to SQLite DB file (e.g., D:/data/hpo.db)")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials to run")
    
    args = parser.parse_args()
    
    run_hpo(n_trials=args.trials, db_path=args.db)