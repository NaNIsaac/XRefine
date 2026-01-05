import optuna
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from xrefine.config import XRefineConfig
from xrefine.train import XRefineTrainer
from xrefine.models.fourier_model import XRR_Fourier_PINN
from xrefine.logger import logger

# Optuna 로그 설정
import optuna.logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. HPO 상태 저장용 설정
DB_PATH = "hpo_study.db"  # 이 파일에 튜닝 이력이 저장됩니다.
STUDY_NAME = "xrefine_hpo_v1"
YAML_PATH = Path("config.yaml")

def objective(trial: optuna.Trial):
    # 설정 로드 및 파라미터 샘플링 (기존과 동일)
    if YAML_PATH.exists():
        base_cfg = XRefineConfig.load_yaml(YAML_PATH)
    else:
        from xrefine.config import CONFIG
        base_cfg = CONFIG.model_copy(deep=True)

    f_scale = trial.suggest_float("f_scale", 10.0, 100.0)
    phys_w = trial.suggest_float("phys_w", 0.01, 1.0, log=True)
    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)

    base_cfg.model.fourier_scale = f_scale
    base_cfg.train.phys_penalty_w = phys_w
    base_cfg.train.lr = lr
    base_cfg.experiment_name = f"hpo_trial_{trial.number}"
    base_cfg.train.epochs = 15 

    # 모델 및 트레이너 초기화
    model = XRR_Fourier_PINN(base_cfg)
    model.to(base_cfg.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    trainer = XRefineTrainer(model, config=base_cfg)
    best_val_loss = float('inf')
    
    desc = f"Trial {trial.number} (fs:{f_scale:.1f}, pw:{phys_w:.2f})"
    epoch_pbar = tqdm(range(base_cfg.train.epochs), desc=desc, leave=False)
    
    for epoch in epoch_pbar:
        curr_w = trainer._get_curriculum_weight(epoch)
        train_loss = trainer.train_epoch(epoch, curr_w)
        
        if np.isnan(train_loss):
            return float('inf')
            
        val_loss, _, _ = trainer.validate(curr_w)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        best_val_loss = min(best_val_loss, val_loss)
        epoch_pbar.set_postfix(val=f"{val_loss:.4f}", best=f"{best_val_loss:.4f}")

    return best_val_loss

if __name__ == "__main__":
    # 2. SQLite DB를 저장소로 지정하여 Study 생성 또는 로드
    # storage="sqlite:///파일명.db" 형식을 사용합니다.
    storage_url = f"sqlite:///{DB_PATH}"
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_url,
        direction="minimize",
        load_if_exists=True, # 이미 존재하면 이어서 하기
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )

    # 현재까지 완료된 Trial 수 확인
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info(f"🟢 HPO Session: {STUDY_NAME} | Already Completed: {completed_trials}")
    
    # 총 50회를 목표로 할 때, 남은 횟수만큼 추가 진행
    total_target_trials = 50
    remaining_trials = max(0, total_target_trials - completed_trials)

    if remaining_trials > 0:
        logger.info(f"🔥 Running {remaining_trials} more trials (Total Target: {total_target_trials})...")
        try:
            study.optimize(objective, n_trials=remaining_trials)
        except KeyboardInterrupt:
            logger.info("🛑 Optimization paused by user. Progress saved in DB.")
    else:
        logger.info("✅ All target trials are already completed.")

    # 최종 결과 출력
    print("\n" + "="*60)
    print(f"🏆 BEST TRIAL: #{study.best_trial.number}")
    print(f"Value: {study.best_value:.6f}")
    print("Params:", study.best_params)
    print("="*60)