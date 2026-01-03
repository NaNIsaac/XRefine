import argparse
from pathlib import Path
from typing import List, Dict, Any

from xrefine.config import XRefineConfig, CONFIG
from xrefine.train import XRefineTrainer
from xrefine.evaluate import XRREvaluator 
from xrefine.models import XRR_Fourier_PINN
from xrefine.logger import logger

def merge_unknown_args(config_dict: Dict[str, Any], unknown: List[str]) -> Dict[str, Any]:
    """CLI 인자를 계층적 dict로 변환 (config. 접두사 자동 처리)"""
    for i in range(0, len(unknown), 2):
        if not unknown[i].startswith('--'):
            continue
        
        key = unknown[i].lstrip('-')
        value_str = unknown[i+1]
        parts = key.split('.')

        if parts[0] == 'config' and len(parts) > 1:
            parts = parts[1:]

        target = config_dict
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]

        try:
            if value_str.lower() in ('true', 'false'):
                val = value_str.lower() == 'true'
            elif '[' in value_str:
                import ast
                val = ast.literal_eval(value_str)
            elif '.' in value_str or 'e' in value_str:
                val = float(value_str)
            else:
                val = int(value_str)
        except (ValueError, SyntaxError):
            val = value_str
            
        target[parts[-1]] = val
    return config_dict

def print_config_summary(config: XRefineConfig):
    """현재 세션의 핵심 파라미터를 표 형태로 출력"""
    line = "=" * 60
    logger.info(f"\n{line}\n[PRE-FLIGHT CHECK: SESSION CONFIGURATION]\n{line}")
    
    # 1. 기본 정보
    logger.info(f"Experiment : {config.experiment_name}")
    logger.info(f"Save Path  : {config.save_root / config.experiment_name}")
    logger.info(f"Device     : {config.device}")
    
    # 2. 하이퍼파라미터 (Table Style)
    logger.info(f"{'-'*60}")
    logger.info(f"{'CATEGORY':<12} | {'PARAMETER':<15} | {'VALUE'}")
    logger.info(f"{'-'*60}")
    logger.info(f"{'Training':<12} | {'LR':<15} | {config.train.lr:.1e}")
    logger.info(f"{'Training':<12} | {'Batch Size':<15} | {config.train.batch_size}")
    logger.info(f"{'Training':<12} | {'Total Epochs':<15} | {config.train.epochs}")
    logger.info(f"{'Training':<12} | {'Sup Weight':<15} | {config.train.sup_w_start} -> {config.train.sup_w_end}")
    
    logger.info(f"{'Model':<12} | {'N Layers':<15} | {config.model.n_layers}")
    logger.info(f"{'Model':<12} | {'Fourier Scale':<15} | {config.model.fourier_scale}")
    
    # 3. 물리 범위 요약 (핵심만)
    logger.info(f"{'-'*60}")
    logger.info("Physics Ranges (First Layer if N>1):")
    logger.info(f"   - Thickness (d_f) : {config.ranges.d_f} [Å]")
    logger.info(f"   - Roughness(sig_f): {config.ranges.sig_f} [Å]")
    logger.info(f"   - SLD (rho_f)     : {config.ranges.rho_f} [10^-6 Å^-2]")
    logger.info(f"{line}\n")

def run_train(config: XRefineConfig, resume: bool):
    """학습 세션 실행"""
    # [Added] 요약 출력
    print_config_summary(config)
    
    save_dir = config.save_root / config.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 설정 백업
    config.save_yaml(save_dir / "config.yaml")

    model = XRR_Fourier_PINN(config=config, device=config.device)
    trainer = XRefineTrainer(model=model, config=config)
    trainer.fit(resume=resume)
    logger.info("Training session completed successfully.")

def run_eval(config: XRefineConfig, checkpoint: str | None, batch_size: int):
    """평가 세션 실행"""
    ckpt_path = checkpoint or str(config.save_root / config.experiment_name / "best.pt")
    if not Path(ckpt_path).exists():
        logger.error(f"❌ Checkpoint not found: {ckpt_path}")
        return

    logger.info(f"🔍 [EVAL] Using checkpoint: {ckpt_path}")
    evaluator = XRREvaluator(checkpoint_path=ckpt_path, batch_size=batch_size)
    evaluator.setup()
    evaluator.evaluate_all()

def main():
    parser = argparse.ArgumentParser(description="XRefine: Physics-Informed AI for XRR")
    parser.add_argument("subcommand", choices=["train", "eval", "config"], help="작업 모드 선택")
    parser.add_argument("--cfg", type=str, default="config.yaml", help="기본 설정 파일 경로")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--batch_size", type=int, default=1000)
    
    args, unknown = parser.parse_known_args()

    try:
        config_obj = XRefineConfig.load_yaml(args.cfg)
    except FileNotFoundError:
        logger.warning(f"{args.cfg} not found. Using default CONFIG object.")
        config_obj = CONFIG

    if unknown:
        config_dict = config_obj.model_dump()
        updated_dict = merge_unknown_args(config_dict, unknown)
        config_obj = XRefineConfig(**updated_dict)

    match args.subcommand:
        case "train":
            run_train(config_obj, args.resume)
        case "eval":
            run_eval(config_obj, args.checkpoint, args.batch_size)
        case "config":
            print(config_obj.model_dump_json(indent=2))
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()