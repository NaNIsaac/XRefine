import argparse
from pathlib import Path
from typing import List, Dict, Any

from xrefine.config import XRefineConfig, CONFIG
from xrefine.train import XRefineTrainer
from xrefine.evaluate import XRREvaluator 
from xrefine.models import XRR_Fourier_PINN, XRR_Transformer_PINN
from xrefine.logger import logger

def get_model(config: XRefineConfig):
    """Config의 model_type에 따라 적절한 모델 인스턴스 반환"""
    model_type = config.model.model_type.lower()
    
    if model_type == 'transformer':
        logger.info("Initializing Model: Transformer-VAE PINN")
        return XRR_Transformer_PINN(config=config).to(config.device)
    
    elif model_type == 'cnn':
        logger.info("Initializing Model: Fourier-CNN PINN")
        return XRR_Fourier_PINN(config=config, device=config.device).to(config.device)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'cnn' or 'transformer'.")

def merge_unknown_args(config_dict: Dict[str, Any], unknown: List[str]) -> Dict[str, Any]:
    """CLI 인자를 계층적 dict로 변환 (config. 접두사 자동 처리)"""
    for i in range(0, len(unknown), 2):
        if not unknown[i].startswith('--'): continue
        
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

def format_config_summery(config: XRefineConfig) -> str:
    """현재 세션의 핵심 파라미터 요약 출력"""
    lines: list[str] = []
    line = "=" * 60
    lines.append(f"\n{line}\n[PRE-FLIGHT CHECK: SESSION CONFIGURATION]\n{line}")
    
    lines.append(f"Experiment : {config.experiment_name}")
    lines.append(f"Model Type : {config.model.model_type.upper()}")
    lines.append(f"Device     : {config.device}")
    
    lines.append(f"{'-'*60}")
    lines.append(f"{'CATEGORY':<12} | {'PARAMETER':<15} | {'VALUE'}")
    lines.append(f"{'-'*60}")
    lines.append(f"{'Training':<12} | {'LR':<15} | {config.train.lr:.1e}")
    lines.append(f"{'Training':<12} | {'Batch Size':<15} | {config.train.batch_size}")
    lines.append(f"{'Training':<12} | {'Total Epochs':<15} | {config.train.epochs}")
    
    # 모델 타입에 따른 파라미터 출력 분기
    if config.model.model_type == 'transformer':
        lines.append(f"{'Model':<12} | {'Hidden Dim':<15} | {config.model.hidden_dim}")
        lines.append(f"{'Model':<12} | {'Enc Depth':<15} | {config.model.encoder_depth}")
        lines.append(f"{'Model':<12} | {'Heads':<15} | {config.model.n_heads}")
    else:
        lines.append(f"{'Model':<12} | {'Mapping Size':<15} | {config.model.mapping_size}")
        lines.append(f"{'Model':<12} | {'Fourier Scale':<15} | {config.model.fourier_scale}")
    
    lines.append(f"{'-'*60}")
    lines.append(f"Physics Ranges (d_f) : {config.ranges.d_f} [Å]")
    lines.append(f"{line}\n")

    return "\n".join(lines)

def run_train(config: XRefineConfig, resume: bool):
    """학습 세션 실행"""
    logger.info("\n" + format_config_summery(config))
    
    save_dir = config.save_root / config.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    config.save_yaml(save_dir / "config.yaml")

    # [Refactored] 모델 생성 팩토리 사용
    model = get_model(config)
    
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
    
    # Evaluator 내부에서 Checkpoint의 config를 로드하여 적절한 모델을 재구성한다고 가정
    # (Evaluator가 config.yaml을 로드하거나 ckpt['config']를 사용해야 함)
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