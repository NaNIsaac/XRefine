"""
XRefine CLI
==========
Main entry point for XRefine using Typer.
"""
import typer
import ast
from pathlib import Path
from typing import List, Optional
from typing_extensions import Annotated

import torch

from xrefine.config import XRefineConfig, CONFIG
from xrefine.train import XRefineTrainer
from xrefine.evaluate import XRREvaluator 
from xrefine.models import get_model
from xrefine.logger import logger

torch.set_float32_matmul_precision('high')

# [Performance] Disable pretty exceptions to prevent browser freeze in Colab
app = typer.Typer(help="XRefine: Physics-Informed AI for XRR Analysis", pretty_exceptions_enable=False)

def merge_unknown_args(config_obj: XRefineConfig, unknown: List[str]) -> XRefineConfig:
    """CLI 인자를 계층적 dict로 변환하여 Config 객체 업데이트"""
    config_dict = config_obj.model_dump() 
    
    for i in range(0, len(unknown), 2):
        if not unknown[i].startswith('--'):
            continue
        
        key = unknown[i].lstrip('-')
        if i + 1 >= len(unknown):
            logger.warning(f"No value provided for option '{key}'")
            continue
            
        value_str = unknown[i+1]
        parts = key.split('.')

        # 'config.' 접두사 무시 (하위 호환성)
        if parts[0] == 'config' and len(parts) > 1:
            parts = parts[1:]

        target = config_dict
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]

        # 자동 타입 변환
        try:
            if value_str.lower() in ('true', 'false'):
                val = value_str.lower() == 'true'
            elif value_str.startswith('[') and value_str.endswith(']'):
                val = ast.literal_eval(value_str)
            elif '.' in value_str or 'e' in value_str:
                val = float(value_str)
            else:
                val = int(value_str)
        except (ValueError, SyntaxError):
            val = value_str
            
        target[parts[-1]] = val
        
    return XRefineConfig(**config_dict)

def format_config_summary(config: XRefineConfig) -> str:
    """현재 세션의 핵심 파라미터 요약 출력"""
    lines: list[str] = []
    line = "=" * 60
    lines.append(f"\n{line}\n[PRE-FLIGHT CHECK: SESSION CONFIGURATION]\n{line}")
    
    lines.append(f"Experiment : {config.experiment_name}")
    lines.append(f"Model Type : {config.model.model_type.upper()}")
    lines.append(f"Device     : {config.device}")
    
    lines.append(f"{'-'*60}")
    lines.append(f"{ 'CATEGORY':<12} | {'PARAMETER':<15} | {'VALUE'}")
    lines.append(f"{'-'*60}")
    lines.append(f"{ 'Training':<12} | {'LR':<15} | {config.train.lr:.1e}")
    lines.append(f"{ 'Training':<12} | {'Batch Size':<15} | {config.train.batch_size}")
    lines.append(f"{ 'Training':<12} | {'Total Epochs':<15} | {config.train.epochs}")
    
    lines.append(f"{'-'*60}")
    lines.append("Sample Structure:")
    for layer in config.sample.layers:
        lines.append(f"  - {layer.name:<10}: d={layer.thickness} Å")
    lines.append(f"Substrate: {config.sample.substrate.name} (SLD={config.sample.substrate.sld})")
    lines.append(f"{line}\n")

    return "\n".join(lines)

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train(
    ctx: typer.Context,
    cfg: Annotated[Path, typer.Option("--cfg", help="Path to config.yaml")] = Path("config.yaml"),
    resume: Annotated[bool, typer.Option("--resume", help="Resume from last checkpoint")] = False,
):
    """Start a PINN training session."""
    try:
        config_obj = XRefineConfig.load_yaml(cfg)
    except FileNotFoundError:
        logger.warning(f"{cfg} not found. Using default CONFIG object.")
        config_obj = CONFIG
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        config_obj = CONFIG

    # Handle unknown arguments for dynamic config overrides
    if ctx.args:
        config_obj = merge_unknown_args(config_obj, ctx.args)

    logger.info("\n" + format_config_summary(config_obj))
    
    save_dir = config_obj.save_root / config_obj.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    config_obj.save_yaml(save_dir / "config.yaml")

    # Initialize Model using Registry
    model = get_model(config_obj).to(config_obj.device)
    
    trainer = XRefineTrainer(model=model, config=config_obj)
    trainer.fit(resume=resume)
    logger.info("Training session completed successfully.")

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def eval(
    ctx: typer.Context,
    checkpoint: Annotated[Optional[str], typer.Option("--checkpoint", help="Path to checkpoint .pt file")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Evaluation batch size")] = 1000,
    cfg: Annotated[Path, typer.Option("--cfg", help="Path to config.yaml")] = Path("config.yaml"),
):
    """Evaluate a trained model on synthetic or real data."""
    try:
        config_obj = XRefineConfig.load_yaml(cfg)
    except Exception:
        config_obj = CONFIG

    if ctx.args:
        config_obj = merge_unknown_args(config_obj, ctx.args)

    # Resolve checkpoint path if not provided
    if checkpoint is None:
        checkpoint = str(config_obj.save_root / config_obj.experiment_name / "best.pt")
    
    if not Path(checkpoint).exists():
        logger.error(f"❌ Checkpoint not found: {checkpoint}")
        return

    logger.info(f"🔍 [EVAL] Using checkpoint: {checkpoint}")
    evaluator = XRREvaluator(checkpoint_path=checkpoint, batch_size=batch_size)
    evaluator.setup()
    evaluator.evaluate_all()

@app.command(name="config")
def config_show():
    """Print the current default configuration in JSON format."""
    print(CONFIG.model_dump_json(indent=2))

if __name__ == "__main__":
    app()
