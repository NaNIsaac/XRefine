# main_enhanced.py
"""
Enhanced Refactoring Entry Point (CLI Integrated)
=================================================
Combines robust Typer CLI from main.py with the Enhanced Backpropagation Model.
"""
import typer
import ast
import torch
from pathlib import Path
from typing import List, Optional
from typing_extensions import Annotated

# Core Imports
from xrefine.config import XRefineConfig, CONFIG
from xrefine.train import XRefineTrainer
from xrefine.evaluate import XRREvaluator 
from xrefine.logger import logger

# Enhanced Model Imports
from xrefine.models.fourier_model import FourierConvEncoder
from xrefine.models.common_enhanced import BasePINNModelEnhanced

torch.set_float32_matmul_precision('high')

app = typer.Typer(
    help="XRefine Enhanced: Physics-Informed AI with Robust Backprop", 
    pretty_exceptions_enable=False
)

# -----------------------------------------------------------------------------
# 1. Enhanced Model Definition
# -----------------------------------------------------------------------------
class FourierConvPINNEnhanced(BasePINNModelEnhanced):
    """
    Fourier+Conv PINN model with enhanced backpropagation.
    
    Combines FourierConvEncoder with enhanced base model for:
    - Soft clamping at boundaries
    - Curriculum learning
    - Dynamic bounds
    """
    def __init__(self, config):
        super().__init__(config)
        self.encoder_net = FourierConvEncoder(config, self.output_dim)
    
    def encode(self, x):
        return self.encoder_net(x)

# -----------------------------------------------------------------------------
# 2. CLI Helpers (from main.py)
# -----------------------------------------------------------------------------
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

        # 'config.' 접두사 무시
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
    lines.append(f"\n{line}\n[ENHANCED SESSION CONFIGURATION]\n{line}")
    
    lines.append(f"Experiment : {config.experiment_name}")
    lines.append(f"Model Class: FourierConvPINNEnhanced (Forced)")
    lines.append(f"Device     : {config.device}")
    
    lines.append(f"{'-'*60}")
    lines.append(f"Enhanced Features Active:")
    lines.append(f"  ✓ Soft clamping for gradient preservation")
    lines.append(f"  ✓ Curriculum learning for progressive constraints")
    lines.append(f"  ✓ Dynamic bounds for physical validity")
    lines.append(f"  ✓ Robust numerical operations")
    
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
    lines.append(f"Substrate: {config.sample.substrate.name}")
    lines.append(f"{line}\n")

    return "\n".join(lines)

# -----------------------------------------------------------------------------
# 3. CLI Commands
# -----------------------------------------------------------------------------
@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train(
    ctx: typer.Context,
    cfg: Annotated[Path, typer.Option("--cfg", help="Path to config.yaml")] = Path("config.yaml"),
    resume: Annotated[bool, typer.Option("--resume", help="Resume from last checkpoint")] = False,
):
    """Start an Enhanced PINN training session."""
    # 1. Load Config
    try:
        config_obj = XRefineConfig.load_yaml(cfg)
    except FileNotFoundError:
        logger.warning(f"{cfg} not found. Using default CONFIG object.")
        config_obj = CONFIG
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        config_obj = CONFIG

    # 2. Merge CLI Args
    if ctx.args:
        config_obj = merge_unknown_args(config_obj, ctx.args)

    logger.info("\n" + format_config_summary(config_obj))
    
    # 3. Setup Save Dir
    save_dir = config_obj.save_root / config_obj.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    config_obj.save_yaml(save_dir / "config.yaml")

    # 4. Initialize Enhanced Model
    logger.info("Initializing FourierConvPINNEnhanced model...")
    model = FourierConvPINNEnhanced(config_obj).to(config_obj.device)
    
    # 5. Start Training
    trainer = XRefineTrainer(model=model, config=config_obj)
    
    # Check for set_epoch support just in case
    if not hasattr(trainer, 'train_epoch'):
        logger.warning("Trainer may need modification to fully support curriculum learning.")

    trainer.fit(resume=resume)
    logger.info("Enhanced training session completed successfully.")

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def eval(
    ctx: typer.Context,
    checkpoint: Annotated[Optional[str], typer.Option("--checkpoint", help="Path to checkpoint .pt file")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Evaluation batch size")] = 1000,
    cfg: Annotated[Path, typer.Option("--cfg", help="Path to config.yaml")] = Path("config.yaml"),
):
    """Evaluate a trained model."""
    try:
        config_obj = XRefineConfig.load_yaml(cfg)
    except Exception:
        config_obj = CONFIG

    if ctx.args:
        config_obj = merge_unknown_args(config_obj, ctx.args)

    if checkpoint is None:
        checkpoint = str(config_obj.save_root / config_obj.experiment_name / "best.pt")
    
    if not Path(checkpoint).exists():
        logger.error(f"❌ Checkpoint not found: {checkpoint}")
        return

    logger.info(f"🔍 [EVAL] Using checkpoint: {checkpoint}")
    
    # Note: XRREvaluator uses standard get_model internally. 
    # Ideally, checkpoint loading handles architecture mismatch if keys align.
    evaluator = XRREvaluator(checkpoint_path=checkpoint, batch_size=batch_size)
    evaluator.setup()
    evaluator.evaluate_all()

@app.command(name="config")
def config_show():
    """Print the current default configuration in JSON format."""
    print(CONFIG.model_dump_json(indent=2))

if __name__ == "__main__":
    app()