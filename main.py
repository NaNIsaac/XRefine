import argparse
import sys
from pathlib import Path
from pydantic import BaseModel, Field

from xrefine.config import CONFIG
from xrefine.train import XRefineTrainer
from xrefine.evaluate import evaluate 
from xrefine.models import XRR_Fourier_PINN
from xrefine.logger import logger

class CLIArgs(BaseModel):
    experiment_name: str | None = Field(None, alias="name")
    lr: float | None = None
    n_layers: int | None = Field(None, alias="layers")
    epochs: int = 200
    steps_per_epoch: int = 100
    resume: bool = False


def run_train(args_ns: argparse.Namespace):
    # Namespace를 Pydantic 모델로 변환하여 유효성 검사 및 업데이트
    cli_params = CLIArgs(**vars(args_ns)).model_dump(exclude_none=True, by_alias=False)
    config = CONFIG.model_copy(update=cli_params)

    save_dir = Path(config.save_root) / config.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(save_dir / "config.yaml")

    logger.info(f"[INIT] {config.experiment_name} | Layers: {config.n_layers} | Device: {config.device}")

    # 모델 및 트레이너 초기화
    model = XRR_Fourier_PINN(config=config, device=config.device)
    trainer = XRefineTrainer(model=model, config=config)

    try:
        trainer.fit(
            max_epochs=args_ns.epochs, 
            steps_per_epoch=args_ns.steps_per_epoch, 
            resume=args_ns.resume
        )
    except KeyboardInterrupt:
        logger.info("\n[STOP] Training paused by user.")
        sys.exit(0)

def run_eval(args_ns: argparse.Namespace):
    if not args_ns.checkpoint:
        args_ns.checkpoint = str(CONFIG.save_root / CONFIG.experiment_name / "best_pinn.pt")
    
    if not Path(args_ns.checkpoint).exists():
        logger.info(f"Error: Checkpoint not found at {args_ns.checkpoint}")
        sys.exit(1)
        
    evaluate(args_ns)


def main():
    parser = argparse.ArgumentParser(description="XRefine: Physics-Informed AI for XRR")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train Command
    train_p = subparsers.add_parser("train", help="Start PINN training")
    train_p.add_argument("--name", type=str, help="Experiment name")
    train_p.add_argument("--epochs", type=int, default=200)
    train_p.add_argument("--steps_per_epoch", type=int, default=100)
    train_p.add_argument("--lr", type=float)
    train_p.add_argument("--layers", type=int, help="Number of film layers")
    train_p.add_argument("--resume", action="store_true", help="Resume from latest.pt")

    # Eval Command
    eval_p = subparsers.add_parser("eval", help="Evaluate model")
    eval_p.add_argument("--checkpoint", type=str)
    eval_p.add_argument("--batch_size", type=int, default=1000)

    # Config Command
    subparsers.add_parser("config", help="Show current configuration")

    args = parser.parse_args()

    match args.command:
        case "train":
            run_train(args)
        case "eval":
            run_eval(args)
        case "config":
            logger.info(CONFIG.model_dump_json(indent=2))
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()