import argparse
import sys
from pathlib import Path

from xrefine.config import CONFIG
from xrefine.train import XRefineTrainer
from xrefine.evaluate import evaluate 
from xrefine.models import XRR_Fourier_PINN

def setup_training(args):
    """훈련을 위한 모델, 시뮬레이터, 트레이너 조립"""
    # 1. Config 병합 (Pydantic)
    config = CONFIG.model_copy()
    if args.name: config.experiment_name = args.name
    if args.lr: config.lr = args.lr
    if args.layers: config.n_layers = args.layers
    if args.warmup: config.warmup_steps = args.warmup

    save_dir = Path(config.save_root) / config.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    config.save_yaml(save_dir / "config.yaml")

    print(f"🚀 [INIT] Starting XRefine PINN for '{config.experiment_name}'")
    print(f"📦 [INFO] Layers: {config.n_layers} | Device: {config.device}")

    # 2. PINN 모델 초기화 (내부에 Simulator 및 Mapper 포함)
    model = XRR_Fourier_PINN(config=config, device=config.device)

    # 3. 현대적 트레이너 초기화
    trainer = XRefineTrainer(
        model=model,
        config=config,
        save_dir=save_dir,
        patience=30
    )

    return trainer, config

def main():
    parser = argparse.ArgumentParser(
        description="XRefine: Modern Physics-Informed AI for XRR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # --- Train Command ---
    train_parser = subparsers.add_parser("train", help="Start Modern PINN Training")
    train_parser.add_argument("--name", type=str, help="Experiment name")
    train_parser.add_argument("--epochs", type=int, default=200, help="Max epochs")
    train_parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch")
    train_parser.add_argument("--lr", type=float, help="Learning rate")
    train_parser.add_argument("--layers", type=int, help="Number of film layers")
    train_parser.add_argument("--warmup", type=int, help="Warmup steps")

    # --- Evaluate Command ---
    eval_parser = subparsers.add_parser("eval", help="Evaluate & Report")
    eval_parser.add_argument("--checkpoint", type=str, help="Path to .pt file")
    eval_parser.add_argument("--batch_size", type=int, default=1000)

    # --- Config Command ---
    subparsers.add_parser("config", help="Print system configuration")

    args = parser.parse_args()

    if args.command == "train":
        trainer, config = setup_training(args)
        try:
            # 8-파라미터 및 q-weighted loss가 적용된 학습 시작
            trainer.fit(max_epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)
        except KeyboardInterrupt:
            print("\n🛑 Training paused. Checkpoints are safe in the experiment directory.")
            sys.exit(0)

    elif args.command == "eval":
        # 체크포인트 자동 탐색 로직 유지
        if not args.checkpoint:
            base_dir = Path(CONFIG.save_root) / CONFIG.experiment_name
            args.checkpoint = str(base_dir / "best.pt")
            
            if not Path(args.checkpoint).exists():
                print(f"❌ [ERROR] Checkpoint not found at {args.checkpoint}")
                sys.exit(1)
        
        evaluate(args)

    elif args.command == "config":
        print("\n--- Current XRefine Configuration ---")
        print(CONFIG.model_dump_json(indent=4))
        print("--------------------------------------\n")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()