import argparse
import sys
from pathlib import Path

# 프로젝트 내부 모듈 임포트
from xrefine.config import CONFIG
from xrefine.train import train_streaming
from xrefine.evaluate import evaluate 

def main():
    parser = argparse.ArgumentParser(
        description="XRefine: AI-Powered X-ray Reflectivity Analysis Toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # --- Train Command ---
    train_parser = subparsers.add_parser("train", help="Start PINN training with dynamic physics scaling")
    train_parser.add_argument("--name", type=str, default=None, help="Experiment name override")
    train_parser.add_argument("--steps", type=int, default=None, help="Total training steps")
    train_parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    train_parser.add_argument("--model", type=str, choices=['pinn', 'fourier'], default=None, help="Model architecture")
    train_parser.add_argument("--layers", type=int, default=None, help="Number of film layers")
    train_parser.add_argument("--warmup", type=int, default=None, help="Warmup steps for curriculum learning")
    
    # --- Evaluate Command ---
    eval_parser = subparsers.add_parser("eval", help="Evaluate model performance & generate reports")
    eval_parser.add_argument("--checkpoint", type=str, default=None, help="Path to best.pt (auto-detects if None)")
    eval_parser.add_argument("--batch_size", type=int, default=1000, help="Number of samples for evaluation")
    
    # --- Config Command ---
    subparsers.add_parser("config", help="Print current system configuration")

    args = parser.parse_args()
    
    if args.command == "train":
        # 1. 베이스 설정 복사 (Pydantic의 이점을 살려 불변성 유지)
        config = CONFIG.model_copy()
        
        # 2. CLI 인자로 설정 덮어쓰기
        if args.name: config.experiment_name = args.name
        if args.lr: config.lr = args.lr
        if args.model: config.model_type = args.model
        if args.layers: config.n_layers = args.layers
        if args.warmup: config.warmup_steps = args.warmup
        
        print(f"🚀 [INIT] Starting Training for '{config.experiment_name}'")
        print(f"📦 [INFO] Model: {config.model_type} | Layers: {config.n_layers} | Steps: {args.steps or config.train_steps}")
        
        try:
            train_streaming(config=config, steps=args.steps)
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user. Progress saved in 'latest.pt'.")
            sys.exit(0)

    elif args.command == "eval":
        checkpoint_path = args.checkpoint
        
        # 3. 체크포인트 자동 탐지 로직 (강화됨)
        if checkpoint_path is None:
            # CONFIG에 설정된 저장 경로에서 가장 최신/베스트 모델 탐색
            base_dir = Path(CONFIG.save_root) / CONFIG.experiment_name
            checkpoint_path = base_dir / "best.pt"
            
            if not checkpoint_path.exists():
                # best.pt가 없으면 latest.pt라도 시도
                checkpoint_path = base_dir / "latest.pt"
                
            if not checkpoint_path.exists():
                print(f"❌ [ERROR] Could not find checkpoint in {base_dir}")
                print("   Please specify manually using: --checkpoint path/to/model.pt")
                sys.exit(1)
            
            print(f"🔍 [AUTO] Detected checkpoint: {checkpoint_path}")
        
        # evaluate 모듈에서 요구하는 형태로 인자 전달
        args.checkpoint = str(checkpoint_path)
        evaluate(args)
        
    elif args.command == "config":
        print("\n--- Current XRefine Configuration ---")
        print(CONFIG.model_dump_json(indent=4))
        print("--------------------------------------\n")
        
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()