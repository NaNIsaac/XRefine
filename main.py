import argparse
import sys
from pathlib import Path
import torch

from xrefine.config import CONFIG, XRefineConfig
# Import functional components
from xrefine.train import train_streaming
from xrefine.evaluate import evaluate # To be implemented/refactored

def main():
    parser = argparse.ArgumentParser(description="XRefine Unified CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # --- Train Command ---
    train_parser = subparsers.add_parser("train", help="Start training a PINN model")
    train_parser.add_argument("--name", type=str, default=None, help="Experiment name override")
    train_parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    train_parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    train_parser.add_argument("--model", type=str, choices=['pinn', 'fourier'], default=None, help="Model type override")
    train_parser.add_argument("--layers", type=int, default=None, help="Number of layers override")
    
    # --- Evaluate Command (Placeholder for now) ---
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pt")
    eval_parser.add_argument("--batch_size", type=int, default=1000, help="Evaluation batch size")
    
    # --- Config Command ---
    config_parser = subparsers.add_parser("config", help="Show current configuration")

    args = parser.parse_args()
    
    if args.command == "train":
        # 1. Load Base Config
        config = CONFIG.model_copy() # Pydantic copy method
        
        # 2. Apply Overrides
        if args.name: config.experiment_name = args.name
        if args.lr: config.lr = args.lr
        if args.model: config.model_type = args.model
        if args.layers: config.n_layers = args.layers
        
        print(f"🚀 Starting Training for '{config.experiment_name}'...")
        train_streaming(config=config, steps=args.steps)
        
    elif args.command == "eval":
        evaluate(args)
        
    elif args.command == "config":
        print(CONFIG.model_dump_json(indent=4))
        
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
