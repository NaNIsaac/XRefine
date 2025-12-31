import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import argparse
import datetime

from xrefine.models.model import XRR_PINN
from xrefine.data.generator import generate_synthetic_batch
from xrefine.config import CONFIG

from xrefine.core.structures import Instrument

def train_streaming(args):
    device = CONFIG['device']
    
    # Setup save directory from CONFIG
    # Use args override if provided (optional), otherwise use CONFIG
    # But user requested "config에 실험 이름과 저장 루트를 적게하자", so prioritize config or make args optional overrides.
    # Let's use config primarily.
    if args.name:
        exp_name = args.name
    else:
        exp_name = CONFIG['experiment_name']
        
    if args.save_dir != "experiments": # Default value check
         root_dir = args.save_dir
    else:
         root_dir = CONFIG['save_root']

    save_dir = Path(root_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    # Save the dictionary directly
    torch.save(CONFIG, save_dir / "config.pt")
    
    # instrument config is already in CONFIG['instrument'] as a dict
    # generator expects 'instrument' key in config dict to be passed to it.
    # However, generator.py logic (to be updated) might expect just the dict or an Instrument object
    # The previous fix converted it to Instrument object.
    # We will update generator.py to accept the dict directly or handle it there.
    # For now, let's keep it simple and pass CONFIG. generator.py will need to handle it.
    
    pinn = XRR_PINN(device=device).to(device)
    optimizer = torch.optim.AdamW(pinn.parameters(), lr=CONFIG['lr'])

    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"🔄 Resuming from {resume_path}...")
            checkpoint = torch.load(resume_path, map_location=device)
            pinn.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step']
        else:
            print(f"⚠️ Checkpoint {resume_path} not found. Starting from scratch.")

    criterion = nn.MSELoss()

    print(f"🔥 PINN Streaming Training Started: {exp_name}")
    print(f"📂 Saving to: {save_dir}")
    
    best_loss = float('inf')
    
    pbar = tqdm(range(start_step, args.steps))
    for step in pbar:
        # Pass CONFIG directly. generator.py will be updated to handle it.
        r_measured, _ = generate_synthetic_batch(CONFIG['batch_size'], CONFIG['q_grid'].to(device), CONFIG)

        optimizer.zero_grad()

        params, r_reconstructed = pinn(torch.log10(r_measured + 1e-12))

        loss = criterion(torch.log10(r_reconstructed + 1e-12), torch.log10(r_measured + 1e-12))

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            current_loss = loss.item()
            pbar.set_description(f"Loss: {current_loss:.6f}")
            
            # Save latest checkpoint
            checkpoint = {
                'step': step,
                'model_state_dict': pinn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                'config': CONFIG
            }
            torch.save(checkpoint, save_dir / "checkpoint_latest.pt")
            
            # Save best checkpoint
            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(checkpoint, save_dir / "checkpoint_best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XRR PINN")
    # Make arguments optional, defaults handled in function via CONFIG
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--save_dir", type=str, default="experiments", help="Directory to save checkpoints")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging and saving interval")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    train_streaming(args)