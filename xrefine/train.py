import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import argparse

from xrefine.models import XRR_PINN, XRR_Fourier_PINN
from xrefine.data.generator import generate_synthetic_batch
from xrefine.data.augmentations import XRRAugmentations
from xrefine.config import CONFIG, XRefineConfig

def train_streaming(config: XRefineConfig, save_dir: Path = None, steps: int = None):
    """
    Main training loop.
    Args:
        config: XRefineConfig object
        save_dir: Optional override for save directory
        steps: Optional override for steps
    """
    device = config.device
    
    # 1. Setup Directories
    exp_name = config.experiment_name
    if save_dir is None:
        save_dir = Path(config.save_root) / exp_name
    else:
        save_dir = save_dir / exp_name
        
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Configuration (As JSON/Text representation since it's Pydantic)
    with open(save_dir / "config.json", "w") as f:
        f.write(config.model_dump_json(indent=4))
    
    # 2. Initialize Model
    if config.model_type == 'fourier':
        print(f"🧠 Using Fourier Feature PINN Model ({config.n_layers} layers)")
        pinn = XRR_Fourier_PINN(config=config, device=device).to(device)
    else:
        print(f"🧠 Using Standard CNN PINN Model ({config.n_layers} layers)")
        pinn = XRR_PINN(config=config, device=device).to(device)
        
    optimizer = torch.optim.AdamW(pinn.parameters(), lr=config.lr)
    
    # 3. Setup Augmenter (Dependency Injection)
    # Calculate delta_q
    q_grid = config.q_grid.to(device)
    if q_grid.ndim > 0:
        delta_q = (q_grid[1] - q_grid[0]).item()
    else:
        delta_q = 0.0005
        
    aug_cfg = config.augmentations
    augmenter = XRRAugmentations(
        delta_q=delta_q,
        intensity_noise_scale=aug_cfg.intensity_noise_scale,
        bg_range=aug_cfg.bg_range,
        res_sigma_range=aug_cfg.res_sigma_range,
        q_shift_sigma=aug_cfg.q_shift_sigma,
        beam_profile=aug_cfg.beam_profile,
        do_footprint=True # Always True for now, or add to config
    )

    criterion = nn.MSELoss()
    
    # Resume Logic (Can be added back, simplified here for Refactor focus)
    start_step = 0
    
    print(f"🔥 PINN Streaming Training Started: {exp_name}")
    print(f"📂 Saving to: {save_dir}")
    
    best_loss = float('inf')
    
    total_steps = steps if steps else 10000
    pbar = tqdm(range(start_step, total_steps))
    
    for step in pbar:
        # Pass Config and Augmenter
        r_measured, _ = generate_synthetic_batch(
            batch_size=config.batch_size, 
            q_grid=q_grid, 
            config=config, 
            augmenter=augmenter
        )

        optimizer.zero_grad()

        # Input: Log Normalized R
        params, r_reconstructed = pinn(torch.log10(r_measured + 1e-12))

        # Loss: Log MSE
        loss = criterion(torch.log10(r_reconstructed + 1e-12), torch.log10(r_measured + 1e-12))

        loss.backward()
        optimizer.step()

        if step % 100 == 0: # Fixed log interval or from args
            current_loss = loss.item()
            pbar.set_description(f"Loss: {current_loss:.6f}")
            
            checkpoint = {
                'step': step,
                'model_state_dict': pinn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                'config': config.model_dump()
            }
            torch.save(checkpoint, save_dir / "checkpoint_latest.pt")
            
            if current_loss < best_loss:
                best_loss = current_loss
                torch.save(checkpoint, save_dir / "checkpoint_best.pt")

if __name__ == "__main__":
    # Internal Test Run
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    
    train_streaming(CONFIG, steps=args.steps)
