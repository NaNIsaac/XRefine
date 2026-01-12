
import torch
import matplotlib.pyplot as plt
import numpy as np
from xrefine.config import CONFIG, XRefineConfig
from xrefine.data.augmentations import XRRAugmentor
from xrefine.core.structures import ParamSet

def test_variable_density_augmentation():
    # 1. Setup Config & Augmentor
    config = XRefineConfig()
    config.augmentations.measure_step_range = (0.01, 0.1) # Test range: 0.01 deg to 0.1 deg
    config.augmentations.intensity_noise_scale = 0.0 # Disable noise for clear view
    config.augmentations.q_shift_sigma = 0.0
    
    augmentor = XRRAugmentor(config).to(config.device)
    
    # 2. Create Dummy Data (Sine wave to see interpolation easily)
    q = config.q_grid.to(config.device)
    # High frequency oscillation
    R_orig = (torch.sin(q * 100) * 0.5 + 0.5).unsqueeze(0) 
    
    # 3. Apply Resampling directly
    # Force a specific target_len manually if possible, or just run forward multiple times
    # Here we rely on the random selection in augmentation
    
    plt.figure(figsize=(12, 5))
    plt.plot(q.cpu().numpy(), R_orig[0].cpu().numpy(), 'k-', lw=1, alpha=0.5, label='Original (High Res)')

    colors = ['r', 'g', 'b']
    for i in range(3):
        # We need to hack the augmentor or just run it multiple times hoping for different densities
        # Since logic is random, we just run forward.
        # Note: forward adds other effects too, but we disabled noise/shift.
        # But forward calls apply_resampling.
        
        R_aug = augmentor.apply_resampling(R_orig.clone())
        
        plt.plot(q.cpu().numpy(), R_aug[0].cpu().numpy(), 
                 ls='--', color=colors[i], lw=1.5, alpha=0.8, 
                 label=f'Augmented Run {i+1}')

    plt.title("Variable Density Augmentation Verification")
    plt.xlabel("Q")
    plt.ylabel("Reflectivity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("debug_augmentation_test.png")
    print("Test image saved to debug_augmentation_test.png")

if __name__ == "__main__":
    test_variable_density_augmentation()
