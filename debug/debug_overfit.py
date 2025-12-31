
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from xrefine.config import CONFIG
from xrefine.models.fourier_model import XRR_Fourier_PINN
from xrefine.data.generator import generate_synthetic_batch
from xrefine.data.augmentations import XRRAugmentations

def debug_overfit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    config = CONFIG
    config.n_layers = 1
    config.batch_size = 1
    config.lr = 1e-3
    
    # 1. Generate ONE sample
    q_grid = config.q_grid.to(device)
    aug_cfg = config.augmentations
    augmenter = XRRAugmentations(
        delta_q=0.0005,
        intensity_noise_scale=aug_cfg.intensity_noise_scale, # Use real noise
        do_footprint=True
    )
    
    # Fixed Seed
    torch.manual_seed(42)
    
    # Generate ONE Single Target
    # We want to overfit this exact noisy R and these exact params
    r_target, params_target = generate_synthetic_batch(1, q_grid, config, augmenter=augmenter)
    
    t_true, r_true, s_true = params_target
    print(f"Target Params: T={t_true.item():.1f}, R={r_true[0,0].item():.1f}, SLD={s_true[0,1].item():.1f}")
    
    # 2. Setup Model
    model = XRR_Fourier_PINN(config=config, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    losses = []
    
    print("🚀 Starting Overfit Test (1000 steps)...")
    model.train()
    
    input_log_r = torch.log10(r_target + 1e-12)
    
    for step in range(1000):
        optimizer.zero_grad()
        
        pred_params, r_recon = model(input_log_r)
        
        # Loss on Reconstruction (Autoencoder)
        loss_r = criterion(torch.log10(r_recon + 1e-12), input_log_r)
        
        # Supervised Loss on Params? 
        # In real training we don't have supervision? 
        # Wait, PINN usually implies Unsupervised (Physics Loss)?
        # IN XRefine normal training `train.py`:
        # loss = criterion(torch.log10(r_reconstructed), torch.log10(r_measured))
        # It IS Unsupervised (Reconstruction Loss).
        
        loss = loss_r
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}: Loss {loss.item():.6f}")
            losses.append(loss.item())
            
    # 4. Evaluation
    model.eval()
    with torch.no_grad():
        pred_params, r_recon = model(input_log_r)
        
        t_pred, r_pred, s_pred = pred_params
        
        print("\nResults:")
        print(f"True Params: T={t_true.item():.1f}, R={r_true[0,0].item():.1f}, SLD={s_true[0,1].item():.1f}")
        print(f"Pred Params: T={t_pred.item():.1f}, R={r_pred[0,0].item():.1f}, SLD={s_pred[0,1].item():.1f}")
        
    # Plot
    plt.figure(figsize=(10, 5))
    plt.semilogy(q_grid.cpu(), r_target.cpu().flatten(), 'k.', alpha=0.5, label='Target (Noisy)')
    plt.semilogy(q_grid.cpu(), r_recon.cpu().flatten(), 'r-', label='Recon (Model)')
    plt.legend()
    plt.title(f"Overfit Test: Loss {losses[-1]:.4f}")
    plt.savefig('debug_overfit.png')
    print("Saved debug_overfit.png")

if __name__ == "__main__":
    debug_overfit()
