import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

from xrefine.config import XRefineConfig
from xrefine.models import XRR_Fourier_PINN
from xrefine.core.structures import ParamSet
from xrefine.predict_real import load_real_data, _optimize_subset
from xrefine.logger import logger

def scan_thickness_landscape(
    checkpoint_path: str, 
    data_path: str,
    q_max: float = 0.45
):
    # Windows Compat
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    finally:
        pathlib.PosixPath = temp

    config = XRefineConfig(**ckpt['config'])
    model = XRR_Fourier_PINN(config=config).to(config.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Load Data
    log_r_obs = load_real_data(data_path, config).to(config.device)
    
    # 1. AI Initial Guess to set base params
    with torch.no_grad():
        p_ai, _, _ = model(log_r_obs)
        
    # 2. Optimize Background/Footprint first (Stage 0 logic) to get clean base
    logger.info("🔧 Pre-optimizing Background & Footprint...")
    p_base = _optimize_subset(
        model, p_ai, log_r_obs, config,
        optimize_keys={'beam_width', 'L', 'i0', 'bkg'},
        q_max=0.15, maxiter=30, popsize=10
    )
    
    # 3. Thickness Scan
    logger.info("📉 Scanning Thickness (d_f) Landscape...")
    
    d_range = np.linspace(10, 1200, 500) # 10A ~ 1200A
    loss_history = []
    
    # Masking
    mask = (log_r_obs > -14.0) & (config.q_grid.to(config.device) < q_max) & (config.q_grid.to(config.device) > 0.03)
    
    # Disable Oxide for clear scan
    p_scan = deepcopy(p_base)
    p_scan.d_ox.fill_(0.0)
    p_scan.sig_ox.fill_(0.0)
    p_scan.rho_ox.fill_(0.0)
    p_scan.sig_f.fill_(3.0) # Fixed roughness
    
    for d_val in d_range:
        p_scan.d_f.fill_(d_val)
        
        with torch.no_grad():
            _, r_sim, _ = model.forward_with_params(p_scan)
            log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
            
            # Hybrid Loss (MSE + Gradient)
            mse = torch.mean(((log_r_sim - log_r_obs)**2) * mask)
            
            diff_sim = torch.diff(log_r_sim, dim=-1)
            diff_obs = torch.diff(log_r_obs, dim=-1)
            grad_err = torch.mean(((diff_sim - diff_obs)**2) * mask[:, 1:])
            
            total_loss = mse + 5.0 * grad_err
            loss_history.append(total_loss.item())
            
    # Find Local Minima (Valleys)
    from scipy.signal import find_peaks
    # Invert loss for peak finding
    peaks, _ = find_peaks(-np.array(loss_history), prominence=0.01)
    
    # Sort by loss value
    proposals = []
    for idx in peaks:
        proposals.append((d_range[idx], loss_history[idx]))
    
    # Add global min if not in peaks (edge case)
    min_idx = np.argmin(loss_history)
    global_min_d = d_range[min_idx]
    if not any(abs(p[0] - global_min_d) < 5.0 for p in proposals):
        proposals.append((global_min_d, loss_history[min_idx]))
        
    # Sort proposals by Loss (ascending)
    proposals.sort(key=lambda x: x[1])
    
    logger.info(f"✅ Top 3 Candidates: {[f'{d:.1f}A (L={l:.4f})' for d, l in proposals[:3]]}")
    
    # Plot Landscape
    plt.figure(figsize=(10, 6))
    plt.plot(d_range, loss_history, 'b-', label='Loss')
    for d_val, l_val in proposals[:3]:
        plt.plot(d_val, l_val, 'ro')
        plt.text(d_val, l_val, f"{d_val:.1f}A", fontsize=9, verticalalignment='bottom')
        
    plt.title(f"Thickness Scan: {Path(data_path).name}")
    plt.xlabel("Thickness d_f (A)")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"debug_scan_landscape_{Path(data_path).stem}.png")
    plt.close()
    
    # Plot Fits for Top Candidates
    q = config.q_grid.cpu().numpy()
    obs = log_r_obs.cpu().numpy().flatten()
    
    plt.figure(figsize=(12, 6))
    plt.semilogy(q, 10**obs, 'k.', alpha=0.2, label='Observed')
    
    colors = ['r', 'g', 'b']
    for i, (d_val, l_val) in enumerate(proposals[:3]):
        p_scan.d_f.fill_(d_val)
        with torch.no_grad():
            _, r_sim, _ = model.forward_with_params(p_scan)
        
        sim = r_sim.cpu().numpy().flatten()
        plt.semilogy(q, sim, '-', color=colors[i], linewidth=2, label=f'd={d_val:.1f}A (L={l_val:.3f})')
        
    plt.title("Fit Comparison for Candidates")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"debug_scan_fits_{Path(data_path).stem}.png")
    logger.info(f"📈 Saved fit comparison to debug_scan_fits_{Path(data_path).stem}.png")

if __name__ == "__main__":
    CHECKPOINT = r"D:\03_Resources\Data\XRR_AI\xrefine\v1\best.pt"
    # DATA_FILE = r"D:\03_Resources\Data\XRR_AI\XRR_data\KBJ\ALD_A_no_XRR.dat"
    DATA_FILE = r"D:\03_Resources\Data\XRR_AI\XRR_data\XRR_data\#8_xrr.dat"
    
    scan_thickness_landscape(CHECKPOINT, DATA_FILE)
