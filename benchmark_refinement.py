import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from typing import Optional, Dict

from xrefine.config import CONFIG, XRefineConfig
from xrefine.models import XRR_Fourier_PINN
from xrefine.core.structures import ParamSet
from xrefine.predict_real import three_stage_refinement
from xrefine.logger import logger

from xrefine.data.generator import generate_synthetic_batch

def run_benchmark(
    checkpoint_path: str,
    target_params: Optional[Dict[str, float]] = None,
    n_random: int = 0,
    save_dir: str = "./benchmark_results"
):
    """
    벤치마크 실행 함수
    """
    # 1. Load Model & Config
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
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    q_grid = config.q_grid.to(config.device)
    
    # 2. Define Test Cases & Generate Data
    test_cases = []
    
    # (A) User Specified Case
    if target_params:
        # Generate 1 batch of random params
        _, p_rand = generate_synthetic_batch(1, q_grid, config)
        
        # Override with user values
        for k, v in target_params.items():
            if hasattr(p_rand, k):
                getattr(p_rand, k).fill_(v)
                
        # Generate corresponding data
        # Note: We use generate_synthetic_batch internally to get physics-consistent R
        # But we need to force these params.
        # Let's manually generate R for this specific p_rand using the same logic as generator
        model_sim = XRR_Fourier_PINN(config).to(config.device)
        _, r_sim, _ = model_sim.forward_with_params(p_rand)
        r_noisy = r_sim * (1 + 0.1 * torch.randn_like(r_sim)) # 10% Noise
        
        # [Normalization] Match load_real_data behavior (Peak=1.0)
        # This brings the data to a scale where i0 ~ 1.0 is expected
        r_noisy_norm = r_noisy / (r_noisy.max() + 1e-12)
        log_r = torch.log10(torch.clamp(r_noisy_norm, min=1e-12))
        
        test_cases.append(("User_Spec", p_rand, log_r))
        
    # (B) Random Cases
    if n_random > 0:
        r_batch, p_batch = generate_synthetic_batch(n_random, q_grid, config)
        
        # Normalize Batch (per sample)
        r_batch_norm = r_batch / (r_batch.max(dim=1, keepdim=True)[0] + 1e-12)
        log_r_batch = torch.log10(torch.clamp(r_batch_norm, min=1e-12))
        
        for i in range(n_random):
            # Extract single item
            p_single = {}
            for k, v in p_batch.items():
                val = getattr(p_batch, k)
                p_single[k] = val[i].view(1, -1)
            
            p_obj = ParamSet(**p_single)
            r_obj = log_r_batch[i].view(1, -1)
            test_cases.append((f"Random_{i+1}", p_obj, r_obj))
            
    if not test_cases:
        logger.warning("No test cases defined.")
        return

    # 3. Run Benchmark
    logger.info(f"🚀 Starting Benchmark on {len(test_cases)} cases...")
    
    for name, p_true, log_r_obs in test_cases:
        logger.info(f"\n--- Benchmark Case: {name} ---")
        
        # 3.2 AI Initial Guess
        with torch.no_grad():
            p_ai, _, _ = model(log_r_obs)
            
        # 3.3 Run Refinement
        p_refined = three_stage_refinement(model, p_ai, log_r_obs, config)
        
        # 3.4 Compare
        errors = {}
        for k in p_true.keys():
            t = getattr(p_true, k).item()
            p = getattr(p_refined, k).item()
            err_percent = abs(p - t) / (t + 1e-9) * 100
            errors[k] = (t, p, err_percent)
            
        logger.info(f"📊 Result for {name}:")
        logger.info(f"{'Param':<10} {'True':<10} {'Pred':<10} {'Error(%)':<10}")
        logger.info("-" * 40)
        for k, (t, p, e) in errors.items():
            logger.info(f"{k:<10} {t:<10.2f} {p:<10.2f} {e:<10.1f}%")
            
        # Plot
        _, r_true, _ = model.forward_with_params(p_true)
        _, r_pred, _ = model.forward_with_params(p_refined)
        
        plt.figure(figsize=(10, 6))
        q = config.q_grid.cpu().numpy()
        obs_np = log_r_obs.detach().cpu().numpy().flatten()
        true_np = r_true.detach().cpu().numpy().flatten()
        pred_np = r_pred.detach().cpu().numpy().flatten()
        
        plt.semilogy(q, 10**obs_np, 'ko', alpha=0.3, label='Observed (Noisy)')
        plt.semilogy(q, true_np, 'g--', label='Ground Truth')
        plt.semilogy(q, pred_np, 'r-', label='Refined Fit')
        plt.title(f"Benchmark: {name}")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(save_path / f"{name}.png")
        plt.close()
        
if __name__ == "__main__":
    CHECKPOINT = r"D:\03_Resources\Data\XRR_AI\xrefine\v1\best.pt"
    
    # Example: User defines a "Thin Film" case manually to test if optimizer fails
    manual_case = {
        'd_f': 400, # Target: ~95 A
        'd_ox': 20.0,
        'sig_f': 3.0,
        'sig_ox': 4.0,
        'bkg': -6.5
    }
    
    run_benchmark(CHECKPOINT, target_params=manual_case, n_random=1)
