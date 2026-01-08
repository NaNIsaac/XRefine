"""
XRefine Benchmark
=================
합성 데이터 기반 Refinement 정확도 벤치마크

Usage:
    uv run python -m xrefine.refine.benchmark
    
    또는 파이썬 코드에서:
    from xrefine.refine.benchmark import run_benchmark
    run_benchmark(checkpoint_path, target_params={"d_f": 400}, n_random=5)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict

from xrefine.config import CONFIG, XRefineConfig
from xrefine.models import XRR_Fourier_PINN
from xrefine.core.structures import ParamSet
from xrefine.data.generator import generate_synthetic_batch
from xrefine.logger import logger

from xrefine.refine.config import RefinementConfig
from xrefine.refine.pipeline import run_refinement


def run_benchmark(
    checkpoint_path: str,
    target_params: Optional[Dict[str, float]] = None,
    n_random: int = 0,
    refine_config: RefinementConfig = None,
    save_dir: str = "./benchmark_results"
):
    """
    합성 데이터 기반 Refinement 벤치마크
    
    Args:
        checkpoint_path: 모델 체크포인트 경로
        target_params: 사용자 지정 파라미터 (선택)
        n_random: 랜덤 테스트 케이스 수
        refine_config: Refinement 설정 (None = 기본값)
        save_dir: 결과 저장 디렉토리
    """
    # 1. Load Model & Config
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    finally:
        pathlib.PosixPath = temp
    
    # Config Override (최신 ranges 적용)
    config = XRefineConfig(**ckpt['config'])
    config.ranges = CONFIG.ranges
    
    model = XRR_Fourier_PINN(config=config).to(config.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    q_grid = config.q_grid.to(config.device)
    
    # 2. Define Test Cases
    test_cases = []
    
    # (A) User-specified Case
    if target_params:
        _, p_rand = generate_synthetic_batch(1, q_grid, config)
        
        for k, v in target_params.items():
            if hasattr(p_rand, k):
                getattr(p_rand, k).fill_(v)
        
        # Generate R for these params
        _, r_sim, _ = model.forward_with_params(p_rand)
        r_noisy = r_sim * (1 + 0.1 * torch.randn_like(r_sim))
        r_noisy_norm = r_noisy / (r_noisy.max() + 1e-12)
        log_r = torch.log10(torch.clamp(r_noisy_norm, min=1e-12))
        
        test_cases.append(("User_Spec", p_rand, log_r))
    
    # (B) Random Cases
    if n_random > 0:
        r_batch, p_batch = generate_synthetic_batch(n_random, q_grid, config)
        r_batch_norm = r_batch / (r_batch.max(dim=1, keepdim=True)[0] + 1e-12)
        log_r_batch = torch.log10(torch.clamp(r_batch_norm, min=1e-12))
        
        for i in range(n_random):
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
    logger.info(f"🚀 Starting Benchmark ({len(test_cases)} cases)...")
    
    all_errors = {}
    
    for name, p_true, log_r_obs in test_cases:
        logger.info(f"\n{'='*50}")
        logger.info(f"📦 Benchmark Case: {name}")
        logger.info(f"{'='*50}")
        
        # AI Initial Guess
        with torch.no_grad():
            p_ai, _, _ = model(log_r_obs)
        
        # Run Refinement (using new module)
        p_refined = run_refinement(model, p_ai, log_r_obs, config, refine_config)
        
        # Compare
        errors = {}
        key_params = ['d_f', 'sig_f', 'rho_f', 'd_ox', 'sig_ox', 'rho_ox']
        
        for k in key_params:
            t = getattr(p_true, k).item()
            p = getattr(p_refined, k).item()
            err_percent = (p - t) / (abs(t) + 1e-9) * 100
            errors[k] = (t, p, err_percent)
        
        # Print Results (key params only)
        logger.info(f"\n📊 Result:")
        logger.info(f"{'Param':<10} {'True':>10} {'Pred':>10} {'Error':>10}")
        logger.info("-" * 42)
        for k, (t, p, e) in errors.items():
            err_str = f"{e:+.1f}%" if abs(e) < 100 else f"{'✗':>10}"
            logger.info(f"{k:<10} {t:>10.1f} {p:>10.1f} {err_str:>10}")
        
        all_errors[name] = errors
        
        # Plot
        _, r_true, _ = model.forward_with_params(p_true)
        _, r_pred, _ = model.forward_with_params(p_refined)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        q = config.q_grid.cpu().numpy()
        obs_np = log_r_obs.detach().cpu().numpy().flatten()
        true_np = r_true.detach().cpu().numpy().flatten()
        pred_np = r_pred.detach().cpu().numpy().flatten()
        
        ax.semilogy(q, 10**obs_np, 'ko', ms=2, alpha=0.3, label='Observed (Noisy)')
        ax.semilogy(q, true_np, 'g--', lw=1.5, label='Ground Truth')
        ax.semilogy(q, pred_np, 'r-', lw=1.5, label='Refined Fit')
        ax.set_xlabel("Q (Å⁻¹)")
        ax.set_ylabel("Reflectivity")
        ax.set_title(f"Benchmark: {name}")
        ax.legend()
        ax.grid(True, alpha=0.2)
        
        # Add param text
        param_text = (
            f"True:  d={p_true.d_f.item():.0f}Å  σ={p_true.sig_f.item():.1f}Å\n"
            f"Pred:  d={p_refined.d_f.item():.0f}Å  σ={p_refined.sig_f.item():.1f}Å"
        )
        ax.text(0.02, 0.02, param_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.savefig(save_path / f"{name}.png", dpi=150)
        plt.close()
    
    logger.info(f"\n✅ Benchmark Complete. Results saved to: {save_path}")
    return all_errors


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="XRefine Benchmark Tool")
    parser.add_argument("-c", "--checkpoint",
                        default=r"D:\03_Resources\Data\XRR_AI\xrefine\v1\best.pt",
                        help="Model checkpoint path")
    parser.add_argument("-n", "--n_random", type=int, default=3,
                        help="Number of random test cases")
    parser.add_argument("-o", "--output", default="./benchmark_results",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Example manual case
    manual_case = {
        'd_f': 354,
        'd_ox': 18.3,
        'sig_f': 5.0,
        'sig_ox': 4.4,
        'rho_f': 49.6,
        'rho_ox': 14.1,
    }
    
    run_benchmark(
        args.checkpoint, 
        target_params=manual_case, 
        n_random=args.n_random,
        save_dir=args.output
    )
