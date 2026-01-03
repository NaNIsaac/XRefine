from pathlib import Path
from dataclasses import fields

import torch
import numpy as np
import matplotlib.pyplot as plt

from xrefine.config import XRefineConfig
from xrefine.models import XRR_Fourier_PINN
from xrefine.physics_utils import tth2q
from xrefine.logger import logger
from xrefine.core.structures import ParamSet

def load_real_data(file_path: str, config: XRefineConfig, mask_val: float = -15.0):
    """실전 데이터를 로드하고 모델 그리드에 맞춰 마스킹 수행"""
    data = np.loadtxt(file_path)
    x_raw, y_raw = data[:, 0], data[:, 1]
    
    x_raw = tth2q(x_raw, config.instrument.wavelength)

    q_min_meas, q_max_meas = x_raw.min(), x_raw.max()
    y_norm = y_raw / (y_raw.max() + 1e-12)
    log_y_raw = np.log10(np.clip(y_norm, 1e-11, 1.0))

    q_target = config.q_grid.numpy()
    log_y_interp = np.interp(q_target, x_raw, log_y_raw)

    # 범위 밖 마스킹
    out_of_range = (q_target < q_min_meas) | (q_target > q_max_meas)
    log_y_interp[out_of_range] = mask_val
    
    return torch.from_numpy(log_y_interp).float().view(1, -1)

def local_refinement(model, p_init, log_r_obs, n_iters=100):
    """
    Early Stopping 및 발산 방지 로직이 포함된 Refinement.
    """
    logger.info(f"🛠️ Starting Refinement (Max {n_iters} iters)...")
    
    p_refined = p_init.detach().clone()
    params_to_opt = [getattr(p_refined, f.name).requires_grad_(True) for f in fields(p_refined)]
    
    # 보폭을 줄여 안정성 확보 (0.005 -> 0.002)
    optimizer = torch.optim.Adam(params_to_opt, lr=0.002)
    # Loss가 정체되면 LR을 더 줄임 (감속기)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    mask = (log_r_obs > -10.0).float()
    valid_points = mask.sum() + 1e-9
    
    best_loss = float('inf')
    best_params_state = None
    
    with torch.enable_grad():
        for i in range(n_iters):
            _, r_recon, penalty = model.forward_with_params(p_refined)
            log_r_recon = torch.log10(torch.clamp(r_recon, min=1e-12))
            
            recon_loss = torch.sum(((log_r_recon - log_r_obs)**2) * mask) / valid_points
            total_loss = recon_loss + (1.0 * penalty) # 물리 제약 강화
            
            curr_loss = total_loss.item()
            
            # --- Early Stopping & Snapshot 로직 ---
            if curr_loss < best_loss:
                best_loss = curr_loss
                # 최적의 상태 저장 (복제본)
                best_params_state = {f.name: getattr(p_refined, f.name).detach().clone() for f in fields(p_refined)}
                patience_counter = 0
            else:
                patience_counter += 1
                
            # 갑작스러운 발산 방지 (이전 최소값의 3배 이상 튀면 중단)
            if curr_loss > best_loss * 3.0:
                logger.warning(f"🚨 Divergence at {i}! Restoring best state.")
                break
                
            # 20회 이상 개선 없으면 조기 종료
            if patience_counter > 20:
                logger.info(f"✅ Converged at iter {i}.")
                break
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)
            
            if (i+1) % 20 == 0:
                logger.debug(f"Iter {i+1} | Loss: {curr_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
                
    # 최적의 상태로 복원하여 반환
    return ParamSet(**best_params_state)

@torch.no_grad()
def run_real_inference(checkpoint_path: str, data_path: str, refine: bool = True):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = XRefineConfig(**ckpt['config']) if isinstance(ckpt['config'], dict) else ckpt['config']
        
    model = XRR_Fourier_PINN(config=config)
    model.load_state_dict(ckpt['model_state_dict'], strict=False) # 구조 변경 대응
    model.eval()

    # 1. 데이터 준비 및 AI 1차 추론
    input_tensor = load_real_data(data_path, config)
    p_ai, r_ai, _ = model(input_tensor)

    # 2. Local Refinement (선택 사항)
    if refine:
        p_final = local_refinement(model, p_ai, input_tensor, n_iters=600)
        _, r_final, _ = model.forward_with_params(p_final)
    else:
        p_final, r_final = p_ai, r_ai

    # 3. 결과 리포트
    print("\n" + "="*60)
    print("[XRefine AI + Physics Refined Report]")
    print(f"File: {Path(data_path).name}")
    print("-" * 60)
    print(p_final)
    print("="*60 + "\n")

    # 4. 시각화
    q_plot = config.q_grid.numpy()
    y_meas_log = input_tensor[0].numpy()
    valid_mask = y_meas_log > -14.0 

    plt.figure(figsize=(12, 7))
    plt.semilogy(q_plot[valid_mask], 10**y_meas_log[valid_mask], 'ko', ms=3, alpha=0.2, label='Measured Data')
    plt.semilogy(q_plot, r_ai[0].numpy(), 'b--', alpha=0.5, label='AI Initial Guess')
    plt.semilogy(q_plot, r_final[0].detach().numpy(), 'r-', lw=2, label='Refined Final Fit')
    
    plt.xlabel(r"q ($\AA^{-1}$)", fontsize=12) # r 문법 수정
    plt.ylabel("Reflectivity", fontsize=12)
    plt.title(f"Refinement Result: {Path(data_path).name}")
    plt.legend()
    plt.grid(True, which='both', alpha=0.1)
    plt.show()

if __name__ == "__main__":
    CHECKPOINT = r"C:\Warehouse\data\XRR_AI\experiments\sandbox_v7\best.pt"
    DATA_FILE = r"C:\Warehouse\data\dat_files\jinhuan\#2_xrr.dat"
    run_real_inference(CHECKPOINT, DATA_FILE, refine=True)