from pathlib import Path
from dataclasses import fields

import torch
import numpy as np
import matplotlib.pyplot as plt

from xrefine.config import XRefineConfig
from xrefine.models import XRR_Fourier_PINN
from xrefine.physics_utils import tth2q
from xrefine.logger import logger

def load_real_data(file_path: str, config: XRefineConfig, mask_val: float = -15.0):
    """실전 데이터를 로드하고 모델 그리드에 맞춰 마스킹 수행"""
    data = np.loadtxt(file_path)
    x_raw, y_raw = data[:, 0], data[:, 1]
    
    # 2-theta to q 변환
    if x_raw.max() > 2.0:
        x_raw = tth2q(x_raw / 2.0, config.instrument.wavelength)

    q_min_meas, q_max_meas = x_raw.min(), x_raw.max()
    y_norm = y_raw / (y_raw.max() + 1e-12)
    log_y_raw = np.log10(np.clip(y_norm, 1e-11, 1.0))

    q_target = config.q_grid.numpy()
    log_y_interp = np.interp(q_target, x_raw, log_y_raw)

    # 범위 밖 마스킹
    out_of_range = (q_target < q_min_meas) | (q_target > q_max_meas)
    log_y_interp[out_of_range] = mask_val
    
    return torch.from_numpy(log_y_interp).float().view(1, -1)

def local_refinement(model: XRR_Fourier_PINN, p_init, log_r_obs, n_iters=60):
    logger.info(f"🛠️ Starting Local Refinement ({n_iters} iterations)...")
    
    # 1. 독립적인 복제본 생성 및 최적화 대상 수집
    p_refined = p_init.detach().clone()
    params_to_opt = []
    
    # fields를 돌며 각 텐서에 대해 requires_grad를 True로 명시적 설정
    for f in fields(p_refined):
        tensor = getattr(p_refined, f.name)
        tensor.requires_grad_(True)  # 핵심: 여기서 그래프의 시작점을 만듭니다.
        params_to_opt.append(tensor)
            
    optimizer = torch.optim.Adam(params_to_opt, lr=0.005)
    mask = (log_r_obs > -10.0).float()
    valid_points = mask.sum() + 1e-9
    
    # --- 핵심 수정 부분: enable_grad() 컨텍스트 추가 ---
    with torch.enable_grad(): 
        model.train() # 필요 시 (BatchNorm 등이 있다면)
        for i in range(n_iters):
            # 2. 물리 시뮬레이션 (p_refined를 사용하므로 그래프가 여기서부터 이어짐)
            _, r_recon, penalty = model.forward_with_params(p_refined)
            
            log_r_recon = torch.log10(torch.clamp(r_recon, min=1e-12))
            
            # 3. Loss 계산
            recon_loss = torch.sum(((log_r_recon - log_r_obs)**2) * mask) / valid_points
            total_loss = recon_loss + (0.1 * penalty)
            
            # 4. 역전파 및 최적화
            optimizer.zero_grad()
            total_loss.backward() # 이제 total_loss는 p_refined의 텐서들과 연결되어 있습니다.
            optimizer.step()
            
            if (i+1) % 20 == 0:
                logger.debug(f"Iter {i+1}/{n_iters} | Loss: {total_loss.item():.6f}")
    
    model.eval()
    return p_refined.detach() # 최적화 종료 후 다시 detach하여 반환

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
        p_final = local_refinement(model, p_ai, input_tensor, n_iters=60)
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