import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution
from typing import Dict, List, Any

from xrefine.config import CONFIG, XRefineConfig
from xrefine.models import XRR_Fourier_PINN
from xrefine.physics_utils import tth2q
from xrefine.logger import logger
from xrefine.core.structures import ParamSet

def load_real_data(file_path: str, config: XRefineConfig):
    """실제 데이터를 로드하고 노이즈에 강인한 정규화를 수행합니다."""
    df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python', dtype=str)
    for col in df.columns:
        df[col] = df[col].str.replace('(', '', regex=False).str.replace(')', '', regex=False).astype(float)

    x_raw, y_raw = df.values[:, 0], df.values[:, 1]
    x_raw = tth2q(x_raw, config.instrument.wavelength)

    # 상위 1% 값을 기준으로 정규화하여 Outlier 영향 감소
    norm_factor = np.quantile(y_raw, 0.99)
    y_norm = y_raw / (norm_factor + 1e-12)
    log_y_raw = np.log10(np.clip(y_norm, 1e-11, 1.0))

    q_target = config.q_grid.numpy()
    log_y_interp = np.interp(q_target, x_raw, log_y_raw)
    log_y_interp[(q_target < x_raw.min()) | (q_target > x_raw.max())] = -15.0
    
    return torch.from_numpy(log_y_interp).float().view(1, -1)

def genx_style_refinement(model, p_ai: ParamSet, log_r_obs: torch.Tensor, config: XRefineConfig):
    logger.info("🧬 Starting GenX-style Evolutionary Refinement...")
    
    # 1. 모델의 실제 디바이스를 명확히 고정
    target_device = next(model.parameters()).device
    range_config = config.ranges
    
    # [Debug] 타겟 데이터를 미리 GPU로 옮겨서 연산 충돌 방지
    log_r_obs_gpu = log_r_obs.to(target_device)
    mask = (log_r_obs_gpu > -14.0)

    # 파라미터 구조 분석 (ParamSet의 iterator 활용)
    param_info = []
    ai_guess_norm = []
    for k, v in p_ai.items():
        p_min, p_max = getattr(range_config, k)
        val_flat = v.detach().cpu().numpy().flatten()
        param_info.append({'key': k, 'size': val_flat.size, 'min': p_min, 'max': p_max})
        norm_val = (val_flat - p_min) / (p_max - p_min + 1e-9)
        ai_guess_norm.extend(np.clip(norm_val, 0.0, 1.0).tolist())

    total_dim = len(ai_guess_norm)

    def objective(trial_vector):
        """Scipy(CPU) -> Torch(GPU) 변환 및 물리 Loss 계산"""
        try:
            p_dict = {}
            cursor = 0
            for info in param_info:
                k, size, p_min, p_max = info['key'], info['size'], info['min'], info['max']
                # 물리 스케일 복원
                phys_val = trial_vector[cursor : cursor + size] * (p_max - p_min) + p_min
                
                # [CRITICAL] 텐서 생성 시점에 즉시 GPU(target_device) 지정 및 2D 형상 확보
                p_dict[k] = torch.tensor(phys_val, dtype=torch.float32, device=target_device).view(1, size)
                cursor += size
            
            # 이미 모든 텐서가 GPU에 있는 ParamSet 조립
            p_set = ParamSet(**p_dict)
            
            with torch.no_grad():
                # forward_with_params 내부의 모든 연산이 동일 디바이스에서 수행됨
                _, r_sim, penalty = model.forward_with_params(p_set)
                
                log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
                
                # 가중치 MSE Loss + Physics Penalty
                loss = torch.mean(((log_r_sim - log_r_obs_gpu)**2) * mask)
                total_val = loss + (penalty * config.train.phys_penalty_w)
                
                # [FIX] Scipy가 요구하는 순수 Python float로 반환
                return float(total_val.item())
        
        except Exception as e:
            # Scipy가 에러를 덮기 전에 실제 원인을 로깅
            logger.error(f"❌ Error in objective: {e}")
            raise e

    # 초기 인구 설정 및 실행
    popsize = 15
    init_pop = np.random.rand(popsize, total_dim)
    init_pop[0] = np.array(ai_guess_norm)

    result = differential_evolution(
        objective,
        bounds=[(0.0, 1.0)] * total_dim,
        strategy='best1bin',
        maxiter=100,
        popsize=popsize,
        init=init_pop,
        disp=True
    )

    # 4. 결과 복원 (2D Tensor 형태 유지)
    final_dict = {}
    cursor = 0
    for info in param_info:
        phys_val = result.x[cursor : cursor + info['size']] * (info['max'] - info['min']) + info['min']
        final_dict[info['key']] = torch.tensor(phys_val, dtype=torch.float32).view(1, info['size'])
        cursor += info['size']
        
    return ParamSet(**final_dict)

@torch.no_grad()
def run_real_inference(checkpoint_path: str, data_path: str):
    # 환경 설정
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = XRefineConfig(**ckpt['config'])
    
    model = XRR_Fourier_PINN(config=config).to(config.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 1. AI 초기 추측
    input_tensor = load_real_data(data_path, config)
    p_ai, r_ai, _ = model(input_tensor.to(config.device))

    # 2. 전역 최적화 실행
    p_refined = genx_style_refinement(model, p_ai, input_tensor, config)
    
    # 3. 최종 결과 생성
    p_final, r_final, _ = model.forward_with_params(p_refined.to(config.device))

    # 4. 시각화 및 로깅
    _plot_results(config, input_tensor[0], r_ai[0], r_final[0], Path(data_path).name)
    logger.info(f"✨ Refinement Complete.\nAI:\n{p_ai}\nRefined:\n{p_final}")

def _plot_results(config, y_meas_log, r_ai, r_ref, title):
    q = config.q_grid.numpy()
    plt.figure(figsize=(12, 7))
    plt.semilogy(q, 10**y_meas_log.cpu().numpy(), 'ko', ms=3, alpha=0.2, label='Measured Data')
    plt.semilogy(q, r_ai.cpu().numpy(), 'b--', alpha=0.5, label='AI Initial Guess')
    plt.semilogy(q, r_ref.cpu().numpy(), 'r-', lw=2, label='GenX-style Refined')
    plt.title(f"XRefine Result: {title}")
    plt.legend()
    plt.grid(True, which='both', alpha=0.1)
    plt.show()

if __name__ == "__main__":
    CHECKPOINT = r"D:\03_Resources\Data\XRR_AI\PINN\v3\best.pt"
    DATA_FILE = r"D:\03_Resources\Data\XRR_AI\XRR_data\XRR_data2\#4.dat"
    run_real_inference(CHECKPOINT, DATA_FILE)