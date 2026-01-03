from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from xrefine.config import XRefineConfig
from xrefine.models import XRR_Fourier_PINN
from xrefine.physics_utils import tth2q
from xrefine.logger import logger

def load_real_data(file_path: str | Path, config: XRefineConfig, mask_val: float = -15.0):
    """
    실전 데이터를 로드하고 모델 q_grid에 맞춰 보간 및 마스킹을 수행합니다.
    """
    # 1. 데이터 로드 (Delimiter는 데이터 형식에 따라 np.loadtxt 인자 조정 가능)
    data = np.loadtxt(file_path)
    x_raw, y_raw = data[:, 0], data[:, 1]

    x_raw = tth2q(x_raw, config.instrument.wavelength)

    # 유효 데이터의 q 범위 기록 (마스킹용)
    q_min_meas, q_max_meas = x_raw.min(), x_raw.max()

    # 2. Intensity 정규화 및 로그 변환
    y_norm = y_raw / (y_raw.max() + 1e-12)
    log_y_raw = np.log10(np.clip(y_norm, 1e-11, 1.0))

    # 3. 모델 q_grid(Fixed 2000pt)에 보간
    q_target = config.q_grid.numpy()
    log_y_interp = np.interp(q_target, x_raw, log_y_raw)

    # 4. 범위 밖 데이터 마스킹 (중요: 선형 외삽으로 인한 가짜 평면 제거)
    # 측정 범위보다 작거나 큰 q 영역을 mask_val로 덮어씌움
    out_of_range = (q_target < q_min_meas) | (q_target > q_max_meas)
    log_y_interp[out_of_range] = mask_val
    
    logger.info(f"📊 Data Loaded: {Path(file_path).name} | Q-range: [{q_min_meas:.3f}, {q_max_meas:.3f}]")
    logger.info(f"📌 Masked {out_of_range.sum()} points out of {len(q_target)}")
    
    return torch.from_numpy(log_y_interp).float().view(1, -1)

@torch.no_grad()
def run_real_inference(checkpoint_path: str, data_path: str):
    # 1. 체크포인트에서 설정 및 모델 복원
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Config 객체 복원
    if isinstance(ckpt['config'], dict):
        config = XRefineConfig(**ckpt['config'])
    else:
        config = ckpt['config']

        
    model = XRR_Fourier_PINN(config=config)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    # 2. 전처리된 텐서 준비 (Masked)
    input_tensor = load_real_data(data_path, config) # (1, 2000)
    
    # 3. AI 추론
    # r_recon은 이제 abeles 내부의 footprint 로직이 반영되어 나옵니다.
    params, r_recon, _ = model(input_tensor)

    # 4. 결과 출력
    print("\n" + "="*50)
    print("[XRefine AI Analysis Report]")
    print(f"File: {Path(data_path).name}")
    print("-" * 50)
    print(params) # ParamSet의 __repr__ 활용
    print("="*50 + "\n")

    # 5. 시각화 (마스크된 데이터는 시각화에서 제외)
    q_plot = config.q_grid.numpy()
    y_meas_log = input_tensor[0].numpy()
    y_recon = r_recon[0].numpy()
    
    # 마스크(-15.0)가 아닌 데이터만 필터링하여 그리기
    valid_mask = y_meas_log > -14.0 

    plt.figure(figsize=(12, 6))
    plt.semilogy(q_plot[valid_mask], 10**y_meas_log[valid_mask], 'ko', ms=2.5, alpha=0.3, label='Measured (Valid)')
    plt.semilogy(q_plot, y_recon, 'r-', lw=2, label='XRefine PINN Fit')
    
    # 임계각(Qc) 부근 하프컷 영역 가이드라인 (선택 사항)
    plt.axvline(q_plot[valid_mask].min(), color='blue', ls='--', alpha=0.3, label='Measurement Start')
    
    plt.xlabel("q ($\\AA^{-1}$)", fontsize=12)
    plt.ylabel("Reflectivity (Log Scale)", fontsize=12)
    plt.title(f"Experimental Data Fitting: {Path(data_path).name}", fontsize=14)
    plt.legend()
    plt.grid(True, which='both', alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 실제 환경에 맞는 경로로 수정 필요
    CHECKPOINT = r"C:\Warehouse\data\XRR_AI\experiments\sandbox_v8\best.pt"
    DATA_FILE = r"C:\Warehouse\data\dat_files\jinhuan\#2_xrr.dat"
    
    run_real_inference(CHECKPOINT, DATA_FILE)