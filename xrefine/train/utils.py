from typing import Dict, List, Optional
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from xrefine.core.structures import ParamSet
from xrefine.config import XRefineConfig

plt.switch_backend('Agg')

class XRRWeightedLoss(nn.Module):
    def __init__(self, q_grid: torch.Tensor):
        super().__init__()
        weights = torch.full_like(q_grid, 1.0)
        self.register_buffer('q_weights', weights.unsqueeze(0))
        self.register_buffer('q_grid', q_grid.unsqueeze(0))

    def forward(self, 
                p_pred: ParamSet, 
                r_recon: torch.Tensor, 
                log_r_target: torch.Tensor, 
                p_true: ParamSet, 
                config: XRefineConfig, 
                curr_sup_w: float, 
                phys_penalty: torch.Tensor):
        
        # 1. 마스킹 및 전처리
        mask = (log_r_target > -10.0).float()
        valid_points = mask.sum() + 1e-9
        log_r_recon = torch.log10(torch.clamp(r_recon, min=1e-11))

        # 2. 크리티컬 엥글(Qc) 영역 보강
        qc_mask = (self.q_grid < 0.08).float() * mask
        loss_qc = torch.sum(qc_mask * (log_r_recon - log_r_target)**2) / (qc_mask.sum() + 1e-9)

        # 3. 프린지 간격(Diff) 보강
        diff_recon = torch.diff(log_r_recon, dim=-1)
        diff_target = torch.diff(log_r_target, dim=-1)
        diff_mask = mask[:, 1:]
        loss_fringe = torch.sum(diff_mask * (diff_recon - diff_target)**2) / (diff_mask.sum() + 1e-9)

        # 4. 베이스 재구성 손실
        base_error = (log_r_recon - log_r_target)**2
        loss_recon_base = (base_error * self.q_weights * mask).sum() / valid_points

        # 5. 파라미터 지도 학습
        loss_param = 0
        w_set = config.get_weight_set(device=log_r_target.device)
        
        for fld in fields(ParamSet):
            key = fld.name
            p_range = getattr(config.ranges, key)
            p_min, p_max = p_range # Tuple unpacking
            
            norm_pred = (getattr(p_pred, key) - p_min) / (p_max - p_min + 1e-9)
            norm_true = (getattr(p_true, key) - p_min) / (p_max - p_min + 1e-9)
            
            # Weighted MSE
            w_val = getattr(w_set, key) if hasattr(w_set, key) else 1.0
            if isinstance(w_val, torch.Tensor): w_val = w_val.item()
            
            loss_param += F.mse_loss(norm_pred, norm_true) * w_val

        # 6. 최종 통합
        total_loss = (loss_recon_base + 15.0 * loss_qc + 30.0 * loss_fringe) + \
                     (curr_sup_w * loss_param) + \
                     (config.train.phys_penalty_w * phys_penalty)

        # [FIX] Return 5 values (Trainer expects 5)
        return total_loss, loss_recon_base, loss_fringe, loss_qc, loss_param

def plot_learning_curves(history: Dict[str, List[float]], save_path: str | Path):
    if 'train_loss' not in history or not history['train_loss']:
        return

    plt.figure(figsize=(10, 6))
    for label, values in history.items():
        if values:
            plt.plot(values, label=label.replace('_', ' ').title(), alpha=0.8)

    plt.yscale('log')
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title("XRefine Global Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_pinn_report(
    q_grid: torch.Tensor, 
    target_log_r: torch.Tensor, 
    r_recon: torch.Tensor, 
    loss_history: List[float], 
    params_pred: ParamSet, 
    params_true: Optional[ParamSet], 
    save_path: Path | str, 
    step: int
):
    """
    PINN 상태 종합 리포트 생성 (불필요한 Residual 제거 및 Parameter 비교 강화)
    """
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])

    # 1. [Left] 학습 곡선 (Convergence)
    ax0 = plt.subplot(gs[0])
    # 최근 1000 스텝만 표시 (너무 길면 변동 안보임)
    recent_loss = loss_history[-1000:] if len(loss_history) > 1000 else loss_history
    ax0.plot(recent_loss, color='#2c3e50', lw=1.5)
    ax0.set_yscale('log')
    ax0.set_title(f"Convergence (Recent {len(recent_loss)} Steps)")
    ax0.set_xlabel("Steps")
    ax0.set_ylabel("Total Loss")
    ax0.grid(True, alpha=0.2)

    # 2. [Right] XRR Profile Fitting
    ax1 = plt.subplot(gs[1])
    q_np = q_grid.cpu().numpy()
    
    # Batch의 첫 번째 샘플 시각화
    y_target = target_log_r[0].detach().cpu().numpy()
    y_recon = torch.log10(torch.clamp(r_recon[0], min=1e-11)).detach().cpu().numpy()

    # 데이터가 겹치는지 확인하기 위해 스타일 차별화
    ax1.plot(q_np, y_target, 'k-', lw=1, alpha=0.2, label='Target (Truth)')
    ax1.plot(q_np, y_recon, 'r--', lw=1.5, label='PINN Prediction')
    
    ax1.set_ylabel("Log10(Reflectivity)")
    ax1.set_xlabel("q ($Å^{-1}$)")
    ax1.set_title(f"Validation Sample Fit at Epoch {step}")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 3. 파라미터 텍스트 오버레이 (그래프 위에 띄우기)
    # 실제값과 예측값을 나란히 비교하여 출력
    info_lines = [f"STEP: {step:04d}"]
    
    if params_true:
        info_lines.append("\n--- Key Parameters (Pred vs True) ---")
        # 주요 파라미터만 선별 출력
        keys = ['d_f', 'sig_f', 'rho_f', 'd_ox']
        for k in keys:
            v_p = getattr(params_pred, k)[0].item()
            v_t = getattr(params_true, k)[0].item()
            err = abs(v_p - v_t) / (abs(v_t) + 1e-9) * 100
            info_lines.append(f"{k:<6}: {v_p:6.1f} | {v_t:6.1f} (Err: {err:4.1f}%)")
    else:
        info_lines.append(f"\nPRED:\n{params_pred}")

    info_text = "\n".join(info_lines)
    
    # 텍스트 박스 위치: 왼쪽 하단
    plt.gcf().text(0.02, 0.02, info_text, fontsize=10, family='monospace', 
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='#34495e', boxstyle='round,pad=0.5'),
        verticalalignment='bottom')

    plt.tight_layout(rect=[0, 0.0, 1, 0.95]) # 텍스트 공간 확보
    plt.savefig(save_path, dpi=120)
    plt.close('all')