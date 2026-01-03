from typing import Dict, List, Optional
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from xrefine.core.structures import ParamSet
from xrefine.config import XRefineConfig

plt.switch_backend('Agg')

class XRRWeightedLoss(nn.Module):
    """
    실전 데이터 대응형 XRR 손실 함수:
    1. Dynamic Masking: 유효하지 않은 데이터(-10 이하)를 Loss 계산에서 제외.
    2. Segmented Recon Loss: 저각도(Footprint)와 고각도 가중치 밸런싱.
    3. Param Supervision: 정규화된 MSE와 파라미터별 가중치 적용.
    """
    def __init__(self, q_grid: torch.Tensor):
        super().__init__()
        # 기본 고각도 감쇠 가중치 (Noise 과적합 방지)
        weights = torch.exp(-2.0 * q_grid) 
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
        
        # --- 1. Dynamic Masking (실전 데이터의 빈 부분 제외) ---
        # -10 이하(예: 사용자가 넣은 -15)인 구간은 0, 나머지는 1로 마스킹
        mask = (log_r_target > -10.0).float()
        valid_points = mask.sum() + 1e-9

        # --- 2. Physics Reconstruction Loss (Log-scale) ---
        log_r_recon = torch.log10(torch.clamp(r_recon, min=1e-11))
        
        # Footprint 영역(Q < 0.1)에 추가 가중치 부여 (L, beam_width 수렴 지원)
        footprint_boost = torch.where(self.q_grid < 0.1, 5.0, 1.0)
        
        recon_error = (log_r_recon - log_r_target)**2
        # 마스크와 영역 가중치를 모두 적용하여 평균 산출
        loss_recon = (recon_error * self.q_weights * footprint_boost * mask).sum() / valid_points

        # --- 3. Parameter Supervision Loss (Normalized & Weighted) ---
        loss_param = 0
        w_set = config.get_weight_set(device=log_r_target.device)

        for fld in fields(ParamSet):
            key = fld.name
            pred_val = getattr(p_pred, key)
            true_val = getattr(p_true, key)
            weight = getattr(w_set, key)

            p_min, p_max = getattr(config.ranges, key)
            
            # 정규화된 MSE (0~1 scale-invariant)
            norm_pred = (pred_val - p_min) / (p_max - p_min + 1e-9)
            norm_true = (true_val - p_min) / (p_max - p_min + 1e-9)
            
            loss_param += nn.functional.mse_loss(norm_pred, norm_true) * weight

        # 최종 통합 손실 (Recon + Param + Physics Penalty)
        total_loss = loss_recon + (curr_sup_w * loss_param) + (config.train.phys_penalty_w * phys_penalty)
        return total_loss, loss_recon, loss_param, phys_penalty


def plot_learning_curves(history: Dict[str, List[float]], save_path: str | Path):
    """전체 학습 과정의 Loss 추이 요약"""
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
    PINN 상태 종합 리포트 생성 시각화
    """
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[3, 1])

    # 1. [Left] 최근 500스텝 수렴 곡선
    ax0 = plt.subplot(gs[:, 0])
    recent_loss = loss_history[-500:] 
    ax0.plot(recent_loss, color='#2c3e50', lw=1.5)
    ax0.set_yscale('log')
    ax0.set_title("Convergence (Recent 500 Steps)")
    ax0.set_xlabel("Steps")
    ax0.grid(True, alpha=0.2)

    # 2. [Right Top] XRR Profile Fitting
    ax1 = plt.subplot(gs[0, 1])
    q_np = q_grid.cpu().numpy()
    # Batch의 첫 번째 샘플 시각화
    y_target = target_log_r[0].detach().cpu().numpy()
    y_recon = torch.log10(torch.clamp(r_recon[0], min=1e-11)).detach().cpu().numpy()

    ax1.plot(q_np, y_recon, 'r-', lw=2, label='PINN Physics Fit', zorder=3)
    ax1.plot(q_np, y_target, 'ko', markersize=2, alpha=0.15, label='Target Data', zorder=2)
    ax1.set_ylabel("Log10(Reflectivity)")
    ax1.set_title(f"Reflectivity Match at Step {step}")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 3. [Right Bottom] Residuals (Error)
    ax2 = plt.subplot(gs[1, 1])
    residuals = y_recon - y_target
    ax2.fill_between(q_np, residuals, 0, color='#3498db', alpha=0.3)
    ax2.plot(q_np, residuals, color='#3498db', lw=1)
    ax2.axhline(0, color='black', lw=1, ls='--')
    ax2.set_ylabel("$\\Delta$ LogR")
    ax2.set_xlabel("q ($Å^{-1}$)")
    ax2.set_ylim(-0.5, 0.5)

    # 4. [Bottom Overlay] 파라미터 비교 텍스트
    # ParamSet의 __repr__을 활용하여 깔끔하게 출력
    info_text = f"STEP: {step:06d}\n\nPRED: {params_pred}\n\nTRUE: {params_true}"
    plt.gcf().text(0.02, 0.02, info_text, fontsize=8.5, family='monospace', 
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='#34495e', boxstyle='round,pad=1'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close('all')