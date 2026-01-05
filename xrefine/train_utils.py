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
    물리 우선순위 대응형 커스텀 손실 함수:
    1. Critical Angle (Qc): SLD(rho) 결정 1순위.
    2. Fringe Gradient (Diff): 두께(d) 및 간섭 주기 결정 1순위.
    3. Parameter Weighting: d, rho, sigma에 높은 가중치 부여.
    """
    def __init__(self, q_grid: torch.Tensor):
        super().__init__()

        weights = torch.full_like(q_grid, 1)
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
        
        # --- 1. 전처리 및 마스킹 ---
        mask = (log_r_target > -10.0).float()
        valid_points = mask.sum() + 1e-9
        log_r_recon = torch.log10(torch.clamp(r_recon, min=1e-11))

        # --- 2. [핵심] Critical Angle (Qc) Loss ---
        # 전반사 임계각 영역(Q < 0.08)은 박막 밀도(rho)를 결정하는 1순위 지표입니다.
        qc_mask = (self.q_grid < 0.08).float() * mask
        loss_qc = torch.sum(qc_mask * (log_r_recon - log_r_target)**2) / (qc_mask.sum() + 1e-9)
        loss_qc *= 15.0 # Qc 정합성 우선순위 반영 가중치

        # --- 3. [핵심] Fringe Spacing (Diff) Loss ---
        # 반사율의 미분값을 비교하면 주기가 틀렸을 때(두께 오류) 매우 민감하게 반응합니다. 
        diff_recon = torch.diff(log_r_recon, dim=-1)
        diff_target = torch.diff(log_r_target, dim=-1)
        diff_mask = mask[:, 1:] # diff 연산으로 길이가 1 줄어듦
        loss_fringe = torch.sum(diff_mask * (diff_recon - diff_target)**2) / (diff_mask.sum() + 1e-9)
        loss_fringe *= 30.0 # 프린지 간격 일치 우선순위 반영 가중치

        # --- 4. Base Reconstruction Loss ---
        # 전체적인 최대값 및 경향성 일치
        base_error = (log_r_recon - log_r_target)**2
        loss_recon_base = (base_error * self.q_weights * mask).sum() / valid_points

        # --- 5. Parameter Supervision (1순위 필드 강조) ---
        loss_param = 0
        w_set = config.get_weight_set(device=log_r_target.device)

        for fld in fields(ParamSet):
            key = fld.name
            pred_val = getattr(p_pred, key)
            true_val = getattr(p_true, key)
            weight = getattr(w_set, key) # WeightConfig에서 d_f, rho_f 등 1순위 가중치 참조

            p_min, p_max = getattr(config.ranges, key)
            norm_pred = (pred_val - p_min) / (p_max - p_min + 1e-9)
            norm_true = (true_val - p_min) / (p_max - p_min + 1e-9)
            
            # 1순위 파라미터(d, rho, sig)는 WeightConfig에서 이미 높은 값을 가짐
            loss_param += F.mse_loss(norm_pred, norm_true) * weight

        # --- 6. 최종 통합 ---
        total_loss = (loss_recon_base + loss_qc + loss_fringe) + \
                     (curr_sup_w * loss_param) + \
                     (config.train.phys_penalty_w * phys_penalty)

        # Trainer의 pbar 출력을 위해 개별 항목 반환
        return total_loss, loss_recon_base, loss_fringe, loss_qc

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