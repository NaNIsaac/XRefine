import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional
from xrefine.core.structures import ParamSet, format_params

# 서버 환경(Colab/SSH)에서 GUI 창이 뜨지 않도록 설정
plt.switch_backend('Agg')

class XRRWeightedLoss(nn.Module):
    """
    XRR 특화 손실 함수:
    1. Reconstruction Loss: Log-scale MSE + Q-weighting
    2. Parameter Loss: Normalized MSE (Curriculum)
    """
    def __init__(self, q_grid: torch.Tensor):
        super().__init__()
        # 고각도로 갈수록 가중치를 줄여 노이즈에 과적합되는 것을 방지
        weights = torch.exp(-2.0 * q_grid) 
        self.register_buffer('q_weights', weights.unsqueeze(0))

    def forward(self, p_pred: ParamSet, r_recon: torch.Tensor, 
                log_r_target: torch.Tensor, p_true: ParamSet, 
                config, curr_sup_w: float, phys_penalty: torch.Tensor):
        
        # 1. Physics Reconstruction Loss (Log-scale)
        log_r_recon = torch.log10(torch.clamp(r_recon, min=1e-11))
        recon_error = (log_r_recon - log_r_target)**2
        loss_recon = (recon_error * self.q_weights).mean()

        # 2. Parameter Supervision Loss (Normalized)
        loss_param = 0
        p_weights = config.supervise_weight_params
        for k in p_pred.keys():
            if k in p_true:
                p_min, p_max = config.get_range(k)
                norm_pred = (p_pred[k] - p_min) / (p_max - p_min + 1e-9)
                norm_true = (p_true[k] - p_min) / (p_max - p_min + 1e-9)
                w = p_weights.get(k, 1.0)
                loss_param += nn.functional.mse_loss(norm_pred, norm_true) * w

        total_loss = loss_recon + (curr_sup_w * loss_param) + (config.phys_penalty_w * phys_penalty)
        return total_loss, loss_recon, loss_param, phys_penalty


def plot_learning_curves(history: Dict[str, List[float]], save_path: str):
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
    save_path: str, 
    step: int
):
    """
    PINN 상태 종합 리포트 생성:
    - XRR Profile Fitting
    - Residuals (Δ LogR)
    - Convergence Curve
    - Parameter Text Comparison
    """
    fig = plt.figure(figsize=(16, 10))
    # 레이아웃 설정: 왼쪽(Loss), 오른쪽 위(XRR), 오른쪽 아래(Residuals)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[3, 1])

    # 1. [Left] 최근 Loss 수렴 곡선
    ax0 = plt.subplot(gs[:, 0])
    recent_loss = loss_history[-500:] # 최근 500스텝만 시각화
    ax0.plot(recent_loss, color='#2c3e50', lw=1.5)
    ax0.set_yscale('log')
    ax0.set_title("Convergence (Recent Steps)")
    ax0.set_xlabel("Steps")
    ax0.grid(True, alpha=0.2)

    # 2. [Right Top] XRR Profile
    ax1 = plt.subplot(gs[0, 1])
    q_np = q_grid.cpu().numpy()
    # target_log_r은 이미 로그 스케일, r_recon은 선형이므로 로그 변환
    y_target = target_log_r[0].detach().cpu().numpy()
    y_recon = torch.log10(torch.clamp(r_recon[0], min=1e-11)).detach().cpu().numpy()

    ax1.plot(q_np, y_recon, 'r-', lw=2, label='PINN Physics Fit (Log)')
    ax1.plot(q_np, y_target, 'ko', markersize=2, alpha=0.15, label='Measured (Log)')
    ax1.set_ylabel("Log10(Reflectivity)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 3. [Right Bottom] Residuals
    ax2 = plt.subplot(gs[1, 1])
    residuals = y_recon - y_target
    ax2.fill_between(q_np, residuals, 0, color='#3498db', alpha=0.3)
    ax2.plot(q_np, residuals, color='#3498db', lw=1)
    ax2.axhline(0, color='black', lw=1, ls='--')
    ax2.set_ylabel("Δ LogR")
    ax2.set_xlabel("q ($Å^{-1}$)")
    ax2.set_ylim(-0.6, 0.6)

    # 4. [Overlay] 파라미터 텍스트 정보
    pred_str = format_params(params_pred, "Predicted")
    true_str = format_params(params_true, "Ground Truth") if params_true else ""
    info_text = f"Step: {step}\n\n{pred_str}\n{true_str}"
    
    plt.gcf().text(0.02, 0.02, info_text, fontsize=9, family='monospace', 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='#34495e'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close('all') # 메모리 누수 방지