import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from typing import Dict, List, Optional

from xrefine.core.structures import ParamSet, format_params
from xrefine.logger import logger
from xrefine.config import XRefineConfig


class QWeightedLogMSE(nn.Module):
    """
    XRR의 물리적 특성을 고려한 가중치 손실 함수.
    $q$가 커질수록 신호대잡음비(SNR)가 낮아지는 것을 가우시안 감쇄 가중치로 보정합니다.
    $$W(q) = \exp\left(-\frac{q^2}{2q_{cut}^2}\right)$$
    """
    def __init__(self, q_grid: torch.Tensor, q_cut: float = 0.5):
        super().__init__()
        # q_grid와 동일한 형태의 가중치 미리 계산 (버퍼 등록으로 GPU 이동 자동화)
        weights = torch.exp(-(q_grid**2) / (2 * q_cut**2))
        self.register_buffer('weights', weights)

    def forward(self, pred_log_r: torch.Tensor, target_log_r: torch.Tensor) -> torch.Tensor:
        # 역산된 R값이 0이 되어 log(-inf)가 되는 것을 방지
        pred_log_r = torch.clamp(pred_log_r, min=-10.0, max=0.0)
        target_log_r = torch.clamp(target_log_r, min=-10.0, max=0.0)
        
        mse = (pred_log_r - target_log_r)**2
        return (mse * self.weights).mean()

def compute_loss(
    params_pred: ParamSet, 
    r_recon: torch.Tensor, 
    target_log_r: torch.Tensor, 
    params_true: ParamSet, 
    config: XRefineConfig, 
    curr_sup_w: float, 
    phys_penalty: torch.Tensor
):
    """
    통합 손실 함수: Reconstruction(물리) + Supervision(라벨) + Penalty(제약조건)
    """
    # 1. Physics Reconstruction Loss (Log Scale)
    recon_log_r = torch.log10(torch.clamp(r_recon, min=1e-10))
    loss_recon = nn.functional.mse_loss(recon_log_r, target_log_r)

    # 2. Parameter Supervision Loss (가중치 적용)
    loss_param = torch.tensor(0.0, device=config.device)
    p_weights = config.supervise_weight_params
    
    for k in params_pred.keys():
        if k in params_true:
            w = p_weights.get(k, 1.0)
            loss_param += nn.functional.mse_loss(params_pred[k], params_true[k]) * w

    # 3. Total Loss Combined
    total_loss = loss_recon + (curr_sup_w * loss_param) + (config.phys_penalty_w * phys_penalty)

    return total_loss, loss_recon, loss_param, phys_penalty

# --- 3. Visualization & Logging ---

def save_pinn_report(
    q_grid: torch.Tensor, 
    r_target: torch.Tensor, 
    r_recon: torch.Tensor, 
    loss_history: List[float], 
    params_pred: ParamSet, 
    params_true: ParamSet, 
    save_path: str, 
    step: Optional[int] = None
):
    """학습 상태 시각화 및 파라미터 로깅"""
    
    # 파라미터 텍스트 생성
    pred_str = format_params(params_pred, "Predicted")
    true_str = format_params(params_true, "Ground Truth")
    
    # 1. 로거에 기록 (파일 및 콘솔로 전송)
    if step is not None:
        logger.info(f"Step {step} Reporting:\n{pred_str}")

    # 2. 시각화 생성
    plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[3, 1])

    # [Left] Convergence
    ax0 = plt.subplot(gs[:, 0])
    history_slice = loss_history[-2000:]
    ax0.plot(history_slice, color='#2c3e50', lw=1.5)
    ax0.set_yscale('log')
    ax0.set_title(f"Convergence Path (Recent {len(history_slice)} steps)")
    ax0.grid(True, which='both', alpha=0.2)

    # [Right Top] XRR Profile
    q_np = q_grid.cpu().numpy()
    target_np = r_target.detach().cpu().numpy()[0]
    recon_np = r_recon.detach().cpu().numpy()[0]

    ax1 = plt.subplot(gs[0, 1])
    ax1.semilogy(q_np, target_np, 'ko', markersize=2, alpha=0.15, label='Measured')
    ax1.semilogy(q_np, recon_np, 'r-', lw=2, label='PINN Physics Recon')
    ax1.set_title(f"XRR Profile Matching (Step: {step if step else 'N/A'})")
    ax1.set_ylabel("Reflectivity (Log)")
    ax1.legend(loc='upper right')

    # [Right Bottom] Residuals
    ax2 = plt.subplot(gs[1, 1])
    res = np.log10(recon_np + 1e-10) - np.log10(target_np + 1e-10)
    ax2.fill_between(q_np, res, 0, color='#3498db', alpha=0.3)
    ax2.plot(q_np, res, color='#3498db', lw=1)
    ax2.axhline(0, color='black', lw=1, ls='--')
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_ylabel("Δ LogR")

    # 정보 텍스트 삽입
    info_text = f"Step: {step}\n{pred_str}\n{true_str}"
    plt.gcf().text(0.02, 0.02, info_text, fontsize=9, family='monospace', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='#34495e'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_learning_curves(history: Dict[str, List[float]], save_path: str):
    """전체 학습 과정의 Loss 추이 요약"""
    if 'train_loss' not in history or not history['train_loss']:
        return

    plt.figure(figsize=(10, 6))
    for label, values in history.items():
        if values:
            plt.plot(values, label=label.replace('_', ' ').title(), alpha=0.8)

    plt.yscale('log')
    plt.xlabel("Logging Intervals")
    plt.ylabel("Loss Value")
    plt.title("XRefine Global Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()