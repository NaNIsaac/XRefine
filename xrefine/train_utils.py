import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# --- 1. Physics-Oriented Loss Functions ---

class QWeightedLogMSE(nn.Module):
    """
    XRR의 물리적 특성(고각 신호 감쇄 및 노이즈)을 고려한 가중치 손실 함수.
    """
    def __init__(self, q_grid, q_cut=0.5):
        super().__init__()
        # q_cut 이후로 가중치가 급격히 감소하도록 설계
        weights = torch.exp(-(q_grid**2) / (2 * q_cut**2))
        self.register_buffer('weights', weights)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_log_r, target_log_r):
        # target_log_r이 이미 로그값이라면, 그 값이 너무 작아지지 않도록 clamp
        target_log_r = torch.clamp(target_log_r, min=-10.0, max=0.0)
        pred_log_r = torch.clamp(pred_log_r, min=-10.0, max=0.0)
        
        loss = self.mse(pred_log_r, target_log_r)
        return (loss * self.weights.unsqueeze(0)).mean()

def compute_loss(params_pred, r_recon, target_log_r, params_true, config, curr_sup_w, phys_penalty):
    """
    8-파라미터 체계 전용 통합 손실 함수.
    """
    criterion = nn.MSELoss()
    # 예측값 언팩 (8개)
    d_f, sig_f, rho_f, d_ox, sig_ox, rho_ox, L, log_bkg = params_pred
    # 정답값 언팩 (4개)
    t_t, r_t, s_t, l_t = params_true 

    # 1. 그래프 재구성 손실 (Physics Loss)
    recon_log_r = torch.log10(torch.clamp(r_recon, min=1e-9))
    loss_recon = criterion(recon_log_r, target_log_r)

    # 2. 파라미터 지도 학습 (Supervised Loss)
    # s_t 인덱스: 0:Amb, 1:Film, 2:SiO2, 3:Sub
    loss_param = criterion(d_f, t_t[:, 0:1]) + \
                 criterion(sig_f, r_t[:, 0:1]) + \
                 (config.sld_weight_factor * criterion(rho_f, s_t[:, 1:2])) + \
                 criterion(d_ox, t_t[:, 1:2]) + \
                 criterion(sig_ox, r_t[:, 1:2]) + \
                 criterion(rho_ox, s_t[:, 2:3]) + \
                 criterion(L, l_t) 

    # 3. 물리적 타당성 페널티 통합
    total_loss = loss_recon + (curr_sup_w * loss_param) + (10.0 * phys_penalty)

    return total_loss, loss_recon, loss_param, phys_penalty

# --- 2. Visualization & Reporting ---

def save_pinn_report(q_grid, r_target, r_recon, loss_history, params_pred, params_true, save_path, step=None):
    """실시간 학습 상태를 물리 파라미터와 함께 시각화"""
    plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[3, 1])

    # [Left] Convergence Plot
    ax0 = plt.subplot(gs[:, 0])
    ax0.plot(loss_history[-2000:], color='#2c3e50', alpha=0.8)
    ax0.set_yscale('log')
    ax0.set_title(f"Training Convergence (Step: {step})")
    ax0.set_xlabel("Recent Steps")
    ax0.set_ylabel("Total Loss")

    # [Right Top] XRR Profile Fit
    q_np = q_grid.cpu().numpy()
    target_np = r_target.cpu().numpy()[0]
    recon_np = r_recon.detach().cpu().numpy()[0]

    ax1 = plt.subplot(gs[0, 1])
    ax1.semilogy(q_np, target_np, 'ko', markersize=2, alpha=0.2, label='Measured')
    ax1.semilogy(q_np, recon_np, 'r-', lw=2, label='PINN Fit')
    ax1.set_title("Reflectivity Profile Matching")
    ax1.legend()

    # [Right Bottom] Residual Analysis
    ax2 = plt.subplot(gs[1, 1])
    residual = np.log10(recon_np + 1e-10) - np.log10(target_np + 1e-10)
    ax2.plot(q_np, residual, color='#3498db', alpha=0.7)
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_ylabel("Δ LogR")

    # [Text] Parameter Comparison
    d_f, sig_f, rho_f, d_ox, sig_ox, rho_ox, L, log_bkg = params_pred
    t_t, r_t, s_t, l_t = params_true
    
    info_text = (
        f"[Target Structure]\n"
        f"Film Thick: {d_f[0,0]:.1f}Å (T:{t_t[0,0]:.1f})\n"
        f"Film SLD:   {rho_f[0,0]:.2f}  (T:{s_t[0,1]:.2f})\n"
        f"SiO2 SLD:   {rho_ox[0,0]:.2f}  (T:{s_t[0,2]:.2f})\n"
        f"Sample L:   {L[0,0]:.2f}mm (T:{l_t[0,0]:.2f})"
    )
    plt.gcf().text(0.02, 0.05, info_text, fontsize=12, family='monospace', 
                   bbox=dict(facecolor='white', alpha=0.9, edgecolor='#e74c3c'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def plot_learning_curves(history, save_path):
    """
    Train/Val Loss 추이를 시각화 (누락되었던 함수)
    """
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    
    if not train_loss:
        return

    plt.figure(figsize=(10, 6))
    
    # Train Loss
    steps_train = np.arange(len(train_loss))
    plt.plot(steps_train, train_loss, 'b-', label='Train Loss', alpha=0.4, lw=1)
    
    # Val Loss (보통 더 적은 주기로 기록되므로 x축 보간)
    if val_loss:
        steps_val = np.linspace(0, len(train_loss) - 1, len(val_loss))
        plt.plot(steps_val, val_loss, 'r-o', label='Val Loss', markersize=4)
        
    plt.yscale('log')
    plt.title("XRefine Learning Curves", fontsize=14)
    plt.xlabel("Epochs / Log Intervals")
    plt.ylabel("Loss (Log Scale)")
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()