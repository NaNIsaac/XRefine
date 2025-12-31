import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def compute_xrefine_loss(params_pred, r_recon, target_log_r, params_true, config, curr_sup_w):
    """물리 기반 하이브리드 Loss 통합 계산기"""
    criterion = nn.MSELoss()
    d_f, sig_f, rho_f, d_ox, sig_ox, L, log_bkg = params_pred
    t_true, r_true, s_true = params_true

    # 1. Physics Loss (그래프 정합성)
    recon_log_r = torch.log10(torch.clamp(r_recon, min=1e-9))
    loss_recon = criterion(recon_log_r, target_log_r)

    # 2. Supervised Loss (파라미터 정합성)
    loss_t = criterion(d_f, t_true[:, 0:1]) 
    loss_r = criterion(sig_f, r_true[:, 0:1])
    loss_s = criterion(rho_f, s_true[:, 1:2])
    
    # SiO2 Thickness 가이드 (Warning 방지)
    if t_true.size(1) > 1:
        loss_tox = criterion(d_ox, t_true[:, 1:2])
    else:
        loss_tox = criterion(d_ox, torch.full_like(d_ox, 15.0))

    loss_param = loss_t + loss_r + (config.sld_weight_factor * loss_s) + (10.0 * loss_tox)

    # 3. Physical Regularization (L-Prior)
    loss_L_prior = criterion(L, torch.full_like(L, 10.0))

    total_loss = loss_recon + (curr_sup_w * loss_param) + (0.5 * loss_L_prior)
    
    return total_loss, loss_recon, loss_param, loss_L_prior

def save_pinn_report(q_grid, r_target, r_recon, loss_history, params_pred, params_true, save_path, step=None):
    """
    강화된 XRR 리포트:
    - 왼쪽: Loss 수렴 곡선
    - 오른쪽 위: XRR Fit (Log scale)
    - 오른쪽 아래: Residual (잔차) 플롯
    """
    plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[3, 1])

    # [1] Convergence Plot (Left)
    ax0 = plt.subplot(gs[:, 0])
    ax0.plot(loss_history[-2000:], color='#2c3e50', lw=1.5)
    ax0.set_yscale('log')
    ax0.set_title(f"Convergence History (Step: {step if step else 'N/A'})", fontsize=14)
    ax0.set_xlabel("Steps")
    ax0.set_ylabel("Total Loss")
    ax0.grid(True, alpha=0.3)

    # 데이터 준비
    q_np = q_grid.cpu().numpy()
    target_np = r_target.cpu().numpy()[0]
    recon_np = r_recon.detach().cpu().numpy()[0]
    
    # [2] XRR Fit Plot (Right Top)
    ax1 = plt.subplot(gs[0, 1])
    ax1.semilogy(q_np, target_np, 'ko', markersize=2, alpha=0.3, label='Target (Measured)')
    ax1.semilogy(q_np, recon_np, 'r-', lw=2, label='PINN Reconstruction')
    ax1.set_title("XRR Profile Matching", fontsize=14)
    ax1.set_ylabel("Reflectivity (R)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2, which='both')

    # [3] Residual Plot (Right Bottom)
    # 잔차: log10 단위의 차이를 보여주어 물리적 불일치 영역을 특정함
    ax2 = plt.subplot(gs[1, 1])
    residual = np.log10(recon_np + 1e-10) - np.log10(target_np + 1e-10)
    ax2.plot(q_np, residual, color='blue', lw=1, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='--', lw=1)
    ax2.set_ylim(-1, 1) # ±1 order of magnitude
    ax2.set_xlabel("q (1/Å)")
    ax2.set_ylabel("Δ Log10(R)")
    ax2.set_title("Residual Analysis", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # [4] 물리 파라미터 텍스트 추가
    d_f, sig_f, rho_f, d_ox, sig_ox, L, log_bkg = params_pred
    t_t, r_t, s_t = params_true
    info_text = (
        f"Thickness: {d_f.mean():.2f} (True:{t_t[0,0]:.2f})\n"
        f"SLD: {rho_f.mean():.2f} (True:{s_t[0,1]:.2f})\n"
        f"Roughness: {sig_f.mean():.2f} (True:{r_t[0,0]:.2f})\n"
        f"L: {L.mean():.2f} | Bkg: {log_bkg.mean():.2f}"
    )
    plt.gcf().text(0.02, 0.02, info_text, fontsize=12, family='monospace', 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

def plot_learning_curves(history, save_path):
    """
    Train/Val Loss 수렴 과정을 시각화
    로그 주기가 달라도(예: Train은 1step, Val은 100step) 자동으로 x축을 맞춰서 그림
    """
    train_loss = history.get('train', [])
    val_loss = history.get('val', [])
    
    if not train_loss:
        print("⚠️ No train loss to plot.")
        return

    plt.figure(figsize=(10, 6))
    
    # 1. Train Loss용 X축 (매 스텝)
    steps_train = np.arange(len(train_loss))
    plt.plot(steps_train, train_loss, 'b-', label='Train Loss', alpha=0.5, lw=1)
    
    # 2. Val Loss용 X축 (인터벌 계산)
    if val_loss:
        # Train 데이터 개수와 Val 데이터 개수의 비율로 x축 위치를 보간(Interpolation)
        # 예: Train 5000개, Val 50개면 x축은 0, 100, 200 ... 순서가 됨
        steps_val = np.linspace(0, len(train_loss) - 1, len(val_loss))
        plt.plot(steps_val, val_loss, 'r-o', label='Val Loss', markersize=4, alpha=0.8)
        
    plt.yscale('log') # XRR 손실값은 로그 스케일 필수
    plt.title("XRefine Learning Dynamics", fontsize=14)
    plt.xlabel("Training Steps")
    plt.ylabel("Total Loss (Log Scale)")
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"📈 Convergence plot saved to: {save_path}")