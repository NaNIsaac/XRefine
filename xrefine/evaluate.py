import torch
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from xrefine.models import XRR_PINN, XRR_Fourier_PINN
from xrefine.data.generator import generate_synthetic_batch
from xrefine.data.augmentations import XRRAugmentations
from xrefine.config import XRefineConfig
from xrefine.train_utils import plot_learning_curves

# -----------------------------------------------------------------------------
# 1. Advanced Metrics
# -----------------------------------------------------------------------------
def calculate_r2(true, pred):
    """결정계수(R²) 계산: 모델의 설명력을 측정"""
    ss_res = np.sum((true - pred) ** 2, axis=0)
    ss_tot = np.sum((true - np.mean(true, axis=0)) ** 2, axis=0)
    return 1 - (ss_res / (ss_tot + 1e-8))

def calculate_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """MAE, RMSE, MAPE, R2 통합 계산"""
    errors = preds - targets
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors, axis=0)
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    mape = np.mean(abs_errors / (np.abs(targets) + 1e-6), axis=0) * 100
    r2 = calculate_r2(targets, preds)

    return {
        "errors": errors,
        "abs_errors": abs_errors,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2
    }

def print_metrics_table(metrics: dict, param_names: list):
    """
    모델의 물리적 추론 성능을 테이블 형태로 출력
    - MAE: 평균 절대 오차 (물리적 단위와 동일)
    - RMSE: 큰 오차에 페널티를 준 오차
    - MAPE: 백분율 오차 (상대적 정확도)
    - R2: 결정계수 (1.0에 가까울수록 완벽한 물리적 모사)
    """
    header = f"{'Parameter':<22} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10} | {'R2':<10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    
    for i, name in enumerate(param_names):
        # 인덱스 에러 방지를 위해 metrics 길이 확인
        if i < len(metrics['mae']):
            mae = metrics['mae'][i]
            rmse = metrics['rmse'][i]
            mape = metrics['mape'][i]
            r2 = metrics['r2'][i]
            
            print(
                f"{name:<22} | {mae:<10.4f} | "
                f"{rmse:<10.4f} | {mape:<10.2f} | "
                f"{r2:<10.4f}"
            )
    print("=" * len(header) + "\n")

# -----------------------------------------------------------------------------
# 2. Visualization & Reporting
# -----------------------------------------------------------------------------
def save_correlation_plot(df, param_names, save_path):
    """True vs Pred 상관관계 및 에러 분포 시각화"""
    n_params = len(param_names)
    fig, axes = plt.subplots(2, n_params, figsize=(5 * n_params, 9))
    
    for i, name in enumerate(param_names):
        clean = name.split(' (')[0]
        y_true = df[f"{clean}_True"].values
        y_pred = df[f"{clean}_Pred"].values
        err = df[f"{clean}_Error"].values

        # Scatter Plot
        ax_s = axes[0][i]
        min_v, max_v = y_true.min(), y_true.max()
        ax_s.plot([min_v, max_v], [min_v, max_v], 'r--', alpha=0.7)
        ax_s.scatter(y_true, y_pred, alpha=0.4, s=10, color='#3498db')
        ax_s.set_title(f"{name}\n$R^2$: {calculate_r2(y_true, y_pred):.4f}")
        ax_s.set_xlabel("Ground Truth")
        ax_s.set_ylabel("Prediction")

        # Error Hist
        ax_h = axes[1][i]
        ax_h.hist(err, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax_h.axvline(0, color='red', linestyle='--')
        ax_h.set_title(f"{clean} Error Dist.")
        ax_h.set_xlabel("Error (Pred - True)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_advanced_reconstruction(qs, r_true, r_pred, target_params, pred_params, save_path, n_samples=6):
    """XRR 프로파일 피팅 결과와 잔차(Residual) 분석 리포트"""
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.2)
    
    indices = np.random.choice(len(r_true), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i], height_ratios=[3, 1], hspace=0.05)
        ax_top = plt.Subplot(fig, inner[0])
        ax_bot = plt.Subplot(fig, inner[1])
        fig.add_subplot(ax_top)
        fig.add_subplot(ax_bot)
        
        # Profile
        ax_top.semilogy(qs, r_true[idx], 'ko', markersize=2, alpha=0.3, label='True')
        ax_top.semilogy(qs, r_pred[idx], 'r-', lw=1.5, label='PINN')
        ax_top.set_title(f"Sample #{idx} | d_err: {pred_params[idx,0]-target_params[idx,0]:.2f}Å")
        ax_top.set_xticks([])
        ax_top.legend(fontsize=8)
        
        # Residual
        residual = np.log10(r_pred[idx] + 1e-10) - np.log10(r_true[idx] + 1e-10)
        ax_bot.plot(qs, residual, color='blue', alpha=0.6)
        ax_bot.axhline(0, color='black', lw=1, linestyle='--')
        ax_bot.set_ylim(-0.5, 0.5)
        ax_bot.set_ylabel("$\\Delta$LogR", fontsize=8)

    plt.savefig(save_path, dpi=150)
    plt.close()

# -----------------------------------------------------------------------------
# 3. Main Evaluation Function
# -----------------------------------------------------------------------------
def evaluate(args):
    checkpoint_path = Path(args.checkpoint)
    print(f"🔍 Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    history = checkpoint.get('history', {})
    
    if history:
        print("📈 Generating Convergence Plot...")
        plot_learning_curves(history, exp_dir / "learning_curves.png")
    else:
        print("⚠️ No history found in checkpoint. Skipping convergence plot.")
    # 1. Config 복원
    config_data = checkpoint.get('config')
    config = XRefineConfig(**config_data) if isinstance(config_data, dict) else config_data
    device = config.device
    
    # 2. 모델 초기화 및 로드
    if config.model_type == 'fourier':
        pinn = XRR_Fourier_PINN(config=config, device=device).to(device)
    else:
        pinn = XRR_PINN(config=config, device=device).to(device)
    pinn.load_state_dict(checkpoint['model_state_dict'])
    pinn.eval()
    
    # 3. 테스트 데이터 생성 (Batch Generation)
    q_grid = config.q_grid.to(device)
    augmenter = XRRAugmentations(delta_q=0.0005, do_footprint=True)
    
    print(f"🧪 Generating {args.batch_size} test samples...")
    r_measured, (t_true, r_true, s_true) = generate_synthetic_batch(
        args.batch_size, q_grid, config=config, augmenter=augmenter
    )
    
    # 4. Inference
    print("🧠 Running Inference...")
    with torch.no_grad():
        target_log_r = torch.log10(torch.clamp(r_measured, min=1e-9))
        params_pred, r_reconstructed = pinn(target_log_r)
        
    # 5. 결과 언팩 및 정규화 해제 (IndexError 방지)
    # [Film Parameters]
    d_f_p, sig_f_p, rho_f_p, d_ox_p, sig_ox_p, L_p, bkg_p = params_pred
    
    # CPU Numpy 변환
    t_pred_np = d_f_p.cpu().numpy()[:, 0]
    r_pred_np = sig_f_p.cpu().numpy()[:, 0]
    s_pred_np = rho_f_p.cpu().numpy()[:, 0]
    L_pred_np = L_p.cpu().numpy()[:, 0]

    # Ground Truth Unpack (Generator 구조 대응)
    t_true_np = t_true.cpu().numpy()[:, 0]     # Main Film Thickness
    r_true_np = r_true.cpu().numpy()[:, 0]     # Top Interface Roughness
    s_true_np = s_true.cpu().numpy()[:, 1]     # Film SLD (Index 1: Amb, [Film], Sub)
    L_true_np = np.ones_like(L_pred_np) * 10.0 # 고정 타겟 10mm

    # 6. Metrics 계산
    preds = np.stack([t_pred_np, r_pred_np, s_pred_np, L_pred_np], axis=1)
    targets = np.stack([t_true_np, r_true_np, s_true_np, L_true_np], axis=1)
    param_names = ["Thickness (Å)", "Roughness (Å)", "SLD (10⁻⁶ Å⁻²)", "Sample Len (mm)"]
    
    metrics = calculate_metrics(preds, targets)
    print_metrics_table(metrics, param_names)

    # 7. 리포트 저장
    exp_dir = checkpoint_path.parent
    df = pd.DataFrame()
    for i, name in enumerate(param_names):
        clean = name.split(' (')[0]
        df[f"{clean}_True"] = targets[:, i]
        df[f"{clean}_Pred"] = preds[:, i]
        df[f"{clean}_Error"] = metrics["errors"][:, i]

    df.to_csv(exp_dir / "eval_results.csv", index=False)
    save_correlation_plot(df, param_names, exp_dir / "eval_correlation.png")
    save_advanced_reconstruction(
        q_grid.cpu().numpy(), r_measured.cpu().numpy(), r_reconstructed.cpu().numpy(),
        targets, preds, exp_dir / "eval_reconstruction.png"
    )
    
    print(f"✅ Evaluation Complete. Reports in: {exp_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()
    evaluate(args)