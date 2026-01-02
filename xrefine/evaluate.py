import torch
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from xrefine.models import XRR_Fourier_PINN
from xrefine.data.generator import generate_synthetic_batch
from xrefine.data.augmentations import XRRAugmentations
from xrefine.config import XRefineConfig
from xrefine.train_utils import plot_learning_curves
from xrefine.logger import logger
from xrefine.core.structures import ParamSet

# -----------------------------------------------------------------------------
# 1. Advanced Metrics
# -----------------------------------------------------------------------------
def calculate_r2(true: np.ndarray, pred: np.ndarray):
    ss_res = np.sum((true - pred) ** 2, axis=0)
    ss_tot = np.sum((true - np.mean(true, axis=0)) ** 2, axis=0)
    return 1 - (ss_res / (ss_tot + 1e-8))

def calculate_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    errors = preds - targets
    abs_errors = np.abs(errors)
    mae = np.mean(abs_errors, axis=0)
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    mape = np.mean(abs_errors / (np.abs(targets) + 1e-6), axis=0) * 100
    r2 = calculate_r2(targets, preds)
    return {"errors": errors, "abs_errors": abs_errors, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

def print_metrics_table(metrics: dict, param_names: list):
    header = f"{'Parameter':<22} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10} | {'R2':<10}"
    xlogger.info("\n" + "=" * len(header))
    xlogger.info(header)
    xlogger.info("-" * len(header))
    for i, name in enumerate(param_names):
        if i < len(metrics['mae']):
            xlogger.info(f"{name:<22} | {metrics['mae'][i]:<10.4f} | {metrics['rmse'][i]:<10.4f} | {metrics['mape'][i]:<10.2f} | {metrics['r2'][i]:<10.4f}")
    xlogger.info("=" * len(header) + "\n")

# -----------------------------------------------------------------------------
# 2. Visualization & Reporting
# -----------------------------------------------------------------------------
def save_correlation_plot(df: pd.DataFrame, param_names: list, save_path: Path):
    n_params = len(param_names)
    fig, axes = plt.subplots(2, n_params, figsize=(5 * n_params, 9))
    
    # 1개 파라미터일 경우 axes가 1D가 되는 현상 방지
    if n_params == 1:
        axes = axes.reshape(2, 1)

    for i, name in enumerate(param_names):
        clean = name.split(' (')[0]
        y_true, y_pred, err = df[f"{clean}_True"], df[f"{clean}_Pred"], df[f"{clean}_Error"]
        
        # Scatter Plot
        ax_s = axes[0][i]
        min_v, max_v = y_true.min(), y_true.max()
        ax_s.plot([min_v, max_v], [min_v, max_v], 'r--', alpha=0.7)
        ax_s.scatter(y_true, y_pred, alpha=0.4, s=10, color='#3498db')
        ax_s.set_title(f"{name}\n$R^2$: {calculate_r2(y_true.values, y_pred.values):.4f}")
        
        # Error Histogram
        ax_h = axes[1][i]
        ax_h.hist(err, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax_h.axvline(0, color='red', linestyle='--')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_advanced_reconstruction(qs, r_true, r_pred, target_vals, pred_vals, save_path, n_samples=6):
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.2)
    indices = np.random.choice(len(r_true), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i], height_ratios=[3, 1], hspace=0.05)
        ax_top, ax_bot = plt.Subplot(fig, inner[0]), plt.Subplot(fig, inner[1])
        fig.add_subplot(ax_top); fig.add_subplot(ax_bot)
        
        ax_top.semilogy(qs, r_true[idx], 'ko', markersize=2, alpha=0.3, label='True')
        ax_top.semilogy(qs, r_pred[idx], 'r-', lw=1.5, label='PINN')
        ax_top.set_title(f"Sample #{idx} | d_err: {pred_vals[idx,0]-target_vals[idx,0]:.2f}Å")
        
        # Residual Plot (Log Scale)
        residual = np.log10(r_pred[idx] + 1e-10) - np.log10(r_true[idx] + 1e-10)
        ax_bot.plot(qs, residual, color='blue', alpha=0.6)
        ax_bot.axhline(0, color='black', lw=1, linestyle='--')
        ax_bot.set_ylim(-0.5, 0.5)
        
    plt.savefig(save_path, dpi=150)
    plt.close()

# -----------------------------------------------------------------------------
# 3. Main Evaluation Function
# -----------------------------------------------------------------------------
def evaluate(args):
    checkpoint_path = Path(args.checkpoint)
    exp_dir = checkpoint_path.parent 
    if not checkpoint_path.exists():
        xlogger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    xlogger.info(f"🔍 Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 1. 수렴 곡선 출력
    history = checkpoint.get('history', {})
    if history:
        plot_learning_curves(history, exp_dir / "learning_curves.png")

    # 2. Config 및 모델 로드
    config_data = checkpoint.get('config')
    config = XRefineConfig(**config_data) if isinstance(config_data, dict) else config_data
    device = config.device
    
    xlogger.info(f"🧠 Initializing Model: {config.model_type}")
    pinn = XRR_Fourier_PINN(config=config, device=device).to(device)
    pinn.load_state_dict(checkpoint['model_state_dict'])
    pinn.eval()
    
    # 3. 데이터 생성 (ParamSet 기반)
    q_grid = config.q_grid.to(device)
    # 중복 풋프린트 방지를 위해 do_footprint=False (Generator에서 수행하므로)
    base_augmenter = XRRAugmentations(delta_q=0.0005, do_footprint=False)
    
    xlogger.info(f"🧪 Generating {args.batch_size} test samples...")
    r_measured, p_true = generate_synthetic_batch(
        args.batch_size, q_grid, config=config, augmenter=base_augmenter
    )
    
    # 4. Inference
    xlogger.info("🧠 Running Inference...")
    with torch.no_grad():
        target_log_r = torch.log10(torch.clamp(r_measured, min=1e-9))
        p_pred, r_reconstructed, _ = pinn(target_log_r)
        
    # 5. 결과 언팩 (ParamSet Key 기반 접근)
    # Numpy 변환 및 차원 정렬 (N=1 가정)
    def to_np(tensor): return tensor.cpu().numpy()[:, 0]

    # 비교할 타겟 파라미터 정의 (I0, bkg 포함)
    eval_keys = ['d_f', 'sig_f', 'rho_f', 'L', 'I0', 'bkg']
    param_display_names = [
        "Thickness (Å)", "Roughness (Å)", "SLD (10⁻⁶ Å⁻²)", 
        "Sample Len (mm)", "Intensity Scale", "Log Background"
    ]

    preds_list = []
    targets_list = []

    for k in eval_keys:
        preds_list.append(to_np(p_pred[k]))
        if k in p_true:
            targets_list.append(to_np(p_true[k]))
        else:
            # ParamSet에 없는 고정 상수 대응 (예: n_layers 확장 시)
            targets_list.append(np.zeros(args.batch_size))

    # Metrics 계산용 Stack
    preds_np = np.stack(preds_list, axis=1)
    targets_np = np.stack(targets_list, axis=1)
    
    metrics = calculate_metrics(preds_np, targets_np)
    print_metrics_table(metrics, param_display_names)

    # 6. 데이터프레임 생성 및 저장
    df = pd.DataFrame()
    for i, name in enumerate(param_display_names):
        clean = name.split(' (')[0].replace(" ", "_")
        df[f"{clean}_True"] = targets_np[:, i]
        df[f"{clean}_Pred"] = preds_np[:, i]
        df[f"{clean}_Error"] = metrics["errors"][:, i]

    df.to_csv(exp_dir / "eval_results.csv", index=False)
    
    # 7. 시각화 리포트
    save_correlation_plot(df, param_display_names, exp_dir / "eval_correlation.png")
    save_advanced_reconstruction(
        q_grid.cpu().numpy(), r_measured.cpu().numpy(), r_reconstructed.cpu().numpy(),
        targets_np, preds_np, exp_dir / "eval_reconstruction.png"
    )
    
    xlogger.info(f"✅ Evaluation Complete. Reports in: {exp_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()
    evaluate(args)