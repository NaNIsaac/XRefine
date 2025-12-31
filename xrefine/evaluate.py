import torch
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from xrefine.models import XRR_PINN, XRR_Fourier_PINN
from xrefine.data.generator import generate_synthetic_batch
from xrefine.data.augmentations import XRRAugmentations
from xrefine.config import XRefineConfig



# -----------------------------------------------------------------------------
# 1. Advanced Metrics
# -----------------------------------------------------------------------------
def calculate_r2(true, pred):
    """Calculate R^2 (Coefficient of Determination)"""
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

    return {
        "errors": errors,
        "abs_errors": abs_errors, # For finding worst cases
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2
    }

# -----------------------------------------------------------------------------
# 2. Reporting & Saving
# -----------------------------------------------------------------------------
def save_results_csv(df: pd.DataFrame, save_path: Path):
    if not save_path: return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"[Save] CSV Report: {save_path.name}")

def save_worst_cases(df: pd.DataFrame, abs_errors: np.ndarray, save_path: Path, top_k=50):
    """Save worst error samples info"""
    if not save_path:
        return

    # Assuming Thickness is index 0
    thick_errs = abs_errors[:, 0]
    worst_indices = np.argsort(thick_errs)[::-1][:top_k]

    worst_df = df.iloc[worst_indices].copy()
    worst_df.insert(0, "Sample_ID", worst_indices)

    worst_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"[Save] Worst Cases Log: {save_path.name}")
    return worst_indices

def print_metrics_table(metrics: dict, param_names: list):
    print("\n" + "=" * 80)
    print(f"{'Parameter':<20} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10} | {'R2':<10}")
    print("-" * 80)
    for i, name in enumerate(param_names):
        print(
            f"{name:<20} | {metrics['mae'][i]:<10.4f} | "
            f"{metrics['rmse'][i]:<10.4f} | {metrics['mape'][i]:<10.2f} | "
            f"{metrics['r2'][i]:<10.4f}"
        )
    print("=" * 80)

# -----------------------------------------------------------------------------
# 3. Visualization
# -----------------------------------------------------------------------------
def save_correlation_plot(df: pd.DataFrame, param_names: list, save_path: Path):
    if not save_path: return

    n_params = len(param_names)
    fig, axes = plt.subplots(2, n_params, figsize=(6 * n_params, 10))

    scatter_color = '#3B8ED0'
    line_color = '#E63946'
    hist_color = '#2A9D8F'

    for i, name in enumerate(param_names):
        clean = name.split(' (')[0]
        y_true = df[f"{clean}_True"].values
        y_pred = df[f"{clean}_Pred"].values
        err = df[f"{clean}_Error"].values

        # 1. Scatter Plot
        ax_s = axes[0][i]
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())

        ax_s.plot([min_v, max_v], [min_v, max_v], color=line_color, lw=2, linestyle='--')
        ax_s.scatter(y_true, y_pred, alpha=0.3, s=15, color=scatter_color, edgecolor='none')

        r2 = calculate_r2(y_true, y_pred)
        ax_s.set_title(f"{name}\n$R^2$ = {r2:.4f}", fontsize=12, fontweight='bold')
        ax_s.set_xlabel("Ground Truth")
        ax_s.set_ylabel("Prediction")
        ax_s.grid(True, linestyle=':', alpha=0.6)

        # 2. Error Histogram
        ax_h = axes[1][i]
        mu, sigma = np.mean(err), np.std(err)

        ax_h.hist(err, bins=60, density=True, alpha=0.7, color=hist_color, edgecolor='black', linewidth=0.5)
        ax_h.axvline(0, color=line_color, linestyle='--', linewidth=1.5)
        ax_h.axvline(mu, color='orange', linestyle=':', linewidth=2, label=f"Mean: {mu:.2f}")

        ax_h.set_title(f"Error Distribution (Std: {sigma:.2f})")
        ax_h.set_xlabel("Error (Pred - True)")
        ax_h.legend(fontsize=9)
        ax_h.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Save] Correlation Plot: {save_path.name}")

def plot_error_heatmap(df: pd.DataFrame, save_path: Path | None = None):
    """Heatmap of Thickness MAE binned by True Thickness vs Roughness"""
    n_bins = 10
    plot_df = df.copy()
    plot_df['Abs_Thick_Err'] = plot_df['Thickness_Error'].abs()

    plot_df['Thick_Bin'] = pd.cut(plot_df['Thickness_True'], bins=n_bins).apply(lambda x: int(x.mid))
    plot_df['Rough_Bin'] = pd.cut(plot_df['Roughness_True'], bins=n_bins).apply(lambda x: int(x.mid))

    pivot_table = plot_df.pivot_table(
        values='Abs_Thick_Err',
        index='Rough_Bin',
        columns='Thick_Bin',
        aggfunc='mean',
        observed=False
    )
    pivot_table = pivot_table.sort_index(ascending=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap='YlOrRd',
        cbar_kws={'label': 'Mean Absolute Error (Å)'},
        linewidths=.5
    )
    plt.title("Prediction Difficulty Map: Thickness MAE", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("True Thickness (Å)", fontsize=12)
    plt.ylabel("True Roughness (Å)", fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Save] Error Heatmap: {save_path.name}")
    plt.close()

def plot_single_curve(ax_main, ax_res, qs, R_true, R_pred, title):
    # Main Curve
    ax_main.semilogy(qs, R_true, 'k-', lw=2.0, alpha=0.6, label='Measured')
    ax_main.semilogy(qs, R_pred, 'r--', lw=1.5, label='AI Recon')
    # ax_main.set_yscale('log') # Already done via semilogy
    ax_main.set_xticklabels([]) 
    ax_main.grid(True, which='both', alpha=0.3)
    ax_main.legend(loc='upper right', fontsize=8)
    ax_main.set_title(title, fontsize=10, fontweight='bold')

    # Residuals
    log_true = np.log10(np.maximum(R_true, 1e-15))
    log_pred = np.log10(np.maximum(R_pred, 1e-15))
    diff = log_pred - log_true

    ax_res.plot(qs, diff, 'b-', lw=1.0, alpha=0.7)
    ax_res.axhline(0, color='k', linestyle='--', lw=1.0)
    ax_res.set_ylim(-2, 2)
    ax_res.set_ylabel(r"$\Delta$ LogR", fontsize=8)
    ax_res.set_xlabel(r"$q (\AA^{-1})$", fontsize=9)
    ax_res.grid(True, alpha=0.3)

def save_advanced_reconstruction(
    preds_params: np.ndarray, target_params: np.ndarray, 
    qs_grid: np.ndarray,
    R_true_batch: np.ndarray, R_pred_batch: np.ndarray,
    save_path: Path, worst_indices: np.ndarray | None = None, n_random: int = 3
):
    if not save_path: return

    # Selection: Top 3 Worst Cases + 3 Random Cases
    if worst_indices is not None and len(worst_indices) >= 3:
        target_indices = np.concatenate([
            worst_indices[:3],
            np.random.choice(len(preds_params), n_random, replace=False)
        ])
        labels = ["Worst #1", "Worst #2", "Worst #3", "Random A", "Random B", "Random C"]
    else:
        target_indices = np.random.choice(len(preds_params), 6, replace=False)
        labels = [f"Random #{i}" for i in range(6)]

    fig = plt.figure(figsize=(18, 10))
    outer_grid = gridspec.GridSpec(2, 3, wspace=0.2, hspace=0.3)

    print("[Plot] Generating Advanced Reconstruction Report...")

    # Iterate over selected indices
    # We must be careful if target_indices are valid
    num_samples = len(preds_params)
    
    for i, idx in enumerate(target_indices):
        if idx >= num_samples: continue

        inner_grid = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_grid[i], height_ratios=[3, 1], hspace=0.05
        )
        ax_main = plt.Subplot(fig, inner_grid[0])
        ax_res = plt.Subplot(fig, inner_grid[1])
        fig.add_subplot(ax_main)
        fig.add_subplot(ax_res)

        # Data
        t_p = target_params[idx] # [Thick, Rough, SLD]
        p_p = preds_params[idx]
        
        # Curves
        # R_true_batch and R_pred_batch are [Batch, Q]
        R_true = R_true_batch[idx]
        R_pred = R_pred_batch[idx]

        # Title Info
        # Assuming index 0 is thickness
        err_d = p_p[0] - t_p[0]
        header = f"[{labels[i]}] Sample #{idx}"
        info = f"Thick: {t_p[0]:.1f}A (Err: {err_d:+.1f}A)"

        plot_single_curve(ax_main, ax_res, qs_grid, R_true, R_pred, f"{header}\n{info}")

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Save] Reconstruction Plot: {save_path.name}")

# -----------------------------------------------------------------------------
# 4. Main Evaluation Function
# -----------------------------------------------------------------------------
def evaluate(args):
    # Load Checkpoint
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    exp_dir = checkpoint_path.parent
    print(f"🔍 Loading checkpoint from {checkpoint_path}...")
    
    # Load CPU first to avoid memory issues if GPU is busy
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract Config
    config_obj = checkpoint.get('config', None)
    if not config_obj:
        print("⚠️ No config found in checkpoint. Cannot proceed reliably.")
        return
        
    # Ensure it's XRefineConfig (it should be if saved correctly, or dict)
    if isinstance(config_obj, dict):
        config = XRefineConfig(**config_obj)
    else:
        config = config_obj
        
    device = config.device
    
    # Initialize Model
    print(f"🧠 Initializing Model: {config.model_type} ({config.n_layers} layers)")
    if config.model_type == 'fourier':
        pinn = XRR_Fourier_PINN(config=config, device=device).to(device)
    else:
        pinn = XRR_PINN(config=config, device=device).to(device)
        
    pinn.load_state_dict(checkpoint['model_state_dict'])
    pinn.eval()
    
    # Setup Augmenter
    # Use config params
    q_grid = config.q_grid.to(device)
    if q_grid.ndim > 0:
        delta_q = (q_grid[1] - q_grid[0]).item()
    else:
        delta_q = 0.0005
        
    aug_cfg = config.augmentations
    augmenter = XRRAugmentations(
        delta_q=delta_q,
        intensity_noise_scale=aug_cfg.intensity_noise_scale,
        bg_range=aug_cfg.bg_range,
        res_sigma_range=aug_cfg.res_sigma_range,
        q_shift_sigma=aug_cfg.q_shift_sigma,
        beam_profile=aug_cfg.beam_profile,
        do_footprint=True 
    )
    
    # Large Batch Generation
    BATCH_SIZE = args.batch_size
    print(f"🧪 Generating {BATCH_SIZE} synthetic samples for evaluation...")
    
    # Generate data
    r_measured_cuda, params_tuple_cuda = generate_synthetic_batch(
        BATCH_SIZE, q_grid, config=config, augmenter=augmenter
    )
    
    # Run Inference
    print("🧠 Running Inference...")
    with torch.no_grad():
        params_pred_tuple, r_reconstructed_cuda = pinn(torch.log10(r_measured_cuda + 1e-12))

    # Move to CPU
    r_measured = r_measured_cuda.cpu().numpy()
    r_reconstructed = r_reconstructed_cuda.cpu().numpy()
    
    # Unpack Truth
    # Generator returns (thick [B,N], rough [B,N+1], sld [B,3])
    # For N=1: 
    # Thick: [B, 1]
    # Rough: [B, 2] (Amb/L1, L1/Sub)
    # SLD: [B, 3] (Amb, L1, Sub)
    
    t_true = params_tuple_cuda[0].cpu().numpy()
    r_true = params_tuple_cuda[1].cpu().numpy()
    s_true = params_tuple_cuda[2].cpu().numpy()
    
    # Unpack Preds
    # Model returns (thick [B,N], rough [B,?], sld [B,?])
    # FourierModel: 
    # Thick: [B, N]
    # Rough: [B, N+1] ! Fixed in decoupling
    # SLD: [B, N] ! Fixed in decoupling? 
    # Wait, FourierModel sld output is size N? No, let's check fourier_model.py
    # sld = out[...] size N. 
    # returned (thick, rough_full, sld_full) ? 
    # r_final = ...
    # return (thick, rough_full, sld_full), r_final
    # rough_full: size N+1
    # sld_full: size N+2
    
    t_pred = params_pred_tuple[0].cpu().numpy()
    r_pred = params_pred_tuple[1].cpu().numpy()
    s_pred = params_pred_tuple[2].cpu().numpy()
    
    # Comparison Strategy
    # Thickness: Direct
    # Roughness: Direct (both N+1)
    # SLD: Direct (both N+2) -- Wait, did Model predict full SLD? 
    # Model predicted 'sld' (size N), constructed 'sld_full' (size N+2) with fixed Amb/Sub.
    # So we compare full profiles? Or just the variable parts?
    # Evaluation should ideally evaluate what the model *predicted*.
    # But evaluating the full physics profile (SLD vector) is also good.
    # Let's verify dimensions match.
    
    # Taking mean or specific layer for summary table
    # Let's take Layer 1 Thickness, Layer 1 Roughness, Layer 1 SLD
    
    t_true_l1 = t_true[:, 0]
    t_pred_l1 = t_pred[:, 0]
    
    # Roughness: Layer 1 is usually index 0 (Amb->L1) or index 1 (L1->Sub)?
    # Generator logic: index 0 is top (Amb/L1), index -1 is bottom (Ln/Sub)
    # Let's look at index 0 (Top Roughness) and index -1 (Substrate Roughness)
    r_true_l1 = r_true[:, 0] 
    r_pred_l1 = r_pred[:, 0]
    
    # SLD: Index 1 is Layer 1 (Index 0 is Amb=0, Index -1 is Sub)
    s_true_l1 = s_true[:, 1]
    s_pred_l1 = s_pred[:, 1]
    
    targets = np.stack([t_true_l1, r_true_l1, s_true_l1], axis=1)
    preds = np.stack([t_pred_l1, r_pred_l1, s_pred_l1], axis=1)
    
    param_names = ["Thickness (A)", "Roughness (A)", "SLD (10-6 A-2)"]
    
    # Calculate Metrics
    metrics = calculate_metrics(preds, targets)
    print_metrics_table(metrics, param_names)

    # DataFrame creation for full report
    df = pd.DataFrame()
    for i, name in enumerate(param_names):
        clean = name.split(' (')[0]
        df[f"{clean}_True"] = targets[:, i]
        df[f"{clean}_Pred"] = preds[:, i]
        df[f"{clean}_Error"] = metrics["errors"][:, i]

    # Save Reports
    print(f"📊 Saving reports to {exp_dir}...")
    save_results_csv(df, exp_dir / "evaluation_results.csv")
    
    worst_indices = save_worst_cases(df, metrics["abs_errors"], exp_dir / "worst_cases_log.csv")
    
    save_correlation_plot(df, param_names, exp_dir / "evaluation_correlation.png")
    
    plot_error_heatmap(df, save_path=exp_dir / "error_heatmap.png")
    
    save_advanced_reconstruction(
        preds, targets, config.q_grid.cpu().numpy(),
        r_measured, r_reconstructed,
        exp_dir / "reconstruction_analysis.png",
        worst_indices=worst_indices
    )
    
    print(f"✅ Evaluation Complete. Reports saved in: {exp_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XRR PINN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pt")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of samples for evaluation")
    
    args = parser.parse_args()
    evaluate(args)
