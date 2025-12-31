import torch
import torch.nn as nn
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from xrefine.models.model import XRR_PINN
from xrefine.data.generator import generate_synthetic_batch
from xrefine.config import CONFIG

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
    device = CONFIG['device']
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    # Create Save Directory (same as checkpoint or specified)
    # The user asked for "saved in experiment folder" which contains the checkpoint
    exp_dir = checkpoint_path.parent
    
    print(f"🔍 Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', None)
    if not config:
        print("⚠️ No config found in checkpoint, using default Config")
        config = CONFIG
    
    # Initialize model
    pinn = XRR_PINN(device=device).to(device)
    pinn.load_state_dict(checkpoint['model_state_dict'])
    pinn.eval()
    
    # Large Batch Generation
    BATCH_SIZE = args.batch_size
    print(f"🧪 Generating {BATCH_SIZE} synthetic samples for evaluation...")
    
    q_grid = config['q_grid'].to(device)
    qs_np = q_grid.cpu().numpy()
    
    # Generate data
    # generate_synthetic_batch returns (reflectivity, (thick, rough, sld))
    # params tuple has [Batch, Layers] shape
    r_measured_cuda, params_tuple_cuda = generate_synthetic_batch(BATCH_SIZE, q_grid, config)
    
    # Run Inference
    print("🧠 Running Inference...")
    with torch.no_grad():
        # returns (thick, rough, sld), r_reconstructed
        params_pred_tuple, r_reconstructed_cuda = pinn(torch.log10(r_measured_cuda + 1e-12))

    # Move to CPU for analysis
    r_measured = r_measured_cuda.cpu().numpy()
    r_reconstructed = r_reconstructed_cuda.cpu().numpy()
    
    # Unpack parameters
    # Tuple: (thickness, roughness, sld)
    # Thickness: [B, n_layers]
    # Roughness: [B, n_layers+1] ? Note: model.py output matches input structure?
    # generate_synthetic_batch returns roughness as [B, n_layers+1] ?
    # Let's check model.py:
    # Model returns: return (thick, rough, sld), r_final
    # XRRParameterEncoder returns: thick, rough, sld, s_len, i0_scale
    # Wait, model forward returns (thick, rough, sld) matching encoder output
    # Encoder output thick is [B, 1], rough is [B, 1], sld is [B, 1] (for n_layers=1)
    # Generator output (for n_layers=1):
    # thick: [B, 1], rough: [B, 2], sld: [B, 3] (Amb, L1, Sub)
    # WE HAVE A MISMATCH potentially if model predicts simplified parameters vs generator full parameters
    
    # Let's check model.py detail:
    # class XRRParameterEncoder(nn.Module):
    #   ...
    #   return thick, rough, sld, s_len, i0_scale
    #   thick, rough, sld are [B, 1]
    
    # class XRR_PINN_V3(nn.Module):
    #   def forward(self, r_obs):
    #       thick, rough, sld, s_len, i0 = self.encoder(r_obs)
    #       ...
    #       rough_full = torch.cat([rough, rough], dim=-1) # Duplicating roughness for 2 interfaces?
    #       sld_full = torch.cat([torch.zeros_like(sld), sld, torch.ones_like(sld)*20.1], dim=-1)
    #       return (thick, rough, sld), r_final
    
    # So the model predicts SINGLE layer properties (1 thick, 1 rough, 1 sld).
    # And constructs full stack assuming Amb=0, Sub=20.1 fixed?
    
    # Generator returns:
    # thickness [B, 1]
    # roughness [B, 2] (Two interfaces)
    # sld [B, 3] (Amb, Layer, Sub)
    
    # Comparison:
    # Thickness: Direct compare.
    # Roughness: Model predicts 1 value 'rough'. Generator has 2.
    #            If model uses `rough` for both interfaces (rough_full = cat([rough, rough])),
    #            we should compare model 'rough' to... average of true roughness? or first?
    # SLD: Model predicts 1 value 'sld'. Generator has 3 (Middle one is layer sld).
    #            We should compare model 'sld' to true 'layer sld'.
    
    t_true = params_tuple_cuda[0].cpu().numpy()[:, 0]
    # Generator provides full roughness [Amb/L1, L1/Sub]
    # We'll stick to first interface or average?
    # Let's use index 0 for comparison if that's what model learned or intended
    # But generator randomizes both. Model is forced to use same for both?
    # "rough_full = torch.cat([rough, rough], dim=-1)" -> Yes model assumes uniform roughness.
    # So we compare prediction to mean of true roughnesses?
    r_true = params_tuple_cuda[1].cpu().numpy().mean(axis=1) 
    
    s_true = params_tuple_cuda[2].cpu().numpy()[:, 1] # Index 1 is layer SLD
    
    t_pred = params_pred_tuple[0].cpu().numpy()[:, 0]
    r_pred = params_pred_tuple[1].cpu().numpy()[:, 0]
    s_pred = params_pred_tuple[2].cpu().numpy()[:, 0]
    
    # Stack for metrics [B, 3]
    targets = np.stack([t_true, r_true, s_true], axis=1)
    preds = np.stack([t_pred, r_pred, s_pred], axis=1)
    
    # Metrics
    metrics = calculate_metrics(preds, targets)
    param_names = ["Thickness (A)", "Roughness (A)", "SLD (10-6 A-2)"]
    
    print_metrics_table(metrics, param_names)

    # DataFrame creation
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
        preds, targets, qs_np,
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
