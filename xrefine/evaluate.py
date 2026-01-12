"""
XRefine Evaluator (Professional Edition)
========================================
Comprehensive performance analysis suite for XRR PINN models.
Features:
1. Best/Worst Case Analysis
2. Learning Dynamics
3. Error Distribution vs Physical Values
4. Parameter Entanglement (Pearson Map)
5. Physical Consistency Checks
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm

from xrefine.models import get_model
from xrefine.data.generator import generate_synthetic_batch
from xrefine.data.augmentations import XRRAugmentor
from xrefine.config import XRefineConfig
from xrefine.logger import logger
from xrefine.core.structures import ParamSet

# Matplotlib Style Setting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

class XRREvaluator:
    def __init__(self, checkpoint_path: str, batch_size: int = 1000):
        self.checkpoint_path = Path(checkpoint_path)
        self.exp_dir = self.checkpoint_path.parent
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.config: Optional[XRefineConfig] = None
        self.model = None
        self.results_df: Optional[pd.DataFrame] = None
        self.metrics_summary: Dict[str, Dict] = {}
        self.history: Optional[Dict] = None
        
        # Raw Data Storage
        self.q_grid = None
        self.r_true = None
        self.r_pred = None
        self.p_true = None
        self.p_pred = None

    def setup(self):
        """Load checkpoint and initialize model."""
        logger.info(f"🔍 Initializing Evaluator from {self.checkpoint_path}...")
        
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        config_data = ckpt.get('config')
        if isinstance(config_data, dict):
            self.config = XRefineConfig(**config_data)
        else:
            self.config = config_data
            
        self.config.q_grid = self.config.q_grid.to(self.device)
        self.q_grid = self.config.q_grid
        
        self.model = get_model(self.config).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        
        self.history = ckpt.get('history', {})
        logger.info(f"Setup complete. Model: {self.config.model.model_type.upper()}")

    @torch.no_grad()
    def run_inference(self):
        """Perform batch inference and store results in memory."""
        q_grid = self.config.q_grid.to(self.device)
        augmentor = XRRAugmentor(self.config).to(self.device)
        
        split_size = 50 
        num_iters = self.batch_size // split_size
        if self.batch_size % split_size != 0: num_iters += 1
        
        r_t_list, r_p_list, p_t_list, p_p_list = [], [], [], []
        
        for _ in tqdm(range(num_iters), desc="Running Inference"):
            # Generate Data
            r_t_batch, p_t_batch = generate_synthetic_batch(split_size, q_grid, self.config, augmenter=augmentor)
            
            # Forward
            r_t_gpu = r_t_batch.to(self.device)
            p_t_batch.to(self.device) # Move to GPU
            
            log_r_in = torch.log10(torch.clamp(r_t_gpu, min=1e-11))
            out = self.model(log_r_in)
            p_p_batch, r_p_batch = out[0], out[1]
            
            # Collect
            r_t_list.append(r_t_batch.cpu())
            r_p_list.append(r_p_batch.detach().cpu())
            p_t_list.append(p_t_batch.to(torch.device('cpu')))
            p_p_list.append(p_p_batch.to(torch.device('cpu')))
            
        # Store globally
        self.r_true = torch.cat(r_t_list)
        self.r_pred = torch.cat(r_p_list)
        self.p_true = ParamSet.cat(p_t_list)
        self.p_pred = ParamSet.cat(p_p_list)

    def calculate_metrics(self):
        """Calculate statistical metrics and create DataFrame."""
        logger.info("📊 Calculating statistics...")
        
        # 1. Curve Metrics (Log MSE)
        log_t = torch.log10(torch.clamp(self.r_true, min=1e-11))
        log_p = torch.log10(torch.clamp(self.r_pred, min=1e-11))
        mse_per_sample = torch.mean((log_t - log_p)**2, dim=1).numpy()
        
        # 2. Parameter Metrics
        all_res = []
        keys = list(self.p_true._params.keys())
        
        for key in keys:
            val_t = self.p_true[key].detach().cpu().numpy().flatten()
            val_p = self.p_pred[key].detach().cpu().numpy().flatten()
            
            err = val_p - val_t
            
            self.metrics_summary[key] = {
                "MAE": np.mean(np.abs(err)),
                "MAPE": np.mean(np.abs(err) / (np.abs(val_t) + 1e-9)) * 100,
                "RMSE": np.sqrt(np.mean(err**2)),
                "R2": 1 - (np.sum(err**2) / (np.sum((val_t - np.mean(val_t))**2) + 1e-9))
            }
            # DataFrame for analysis
            df_k = pd.DataFrame({
                'Sample_ID': np.arange(len(val_t)),
                'Parameter': key,
                'True': val_t,
                'Pred': val_p,
                'Error': err,
                'AbsError': np.abs(err),
                'LogMSE': mse_per_sample
            })
            all_res.append(df_k)
            
        self.results_df = pd.concat(all_res)
        self.results_df.to_csv(self.exp_dir / "eval_metrics_full.csv", index=False)

    def _split_params(self) -> Tuple[List[str], List[str]]:
        keys = sorted(list(self.metrics_summary.keys()))
        core_keywords = ['thickness', 'roughness', 'sld']
        core = [k for k in keys if any(c in k for c in core_keywords)]
        inst = [k for k in keys if k not in core]
        return core, inst

    # =========================================================================
    # Visualizations
    # =========================================================================

    def plot_best_worst_cases(self, n=3):
        """[1] 상위/하위 n개 샘플 비교 시각화"""
        # Group by Sample ID to get average LogMSE (already same per sample, but safe to grab first)
        sample_mse = self.results_df.groupby('Sample_ID')['LogMSE'].first().sort_values()
        
        best_indices = sample_mse.index[:n].values
        worst_indices = sample_mse.index[-n:].values[::-1] # Worst first
        
        indices = np.concatenate([best_indices, worst_indices])
        titles = [f"Best #{i+1}" for i in range(n)] + [f"Worst #{i+1}" for i in range(n)]
        
        q = self.q_grid.cpu().numpy()
        
        fig = plt.figure(figsize=(20, 4 * len(indices)))
        gs = gridspec.GridSpec(len(indices), 2, width_ratios=[2, 1])
        
        for i, (idx, title) in enumerate(zip(indices, titles)):
            # A. Plot Curve
            ax_plot = plt.subplot(gs[i, 0])
            yt = self.r_true[idx].numpy()
            yp = self.r_pred[idx].numpy()
            mse = sample_mse[idx]
            
            ax_plot.semilogy(q, yt, 'k-', alpha=0.3, lw=2, label='Measured')
            ax_plot.semilogy(q, yp, 'r--', alpha=0.8, lw=1.5, label='Predicted')
            ax_plot.set_title(f"{title} (LogMSE: {mse:.4f})", fontsize=14, fontweight='bold')
            ax_plot.legend(loc='upper right')
            ax_plot.grid(True, alpha=0.3)
            
            # B. Parameter Comparison Table
            ax_tbl = plt.subplot(gs[i, 1])
            ax_tbl.axis('off')
            
            # Get params for this sample
            sample_df = self.results_df[self.results_df['Sample_ID'] == idx]
            
            # Format text
            text_data = [["Parameter", "True", "Pred", "Diff", "Err%"]]
            colors = []
            
            for _, row in sample_df.iterrows():
                p_name = row['Parameter'].replace("thickness", "d").replace("roughness", "sig")
                diff = row['Pred'] - row['True']
                err_p = abs(diff) / (abs(row['True']) + 1e-9) * 100
                
                # Highlight critical errors (e.g. > 10%)
                row_color = "#ffcccc" if err_p > 10 else "#ccffcc"
                colors.append(row_color)
                
                text_data.append([
                    p_name, 
                    f"{row['True']:.2f}", 
                    f"{row['Pred']:.2f}", 
                    f"{diff:+.2f}", 
                    f"{err_p:.1f}%"
                ])
            
            table = ax_tbl.table(cellText=text_data, loc='center', cellLoc='left', colWidths=[0.35, 0.15, 0.15, 0.15, 0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.8)
            
            # Header Bold
            for (r, c), cell in table.get_celld().items():
                if r == 0:
                    cell.set_text_props(fontweight='bold')
                    cell.set_facecolor("#eeeeee")
                elif r > 0 and c == 4: # Err% Column
                    err_val = float(text_data[r][4].strip('%'))
                    if err_val > 20: cell.set_facecolor("#ffaaaa") # Red for bad
                    elif err_val < 5: cell.set_facecolor("#aaffaa") # Green for good
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / "eval_1_best_worst.png", dpi=100)
        plt.close()

    def plot_training_curves(self):
        """[2] 학습 곡선 (Loss & Weights)"""
        if not self.history: return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        ax = axes[0]
        ax.plot(self.history.get('train_loss', []), label='Train')
        ax.plot(self.history.get('val_loss', []), label='Val')
        ax.set_yscale('log')
        ax.set_title("Loss History")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning Rate
        ax = axes[1]
        ax.plot(self.history.get('lr', []), color='g')
        ax.set_title("Learning Rate")
        ax.grid(True, alpha=0.3)
        
        # Supervision Weight (Important!)
        ax = axes[2]
        ax.plot(self.history.get('sup_weight', []), color='orange')
        ax.set_title("Supervision Weight (Curriculum)")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / "eval_2_learning_curves.png", dpi=100)
        plt.close()

    def plot_error_vs_value(self):
        """[3] Error vs Value 산점도 (물리적 편향 확인)"""
        core, _ = self._split_params()
        if not core: return
        
        n_cols = 3
        n_rows = (len(core) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        
        for i, key in enumerate(core):
            sub = self.results_df[self.results_df['Parameter'] == key]
            ax = axes[i]
            
            # X: True Value, Y: Abs Error
            sc = ax.scatter(sub['True'], sub['AbsError'], alpha=0.2, c=sub['LogMSE'], cmap='viridis')
            
            # Trend line
            sns.regplot(x='True', y='AbsError', data=sub, ax=ax, scatter=False, color='red', line_kws={'alpha':0.5})
            
            ax.set_title(f"{key}\nErr vs Value")
            ax.set_xlabel("True Value")
            ax.set_ylabel("Absolute Error")
            ax.grid(True, alpha=0.3)
            
        plt.colorbar(sc, ax=axes, label='Reconstruction Error (LogMSE)')
        plt.savefig(self.exp_dir / "eval_3_error_vs_value.png", dpi=100)
        plt.close()

    def plot_pearson_map(self):
        """[4] Parameter Entanglement (Pearson Correlation of Errors)"""
        # Pivot: Rows=Samples, Cols=Parameter Errors
        pivot_df = self.results_df.pivot(index='Sample_ID', columns='Parameter', values='Error')
        
        corr = pivot_df.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        plt.title("Parameter Error Correlation (Entanglement Map)\nRed: Params move together (Confounded)", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.exp_dir / "eval_4_pearson_map.png", dpi=100)
        plt.close()

    def report_physical_consistency(self):
        """[5] 물리적 일관성 체크 리포트"""
        # Condition: Roughness < 0.5 * Thickness
        # We need to pair roughness and thickness for the same layer
        
        layer_names = set([k.split('.')[0] for k in self.metrics_summary.keys() if '.' in k])
        
        lines = ["\n[Physical Consistency Check]"]
        lines.append(f"{'{Layer':<15} | {'Violation Ratio':<15} | {'Avg Violation (Å)'}")
        lines.append("-" * 50)
        
        pivot_pred = self.results_df.pivot(index='Sample_ID', columns='Parameter', values='Pred')
        
        for layer in layer_names:
            t_key = f"{layer}.thickness"
            r_key = f"{layer}.roughness"
            
            if t_key in pivot_pred and r_key in pivot_pred:
                t_vals = pivot_pred[t_key]
                r_vals = pivot_pred[r_key]
                
                # Check: r < 0.5 * t
                violation = r_vals > (0.5 * t_vals)
                ratio = violation.mean() * 100
                magnitude = (r_vals - 0.5 * t_vals)[violation].mean() if violation.any() else 0.0
                
                lines.append(f"{layer:<15} | {ratio:>13.1f}% | {magnitude:.2f}")
                
        logger.info("\n".join(lines))

    def evaluate_all(self):
        self.run_inference()
        self.calculate_metrics()
        
        logger.info("🎨 Generating Visualization Reports...")
        self.plot_best_worst_cases()
        self.plot_training_curves()
        self.plot_error_vs_value()
        self.plot_pearson_map()
        self.report_physical_consistency()
        
        logger.info(f"\n✅ Evaluation Complete! Reports saved in: {self.exp_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()
    evaluator = XRREvaluator(args.checkpoint, args.batch_size)
    evaluator.setup()
    evaluator.evaluate_all()