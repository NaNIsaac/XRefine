"""
XRefine Evaluator
================
Performance analysis and visualization for trained models.
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

class XRREvaluator:
    """
    Evaluation suite for XRR PINN models.
    Supports dynamic layer configurations.
    """
    def __init__(self, checkpoint_path: str, batch_size: int = 1000):
        self.checkpoint_path = Path(checkpoint_path)
        self.exp_dir = self.checkpoint_path.parent
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.config: Optional[XRefineConfig] = None
        self.model = None
        self.results_df: Optional[pd.DataFrame] = None
        self.sample_metrics: Optional[pd.DataFrame] = None 
        self.metrics_summary: Dict[str, Dict] = {}
        self.history: Optional[Dict] = None

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
        
        self.model = get_model(self.config).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        
        self.history = ckpt.get('history', {})
        logger.info(f"Setup complete. Model: {self.config.model.model_type.upper()}")

    @torch.no_grad()
    def run_inference(self) -> Tuple[torch.Tensor, ParamSet, ParamSet, torch.Tensor]:
        """Perform batch inference."""
        q_grid = self.config.q_grid.to(self.device)
        augmentor = XRRAugmentor(self.config).to(self.device)
        
        split_size = 50 
        num_iters = self.batch_size // split_size
        if self.batch_size % split_size != 0: num_iters += 1
        
        r_m_list, r_r_list, p_t_list, p_p_list = [], [], [], []
        for _ in tqdm(range(num_iters), desc="Running Inference"):
            r_m, p_t = generate_synthetic_batch(split_size, q_grid, self.config, augmenter=augmentor)
            
            r_m = r_m.to(self.device)
            p_t.to(self.device)
            
            log_r_in = torch.log10(torch.clamp(r_m, min=1e-11))
            out = self.model(log_r_in)
            p_p, r_r = out[0], out[1]
            
            r_m_list.append(r_m.cpu())
            r_r_list.append(r_r.cpu())
            p_t_list.append(p_t.to(torch.device('cpu')))
            p_p_list.append(p_p.to(torch.device('cpu')))
            
        return torch.cat(r_m_list), ParamSet.cat(p_t_list), ParamSet.cat(p_p_list), torch.cat(r_r_list)

    def calculate_statistics(self, r_t: torch.Tensor, r_r: torch.Tensor, p_t: ParamSet, p_p: ParamSet):
        """Calculate statistical metrics for each parameter."""
        logger.info("📊 Calculating statistics...")
        
        log_t = torch.log10(torch.clamp(r_t, min=1e-11))
        log_r = torch.log10(torch.clamp(r_r, min=1e-11))
        mse_per_sample = torch.mean((log_t - log_r)**2, dim=1).numpy()
        
        actual_size = len(mse_per_sample)
        self.sample_metrics = pd.DataFrame({'idx': np.arange(actual_size), 'log_mse': mse_per_sample})
        
        all_res = []
        keys = list(p_t._params.keys())
        
        for key in keys:
            val_t = p_t[key].detach().cpu().numpy().flatten()
            val_p = p_p[key].detach().cpu().numpy().flatten()
            
            err = val_p - val_t
            
            self.metrics_summary[key] = {
                "MAE": np.mean(np.abs(err)),
                "MAPE": np.mean(np.abs(err) / (np.abs(val_t) + 1e-9)) * 100,
                "RMSE": np.sqrt(np.mean(err**2)),
                "R2": 1 - (np.sum(err**2) / (np.sum((val_t - np.mean(val_t))**2) + 1e-9))
            }
            all_res.append(pd.DataFrame({'Parameter': key, 'True': val_t, 'Pred': val_p, 'Error': err}))
            
        self.results_df = pd.concat(all_res)
        self.results_df.to_csv(self.exp_dir / "eval_metrics_full.csv", index=False)

    def _split_params(self) -> Tuple[List[str], List[str]]:
        """Separate Core Physics parameters from Instrument/Nuisance parameters."""
        keys = sorted(list(self.metrics_summary.keys()))
        core_keywords = ['thickness', 'roughness', 'sld']
        
        core_keys = [k for k in keys if any(c in k for c in core_keywords)]
        inst_keys = [k for k in keys if k not in core_keys]
        
        return core_keys, inst_keys

    def save_correlation_plots(self):
        """Save separate correlation plots for Core and Instrument parameters."""
        core_keys, inst_keys = self._split_params()
        
        for group_name, keys in [("Physics", core_keys), ("Instrument", inst_keys)]:
            if not keys: continue
            
            n_cols = 4
            n_rows = (len(keys) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axes = axes.flatten()
            
            for i, key in enumerate(keys):
                sub = self.results_df[self.results_df['Parameter'] == key]
                ax = axes[i]
                
                # Scatter
                ax.scatter(sub['True'], sub['Pred'], alpha=0.15, s=6, color='#2c3e50')
                
                # Identity Line
                mi, ma = min(sub['True'].min(), sub['Pred'].min()), max(sub['True'].max(), sub['Pred'].max())
                ax.plot([mi, ma], [mi, ma], 'r--', lw=1.5)
                
                r2 = self.metrics_summary[key]['R2']
                # Shorten title
                title_key = key.replace("thickness", "d").replace("roughness", "sig").replace("sld", "rho")
                ax.set_title(f"{title_key}\nR²: {r2:.3f}", fontsize=10)
                
            # Hide empty axes
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
                
            plt.tight_layout()
            plt.savefig(self.exp_dir / f"report_corr_{group_name.lower()}.png", dpi=150)
            plt.close()

    def save_visual_report(self, q: torch.Tensor, r_t: torch.Tensor, r_r: torch.Tensor):
        """Generate plots for ranked cases."""
        q_np = q.cpu().numpy()
        sns.set_theme(style="whitegrid")
        
        # Ranked Cases
        if self.sample_metrics is not None:
            sorted_idx = self.sample_metrics.sort_values('log_mse')['idx'].values
            n_samples = len(sorted_idx)
            target_idx = [sorted_idx[0], sorted_idx[n_samples//2], sorted_idx[int(n_samples*0.85)], sorted_idx[-1]]
            titles = ["Best Fit", "Median Fit", "Top 15% Error", "Worst Fit"]
            
            fig2 = plt.figure(figsize=(18, 11))
            gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.22)
            for i, (idx, title) in enumerate(zip(target_idx, titles)):
                s_gs = gridspec.GridSpecFromSubplotSpec(2, 1, gs[i], height_ratios=[3, 1], hspace=0.05)
                ax_t, ax_b = fig2.add_subplot(s_gs[0]), fig2.add_subplot(s_gs[1])
                yt, yr = r_t[idx].numpy(), r_r[idx].numpy()
                ax_t.semilogy(q_np, yt, 'ko', ms=2, alpha=0.3, label='Observed')
                ax_t.semilogy(q_np, yr, 'r-', lw=1.2, label='Predicted')
                ax_t.set_title(f"{title} | Log-MSE: {self.sample_metrics.iloc[idx]['log_mse']:.5f}")
                res = np.log10(np.clip(yr, 1e-12, None)) - np.log10(np.clip(yt, 1e-12, None))
                ax_b.plot(q_np, res, color='#2980b9', lw=1)
                ax_b.axhline(0, color='black', ls='--')
                ax_b.set_ylim(-0.5, 0.5)
                
            plt.savefig(self.exp_dir / "report_ranked_cases.png", dpi=200)
            plt.close()

    def save_training_curves(self):
        """Recovered: Plot training history."""
        if not self.history or 'train_loss' not in self.history:
            logger.warning("No training history found.")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss
        ax1 = axes[0]
        ax1.plot(self.history['train_loss'], label='Train')
        if 'val_loss' in self.history:
            ax1.plot(self.history['val_loss'], label='Val')
        ax1.set_yscale('log')
        ax1.set_title("Loss Curves")
        ax1.legend()
        
        # LR / Weight
        ax2 = axes[1]
        ax2_r = ax2.twinx()
        if 'lr' in self.history:
            ax2.plot(self.history['lr'], 'g-', label='LR')
        if 'sup_weight' in self.history:
            ax2_r.plot(self.history['sup_weight'], 'm--', label='Sup. Weight')
            
        ax2.set_title("Training Dynamics")
        ax2.set_xlabel("Epochs")
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / "report_learning_curve.png", dpi=150)
        plt.close()

    def report(self):
        """Print summary report to log."""
        header = f"{'{Parameter':<25} | {'MAE':<10} | {'MAPE(%)':<10} | {'R2 Score':<10}"
        logger.info("\n" + header + "\n" + "-" * 65)
        
        # Sort keys so Core params come first
        core_keys, inst_keys = self._split_params()
        
        for k in core_keys + inst_keys:
            v = self.metrics_summary[k]
            prefix = "[*] " if k in core_keys else "    "
            logger.info(f"{prefix}{k:<21} | {v['MAE']:<10.4f} | {v['MAPE']:<10.2f} | {v['R2']:<10.4f}")

    def evaluate_all(self):
        """Run full evaluation pipeline."""
        rm, pt, pp, rr = self.run_inference()
        self.calculate_statistics(rm, rr, pt, pp)
        self.report()
        self.save_correlation_plots() # Split into Physics / Instrument
        self.save_visual_report(self.config.q_grid, rm, rr)
        self.save_training_curves()   # Recovered
        
        logger.info("\n" + "="*60)
        logger.info(f"✅ Evaluation Complete!")
        logger.info(f"📂 Results saved in: {self.exp_dir.absolute()}")
        logger.info("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()
    evaluator = XRREvaluator(args.checkpoint, args.batch_size)
    evaluator.setup()
    evaluator.evaluate_all()
