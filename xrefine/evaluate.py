"""
XRefine Evaluator (v2.5 Professional Edition)
=============================================
Comprehensive performance analysis suite for XRR PINN models.
Updated for Strict Complex Mode & Explicit Simulator Injection.

Features:
1. Best/Worst Case Analysis
2. Learning Dynamics
3. Error Distribution vs Physical Values
4. Parameter Entanglement (Pearson Map)
5. Physical Consistency Checks
"""

from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

# [Imports Updated]
from xrefine.checkpoint import load_checkpoint
from xrefine.config import XRefineConfig
from xrefine.core.abeles import AbelesMatrix
from xrefine.core.smearing import abeles_constant_smearing
from xrefine.core.structures import ParamSet
from xrefine.data.augmentations import XRRAugmentor
from xrefine.data.generator import generate_synthetic_batch
from xrefine.logger import logger
from xrefine.models import get_model

# Matplotlib Style Setting
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("muted")


class XRREvaluator:
    def __init__(self, checkpoint_path: str, batch_size: int = 1000):
        self.checkpoint_path = Path(checkpoint_path)
        self.exp_dir = self.checkpoint_path.parent
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config: XRefineConfig | None = None
        self.model = None
        self.simulator = None  # [New]
        self.results_df: pd.DataFrame | None = None
        self.metrics_summary: dict[str, dict] = {}
        self.history: dict | None = None

        # Raw Data Storage
        self.q_grid = None
        self.r_true = None
        self.r_pred = None
        self.p_true = None
        self.p_pred = None

    def setup(self):
        """Load checkpoint, config, model, and simulator."""
        logger.info(f"🔍 Initializing Evaluator from {self.checkpoint_path}...")

        ckpt = load_checkpoint(self.checkpoint_path, map_location=self.device)

        config_data = ckpt.get("config")
        if isinstance(config_data, dict):
            self.config = XRefineConfig(**config_data)
        else:
            self.config = config_data

        # Device & Grid Setup
        self.config.q_grid = self.config.q_grid.to(self.device)
        self.q_grid = self.config.q_grid

        self.model = get_model(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.simulator = AbelesMatrix(device=self.device)

        self.history = ckpt.get("history", {})
        logger.info(f"Setup complete. Model: {self.config.model.model_type.upper()}")

    @torch.no_grad()
    def run_inference(self):
        """Perform batch inference and store results in memory."""
        augmentor = XRRAugmentor(self.config).to(self.device)

        split_size = 50
        num_iters = self.batch_size // split_size
        if self.batch_size % split_size != 0:
            num_iters += 1

        r_t_list, r_p_list, p_t_list, p_p_list = [], [], [], []

        logger.info(f"🚀 Running Inference on {self.batch_size} synthetic samples...")

        for _ in tqdm(range(num_iters), desc="Inference"):
            r_t_batch, p_t_batch = generate_synthetic_batch(
                batch_size=split_size,
                q_grid=self.q_grid,
                config=self.config,
                simulator=self.simulator,  # [Critical Fix]
                augmenter=augmentor,
            )

            # Data Move
            r_t_gpu = r_t_batch.to(self.device)
            # ParamSet tensors are already on GPU from generator, but strictly:
            # p_t_batch is a ParamSet, handled internally.

            # Preprocessing (Log & Clamp)
            # same preprocessing as training: clamp at 1e-15 preserves the masking value
            log_r_in = torch.log10(torch.clamp(r_t_gpu, min=1e-15))

            # Forward (No Fixed Values)
            out = self.model(log_r_in, fixed_values=None)

            # Model Output Handling (Tuple vs Tensor)
            if isinstance(out, tuple):
                p_p_batch, r_p_batch = out[0], out[1]
                # aux_output (out[2]) is ignored
            else:
                p_p_batch, r_p_batch = out  # Fallback for simple models

            # Collect (CPU Offloading to save GPU memory)
            r_t_list.append(r_t_batch.cpu())
            r_p_list.append(r_p_batch.detach().cpu())
            p_t_list.append(p_t_batch.to(torch.device("cpu")))
            p_p_list.append(p_p_batch.to(torch.device("cpu")))

        # Store globally
        self.r_true = torch.cat(r_t_list)
        self.r_pred = torch.cat(r_p_list)
        # ParamSet concatenation
        self.p_true = ParamSet.cat(p_t_list)
        self.p_pred = ParamSet.cat(p_p_list)

    def calculate_metrics(self):
        """Calculate statistical metrics and create DataFrame."""
        logger.info("📊 Calculating statistics...")

        log_t = torch.log10(torch.clamp(self.r_true, min=1e-15))
        log_p = torch.log10(torch.clamp(self.r_pred, min=1e-15))
        mse_per_sample = torch.mean((log_t - log_p) ** 2, dim=1).numpy()

        all_res = []
        # DynamicParamSet uses _params dict
        keys = list(self.p_true._params.keys())

        for key in keys:
            val_t = self.p_true[key].detach().cpu().numpy().flatten()
            val_p = self.p_pred[key].detach().cpu().numpy().flatten()

            err = val_p - val_t

            self.metrics_summary[key] = {
                "MAE": np.mean(np.abs(err)),
                "MAPE": np.mean(np.abs(err) / (np.abs(val_t) + 1e-9)) * 100,
                "RMSE": np.sqrt(np.mean(err**2)),
                "R2": 1 - (np.sum(err**2) / (np.sum((val_t - np.mean(val_t)) ** 2) + 1e-9)),
            }
            # DataFrame for analysis
            df_k = pd.DataFrame(
                {
                    "Sample_ID": np.arange(len(val_t)),
                    "Parameter": key,
                    "True": val_t,
                    "Pred": val_p,
                    "Error": err,
                    "AbsError": np.abs(err),
                    "LogMSE": mse_per_sample,
                }
            )
            all_res.append(df_k)

        self.results_df = pd.concat(all_res)
        self.results_df.to_csv(self.exp_dir / "eval_metrics_full.csv", index=False)

    def _split_params(self) -> tuple[list[str], list[str]]:
        keys = sorted(list(self.metrics_summary.keys()))
        core_keywords = ["thickness", "roughness", "sld"]  # 'sld_imag' also contains 'sld'

        core = [k for k in keys if any(c in k for c in core_keywords)]
        inst = [k for k in keys if k not in core]
        return core, inst

    # Visualizations

    def plot_best_worst_cases(self, n=3):
        """Plot the n best and n worst samples."""
        sample_mse = self.results_df.groupby("Sample_ID")["LogMSE"].first().sort_values()

        best_indices = sample_mse.index[:n].values
        worst_indices = sample_mse.index[-n:].values[::-1]  # Worst first

        indices = np.concatenate([best_indices, worst_indices])
        titles = [f"Best #{i + 1}" for i in range(n)] + [f"Worst #{i + 1}" for i in range(n)]

        q = self.q_grid.cpu().numpy()

        fig = plt.figure(figsize=(20, 4 * len(indices)))
        gs = gridspec.GridSpec(len(indices), 2, width_ratios=[2, 1])

        for i, (idx, title) in enumerate(zip(indices, titles, strict=False)):
            # A. Plot Curve
            ax_plot = plt.subplot(gs[i, 0])
            yt = self.r_true[idx].numpy()
            yp = self.r_pred[idx].numpy()
            mse = sample_mse[idx]

            ax_plot.semilogy(q, yt, "k-", alpha=0.3, lw=2, label="Measured")
            ax_plot.semilogy(q, yp, "r--", alpha=0.8, lw=1.5, label="Predicted")
            ax_plot.set_title(f"{title} (LogMSE: {mse:.4f})", fontsize=14, fontweight="bold")
            ax_plot.legend(loc="upper right")
            ax_plot.grid(True, alpha=0.3)

            # B. Parameter Comparison Table
            ax_tbl = plt.subplot(gs[i, 1])
            ax_tbl.axis("off")

            sample_df = self.results_df[self.results_df["Sample_ID"] == idx]

            # Clean Table Logic
            text_data = [["Parameter", "True", "Pred", "Diff", "Err%"]]
            colors = []

            # Sort by Parameter name for consistency
            sample_df = sample_df.sort_values("Parameter")

            for _, row in sample_df.iterrows():
                # Shorten Names
                p_name = row["Parameter"].replace("thickness", "d").replace("roughness", "sig")
                p_name = p_name.replace("sld_imag", "iRho").replace("sld", "Rho")

                diff = row["Pred"] - row["True"]
                err_p = abs(diff) / (abs(row["True"]) + 1e-9) * 100

                text_data.append(
                    [
                        p_name,
                        f"{row['True']:.2f}",
                        f"{row['Pred']:.2f}",
                        f"{diff:+.2f}",
                        f"{err_p:.1f}%",
                    ]
                )

            table = ax_tbl.table(
                cellText=text_data,
                loc="center",
                cellLoc="left",
                colWidths=[0.35, 0.15, 0.15, 0.15, 0.15],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

            # Color Coding
            for (r, c), cell in table.get_celld().items():
                if r == 0:
                    cell.set_text_props(fontweight="bold")
                    cell.set_facecolor("#eeeeee")
                elif r > 0 and c == 4:
                    err_val = float(text_data[r][4].strip("%"))
                    if err_val > 20:
                        cell.set_facecolor("#ffaaaa")
                    elif err_val < 5:
                        cell.set_facecolor("#aaffaa")

        plt.tight_layout()
        plt.savefig(self.exp_dir / "eval_1_best_worst.png", dpi=100)
        plt.close()

    def plot_training_curves(self):
        """Plot training curves (loss and weights)."""
        if not self.history:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss
        ax = axes[0]
        ax.plot(self.history.get("train_loss", []), label="Train")
        ax.plot(self.history.get("val_loss", []), label="Val")
        ax.set_yscale("log")
        ax.set_title("Loss History")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning Rate
        ax = axes[1]
        ax.plot(self.history.get("lr", []), color="g")
        ax.set_title("Learning Rate")
        ax.grid(True, alpha=0.3)

        # Supervision Weight
        ax = axes[2]
        ax.plot(self.history.get("sup_weight", []), color="orange")
        ax.set_title("Supervision Weight")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.exp_dir / "eval_2_learning_curves.png", dpi=100)
        plt.close()

    def plot_error_vs_value(self):
        """Scatter plot of error vs value to show physical bias."""
        core, _ = self._split_params()
        if not core:
            return

        n_cols = 3
        n_rows = (len(core) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, key in enumerate(core):
            sub = self.results_df[self.results_df["Parameter"] == key]
            ax = axes[i]

            sc = ax.scatter(
                sub["True"], sub["AbsError"], alpha=0.2, c=sub["LogMSE"], cmap="viridis"
            )
            sns.regplot(
                x="True",
                y="AbsError",
                data=sub,
                ax=ax,
                scatter=False,
                color="red",
                line_kws={"alpha": 0.5},
            )

            ax.set_title(f"{key}\nErr vs Value")
            ax.set_xlabel("True Value")
            ax.set_ylabel("Absolute Error")
            ax.grid(True, alpha=0.3)

        # Hide empty plots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(self.exp_dir / "eval_3_error_vs_value.png", dpi=100)
        plt.close()

    def plot_pearson_map(self):
        """[4] Parameter Entanglement (Pearson Correlation of Errors)"""
        pivot_df = self.results_df.pivot(index="Sample_ID", columns="Parameter", values="Error")
        corr = pivot_df.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, vmin=-1, vmax=1)
        plt.title("Parameter Error Correlation (Entanglement Map)", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.exp_dir / "eval_4_pearson_map.png", dpi=100)
        plt.close()

    def save_correlation_plots(self):
        """[Recovered] Save separate correlation plots for Core and Instrument parameters."""
        core_keys, inst_keys = self._split_params()

        for group_name, keys in [("Physics", core_keys), ("Instrument", inst_keys)]:
            if not keys:
                continue

            n_cols = 4
            n_rows = (len(keys) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axes = axes.flatten()

            for i, key in enumerate(keys):
                sub = self.results_df[self.results_df["Parameter"] == key]
                ax = axes[i]

                # Scatter
                ax.scatter(sub["True"], sub["Pred"], alpha=0.15, s=6, color="#2c3e50")

                # Identity Line
                mi, ma = (
                    min(sub["True"].min(), sub["Pred"].min()),
                    max(sub["True"].max(), sub["Pred"].max()),
                )
                ax.plot([mi, ma], [mi, ma], "r--", lw=1.5)

                r2 = self.metrics_summary[key]["R2"]
                # Shorten title
                title_key = (
                    key.replace("thickness", "d")
                    .replace("roughness", "sig")
                    .replace("sld_imag", "iRho")
                    .replace("sld", "Rho")
                )
                ax.set_title(f"{title_key}\nR²: {r2:.3f}", fontsize=10)
                ax.grid(True, alpha=0.3)

            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            plt.savefig(self.exp_dir / f"report_corr_{group_name.lower()}.png", dpi=150)
            plt.close()

    def report(self):
        """Print summary report to log."""
        header = f"{'Parameter':<25} | {'MAE':<10} | {'MAPE(%)':<10} | {'R2 Score':<10}"
        logger.info("\n" + header + "\n" + "-" * 65)

        core_keys, inst_keys = self._split_params()

        for k in core_keys + inst_keys:
            v = self.metrics_summary[k]
            prefix = "[*] " if k in core_keys else "    "
            logger.info(
                f"{prefix}{k:<21} | {v['MAE']:<10.4f} | {v['MAPE']:<10.2f} | {v['R2']:<10.4f}"
            )

    def report_physical_consistency(self):
        """Report physical-consistency checks."""
        layer_names = set([k.split(".")[0] for k in self.metrics_summary.keys() if "." in k])

        lines = ["\n[Physical Consistency Check (Roughness < 0.5 * Thickness)]"]
        lines.append(f"{'Layer':<15} | {'Violation Ratio':<15} | {'Avg Violation (Å)'}")
        lines.append("-" * 50)

        pivot_pred = self.results_df.pivot(index="Sample_ID", columns="Parameter", values="Pred")

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

        self.report()

        logger.info("🎨 Generating Visualization Reports...")
        self.plot_best_worst_cases()
        self.plot_training_curves()
        self.plot_error_vs_value()
        self.plot_pearson_map()
        self.save_correlation_plots()
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
