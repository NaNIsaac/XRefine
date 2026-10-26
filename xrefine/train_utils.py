from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib import gridspec
from torch import nn

from xrefine.config import XRefineConfig
from xrefine.core.structures import ParamSet

plt.switch_backend("Agg")


class XRRWeightedLoss(nn.Module):
    def __init__(self, q_grid: torch.Tensor):
        super().__init__()
        # equal weight (1.0) in all Q regions: the log scale already flattens the dynamic range, so
        # uniform weights are physically reasonable
        weights = torch.full_like(q_grid, 1.0)

        self.register_buffer("q_weights", weights.unsqueeze(0))
        self.register_buffer("q_grid", q_grid.unsqueeze(0))

    def forward(
        self,
        p_pred: ParamSet,
        r_recon: torch.Tensor,
        log_r_target: torch.Tensor,
        p_true: ParamSet,
        config: XRefineConfig,
        curr_sup_w: float,
        phys_penalty: torch.Tensor,
    ):

        mask = (log_r_target > -10.0).float()
        valid_points = mask.sum() + 1e-9
        log_r_recon = torch.log10(torch.clamp(r_recon, min=1e-11))

        # emphasize the critical-angle region (Q < 0.08), which determines the SLD (rho)
        qc_mask = (self.q_grid < 0.08).float() * mask
        loss_qc = torch.sum(qc_mask * (log_r_recon - log_r_target) ** 2) / (qc_mask.sum() + 1e-9)

        # the fringe spacing (diff) determines the thickness d; with uniform weights the high-angle
        # fringe derivatives count fully
        diff_recon = torch.diff(log_r_recon, dim=-1)
        diff_target = torch.diff(log_r_target, dim=-1)
        diff_mask = mask[:, 1:]
        loss_fringe = torch.sum(diff_mask * (diff_recon - diff_target) ** 2) / (
            diff_mask.sum() + 1e-9
        )

        base_error = (log_r_recon - log_r_target) ** 2
        loss_recon_base = (base_error * self.q_weights * mask).sum() / valid_points

        # supervised parameters; d_f, rho_f, sig_f are top priority
        loss_param = 0
        w_set = config.get_weight_set(device=log_r_target.device)
        for fld in fields(ParamSet):
            key = fld.name
            p_min, p_max = getattr(config.ranges, key)
            # normalized to 0..1 for the MSE
            norm_pred = (getattr(p_pred, key) - p_min) / (p_max - p_min + 1e-9)
            norm_true = (getattr(p_true, key) - p_min) / (p_max - p_min + 1e-9)
            loss_param += F.mse_loss(norm_pred, norm_true) * getattr(w_set, key)

        # priority weights: Fringe (30) > Qc (15) > Base (1)
        total_loss = (
            (loss_recon_base + 15.0 * loss_qc + 30.0 * loss_fringe)
            + (curr_sup_w * loss_param)
            + (config.train.phys_penalty_w * phys_penalty)
        )

        return total_loss, loss_recon_base, loss_fringe, loss_qc, loss_param


def plot_learning_curves(history: dict[str, list[float]], save_path: str | Path):
    """Plot the loss history over the whole training run."""
    if "train_loss" not in history or not history["train_loss"]:
        return

    plt.figure(figsize=(10, 6))
    for label, values in history.items():
        if values:
            plt.plot(values, label=label.replace("_", " ").title(), alpha=0.8)

    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title("XRefine Global Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_pinn_report(
    q_grid: torch.Tensor,
    target_log_r: torch.Tensor,
    r_recon: torch.Tensor,
    loss_history: list[float],
    params_pred: ParamSet,
    params_true: ParamSet | None,
    save_path: Path | str,
    step: int,
):
    """Plot a combined PINN status report."""
    plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[3, 1])

    ax0 = plt.subplot(gs[:, 0])
    recent_loss = loss_history[-500:]
    ax0.plot(recent_loss, color="#2c3e50", lw=1.5)
    ax0.set_yscale("log")
    ax0.set_title("Convergence (Recent 500 Steps)")
    ax0.set_xlabel("Steps")
    ax0.grid(True, alpha=0.2)

    ax1 = plt.subplot(gs[0, 1])
    q_np = q_grid.cpu().numpy()
    # first sample of the batch
    y_target = target_log_r[0].detach().cpu().numpy()
    y_recon = torch.log10(torch.clamp(r_recon[0], min=1e-11)).detach().cpu().numpy()

    ax1.plot(q_np, y_recon, "r-", lw=2, label="PINN Physics Fit", zorder=3)
    ax1.plot(q_np, y_target, "ko", markersize=2, alpha=0.15, label="Target Data", zorder=2)
    ax1.set_ylabel("Log10(Reflectivity)")
    ax1.set_title(f"Reflectivity Match at Step {step}")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(gs[1, 1])
    residuals = y_recon - y_target
    ax2.fill_between(q_np, residuals, 0, color="#3498db", alpha=0.3)
    ax2.plot(q_np, residuals, color="#3498db", lw=1)
    ax2.axhline(0, color="black", lw=1, ls="--")
    ax2.set_ylabel("$\\Delta$ LogR")
    ax2.set_xlabel("q ($Å^{-1}$)")
    ax2.set_ylim(-0.5, 0.5)

    info_text = f"STEP: {step:06d}\n\nPRED: {params_pred}\n\nTRUE: {params_true}"
    plt.gcf().text(
        0.02,
        0.02,
        info_text,
        fontsize=8.5,
        family="monospace",
        bbox={
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "#34495e",
            "boxstyle": "round,pad=1",
        },
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close("all")
