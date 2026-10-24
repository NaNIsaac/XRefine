from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xrefine.config import SampleConfig, XRefineConfig
from xrefine.core.structures import ParamSet

plt.switch_backend("Agg")


def get_param_range(config: XRefineConfig, key: str) -> tuple[float, float]:
    """
    Dynamically resolve parameter range [min, max] from key string.
    Key examples: 'Film.thickness', 'Substrate.roughness', 'i0'
    """
    parts = key.split(".")

    # 1. Global/Instrument Params
    if len(parts) == 1:
        if hasattr(config.sample, key):
            return getattr(config.sample, key)
        # Fallback for old/unknown keys?
        return (0.0, 1.0)

    # 2. Layer/Substrate Params
    obj_name, attr = parts[0], parts[1]

    if obj_name == "Substrate":
        return getattr(config.sample.substrate, attr)

    # Check Layers
    for layer in config.sample.layers:
        if layer.name == obj_name:
            return getattr(layer, attr)

    return (0.0, 1.0)  # Fallback


class XRRWeightedLoss(nn.Module):
    def __init__(self, q_grid: torch.Tensor):
        super().__init__()
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

        # -----------------------------------------------------------
        # 1. Preprocessing
        # -----------------------------------------------------------
        mask = (log_r_target > -10.0).float()
        valid_points = mask.sum() + 1e-9

        # r_recon comes in Linear Scale from model.simulate
        log_r_recon = torch.log10(torch.clamp(r_recon, min=1e-11))

        r_target_linear = torch.pow(10.0, log_r_target)
        r_recon_linear = r_recon

        # -----------------------------------------------------------
        # 2. Linear Loss
        # -----------------------------------------------------------
        loss_linear = torch.sum(((r_recon_linear - r_target_linear) ** 2) * mask) / valid_points

        # -----------------------------------------------------------
        # 3. Log Gradient Loss
        # -----------------------------------------------------------
        diff_recon = torch.diff(log_r_recon, dim=-1)
        diff_target = torch.diff(log_r_target, dim=-1)
        diff_mask = mask[:, 1:]

        loss_grad = torch.sum(((diff_recon - diff_target) ** 2) * diff_mask) / (
            diff_mask.sum() + 1e-9
        )

        # -----------------------------------------------------------
        # 4. Base Log MSE
        # -----------------------------------------------------------
        loss_log_base = torch.sum(((log_r_recon - log_r_target) ** 2) * mask) / valid_points

        # -----------------------------------------------------------
        # 5. Parameter Supervision (Dynamic)
        # -----------------------------------------------------------
        loss_param = torch.tensor(0.0, device=log_r_target.device)

        # We assume p_true has all keys that p_pred has
        for key in p_pred._params.keys():
            # Skip non-learnable or fixed params if we track them?
            # DynamicParamSet usually only contains learnable stuff + sampled globals.

            p_min, p_max = get_param_range(config, key)

            val_pred = p_pred[key]
            val_true = p_true[key]

            # Normalize to 0-1 for balanced loss
            norm_pred = (val_pred - p_min) / (p_max - p_min + 1e-9)
            norm_true = (val_true - p_min) / (p_max - p_min + 1e-9)

            # Weighting Logic
            w_val = 1.0

            # Parse key: "LayerName.attr" or "GlobalParam"
            if "." in key:
                layer_name, attr = key.split(".", 1)

                # 1. Check Specific Override (e.g. Layer2.thickness)
                specific_map = config.train.weights.specific_layer_weights.get(layer_name, {})
                if attr in specific_map:
                    w_val = specific_map[attr]
                else:
                    # 2. Check Generic Layer Weight (e.g. thickness)
                    w_val = config.train.weights.layer_weights.get(attr, 1.0)
            else:
                # Look up global weight (e.g. "i0")
                w_val = config.train.weights.global_weights.get(key, 1.0)

            loss_param = loss_param + F.mse_loss(norm_pred, norm_true) * w_val

        # -----------------------------------------------------------
        # 6. Total Loss
        # -----------------------------------------------------------
        w_linear = getattr(config.train, "w_linear", 100.0)
        w_gradient = getattr(config.train, "w_gradient", 50.0)
        w_log_base = getattr(config.train, "w_log_base", 10.0)

        total_loss = (
            (w_linear * loss_linear)
            + (w_gradient * loss_grad)
            + (w_log_base * loss_log_base)
            + (curr_sup_w * loss_param)
            + (config.train.phys_penalty_w * phys_penalty)
        )

        return total_loss, loss_log_base, loss_grad, loss_linear, loss_param


def plot_learning_curves(history: dict[str, list[float]], save_path: str | Path):
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
    save_path: Path | str | None,
    step: int,
):
    """
    PINN Status Report (Dynamic)
    Returns: matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])

    # 1. Learning Curves
    ax0 = plt.subplot(gs[0])
    recent_loss = loss_history[-1000:] if len(loss_history) > 1000 else loss_history
    ax0.plot(recent_loss, color="#2c3e50", lw=1.5)
    ax0.set_yscale("log")
    ax0.set_title(f"Convergence (Recent {len(recent_loss)} Steps)")
    ax0.set_xlabel("Steps")
    ax0.set_ylabel("Total Loss")
    ax0.grid(True, alpha=0.2)

    # 2. XRR Profile
    ax1 = plt.subplot(gs[1])
    q_np = q_grid.cpu().numpy()

    y_target = target_log_r[0].detach().cpu().numpy()
    y_recon = torch.log10(torch.clamp(r_recon[0], min=1e-11)).detach().cpu().numpy()

    ax1.plot(q_np, y_target, "k-", lw=1, alpha=0.2, label="Target (Truth)")
    ax1.plot(q_np, y_recon, "r--", lw=1.5, label="PINN Prediction")

    ax1.set_ylabel("Log10(Reflectivity)")
    ax1.set_xlabel("q ($Å^{-1}$)")
    ax1.set_title(f"Validation Sample Fit at Epoch {step}")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # 3. Dynamic Parameter Text
    info_lines = [f"STEP: {step:04d}"]

    if params_true:
        info_lines.append("\n--- Parameters (Pred vs True) ---")
        # Sort keys to group by layer?
        keys = sorted(list(params_pred._params.keys()))

        # Limit display if too many params
        display_keys = keys[:15]  # Show top 15

        for k in display_keys:
            v_p = params_pred[k][0].item()
            v_t = params_true[k][0].item()

            # Formatting logic
            fmt = ".1f"
            if "dq" in k or "i0" in k:
                fmt = ".4f"

            err = abs(v_p - v_t) / (abs(v_t) + 1e-9) * 100

            # Shorten key name for display
            short_k = k.replace("thickness", "d").replace("roughness", "sig").replace("sld", "rho")

            info_lines.append(f"{short_k:<15}: {v_p:{fmt}} | {v_t:{fmt}} (Err: {err:4.1f}%)")

        if len(keys) > 15:
            info_lines.append(f"... and {len(keys) - 15} more")

    info_text = "\n".join(info_lines)

    plt.gcf().text(
        0.02,
        0.02,
        info_text,
        fontsize=9,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="#34495e", boxstyle="round,pad=0.5"),
        verticalalignment="bottom",
    )

    plt.tight_layout(rect=[0, 0.0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=120)
        # We don't close here if we return the figure,
        # BUT standard pattern is create->save->close.
        # Since we want to return fig for tensorboard, we leave it open.
        # Caller must close it.

    return fig
