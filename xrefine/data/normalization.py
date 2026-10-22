"""
Normalization Engine
====================
Central module for all data normalization logic.
Strategies are selected via config.data.normalization_mode.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from xrefine.config import XRefineConfig

from xrefine.logger import logger


def robust_plateau_normalize(
    y: np.ndarray, q: np.ndarray | None = None, q_limit: float = 0.08
) -> tuple[np.ndarray, float]:
    """Returns: (Normalized intensity, Scale factor)"""
    if q is not None:
        mask = q < q_limit
        target = y[mask] if mask.sum() > 3 else y[:20]
    else:
        target = y[:20]

    if len(target) == 0:
        logger.warning("Plateau normalization: empty target region, using max.")
        scale = y.max() + 1e-12
        return y / scale, scale

    sorted_vals = np.sort(target)[::-1]
    k = min(5, len(sorted_vals))
    scale = np.mean(sorted_vals[:k]) + 1e-12

    return y / scale, scale


def anchor_normalize(
    y: np.ndarray, q: np.ndarray | None, tth_array: np.ndarray | None, target_tth: float = 0.2
) -> tuple[np.ndarray, float]:
    """Returns: (Normalized intensity, Scale factor)"""
    if tth_array is None:
        logger.warning("Anchor normalization requires tth_array. Falling back to max.")
        scale = y.max() + 1e-12
        return y / scale, scale

    idx = (np.abs(tth_array - target_tth)).argmin()
    actual_tth = tth_array[idx]

    start, end = max(0, idx - 2), min(len(y), idx + 3)

    if end > start:
        scale = np.mean(y[start:end])
    else:
        scale = y[idx]

    if scale <= 0:
        logger.warning(f"Anchor normalization: scale={scale:.2e} at {target_tth}° is invalid.")
        scale = y.max()

    logger.info(
        f"  [Anchor Norm] tth={actual_tth:.3f}° (target {target_tth}°), "
        f"scale={scale:.4e}, result max={y.max() / scale:.4f}"
    )

    return y / (scale + 1e-12), scale


def apply_normalization(
    y_raw: np.ndarray,
    config: "XRefineConfig",
    q_array: np.ndarray | None = None,
    tth_array: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Returns: (Normalized intensity, Scale factor)"""
    mode = config.data.normalization_mode.lower()

    if mode == "plateau":
        return robust_plateau_normalize(y_raw, q_array, config.data.plateau_q_limit)

    elif mode == "max":
        scale = y_raw.max() + 1e-12
        return y_raw / scale, scale

    elif mode == "anchor":
        return anchor_normalize(y_raw, q_array, tth_array, config.data.anchor_tth)

    elif mode == "none":
        return y_raw, 1.0

    else:
        logger.warning(f"Unknown normalization mode '{mode}', using max.")
        scale = y_raw.max() + 1e-12
        return y_raw / scale, scale


# =============================================================================
# Tensor Versions (for Generator)
# =============================================================================


def apply_normalization_tensor(
    reflectivity: torch.Tensor, q_grid: torch.Tensor, config: "XRefineConfig"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns: (Normalized tensor, Scale tensor (B,1))"""
    mode = config.data.normalization_mode.lower()

    # Ensure 2D shape (B, Q)
    if reflectivity.dim() == 3:
        reflectivity = reflectivity.squeeze(1)

    if mode == "plateau":
        q_limit = config.data.plateau_q_limit
        low_q_mask = q_grid < q_limit
        low_q_idx = low_q_mask.sum().item()

        if low_q_idx > 3:
            target = reflectivity[:, :low_q_idx]
        else:
            target = reflectivity[:, :20]

        k = min(5, target.shape[-1])
        top_k = torch.topk(target, k=k, dim=-1)[0]
        scale = torch.mean(top_k, dim=-1, keepdim=True) + 1e-12

        return reflectivity / scale, scale

    elif mode == "max":
        scale = reflectivity.max(dim=-1, keepdim=True)[0] + 1e-12
        return reflectivity / scale, scale

    elif mode == "anchor":
        # Fallback to plateau for generator
        return apply_normalization_tensor(
            reflectivity,
            q_grid,
            type(
                "Config",
                (),
                {
                    "data": type(
                        "Data",
                        (),
                        {
                            "normalization_mode": "plateau",
                            "plateau_q_limit": config.data.plateau_q_limit,
                        },
                    )()
                },
            )(),
        )

    elif mode == "none":
        scale = torch.ones(reflectivity.shape[0], 1, device=reflectivity.device)
        return reflectivity, scale

    else:
        scale = reflectivity.max(dim=-1, keepdim=True)[0] + 1e-12
        return reflectivity / scale, scale
