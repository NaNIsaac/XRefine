"""
Normalization Engine
====================
Central module for all data normalization logic.
Strategies are selected via config.data.normalization_mode.
"""
import numpy as np
import torch
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from xrefine.config import XRefineConfig

from xrefine.logger import logger


def robust_plateau_normalize(
    y: np.ndarray, 
    q: Optional[np.ndarray] = None, 
    q_limit: float = 0.08
) -> np.ndarray:
    """
    Normalize by the top values in the plateau (total reflection) region.
    
    The plateau region is where Q < q_limit (typically < 0.08 Å⁻¹).
    Uses top-5 mean to be robust against noise spikes.
    
    Args:
        y: Raw intensity array
        q: Q-values array (Å⁻¹)
        q_limit: Upper Q limit for plateau region
        
    Returns:
        Normalized intensity array with plateau ≈ 1.0
    """
    if q is not None:
        mask = q < q_limit
        target = y[mask] if mask.sum() > 3 else y[:20]
    else:
        # Fallback: use first 20 points
        target = y[:20]
    
    if len(target) == 0:
        logger.warning("Plateau normalization: empty target region, using max.")
        return y / (y.max() + 1e-12)
    
    # Top-k mean for robustness
    sorted_vals = np.sort(target)[::-1]
    k = min(5, len(sorted_vals))
    scale = np.mean(sorted_vals[:k])
    
    return y / (scale + 1e-12)


def anchor_normalize(
    y: np.ndarray, 
    q: Optional[np.ndarray], 
    tth_array: Optional[np.ndarray], 
    target_tth: float = 0.2
) -> np.ndarray:
    """
    Normalize to make intensity at a specific 2theta angle equal to 1.0.
    
    Uses local averaging around anchor point for stability.
    
    Args:
        y: Raw intensity array
        q: Q-values array (unused, for API consistency)
        tth_array: 2theta values array (degrees)
        target_tth: Target 2theta angle for normalization anchor
        
    Returns:
        Normalized intensity array with I(target_tth) ≈ 1.0
    """
    if tth_array is None:
        logger.warning("Anchor normalization requires tth_array. Falling back to max.")
        return y / (y.max() + 1e-12)
    
    # Find closest index to target angle
    idx = (np.abs(tth_array - target_tth)).argmin()
    actual_tth = tth_array[idx]
    
    # Local averaging for stability (±2 points)
    start, end = max(0, idx - 2), min(len(y), idx + 3)
    
    if end > start:
        scale = np.mean(y[start:end])
    else:
        scale = y[idx]
    
    if scale <= 0:
        logger.warning(f"Anchor normalization: scale={scale:.2e} at {target_tth}° is invalid.")
        scale = y.max()
    
    logger.info(f"  [Anchor Norm] tth={actual_tth:.3f}° (target {target_tth}°), "
                f"scale={scale:.4e}, result max={y.max()/scale:.4f}")
    
    return y / (scale + 1e-12)


def apply_normalization(
    y_raw: np.ndarray, 
    config: "XRefineConfig", 
    q_array: Optional[np.ndarray] = None, 
    tth_array: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Central dispatcher for normalization strategies.
    
    Selects normalization method based on config.data.normalization_mode.
    
    Args:
        y_raw: Raw intensity array
        config: XRefineConfig with data.normalization_mode setting
        q_array: Q-values array (Å⁻¹), required for 'plateau' mode
        tth_array: 2theta values array (degrees), required for 'anchor' mode
        
    Returns:
        Normalized intensity array
    """
    mode = config.data.normalization_mode.lower()
    
    if mode == "plateau":
        return robust_plateau_normalize(y_raw, q_array, config.data.plateau_q_limit)
    
    elif mode == "max":
        return y_raw / (y_raw.max() + 1e-12)
    
    elif mode == "anchor":
        return anchor_normalize(y_raw, q_array, tth_array, config.data.anchor_tth)
    
    elif mode == "none":
        # No normalization - user must ensure I0 range is appropriate
        return y_raw
    
    else:
        logger.warning(f"Unknown normalization mode '{mode}', using max.")
        return y_raw / (y_raw.max() + 1e-12)


# =============================================================================
# Tensor Versions (for Generator)
# =============================================================================

def apply_normalization_tensor(
    reflectivity: torch.Tensor,
    q_grid: torch.Tensor,
    config: "XRefineConfig"
) -> torch.Tensor:
    """
    Apply normalization to batched tensor data (for training data generation).
    
    Args:
        reflectivity: Shape (B, Q) or (B, 1, Q) - batched reflectivity
        q_grid: Shape (Q,) - Q-values grid
        config: XRefineConfig with normalization settings
        
    Returns:
        Normalized reflectivity tensor
    """
    mode = config.data.normalization_mode.lower()
    
    # Ensure 2D shape (B, Q)
    if reflectivity.dim() == 3:
        reflectivity = reflectivity.squeeze(1)
    
    if mode == "plateau":
        q_limit = config.data.plateau_q_limit
        
        # Find plateau region
        low_q_mask = q_grid < q_limit
        low_q_idx = low_q_mask.sum().item()
        
        if low_q_idx > 3:
            target = reflectivity[:, :low_q_idx]
        else:
            target = reflectivity[:, :20]
        
        # Top-k mean for each batch
        k = min(5, target.shape[-1])
        top_k = torch.topk(target, k=k, dim=-1)[0]
        scale = torch.mean(top_k, dim=-1, keepdim=True)
        
        return reflectivity / (scale + 1e-12)
    
    elif mode == "max":
        scale = reflectivity.max(dim=-1, keepdim=True)[0]
        return reflectivity / (scale + 1e-12)
    
    elif mode == "anchor":
        # For anchor mode in generator, approximate with plateau
        # (exact angle mapping is complex for generated data)
        return apply_normalization_tensor(reflectivity, q_grid, 
                                          type('Config', (), {'data': type('Data', (), {
                                              'normalization_mode': 'plateau',
                                              'plateau_q_limit': config.data.plateau_q_limit
                                          })()})())
    
    elif mode == "none":
        return reflectivity
    
    else:
        logger.warning(f"Unknown normalization mode '{mode}' in tensor, using max.")
        scale = reflectivity.max(dim=-1, keepdim=True)[0]
        return reflectivity / (scale + 1e-12)
