"""XRefine Refine Module
=====================
Core tools for fine refinement (fitting) of XRR data.

Usage:
    from xrefine.refine import run_real_inference, RefinementConfig, LOG_MSE
"""

from xrefine.refine.config import DEFAULT_CONFIG, RefinementConfig, StageConfig
from xrefine.refine.losses import (
    CORRELATION,
    FRINGE_WEIGHTED_LOG,
    LINEAR_MSE,
    LOG_MSE,
    LossFn,
)
from xrefine.refine.optimizer import optimize_step
from xrefine.refine.pipeline import run_real_inference, run_refinement

__all__ = [
    "CORRELATION",
    "DEFAULT_CONFIG",
    "FRINGE_WEIGHTED_LOG",
    "LINEAR_MSE",
    "LOG_MSE",
    "LossFn",
    "RefinementConfig",
    "StageConfig",
    "optimize_step",
    "run_real_inference",
    "run_refinement",
]
