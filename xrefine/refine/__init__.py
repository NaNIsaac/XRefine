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
    "run_refinement",
    "run_real_inference",
    "RefinementConfig",
    "StageConfig",
    "DEFAULT_CONFIG",
    "optimize_step",
    "LossFn",
    "LOG_MSE",
    "LINEAR_MSE",
    "CORRELATION",
    "FRINGE_WEIGHTED_LOG",
    "FRINGE_WEIGHTED_LINEAR",
]
