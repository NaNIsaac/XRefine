"""
XRefine Refine Module
=====================
XRR 데이터 정밀 피팅(Refinement)을 위한 핵심 도구 모음.

Usage:
    from xrefine.refine import run_real_inference, RefinementConfig, LOG_MSE
"""

# 1. Pipeline (Main Entry Points)
from xrefine.refine.pipeline import (
    run_refinement, 
    run_real_inference
)

# 2. Configuration
from xrefine.refine.config import (
    RefinementConfig, 
    StageConfig, 
    DEFAULT_CONFIG
)

# 3. Core Logic
from xrefine.refine.optimizer import optimize_step

# 4. Loss Functions (Presets)
from xrefine.refine.losses import (
    LossFn,
    LOG_MSE, 
    LINEAR_MSE, 
    CORRELATION, 
    FRINGE_WEIGHTED_LOG,
)

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