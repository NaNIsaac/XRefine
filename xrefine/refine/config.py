"""
Refinement Configuration (Smart-Lock Configurable)
==================================================
Added 'relative_bounds' to StageConfig.
Now you can define dynamic constraints (Smart-Lock) directly in the stage definition.
"""

from dataclasses import dataclass, field
from typing import Optional

from xrefine.refine.losses import (
    CORRELATION,
    FRINGE_WEIGHTED_LOG,
    HYBRID_LOSS,
    LINEAR_MSE,
    LOG_MSE,
    LossFn,
)


@dataclass
class StageConfig:
    name: str = "Unnamed Stage"
    optimize_keys: set[str] = field(default_factory=set)
    fixed_params: dict = field(default_factory=dict)
    loss_fn: LossFn = LOG_MSE
    q_min: float | None = None
    q_max: float | None = None
    use_grid_search: bool = False
    grid_target_key: str = "Target_Film.thickness"
    grid_range: tuple = (20.0, 1200.0)
    grid_points: int = 500

    # hard bounds: absolute limits (e.g. roughness min 2.0)
    temp_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    # smart bounds: limits relative to the current value
    # key: parameter name, value: allowed range (ratio or delta)
    # {'Target_Film.sld': 0.05} -> +-5% of the current value
    # {'tth_offset': 0.01} -> +-0.01 around the current value (absolute)
    relative_bounds: dict[str, float] = field(default_factory=dict)

    maxiter: int = 100
    popsize: int = 30
    description: str = ""


@dataclass
class RefinementConfig:
    q_cutoff: float = 0.45
    trust_ai: bool = True
    ai_trust_range: float = 0.4
    verbose: bool = True
    stages: list[StageConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.stages:
            self.stages = self.default_stages()

    @staticmethod
    def default_stages() -> list[StageConfig]:
        """
        Improved refinement stages for better low-angle and fringe fitting.

        Key changes:
        1. Stage 0: Pre-critical footprint (beam_width, L, tth_offset)
        2. beam_width is now optimized, not fixed (Critical for low-angle fit!)
        3. Better separation of physical effects
        """

        footprint_params = {"beam_width", "L", "tth_offset"}
        sld_params = {"Target_Film.sld", "Substrate.sld", "Top_Layer.sld"}
        shape_params = {
            "Target_Film.roughness",
            "Substrate.roughness",
            "Top_Layer.roughness",
            "Target_Film.sld_imag",
            "Substrate.sld_imag",
            "dq",
        }
        all_params = (
            footprint_params
            | sld_params
            | shape_params
            | {"Target_Film.thickness", "Top_Layer.thickness", "bkg"}
        )

        # scale and footprint (low angle): linear loss to match the amplitude
        # main film (thickness, SLD): correlation/log loss to match the frequency
        # fine tuning (roughness etc.): log/hybrid loss for detail

        return [
            # Stage 1: Scale & Footprint (Absolute Amplitude)
            # Match I0, Beam Width, Background first.
            StageConfig(
                name="1. Scale & Footprint",
                optimize_keys={"i0", "beam_width", "bkg", "tth_offset"},
                loss_fn=LINEAR_MSE,  # Amplitude Matching
                q_max=0.06,  # Focus on critical angle & start of fringes
                maxiter=50,
                popsize=40,
                temp_bounds={"i0": (0.5, 3.0), "beam_width": (0.01, 0.5), "bkg": (-10.0, -4.0)},
            ),
            # Stage 2: Main Film Structure (Periodicity)
            # Fix Thickness and SLD to get correct fringe frequency
            StageConfig(
                name="2. Main Film Structure",
                optimize_keys={"Target_Film.thickness", "Target_Film.sld", "Substrate.sld", "dq"},
                loss_fn=CORRELATION,  # Phase/Frequency Matching (Invariant to Scale/Damping)
                q_min=0.03,
                q_max=0.35,
                maxiter=80,
                popsize=50,
                use_grid_search=True,  # Trust Grid for thickness
                grid_target_key="Target_Film.thickness",
                grid_points=300,
                temp_bounds={
                    "Target_Film.thickness": (10.0, 2000.0),
                    "Target_Film.sld": (0.0, 150.0),
                    "dq": (0.003, 0.03),
                },
            ),
            # Stage 3: Fine Tuning (Shape & Damping)
            # Now fit Roughness and detailed shape
            StageConfig(
                name="3. Fine Tuning",
                optimize_keys=all_params,
                loss_fn=LOG_MSE,  # Sensitive to damping (roughness)
                maxiter=150,
                popsize=50,
                temp_bounds={
                    "Top_Layer.thickness": (0.0, 100.0),
                    "Top_Layer.roughness": (0.0, 8.0),
                    "Substrate.roughness": (0.0, 8.0),
                    "Target_Film.roughness": (0.0, 20.0),
                    "Target_Film.sld": (0.0, 150.0),
                    "i0": (0.5, 3.0),
                    "dq": (0.003, 0.03),
                },
                relative_bounds={
                    "Target_Film.thickness": 0.1,  # Don't drift too far from Stage 2
                    "Target_Film.sld": 0.1,
                },
            ),
        ]


DEFAULT_CONFIG = RefinementConfig()
