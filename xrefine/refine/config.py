"""
Refinement Configuration (Smart-Lock Configurable)
==================================================
Added 'relative_bounds' to StageConfig.
Now you can define dynamic constraints (Smart-Lock) directly in the stage definition.
"""
from dataclasses import dataclass, field
from typing import Set, Optional, List, Dict, Tuple
from xrefine.refine.losses import (
    LossFn, LOG_MSE, LINEAR_MSE, CORRELATION, FRINGE_WEIGHTED_LOG, HYBRID_LOSS
)

@dataclass
class StageConfig:
    name: str = "Unnamed Stage"
    optimize_keys: Set[str] = field(default_factory=set)
    fixed_params: dict = field(default_factory=dict)
    loss_fn: LossFn = LOG_MSE
    q_min: Optional[float] = None
    q_max: Optional[float] = None
    use_grid_search: bool = False
    grid_target_key: str = "Target_Film.thickness" 
    grid_range: tuple = (20.0, 1200.0)
    grid_points: int = 500
    
    # [Hard Bounds] 절대적인 범위 제한 (예: 거칠기 Min 2.0)
    temp_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # [NEW: Smart Bounds] 현재 값 기준 상대적 제한
    # Key: 파라미터 이름, Value: 허용 범위 (Ratio or Delta)
    # 예: {'Target_Film.sld': 0.05} -> 현재 값의 ±5%
    # 예: {'tth_offset': 0.01} -> 현재 값 ±0.01 (절대값)
    relative_bounds: Dict[str, float] = field(default_factory=dict)
    
    maxiter: int = 100
    popsize: int = 30
    description: str = ""

@dataclass 
class RefinementConfig:
    q_cutoff: float = 0.45
    trust_ai: bool = True  
    ai_trust_range: float = 0.4 
    verbose: bool = True
    stages: List[StageConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.stages:
            self.stages = self.default_stages()

    @staticmethod
    def default_stages() -> List[StageConfig]:
        """
        Improved refinement stages for better low-angle and fringe fitting.
        
        Key changes:
        1. Stage 0: Pre-critical footprint (beam_width, L, tth_offset)
        2. beam_width is now optimized, not fixed (Critical for low-angle fit!)
        3. Better separation of physical effects
        """
        
        footprint_params = {'beam_width', 'L', 'tth_offset'}
        sld_params = {'Target_Film.sld', 'Substrate.sld', 'Top_Layer.sld'}
        shape_params = {
            'Target_Film.roughness', 'Substrate.roughness', 'Top_Layer.roughness',
            'Target_Film.sld_imag', 'Substrate.sld_imag', 'dq'
        }
        all_params = footprint_params | sld_params | shape_params | {
            'Target_Film.thickness', 'Top_Layer.thickness', 'bkg'
        }

        return [
            # Stage 1: Pre-Critical Footprint (Q < 0.04)
            # Beam width 최적화가 핵심! (0.1 고정 해제)
            StageConfig(
                name="1. Pre-Critical Footprint",
                optimize_keys=footprint_params | {'bkg'},
                loss_fn=LINEAR_MSE,
                q_max=0.045,  
                maxiter=80, popsize=40,
                temp_bounds={
                    'L': (5.0, 100.0),
                    'beam_width': (0.02, 0.5),  # [KEY]
                    'tth_offset': (-0.1, 0.1),
                    'bkg': (-10.0, -4.0)
                }
            ),
            
            # Stage 2: Critical Angle & SLD
            StageConfig(
                name="2. Critical Angle & SLD",
                optimize_keys=sld_params | {'dq', 'Substrate.roughness'},
                loss_fn=LINEAR_MSE,
                q_min=0.03, q_max=0.09,
                maxiter=80, popsize=40,
                temp_bounds={
                    'dq': (0.001, 0.03),
                    'Substrate.roughness': (0.0, 15.0)
                }
            ),
            
            # Stage 3: Thickness
            StageConfig(
                name="3. Fix Thickness",
                use_grid_search=True,
                grid_target_key="Target_Film.thickness",
                loss_fn=CORRELATION,
                grid_points=300
            ),
            
            # Stage 4: Fringe Depth
            StageConfig(
                name="4. Fringe Depth",
                optimize_keys=shape_params | {'Top_Layer.thickness'},
                loss_fn=LOG_MSE,
                q_max=0.35,
                maxiter=100, popsize=40
            ),

            # Stage 5: Final Polish
            StageConfig(
                name="5. Final Polish",
                optimize_keys=all_params,
                loss_fn=HYBRID_LOSS,
                maxiter=150, popsize=50,
                temp_bounds={
                    'bkg': (-7.5, -4.0),
                    'Target_Film.roughness': (0.0, 40.0),
                    'beam_width': (0.02, 0.5),
                    'dq': (0.001, 0.03)
                },
                relative_bounds={
                    'Target_Film.sld': 0.10,
                    'Substrate.sld': 0.10,
                    'tth_offset': 0.02,
                    'L': 0.2,
                    'Target_Film.thickness': 0.1
                }
            ),
        ]

DEFAULT_CONFIG = RefinementConfig()