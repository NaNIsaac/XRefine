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
        
        footprint_params = {'tth_offset', 'L', 'i0'} 
        plateau_params = {'Target_Film.sld', 'Substrate.sld', 'Top_Layer.sld'}
        shape_params = {
            'Target_Film.roughness', 'Substrate.roughness', 'Top_Layer.roughness',
            'Target_Film.sld_imag', 'Substrate.sld_imag', 'dq'
        }
        all_params = footprint_params | plateau_params | shape_params | {
            'Target_Film.thickness', 'Top_Layer.thickness', 'bkg', 'beam_width'
        }

        return [
            # Stage 1~4 동일 (라운드 니, 풋프린트 등 기존 성공 전략 유지)
            StageConfig(name="1. Scale & Footprint", optimize_keys=footprint_params | {'bkg'}, fixed_params={'beam_width': 0.1}, loss_fn=LINEAR_MSE, q_max=0.05, maxiter=60, popsize=40, temp_bounds={'L': (5.0, 50.0), 'i0': (0.5, 2.0)}),
            # Stage 2: Roundness (dq, sub_rough) 조기 투입
            StageConfig(name="2. Anchor SLD & Roundness", optimize_keys=plateau_params | {'dq', 'Substrate.roughness'}, fixed_params={'beam_width': 0.1}, loss_fn=LINEAR_MSE, q_max=0.09, maxiter=60, popsize=40, temp_bounds={'dq': (0.002, 0.03), 'Substrate.roughness': (0.0, 10.0)}),
            StageConfig(name="3. Fix Thickness", use_grid_search=True, grid_target_key="Target_Film.thickness", fixed_params={'beam_width': 0.1}, loss_fn=CORRELATION, grid_points=300),
            StageConfig(name="4. Shape (Roughness Only)", optimize_keys=shape_params, fixed_params={'beam_width': 0.1}, loss_fn=LOG_MSE, q_max=0.30, maxiter=100, popsize=40),

            # ---------------------------------------------------------
            # [Stage 5] Final Polish (Hybrid R1 + LogR2)
            # ---------------------------------------------------------
            # Linear(임계각 사수)와 Log(프린지 사수)가 20:1 비율로 교차 검증하며 최적화.
            StageConfig(
                name="5. Final Polish (Hybrid)",
                optimize_keys=all_params, 
                loss_fn=HYBRID_LOSS, # [NEW] Hybrid Loss 사용
                maxiter=150, popsize=50,
                
                # Absolute Bounds
                temp_bounds={
                    'bkg': (-6.5, -4.5), 
                    'Target_Film.roughness': (2.0, 40.0),
                    'beam_width': (0.09, 0.11),
                    'dq': (0.002, 0.03)
                },
                
                # Smart-Lock (Relative Bounds)
                relative_bounds={
                    'Target_Film.sld': 0.05,
                    'Substrate.sld': 0.05,
                    'tth_offset': 0.01,
                    'L': 0.1,
                    'Target_Film.thickness': 0.2
                }
            ),
        ]

DEFAULT_CONFIG = RefinementConfig()