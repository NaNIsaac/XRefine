"""
Refinement Configuration
========================
피팅 파이프라인 설정 (1급 객체 Loss Function 지원)

Usage:
    from xrefine.refine.config import RefinementConfig, StageConfig
    from xrefine.refine.losses import LOG_MSE, LINEAR_MSE, CORRELATION
    
    config = RefinementConfig(
        stages=[
            StageConfig("My Stage", loss_fn=LINEAR_MSE, optimize_keys={'d_f', 'sig_f'})
        ]
    )
"""
from dataclasses import dataclass, field
from typing import Set, Optional, List, Callable, Any
from xrefine.refine.losses import (
    LossFn, log_mse_loss, linear_mse_loss, correlation_loss,
    LOG_MSE, LINEAR_MSE, CORRELATION, LOG_MSE_WITH_GRAD
)


@dataclass
class StageConfig:
    """단일 최적화 스테이지 설정"""
    
    name: str = "Unnamed Stage"
    
    # 최적화할 파라미터 키
    optimize_keys: Set[str] = field(default_factory=set)
    
    # 강제 고정할 파라미터 (key: value 형태)
    fixed_params: dict = field(default_factory=dict)
    
    # Loss 함수 (1급 객체)
    loss_fn: LossFn = log_mse_loss
    
    # Q-Range (None = config 기본값 사용)
    q_min: Optional[float] = None
    q_max: Optional[float] = None
    
    # Grid Search 옵션 (두께 탐색용)
    use_grid_search: bool = False
    grid_range: tuple = (20.0, 1200.0)
    grid_points: int = 500
    
    # Optimizer 설정
    maxiter: int = 100
    popsize: int = 30
    
    # 추가 옵션
    description: str = ""
    
    def __repr__(self):
        loss_name = getattr(self.loss_fn, '__name__', 'custom')
        return f"StageConfig({self.name}, loss={loss_name})"
    
    def with_loss(self, loss_fn: LossFn) -> 'StageConfig':
        """Loss 함수를 변경한 새 StageConfig 반환 (Fluent API)"""
        return StageConfig(
            name=self.name,
            optimize_keys=self.optimize_keys,
            fixed_params=self.fixed_params,
            loss_fn=loss_fn,
            q_min=self.q_min,
            q_max=self.q_max,
            use_grid_search=self.use_grid_search,
            grid_range=self.grid_range,
            grid_points=self.grid_points,
            maxiter=self.maxiter,
            popsize=self.popsize
        )


@dataclass 
class RefinementConfig:
    """전체 피팅 파이프라인 설정"""
    
    # 기본 설정
    q_cutoff: float = 0.45  # 고각도 노이즈 커팅
    min_thickness_ratio: float = 2.0  # d_min = min_thickness_ratio / q_critical
    thickness_fine_range: float = 50.0  # ± 범위 (최종 미세 조정용)
    
    # 디버그 출력
    verbose: bool = True
    
    # 스테이지 순서 정의
    stages: List[StageConfig] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.stages:
            self.stages = self.default_stages()
    
    @staticmethod
    def default_stages() -> List[StageConfig]:
        """기본 5단계 파이프라인 (마지막에 bkg 재조정)"""
        return [
            # Stage 0: Footprint & Background
            StageConfig(
                name="Footprint & Background",
                optimize_keys={'beam_width', 'L', 'i0', 'bkg'},
                loss_fn=LOG_MSE,
                q_max=0.15,
                maxiter=50, popsize=20,
                description="Low-Q 영역에서 풋프린트/배경 최적화"
            ),
            
            # Stage 1: Thickness Grid Search
            StageConfig(
                name="Thickness Grid Search",
                optimize_keys={'d_f'},
                use_grid_search=True,
                loss_fn=CORRELATION,
                grid_range=(20.0, 1200.0),
                grid_points=500,
                description="Correlation 기반 전역 두께 탐색"
            ),
            
            # Stage 2: Linear R Fitting
            StageConfig(
                name="Linear R Fitting",
                optimize_keys={'sig_f', 'rho_f', 'd_ox', 'sig_ox', 'rho_ox', 'L', 'i0', 'bkg', 'dq'},
                loss_fn=LINEAR_MSE,
                maxiter=100, popsize=30,
                description="Linear R로 프린지 진폭 맞추기"
            ),
            
            # Stage 3: Log R Fine-Tuning
            StageConfig(
                name="Log R Fine-Tuning",
                optimize_keys={'sig_f', 'rho_f', 'd_ox', 'sig_ox', 'rho_ox', 'L', 'i0', 'bkg', 'dq'},
                loss_fn=LOG_MSE,
                maxiter=100, popsize=30,
                description="Log R로 전체 형상 미세 조정"
            ),
            
            # Stage 4: Thickness Fine-Tuning
            StageConfig(
                name="Thickness Fine-Tuning",
                optimize_keys={'d_f', 'sig_f', 'rho_f', 'd_ox', 'sig_ox', 'rho_ox', 'L', 'i0', 'bkg', 'dq'},
                loss_fn=LOG_MSE,
                maxiter=50, popsize=20,
                description="좁은 범위에서 두께 미세 조정"
            ),
            
            # Stage 5: Background 재조정 (물리 파라미터 고정)
            # StageConfig(
            #     name="Background Re-Adjustment",
            #     optimize_keys={'bkg'},  # bkg만 최적화
            #     loss_fn=LOG_MSE,
            #     q_min=0.15,  # 고각도 영역에서 bkg 영향 확인
            #     maxiter=30, popsize=15,
            #     description="물리 파라미터 고정 후 배경 재조정"
            # )
        ]
    
    @staticmethod
    def simple_log_only() -> 'RefinementConfig':
        """간단 설정: Log MSE만 사용 (빠름)"""
        return RefinementConfig(
            stages=[
                StageConfig(
                    name="Full Optimization",
                    optimize_keys={'d_f', 'sig_f', 'rho_f', 'd_ox', 'sig_ox', 'rho_ox', 'L', 'i0', 'bkg', 'dq'},
                    loss_fn=LOG_MSE,
                    maxiter=200, popsize=40
                )
            ]
        )
    
    @staticmethod
    def quick_thickness() -> 'RefinementConfig':
        """두께만 빠르게 찾기"""
        return RefinementConfig(
            stages=[
                StageConfig(
                    name="Quick Thickness",
                    optimize_keys={'d_f'},
                    use_grid_search=True,
                    loss_fn=CORRELATION
                )
            ]
        )
    
    @staticmethod
    def high_quality() -> 'RefinementConfig':
        """고품질 피팅 (시간 오래 걸림)"""
        stages = RefinementConfig.default_stages()
        for s in stages:
            s.maxiter = s.maxiter * 2
            s.popsize = s.popsize * 2
        return RefinementConfig(stages=stages)


# 기본 설정 싱글톤
DEFAULT_CONFIG = RefinementConfig()
