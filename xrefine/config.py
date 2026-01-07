import torch
import yaml
from pathlib import Path
from typing import Literal, Self, Tuple
from pydantic import BaseModel, Field, ConfigDict

from xrefine.physics_utils import tth2q
from xrefine.core.structures import ParamSet

# 1. 장비 사양 설정
class InstrumentConfig(BaseModel):
    wavelength: float = 1.5406  # Cu K-alpha [Å]
    beam_width: float = 0.1     # [mm]
    res: float = 0.002          # dq resolution [Å^-1]
    I0: float = 1.0             # Intensity scaling
    Ibkg_init: float = 1e-7     # Initial background noise

# 2. 데이터 증강 하이퍼파라미터
class AugmentationConfig(BaseModel):
    intensity_noise_scale: float = 0.2
    bg_range: Tuple[float, float] = (-9.0, -5.0)  # Log10 scale
    res_sigma_range: Tuple[float, float] = (0.0001, 0.005)
    q_shift_sigma: float = 0.001
    beam_profile: Literal['gaussian', 'square'] = 'gaussian'

# 3. 모델 아키텍처 설정
class ModelArchConfig(BaseModel):
    encoder_depth: int = 6
    n_channels: int = 128
    kernel_size: int = 15
    dilation: int = 2
    fourier_scale: float = 40.0
    mapping_size: int = 64
    hidden_dim: int = 1024
    n_layers: int = 1

# 4. 물리 파라미터 샘플링 범위 (Source of Truth for ParamSet)
class PhysicalRangeConfig(BaseModel):
    # Main Film
    d_f: Tuple[float, float] = (10.0, 1000.0)
    sig_f: Tuple[float, float] = (0.5, 20.0)
    rho_f: Tuple[float, float] = (5.0, 100.0)
    
    # Oxide Layer
    d_ox: Tuple[float, float] = (5.0, 50.0)
    sig_ox: Tuple[float, float] = (0.5, 10.0)
    rho_ox: Tuple[float, float] = (14.0, 20.0)
    
    # Instrumental
    L: Tuple[float, float] = (5.0, 25.0)  # Expanded range for better footprint fitting
    beam_width: Tuple[float, float] = (0.01, 0.3)
    bkg: Tuple[float, float] = (-9.0, -5.0)
    i0: Tuple[float, float] = (0.8, 1.2)
    
    substrate_sld: float = 20.1  # Fixed [10^-6 Å^-2]

# 5. 파라미터별 손실 함수 가중치
class WeightConfig(BaseModel):
    d_f: float = 1.0
    sig_f: float = 1.0
    rho_f: float = 1.0
    d_ox: float = 0.5
    sig_ox: float = 0.5
    rho_ox: float = 0.5
    L: float = 5.0
    i0: float = 1.0
    bkg: float = 1.0
    beam_width: float = 10.0

# 6. 학습 프로세스 하이퍼파라미터
class TrainingConfig(BaseModel):
    # 기초 하이퍼파라미터
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 300
    steps_per_epoch: int = 100
    warmup_steps: int = 1000
    
    # Curriculum Learning & Physics Loss 제어
    sup_w_start: float = 100.0    # 초기 파라미터 지도 학습 가중치
    sup_w_end: float = 2000.0     # 최종 파라미터 지도 학습 가중치
    phys_penalty_w: float = 10.0  # 물리적 제약 조건(Penalty) 가중치
    
    # 파라미터별 개별 가중치 통합
    weights: WeightConfig = Field(default_factory=WeightConfig)

# 7. 프로젝트 통합 설정 (Root Config)
class XRefineConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='ignore')

    # 경로 및 환경
    experiment_name: str = 'XRefine_v1'
    save_root: Path = Path(r"C:\Warehouse\data\XRR_AI\PINN")
    log_dir: Path = Path("./logs")
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 측정 그리드 설정
    q_min_tth: float = 0.05
    q_max_tth: float = 12.0
    q_len: int = 2000

    # 서브 설정 객체들
    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    ranges: PhysicalRangeConfig = Field(default_factory=PhysicalRangeConfig)
    augmentations: AugmentationConfig = Field(default_factory=AugmentationConfig)
    train: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelArchConfig = Field(default_factory=ModelArchConfig)

    @property
    def q_grid(self) -> torch.Tensor:
        q_start = tth2q(self.q_min_tth, self.instrument.wavelength)
        q_end = tth2q(self.q_max_tth, self.instrument.wavelength)
        return torch.linspace(q_start, q_end, self.q_len)

    def get_weight_set(self, device: str = 'cpu') -> ParamSet:
        """학습 시 ParamSet 형태로 가중치를 즉시 활용하기 위한 팩토리 메서드"""
        w = self.train.weights
        return ParamSet(
            **{k: torch.tensor(v, device=device) for k, v in w.model_dump().items()}
        )

    def save_yaml(self, path: str | Path):
        """현재 설정을 YAML로 직렬화하여 저장"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.model_dump(mode='json'), f, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> Self:
        """YAML 파일로부터 설정 복원"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)

# 전역 설정 인스턴스
CONFIG = XRefineConfig()