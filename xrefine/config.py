from typing import Literal
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict
import torch
import numpy as np
import yaml

from xrefine.core.structures import ParamSet

# --- 물리 상수 및 유틸리티 ---
def tth2q(tth: float, wavelength: float = 1.5406) -> float:
    """2-theta (deg)를 Scattering Vector q (1/Å)로 변환"""
    rad = np.deg2rad(0.5 * tth)
    return float((4 * np.pi / wavelength) * np.sin(rad))

class InstrumentConfig(BaseModel):
    """장비 하드웨어 고정 사양 및 노이즈 기저치"""
    wavelength: float = 1.5406    # Cu K-alpha (Å)
    beam_width: float = 0.1      # mm (FWHM)
    res: float = 0.002           # dq resolution (Smearing 정도)
    I0: float = 1.0              # 데이터가 정규화되었으므로 1.0 기준
    Ibkg_init: float = 1e-7      # 초기 배경 노이즈 추정치

class AugmentationConfig(BaseModel):
    """학습 데이터 생성 시 적용할 변동성(Noise) 설정"""
    intensity_noise_scale: float = 0.2
    bg_range: tuple[float, float] = (-8.0, -5.0)
    res_sigma_range: tuple[float, float] = (0.0001, 0.005)
    q_shift_sigma: float = 0.001
    beam_profile: Literal['gaussian', 'square'] = 'gaussian'

class XRefineConfig(BaseModel):
    """전체 프로젝트 통합 설정 클래스"""
    
    # --- 실험 및 저장 경로 ---
    experiment_name: str = 'test'
    # save_root: str = r'D:\03_Resources\Data\XRR_AI\PINN'
    save_root: Path = Path(r'C:\Warehouse\data\XRR_AI\PINN')
    log_dir: Path = Path(r'logs')
    # --- 물리적 측정 범위 (q-grid) ---
    q_min_tth: float = 0.05
    q_max_tth: float = 12.0
    q_len: int = 2000

    # --- [Film] 메인 타겟 박막 범위 ---
    thick_range: tuple[float, float] = (10.0, 1000.0)
    rough_range: tuple[float, float] = (0.5, 20.0)
    sld_range: tuple[float, float] = (5.0, 100.0)

    # --- [SiO2] 하부 산화막 범위 (통일성을 위해 명시) ---
    sio2_thick_range: tuple[float, float] = (5.0, 50.0)
    sio2_rough_range: tuple[float, float] = (0.5, 10.0)
    sio2_sld_range: tuple[float, float] = (14, 20)

    sample_len_range: tuple[float, float] = (9.0, 11.0)  # mm
    beam_width_range: tuple[float, float] = (0.01, 0.3)  # mm
    substrate_sld: float = 20.1  # Si Substrate 고정

    # --- 서브 설정 객체 ---
    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    augmentations: AugmentationConfig = Field(default_factory=AugmentationConfig)

    # --- 학습 파라미터 (Curriculum Learning) ---
    batch_size: int = 50
    lr: float = 2e-4             # Fourier 모델은 낮은 LR이 안정적
    train_steps: int = 10000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 하이브리드 가중치 (Warmup)
    sup_w_start: float = 1000.0
    sup_w_end: float = 10.0
    warmup_steps: int = 3000
    sld_weight_factor: float = 3.0

    supervise_weight_params: ParamSet = {
        "d_f": 1.0,
        "sig_f": 1.0,
        "rho_f": 1.0,
        "d_ox": 1.0,
        "sig_ox": 1.0,
        "rho_ox": 1.0,
        "i0": 1.0,
        "bkg": 1.0,
        "L": 5.0,
        "beam_width": 20.0
    }
    # --- 모델 아키텍처 설정 ---
    model_type: Literal['pinn', 'fourier'] = 'fourier'
    n_layers: int = 1
    fourier_scale: float = 40.0
    hidden_dim: int = 512

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # --- 런타임 계산 속성 (Serialization 무시) ---
    @property
    def q_grid(self) -> torch.Tensor:
        q_start = tth2q(self.q_min_tth, self.instrument.wavelength)
        q_end = tth2q(self.q_max_tth, self.instrument.wavelength)
        return torch.linspace(q_start, q_end, self.q_len)

    # --- YAML 저장 및 로드 기능 ---
    def save_yaml(self, path: str | Path):
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> 'XRefineConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


CONFIG = XRefineConfig()