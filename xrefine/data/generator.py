import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, Tuple
from dataclasses import fields

from xrefine.core.abeles import AbelesMatrix  # [FIX] 직접 import
from xrefine.config import XRefineConfig
from xrefine.core.structures import ParamSet
from xrefine.data.augmentations import XRRAugmentor
from xrefine.logger import logger

def generate_synthetic_batch(
    batch_size: int, 
    q_grid: Tensor, 
    config: XRefineConfig, 
    simulator: Optional[AbelesMatrix] = None, 
    augmenter: Optional[XRRAugmentor] = None
) -> Tuple[Tensor, ParamSet]:
    """
    ParamSet 메타데이터를 기반으로 모든 파라미터를 한 번에 샘플링하고
    물리 연산과 증강을 순차적으로 수행하는 고성능 데이터 생성기.
    """
    device = config.device # q_grid.device 대신 config.device 사용 (안전)
    
    # Q-grid 동기화
    q_grid = q_grid.to(device)
    
    if simulator is None:
        simulator = AbelesMatrix(device=device)
    
    # config.model.n_layers가 없으면 기본값 1
    n = getattr(config.model, 'n_layers', 1)
    sampled_values = {}

    # --- Step 1. 통합 샘플링 (물리 + 장비 파라미터 전역 생성) ---
    for fld in fields(ParamSet):
        mult = fld.metadata.get('mult', 1)
        width = n if mult == 'n' else 1
        
        # Pydantic Config의 Tuple (min, max) 참조
        p_range = getattr(config.ranges, fld.name)
        low, high = p_range
        
        # Vectorized Sampling
        sampled_values[fld.name] = torch.rand(
            (batch_size, width), device=device, dtype=torch.float32
        ) * (high - low) + low

    params_gt = ParamSet(**sampled_values)

    # --- Step 2. 물리 구조 조립 ---
    # ParamSet 내부 메서드 활용하여 일관성 유지
    thickness, roughness, sld_full = params_gt.assemble_structure(config.ranges.substrate_sld)

    # --- Step 3. 순수 물리 시뮬레이션 ---
    q_batch = q_grid.view(1, -1).expand(batch_size, -1) # (B, Q)
    
    # [NEW] 2theta Offset 적용 (현실적인 기기 오차 모사)
    if hasattr(params_gt, 'tth_offset'):
        # 근사식: q_new = q_old + offset
        # q = 4pi/lambda * sin(theta) => dq approx (4pi/lambda) * cos(theta) * d_theta
        # 여기서는 간단히 q_shift로 적용 (모델이 tth_offset을 학습하므로 중요함)
        tth_offset_rad = params_gt.tth_offset * (3.14159 / 180.0) / 2.0
        q_offset = (4 * 3.14159 / config.instrument.wavelength) * torch.sin(tth_offset_rad)
        q_batch = q_batch + q_offset

    # 시뮬레이터 호출 (AbelesMatrix)
    refl_pure = simulator(
        q=q_batch,
        thickness=thickness,
        roughness=roughness,
        sld=sld_full,
        sample_length=params_gt.L,
        beam_width=params_gt.beam_width,
        wavelength=config.instrument.wavelength
    )

    # --- Step 4. 통합 증강 및 기기 효과 ---
    if augmenter is not None:
        # Augmentor 내부에서 I0, Background, Noise 등을 모두 처리한다고 가정
        r_final = augmenter(q_batch, refl_pure, params_gt)
    else:
        # 기본 처리 (I0 + Bkg)
        r_final = refl_pure * params_gt.i0 + torch.pow(10.0, params_gt.bkg)
        # Simple Gaussian Noise
        noise = torch.randn_like(r_final) * config.augmentations.intensity_noise_scale * r_final
        r_final = r_final + noise

    # 수치 안정성 및 Log 변환 (모델 입력용)
    r_final = torch.clamp(r_final, min=1e-12, max=2.0)
    log_r = torch.log10(r_final)
    
    if torch.isnan(log_r).any():
        log_r = torch.nan_to_num(log_r, nan=-10.0)

    return log_r, params_gt

class XRRStreamingDataset(IterableDataset):
    """GPU에서 배치를 직접 생성하여 전송 오버헤드를 없앤 스트리밍 데이터셋"""
    def __init__(self, config: XRefineConfig, q_grid: torch.Tensor, steps_per_epoch: int):
        super().__init__()
        self.config = config
        self.q_grid = q_grid
        self.steps_per_epoch = steps_per_epoch
        # Augmentor 초기화 (없으면 None)
        try:
            self.augmenter = XRRAugmentor(config).to(config.device)
        except:
            self.augmenter = None # Augmentor 클래스가 없거나 에러 시 기본 로직 사용

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            r_noisy, params_gt = generate_synthetic_batch(
                batch_size=self.config.train.batch_size,
                q_grid=self.q_grid,
                config=self.config,
                augmenter=self.augmenter
            )
            yield r_noisy, params_gt

def get_xrr_dataloader(config: XRefineConfig, q_grid: Tensor) -> DataLoader:
    """
    Worker 간 컨텍스트 스위칭이 없는 최속의 DataLoader 설정.
    """
    dataset = XRRStreamingDataset(
        config, q_grid, steps_per_epoch=config.train.steps_per_epoch
    )
    # num_workers=0: 메인 프로세스(GPU 소유)에서 직접 생성 -> CUDA Context 복사 비용 없음
    return DataLoader(dataset, batch_size=None, num_workers=0)