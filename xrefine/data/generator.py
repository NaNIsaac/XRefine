import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, Tuple
from dataclasses import fields

from xrefine.core.interface import Simulator
from xrefine.config import XRefineConfig
from xrefine.core.structures import ParamSet
from xrefine.data.augmentations import XRRAugmentor
from xrefine.logger import logger

def generate_synthetic_batch(
    batch_size: int, 
    q_grid: Tensor, 
    config: XRefineConfig, 
    simulator: Optional[Simulator] = None, 
    augmenter: Optional[XRRAugmentor] = None
) -> Tuple[Tensor, ParamSet]:
    """
    ParamSet 메타데이터를 기반으로 모든 파라미터를 한 번에 샘플링하고
    물리 연산과 증강을 순차적으로 수행하는 고성능 데이터 생성기.
    """
    device = q_grid.device
    if simulator is None:
        simulator = Simulator(method='abeles', device=device)
    
    n = config.model.n_layers
    sampled_values = {}

    # --- Step 1. 통합 샘플링 (물리 + 장비 파라미터 전역 생성) ---
    # ParamSet의 정의(스키마)를 바탕으로 모든 필드를 한 번에 샘플링합니다.
    for fld in fields(ParamSet):
        # 1. 차원 결정 (레이어 종속 'n' vs 단일 값 1)
        mult = fld.metadata.get('mult', 1)
        width = n if mult == 'n' else 1
        
        # 2. 범위 가져오기 (config.ranges에서 필드명으로 직접 참조)
        # config.ranges.d_f, config.ranges.i0 등을 직접 호출
        p_range = getattr(config.ranges, fld.name)
        low, high = p_range
        
        # 3. Vectorized Sampling on GPU
        sampled_values[fld.name] = torch.rand(
            (batch_size, width), device=device, dtype=torch.float32
        ) * (high - low) + low

    # Ground Truth 객체 생성 (이미 모든 장비 파라미터 L, i0, bkg, bw 포함)
    params_gt = ParamSet(**sampled_values)

    # --- Step 2. 물리 구조 조립 (Simulation Input Prep) ---
    # 시뮬레이터(Abeles) 입력 규격에 맞게 텐서 결합
    # Thickness: [Film_n, Oxide]
    thickness = torch.cat([params_gt.d_f, params_gt.d_ox], dim=1)
    
    # Roughness: [Film_n, Oxide_top, Substrate_interface]
    # (주의: 계면 개수는 층 개수 + 1)
    roughness = torch.cat([params_gt.sig_f, params_gt.sig_ox, params_gt.sig_ox], dim=1)
    
    # SLD: [Ambient(0), Film_n, Oxide, Substrate]
    sub_sld = torch.full((batch_size, 1), config.ranges.substrate_sld, device=device)
    sld_full = torch.cat([
        torch.zeros((batch_size, 1), device=device),
        params_gt.rho_f,
        params_gt.rho_ox,
        sub_sld
    ], dim=1)

    # --- Step 3. 순수 물리 시뮬레이션 (Pure Reflectivity) ---
    q_batch = q_grid.view(1, -1).expand(batch_size, -1)
    
    # 시뮬레이터는 기본 장비 분해능(res)만 반영하여 계산
    refl_pure = simulator(
        q_batch, thickness, roughness, sld_full,
        sample_length=params_gt.L,
        beam_width=params_gt.beam_width,
        dq=params_gt.dq
    )

    # --- Step 4. 통합 증강 (ParamSet 주입 및 보정) ---
    if augmenter is not None:
        r_noisy = augmenter(q_batch, refl_pure, params_gt)
    else:
        r_noisy = refl_pure * params_gt.i0 + torch.pow(10.0, params_gt.bkg)

    # 수치 안정성 확보
    r_noisy = torch.clamp(r_noisy, min=1e-11)

    return r_noisy, params_gt

class XRRStreamingDataset(IterableDataset):
    """GPU에서 배치를 직접 생성하여 전송 오버헤드를 없앤 스트리밍 데이터셋"""
    def __init__(self, config: XRefineConfig, q_grid: torch.Tensor, steps_per_epoch: int):
        super().__init__()
        self.config = config
        self.q_grid = q_grid.to(config.device)
        self.steps_per_epoch = steps_per_epoch
        # 전역 증강기 초기화
        self.augmenter = XRRAugmentor(config).to(config.device)

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
    # num_workers=0은 배치를 메인 프로세스의 GPU에서 직접 생성함을 의미 (가장 빠름)
    return DataLoader(dataset, batch_size=None, num_workers=0)