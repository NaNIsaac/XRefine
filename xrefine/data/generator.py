import os
from typing import Optional, Any, Tuple

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import lib_platform

from xrefine.core.interface import Simulator
from xrefine.config import XRefineConfig
from xrefine.core.structures import ParamSet
from xrefine.data.augmentations import XRRAugmentor  # 수정된 클래스 임포트
from xrefine.logger import logger

def generate_synthetic_batch(
    batch_size: int, 
    q_grid: Tensor, 
    config: XRefineConfig, 
    simulator: Optional[Simulator] = None, 
    augmenter: Optional[XRRAugmentor] = None
) -> Tuple[Tensor, ParamSet[Tensor]]:
    """
    고성능 배치 기반 물리 데이터 생성기.
    NumPy 변환 없이 모든 과정을 GPU 텐서 위에서 처리합니다.
    """
    device = q_grid.device
    if simulator is None:
        simulator = Simulator(method='abeles', device=device)
    
    # --- Step 1. 물리 파라미터 샘플링 (Vectorized) ---
    n = config.n_layers
    def get_rand(shape, range_tuple):
        return torch.rand(*shape, device=device) * (range_tuple[1] - range_tuple[0]) + range_tuple[0]

    d_f = get_rand((batch_size, n), config.thick_range)
    sig_f = get_rand((batch_size, n), config.rough_range)
    rho_f = get_rand((batch_size, n), config.sld_range)
    
    d_ox = get_rand((batch_size, 1), config.sio2_thick_range)
    sig_ox = get_rand((batch_size, 1), config.sio2_rough_range)
    rho_ox = get_rand((batch_size, 1), config.sio2_sld_range) 
    
    # --- Step 2. 물리 구조 조립 및 시뮬레이션 ---
    # 두께: [Film_layers, SiO2]
    thickness = torch.cat([d_f, d_ox], dim=1)
    # 거칠기: [Film_interfaces, SiO2_interface, Substrate_interface]
    roughness = torch.cat([sig_f, sig_ox, sig_ox], dim=1) 
    # SLD: [Ambient(0), Film_layers, SiO2, Substrate]
    sld = torch.cat([
        torch.zeros((batch_size, 1), device=device), 
        rho_f, 
        rho_ox, 
        torch.full((batch_size, 1), config.substrate_sld, device=device)
    ], dim=1)

    q_batch = q_grid.view(1, -1).expand(batch_size, -1)
    
    # Pure Reflectivity (No noise yet)
    # Instrument resolution은 시뮬레이션 레벨에서 적용 (Smearing 전 단계)
    refl = simulator(
        q_batch, thickness, roughness, sld, 
        dq=torch.full((batch_size, 1), config.instrument.res, device=device)
    )

    # --- Step 3. GPU 기반 물리 증강 (Augmentation) ---
    if augmenter is not None:
        # Loop 제거: 배치를 통째로 주입하여 GPU에서 병렬 처리
        r_noisy, meta = augmenter(q_batch, refl)
        
        i0_true = meta['i0']
        bkg_true = meta['bkg']
        sample_len = meta['sample_len']
        beam_width = meta['beam_width']
    else:
        # 증강이 없을 경우 기본값 할당 (Tensor 형태 유지)
        r_noisy = refl
        i0_true = torch.ones((batch_size, 1), device=device)
        bkg_true = torch.full((batch_size, 1), -8.0, device=device)
        sample_len = torch.full((batch_size, 1), 10.0, device=device)
        beam_width = torch.full((batch_size, 1), 0.1, device=device)

    # 수치 안정성 확보
    r_noisy = torch.clamp(r_noisy, min=1e-10)

    # 모든 가변 변수가 포함된 통합 ParamSet 반환
    return r_noisy, {
        'd_f': d_f, 'sig_f': sig_f, 'rho_f': rho_f,
        'd_ox': d_ox, 'sig_ox': sig_ox, 'rho_ox': rho_ox,
        'L': sample_len,
        'i0': i0_true,
        'bkg': bkg_true,
        'beam_width': beam_width
    }

class XRRStreamingDataset(IterableDataset):
    """무한 물리 데이터를 GPU에서 스트리밍하는 데이터셋"""
    def __init__(self, config: XRefineConfig, q_grid: torch.Tensor, steps_per_epoch: int):
        super().__init__()
        self.config = config
        self.q_grid = q_grid.to(config.device) # 데이터셋 레벨에서 GPU에 미리 로드
        self.steps_per_epoch = steps_per_epoch
        # 데이터셋 생성 시 Augmentor 초기화 (GPU)
        self.augmenter = XRRAugmentor(config).to(config.device)

    def __iter__(self):
        # 파이토치 DataLoader 워커 내에서 실행
        for _ in range(self.steps_per_epoch):
            # 매 스텝마다 배치를 GPU에서 직접 생성
            r_noisy, params_true = generate_synthetic_batch(
                batch_size=self.config.batch_size,
                q_grid=self.q_grid,
                config=self.config,
                augmenter=self.augmenter
            )
            yield r_noisy, params_true

def get_xrr_dataloader(config: XRefineConfig, q_grid: Tensor, steps_per_epoch: int = 100) -> DataLoader:
    """
    GPU 생성이므로 num_workers=0 설정이 가장 빠름 (Context Switch 방지).
    메모리 복사가 없으므로 병렬 워커가 필요 없습니다.
    """
    dataset = XRRStreamingDataset(config, q_grid, steps_per_epoch)
    
    # GPU에서 직접 생성할 때는 num_workers > 0 일 경우 CUDA 초기화 에러가 날 수 있음
    return DataLoader(
        dataset, 
        batch_size=None, # IterableDataset이므로 None
        num_workers=0,   # GPU Direct generation
        pin_memory=False # 이미 GPU에 있으므로 False
    )