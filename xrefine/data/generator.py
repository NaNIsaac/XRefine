from typing import Optional, Any, Tuple
import os

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import lib_platform

from xrefine.core.interface import Simulator
from xrefine.config import XRefineConfig
from xrefine.core.corrections import DifferentiableFootprint

def generate_synthetic_batch(
    batch_size: int, 
    q_grid: Tensor, 
    config: Any, 
    simulator: Optional[Simulator] = None, 
    augmenter: Optional[Any] = None
) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """
    XRR 데이터를 생성하는 핵심 함수. 
    모델과 동일한 8-파라미터 체계와 풋프린트 물리 로직을 따릅니다.
    """
    device = q_grid.device
    
    # 1. 물리 엔진 컴포넌트 준비
    if simulator is None:
        simulator = Simulator(method='abeles', device=device)
    
    # 모델과 동일한 풋프린트 레이어를 사용하여 정합성 유지
    footprint_layer = DifferentiableFootprint(
        beam_width=config.instrument.beam_width,
        wavelength=config.instrument.wavelength
    ).to(device)

    n = config.n_layers

    # 랜덤 샘플링 헬퍼 함수
    def get_rand(shape, range_tuple):
        return torch.rand(*shape, device=device) * (range_tuple[1] - range_tuple[0]) + range_tuple[0]

    # --- Step 1. 물리 파라미터 샘플링 ---
    # [Main Film]
    d_f   = get_rand((batch_size, n), config.thick_range)
    sig_f = get_rand((batch_size, n), config.rough_range)
    rho_f = get_rand((batch_size, n), config.sld_range)
    
    # [SiO2 Layer]
    d_ox   = get_rand((batch_size, 1), config.sio2_thick_range)
    sig_ox = get_rand((batch_size, 1), config.sio2_rough_range)
    rho_ox = get_rand((batch_size, 1), config.sio2_sld_range) 
    
    # [Instrumental & Substrate]
    sample_len = get_rand((batch_size, 1), config.sample_len_range)
    sub_sld    = torch.full((batch_size, 1), config.substrate_sld, device=device)
    amb_sld    = torch.zeros((batch_size, 1), device=device)

    # --- Step 2. 물리 구조 조립 (Topology Matching) ---
    # Thickness: [Film, SiO2]
    thickness = torch.cat([d_f, d_ox], dim=1)
    # Roughness: [Amb/Film, Film/SiO2, SiO2/Sub]
    roughness = torch.cat([sig_f, sig_ox, sig_ox], dim=1) 
    # SLD: [Amb, Film, SiO2, Sub]
    sld = torch.cat([amb_sld, rho_f, rho_ox, sub_sld], dim=1)

    # --- Step 3. 물리 시뮬레이션 ---
    if q_grid.ndim == 1:
        q_grid = q_grid.view(1, -1)
    q_batch = q_grid.expand(batch_size, -1)
    
    # (A) Abeles 기초 반사율 계산
    dq_val = config.instrument.res
    dq = torch.full((batch_size, 1), dq_val, device=device)
    r_model = simulator(q_batch, thickness, roughness, sld, dq=dq)

    # (B) [핵심 수정] 풋프린트 물리 강제 적용
    # 이제 검정색 선(Target)도 모델(PINN Fit)과 동일하게 저각도에서 감쇠합니다.
    r_footprint = footprint_layer(q_batch, r_model, sample_len)

    # --- Step 4. 노이즈 및 증강 (Post-processing) ---
    if augmenter is not None:
        # GPU 연산을 거친 텐서를 넘겨주되, 증강기 내부 로직에 따라 처리
        r_np = r_footprint.detach().cpu().numpy()
        q_np = q_batch.detach().cpu().numpy()
        
        r_aug_list = []
        for i in range(batch_size):
            # Q-shift, Background, Poisson 노이즈 적용
            _, r_aug = augmenter(q_np[i], r_np[i])
            r_aug_list.append(r_aug)
            
        r_noisy = torch.tensor(np.array(r_aug_list), dtype=torch.float32, device=device)
    else:
        r_noisy = r_footprint

    # 수치적 안정성을 위한 클램핑
    r_noisy = torch.clamp(r_noisy, min=1e-10)

    return r_noisy, (thickness, roughness, sld, sample_len)


class XRRStreamingDataset(IterableDataset):
    """무한한 물리 데이터를 CPU에서 생성하여 스트리밍하는 데이터셋"""
    def __init__(self, config: XRefineConfig, q_grid: torch.Tensor, steps_per_epoch: int):
        super().__init__()
        self.config = config
        self.q_grid = q_grid
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self):
        # 파이토치 DataLoader의 각 worker가 이 generator를 독립적으로 실행
        for _ in range(self.steps_per_epoch):
            r_noisy, params_true = generate_synthetic_batch(
                batch_size=self.config.batch_size,
                q_grid=self.q_grid,
                config=self.config
            )
            yield r_noisy, params_true

def worker_init_fn(worker_id):
    """멀티프로세싱 시 각 worker가 고유한 난수 시퀀스를 갖도록 설정"""
    process_seed = torch.initial_seed()
    base_seed = process_seed % (2**32)
    np.random.seed(base_seed + worker_id)
    torch.manual_seed(base_seed + worker_id)

def get_xrr_dataloader(config, q_grid, steps_per_epoch=100):
    """코랩 환경의 I/O 병목을 피하기 위해 최적화된 DataLoader 반환"""
    dataset = XRRStreamingDataset(config, q_grid, steps_per_epoch)
    workers = 0 if lib_platform.is_platform_windows else min(2, os.cpu_count())
    return DataLoader(
        dataset, 
        batch_size=None,
        num_workers=workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True if config.device == 'cuda' else False
    )