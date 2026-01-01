from typing import Optional, Any, Tuple
import os

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import lib_platform

from xrefine.core.interface import Simulator
from xrefine.config import XRefineConfig
from xrefine.core.structures import ParamSet
from xrefine.data.augmentations import XRRAugmentations


def generate_synthetic_batch(
    batch_size: int, 
    q_grid: Tensor, 
    config: Any, 
    simulator: Optional[Simulator] = None, 
    augmenter: Optional[XRRAugmentations] = None
) -> Tuple[Tensor, ParamSet[Tensor]]:
    device = q_grid.device
    if simulator is None:
        simulator = Simulator(method='abeles', device=device)
    
    n = config.n_layers
    def get_rand(shape, range_tuple):
        return torch.rand(*shape, device=device) * (range_tuple[1] - range_tuple[0]) + range_tuple[0]

    # --- Step 1. 물리 파라미터 샘플링 ---
    d_f = get_rand((batch_size, n), config.thick_range)
    sig_f = get_rand((batch_size, n), config.rough_range)
    rho_f = get_rand((batch_size, n), config.sld_range)
    d_ox = get_rand((batch_size, 1), config.sio2_thick_range)
    sig_ox = get_rand((batch_size, 1), config.sio2_rough_range)
    rho_ox = get_rand((batch_size, 1), config.sio2_sld_range) 
    sample_len = get_rand((batch_size, 1), config.sample_len_range)
    beam_width = get_rand((batch_size, 1), config.beam_width_range)

    # --- Step 2. 물리 구조 조립 및 시뮬레이션 ---
    thickness = torch.cat([d_f, d_ox], dim=1)
    roughness = torch.cat([sig_f, sig_ox, sig_ox], dim=1) 
    sld = torch.cat(
        [torch.zeros((batch_size, 1), device=device), rho_f, rho_ox, 
        torch.full((batch_size, 1), config.substrate_sld, device=device)], dim=1
    )

    q_batch = q_grid.view(1, -1).expand(batch_size, -1)
    refl = simulator(q_batch, thickness, roughness, sld, 
        dq=torch.full((batch_size, 1), config.instrument.res, device=device))

    # --- Step 3. 파라미터 주입 (Injection) ---
    i0_list = []
    bkg_list = []
    beam_width_list = []
    if augmenter is not None:
        refl_np = refl.detach().cpu().numpy()
        q_np = q_batch.detach().cpu().numpy()
        r_aug_list = []
        for i in range(batch_size):
            _, r_aug, meta = augmenter(q_np[i], refl_np[i])
            r_aug_list.append(r_aug)
            i0_list.append(meta['I0'])
            bkg_list.append(meta['bkg'])
            beam_width_list.append(meta['beam_width'])

        r_noisy = torch.tensor(np.array(r_aug_list), dtype=torch.float32, device=device)
        i0_true = torch.tensor(i0_list, dtype=torch.float32, device=device).view(-1, 1)
        bkg_true = torch.tensor(bkg_list, dtype=torch.float32, device=device).view(-1, 1)
        beam_width = torch.tensor(beam_width_list, dtype=torch.float32, device=device).view(-1, 1)
    else:
        r_noisy = refl
        i0_true = torch.ones((batch_size, 1), device=device)
        bkg_true = torch.full((batch_size, 1), -8.0, device=device) # Default bkg
        beam_width = torch.full((batch_size, 1), 0.01, device=device) # Default beam width

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

def get_xrr_dataloader(config, q_grid, steps_per_epoch=100) -> DataLoader:
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