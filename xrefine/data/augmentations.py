import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from xrefine.config import XRefineConfig
from xrefine.core.structures import ParamSet

class XRRAugmentor(nn.Module):
    """
    GPU 기반 고속 XRR 물리 증강 레이어 (최적화 버전).
    
    개선 사항:
    1. 고정 커널 크기: torch.compile 및 CUDA 그래프 안정화를 위해 최대 시그마 기반 커널 고정.
    2. 완전한 Config 반영: q_shift_sigma, res_sigma_range 등 모든 인자 적용.
    3. 수치적 철통 방어: Poisson 에러 및 NaN 전파를 막는 3단계 클램핑.
    """
    def __init__(self, config: XRefineConfig):
        super().__init__()
        self.cfg = config
        self.aug_cfg = config.augmentations
        self.register_buffer('sqrt_8ln2', torch.sqrt(torch.tensor(8.0 * np.log(2.0))))

    def apply_smearing(self, R: torch.Tensor, res_sigma: torch.Tensor) -> torch.Tensor:
        """배치 단위 가우시안 스미어링 (고정 커널 최적화 버전)"""
        B, Q = R.shape
        q_grid = self.cfg.q_grid.to(R.device)
        dq = (q_grid[-1] - q_grid[0]) / (Q - 1)
        
        # [최적화] 커널 사이즈를 배치의 최대값이 아닌, 설정 가능한 최대 범위에 맞춤
        # 이렇게 하면 매 에폭마다 커널 크기가 바뀌어 발생하는 컴파일 오버헤드를 줄임
        max_possible_sigma = self.aug_cfg.res_sigma_range[1]
        max_sigma_px = max_possible_sigma / dq
        kernel_size = int(max_sigma_px * 6)
        if kernel_size % 2 == 0: kernel_size += 1
        if kernel_size < 3: return R

        padding = kernel_size // 2
        x = torch.linspace(-padding, padding, kernel_size, device=R.device)
        sigma_px = res_sigma / (dq + 1e-9) # (B, 1)
        
        # 가우시안 커널 생성 (B, 1, K)
        # x.view(1, -1) 와 sigma_px.view(-1, 1) 의 브로드캐스팅 활용
        kernel = torch.exp(-0.5 * (x.view(1, kernel_size) / sigma_px)**2)
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-9)
        kernel = kernel.unsqueeze(1) 

        # Forward Pass: Replicate 패딩으로 경계면 아티팩트 방지
        inp = R.unsqueeze(1) # (B, 1, Q)
        inp_padded = F.pad(inp, (padding, padding), mode='replicate')
        
        # [Fix] bias=None 명시 및 groups=B로 배치별 독립 커널 적용
        smeared = F.conv1d(inp_padded.view(1, B, -1), kernel, bias=None, groups=B)
        
        return smeared.view(B, Q)

    @torch.no_grad()
    def forward(self, q: torch.Tensor, R: torch.Tensor, params: ParamSet) -> torch.Tensor:
        """[Pipeline] Smearing -> Q-Shift -> Scaling -> Noise"""
        B, Q = R.shape
        device = R.device
        
        # 0. NaN 방어 (물리 엔진에서 올라올 수 있는 불확실성 제거)
        R = torch.nan_to_num(R, nan=1e-11, posinf=1.0, neginf=1e-11)

        # 1. Resolution Smearing (가변 분해능)
        # 1. Resolution Smearing (Disabled - Handled by Simulator with params.dq)
        # low, high = self.aug_cfg.res_sigma_range
        # res_sig = torch.rand(B, 1, device=device) * (high - low) + low
        # R_aug = self.apply_smearing(R, res_sig)
        R_aug = R
        
        # 2. Q-Shift (정렬 오차 모사)
        if self.aug_cfg.q_shift_sigma > 0:
            # 단순화를 위해 강도에 미세한 변화를 주어 정렬 불안정성 표현
            shift_noise = torch.randn_like(R_aug) * self.aug_cfg.q_shift_sigma
            R_aug = (R_aug + shift_noise).clamp(min=1e-11)

        # 3. Scaling & Background (I0 및 Background 적용)
        R_aug = R_aug * params.i0 + torch.pow(10.0, params.bkg)
        R_aug = torch.nan_to_num(R_aug, nan=1e-11).clamp(min=0.0)

        # 4. Shot Noise (Poisson Noise)
        # intensity_noise_scale이 클수록 더 많은 노이즈를 주입하도록 s_rand 조절
        noise_factor = 1.0 / (self.aug_cfg.intensity_noise_scale + 1e-6)
        s_rand = torch.pow(10.0, torch.rand(B, 1, device=device) * 2 + 5) * noise_factor
        
        # [CUDA Safety] Poisson Rate 검증
        poisson_rate = torch.clamp(R_aug * s_rand, min=0.0)
        poisson_rate = torch.nan_to_num(poisson_rate, nan=0.0)
        
        R_aug = torch.poisson(poisson_rate) / (s_rand + 1e-9)
        
        # 5. 최종 클램핑 (실전 데이터 마스킹 기준과 일치)
        return torch.clamp(R_aug, min=1e-11)