import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import fields

from xrefine.core.abeles import AbelesMatrix
from xrefine.core.structures import ParamSet
from xrefine.config import XRefineConfig

class FourierFeatureMapping(nn.Module):
    """
    [핵심 기술] 입력 q 신호를 고주파 영역으로 투영하여 
    미세한 주파수 변화(두께 변화)를 인코딩함.
    Transformer 모델에서도 Input Projection 대신 사용할 수 있음.
    """
    def __init__(self, mapping_size: int = 64, scale: float = 40.0):
        super().__init__()
        # scale이 높을수록 미세한 간섭 무늬 변화에 예민해짐
        self.register_buffer('B', torch.randn(1, mapping_size) * scale)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 1, Q) or (B, Q)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Fourier Projection
        # (B, 1, Q) -> transpose -> (B, Q, 1) @ (1, M) -> (B, Q, M)
        x_proj = (x.transpose(1, 2) @ self.B)
        x_proj = x_proj * 2 * torch.pi
        
        # Output: (B, Q, 2*M) -> Transpose -> (B, 2*M, Q) (CNN용)
        # Transformer용으로 쓸 때는 Transpose를 안 하면 됨
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out.transpose(1, 2) 

class XRRPhysicsModel(nn.Module):
    """
    Dilated CNN Encoder: 넓은 수용장으로 Fringe 주기를 파악
    """
    def __init__(self, config: XRefineConfig):
        super().__init__()
        # Config에서 파라미터 로드
        mapping_size = config.model.mapping_size
        scale = config.model.fourier_scale
        base_channels = config.model.hidden_dim
        kernel_size = config.model.kernel_size
        
        self.fourier = FourierFeatureMapping(mapping_size=mapping_size, scale=scale)
        
        # Input Channels: Intensity(1) + Fourier Features(2*M)
        enc_in_channels = 1 + (mapping_size * 2) 

        layers = []
        curr_dim = enc_in_channels
        
        for i in range(config.model.encoder_depth):
            out_dim = base_channels * (2 ** min(i // 2, 2))
            current_dilation = 2**i
            
            layers.append(nn.Sequential(
                nn.Conv1d(curr_dim, out_dim, 
                    kernel_size=kernel_size, 
                    padding=((kernel_size - 1) * current_dilation) // 2,
                    dilation=current_dilation,
                    bias=False),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool1d(2) 
            ))
            curr_dim = out_dim

        self.encoder = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # ParamSet 필드 구성을 동적으로 파악하여 출력 차원 결정
        # n_layers = Film 개수 (config.model.n_layers가 없으면 기본값 1)
        n = getattr(config.model, 'n_layers', 1) 
        self.output_dim = sum([n if f.metadata.get('mult') == 'n' else 1 for f in fields(ParamSet)])

        self.regressor = nn.Sequential(
            nn.Linear(curr_dim, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(base_channels, base_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(base_channels // 2, self.output_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2: x = x.unsqueeze(1)
        
        # Intensity Normalization (-9 ~ 0 -> 0 ~ 1)
        x_norm = (torch.clamp(x, min=-9.0, max=0.0) + 9.0) / 9.0
        
        # Fourier Features 결합
        fourier_feat = self.fourier(x_norm) # (B, 2M, Q)
        x_in = torch.cat([x_norm, fourier_feat], dim=1) # (B, 1+2M, Q)
        
        feat = self.encoder(x_in)
        feat = self.global_pool(feat).view(feat.size(0), -1)
        
        # Sigmoid로 0~1 사이 값 출력 (이후 물리 범위로 스케일링)
        return torch.sigmoid(self.regressor(feat))

class XRR_Fourier_PINN(nn.Module):
    def __init__(self, config: XRefineConfig, device='cpu'):
        super().__init__()
        self.config = config
        
        # 모델 및 시뮬레이터 초기화
        self.encoder = XRRPhysicsModel(config=config)
        self.simulator = AbelesMatrix(device=device)
        
        # q_grid 등록 (Buffer로 등록하여 state_dict에 포함되지만 grad는 없음)
        self.register_buffer('q_grid', config.q_grid)
        
    @property
    def device(self):
        return next(self.parameters()).device

    def _unnormalize(self, out: Tensor) -> ParamSet:
        """
        [Zero-Magic Number] 
        ParamSet 메타데이터를 활용하여 자동 슬라이싱 및 물리 스케일 변환
        """
        # Film Layer 개수 확인
        n = getattr(self.config.model, 'n_layers', 1)
        
        # 메타데이터 기반 텐서 폭(width) 계산
        widths = [n if f.metadata.get('mult') == 'n' else 1 for f in fields(ParamSet)]
        tensors = torch.split(out, widths, dim=1)
        
        params_dict = {}
        for f, val in zip(fields(ParamSet), tensors):
            # config.ranges에서 필드명으로 직접 범위 참조
            # 예: config.ranges.d_f = (10.0, 1000.0)
            p_min, p_max = getattr(self.config.ranges, f.name)
            
            # Linear Rescaling: 0~1 -> Min~Max
            unnorm_val = val * (p_max - p_min) + p_min
            
            params_dict[f.name] = unnorm_val

        return ParamSet(**params_dict)

    def validate_physics(self, p: ParamSet) -> Tensor:
        """물리적 제약 조건(Roughness < 50% Thickness) 페널티"""
        # 두께가 너무 얇은데 거칠기가 크면 물리적으로 불가능
        penalty_f = torch.mean(torch.relu(p.sig_f - 0.5 * p.d_f)) 
        penalty_ox = torch.mean(torch.relu(p.sig_ox - 0.5 * p.d_ox))
        return penalty_f + penalty_ox

    def forward(self, r_obs_log: Tensor) -> tuple[ParamSet, Tensor, Tensor]:
        """
        학습 시 호출되는 Main Forward
        """
        # NaN 처리
        if torch.isnan(r_obs_log).any():
            r_obs_log = torch.nan_to_num(r_obs_log, nan=-10.0)
            
        # 1. AI 추론
        out = self.encoder(r_obs_log)
        params = self._unnormalize(out.float())

        # 2. 물리 파라미터 조립
        thickness, roughness, sld_full = params.assemble_structure(self.config.ranges.substrate_sld)
        
        # 3. 시뮬레이션 준비
        batch_size = out.size(0)
        
        # Q-Grid 확장 및 2theta Offset 적용
        q_batch = self.q_grid.unsqueeze(0).expand(batch_size, -1) # (B, Q)
        
        # tth_offset 보정
        if hasattr(params, 'tth_offset'):
            # dq ≈ (4π/λ) * cos(th) * (dth/2)
            # 근사식: q_new = q_old + offset_in_q
            # 여기서는 간단히 q shift로 근사하거나, 정확히 다시 계산할 수 있음
            # 일단 간단한 Shift 적용:
            q_batch = q_batch + params.tth_offset * 0.01 # Scaling factor 주의
            
        # 4. 시뮬레이터 실행 (AbelesMatrix 클래스 호출)
        r_sim = self.simulator(
            q=q_batch,
            thickness=thickness,
            roughness=roughness,
            sld=sld_full,
            sample_length=params.L,
            beam_width=params.beam_width,
            wavelength=self.config.instrument.wavelength
        )
        
        # 5. 스케일 및 배경 보정
        # r_final = I0 * R + Bkg
        r_final = (params.i0 * r_sim) + torch.pow(10.0, params.bkg)
        r_final = torch.clamp(r_final, min=1e-11, max=2.0)

        # 6. 물리 페널티 계산
        phys_penalty = self.validate_physics(params)

        # PINN 구조상 (파라미터, 예측 그래프, 페널티)를 반환해야 함
        return params, r_final, phys_penalty

    def forward_with_params(self, params: ParamSet) -> tuple[ParamSet, torch.Tensor, torch.Tensor]:
        """
        Refinement 단계에서 사용 (외부 파라미터 주입)
        Device Safe하게 구현됨.
        """
        device = self.device
        params.to(device)

        thickness, roughness, sld_full = params.assemble_structure(self.config.ranges.substrate_sld)
        
        q_batch = self.q_grid.unsqueeze(0).expand(thickness.size(0), -1).to(device)
        
        # Offset 적용 (Optional)
        if hasattr(params, 'tth_offset'):
             q_batch = q_batch + params.tth_offset * 0.01

        r_sim = self.simulator(
            q=q_batch, 
            thickness=thickness, 
            roughness=roughness, 
            sld=sld_full,
            sample_length=params.L,
            beam_width=params.beam_width,
            wavelength=self.config.instrument.wavelength
        )
        
        r_final = (params.i0 * r_sim) + torch.pow(10.0, params.bkg)
        r_final = torch.clamp(r_final, min=1e-11, max=2.0)
        
        penalty = self.validate_physics(params)
        
        return params, r_final, penalty