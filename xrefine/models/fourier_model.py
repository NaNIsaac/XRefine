import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import fields

from xrefine.core.interface import Simulator
from xrefine.core.corrections import DifferentiableFootprint
from xrefine.core.structures import ParamSet
from xrefine.config import XRefineConfig

class FourierFeatureMapping(nn.Module):
    """
    입력 q 신호를 고주파 영역으로 투영하여 
    미세한 주파수 변화(두께 변화)를 인코딩함.
    """
    def __init__(self, mapping_size: int = 64, scale: float = 40.0):
        super().__init__()
        # scale이 높을수록 미세한 간섭 무늬 변화에 예민해짐
        self.register_buffer('B', torch.randn(1, mapping_size) * scale)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 1, Q) -> Fourier Projection
        x_proj = (x.transpose(1, 2) @ self.B).transpose(1, 2)
        x_proj = x_proj * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)

class XRRPhysicsModel(nn.Module):
    """
    Dilated CNN Encoder: 넓은 수용장으로 Fringe 주기를 파악
    """
    def __init__(self, config: XRefineConfig):
        super().__init__()
        self.fourier = FourierFeatureMapping(
            mapping_size=config.model.mapping_size, 
            scale=config.model.fourier_scale
        )
        enc_in_channels = 1 + (config.model.mapping_size * 2) 

        layers = []
        curr_dim = enc_in_channels
        for i in range(config.model.encoder_depth):
            out_dim = config.model.n_channels * (2 ** min(i // 2, 2))
            layers.append(nn.Sequential(
                nn.Conv1d(curr_dim, out_dim, 
                    kernel_size=config.model.kernel_size, 
                    padding=((config.model.kernel_size-1) * config.model.dilation) // 2,
                    dilation=config.model.dilation,
                    bias=False),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool1d(2)
            ))
            curr_dim = out_dim

        self.encoder = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # ParamSet 필드 구성을 동적으로 파악하여 출력 차원 결정
        n = config.model.n_layers
        self.output_dim = sum([n if f.metadata.get('mult') == 'n' else 1 for f in fields(ParamSet)])

        self.regressor = nn.Sequential(
            nn.Linear(curr_dim, config.model.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(config.model.hidden_dim, config.model.hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(config.model.hidden_dim // 2, self.output_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2: x = x.unsqueeze(1)
        # Intensity Normalization (-9 ~ 0 -> 0 ~ 1)
        x = (torch.clamp(x, min=-9.0, max=0.0) + 9.0) / 9.0
        
        x = torch.cat([x, self.fourier(x)], dim=1)
        feat = self.encoder(x)
        feat = self.global_pool(feat).view(feat.size(0), -1)
        return torch.sigmoid(self.regressor(feat))

class XRR_Fourier_PINN(nn.Module):
    def __init__(self, config: XRefineConfig, simulator: nn.Module | None = None, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.register_buffer('q_grid', config.q_grid)

        self.encoder = XRRPhysicsModel(config=config).to(device)
        self.simulator = simulator if simulator else Simulator(method='abeles', device=device)


    def _unnormalize(self, out: Tensor) -> ParamSet:
        """
        [Zero-Magic Number] 
        ParamSet 메타데이터를 활용하여 자동 슬라이싱 및 물리 스케일 변환
        """
        n = self.config.model.n_layers
        
        # 1. 메타데이터 기반 텐서 폭(width) 계산
        widths = [n if f.metadata.get('mult') == 'n' else 1 for f in fields(ParamSet)]
        tensors = torch.split(out, widths, dim=1)
        
        params_dict = {}
        for f, val in zip(fields(ParamSet), tensors):
            # config.ranges에서 필드명으로 직접 범위 참조
            p_min, p_max = getattr(self.config.ranges, f.name)
            
            # Linear Rescaling
            unnorm_val = val * (p_max - p_min) + p_min
            
            # Log 필드(예: bkg)의 경우 최종 결과 리턴 시에는 unnormalize 상태(log10) 유지
            # 시뮬레이션 직전에만 pow(10, x) 적용하는 것이 디버깅에 유리함
            params_dict[f.name] = unnorm_val

        return ParamSet(**params_dict)

    def validate_physics(self, p: ParamSet) -> Tensor:
        """물리적 제약 조건(Roughness < 50% Thickness) 페널티"""
        penalty_f = torch.mean(torch.relu(p.sig_f - 0.5 * p.d_f)) 
        penalty_ox = torch.mean(torch.relu(p.sig_ox - 0.5 * p.d_ox))
        return penalty_f + penalty_ox

    def forward(self, r_obs_log: Tensor) -> tuple[ParamSet, Tensor, Tensor]:
        """
        PINN Forward Pipeline:
        Encoder -> Unnormalize -> Physics Simulation -> Correction -> Scaling
        """
        if torch.isnan(r_obs_log).any():
            r_obs_log = torch.nan_to_num(r_obs_log, nan=-10.0)
            
        # 1. AI 추론 및 구조화된 파라미터 복원
        out = self.encoder(r_obs_log)
        params = self._unnormalize(out)
        
        batch_size = out.size(0)
        
        # --- 2. 물리 시뮬레이션 영역 ---
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # A. 텐서 조립 (Ambient-Film-Oxide-Substrate)
            thickness = torch.cat([params.d_f, params.d_ox], dim=-1)
            roughness = torch.cat([params.sig_f, params.sig_ox, params.sig_ox], dim=-1)

            amb = torch.zeros(batch_size, 1, device=self.device)
            sub = torch.full((batch_size, 1), self.config.ranges.substrate_sld, device=self.device)
            sld_full = torch.cat([amb, params.rho_f, params.rho_ox, sub], dim=-1)

            # B. 물리 법칙 적용 (Abeles Matrix)
            q_batch = self.q_grid.unsqueeze(0).expand(batch_size, -1)
            dq = torch.tensor([[self.config.instrument.res]], device=self.device)

            r_sim = self.simulator(
                q_batch,
                thickness,
                roughness,
                sld_full,
                sample_length=params.L,
                beam_width=params.beam_width,
                dq=dq)
            

        r_final = (params.i0 * r_sim) + torch.pow(10.0, params.bkg)
        r_final = torch.clamp(r_final, min=1e-11, max=2.0)

        # 4. 페널티 계산
        phys_penalty = self.validate_physics(params)

        return params, r_final, phys_penalty
    
    def forward_with_params(self, params: ParamSet) -> tuple[ParamSet, Tensor, Tensor]:
        """
        [Physics Core] 파라미터 세트를 직접 입력받아 물리 법칙에 따른 반사율 곡선을 생성합니다.
        인코더를 거치지 않으므로 Local Refinement(미세 조정) 및 디버깅에 사용됩니다.
        """
        # 1. 배치 사이즈 확인 (파라미터 텐서 중 하나를 기준으로 함)
        batch_size = params.d_f.size(0)
        
        # 2. 물리 시뮬레이션 영역 (Abeles Matrix + Integrated Footprint)
        # 고정밀 최적화를 위해 이 구간은 보통 Autocast를 비활성화(float32)하여 수행합니다.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # A. 다층 구조 텐서 조립 (Ambient-Film-Oxide-Substrate)
            # thickness: (B, 2), roughness: (B, 3)
            thickness = torch.cat([params.d_f, params.d_ox], dim=-1)
            roughness = torch.cat([params.sig_f, params.sig_ox, params.sig_ox], dim=-1)

            # B. SLD 구성 (Ambient SLD는 항상 0, Substrate SLD는 고정)
            amb = torch.zeros(batch_size, 1, device=self.device)
            sub = torch.full((batch_size, 1), self.config.ranges.substrate_sld, device=self.device)
            sld_full = torch.cat([amb, params.rho_f, params.rho_ox, sub], dim=-1)

            # C. 통합 시뮬레이터 호출
            # q_grid를 배치 크기에 맞게 확장
            q_batch = self.q_grid.unsqueeze(0).expand(batch_size, -1)
            dq = torch.tensor([[self.config.instrument.res]], device=self.device)

            # 풋프린트(L, beam_width)가 포함된 통합 물리 엔진 실행
            r_recon = self.simulator(
                q=q_batch,
                thickness=thickness,
                roughness=roughness,
                sld=sld_full,
                sample_length=params.L,          # [필수] 기하학 보정 인자 전달
                beam_width=params.beam_width,    # [필수] 기하학 보정 인자 전달
                dq=dq
            )

        # 3. 측정 환경 합성 (Scaling & Background)
        # R_final = I0 * R_physics + 10^Background
        r_final = (params.i0 * r_recon) + torch.pow(10.0, params.bkg)
        
        # 수치적 안정성을 위한 클램핑 (10^0 이상의 강도는 물리적으로 불가능)
        r_final = torch.clamp(r_final, min=1e-11, max=2.0)

        # 4. 물리적 제약 조건 페널티 계산
        penalty = self.validate_physics(params)

        return params, r_final, penalty