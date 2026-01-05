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
            current_dilation = 2**i
            kernel_size = config.model.kernel_size
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
        target_device = device or config.device
        self.to(target_device)

        self.register_buffer('q_grid', config.q_grid)
        self.encoder = XRRPhysicsModel(config=config).to(device)
        self.simulator = simulator if simulator else Simulator(method='abeles', device=device)
        
    @property
    def device(self):
        """
        모델의 파라미터가 위치한 장치를 실시간으로 반환합니다.
        문자열 변수가 아니라 '실제 텐서의 주소'를 참조하므로 절대 꼬이지 않습니다.
        """
        return next(self.parameters()).device

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
        if torch.isnan(r_obs_log).any():
            r_obs_log = torch.nan_to_num(r_obs_log, nan=-10.0)
            
        # 1. AI 추론 (이미 r_obs_log가 GPU에 있으므로 out도 GPU)
        out = self.encoder(r_obs_log)
        params = self._unnormalize(out)
        

        batch_size = out.size(0)
        
        # --- 2. 물리 시뮬레이션 영역 ---
        with torch.amp.autocast(device_type='cuda', enabled=False):
            thickness = torch.cat([params.d_f, params.d_ox], dim=-1)
            roughness = torch.cat([params.sig_f, params.sig_ox, params.sig_ox], dim=-1)

            # A. Ambient/Substrate 장치 고정
            amb = torch.zeros(batch_size, 1, device=self.device)
            sub = torch.full((batch_size, 1), self.config.ranges.substrate_sld, device=self.device)
            sld_full = torch.cat([amb, params.rho_f, params.rho_ox, sub], dim=-1)

            # B. [중요] q_grid와 dq 장치 고정
            # self.q_grid가 buffer일지라도 .to(curr_device)를 명시하는 것이 가장 안전합니다.
            q_batch = self.q_grid.to(self.device).unsqueeze(0).expand(batch_size, -1)
            dq = torch.tensor([[self.config.instrument.res]], device=self.device)

            # C. 시뮬레이터 실행
            r_sim = self.simulator(
                q_batch,
                thickness,
                roughness,
                sld_full,
                sample_length=params.L,
                beam_width=params.beam_width,
                dq=dq
            )
            
            # r_sim이 혹시나 CPU로 튀었을 경우를 대비해 다시 한 번 장치 확인
            if r_sim.device != self.device:
                r_sim = r_sim.to(self.device)

        # 3. 최종 연산 (이제 params와 r_sim은 100% 같은 장치입니다)
        r_final = (params.i0 * r_sim) + torch.pow(10.0, params.bkg)
        r_final = torch.clamp(r_final, min=1e-11, max=2.0)

        phys_penalty = self.validate_physics(params)

        return params, r_final, phys_penalty
    
    def forward_with_params(self, params: ParamSet) -> tuple[ParamSet, torch.Tensor, torch.Tensor]:
        """직접 파라미터를 받아 물리 시뮬레이션을 수행 (장치 일관성 확보 버전)"""
        batch_size = params.d_f.size(0)
        device = params.d_f.device # [핵심] 입력 데이터가 위치한 장치 파악

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # 1. Thickness & Roughness 조립
            thickness = torch.cat([params.d_f, params.d_ox], dim=-1)
            roughness = torch.cat([params.sig_f, params.sig_ox, params.sig_ox], dim=-1)

            # 2. SLD 구성 (Ambient 및 Substrate 장치 명시)
            # amb와 sub를 생성할 때 device를 반드시 지정해야 합니다.
            amb = torch.zeros(batch_size, 1, device=device) # CPU -> GPU
            sub = torch.full((batch_size, 1), self.config.ranges.substrate_sld, device=device) # CPU -> GPU
            
            # 이제 모든 텐서가 동일한 장치에 있으므로 cat 연산이 성공합니다.
            sld_full = torch.cat([amb, params.rho_f, params.rho_ox, sub], dim=-1)

            # 3. 시뮬레이터 입력 데이터 장치 확인
            q_batch = self.q_grid.to(device).unsqueeze(0).expand(batch_size, -1)
            dq = torch.tensor([[self.config.instrument.res]], device=device)

            r_recon = self.simulator(
                q=q_batch,
                thickness=thickness,
                roughness=roughness,
                sld=sld_full,
                sample_length=params.L,
                beam_width=params.beam_width,
                dq=dq
            )

        # 4. Scaling 및 후처리
        r_final = (params.i0 * r_recon) + torch.pow(10.0, params.bkg)
        r_final = torch.clamp(r_final, min=1e-11, max=2.0)
        penalty = self.validate_physics(params)

        return params, r_final, penalty