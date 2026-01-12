
import torch
from xrefine.config import XRefineConfig
from xrefine.models.vit import XRR_ViT_PINN
from xrefine.core.structures import ParamSet
from xrefine.train.utils import XRRWeightedLoss

def test_model_flow():
    print("🚀 Testing Model Flow...")
    
    # 1. Config & Model Init
    config = XRefineConfig()
    config.model.model_type = 'vit'
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {config.device}")
    
    model = XRR_ViT_PINN(config).to(config.device)
    print("✅ Model initialized successfully.")
    
    # 2. Check Learnable Fields
    print(f"Learnable Fields: {model.learnable_fields}")
    print(f"Output Dim: {model.output_dim}")
    
    # 3. Create Dummy Input
    batch_size = 4
    q_len = config.q_len
    r_obs_log = torch.randn(batch_size, q_len).to(config.device)
    
    # 4. Create Dummy Fixed Params (GT)
    # ParamSet의 모든 필드를 채워야 함 (Generator 역할)
    # 여기서는 임의로 생성
    fixed_values = ParamSet(
        d_f=torch.rand(batch_size, 1).to(config.device),
        sig_f=torch.rand(batch_size, 1).to(config.device),
        rho_f=torch.rand(batch_size, 1).to(config.device),
        d_ox=torch.rand(batch_size, 1).to(config.device),
        sig_ox=torch.rand(batch_size, 1).to(config.device),
        rho_ox=torch.rand(batch_size, 1).to(config.device),
        L=torch.rand(batch_size, 1).to(config.device),
        i0=torch.rand(batch_size, 1).to(config.device),
        bkg=torch.rand(batch_size, 1).to(config.device),
        beam_width=torch.rand(batch_size, 1).to(config.device),
        dq=torch.rand(batch_size, 1).to(config.device),
        tth_offset=torch.rand(batch_size, 1).to(config.device),
        rho_sub=torch.full((batch_size, 1), 20.0).to(config.device), # Fixed
        sig_sub=torch.full((batch_size, 1), 3.0).to(config.device)   # Fixed
    )
    
    # 5. Forward Pass
    print("Running Forward Pass...")
    params_pred, r_recon, penalty = model(r_obs_log, fixed_values=fixed_values)
    print("✅ Forward pass successful.")
    
    # 6. Check if Fixed Params are merged correctly
    print("Checking Fixed Params merge...")
    assert torch.allclose(params_pred.rho_sub, fixed_values.rho_sub), "❌ rho_sub not merged correctly!"
    print("✅ rho_sub merged correctly.")
    
    # 7. Loss Calculation Check (Fixed param should be ignored)
    print("Checking Loss Calculation...")
    criterion = XRRWeightedLoss(config.q_grid).to(config.device)
    
    # Dummy Target (p_true)
    p_true = fixed_values # GT를 그대로 사용
    
    loss_main, _, _, _, loss_param = criterion(
        params_pred, r_recon, r_obs_log, p_true, 
        config, curr_sup_w=1.0, phys_penalty=penalty
    )
    print(f"✅ Loss calculation successful. Loss Param: {loss_param.item()}")
    
    print("🎉 All Tests Passed!")

if __name__ == "__main__":
    test_model_flow()
