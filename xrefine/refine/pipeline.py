"""
Refinement Pipeline (Clean One-Shot Output)
===========================================
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from copy import deepcopy

from xrefine.config import XRefineConfig
from xrefine.models import get_model
from xrefine.physics_utils import tth2q
from xrefine.core.structures import ParamSet
from xrefine.core.abeles import AbelesMatrix
from xrefine.core.smearing import abeles_constant_smearing
from xrefine.logger import logger

from xrefine.refine.config import RefinementConfig, StageConfig, DEFAULT_CONFIG
from xrefine.constants import LOG_MASK_VALUE, LOG_VALID_THRESHOLD
from xrefine.refine.optimizer import optimize_step

# -----------------------------------------------------------------------------
# Metric & Report Utils
# -----------------------------------------------------------------------------
def calculate_fom(model, params, log_r_obs, config, cutoff):
    device = config.device
    q_grid = config.q_grid.to(device)
    with torch.no_grad():
        t, r, s = params.assemble_structure()
        r_sim = model.simulator(q_grid, t, r, s) 
        i0 = params.i0 if hasattr(params, 'i0') else 1.0
        bkg = params.bkg if hasattr(params, 'bkg') else -10.0
        r_sim = r_sim * i0 + (10**bkg)
        log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
        
        mask = (log_r_obs > LOG_VALID_THRESHOLD) & (q_grid <= cutoff)
        diff = (log_r_sim - log_r_obs)[mask]
        mse = torch.mean(diff**2).item()
    return {"LogMSE": mse}

def report_stage_horizontal(stage_name, p_init, p_final, fom_init, fom_final):
    """
    Pandas DataFrame을 이용한 가로형 출력.
    모든 출력을 하나의 문자열로 합쳐서 로거 타임스탬프 간섭 방지.
    """
    imse, fmse = fom_init['LogMSE'], fom_final['LogMSE']
    imp = (imse - fmse) / imse * 100 if imse != 0 else 0
    
    # 1. 문자열 버퍼 생성
    lines = []
    lines.append("="*90)
    lines.append(f"Result: {stage_name} | LogMSE: {imse:.5f} -> {fmse:.5f} ({imp:+.1f}%)")
    lines.append("-" * 90)

    all_keys = list(p_final._params.keys())
    
    groups = {
        "Target": [k for k in all_keys if "Target_Film" in k],
        "Oxide":  [k for k in all_keys if "Top_Layer" in k],
        "System": [k for k in all_keys if "Target" not in k and "Top" not in k]
    }
    
    name_map = {
        'thickness': 'd', 'roughness': 'sig', 
        'sld': 'rho', 'sld_imag': 'iRho',
        'beam_width': 'Beam', 'tth_offset': 'Offset',
        'i0': 'I0', 'bkg': 'Bkg', 'L': 'Len', 'dq': 'Res'
    }

    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'right')

    for label, keys in groups.items():
        if not keys: continue
        
        # Sort
        order = {'thickness': 1, 'roughness': 2, 'sld': 3, 'sld_imag': 4}
        keys.sort(key=lambda x: order.get(x.split('.')[-1], 99))
        
        data = {}
        for k in keys:
            short = k.split('.')[-1]
            col = name_map.get(short, short)
            if "Substrate" in k: col = f"Sub_{col}"
            
            v_i = p_init._params[k].item()
            v_f = p_final._params[k].item()
            
            diff = v_f - v_i
            if abs(diff) > 1e-4:
                val_str = f"{v_f:.3f} ({diff:+.3f})"
            else:
                val_str = f"{v_f:.3f}"
            
            data[col] = [val_str]

        df = pd.DataFrame(data, index=[label])
        
        # DataFrame을 문자열로 변환 (index 포함)
        lines.append(df.to_string())
        lines.append("") # 공백 라인

    lines.append("="*90)
    
    # [Fix] 전체를 하나의 문자열로 합치고 맨 앞에 \n 추가
    full_log_str = "\n" + "\n".join(lines)
    logger.info(full_log_str)

# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------
def load_real_data(file_path: str, config: XRefineConfig) -> torch.Tensor:
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python', dtype=str)
        for col in df.columns:
            df[col] = df[col].str.replace('(', '', regex=False).str.replace(')', '', regex=False).astype(float)
        x_raw, y_raw = df.values[:, 0], df.values[:, 1]
        x_raw = tth2q(x_raw, config.instrument.wavelength)
        y_norm = y_raw / (y_raw.max() + 1e-12)
        log_y_raw = np.log10(np.clip(y_norm, 1e-11, 1.0))
        q_target = config.q_grid.cpu().numpy()
        log_y_interp = np.interp(q_target, x_raw, log_y_raw)
        log_y_interp[(q_target < x_raw.min()) | (q_target > x_raw.max())] = LOG_MASK_VALUE
        return torch.from_numpy(log_y_interp).float().view(1, -1)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def grid_search_thickness(model, p_init, log_r_obs, xrr_config, refine_config, stage_config):
    target_key = stage_config.grid_target_key 
    if target_key not in p_init._params: return 0.0
    device = xrr_config.device
    q_grid = xrr_config.q_grid.to(device)
    log_r_obs_gpu = log_r_obs.to(device)
    r_obs_linear = 10 ** log_r_obs_gpu.squeeze()
    peak_idx = int(torch.argmax(r_obs_linear).item())
    q_critical = q_grid[peak_idx].item()
    q_start = q_critical + 0.01
    q_end = refine_config.q_cutoff 
    mask = (log_r_obs_gpu > LOG_VALID_THRESHOLD) & (q_grid > q_start) & (q_grid < q_end)
    d_min, d_max = stage_config.grid_range
    d_grid = np.linspace(d_min, d_max, stage_config.grid_points)
    p_scan = p_init.clone()
    best_loss = 1e9
    best_d = p_init._params[target_key].item()
    for d_val in d_grid:
        p_scan._params[target_key].fill_(d_val)
        with torch.no_grad():
            if hasattr(model, 'forward_with_params'): _, r_sim, _ = model.forward_with_params(p_scan)
            else:
                t, r, s = p_scan.assemble_structure()
                r_sim = model.simulator(q_grid, t, r, s)
            r_sim_norm = r_sim / (r_sim.max() + 1e-12)
            log_r_sim = torch.log10(torch.clamp(r_sim_norm, min=1e-12))
            sim_pattern = log_r_sim[mask]
            vx = sim_pattern - sim_pattern.mean()
            vy = log_r_obs_gpu[mask] - log_r_obs_gpu[mask].mean()
            corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-9)
            loss = 1.0 - corr.item()
            if loss < best_loss:
                best_loss = loss
                best_d = d_val
    return best_d

def run_refinement(model, p_init, log_r_obs, xrr_config, refine_config=None):
    if refine_config is None: refine_config = DEFAULT_CONFIG
    p_current = p_init
    log_r_obs_gpu = log_r_obs.to(xrr_config.device)
    logger.info("Starting Refinement Pipeline...")

    for i, stage in enumerate(refine_config.stages):
        stage_name = stage.name
        p_start_stage = p_current.clone()
        fom_init = calculate_fom(model, p_current, log_r_obs_gpu, xrr_config, refine_config.q_cutoff)
        
        current_stage_config = deepcopy(stage)
        
        # [NEW] Smart-Lock Logic: Relative Bounds Processing
        # Config에 relative_bounds가 정의되어 있으면, 현재 값(p_current) 기준으로 temp_bounds를 생성합니다.
        if stage.relative_bounds:
            if not current_stage_config.temp_bounds:
                current_stage_config.temp_bounds = {}
            
            for key, margin in stage.relative_bounds.items():
                if key in p_current._params:
                    current_val = p_current._params[key].item()
                    
                    # Heuristic: 값의 성격에 따라 Ratio(%) 또는 Delta(Absolute) 적용
                    # Offset, Bkg, Res 등은 0에 가깝거나 음수이므로 Delta 방식이 안전함
                    if any(x in key for x in ['offset', 'bkg', 'dq', 'res', 'beam']):
                        # Absolute Delta Mode
                        p_min = current_val - margin
                        p_max = current_val + margin
                    else:
                        # Percentage Ratio Mode (e.g. 0.05 = 5%)
                        p_min = current_val * (1.0 - margin)
                        p_max = current_val * (1.0 + margin)
                        # 음수 값일 경우 min/max가 뒤집히는 것 방지
                        if p_min > p_max: p_min, p_max = p_max, p_min
                    
                    # 기존 temp_bounds를 덮어쓰지 않고, 더 좁은 범위(교집합)를 적용할 수도 있지만,
                    # 여기서는 Smart-Lock이 우선권을 가짐
                    current_stage_config.temp_bounds[key] = (p_min, p_max)
            
            stage_name += " (Smart-Lock)"

        # [Trust AI Logic] for Thickness Grid Search
        if refine_config.trust_ai and stage.use_grid_search:
            target_key = stage.grid_target_key
            current_d = 0.0
            if target_key in p_current._params:
                current_d = p_current._params[target_key].item()
            if current_d > 10.0:
                margin = refine_config.ai_trust_range
                new_min = max(20.0, current_d * (1.0 - margin))
                new_max = current_d * (1.0 + margin)
                current_stage_config.grid_range = (new_min, new_max)
                stage_name += f" (Trust Grid)"

        if current_stage_config.use_grid_search:
            best_val = grid_search_thickness(
                model, p_current, log_r_obs, xrr_config, refine_config, current_stage_config
            )
            p_current = p_current.clone()
            target_key = current_stage_config.grid_target_key
            if target_key in p_current._params:
                p_current._params[target_key].fill_(best_val)
        else:
            if current_stage_config.q_max is None:
                current_stage_config.q_max = refine_config.q_cutoff

            valid_keys = {k for k in current_stage_config.optimize_keys if k in p_current._params}
            if valid_keys:
                current_stage_config.optimize_keys = valid_keys
                p_current = optimize_step(
                    model, p_current, log_r_obs, xrr_config, current_stage_config, verbose=False
                )
        
        fom_final = calculate_fom(model, p_current, log_r_obs_gpu, xrr_config, refine_config.q_cutoff)
        report_stage_horizontal(stage_name, p_start_stage, p_current, fom_init, fom_final)

    logger.info("Refinement Complete.")
    return p_current

@torch.no_grad()
def run_real_inference(checkpoint_path, data_path, output_dir="./results", refine_config=None, q_cutoff=None):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config_data = ckpt.get('config')
    xrr_config = XRefineConfig(**config_data) if isinstance(config_data, dict) else config_data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xrr_config.device = device
    xrr_config.q_grid = xrr_config.q_grid.to(device)
    model = get_model(xrr_config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    simulator = AbelesMatrix(device=device)
    model.simulator = simulator 
    input_tensor = load_real_data(data_path, xrr_config)
    q_grid_np = xrr_config.q_grid.cpu().numpy()
    actual_cutoff = q_cutoff if q_cutoff else 0.45
    logger.info(f"Cutoff: Q > {actual_cutoff:.3f}")
    if refine_config is None: refine_config = DEFAULT_CONFIG
    refine_config.q_cutoff = actual_cutoff
    mask_indices = q_grid_np > actual_cutoff
    masked_input = input_tensor.clone()
    masked_input[0, mask_indices] = LOG_MASK_VALUE
    masked_input_gpu = masked_input.to(device)
    out = model(masked_input_gpu, fixed_values=None)
    p_ai = out[0] if isinstance(out, tuple) else out
    logger.info("AI Initial Guess Done.")
    
    p_refined = run_refinement(model, p_ai, masked_input, xrr_config, refine_config)
    
    def simulate_full(params):
        t, r, s = params.assemble_structure()
        L = params.L if hasattr(params, 'L') else None
        beam_width = params.beam_width if hasattr(params, 'beam_width') else None
        dq_val = params.dq if hasattr(params, 'dq') else torch.tensor(0.0, device=device)
        r_smeared = abeles_constant_smearing(
            q=xrr_config.q_grid.unsqueeze(0), thickness=t, roughness=r, sld=s,
            sample_length=L, beam_width=beam_width, dq=dq_val.view(1, 1), gauss_num=51, abeles_func=simulator 
        )
        i0_val = params.i0.item() if hasattr(params, 'i0') else 1.0
        bkg_val = params.bkg.item() if hasattr(params, 'bkg') else -10.0
        r_final = r_smeared * i0_val + (10 ** bkg_val)
        return r_final.view(1, -1)

    r_final = simulate_full(p_refined)
    r_ai_sim = simulate_full(p_ai)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_name = Path(data_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(figsize=(10, 6))
    q = q_grid_np
    y_obs = 10**masked_input[0].cpu().numpy()
    valid_mask = (masked_input[0].cpu().numpy() > LOG_VALID_THRESHOLD) & (q <= actual_cutoff)
    ax.semilogy(q[valid_mask], y_obs[valid_mask], 'ko', ms=3, alpha=0.4, label='Measured Data')
    ax.semilogy(q, r_ai_sim[0].cpu().numpy(), 'b--', lw=1, alpha=0.7, label='AI Init (Smeared)')
    ax.semilogy(q, r_final[0].cpu().numpy(), 'r-', lw=2, label='Refined (Smeared)')
    ax.axvline(x=actual_cutoff, color='gray', linestyle=':', label='Cutoff Limit')
    ax.set_xlabel("Q ($Å^{-1}$)", fontsize=12)
    ax.set_ylabel("Reflectivity", fontsize=12)
    ax.set_title(f"XRefine Result: {data_name}", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    if valid_mask.any():
        q_min, q_max = q[valid_mask].min(), q[valid_mask].max()
        ax.set_xlim(0, max(q_max * 1.1, 0.2))
        y_min_data = y_obs[valid_mask].min()
        ax.set_ylim(bottom=max(y_min_data * 0.1, 1e-10), top=2.0)
    plt.tight_layout()
    plt.savefig(output_path / f"{data_name}_{timestamp}_result.png", dpi=150)
    plt.close()
    final_fom = calculate_fom(model, p_refined, masked_input_gpu, xrr_config, actual_cutoff)
    param_dict = {k: v.item() for k, v in p_refined._params.items()}
    param_dict.update(final_fom)
    pd.DataFrame([param_dict]).to_csv(output_path / f"{data_name}_{timestamp}_params.csv", index=False)
    return p_refined

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to .dat file")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="./results")
    args = parser.parse_args()
    run_real_inference(args.checkpoint, args.data, args.out)