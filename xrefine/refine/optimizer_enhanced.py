"""
Enhanced Optimizer with Dynamic Physical Bounds (Differential Evolution)
=========================================================================
Optional enhancement: Dynamic bounds based on physical constraints.

This provides an alternative to the gradient-based optimizer_robust.py,
using differential evolution with dynamic constraint checking.

Usage:
    from xrefine.refine.optimizer_enhanced import optimize_step_enhanced
    p_refined = optimize_step_enhanced(model, p_init, log_r_obs, config, stage)
"""
import torch
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from typing import Tuple, Dict

from xrefine.core.structures import ParamSet
from xrefine.config import XRefineConfig
from xrefine.refine.config import StageConfig
from xrefine.refine.losses import LossFn
from xrefine.constants import LOG_VALID_THRESHOLD
from xrefine.logger import logger


def get_bounds_from_config(config: XRefineConfig, key: str) -> Tuple[float, float]:
    """Get bounds from config (same as original optimizer.py)"""
    sample = config.sample
    parts = key.split('.')
    try:
        if len(parts) == 1 and hasattr(sample, key):
            val = getattr(sample, key)
            if isinstance(val, list) and len(val) == 2: 
                return tuple(val)
        if parts[0] == "Substrate" and hasattr(sample.substrate, parts[1]):
            return tuple(getattr(sample.substrate, parts[1]))
        for layer in sample.layers:
            if layer.name == parts[0] and hasattr(layer, parts[1]):
                return tuple(getattr(layer, parts[1]))
    except: 
        pass
    return (0.0, 1.0)


def compute_dynamic_bounds(
    key: str, 
    params: Dict[str, torch.Tensor], 
    base_bounds: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Compute dynamic bounds based on physical constraints.
    
    For roughness: max = min(base_max, 0.5 * thickness)
    This ensures roughness never exceeds half the layer thickness.
    
    Args:
        key: Parameter key (e.g., 'Film.roughness')
        params: Current parameter dictionary
        base_bounds: Base bounds from config
        
    Returns:
        Adjusted bounds (min_val, max_val)
    """
    min_val, max_val = base_bounds
    
    # Only apply dynamic bounds to roughness parameters
    if key.endswith('.roughness'):
        layer_name = key.split('.')[0]
        thickness_key = f"{layer_name}.thickness"
        
        if thickness_key in params:
            thickness = params[thickness_key].item()
            
            # Physical constraint: roughness < 0.5 * thickness
            physical_max = thickness * 0.5
            max_val = min(max_val, physical_max)
            
            # Safety: ensure max >= min
            if max_val < min_val:
                max_val = max(min_val, thickness * 0.3)  # Relaxed to 30% if needed
    
    return (min_val, max_val)


def optimize_step_enhanced(
    model,
    p_init: ParamSet,
    log_r_obs: torch.Tensor,
    xrr_config: XRefineConfig,
    stage_config: StageConfig,
    verbose: bool = False,
    use_dynamic_bounds: bool = True
) -> ParamSet:
    """
    Enhanced optimization step with dynamic bounds using differential evolution.
    
    This is an alternative to optimizer.optimize_step() that adds:
    - Dynamic bounds based on physical constraints
    - Hard rejection of constraint violations
    - Soft penalties for minor violations
    
    Args:
        model: PINN model or simulator
        p_init: Initial parameters
        log_r_obs: Observed reflectivity (log10 scale)
        xrr_config: XRefine configuration
        stage_config: Stage-specific refinement config
        verbose: Print debug info
        use_dynamic_bounds: Enable dynamic constraint checking
        
    Returns:
        Optimized ParamSet
    """
    target_device = xrr_config.device
    q_grid = xrr_config.q_grid.to(target_device)
    log_r_obs_gpu = log_r_obs.to(target_device)

    # Masking
    q_min = stage_config.q_min or 0.0
    q_max = stage_config.q_max or xrr_config.q_grid.max().item()
    mask = (log_r_obs_gpu > LOG_VALID_THRESHOLD) & (q_grid > q_min) & (q_grid < q_max)

    # Param Setup
    optimize_keys = stage_config.optimize_keys
    fixed_params = {}
    opt_param_info = []
    opt_guess_norm = []

    for k, v_tensor in p_init._params.items():
        val_flat = v_tensor.detach().cpu().numpy().flatten()
        current_val = float(val_flat[0])

        if k in stage_config.fixed_params:
            fixed_val = stage_config.fixed_params[k]
            fixed_params[k] = torch.tensor([[fixed_val]], dtype=torch.float32, device=target_device)
            continue

        if k in optimize_keys:
            # Check for temporary bounds override first
            if k in stage_config.temp_bounds:
                p_min, p_max = stage_config.temp_bounds[k]
                if current_val > p_max or current_val < p_min:
                    current_val = np.clip(current_val, p_min, p_max)
            else:
                # Get base bounds from config
                p_min, p_max = get_bounds_from_config(xrr_config, k)
                
                # [ENHANCED] Apply dynamic bounds if enabled
                if use_dynamic_bounds:
                    # Build temp param dict for dynamic bound calculation
                    temp_params = {k: v for k, v in fixed_params.items()}
                    temp_params[k] = torch.tensor([[current_val]], dtype=torch.float32)
                    
                    p_min, p_max = compute_dynamic_bounds(k, temp_params, (p_min, p_max))
            
            # Clip current value to bounds
            current_val = np.clip(current_val, p_min, p_max)
            
            opt_param_info.append({'key': k, 'size': 1, 'min': p_min, 'max': p_max})
            norm_val = (current_val - p_min) / (p_max - p_min + 1e-9)
            opt_guess_norm.append(norm_val)
        else:
            fixed_params[k] = v_tensor.clone().detach()

    total_dim = len(opt_guess_norm)
    if total_dim == 0: 
        if verbose:
            logger.info("No parameters to optimize in this stage")
        return p_init

    loss_fn: LossFn = stage_config.loss_fn

    # Objective Function
    def objective(trial_vector):
        try:
            p_dict = {k: v for k, v in fixed_params.items()}
            cursor = 0
            for info in opt_param_info:
                phys_val = trial_vector[cursor] * (info['max'] - info['min']) + info['min']
                p_dict[info['key']] = torch.tensor([[phys_val]], dtype=torch.float32, device=target_device)
                cursor += 1

            # [ENHANCED] Check dynamic bounds violations (hard rejection)
            if use_dynamic_bounds:
                for info in opt_param_info:
                    if info['key'].endswith('.roughness'):
                        dynamic_min, dynamic_max = compute_dynamic_bounds(
                            info['key'], p_dict, (info['min'], info['max'])
                        )
                        current_r = p_dict[info['key']].item()
                        if current_r > dynamic_max:
                            return 1e6  # Hard boundary violation

            p_set = ParamSet(p_dict, p_init.layer_names, device=target_device)

            with torch.no_grad():
                if hasattr(model, 'forward_with_params'):
                    _, r_sim, _ = model.forward_with_params(p_set)
                else:
                    t, r, s = p_set.assemble_structure()
                    r_sim = model.simulator(q_grid, t, r, s)

                loss = loss_fn(r_sim, log_r_obs_gpu, mask)
                
                # [ENHANCED] Soft penalties for other constraints
                soft_penalty = 0.0
                for layer_name in p_init.layer_names:
                    t_key = f"{layer_name}.thickness"
                    if t_key in p_dict:
                        t_val = p_dict[t_key].item()
                        # Soft penalty for very thin layers (< 5 Angstrom)
                        if t_val < 5.0:
                            soft_penalty += 0.1 * (5.0 - t_val)
                
                return float((loss + soft_penalty).item())
        except Exception as e:
            if verbose:
                logger.warning(f"Objective evaluation failed: {e}")
            return 1e9

    # Latin Hypercube Sampling for initial population
    sampler = qmc.LatinHypercube(d=total_dim)
    init_pop = sampler.random(n=stage_config.popsize)
    init_pop[0] = np.array(opt_guess_norm)  # Include current values

    if verbose:
        logger.info(f"Starting differential evolution: {total_dim} parameters, "
                   f"popsize={stage_config.popsize}, maxiter={stage_config.maxiter}")

    result = differential_evolution(
        objective, 
        bounds=[(0, 1)] * total_dim, 
        strategy='best1bin',
        maxiter=stage_config.maxiter, 
        popsize=stage_config.popsize, 
        mutation=(0.5, 1.0), 
        recombination=0.7, 
        init=init_pop, 
        disp=verbose, 
        workers=1
    )

    # Build final parameter set
    final_dict = {k: v for k, v in fixed_params.items()}
    cursor = 0
    for info in opt_param_info:
        phys_val = result.x[cursor] * (info['max'] - info['min']) + info['min']
        final_dict[info['key']] = torch.tensor([[phys_val]], dtype=torch.float32, device=target_device)
        cursor += 1

    # [ENHANCED] Final validation with dynamic bounds
    if use_dynamic_bounds:
        for info in opt_param_info:
            if info['key'].endswith('.roughness'):
                dynamic_min, dynamic_max = compute_dynamic_bounds(
                    info['key'], final_dict, (info['min'], info['max'])
                )
                current_val = final_dict[info['key']].item()
                if current_val > dynamic_max:
                    # Clip to 95% of dynamic max for safety
                    clipped_val = dynamic_max * 0.95
                    final_dict[info['key']] = torch.tensor(
                        [[clipped_val]], dtype=torch.float32, device=target_device
                    )
                    if verbose:
                        logger.info(f"Clipped {info['key']}: {current_val:.2f} → {clipped_val:.2f} "
                                   f"(dynamic max: {dynamic_max:.2f})")

    if verbose:
        logger.info(f"Optimization complete: loss = {result.fun:.6f}")

    return ParamSet(final_dict, p_init.layer_names, device=target_device)
