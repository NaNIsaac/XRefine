"""
Refinement Optimizer (Robust Gradient Descent)
==============================================
Update: Implements 'Adaptive Gradient Clipping'.
- Early Stages (Scale, SLD): Loose clipping (Allow big moves).
- Later Stages (Thickness, Shape): Strict clipping (Prevent explosion).
"""
import torch
import torch.optim as optim
from xrefine.logger import logger
from xrefine.constants import LOG_VALID_THRESHOLD

def optimize_step(model, p_init, log_r_obs, xrr_config, stage_config, verbose=False):
    # 1. Setup Data & Config
    device = xrr_config.device
    log_r_obs_gpu = log_r_obs.to(device)
    q_grid = xrr_config.q_grid.to(device)

    q_max_limit = stage_config.q_max if stage_config.q_max else 100.0
    mask = (log_r_obs_gpu > LOG_VALID_THRESHOLD) & (q_grid <= q_max_limit)
    if stage_config.q_min:
        mask &= (q_grid >= stage_config.q_min)

    # 2. Collect Parameters
    p_current = p_init.clone()
    params_to_opt = []
    param_names = []

    for key, tensor in p_current._params.items():
        if key in stage_config.optimize_keys:
            tensor.requires_grad = True
            params_to_opt.append(tensor)
            param_names.append(key)
        else:
            tensor.requires_grad = False
            
    if not params_to_opt:
        if verbose: logger.warning(f"Stage {stage_config.name}: No parameters to optimize.")
        return p_current

    # 3. Select Optimizer & Clipping Strategy
    use_lbfgs = "Final" in stage_config.name or "LBFGS" in stage_config.name
    
    # [ADAPTIVE STRATEGY]
    # 두께(Thickness)나 형태(Shape)를 건드리는 후반부엔 엄격하게(1.0)
    # 스케일(Scale)이나 임계각(Anchor)을 잡는 초반부엔 널널하게(10.0 or None)
    is_sensitive_stage = any(x in stage_config.name for x in ["Thickness", "Shape", "Polish"])
    
    if is_sensitive_stage:
        clip_norm = 1.0   # 꽉 조임 (두께 폭발 방지)
    else:
        clip_norm = 20.0  # 널널함 (I0, SLD 등은 크게 움직여야 함)

    if use_lbfgs:
        optimizer = optim.LBFGS(
            params_to_opt, 
            lr=1.0, 
            max_iter=20, 
            history_size=10, 
            line_search_fn="strong_wolfe"
        )
        num_steps = max(1, stage_config.maxiter // 4)
    else:
        # Adam LR도 스테이지별 미세 조정
        base_lr = 0.005 if is_sensitive_stage else 0.02 # 초반엔 더 빠르게(0.02)
        optimizer = optim.Adam(params_to_opt, lr=base_lr)
        num_steps = stage_config.maxiter

    loss_fn = stage_config.loss_fn
    bounds_map = stage_config.temp_bounds or {}

    # 4. Optimization Loop
    best_loss = float('inf')
    p_best = p_current.clone()

    def closure():
        optimizer.zero_grad()
        
        t, r, s = p_current.assemble_structure()
        r_sim = model.simulator(q_grid, t, r, s)
        
        i0 = p_current.i0 if hasattr(p_current, 'i0') else 1.0
        bkg = p_current.bkg if hasattr(p_current, 'bkg') else -10.0
        r_sim = r_sim * i0 + (10**bkg)
        
        loss = loss_fn(r_sim, log_r_obs_gpu, mask)
        
        if loss.requires_grad:
            loss.backward()
            
            # [ADAPTIVE CLIPPING applied here]
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(params_to_opt, max_norm=clip_norm)
            
        return loss

    for step in range(num_steps):
        try:
            if use_lbfgs:
                optimizer.step(closure)
            else:
                closure()
                optimizer.step()
            
            # Physical Clamping
            with torch.no_grad():
                for name, tensor in zip(param_names, params_to_opt):
                    if name in bounds_map:
                        min_v, max_v = bounds_map[name]
                        tensor.clamp_(min=min_v, max=max_v)

            # Keep Best
            with torch.no_grad():
                current_loss = closure().item()
            
            if current_loss < best_loss:
                best_loss = current_loss
                p_best = p_current.clone()
                
        except RuntimeError as e:
            logger.warning(f"Optimization interrupted at step {step}: {e}")
            break

    return p_best