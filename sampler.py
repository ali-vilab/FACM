"""
FLOW-ANCHORED CONSISTENCY MODELS
Copyright (c) 2024 The FACM Authors. All Rights Reserved.
"""


import torch
import math
import numpy as np


def apply_timestep_shift(t_steps, timestep_shift):
    """
    Apply timestep shift transformation to time steps
    
    Args:
        t_steps: Original time steps
        timestep_shift: Shift parameter
        
    Returns:
        Transformed time steps
    """
    if timestep_shift <= 0:
        return t_steps
        
    def compute_tm(t_n, shift):
        numerator = shift * t_n
        denominator = 1 + (shift - 1) * t_n
        return numerator / denominator
    
    return torch.tensor([compute_tm(t_n, timestep_shift) for t_n in t_steps], 
                       dtype=torch.float64, device=t_steps.device)


def prepare_model_input(x_cur, y, y_null, cfg_scale, guidance_low, guidance_high, 
                       t_cur):
    """
    Prepare model input for conditional/unconditional generation
    
    Args:
        x_cur: Current state
        y: Conditional labels
        y_null: Null/unconditional labels
        cfg_scale: Classifier-free guidance scale
        guidance_low/high: Guidance boundaries
        t_cur: Current time
        
    Returns:
        Tuple of (model_input, y_cur, kwargs)
    """
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y
    
    kwargs = dict(y=y_cur)
    return model_input, y_cur, kwargs


def apply_classifier_free_guidance(d_cur, cfg_scale, guidance_low, guidance_high, 
                                  t_cur):
    """
    Apply classifier-free guidance to model output
    
    Args:
        d_cur: Model output
        cfg_scale: Guidance scale
        guidance_low/high: Guidance boundaries
        t_cur: Current time
        
    Returns:
        Guided output
    """
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
    return d_cur


@torch.no_grad()
def euler_sampler(
    model,
    latents,
    y,
    num_steps=20,
    heun=False,
    cfg_scale=1.0,
    guidance_low=0.125,
    guidance_high=1.0,
    timestep_shift=0.,
):
    """
    Euler sampler for diffusion models
    
    Args:
        model: The diffusion model
        latents: Initial noise latents
        y: Conditional labels
        num_steps: Number of sampling steps
        heun: Whether to use Heun's method (2nd order)
        cfg_scale: Classifier-free guidance scale
        guidance_low/high: Time range for applying guidance
        timestep_shift: Timestep shift parameter
        expanded_t: Whether to use expanded time input
        
    Returns:
        Generated samples
    """
    # Setup conditioning
    y_null = torch.tensor([1000] * y.size(0), device=y.device)
    
    _dtype = latents.dtype
    device = latents.device
    
    x_next = latents.to(torch.float64)

    # Create time schedule
    t_steps = torch.linspace(0, 1, num_steps + 1, dtype=torch.float64)
    t_steps = apply_timestep_shift(t_steps, timestep_shift)
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        # Prepare model input
        model_input, y_cur, kwargs = prepare_model_input(
            x_cur, y, y_null, cfg_scale, guidance_low, guidance_high, 
            t_cur
        )
        
        # Prepare time input
        time_input = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_cur
        r_input = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_cur
        
        # Forward pass
        d_cur = model(
            model_input.to(dtype=_dtype), 
            time_input.to(dtype=_dtype), 
            r_input.to(dtype=_dtype), 
            **kwargs
        ).to(torch.float64)
        
        # Apply guidance
        d_cur = apply_classifier_free_guidance(
            d_cur, cfg_scale, guidance_low, guidance_high, t_cur
        )
        
        # Euler step
        x_next = x_cur + (t_next - t_cur) * d_cur
        
        # Heun's method (2nd order correction)
        if heun and (i < num_steps - 1):
            model_input, y_cur, kwargs = prepare_model_input(
                x_next, y, y_null, cfg_scale, guidance_low, guidance_high, 
                t_next
            )
            
            time_input = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_next
            r_input = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_next
            
            d_prime = model(
                model_input.to(dtype=_dtype), 
                time_input.to(dtype=_dtype), 
                r_input.to(dtype=_dtype), 
                **kwargs
            ).to(torch.float64)
            
            d_prime = apply_classifier_free_guidance(
                d_prime, cfg_scale, guidance_low, guidance_high, t_next
            )
            
            # 2nd order correction
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


@torch.no_grad()
def consistency_model_sampler(
    model,
    latents,
    y,
    num_steps=1,
    cfg_scale=1.0,
    guidance_low=0.1,
    guidance_high=1.0,
):
    """
    Consistency model sampler
    
    Args:
        model: The consistency model
        latents: Initial noise latents
        y: Conditional labels
        num_steps: Number of sampling steps
        cfg_scale: Classifier-free guidance scale
        guidance_low/high: Time range for applying guidance
        
    Returns:
        Generated samples
    """
    # Setup conditioning
    y_null = torch.tensor([1000] * y.size(0), device=y.device)
    
    _dtype = latents.dtype
    device = latents.device
    
    # Create time schedule
    t_steps = torch.linspace(0, 1, num_steps + 1, dtype=torch.float64)
    
    x_next = latents.to(torch.float64)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next.to(torch.float64)
        
        # Prepare model input
        model_input, y_cur, kwargs = prepare_model_input(
            x_cur, y, y_null, cfg_scale, guidance_low, guidance_high, 
            t_cur
        )
        
        # Prepare time input for consistency model
        time_input = torch.ones(model_input.size(0), device=device, dtype=torch.float64) * t_cur
        r_input = torch.ones(model_input.size(0), device=device, dtype=torch.float64)
        
        # Forward pass
        d_cur = model(
            model_input.to(dtype=_dtype), 
            time_input.to(dtype=_dtype), 
            r_input.to(dtype=_dtype), 
            **kwargs
        ).to(torch.float64)
        
        # Apply guidance
        d_cur = apply_classifier_free_guidance(
            d_cur, cfg_scale, guidance_low, guidance_high, t_cur
        )
        
        # Consistency model step
        x_end = x_cur + (t_steps[-1] - t_cur) * d_cur
        noise = torch.randn_like(x_end)
        x_next = t_next * x_end + (1 - t_next) * noise 

    return x_end