"""
FLOW-ANCHORED CONSISTENCY MODELS
Copyright (c) 2024 The FACM Authors. All Rights Reserved.
Including reproduction of:
- Meanflow (https://arxiv.org/abs/2505.13447)
- sCM (https://arxiv.org/abs/2410.11081)
"""


import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm

from utils import mean_flat


class FACMLoss:
    """Flow-Anchored Consistency Model Loss for mixed-objective training"""
    
    def sample_t(self, device, size=1, type='con', args=None):
        """Sample time values according to different distributions"""
        if type == 'default':
            P_mean, P_std = -args.mean, args.std
            sigma = torch.randn(size).reshape(-1, 1, 1, 1)
            sigma = (sigma * P_std + P_mean).exp()
            samples = torch.arctan(sigma) * (2.0/np.pi)
        elif type == 'log':
            mu, sigma = -args.mean, args.std
            normal_samples = torch.randn(size, 1, device=device) * sigma + mu
            samples = 1 / (1 + torch.exp(-normal_samples))
        else:
            raise ValueError(f"Invalid sample type: {type}")
        return 1 - samples.to(device).reshape(-1, 1, 1, 1)
    
    @torch.no_grad()
    def get_velocity(self, x_t, target, t, y, model, args):
        """Get velocity from teacher model or ground truth with CFG"""
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
        y_null[t.flatten() < args.glow] = y[t.flatten() < args.glow]
        y_null[t.flatten() > args.ghigh] = y[t.flatten() > args.ghigh]
        
        if args.cfgw > 1.0:
            cuda_state = torch.cuda.get_rng_state()
            v_cond = model(x_t, t.reshape(-1), t.reshape(-1), y) if args.distill else target
            torch.cuda.set_rng_state(cuda_state)
            v_uncond = model(x_t, t.reshape(-1), t.reshape(-1), y_null)
            velocity = v_uncond + args.cfgw * (v_cond - v_uncond)
        else:
            cuda_state = torch.cuda.get_rng_state()
            velocity = model(x_t, t.reshape(-1), t.reshape(-1), y) if args.distill else target
        
        return velocity, cuda_state

    def norm_l2_loss(self, pred, target, p=0.5, c=1e-3):
        """Norm L2 loss with outlier resistance"""
        e = torch.mean((pred - target) ** 2, dim=(1, 2, 3), keepdim=False)
        loss = e / (e + c).pow(p).detach()
        return loss

    def flow_matching_loss(self, pred, target):
        """Flow Matching loss: MSE + cosine similarity"""
        mse_loss = mean_flat((pred - target) ** 2)
        cos_loss = mean_flat(1 - F.cosine_similarity(pred, target, dim=1))
        return mse_loss + cos_loss

    def interpolant(self, t):
        """Linear interpolant parameters for rectified flow"""
        alpha_t = t
        sigma_t = 1 - t
        d_alpha_t = 1
        d_sigma_t = -1
        t_end = 1.0
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t, t_end

    def __call__(self, accelerator, model, compiled_model, ema, images, iters,
                model_kwargs=None, args=None, freezed_teacher=None):
        """FACM mixed-objective training loss computation"""
        if model_kwargs is None:
            model_kwargs = {}

        # Setup model
        unwrapped_model = model.module if hasattr(model, 'module') else model
        
        b, c, h, w = images.shape

        # Sample noise and time
        noise = torch.randn_like(images)
        t = self.sample_t(images.device, size=b, type=args.t_type, args=args)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t, t_end = self.interpolant(t)

        # Create noisy trajectory point
        x_t = alpha_t * images + sigma_t * noise  
        target = d_alpha_t * images + d_sigma_t * noise

        # Get flow velocity from teacher/ground truth
        reference_model = compiled_model if freezed_teacher is None else freezed_teacher
        v, cuda_state = self.get_velocity(
            x_t, target, t, model_kwargs['y'], reference_model, args
        )

        if unwrapped_model.r_embedder is not None:
            t_fm = t  # Auxiliary time condition
            r = t_end * torch.ones_like(t).reshape(-1)
        else:
            t_fm = 2 - t if t_end == 1.0 else -t
            r = None  # Expanded time interval
        
        # Flow Matching (FM) Loss - The Anchor
        torch.cuda.set_rng_state(cuda_state)
        F_fm = compiled_model(x_t, t_fm.reshape(-1), t_fm.reshape(-1), **model_kwargs)
        fm_loss = self.flow_matching_loss(F_fm, v) 

        # Consistency Model (CM) Loss - The Accelerator
        def model_wrapper(x_input, t_input):
            torch.cuda.set_rng_state(cuda_state)
            output = unwrapped_model(x_input, t_input.reshape(-1), r, **model_kwargs) 
            return output

        # Compute average velocity via JVP
        v_x = v
        v_t = torch.ones_like(t)
        unwrapped_model.disable_fused_attn() 
        F_avg, F_avg_grad = torch.func.jvp(model_wrapper, (x_t, t), (v_x, v_t))
        unwrapped_model.enable_fused_attn() 
        
        F_avg_grad = F_avg_grad.detach()
        F_avg_sg = F_avg.detach()
        
        # Compute average velocity target
        v_bar = v + (t_end - t) * F_avg_grad
        g = F_avg_sg - v_bar
        
        # Compute interpolated target with relaxation
        alpha = 1 - alpha_t ** args.p
        target = F_avg_sg - alpha * g.clamp(min=-1, max=1)

        # Weight CM loss by time
        beta = torch.cos(alpha_t * np.pi / 2).flatten()
        cm_loss = self.norm_l2_loss(F_avg, target) * beta.flatten()

        # Combined FACM loss
        total_loss = cm_loss.mean() + fm_loss.mean()
        
        return total_loss, cm_loss.mean(), fm_loss.mean()


class MeanFlowLoss(FACMLoss):
    """Mean Flow Loss accroding to (https://arxiv.org/abs/2505.13447)"""

    def __call__(self, accelerator, model, compiled_model, ema, images, iters,
                model_kwargs=None, args=None, freezed_teacher=None):

        if model_kwargs is None:
            model_kwargs = {}

        # Setup model
        unwrapped_model = model.module if hasattr(model, 'module') else model
        
        b, c, h, w = images.shape

        # Sample noise and time
        noise = torch.randn_like(images)
        t_ = self.sample_t(images.device, size=b, type=args.t_type, args=args)
        r_ = self.sample_t(images.device, size=b, type=args.t_type, args=args)
        
        # p% of samples r_ -> t_
        num_r2t = int(getattr(args, 'r2t_ratio', 0.) * b)
        indices = torch.randperm(b, device=images.device)[:num_r2t]
        t_[indices] = r_[indices]
        t = torch.minimum(t_, r_)
        r = torch.maximum(t_, r_)

        alpha_t, sigma_t, d_alpha_t, d_sigma_t, t_end = self.interpolant(t)

        # Create noisy trajectory point
        x_t = alpha_t * images + sigma_t * noise  
        target = d_alpha_t * images + d_sigma_t * noise

        # Get dxt/dt from teacher/ground truth
        reference_model = compiled_model if freezed_teacher is None else freezed_teacher
        v, cuda_state = self.get_velocity(
            x_t, target, t, model_kwargs['y'], reference_model, args
        )

        # Define model wrapper function
        def model_wrapper(x_input, t_input, r_input):
            torch.cuda.set_rng_state(cuda_state)
            output = unwrapped_model(x_input, t_input.reshape(-1), r_input.reshape(-1), **model_kwargs) 
            return output

        # Compute average velocity via JVP
        v_x = v
        v_t = torch.ones_like(t)
        v_r = torch.zeros_like(r)
        unwrapped_model.disable_fused_attn() 
        F_avg, F_avg_grad = torch.func.jvp(model_wrapper, (x_t, t, r), (v_x, v_t, v_r))
        unwrapped_model.enable_fused_attn() 
        
        F_avg_grad = F_avg_grad.detach()
        
        # Compute average velocity target
        v_bar = v + (r - t) * F_avg_grad
        
        # Compute Mean Flow Loss
        mf_loss = self.norm_l2_loss(F_avg, v_bar, p=1.0)
        
        return mf_loss.mean(), mf_loss.mean(), torch.zeros_like(mf_loss).mean()


class sCMLoss(FACMLoss):
    """sCM Loss accroding to (https://arxiv.org/abs/2410.11081)"""

    def __call__(self, accelerator, model, compiled_model, ema, images, iters,
                model_kwargs=None, args=None, freezed_teacher=None):
        """sCM Loss computation"""
        if model_kwargs is None:
            model_kwargs = {}

        # Setup model
        unwrapped_model = model.module if hasattr(model, 'module') else model
        
        b, c, h, w = images.shape

        # Sample noise and time
        noise = torch.randn_like(images)
        t = self.sample_t(images.device, size=b, type=args.t_type, args=args)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t, t_end = self.interpolant(t)

        # Create noisy trajectory point
        x_t = alpha_t * images + sigma_t * noise  
        target = d_alpha_t * images + d_sigma_t * noise

        # Get dxt/dt from teacher/ground truth
        reference_model = compiled_model if freezed_teacher is None else freezed_teacher
        v, cuda_state = self.get_velocity(
            x_t, target, t, model_kwargs['y'], reference_model, args
        )

        # stable Consistency Model (sCM) Loss
        def model_wrapper(x_input, t_input):
            torch.cuda.set_rng_state(cuda_state)
            output = unwrapped_model(x_input, t_input.reshape(-1), None, **model_kwargs) 
            return output

        # Compute output via JVP
        v_x = v
        v_t = torch.ones_like(t)
        unwrapped_model.disable_fused_attn() 
        F_avg, F_avg_grad = torch.func.jvp(model_wrapper, (x_t, t), (v_x, v_t))
        unwrapped_model.enable_fused_attn() 
        
        F_avg_grad = F_avg_grad.detach()
        F_avg_sg = F_avg.detach()
        
        # Compute tangent vector
        alpha = torch.cos(alpha_t * np.pi / 2)
        r_factor = min(1.0, iters / getattr(args, 'warmup_steps', args.accumulation * 10000))
        v_bar = v + (t_end - t) * F_avg_grad * r_factor
        g = alpha * (F_avg_sg - v_bar)
        
        # Compute normalized tangent vector
        target = F_avg_sg - g
        g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
        g_norm = g_norm * np.sqrt(g_norm.numel() / g.numel())
        g_normalized = g / (g_norm + 0.1)

        # Adaptive weight sCM loss
        target = F_avg_sg - g_normalized
        squre_error = self.norm_l2_loss(F_avg, target, p=0.0)
        cm_loss = unwrapped_model.AdaLoss(t, squre_error)
        
        return cm_loss.mean(), cm_loss.mean(), torch.zeros_like(cm_loss).mean()