# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import argparse
import logging
import math
import os
import pdb
from collections import defaultdict, OrderedDict
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
from pathlib import Path
import math
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf


##### model imports
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport, Sampler

##### general utils
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import *
from utils.wandb_utils import *
from utils.dist_utils import *


##### Eval utils
from stage2.transport.transport import ModelType, WeightType
import stage2.transport.path as path
from stage2.transport.utils import mean_flat
from stage2.transport.integrators import ode

# -----------------------------------------------------------------------------
# Dual Transport Helpers
# -----------------------------------------------------------------------------

def compute_ode_drift(transport, x, t, model_output):
    """
    Computes dx/dt based on transport type and model output.
    Replicates transport.get_drift() logic but decouples model invocation.
    """
    drift_mean, drift_var = transport.path_sampler.compute_drift(x, t)
    
    if transport.model_type == ModelType.NOISE:
        sigma_t, _ = transport.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
        score = model_output / -sigma_t
        return (-drift_mean + drift_var * score)
    elif transport.model_type == ModelType.SCORE:
        # Assuming model outputs score directly
        return (-drift_mean + drift_var * model_output)
    elif transport.model_type == ModelType.VELOCITY:
        return model_output
    elif transport.model_type == ModelType.DATA:
        # User wants x0 (data) prediction. model_output is predicted x1.
        # We need to map x1 prediction to velocity to integrate.
        t_expand = path.expand_t_like_x(t, x)
        alpha_t, d_alpha_t = transport.path_sampler.compute_alpha_t(t_expand)
        sigma_t, d_sigma_t = transport.path_sampler.compute_sigma_t(t_expand)
        pred_x0 = (x - alpha_t * model_output) / sigma_t
        pred_ut = d_alpha_t * model_output + d_sigma_t * pred_x0
        return pred_ut
    else:
        raise NotImplementedError(f"Unknown transport model type: {transport.model_type}")

def joint_training_losses(model, x1_pix, x1_rae, transport_pix, transport_rae, model_kwargs):
    """
    Compute joint loss for Pixel (x) and RAE (z) streams.
    """
    # 1. Sample t and noise for Pixel stream
    t_pix, x0_pix, _, t_ori = transport_pix.sample(x1_pix) 
    
    # 2. Sample t and noise for RAE stream (Decoupled t)
    t_rae, x0_rae, _, _ = transport_rae.sample(x1_rae, t_ori) # Same timestep
    print(t_pix[0], t_rae[0])
    
    # 3. Plan (Interpolate)
    # Pixel Plan
    _, xt_pix, ut_pix = transport_pix.path_sampler.plan(t_pix, x0_pix, x1_pix)
    
    # RAE Plan
    _, xt_rae, ut_rae = transport_rae.path_sampler.plan(t_rae, x0_rae, x1_rae)
    
    # 4. Model Forward
    # PixelDDT expects (x_pix, x_rae, t_pix, t_rae, y)
    model_output_pix, model_output_rae = model(
        x_pix=xt_pix, 
        x_rae=xt_rae,
        t_pix=t_pix, 
        t_rae=t_rae, 
        **model_kwargs
    )
    
    # 5. Compute Losses
    # Helper to compute single stream loss
    def compute_stream_loss(trans, pred, target, xt, x0, t):
        if trans.model_type == ModelType.VELOCITY:
            return mean_flat(((pred - target) ** 2))
        elif trans.model_type == ModelType.DATA:
            # Use velocity loss. Hard coded. TODO: support different loss types.
            t_expand = path.expand_t_like_x(t, xt)
            alpha_t, d_alpha_t = trans.path_sampler.compute_alpha_t(t_expand)
            sigma_t, d_sigma_t = trans.path_sampler.compute_sigma_t(t_expand)
            pred_x0 = (xt - alpha_t * pred) / sigma_t
            pred_ut = d_alpha_t * pred + d_sigma_t * pred_x0
            return mean_flat(((pred_ut - target) ** 2))
        else:
            # Re-implement transport.training_losses logic for score/noise
            _, drift_var = trans.path_sampler.compute_drift(xt, t)
            # handle t expansion for sigma_t computation
            sigma_t, _ = trans.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            
            if trans.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif trans.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif trans.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            
            if trans.model_type == ModelType.NOISE:
                return mean_flat(weight * ((pred - x0) ** 2))
            else:
                return mean_flat(weight * ((pred * sigma_t + x0) ** 2))

    loss_pix = compute_stream_loss(transport_pix, model_output_pix, ut_pix, xt_pix, x0_pix, t_pix)
    # loss_rae = compute_stream_loss(transport_rae, model_output_rae, ut_rae, xt_rae, x0_rae, t_rae)
    
    return loss_pix #, loss_rae

@torch.no_grad()
def joint_sample_ode(
    z_pix, 
    z_rae, 
    model_fn, 
    transport_pix, 
    transport_rae,
    sampler_type="dopri5",
    num_steps=50, 
    atol=1e-5,
    rtol=1e-5,
    **kwargs
):
    """
    ODE solver for coupled generation supporting 'dopri5', 'euler', etc. via integrals.ode.
    Handles different time schedules (t_pix vs t_rae) by integrating over a shared linear 's'
    and mapping to specific t_sched(s) inside the drift function.
    """
    B = z_pix.shape[0]
    device = z_pix.device
    
    # 1. Define Time Mapping (s -> t)
    # s is linear time [0, 1] (or [eps, 1-eps])
    # t is the actual schedule time used for transport diffusion
    # t(s) = shift * s / (1 + (shift - 1) * s)
    # dt/ds = shift / (1 + (shift - 1) * s)^2
    
    def map_time_and_deriv(s, shift):
        # s is a tensor of shape (N,) or scalar
        denom = 1 + (shift - 1) * s
        t = shift * s / denom
        dt_ds = shift / (denom ** 2)
        return t, dt_ds

    # 2. Define Joint Drift Function
    def joint_drift_fn(x_joint, s, model, **model_kwargs):
        """
        x_joint: tuple (x_pix, x_rae)
        s: current integration time (scalar tensor)
        """
        x_pix, x_rae = x_joint
        
        # Ensure s is proper shape for mapping
        curr_s = s if isinstance(s, torch.Tensor) else torch.tensor(s, device=device)
        # expand to match batch size if necessary but map_time handles broadcasting usually
        # map_time_and_deriv handles scalar s just fine, returning scalar t, dt_ds.
        
        # Calculate specific t and time dilation factor dt/ds
        # Since 's' is already broadcasted to (B,) by integrators.py _fn,
        # t_pix and t_rae are also (B,).
        t_pix, dt_pix_ds = map_time_and_deriv(curr_s, transport_pix.time_dist_shift)
        t_rae, dt_rae_ds = map_time_and_deriv(curr_s, transport_rae.time_dist_shift) # align with t_pix
        
        # Broadcast t for model input (B,)
        # Check if t needs broadcasting or not (if s was scalar)
        if t_pix.ndim == 0:
            t_input_pix = t_pix.repeat(B)
            t_input_rae = t_rae.repeat(B)
            dt_broad_pix = dt_pix_ds # Scalar is fine for multiplication
            dt_broad_rae = dt_rae_ds
        else:
            t_input_pix = t_pix # Already (B,)
            t_input_rae = t_rae 
            # Expand dt derivative for broadcasting against (B, C, H, W)
            # dt_pix_ds is (B,). Need (B, 1, 1, 1)
            dt_broad_pix = dt_pix_ds.view(B, *([1]*(x_pix.ndim-1)))
            dt_broad_rae = dt_rae_ds.view(B, *([1]*(x_rae.ndim-1)))
        
        # Model Forward
        print(t_input_pix, t_input_rae)
        model_out_pix, model_out_rae = model_fn(
            x_pix=x_pix, 
            x_rae=x_rae, 
            t_pix=t_input_pix, 
            t_rae=t_input_rae, 
            **model_kwargs
        )
        
        # Compute "Physical" Drift dx/dt (in schedule time)
        drift_pix_dt = compute_ode_drift(transport_pix, x_pix, t_input_pix, model_out_pix)
        drift_rae_dt = compute_ode_drift(transport_rae, x_rae, t_input_rae, model_out_rae)
        
        # Compute "Integration" Drift dx/ds = (dx/dt) * (dt/ds)
        drift_pix_ds = drift_pix_dt * dt_broad_pix
        drift_rae_ds = drift_rae_dt * dt_broad_rae
        
        return (drift_pix_ds, drift_rae_ds)

    # 3. Setup ODE Integrator
    # We configure it with shift=1.0 so it integrates 's' linearly.
    # We manually handle the shift inside joint_drift_fn.
    # Note: check_interval usually returns [0, 1] for Linear path with shift=1.
    
    # Get common interval. Since we map s->t manually, we should use the "unshifed" interval.
    # We can ask transport for interval with shift=1 (linear).
    # transport.check_interval uses self.train_eps.
    # Construct a dummy transport or just invoke check_interval logic manually?
    # Or temporarily overwrite shift? No, side effects bad.
    
    # We can trust that for eps=1e-3, range is [1e-3, 1-1e-3] approximately.
    # Let's verify transport.check_interval logic.
    t0, t1 = transport_pix.check_interval(transport_pix.train_eps, transport_pix.sample_eps, eval=True)
    # The returned t0, t1 are start/end POINTS.
    # If path is Linear/GVP, t0=0, t1=1-eps.
    
    # Integrator expects t0, t1.
    # If we pass time_dist_shift=1.0 to `ode`, it generates linear `self.t` from [t0, t1].
    # Which is exactly what we want 's' to be.

    if sampler_type == "naive":
        return joint_sample_ode_naive(
            z_pix, 
            z_rae, 
            model_fn, 
            transport_pix, 
            transport_rae,
            num_steps=num_steps, 
            **kwargs
        )
    
    _ode = ode(
        drift=joint_drift_fn,
        t0=t0, # Start s
        t1=t1, # End s
        sampler_type=sampler_type,
        num_steps=num_steps,
        atol=atol,
        rtol=rtol,
        time_dist_shift=1.0, # Linear integration in s
    )
    
    # 4. Sample
    # Initial State
    x_init = (z_pix, z_rae)
    
    # ode.sample returns a list of samples (trajectory). We take the last one.
    # Filter kwargs for ODE solver to remove intermediate tracking params not supported by model_fn
    ode_kwargs = {k: v for k, v in kwargs.items() if k not in ['intermediates', 'capture_intermediates']}
    samples = _ode.sample(x_init, model_fn, **ode_kwargs)
    
    samples_pix_hist, samples_rae_hist = samples
    
    x_pix_final = samples_pix_hist[-1]
    x_rae_final = samples_rae_hist[-1]
        
    return x_pix_final, x_rae_final

def joint_sample_ode_naive(
    z_pix, 
    z_rae, 
    model_fn, 
    transport_pix, 
    transport_rae,
    num_steps=50, 
    **kwargs
):
    """
    Naive Euler ODE solver for coupled generation with support for different transport schedules.
    Guarantees correctness by explicit stepping without complex ODE integrators.
    Now correctly handles non-linear time schedules (time_dist_shift).
    """
    B = z_pix.shape[0]
    device = z_pix.device
    
    # 1. Define shared linear time grid 's'
    # Use transport_pix for the canonical 's' range.
    t0, t1 = transport_pix.check_interval(transport_pix.train_eps, transport_pix.sample_eps, eval=True)
    # Important: The standard ODE solver integrates DIFFERENTLY!
    # Looking at integrators.py:
    # self.t = 1 - th.linspace(t0, t1, num_steps)
    # It integrates BACKWARDS from 1 to 0 (or Forward 0 to 1?).
    # let's verify: t0=eps, t1=1-eps. linspace(eps, 1-eps). 1 - linspace -> (1-eps, eps).
    # So it goes from HIGH to LOW t values?
    # If path is x_t = (1-t)x0 + t*x1 (Linear), t=0 is noise, t=1 is data.
    # We want to go from Noise (t=0?) to Data (t=1).
    # IF integrators.py does "1 - linspace", then t starts at ~1 and goes to ~0. This means it generates NOISE from DATA (forward process).
    # OR t=1 IS Noise and t=0 IS Data?
    
    # Let's check transport.py / path.py again.
    # Standard SiT: t=0 is Data, t=1 is Noise. (Usually).
    # If so, then sampling (Noise->Data) goes t: 1 -> 0.
    # The integrators.py code "1 - linspace(t0, t1)" produces a decreasing sequence [1-eps, ..., eps].
    # Which means we step with distinct negative dt.
    
    # In my naive implementation:
    # ss = torch.linspace(t0, t1) which is Increasing [eps, 1-eps].
    # This means I am integrating from eps -> 1-eps. (Low -> High).
    
    # IF correct direction is High -> Low (1->0), then my naive sampler is going BACKWARDS.
    
    # Let's align naive sampler with integrators.py logic:
    # "self.t = 1 - th.linspace(t0, t1, num_steps)"
    # So ss should be:
    ss = 1.0 - torch.linspace(t0, t1, num_steps + 1, device=device)
    # This gives us t: 1 -> 0.
    # ds = s_next - s_curr will be NEGATIVE.
    # This seems correct if we are integrating backwards in time.
    
    x_pix = z_pix.detach().clone()
    x_rae = z_rae.detach().clone()
    
    # Helper: Map linear s -> t(s) with shift
    def get_t(s, shift):
        return shift * s / (1 + (shift - 1) * s)
    
    # Helper: Derivative dt/ds
    def get_dt_ds(s, shift):
        denom = 1 + (shift - 1) * s
        return shift / (denom ** 2)

    # Extract intermediate tracking args to avoid passing to model
    capture_intermediates = kwargs.pop('capture_intermediates', False)
    intermediates = kwargs.pop('intermediates', None)

    # Internal helper for drift evaluation (Standard RK4 needs 4 evals)
    # Reverting to Euler as requested, but keeping the correct time direction (1 -> 0)
    
    # 1. Define shared linear time grid 's'
    # Use transport_pix for the canonical 's' range.
    t0, t1 = transport_pix.check_interval(transport_pix.train_eps, transport_pix.sample_eps, eval=True)
    
    # Integrators.py uses 1 - ... for sampling (High t to Low t) for Noise -> Data
    # t starts at ~1 (Noise) and goes to ~0 (Data)
    ss = 1.0 - torch.linspace(t0, t1, num_steps + 1, device=device)

    # Euler Integration Loop
    for i in range(num_steps):
        s_curr = ss[i]
        s_next = ss[i+1] # Integrate forward (in s space, which goes 1->0)
        ds = s_next - s_curr # Negative value
        
        # Calculate t at current step for both streams
        t_pix_scalar = get_t(s_curr, transport_pix.time_dist_shift)
        t_rae_scalar = get_t(s_curr, transport_pix.time_dist_shift) # align with t_pix_scalar
        
        t_pix = t_pix_scalar.repeat(B)
        t_rae = t_rae_scalar.repeat(B)
        
        # Calculate derivatives dt/ds (scalar)
        dt_ds_pix = get_dt_ds(s_curr, transport_pix.time_dist_shift)
        dt_ds_rae = get_dt_ds(s_curr, transport_pix.time_dist_shift) # align with dt_ds_pix
        
        # Predict Model Output (Velocity/Score/Noise)
        model_out_pix, model_out_rae = model_fn(
            x_pix=x_pix, 
            x_rae=x_rae, 
            t_pix=t_pix, 
            t_rae=t_rae, 
            **kwargs
        )
        
        if i == 0 or i == num_steps - 1 or i % 10 == 0:
            # Calulate predicted x0 for debug
            # Use scalar t for broadcasting
            x0_debug = x_pix - t_pix_scalar * model_out_pix
            print(f"Step {i} | s: {s_curr:.4f} -> {s_next:.4f} (ds={ds:.4f})")
            print(f"  Pix: t={t_pix_scalar.item():.4f}, dt/ds={dt_ds_pix.item():.4f}")
            print(f"  v_pix: mean={model_out_pix.mean():.4f}, std={model_out_pix.std():.4f}")
            print(f"  x_pix: mean={x_pix.mean():.4f}, std={x_pix.std():.4f}")
            print(f"  x0_pred: mean={x0_debug.mean():.4f}, std={x0_debug.std():.4f}, min={x0_debug.min():.4f}, max={x0_debug.max():.4f}")
        
        if capture_intermediates and intermediates is not None:
            # Calculate x0 prediction for visualization
            # Assuming standard Rectified Flow / Velocity:
            # t=1 is Noise, t=0 is Data.
            # v = dx/dt = Noise - Data.
            # x_t = Data + t * (Noise - Data) = Data + t * v
            # Convert to Physical Drift dx/dt
            drift_pix_dt = compute_ode_drift(transport_pix, x_pix, t_pix, model_out_pix)
            drift_rae_dt = compute_ode_drift(transport_rae, x_rae, t_rae, model_out_rae)

        if capture_intermediates and intermediates is not None:
            # Assuming linear schedule for debugging/vis:
            # v = (x_t - x_data) / t_scale -> roughly x_data = x_t - t * v (in DDPM notation, Data is x0)
            # Actually with ICPlan, Data is x1, Noise is x0.
            # So Data = x_t - t * v
            
            # Helper to broadcast t for subtraction
            def pred_x0(x, v, t_scalar):
                # t_scalar is (1,) tensor from get_t
                return x - t_scalar * v

            x0_pix = pred_x0(x_pix, drift_pix_dt, t_pix_scalar)
            x0_rae = pred_x0(x_rae, drift_rae_dt, t_rae_scalar)

            intermediates.append({
                'step': i,
                't_pix': t_pix_scalar.item(),
                'x_pix': x0_pix.detach().cpu(), # Storing PREDICTED Data
                'x_rae': x0_rae.detach().cpu(), 
                'v_pix': drift_pix_dt.detach().cpu(),
                'v_rae': drift_rae_dt.detach().cpu()
            })

        # Step State: x_next = x_curr + (dx/dt * dt/ds) * ds
        x_pix = x_pix + drift_pix_dt * (dt_ds_pix * ds)
        x_rae = x_rae + drift_rae_dt * (dt_ds_rae * ds)
        
    return x_pix, x_rae

from eval import evaluate_generation_distributed

def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 transport model on RAE latents.")
    parser.add_argument("--config", type=str, required=True, help="YAML config containing stage_1 and stage_2 sections.")
    parser.add_argument("--data-path", type=Path, required=True, help="Directory with ImageFolder structure for training.")
    parser.add_argument("--results-dir", type=str, default="ckpts", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256, help="Input image resolution.")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32", help="Compute precision for training.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--compile", action="store_true", help="Use torch compile (for rae.encode and model.forward).")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")
    args = parser.parse_args()
    return args

def main():
    """Trains a new SiT model using config-driven hyperparameters."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")
    rank, world_size, device = setup_distributed()
    full_cfg = OmegaConf.load(args.config)
    (
        rae_config,
        model_config,
        transport_config_pix,
        transport_config_rae,
        sampler_config_pix,
        sampler_config_rae,
        guidance_config,
        misc_config_pix,
        misc_config_rae,
        training_config,
        eval_config
    ) = parse_configs(full_cfg)

    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 sections.")

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc_pix = to_dict(misc_config_pix)
    misc_rae = to_dict(misc_config_rae)
    transport_cfg_pix = to_dict(transport_config_pix)
    transport_cfg_rae = to_dict(transport_config_rae)
    sampler_cfg_pix = to_dict(sampler_config_pix)
    sampler_cfg_rae = to_dict(sampler_config_rae)
    guidance_cfg = to_dict(guidance_config)
    training_cfg = to_dict(training_config)

    num_classes = int(misc_pix.get("num_classes", 1000))
    null_label = int(misc_pix.get("null_label", num_classes))
    
    latent_size = tuple(int(dim) for dim in misc_pix.get("latent_size", (3, 256, 256)))
    latent_size_rae = tuple(int(dim) for dim in misc_rae.get("latent_size", (768, 16, 16)))
    shift_dim = misc_pix.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc_pix.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    shift_dim_rae = misc_rae.get("time_dist_shift_dim", shift_dim)
    shift_base_rae = misc_rae.get("time_dist_shift_base", shift_base)
    time_dist_shift_rae = math.sqrt(shift_dim_rae / shift_base_rae)

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    num_epochs = int(training_cfg.get("epochs", 1400))
    global_batch_size = training_cfg.get("global_batch_size", None) # optional global batch size for override
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
    else:
        batch_size = int(training_cfg.get("batch_size", 16))
        global_batch_size = batch_size * world_size * grad_accum_steps
    num_workers = int(training_cfg.get("num_workers", 4))
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 2500)) 
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4)) # ckpt interval is epoch based
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    default_seed = int(training_cfg.get("global_seed", 0))
    
    if eval_config:
        """
        FID online evaluation setup
        """
        do_eval = True
        eval_interval = int(eval_config.get("eval_interval", 5000))
        eval_model = eval_config.get("eval_model", False) # by default eval ema. This decides whether to **additionally** eval the non-ema model.
        eval_data = eval_config.get("data_path", None)
        reference_npz_path = eval_config.get("reference_npz_path", None)
        assert eval_data, "eval.data_path must be specified to enable evaluation."
        assert reference_npz_path, "eval.reference_npz_path must be specified to enable evaluation."
    else:
        do_eval = False
    
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    use_fp16 = args.precision == "fp16"
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    autocast_enabled = use_fp16 or use_bf16
    autocast_kwargs = dict(dtype=autocast_dtype, enabled=autocast_enabled)
    scaler = GradScaler(enabled=use_fp16)

    transport_params_pix = dict(transport_cfg_pix.get("params", {}))
    transport_params_pix.pop("time_dist_shift", None)
    transport_params_rae = dict(transport_cfg_rae.get("params", {}))
    transport_params_rae.pop("time_dist_shift", None)

    sampler_mode_pix = sampler_cfg_pix.get("mode", "ODE").upper()
    sampler_params_pix = dict(sampler_cfg_pix.get("params", {}))
    sampler_mode_rae = sampler_cfg_rae.get("mode", "ODE").upper()
    sampler_params_rae = dict(sampler_cfg_rae.get("params", {}))

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if cfg_scale_override is not None:
        guidance_scale = float(cfg_scale_override)
    guidance_method = guidance_cfg.get("method", "cfg")

    def guidance_value(key: str, default: float) -> float:
        if key in guidance_cfg:
            return guidance_cfg[key]
        dashed_key = key.replace("_", "-")
        return guidance_cfg.get(dashed_key, default)

    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))
    
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    
    #### Model init
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device) 
    if args.compile:
        try:
            rae.encode = torch.compile(rae.encode)
        except:
            print('RAE ENCODE compile meets error, falling back to no compile')
        try:
            model.forward = torch.compile(model.forward)
        except:
            print('MODEL FORWARD compile meets error, falling back to no compile')
    else:
        print("Not compiling")
    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()
    # model.requires_grad_(True) # Do NOT force unfreeze. Let model.__init__ decide what is frozen (e.g. DDT).
    model.train() # Set to train mode (Dropout, BatchNorm, etc.)
    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    # ddp_model = torch.compile(ddp_model) # fix shape compile, see if it works
    model = ddp_model.module
    ddp_model.train()
    # no need to put RAE into DDP since it's frozen
    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")
    
    #### Opt, Schedl init
    optimizer, optim_msg = build_optimizer([p for p in model.parameters() if p.requires_grad], training_cfg)

    ### AMP init
    scaler, autocast_kwargs = get_autocast_scaler(args)
    
    
    ### Data init
    stage2_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    loader, sampler = prepare_dataloader(
        args.data_path, micro_batch_size, num_workers, rank, world_size, transform=stage2_transform
    )
    if do_eval:
        eval_dataset = ImageFolder(
            str(eval_data),
            transform=transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
                transforms.ToTensor(),
            ])
        )
        logger.info(f"Evaluation dataset loaded from {eval_data}, containing {len(eval_dataset)} images.")
        
    loader_batches = len(loader)
    if loader_batches % grad_accum_steps != 0:
        raise ValueError("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
    steps_per_epoch = loader_batches // grad_accum_steps
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")
    
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    
    #### Transport init
    transport_pix = create_transport(
        **transport_params_pix,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = Sampler(transport_pix)

    if sampler_mode_pix == "ODE":
        eval_sampler_pix= transport_sampler.sample_ode(**sampler_params_pix)
    elif sampler_mode_pix == "SDE":
        eval_sampler_pix = transport_sampler.sample_sde(**sampler_params_pix)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode_pix}.")
    
    transport_rae = create_transport(
        **transport_params_rae, 
        time_dist_shift=time_dist_shift_rae,
    )
    transport_sampler_rae = Sampler(transport_rae)
    if sampler_mode_rae == "ODE":
        eval_sampler_rae = transport_sampler_rae.sample_ode(**sampler_params_rae)
    elif sampler_mode_rae == "SDE": 
        eval_sampler_rae = transport_sampler_rae.sample_sde(**sampler_params_rae)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode_rae}.")
    
    ### Guidance Init
    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guidance_model_cfg = guidance_cfg.get("guidance_model")
        if guidance_model_cfg is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guidance_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    
    log_steps = 0
    running_loss = 0.0
    start_time = time()
    use_guidance = guidance_scale > 1.0
    zs = torch.randn(micro_batch_size, *latent_size, device=device, dtype=torch.float32) # always use float for noise sampling
    zs_rae = torch.randn(micro_batch_size, *latent_size_rae, device=device, dtype=torch.float32)
    n = micro_batch_size
    if use_guidance:
        zs = torch.cat([zs, zs], dim=0)
        zs_rae = torch.cat([zs_rae, zs_rae], dim=0)
        y_null = torch.full((n,), null_label, device=device)
        ys = torch.cat([ys, y_null], dim=0)
        sample_model_kwargs = dict(
            cfg_scale=guidance_scale,
            cfg_interval=(t_min, t_max),
        )
        if guidance_method == "autoguidance":
            if guid_model_forward is None:
                raise RuntimeError("Guidance model forward is not initialized.")
            sample_model_kwargs["additional_model_forward"] = guid_model_forward
            ema_model_fn = ema_model.forward_with_autoguidance
            model_fn = model.forward_with_autoguidance
        else:
            ema_model_fn = ema_model.forward_with_cfg
            model_fn = model.forward_with_cfg
    else:
        sample_model_kwargs = dict()
        ema_model_fn = ema_model.forward
        model_fn = model.forward

    ### Resuming and checkpointing
    start_epoch = 0
    global_step = 0
    maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
    if maybe_resume_ckpt_path is not None:
        logger.info(f"Experiment resume checkpoint found at {maybe_resume_ckpt_path}, automatically resuming...")
        ckpt_path = Path(maybe_resume_ckpt_path)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                ckpt_path,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )
            logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # starting from fresh, save worktree and configs
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")
    ### Logging experiment details
    if rank == 0:
        num_params = sum(p.numel() for p in rae.parameters())
        logger.info(f"Stage-1 RAE parameters: {num_params/1e6:.2f}M")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Stage-2 Model parameters: {num_params/1e6:.2f}M")
        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            logger.info("Not clipping gradients.")
        # print optim and schel
        logger.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler.")
        logger.info(f"Training for {num_epochs} epochs, batch size {micro_batch_size} per GPU.")
        logger.info(f"Dataset contains {len(loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")

    dist.barrier() 
    for epoch in range(start_epoch, num_epochs):
        model.train()
        sampler.set_epoch(epoch)
        epoch_metrics = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        optimizer.zero_grad()
        accum_counter = 0
        step_loss_accum = 0.0
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0  and rank == 0:
            logger.info(f"Saving checkpoint at epoch {epoch}...")
            ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt" 
            save_checkpoint(
                ckpt_path,
                global_step,
                epoch,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )
        for step, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad(): # TODO: wrap this in autocast?
                z = rae.encode(images)
            
            # PixelDDT Input setup
            x_pix = images
            x_rae = z

            optimizer.zero_grad(set_to_none=True)
            model_kwargs = dict(y=labels)
            
            with autocast(**autocast_kwargs):
                loss_pix = joint_training_losses(
                    ddp_model, 
                    x_pix, 
                    x_rae, 
                    transport_pix, 
                    transport_rae, 
                    model_kwargs
                )
                loss = loss_pix.mean()

            loss.float()
            if scaler:
                scaler.scale(loss / grad_accum_steps).backward()
            else:
                (loss / grad_accum_steps).backward()
            if clip_grad:
                if scaler:
                    scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
            if global_step % grad_accum_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                update_ema(ema_model, ddp_model.module, decay=ema_decay)
            running_loss += loss.item()
            epoch_metrics['loss'] += loss.detach()
            epoch_metrics['loss_pix'] += loss_pix.mean().detach()
            
            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                avg_loss = running_loss / log_interval # flow loss often has large variance so we record avg loss
                steps = torch.tensor(log_interval, device=device)
                stats = {
                    "train/loss": avg_loss, 
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if args.wandb:
                    wandb_utils.log(
                        stats,
                        step=global_step,
                    )
                running_loss = 0.0
            if global_step % sample_every == 0:
                model.eval()
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    n_vis = min(8, micro_batch_size)
                    
                    if use_guidance:
                         zs_cond = zs[:n_vis]
                         zs_uncond = zs[micro_batch_size : micro_batch_size + n_vis]
                         zs_samples = torch.cat([zs_cond, zs_uncond], dim=0)
                         
                         zs_rae_cond = zs_rae[:n_vis]
                         zs_rae_uncond = zs_rae[micro_batch_size : micro_batch_size + n_vis]
                         zs_rae_samples = torch.cat([zs_rae_cond, zs_rae_uncond], dim=0)
                         
                         y_cond = labels[:n_vis]
                         y_uncond = torch.full_like(y_cond, null_label)
                         visual_sample_model_kwargs = deepcopy(sample_model_kwargs)
                         visual_sample_model_kwargs['y'] = torch.cat([y_cond, y_uncond], dim=0)
                    else:
                         zs_samples = zs[:n_vis]
                         zs_rae_samples = zs_rae[:n_vis]
                         visual_sample_model_kwargs = deepcopy(sample_model_kwargs)
                         visual_sample_model_kwargs['y'] = labels[:n_vis]

                    # with autocast(**autocast_kwargs): # Disable autocast for sampling to mimic exact pipeline
                    samples_pix, samples_rae = joint_sample_ode(
                        zs_samples,
                        zs_rae_samples,
                        ema_model_fn,
                        transport_pix=transport_pix,
                        transport_rae=transport_rae,
                        num_steps=sampler_params_pix.get("num_steps", 50),
                        sampler_type="dopri5",
                        **visual_sample_model_kwargs
                    )
                
                    samples_pix = samples_pix.float()
                    samples_rae = samples_rae.float()

                    if use_guidance:
                        samples_pix, _ = samples_pix.chunk(2, dim=0)
                        samples_rae, _ = samples_rae.chunk(2, dim=0)
                    
                    samples = samples_pix.cpu().float()
                    samples = samples.clamp(0, 1)

                    # Decode RAE samples
                    decoded_rae = rae.decode(samples_rae)
                    decoded_rae = decoded_rae.cpu().float().clamp(0, 1)
                    
                    dist.barrier()
                    print(f"Rank{rank}")
                    if args.wandb and rank == 0:
                        wandb_utils.log_image(samples, global_step, key="samples/pixel", caption="Pixel Stream Samples")
                        wandb_utils.log_image(decoded_rae, global_step, key="samples/rae", caption="RAE Stream Samples")
                logger.info("Generating EMA samples done.")
                model.train()
            if do_eval and (eval_interval > 0 and global_step % eval_interval == 0):
                logger.warning("Evaluation disabled for PixelDDT dual-stream model pending refactor of evaluation pipeline.")
                # logger.info("Starting evaluation...")
                # model.eval()
                # eval_models = [(ema_model_fn, "ema")]
                # if eval_model:
                #     eval_models.append((model_fn, "model"))
                # for fn, mod_name in eval_models:
                #     eval_stats = evaluate_generation_distributed(
                #         fn,
                #         eval_sampler,
                #         latent_size,
                #         sample_model_kwargs,
                #         use_guidance,
                #         rae, 
                #         eval_dataset,
                #         len(eval_dataset),
                #         rank = rank,
                #         world_size = world_size,
                #         device = device,
                #         batch_size = micro_batch_size,
                #         experiment_dir = experiment_dir,
                #         global_step = global_step,
                #         autocast_kwargs = autocast_kwargs,
                #         reference_npz_path = reference_npz_path
                #     )
                #     # log with prefix
                #     eval_stats = {f"eval_{mod_name}/{k}": v for k, v in eval_stats.items()} if eval_stats is not None else {}
                #     if args.wandb:
                #         wandb_utils.log(eval_stats, step=global_step)
                #     model.train()
                # logger.info("Evaluation done.")
            global_step += 1
            num_batches += 1
        if rank == 0 and num_batches > 0:
            avg_loss = epoch_metrics['loss'].item() / num_batches 
            epoch_stats = {
                "epoch/loss": avg_loss,
            }
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)
    # save the final ckpt
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt" 
        save_checkpoint(
            ckpt_path,
            global_step,
            num_epochs,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
        )
    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()



if __name__ == "__main__":
    main()
