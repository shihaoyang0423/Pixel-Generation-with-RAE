# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
train_stage3_projcet.py — Project version of train_stage3.py.

Key difference from train_stage3.py:
  Sequential denoising: RAE fully denoises first (Phase 1, 50 steps),
  then its clean bottleneck features condition the Pixel DDT (Phase 3, 50 steps).

Reverted "Pro" changes (back to original baseline behaviour):
  - Pixel normalisation: [0, 1]  (x_pix = images, no 2x-1)
  - No sigma_t.clamp(min=0.05) in DATA drift / loss
  - train_eps / sample_eps: transport defaults (1e-3 for DATA/Linear)
  - time_dist_type: read from yaml, not hardcoded
"""
import argparse
import logging
import math
import os
import sys
from collections import defaultdict
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
from copy import deepcopy
from time import time
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf
from typing import Optional, Tuple

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

##### transport / eval utils
from stage2.transport.transport import ModelType, WeightType
import stage2.transport.path as path
from stage2.transport.utils import mean_flat
from stage2.transport.integrators import ode

# -----------------------------------------------------------------------------
# Transport helpers (identical to base version but DATA branch has NO clamp)
# -----------------------------------------------------------------------------

def compute_ode_drift(transport, x, t, model_output):
    """
    Computes dx/dt based on transport type and model output.
    DATA branch: uses plain / sigma_t — NO clamp (reverted from Pro version).
    """
    drift_mean, drift_var = transport.path_sampler.compute_drift(x, t)

    if transport.model_type == ModelType.NOISE:
        sigma_t, _ = transport.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
        score = model_output / -sigma_t
        return (-drift_mean + drift_var * score)
    elif transport.model_type == ModelType.SCORE:
        return (-drift_mean + drift_var * model_output)
    elif transport.model_type == ModelType.VELOCITY:
        return model_output
    elif transport.model_type == ModelType.DATA:
        t_expand = path.expand_t_like_x(t, x)
        alpha_t, d_alpha_t = transport.path_sampler.compute_alpha_t(t_expand)
        sigma_t, d_sigma_t = transport.path_sampler.compute_sigma_t(t_expand)
        # No clamp — original baseline behaviour
        pred_x0 = (x - alpha_t * model_output) / sigma_t
        pred_ut = d_alpha_t * model_output + d_sigma_t * pred_x0
        return pred_ut
    else:
        raise NotImplementedError(f"Unknown transport model type: {transport.model_type}")


# -----------------------------------------------------------------------------
# Project training losses — independent t sampling, clean conditioning
# -----------------------------------------------------------------------------

def project_training_losses(model, x1_pix, x1_rae, transport_pix, transport_rae, model_kwargs):
    """
    Compute joint loss for Pixel and RAE streams (project / sequential version).

    Key differences from joint_training_losses:
      - t_pix and t_rae are sampled INDEPENDENTLY (no shared t_ori).
      - model.forward() receives x1_rae (clean) so it can extract clean
        conditioning features at t=0 (simulating post-RAE-ODE state).
      - DATA loss branch has NO sigma_t clamp.

    When model.pixel_only is True, the RAE stream is skipped entirely:
      - No transport sampling for RAE.
      - No rae.encode() call (should be skipped upstream in training loop).
      - Returns (loss_pix, zeros) — RAE loss is identically zero.
    """
    def compute_stream_loss(trans, pred, target, xt, x0, t):
        if trans.model_type == ModelType.VELOCITY:
            return mean_flat(((pred - target) ** 2))
        elif trans.model_type == ModelType.DATA:
            t_expand = path.expand_t_like_x(t, xt)
            alpha_t, d_alpha_t = trans.path_sampler.compute_alpha_t(t_expand)
            sigma_t, d_sigma_t = trans.path_sampler.compute_sigma_t(t_expand)
            # No clamp — original baseline behaviour
            pred_x0 = (xt - alpha_t * pred) / sigma_t
            pred_ut = d_alpha_t * pred + d_sigma_t * pred_x0
            return mean_flat(((pred_ut - target) ** 2))
        else:
            _, drift_var = trans.path_sampler.compute_drift(xt, t)
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

    # 1. Sample t and noise for Pixel stream (always)
    t_pix, x0_pix, _, _ = transport_pix.sample(x1_pix)
    _, xt_pix, ut_pix = transport_pix.path_sampler.plan(t_pix, x0_pix, x1_pix)

    raw_model = getattr(model, 'module', model)
    if getattr(raw_model, 'pixel_only', False):
        # pixel_only: skip RAE transport sampling and RAE forward entirely
        model_output_pix, _ = model(x_pix=xt_pix, t_pix=t_pix, **model_kwargs)
        loss_pix = compute_stream_loss(transport_pix, model_output_pix, ut_pix, xt_pix, x0_pix, t_pix)
        return loss_pix, torch.zeros(1, device=x1_pix.device, dtype=x1_pix.dtype)

    # 2. Sample t and noise for RAE stream (independent — no shared t_ori)
    t_rae, x0_rae, _, _ = transport_rae.sample(x1_rae)

    # 3. Interpolate to get noisy inputs
    _, xt_rae, ut_rae = transport_rae.path_sampler.plan(t_rae, x0_rae, x1_rae)

    # 4. Model forward — passes x1_rae for clean conditioning
    model_output_pix, model_output_rae = model(
        x_pix=xt_pix,
        x_rae=xt_rae,
        t_pix=t_pix,
        t_rae=t_rae,
        x1_rae=x1_rae,   # clean RAE for pixel conditioning (Phase 2 simulation)
        **model_kwargs
    )

    # 5. Compute losses
    loss_pix = compute_stream_loss(transport_pix, model_output_pix, ut_pix, xt_pix, x0_pix, t_pix)
    loss_rae = compute_stream_loss(transport_rae, model_output_rae, ut_rae, xt_rae, x0_rae, t_rae)

    return loss_pix, loss_rae


# -----------------------------------------------------------------------------
# Project sequential ODE sampler (3-phase: RAE → extract → Pixel)
# -----------------------------------------------------------------------------

@torch.no_grad()
def project_sample_ode_naive(
    z_pix,
    z_rae,
    model,
    transport_pix,
    transport_rae,
    num_steps=50,
    cfg_scale=1.0,
    cfg_interval=(0.0, 1.0),
    guidance_method="cfg",
    guidance_model=None,
    ag_phase="pixel",
    verbose=False,
    **kwargs
):
    """
    Sequential ODE sampler (project version — Euler method).

    Phase 1: RAE ODE       — denoise z_rae for num_steps steps.
    Phase 2: Extract cond  — run DDT encoder on clean z_rae at t=0.
    Phase 3: Pixel ODE     — denoise z_pix for num_steps steps using fixed s_clean.

    model: PixelDDT instance (PixelDDTv2_project.PixelDDT) — passed directly.
    kwargs: must contain at least 'y' (class labels tensor).

    guidance_method == "cfg" and cfg_scale > 1.0:
        z_pix/z_rae must already be doubled [cond; uncond],
        y must be doubled [y_cond; y_null].

    guidance_method == "autoguidance" and cfg_scale > 1.0:
        z_pix/z_rae/y stay at normal batch size. guidance_model supplies
        the bad-model prediction, mixed as bad + scale * (good - bad).
    """
    B_full = z_pix.shape[0]   # may be 2B if CFG
    device = z_pix.device
    guidance_method = (guidance_method or "cfg").lower()
    using_guidance = cfg_scale > 1.0
    using_cfg = using_guidance and guidance_method == "cfg"
    using_ag = using_guidance and guidance_method == "autoguidance"
    ag_phase = (ag_phase or "pixel").lower()
    if using_guidance and guidance_method not in {"cfg", "autoguidance"}:
        raise ValueError(f"Unknown guidance_method: {guidance_method!r}")
    if using_ag and guidance_model is None:
        raise ValueError("guidance_model is required when guidance_method='autoguidance'.")
    if using_ag and ag_phase not in {"pixel", "rae", "both"}:
        raise ValueError(f"Unknown ag_phase: {ag_phase!r}; expected pixel/rae/both.")
    y = kwargs['y']

    def apply_autoguidance(good_out, bad_out, t_batch, channels):
        good_eps = good_out[:, :channels]
        bad_eps = bad_out[:, :channels]
        rest = good_out[:, channels:]
        guid_t_min, guid_t_max = cfg_interval
        mask = ((t_batch >= guid_t_min) & (t_batch <= guid_t_max))
        mask = mask.view(-1, *([1] * (len(good_eps.shape) - 1)))
        guided_eps = torch.where(
            mask,
            bad_eps + cfg_scale * (good_eps - bad_eps),
            good_eps,
        )
        return torch.cat([guided_eps, rest], dim=1) if rest.shape[1] > 0 else guided_eps

    def get_t(s, shift):
        return shift * s / (1 + (shift - 1) * s)

    def get_dt_ds(s, shift):
        denom = 1 + (shift - 1) * s
        return shift / (denom ** 2)

    # pixel_only mode: skip Phase 1 (RAE ODE) and Phase 2 (extract_clean_s)
    if getattr(model, 'pixel_only', False):
        t0_pix, t1_pix = transport_pix.check_interval(transport_pix.train_eps, transport_pix.sample_eps, eval=True)
        ss_pix = 1.0 - torch.linspace(t0_pix, t1_pix, num_steps + 1, device=device)
        x_pix = z_pix.detach().clone()
        for i in range(num_steps):
            s_curr = ss_pix[i]
            s_next = ss_pix[i + 1]
            ds = s_next - s_curr
            t_pix_scalar = get_t(s_curr, transport_pix.time_dist_shift)
            t_pix_batch = t_pix_scalar.repeat(B_full)
            dt_ds_pix = get_dt_ds(s_curr, transport_pix.time_dist_shift)
            if using_cfg:
                model_out_pix = model.forward_pix_with_cfg(
                    x_pix, t_pix_batch, y, None,
                    cfg_scale=cfg_scale, cfg_interval=cfg_interval
                )
            elif using_ag and ag_phase in {"pixel", "both"}:
                good_pix = model.forward_pix(x_pix, t_pix_batch, y, None)
                bad_pix = guidance_model.forward_pix(x_pix, t_pix_batch, y, None)
                model_out_pix = apply_autoguidance(good_pix, bad_pix, t_pix_batch, model.in_channels)
            else:
                model_out_pix = model.forward_pix(x_pix, t_pix_batch, y, None)
            drift_pix_dt = compute_ode_drift(transport_pix, x_pix, t_pix_batch, model_out_pix)
            x_pix = x_pix + drift_pix_dt * (dt_ds_pix * ds)
            if verbose and (i == 0 or i == num_steps - 1 or i % 10 == 0):
                print(f"[Pix pixel_only] step {i} | s={s_curr:.4f} t_pix={t_pix_scalar.item():.4f}")
        return x_pix, None

    # -----------------------------------------------------------------------
    # Phase 1: RAE ODE — forward_rae (or forward_rae_with_cfg)
    # -----------------------------------------------------------------------
    t0_rae, t1_rae = transport_rae.check_interval(transport_rae.train_eps, transport_rae.sample_eps, eval=True)
    ss_rae = 1.0 - torch.linspace(t0_rae, t1_rae, num_steps + 1, device=device)

    x_rae = z_rae.detach().clone()

    for i in range(num_steps):
        s_curr = ss_rae[i]
        s_next = ss_rae[i + 1]
        ds = s_next - s_curr

        t_rae_scalar = get_t(s_curr, transport_rae.time_dist_shift)
        t_rae_batch = t_rae_scalar.repeat(B_full)
        dt_ds_rae = get_dt_ds(s_curr, transport_rae.time_dist_shift)

        if using_cfg:
            model_out_rae = model.forward_rae_with_cfg(
                x_rae, t_rae_batch, y, cfg_scale=cfg_scale, cfg_interval=cfg_interval
            )
        elif using_ag and ag_phase in {"rae", "both"}:
            good_rae = model.forward_rae(x_rae, t_rae_batch, y)
            bad_rae = guidance_model.forward_rae(x_rae, t_rae_batch, y)
            model_out_rae = apply_autoguidance(good_rae, bad_rae, t_rae_batch, model.ddt.in_channels)
        else:
            model_out_rae = model.forward_rae(x_rae, t_rae_batch, y)

        drift_rae_dt = compute_ode_drift(transport_rae, x_rae, t_rae_batch, model_out_rae)
        x_rae = x_rae + drift_rae_dt * (dt_ds_rae * ds)

        if verbose and (i == 0 or i == num_steps - 1 or i % 10 == 0):
            print(f"[RAE Phase1] step {i} | s={s_curr:.4f} t_rae={t_rae_scalar.item():.4f}")

    x_rae_clean = x_rae  # fully denoised RAE latent

    # -----------------------------------------------------------------------
    # Phase 2: Extract clean conditioning features from denoised RAE
    # With CFG, extract s_clean separately for cond/uncond halves so injected
    # pixel layers see y_null features on the uncond batch slot.
    # x_rae_clean[:B] and x_rae_clean[B:] are equal after Phase 1 CFG guidance.
    # -----------------------------------------------------------------------
    if using_cfg:
        B = B_full // 2
        x_rae_half = x_rae_clean[:B]
        y_cond = y[:B]
        y_uncond = y[B:]
        s_clean_cond = model.extract_clean_s(x_rae_half, y_cond)       # [B, L, C_enc]
        s_clean_uncond = model.extract_clean_s(x_rae_half, y_uncond)   # [B, L, C_enc]
        s_clean = torch.cat([s_clean_cond, s_clean_uncond], dim=0)     # [2B, L, C_enc]
    else:
        s_clean = model.extract_clean_s(x_rae_clean, y)                  # [B, L, C_enc]

    # -----------------------------------------------------------------------
    # Phase 3: Pixel ODE — forward_pix (or forward_pix_with_cfg)
    # -----------------------------------------------------------------------
    t0_pix, t1_pix = transport_pix.check_interval(transport_pix.train_eps, transport_pix.sample_eps, eval=True)
    ss_pix = 1.0 - torch.linspace(t0_pix, t1_pix, num_steps + 1, device=device)

    x_pix = z_pix.detach().clone()

    for i in range(num_steps):
        s_curr = ss_pix[i]
        s_next = ss_pix[i + 1]
        ds = s_next - s_curr

        t_pix_scalar = get_t(s_curr, transport_pix.time_dist_shift)
        t_pix_batch = t_pix_scalar.repeat(B_full)
        dt_ds_pix = get_dt_ds(s_curr, transport_pix.time_dist_shift)

        if using_cfg:
            model_out_pix = model.forward_pix_with_cfg(
                x_pix, t_pix_batch, y, s_clean,
                cfg_scale=cfg_scale, cfg_interval=cfg_interval
            )
        elif using_ag and ag_phase in {"pixel", "both"}:
            good_pix = model.forward_pix(x_pix, t_pix_batch, y, s_clean)
            bad_pix = guidance_model.forward_pix(x_pix, t_pix_batch, y, s_clean)
            model_out_pix = apply_autoguidance(good_pix, bad_pix, t_pix_batch, model.in_channels)
        else:
            model_out_pix = model.forward_pix(x_pix, t_pix_batch, y, s_clean)

        drift_pix_dt = compute_ode_drift(transport_pix, x_pix, t_pix_batch, model_out_pix)
        x_pix = x_pix + drift_pix_dt * (dt_ds_pix * ds)

        if verbose and (i == 0 or i == num_steps - 1 or i % 10 == 0):
            print(f"[Pix Phase3] step {i} | s={s_curr:.4f} t_pix={t_pix_scalar.item():.4f}")

    return x_pix, x_rae_clean


# -----------------------------------------------------------------------------
# Checkpoint helpers (unchanged)
# -----------------------------------------------------------------------------

from eval import evaluate_generation_distributed

def save_checkpoint(path_str, step, epoch, model, ema_model, optimizer, scheduler):
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path_str), exist_ok=True)
    torch.save(state, path_str)


def load_checkpoint(path_str, model, ema_model, optimizer, scheduler):
    try:
        checkpoint = torch.load(path_str, map_location="cpu")
    except Exception as e:
        print(f"[ERROR] Checkpoint file is corrupt (incomplete save?): {e}", flush=True)
        sys.exit(2)
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


# -----------------------------------------------------------------------------
# Arg parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 transport model (project sequential version).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--results-dir", type=str, default="ckpts")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--global-seed", type=int, default=None)
    args = parser.parse_args()
    return args


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
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
    global_batch_size = training_cfg.get("global_batch_size", None)
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % world_size == 0
    else:
        batch_size = int(training_cfg.get("batch_size", 16))
        global_batch_size = batch_size * world_size * grad_accum_steps
    num_workers = int(training_cfg.get("num_workers", 4))
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 2500))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4))
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    default_seed = int(training_cfg.get("global_seed", 0))
    rae_loss_weight = float(training_cfg.get("rae_loss_weight", 0.0))

    if eval_config:
        do_eval = True
        eval_interval = int(eval_config.get("eval_interval", 5000))
        eval_model = eval_config.get("eval_model", False)
        eval_data = eval_config.get("data_path", None)
        reference_npz_path = eval_config.get("reference_npz_path", None)
        assert eval_data, "eval.data_path must be specified."
        assert reference_npz_path, "eval.reference_npz_path must be specified."
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
        raise ValueError("bf16 not supported on this device.")
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    autocast_enabled = use_fp16 or use_bf16
    autocast_kwargs = dict(dtype=autocast_dtype, enabled=autocast_enabled)
    scaler = GradScaler(enabled=use_fp16)

    # --- Transport params (no hardcoded eps overrides, no hardcoded logit-normal) ---
    transport_params_pix = dict(transport_cfg_pix.get("params", {}))
    transport_params_pix.pop("time_dist_shift", None)
    # Project version: use yaml time_dist_type as-is, no override
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

    def guidance_value(key, default):
        if key in guidance_cfg:
            return guidance_cfg[key]
        return guidance_cfg.get(key.replace("_", "-"), default)

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
        except Exception:
            print('RAE ENCODE compile error, falling back')
        try:
            model.forward = torch.compile(model.forward)
        except Exception:
            print('MODEL FORWARD compile error, falling back')
    else:
        print("Not compiling")
    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()
    model.train()
    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    model = ddp_model.module
    ddp_model.train()

    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")

    #### Opt, Sched init
    optimizer, optim_msg = build_optimizer([p for p in model.parameters() if p.requires_grad], training_cfg)
    scaler, autocast_kwargs = get_autocast_scaler(args)

    #### Data init
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
        logger.info(f"Eval dataset: {eval_data} ({len(eval_dataset)} images)")

    loader_batches = len(loader)
    if loader_batches % grad_accum_steps != 0:
        raise ValueError("loader batches must be divisible by grad_accum_steps.")
    steps_per_epoch = loader_batches // grad_accum_steps
    if steps_per_epoch <= 0:
        raise ValueError("grad_accum results in zero optimizer steps per epoch.")

    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    else:
        scheduler = None
        sched_msg = None

    #### Transport init
    transport_pix = create_transport(
        **transport_params_pix,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = Sampler(transport_pix)
    if sampler_mode_pix == "ODE":
        eval_sampler_pix = transport_sampler.sample_ode(**sampler_params_pix)
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
            raise ValueError("guidance_model config required for autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guidance_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    log_steps = 0
    running_loss = 0.0
    running_loss_pix = 0.0
    running_loss_rae_w = 0.0
    start_time = time()
    use_guidance = guidance_scale > 1.0
    # Fixed noise for visual sampling
    zs = torch.randn(micro_batch_size, *latent_size, device=device, dtype=torch.float32)
    zs_rae = torch.randn(micro_batch_size, *latent_size_rae, device=device, dtype=torch.float32)
    n = micro_batch_size
    if use_guidance:
        zs = torch.cat([zs, zs], dim=0)
        zs_rae = torch.cat([zs_rae, zs_rae], dim=0)

    ### Resume / checkpoint
    start_epoch = 0
    global_step = 0
    maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
    if maybe_resume_ckpt_path is not None:
        logger.info(f"Resuming from {maybe_resume_ckpt_path}...")
        ckpt_path = Path(maybe_resume_ckpt_path)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                str(ckpt_path), ddp_model, ema_model, optimizer, scheduler
            )
            logger.info(f"[Rank {rank}] Resumed (epoch={start_epoch}, step={global_step}).")
            if scheduler is not None:
                sched_cfg_cur = dict(training_cfg.get("scheduler", {}))
                opt_cfg_cur = dict(training_cfg.get("optimizer", {}))
                config_base_lr = float(sched_cfg_cur.get("base_lr", opt_cfg_cur.get("lr", scheduler.base_lrs[0])))
                if abs(scheduler.base_lrs[0] - config_base_lr) > 1e-12:
                    logger.info(f"[Rank {rank}] Scheduler base_lr overridden: {scheduler.base_lrs[0]:.2e} -> {config_base_lr:.2e}")
                scheduler.base_lrs = [config_base_lr] * len(scheduler.base_lrs)
                current_lrs = scheduler.get_lr()
                for param_group, lr in zip(optimizer.param_groups, current_lrs):
                    param_group["lr"] = lr
                scheduler._last_lr = list(current_lrs)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")

    ### Log details
    if rank == 0:
        num_params = sum(p.numel() for p in rae.parameters())
        logger.info(f"Stage-1 RAE parameters: {num_params/1e6:.2f}M")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Stage-2 Model trainable parameters: {num_params/1e6:.2f}M")
        if clip_grad is not None:
            logger.info(f"Gradient clipping: max_norm={clip_grad}")
        logger.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler.")
        logger.info(f"Training {num_epochs} epochs, micro_batch={micro_batch_size}/GPU.")
        logger.info(f"[Sequential project] time_dist_shift_pix={time_dist_shift:.4f}, "
                    f"time_dist_shift_rae={time_dist_shift_rae:.4f}")

    dist.barrier()
    for epoch in range(start_epoch, num_epochs):
        model.train()
        model.ddt.eval()   # keep frozen DDT backbone in eval mode
        sampler.set_epoch(epoch)
        epoch_metrics = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        optimizer.zero_grad()

        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0 and epoch > 0 and rank == 0:
            logger.info(f"Saving checkpoint at epoch {epoch}...")
            ckpt_path_str = f"{checkpoint_dir}/ep-{epoch:07d}.pt"
            save_checkpoint(ckpt_path_str, global_step, epoch, ddp_model, ema_model, optimizer, scheduler)

        # FID eval trigger: write .fid_needed signal file at every 10th epoch if not yet evaluated.
        # NOTE: sys.exit(3) inside DDP is NOT propagated by torchrun (torchrun exits with code 1
        # regardless). The retry script detects FID need via the .fid_needed file, not exit code.
        if epoch % 10 == 0 and epoch > 0:
            dist.barrier()  # ensure rank-0 checkpoint save is flushed to filesystem
            marker_path = os.path.join(checkpoint_dir, f"ep-{epoch:07d}.fid_done")
            should_exit = torch.zeros(1, device=device, dtype=torch.int32)
            if rank == 0 and not os.path.exists(marker_path):
                should_exit.fill_(1)
            dist.broadcast(should_exit, src=0)
            if should_exit.item() == 1:
                if rank == 0:
                    fid_needed_path = os.path.join(checkpoint_dir, f"ep-{epoch:07d}.fid_needed")
                    open(fid_needed_path, 'w').close()
                    logger.info(f"[FID] Epoch {epoch}: signal file written, exiting for FID evaluation.")
                cleanup_distributed()
                sys.exit(3)

        for step, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            # Skip RAE encoding entirely in pixel_only mode
            if not model.pixel_only:
                with torch.no_grad():
                    z = rae.encode(images)
                x_rae = z
            else:
                x_rae = None

            # Pixel input: [0, 1] range (no 2x-1 normalisation — project baseline)
            x_pix = images

            optimizer.zero_grad(set_to_none=True)
            model_kwargs = dict(y=labels)

            with autocast(**autocast_kwargs):
                loss_pix, loss_rae = project_training_losses(
                    ddp_model,
                    x_pix,
                    x_rae,
                    transport_pix,
                    transport_rae,
                    model_kwargs
                )
                loss = loss_pix.mean() + rae_loss_weight * loss_rae.mean()
                if torch.isnan(loss):
                    print(f"[ERROR] NaN loss at epoch {epoch} step {global_step}. Exiting.", flush=True)
                    sys.exit(1)

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
            running_loss_pix += loss_pix.mean().item()
            running_loss_rae_w += (rae_loss_weight * loss_rae).mean().item()
            epoch_metrics['loss'] += loss.detach()
            epoch_metrics['loss_pix'] += loss_pix.mean().detach()
            epoch_metrics['loss_rae'] += loss_rae.mean().detach()

            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                avg_loss = running_loss / log_interval
                stats = {
                    "train/loss": avg_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                if rae_loss_weight > 0:
                    stats["train/loss_pix"] = running_loss_pix / log_interval
                    stats["train/loss_rae_w"] = running_loss_rae_w / log_interval
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if args.wandb:
                    wandb_utils.log(stats, step=global_step)
                running_loss = 0.0
                running_loss_pix = 0.0
                running_loss_rae_w = 0.0

            if global_step % sample_every == 0:
                model.eval()
                logger.info("Generating EMA samples (sequential project mode)...")
                with torch.no_grad():
                    n_vis = min(9, micro_batch_size)

                    if use_guidance:
                        zs_samples = torch.cat([zs[:n_vis], zs[n:n + n_vis]], dim=0)
                        zs_rae_samples = torch.cat([zs_rae[:n_vis], zs_rae[n:n + n_vis]], dim=0)
                        y_cond = labels[:n_vis]
                        y_uncond = torch.full_like(y_cond, null_label)
                        vis_y = torch.cat([y_cond, y_uncond], dim=0)
                    else:
                        zs_samples = zs[:n_vis]
                        zs_rae_samples = zs_rae[:n_vis]
                        vis_y = labels[:n_vis]

                    # Sequential project sampler — pass model object directly
                    samples_pix, samples_rae = project_sample_ode_naive(
                        zs_samples,
                        zs_rae_samples,
                        ema_model,
                        transport_pix=transport_pix,
                        transport_rae=transport_rae,
                        num_steps=sampler_params_pix.get("num_steps", 50),
                        cfg_scale=guidance_scale,
                        cfg_interval=(t_min, t_max),
                        y=vis_y,
                    )

                    samples_pix = samples_pix.float()

                    if use_guidance:
                        samples_pix, _ = samples_pix.chunk(2, dim=0)

                    # [0, 1] range — just clamp (no (x+1)/2 shift)
                    samples = samples_pix.cpu().float().clamp(0, 1)

                    dist.barrier()
                    print(f"Rank{rank}")
                    if args.wandb and rank == 0:
                        wandb_utils.log_image(samples, global_step, key="samples/pixel", caption="Pixel Stream Samples")
                        if samples_rae is not None:
                            samples_rae = samples_rae.float()
                            if use_guidance:
                                samples_rae, _ = samples_rae.chunk(2, dim=0)
                            decoded_rae = rae.decode(samples_rae)
                            decoded_rae = decoded_rae.cpu().float().clamp(0, 1)
                            wandb_utils.log_image(decoded_rae, global_step, key="samples/rae", caption="RAE Stream Samples")
                logger.info("EMA sample generation done.")
                model.train()
                model.ddt.eval()

            if do_eval and (eval_interval > 0 and global_step % eval_interval == 0):
                logger.warning("Evaluation disabled for sequential PixelDDT (pending refactor).")

            global_step += 1
            num_batches += 1

        if rank == 0 and num_batches > 0:
            avg_loss = epoch_metrics['loss'].item() / num_batches
            epoch_stats = {"epoch/loss": avg_loss}
            logger.info(f"[Epoch {epoch}] " + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items()))
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)

    # Save final checkpoint
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        save_checkpoint(
            f"{checkpoint_dir}/ep-last.pt",
            global_step, num_epochs, ddp_model, ema_model, optimizer, scheduler
        )
    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
