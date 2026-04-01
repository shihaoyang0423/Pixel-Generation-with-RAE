# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
DDP sampling script for PixelDDT (Dual-Stream pixel diffusion model).

Generates N images using joint_sample_ode over the pixel+RAE dual streams,
saves as PNG files and builds a single .npz for FID evaluation.

Usage (from RAE/ root):
    torchrun --standalone --nnodes=1 --nproc_per_node=8 \\
        src/sample_pix_ddp.py \\
        --config configs/stage2/sampling/ImageNet256/PixelDDT-S.yaml \\
        --ckpt /path/to/ep-0000099.pt \\
        --sample-dir samples/PixelDDT-S-ep99 \\
        --num-fid-samples 50000 \\
        --per-proc-batch-size 32 \\
        --precision fp32 \\
        --label-sampling equal \\
        --naive
"""

import argparse
import math
import os
import sys
from collections.abc import Callable

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast
from tqdm import tqdm
from omegaconf import OmegaConf

# Ensure src/ is on sys.path so package imports work whether this file is run
# directly (python src/sample_pix_ddp.py) or via torchrun.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_SRC_DIR)
for _p in [_SRC_DIR, _PROJ_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport

# joint_sample_ode lives in train_stage3; importing at module level is safe
# because everything in train_stage3 is guarded by if __name__ == "__main__".
from train_stage3 import joint_sample_ode  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (identical logic to sample_ddp.py)
# ---------------------------------------------------------------------------

def create_npz_from_sample_folder(sample_dir: str, num: int) -> str:
    """Collect PNG files and pack them into a single .npz for FID tools."""
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        img = Image.open(f"{sample_dir}/{i:06d}.png")
        samples.append(np.asarray(img).astype(np.uint8))
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3), \
        f"Unexpected shape: {samples.shape}"
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def build_label_sampler(
    sampling_mode: str,
    num_classes: int,
    num_fid_samples: int,
    total_samples: int,
    samples_needed_this_device: int,
    batch_size: int,
    device: torch.device,
    rank: int,
    iterations: int,
    seed: int,
) -> Callable[[int], torch.Tensor]:
    """Return a callable that yields a batch of class labels for each step."""

    if sampling_mode == "random":
        def random_sampler(_step_idx: int) -> torch.Tensor:
            return torch.randint(0, num_classes, (batch_size,), device=device)
        return random_sampler

    if sampling_mode == "equal":
        if num_fid_samples % num_classes != 0:
            raise ValueError(
                f"Equal label sampling requires num_fid_samples ({num_fid_samples}) "
                f"to be divisible by num_classes ({num_classes})."
            )
        labels_per_class = num_fid_samples // num_classes
        base_pool = torch.arange(num_classes, dtype=torch.long).repeat_interleave(labels_per_class)

        generator = torch.Generator()
        generator.manual_seed(seed)
        perm = torch.randperm(base_pool.numel(), generator=generator)
        base_pool = base_pool[perm]

        if total_samples > num_fid_samples:
            tail = torch.randint(
                0, num_classes, (total_samples - num_fid_samples,), generator=generator
            )
            global_pool = torch.cat([base_pool, tail], dim=0)
        else:
            global_pool = base_pool

        start = rank * samples_needed_this_device
        device_pool = global_pool[start : start + samples_needed_this_device]
        device_pool = device_pool.view(iterations, batch_size)

        def equal_sampler(step_idx: int) -> torch.Tensor:
            return device_pool[step_idx].to(device)

        return equal_sampler

    raise ValueError(f"Unknown label sampling mode: {sampling_mode!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("DDP sampling requires at least one GPU.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    # ---- DDP setup ----
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 requested but not supported on this device.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    # ---- Parse dual-stream config ----
    cfg = OmegaConf.load(args.config)
    (
        _rae_config,        # unused at inference – PixelDDT embeds its own DDT backbone
        model_config,
        transport_config_pix,
        transport_config_rae,
        sampler_config_pix,
        _sampler_config_rae,  # both streams share num_steps set by --num-steps
        guidance_config,
        misc_config_pix,
        misc_config_rae,
        _training_config,
        _eval_config,
    ) = parse_configs(cfg)

    def to_dict(x):
        return {} if x is None else OmegaConf.to_container(x, resolve=True)

    misc_pix        = to_dict(misc_config_pix)
    misc_rae        = to_dict(misc_config_rae)
    transport_pix_d = to_dict(transport_config_pix)
    transport_rae_d = to_dict(transport_config_rae)
    sampler_pix_d   = to_dict(sampler_config_pix)
    guidance_d      = to_dict(guidance_config)

    # ---- Latent sizes & time-dist shifts (mirror train_stage3.py exactly) ----
    latent_size_pix = tuple(int(d) for d in misc_pix.get("latent_size", (3, 256, 256)))
    latent_size_rae = tuple(int(d) for d in misc_rae.get("latent_size", (768, 16, 16)))

    shift_dim_pix  = misc_pix.get("time_dist_shift_dim", math.prod(latent_size_pix))
    shift_base_pix = misc_pix.get("time_dist_shift_base", 4096)
    time_dist_shift_pix = math.sqrt(shift_dim_pix / shift_base_pix)

    shift_dim_rae  = misc_rae.get("time_dist_shift_dim", shift_dim_pix)
    shift_base_rae = misc_rae.get("time_dist_shift_base", shift_base_pix)
    time_dist_shift_rae = math.sqrt(shift_dim_rae / shift_base_rae)

    num_classes = int(misc_pix.get("num_classes", 1000))
    null_label  = int(misc_pix.get("null_label", num_classes))

    # if rank == 0:
    #     print(
    #         f"latent_size_pix={latent_size_pix}, latent_size_rae={latent_size_rae}\n"
    #         f"time_dist_shift_pix={time_dist_shift_pix:.4f}, "
    #         f"time_dist_shift_rae={time_dist_shift_rae:.4f}"
    #     )

    # ---- Build PixelDDT model ----
    # instantiate_from_config loads the frozen DDT backbone (stage2_ckpt in yaml).
    # Then --ckpt loads the trained PixelDDT weights (EMA or model key).
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)

    if args.ckpt is not None:
        ckpt_data = torch.load(args.ckpt, map_location="cpu")
        if "ema" in ckpt_data:
            state_dict = ckpt_data["ema"]
            if rank == 0:
                print(f"Loading EMA weights from {args.ckpt}")
        elif "model" in ckpt_data:
            state_dict = ckpt_data["model"]
            if rank == 0:
                print(f"Loading model weights from {args.ckpt}")
        else:
            state_dict = ckpt_data
            if rank == 0:
                print(f"Loading raw state dict from {args.ckpt}")
        model.load_state_dict(state_dict, strict=True)
    else:
        if rank == 0:
            print("WARNING: --ckpt not provided; using randomly initialised PixelDDT weights.")

    model.eval()

    # ---- Build transports (mirror train_stage3.py) ----
    transport_params_pix = dict(transport_pix_d.get("params", {}))
    transport_params_pix.pop("time_dist_shift", None)  # injected explicitly below

    transport_params_rae = dict(transport_rae_d.get("params", {}))
    transport_params_rae.pop("time_dist_shift", None)

    transport_pix = create_transport(**transport_params_pix, time_dist_shift=time_dist_shift_pix)
    transport_rae = create_transport(**transport_params_rae, time_dist_shift=time_dist_shift_rae)

    # ---- Sampler settings ----
    sampler_params = dict(sampler_pix_d.get("params", {}))
    num_steps = args.num_steps if args.num_steps is not None else sampler_params.get("num_steps", 50)
    # "naive" = fixed-step Euler (reliable for DDP); "dopri5" = adaptive (requires torchdiffeq)
    sampler_type = "naive" if args.naive else "dopri5"

    # ---- Guidance settings ----
    guidance_scale = float(guidance_d.get("scale", 1.0))
    if args.cfg_scale is not None:
        guidance_scale = float(args.cfg_scale)

    def _gval(key: str, default: float) -> float:
        """Read a guidance key that may be spelled with _ or -."""
        if key in guidance_d:
            return float(guidance_d[key])
        return float(guidance_d.get(key.replace("_", "-"), default))

    t_min = _gval("t_min", 0.0)
    t_max = _gval("t_max", 1.0)
    using_cfg = guidance_scale > 1.0

    # if rank == 0:
    #     print(
    #         f"guidance_scale={guidance_scale}, using_cfg={using_cfg}, "
    #         f"sampler_type={sampler_type}, num_steps={num_steps}, precision={args.precision}"
    #     )

    # ---- Output folder ----
    ckpt_name = (
        "pretrained" if args.ckpt is None
        else os.path.splitext(os.path.basename(args.ckpt))[0]
    )
    folder_name = os.environ.get(
        "SAVE_FOLDER",
        f"PixelDDT-{ckpt_name}-cfg{guidance_scale:.2f}"
        f"-bs{args.per_proc_batch_size}-{sampler_type}{num_steps}-{args.precision}",
    )
    sample_folder_dir = os.path.join(args.sample_dir, folder_name)

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving PNG samples to: {sample_folder_dir}")
    dist.barrier()

    # ---- Sampling bookkeeping ----
    n = args.per_proc_batch_size
    global_batch_size = n * world_size

    existing = [
        f for f in os.listdir(sample_folder_dir)
        if f.endswith(".png") and os.path.isfile(os.path.join(sample_folder_dir, f))
    ]
    num_already_saved = len(existing)

    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if total_samples % world_size != 0:
        raise ValueError("total_samples must be divisible by world_size.")
    samples_per_gpu = total_samples // world_size
    if samples_per_gpu % n != 0:
        raise ValueError("samples_per_gpu must be divisible by per_proc_batch_size.")
    iterations = samples_per_gpu // n

    # if rank == 0:
    #     print(
    #         f"num_fid_samples={args.num_fid_samples}, total_samples={total_samples}, "
    #         f"iterations={iterations} per GPU, already saved={num_already_saved}"
    #     )

    label_sampler = build_label_sampler(
        sampling_mode=args.label_sampling,
        num_classes=num_classes,
        num_fid_samples=args.num_fid_samples,
        total_samples=total_samples,
        samples_needed_this_device=samples_per_gpu,
        batch_size=n,
        device=device,
        rank=rank,
        iterations=iterations,
        seed=args.global_seed,
    )

    # ---- Sampling loop ----
    pbar = tqdm(range(iterations), desc=f"Rank {rank}") if rank == 0 else range(iterations)
    total_saved = (num_already_saved // world_size) * world_size  # align across ranks

    for step_idx in pbar:
        with autocast(**autocast_kwargs):
            z_pix = torch.randn(n, *latent_size_pix, device=device)
            z_rae = torch.randn(n, *latent_size_rae, device=device)
            y = label_sampler(step_idx)

            if using_cfg:
                # Duplicate tensors: first half = conditional, second = unconditional
                z_pix = torch.cat([z_pix, z_pix], dim=0)
                z_rae = torch.cat([z_rae, z_rae], dim=0)
                y_null = torch.full((n,), null_label, device=device)
                y_cfg = torch.cat([y, y_null], dim=0)
                model_kwargs = dict(
                    y=y_cfg,
                    cfg_scale=guidance_scale,
                    cfg_interval=(t_min, t_max),
                )
                model_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                model_fn = model.forward

            # Dual-stream ODE integration
            # joint_sample_ode returns (x_pix_final, x_rae_final)
            # x_pix_final IS the generated image (pixel space, [0,1] range)
            samples_pix, _ = joint_sample_ode(
                z_pix,
                z_rae,
                model_fn,
                transport_pix=transport_pix,
                transport_rae=transport_rae,
                sampler_type=sampler_type,
                num_steps=num_steps,
                verbose=False,
                **model_kwargs,
            )

            if using_cfg:
                # Take only the conditional half
                samples_pix, _ = samples_pix.chunk(2, dim=0)

            # Convert to uint8 numpy for saving
            samples_np = (
                samples_pix.clamp(0, 1)
                .mul(255)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()
            )

        for local_idx, sample in enumerate(samples_np):
            index = local_idx * world_size + rank + total_saved
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

        total_saved += global_batch_size
        dist.barrier()

    # ---- Finalise ----
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DDP sampler for PixelDDT (dual-stream pixel diffusion)."
    )
    parser.add_argument("--config",  type=str, required=True,
                        help="Path to sampling YAML config (must contain transport_rae section).")
    parser.add_argument("--ckpt",    type=str, default=None,
                        help="Path to trained PixelDDT checkpoint (ep-*.pt). "
                             "Loads 'ema' key if present, otherwise 'model'.")
    parser.add_argument("--sample-dir", type=str, default="samples",
                        help="Root directory under which sample folders are created.")
    parser.add_argument("--per-proc-batch-size", type=int, default=32,
                        help="Number of images generated per GPU per step.")
    parser.add_argument("--num-fid-samples", type=int, default=50_000,
                        help="Total number of images to generate.")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="ODE steps (overrides config value).")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--label-sampling", type=str, choices=["random", "equal"], default="equal",
                        help="'equal' ensures each class appears num_fid_samples/1000 times.")
    parser.add_argument("--cfg-scale", type=float, default=None,
                        help="Override CFG guidance scale from config.")
    parser.add_argument("--naive", action="store_true",
                        help="Use naive fixed-step Euler sampler (default: dopri5 adaptive ODE).")

    args = parser.parse_args()
    main(args)
