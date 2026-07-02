# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
sample_pix_ddt_project.py — DDP sampling script for the sequential-denoising
PixelDDT (project version).

Inference runs in three phases:
  Phase 1 — RAE ODE (num_steps Euler steps)  z_rae → z_rae_clean
  Phase 2 — Extract conditioning              z_rae_clean → s_clean
  Phase 3 — Pixel ODE (num_steps Euler steps) z_pix → x_pix_final

Reverted from "Pro" changes:
  - Pixel output range: [0, 1]  (samples.clamp(0, 1), no (x+1)/2)
  - No train_eps / sample_eps hardcoded to 0
  - time_dist_type read from yaml, not hardcoded

Usage (from RAE/ root):
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \\
        src/sample_pix_ddt_project.py \\
        --config configs/stage2/sampling/ImageNet256/PixDDT_small_v2_lora_project.yaml \\
        --ckpt /path/to/ep-0000050.pt \\
        --sample-dir ~/data/samples/project \\
        --num-fid-samples 50000 \\
        --per-proc-batch-size 64 \\
        --precision fp32 \\
        --label-sampling equal \\
        --class-ids 84 291 292 340 388 579 817 907 953 985 \\
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

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_SRC_DIR)
for _p in [_SRC_DIR, _PROJ_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport

# Import project-specific sequential sampler from the free_layer training script
# (supports pixel_only mode — skips Phase 1 & 2 when rae_inject_layers=[] and rae_inject_final_layer=False)
from train_stage3_projcet_free_layer import project_sample_ode_naive  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (identical to sample_pix_ddt.py)
# ---------------------------------------------------------------------------

def create_npz_from_sample_folder(sample_dir: str, num: int) -> str:
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
    class_ids: list[int] | None,
    num_fid_samples: int,
    total_samples: int,
    batch_size: int,
    device: torch.device,
    rank: int,
    world_size: int,
    iterations: int,
    seed: int,
) -> Callable[[int], torch.Tensor]:
    if class_ids is None:
        class_pool = torch.arange(num_classes, dtype=torch.long)
    else:
        if not class_ids:
            raise ValueError("--class-ids must contain at least one class ID.")
        if len(set(class_ids)) != len(class_ids):
            raise ValueError(f"--class-ids contains duplicates: {class_ids}")
        invalid_ids = [class_id for class_id in class_ids if not 0 <= class_id < num_classes]
        if invalid_ids:
            raise ValueError(
                f"Class IDs must be in [0, {num_classes - 1}], got: {invalid_ids}"
            )
        class_pool = torch.tensor(class_ids, dtype=torch.long)

    if sampling_mode == "random":
        def random_sampler(_step_idx: int) -> torch.Tensor:
            indices = torch.randint(0, class_pool.numel(), (batch_size,), device=device)
            return class_pool.to(device)[indices]
        return random_sampler

    if sampling_mode == "equal":
        selected_class_count = class_pool.numel()
        if num_fid_samples % selected_class_count != 0:
            raise ValueError(
                f"Equal label sampling requires num_fid_samples ({num_fid_samples}) "
                f"to be divisible by the number of selected classes "
                f"({selected_class_count})."
            )
        labels_per_class = num_fid_samples // selected_class_count
        base_pool = class_pool.repeat_interleave(labels_per_class)

        generator = torch.Generator()
        generator.manual_seed(seed)
        perm = torch.randperm(base_pool.numel(), generator=generator)
        base_pool = base_pool[perm]

        if total_samples > num_fid_samples:
            tail_indices = torch.randint(
                0,
                selected_class_count,
                (total_samples - num_fid_samples,),
                generator=generator,
            )
            tail = class_pool[tail_indices]
            global_pool = torch.cat([base_pool, tail], dim=0)
        else:
            global_pool = base_pool

        expected_total = iterations * batch_size * world_size
        if global_pool.numel() != expected_total:
            raise ValueError(
                f"Label pool has {global_pool.numel()} entries, expected {expected_total}."
            )

        # PNG indices are assigned in (step, local_idx, rank) order. Slice the
        # label pool in the same order so truncating a rounded-up tail preserves
        # exact class counts in the first num_fid_samples images.
        device_pool = global_pool.view(iterations, batch_size, world_size)[:, :, rank]

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
        _rae_config,
        model_config,
        transport_config_pix,
        transport_config_rae,
        sampler_config_pix,
        _sampler_config_rae,
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

    # ---- Latent sizes & time-dist shifts (mirror train_stage3_projcet.py) ----
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

    # ---- Build PixelDDT model (project version) ----
    # A training checkpoint contains the complete PixelDDT state, including the
    # frozen DDT. Avoid requiring the YAML's original stage2_ckpt path when a
    # full checkpoint will immediately replace every parameter below.
    if args.ckpt is not None:
        model_config["params"]["load_stage2_ckpt"] = False
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

    # ---- Build transports (mirror train_stage3_projcet.py — no hardcoded eps) ----
    transport_params_pix = dict(transport_pix_d.get("params", {}))
    transport_params_pix.pop("time_dist_shift", None)
    # Project version: use yaml time_dist_type as-is, no eps override

    transport_params_rae = dict(transport_rae_d.get("params", {}))
    transport_params_rae.pop("time_dist_shift", None)

    transport_pix = create_transport(**transport_params_pix, time_dist_shift=time_dist_shift_pix)
    transport_rae = create_transport(**transport_params_rae, time_dist_shift=time_dist_shift_rae)

    # ---- Sampler settings ----
    sampler_params = dict(sampler_pix_d.get("params", {}))
    num_steps = args.num_steps if args.num_steps is not None else sampler_params.get("num_steps", 50)

    # ---- Guidance settings ----
    guidance_scale = float(guidance_d.get("scale", 1.0))
    if args.cfg_scale is not None:
        guidance_scale = float(args.cfg_scale)

    def _gval(key: str, default: float) -> float:
        if key in guidance_d:
            return float(guidance_d[key])
        return float(guidance_d.get(key.replace("_", "-"), default))

    t_min = _gval("t_min", 0.0)
    if args.t_min is not None:
        t_min = float(args.t_min)
    t_max = _gval("t_max", 1.0)
    if args.t_max is not None:
        t_max = float(args.t_max)
    using_cfg = guidance_scale > 1.0
    s_clean_cfg_mode = args.s_clean_cfg_mode

    # ---- Output folder ----
    ckpt_name = (
        "pretrained" if args.ckpt is None
        else os.path.splitext(os.path.basename(args.ckpt))[0]
    )
    folder_name = os.environ.get(
        "SAVE_FOLDER",
        f"PixelDDT-project-{ckpt_name}-cfg{guidance_scale:.2f}"
        f"-sclean-{s_clean_cfg_mode}"
        f"-bs{args.per_proc_batch_size}-euler{num_steps}-{args.precision}",
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

    label_sampler = build_label_sampler(
        sampling_mode=args.label_sampling,
        num_classes=num_classes,
        class_ids=args.class_ids,
        num_fid_samples=args.num_fid_samples,
        total_samples=total_samples,
        batch_size=n,
        device=device,
        rank=rank,
        world_size=world_size,
        iterations=iterations,
        seed=args.global_seed,
    )

    # ---- Sampling loop ----
    pbar = tqdm(range(iterations), desc=f"Rank {rank}") if rank == 0 else range(iterations)
    total_saved = (num_already_saved // world_size) * world_size

    for step_idx in pbar:
        with autocast(**autocast_kwargs):
            z_pix = torch.randn(n, *latent_size_pix, device=device)
            z_rae = torch.randn(n, *latent_size_rae, device=device)
            y = label_sampler(step_idx)

            if using_cfg:
                z_pix = torch.cat([z_pix, z_pix], dim=0)
                z_rae = torch.cat([z_rae, z_rae], dim=0)
                y_null = torch.full((n,), null_label, device=device)
                y_cfg = torch.cat([y, y_null], dim=0)
                sample_y = y_cfg
            else:
                sample_y = y

            # Sequential project sampler: Phase 1 → Phase 2 → Phase 3
            samples_pix, _ = project_sample_ode_naive(
                z_pix,
                z_rae,
                model,
                transport_pix=transport_pix,
                transport_rae=transport_rae,
                num_steps=num_steps,
                cfg_scale=guidance_scale,
                cfg_interval=(t_min, t_max),
                rae_cfg_scale=args.rae_cfg_scale,
                s_clean_cfg_mode=s_clean_cfg_mode,
                verbose=False,
                y=sample_y,
            )

            if using_cfg:
                samples_pix, _ = samples_pix.chunk(2, dim=0)

            # Output range [0, 1] — project baseline (no (x+1)/2)
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
        description="DDP sampler for sequential-denoising PixelDDT (project version)."
    )
    parser.add_argument("--config",  type=str, required=True,
                        help="Path to sampling YAML config.")
    parser.add_argument("--ckpt",    type=str, default=None,
                        help="Path to trained PixelDDT checkpoint (ep-*.pt).")
    parser.add_argument("--sample-dir", type=str, default="samples",
                        help="Root directory under which sample folders are created.")
    parser.add_argument("--per-proc-batch-size", type=int, default=32,
                        help="Number of images generated per GPU per step.")
    parser.add_argument("--num-fid-samples", type=int, default=50_000,
                        help="Total number of images to generate.")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="ODE steps per phase (overrides config value).")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--label-sampling", type=str, choices=["random", "equal"], default="equal")
    parser.add_argument(
        "--class-ids",
        type=int,
        nargs="+",
        default=None,
        help="Restrict label sampling to these ImageNet class IDs (default: all classes).",
    )
    parser.add_argument("--cfg-scale", type=float, default=None,
                        help="Override CFG guidance scale from config.")
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--s-clean-cfg-mode", type=str, choices=["shared", "split", "null", "traj"], default="shared",
                        help="How CFG builds s_clean for cond/uncond Pixel batches. "
                             "traj = run Phase 1 as two independent cond/uncond RAE trajectories "
                             "(uncond s_clean from a genuinely unconditional latent).")
    parser.add_argument("--rae-cfg-scale", type=float, default=None,
                        help="Decouple Phase-1 (RAE) CFG scale from the Pixel CFG scale. "
                             "None (default) = couple to --cfg-scale (current behaviour); "
                             "1.0 = RAE-CFG OFF (Phase 1 = a single natural conditional latent), "
                             "while Pixel CFG still uses --cfg-scale. Ignored when mode=traj.")
    parser.add_argument("--naive", action="store_true",
                        help="Ignored (project version always uses Euler; kept for CLI compat).")

    args = parser.parse_args()
    main(args)
