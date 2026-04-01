"""
Overfitting sanity check for PixelDDT (single-image, supports DDP).

Train on one ImageNet image until loss converges, then sample conditioned on
that image's class label.

Usage (from RAE/ root):
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        src/overfit_single.py \
        --config configs/stage2/training/x0_pred_768.yaml \
        --data-path /data/shihaoyang/data/imagenet/256/train \
        --output-dir overfit_out_ddp \
        --train-steps 10000 \
        --lr 1e-3
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from omegaconf import OmegaConf
from time import time

# ── reuse existing infrastructure ─────────────────────────────────────────────
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs, center_crop_arr
from train_stage3 import joint_training_losses, joint_sample_ode
from utils.dist_utils import setup_distributed, cleanup_distributed

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    p = argparse.ArgumentParser(description="Overfit PixelDDT on a single image.")
    p.add_argument("--config",      type=str, required=True)
    p.add_argument("--data-path",   type=str, required=True, help="ImageNet train dir (ImageFolder layout)")
    p.add_argument("--output-dir",  type=str, default="overfit_out")
    p.add_argument("--train-steps", type=int, default=10000)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--log-every",   type=int, default=200)
    p.add_argument("--num-samples", type=int, default=4,  help="How many independent samples to draw at the end")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--image-size",  type=int, default=256, choices=[256, 512])
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()
    is_main = rank == 0
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    seed = args.seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # ── parse config ──────────────────────────────────────────────────────────
    full_cfg = OmegaConf.load(args.config)
    (
        rae_config, model_config,
        transport_config_pix, transport_config_rae,
        sampler_config_pix,   sampler_config_rae,
        guidance_config,
        misc_config_pix, misc_config_rae,
        training_config, eval_config,
    ) = parse_configs(full_cfg)

    def to_dict(cfg):
        return OmegaConf.to_container(cfg, resolve=True) if cfg else {}

    misc_pix          = to_dict(misc_config_pix)
    misc_rae          = to_dict(misc_config_rae)
    transport_params_pix = dict(to_dict(transport_config_pix).get("params", {}))
    transport_params_rae = dict(to_dict(transport_config_rae).get("params", {}))
    sampler_params_pix   = to_dict(sampler_config_pix).get("params", {})

    latent_size     = tuple(int(d) for d in misc_pix.get("latent_size", (3, 256, 256)))
    latent_size_rae = tuple(int(d) for d in misc_rae.get("latent_size", (768, 16, 16)))
    shift_pix = math.sqrt(
        misc_pix.get("time_dist_shift_dim", math.prod(latent_size)) /
        misc_pix.get("time_dist_shift_base", 4096)
    )
    shift_rae = math.sqrt(
        misc_rae.get("time_dist_shift_dim", math.prod(latent_size_rae)) /
        misc_rae.get("time_dist_shift_base", 4096)
    )

    # ── build models ──────────────────────────────────────────────────────────
    if is_main:
        print("Loading RAE...")
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    if is_main:
        print("Loading PixelDDT...")
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    model.train()
    if dist.is_initialized():
        ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    else:
        ddp_model = model

    trainable = [p for p in ddp_model.parameters() if p.requires_grad]
    if is_main:
        print(f"Trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M")

    # ── transport objects ─────────────────────────────────────────────────────
    transport_params_pix.pop("time_dist_shift", None)
    transport_params_rae.pop("time_dist_shift", None)
    transport_pix = create_transport(**transport_params_pix, time_dist_shift=shift_pix)
    transport_rae = create_transport(**transport_params_rae, time_dist_shift=shift_rae)

    # ── grab the very first image from the dataset ────────────────────────────
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    img_tensor, label = dataset[0]
    img_tensor   = img_tensor.unsqueeze(0).to(device)       # [1, 3, 256, 256]
    label_tensor = torch.tensor([label], device=device)

    if is_main:
        print(f"Target image: class={label} ({dataset.classes[label]}), shape={list(img_tensor.shape)}")
        save_image(img_tensor, os.path.join(args.output_dir, "gt.png"))
        print(f"  ground-truth saved to {args.output_dir}/gt.png")

    with torch.no_grad():
        z = rae.encode(img_tensor)  # [1, 768, 16, 16]

    # ── optimizer — no scheduler, just overfit ────────────────────────────────
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    # ── training loop ─────────────────────────────────────────────────────────
    if is_main:
        print(f"\nTraining for {args.train_steps} steps (lr={args.lr})...")
    t0 = time()
    for step in range(args.train_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = joint_training_losses(
            ddp_model, img_tensor, z,
            transport_pix, transport_rae,
            model_kwargs=dict(y=label_tensor),
        ).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if is_main and (step % args.log_every == 0 or step == args.train_steps - 1):
            print(f"  step {step:6d}/{args.train_steps}  loss={loss.item():.5f}  "
                  f"elapsed={time()-t0:.0f}s")

    if dist.is_initialized():
        dist.barrier()
    if is_main:
        print("\nTraining done. Sampling...")

    # ── sampling ──────────────────────────────────────────────────────────────
    ddp_model.eval()
    model = ddp_model.module
    num_steps = sampler_params_pix.get("num_steps", 50)
    N = args.num_samples if is_main else 1

    torch.manual_seed(args.seed)
    z_pix_noise = torch.randn(N, *latent_size,     device=device)
    z_rae_noise = torch.randn(N, *latent_size_rae, device=device)
    labels_rep  = label_tensor.repeat(N)  # same class for all samples

    samples_pix, samples_rae = joint_sample_ode(
        z_pix=z_pix_noise,
        z_rae=z_rae_noise,
        model_fn=model.forward,
        transport_pix=transport_pix,
        transport_rae=transport_rae,
        sampler_type="naive",
        num_steps=num_steps,
        y=labels_rep,
    )

    if is_main:
        # pixel stream output
        samples_pix = samples_pix.float().clamp(0, 1)
        save_image(samples_pix, os.path.join(args.output_dir, "sample_pixel.png"), nrow=N)

        # RAE stream output (decoded through RAE decoder)
        with torch.no_grad():
            decoded_rae = rae.decode(samples_rae.float()).clamp(0, 1)
        save_image(decoded_rae, os.path.join(args.output_dir, "sample_rae.png"), nrow=N)

        print(f"\nResults saved to {args.output_dir}/")
        print(f"  gt.png           — the training image")
        print(f"  sample_pixel.png — pixel stream output ({N} samples)")
        print(f"  sample_rae.png   — RAE stream output   ({N} samples)")

    if dist.is_initialized():
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
