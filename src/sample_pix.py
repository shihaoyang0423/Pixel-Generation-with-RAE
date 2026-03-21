# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained PixelDDT (Dual-Stream) model.
"""
import torch
import torch.nn as nn
import math
from time import time
import argparse
import sys
import os
from torchvision.utils import save_image

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import instantiate_from_config
from stage2.transport import create_transport, Sampler
from utils.train_utils import parse_configs
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from omegaconf import OmegaConf

# Import joint sampling functions from train_stage3
# Assuming train_stage3.py is in the same directory (src/)
try:
    from train_stage3 import joint_sample_ode
except ImportError:
    from src.train_stage3 import joint_sample_ode

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parse Dual-Stream Configs
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
    ) = parse_configs(args.config)
    
    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc_pix = to_dict(misc_config_pix)
    misc_rae = to_dict(misc_config_rae)
    transport_cfg_pix = to_dict(transport_config_pix)
    transport_cfg_rae = to_dict(transport_config_rae)
    sampler_cfg_pix = to_dict(sampler_config_pix)
    # sampler_cfg_rae = to_dict(sampler_config_rae)
    guidance_cfg = to_dict(guidance_config)

    # Instantiate Models
    print("Loading RAE...")
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    
    print("Loading PixelDDT Model...")
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    if args.ckpt is not None:
        print(f"Loading checkpoint from {args.ckpt}...")
        try:
            checkpoint = torch.load(args.ckpt, map_location=device)
            # Support both EMA and raw model checkpoints
            if "ema" in checkpoint:
                print("Loading EMA weights...")
                state_dict = checkpoint["ema"]
            elif "model" in checkpoint:
                print("Loading raw model weights...")
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict, strict=True)
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Trying to load as raw state dict...")
            model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
            

    model.eval()

    # Calculate Time Dist Shift (from train_stage3.py logic)
    latent_size_pix = tuple(int(dim) for dim in misc_pix.get("latent_size", (3, 256, 256)))
    latent_size_rae = tuple(int(dim) for dim in misc_rae.get("latent_size", (768, 16, 16)))
    
    shift_dim_pix = misc_pix.get("time_dist_shift_dim", math.prod(latent_size_pix))
    shift_base_pix = misc_pix.get("time_dist_shift_base", 4096)
    time_dist_shift_pix = math.sqrt(shift_dim_pix / shift_base_pix)
    
    shift_dim_rae = misc_rae.get("time_dist_shift_dim", shift_dim_pix)
    shift_base_rae = misc_rae.get("time_dist_shift_base", shift_base_pix)
    time_dist_shift_rae = math.sqrt(shift_dim_rae / shift_base_rae)

    print(f"Time shift Pix: {time_dist_shift_pix:.4f}, RAE: {time_dist_shift_rae:.4f}")

    # Create Transport Objects
    transport_pix = create_transport(
        **transport_cfg_pix['params'],
        time_dist_shift=time_dist_shift_pix
    )
    transport_rae = create_transport(
        **transport_cfg_rae['params'],
        time_dist_shift=time_dist_shift_rae
    )

    # Sampler Config
    # Default to ODE/dopri5 unless 'naive' is requested or specified
    sampler_mode = sampler_cfg_pix.get("mode", "ODE")
    sampler_params = sampler_cfg_pix.get("params", {})
    num_steps = sampler_params.get("num_steps", 50)
    
    # Use naive sampler if specified in args or config
    use_naive = args.naive or (sampler_mode.lower() == "naive")
    sampler_type = "naive" if use_naive else "dopri5" # or 'euler' if supported by integrators
    
    print(f"Using sampler type: {sampler_type} with {num_steps} steps.")

    # Labels to condition the model with:
    # default to some ImageNet classes if not provided
    class_labels = [207, 360, 9, 99] 
    n = len(class_labels)

    # Create sampling noise
    # Pixel Stream Noise
    z_pix = torch.randn(n, *latent_size_pix, device=device)
    # RAE Stream Noise
    z_rae = torch.randn(n, *latent_size_rae, device=device)
    
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance
    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if args.cfg_scale is not None:
        guidance_scale = args.cfg_scale
        
    print(f"Guidance scale: {guidance_scale}")

    # Duplicate inputs for CFG
    z_pix = torch.cat([z_pix, z_pix], 0)
    z_rae = torch.cat([z_rae, z_rae], 0)
    
    y_null = torch.tensor([1000] * n, device=device) # Assuming 1000 is null label
    y = torch.cat([y, y_null], 0)
    
    t_min = float(guidance_cfg.get("t_min", 0.0))
    t_max = float(guidance_cfg.get("t_max", 1.0))

    if guidance_scale > 1.0:
        model_kwargs = dict(
            y=y, 
            cfg_scale=guidance_scale,
            cfg_interval=(t_min, t_max)
        )
        model_fn = model.forward_with_cfg
    else:
        model_kwargs = dict(y=y)
        model_fn = model.forward

    # Sample
    start_time = time()
    
    samples_pix, samples_rae = joint_sample_ode(
        z_pix=z_pix,
        z_rae=z_rae,
        model_fn=model_fn,
        transport_pix=transport_pix,
        transport_rae=transport_rae,
        sampler_type=sampler_type,
        num_steps=num_steps,
        **model_kwargs
    )
    
    # Remove null class samples
    samples_pix, _ = samples_pix.chunk(2, dim=0)
    samples_rae, _ = samples_rae.chunk(2, dim=0)
    
    print(f"Sampling took {time() - start_time:.2f} seconds.")

    # Post-process
    # 1. Pixel stream output (direct images)
    samples_pix = samples_pix.clamp(0, 1)
    
    # 2. RAE stream output (latents -> decode)
    # Note: RAE latents usually don't need clamping before decode, but output image does
    decoded_rae = rae.decode(samples_rae)
    decoded_rae = decoded_rae.clamp(0, 1)

    # Save images
    os.makedirs("samples", exist_ok=True)
    save_path_pix = "samples/sample_pix_ddt.png"
    save_path_rae = "samples/sample_rae_ddt.png"
    
    save_image(samples_pix, save_path_pix, nrow=int(math.sqrt(n)), normalize=False)
    save_image(decoded_rae, save_path_rae, nrow=int(math.sqrt(n)), normalize=True, value_range=(0,1))
    
    print(f"Saved pixel stream samples to {save_path_pix}")
    print(f"Saved RAE stream samples to {save_path_rae}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a checkpoint (e.g. .pt file) to load weights from.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=None, help="Override CFG scale")
    parser.add_argument("--naive", action="store_true", help="Force use of naive Euler sampler")
    
    args = parser.parse_known_args()[0]
    main(args)
