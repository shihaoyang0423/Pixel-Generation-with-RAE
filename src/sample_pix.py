# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained PixelDDT (Dual-Stream) model.
Includes visualization of intermediate predictions (x0).
"""
import torch
import torch.nn as nn
import math
from time import time
import argparse
import sys
import os
from torchvision.utils import save_image, make_grid

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import instantiate_from_config
from stage2.transport import create_transport
from utils.train_utils import parse_configs
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from omegaconf import OmegaConf

# Import joint sampling functions from train_stage3
try:
    from train_stage3 import joint_sample_ode
except ImportError:
    try:
        from src.train_stage3 import joint_sample_ode
    except ImportError:
         sys.path.append(os.path.dirname(os.path.abspath(__file__)))
         from train_stage3 import joint_sample_ode

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

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

    # Calculate Time Dist Shift
    latent_size_pix = tuple(int(dim) for dim in misc_pix.get("latent_size", (3, 256, 256)))
    latent_size_rae = tuple(int(dim) for dim in misc_rae.get("latent_size", (768, 16, 16)))
    
    shift_dim_pix = misc_pix.get("time_dist_shift_dim", math.prod(latent_size_pix))
    shift_base_pix = misc_pix.get("time_dist_shift_base", 4096)
    time_dist_shift_pix = math.sqrt(shift_dim_pix / shift_base_pix)
    
    shift_dim_rae = misc_rae.get("time_dist_shift_dim", shift_dim_pix)
    shift_base_rae = misc_rae.get("time_dist_shift_base", shift_base_pix)
    time_dist_shift_rae = math.sqrt(shift_dim_rae / shift_base_rae)

    print(f"Time shift Pix: {time_dist_shift_pix:.4f}, RAE: {time_dist_shift_rae:.4f}")
    
    # DEBUG: Enforce shift calculation print and transport checks
    print(f"DEBUG: shift_dim_pix={shift_dim_pix}, shift_base_pix={shift_base_pix}")

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
    sampler_mode = sampler_cfg_pix.get("mode", "ODE")
    sampler_params = sampler_cfg_pix.get("params", {})
    num_steps = sampler_params.get("num_steps", 50)
    
    use_naive = args.naive or (sampler_mode.lower() == "naive")
    sampler_type = "naive" if use_naive else "dopri5"
    print(f"Using sampler type: {sampler_type} with {num_steps} steps.")

    class_labels = [207, 360, 9, 99] 
    n = len(class_labels)

    z_pix = torch.randn(n, *latent_size_pix, device=device)
    z_rae = torch.randn(n, *latent_size_rae, device=device)
    y = torch.tensor(class_labels, device=device)

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if args.cfg_scale is not None:
        guidance_scale = args.cfg_scale
        
    print(f"Guidance scale: {guidance_scale}")

    # For CFG
    z_pix = torch.cat([z_pix, z_pix], 0)
    z_rae = torch.cat([z_rae, z_rae], 0)
    y_null = torch.tensor([1000] * n, device=device)
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

    # Track intermediates
    intermediate_states = []

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
        intermediates=intermediate_states, # Pass list for capturing
        capture_intermediates=True,        # Trigger capture
        **model_kwargs
    )
    
    # Process final samples (remove null class part)
    samples_pix, _ = samples_pix.chunk(2, dim=0)
    samples_rae, _ = samples_rae.chunk(2, dim=0)
    
    print(f"Sampling took {time() - start_time:.2f} seconds.")

    # Save Final Images
    samples_pix = samples_pix.clamp(0, 1)
    
    with torch.no_grad():
        decoded_rae = rae.decode(samples_rae)
    decoded_rae = decoded_rae.clamp(0, 1)
    
    save_image(samples_pix, os.path.join(args.output_dir, f"final_pix.png"), nrow=int(math.sqrt(n)), normalize=False)
    # Correct output
    save_image(decoded_rae, os.path.join(args.output_dir, f"final_rae.png"), nrow=int(math.sqrt(n)), normalize=True, value_range=(0, 1))

    # Process and Save Intermediates
    print(f"Processing {len(intermediate_states)} intermediate steps...")
    
    # Directory for steps
    steps_dir = os.path.join(args.output_dir, "steps")
    os.makedirs(steps_dir, exist_ok=True)

    for i, state in enumerate(intermediate_states):
        step = state['step']
        t_val = state['t_pix']
        
        # Get tensors 
        x_pix = state['x_pix'].to(device)
        v_pix = state['v_pix'].to(device)
        x_rae = state['x_rae'].to(device)
        v_rae = state['v_rae'].to(device)

        # Handle CFG splitting
        x_pix, _ = x_pix.chunk(2, dim=0)
        v_pix, _ = v_pix.chunk(2, dim=0)
        x_rae, _ = x_rae.chunk(2, dim=0)
        v_rae, _ = v_rae.chunk(2, dim=0)
        
        # x_pix and x_rae are now already predicted x0 values from the sampler
        x0_pred_pix = x_pix
        x0_pred_rae_latent = x_rae
        
        # Clamp/Process
        x0_pred_pix = x0_pred_pix.clamp(0, 1)

        with torch.no_grad():
             x0_pred_rae_img = rae.decode(x0_pred_rae_latent)
        x0_pred_rae_img = x0_pred_rae_img.clamp(0, 1)
        
        # Resize RAE output to match Pix output if necessary
        if x0_pred_pix.shape[-2:] != x0_pred_rae_img.shape[-2:]:
            x0_pred_rae_img = nn.functional.interpolate(
                x0_pred_rae_img, 
                size=x0_pred_pix.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )

        # Concatenate vertically
        combined = torch.cat([x0_pred_pix, x0_pred_rae_img], dim=0)

        # Save with nrow=n so that first row is batch, second row is batch
        save_path = os.path.join(steps_dir, f"step_{step:03d}.png")
        save_image(
            combined, 
            save_path, 
            nrow=n, 
            normalize=False
        )
        if i % 10 == 0:
             print(f"Propagating step {i}: Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a checkpoint.")
    parser.add_argument("--output_dir", type=str, default="samples/vis", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=None, help="Override CFG scale")
    parser.add_argument("--naive", action="store_true", help="Force use of naive Euler sampler")
    
    args = parser.parse_known_args()[0]
    main(args)
