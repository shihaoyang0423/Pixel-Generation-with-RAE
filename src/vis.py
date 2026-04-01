import torch
import torch.nn as nn
import math
import argparse
import sys
import os
from torchvision.utils import save_image, make_grid
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import instantiate_from_config
from stage2.transport import create_transport, Sampler
from utils.train_utils import parse_configs, center_crop_arr
from stage1 import RAE

def get_t(s, shift):
    """
    Map linear s in [0, 1] to time schedule t
    """
    return shift * s / (1 + (shift - 1) * s)

def main(args):
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False) # No grad for visualization

    # Parse Config
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
        return OmegaConf.to_container(cfg_section, resolve=True) if cfg_section else {}

    misc_pix = to_dict(misc_config_pix)
    misc_rae = to_dict(misc_config_rae)
    transport_cfg_pix = to_dict(transport_config_pix)
    transport_cfg_rae = to_dict(transport_config_rae)

    # Calculate Time Dist Shift (Exact pipeline from sample_pix.py)
    latent_size_pix = tuple(int(dim) for dim in misc_pix.get("latent_size", (3, 256, 256)))
    latent_size_rae = tuple(int(dim) for dim in misc_rae.get("latent_size", (768, 16, 16)))
    
    shift_dim_pix = misc_pix.get("time_dist_shift_dim", math.prod(latent_size_pix))
    shift_base_pix = misc_pix.get("time_dist_shift_base", 4096)
    time_dist_shift_pix = math.sqrt(shift_dim_pix / shift_base_pix)
    
    shift_dim_rae = misc_rae.get("time_dist_shift_dim", shift_dim_pix)
    shift_base_rae = misc_rae.get("time_dist_shift_base", shift_base_pix)
    time_dist_shift_rae = math.sqrt(shift_dim_rae / shift_base_rae)

    print(f"Time Dist Shifts - Pix: {time_dist_shift_pix:.4f}, RAE: {time_dist_shift_rae:.4f}")

    # Create Transport Objects
    params_pix = transport_cfg_pix.get("params", {})
    if "time_dist_shift" in params_pix:
        params_pix.pop("time_dist_shift") # Duplicate check
    transport_pix = create_transport(**params_pix, time_dist_shift=time_dist_shift_pix)
    
    params_rae = transport_cfg_rae.get("params", {})
    if "time_dist_shift" in params_rae:
        params_rae.pop("time_dist_shift")
    transport_rae = create_transport(**params_rae, time_dist_shift=time_dist_shift_rae)

    # Load RAE
    print("Loading RAE...")
    rae = instantiate_from_config(rae_config).to(device)
    rae.eval()

    # Load Image
    print(f"Loading image from {args.image_path}")
    image_size = args.image_size
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.ToTensor(),
    ])
    
    try:
        pil_img = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    x_pix = transform(pil_img).unsqueeze(0).to(device) # (1, 3, H, W)
    x_pix = x_pix.clamp(0, 1) # Data (t=1)

    # Encode Image to RAE Latent
    print("Encoding image...")
    z_rae = rae.encode(x_pix) # Data (t=1) for RAE stream
    
    # Generate Fixed Noise (t=0)
    eps_pix = torch.randn_like(x_pix)
    eps_rae = torch.randn_like(z_rae)

    # Visualization Loop over 's' (linear time)
    # We visualize the Forward Process (Corruption towards Data? or Noise?)
    # In SiT/MinRF with linear path: plan(t, x0, x1) = (1-t)x0 + t*x1.
    # where t \in [0, 1]. x0=Noise, x1=Data.
    # So t=0 is Noise, t=1 is Data.
    # We iterate s from 0 to 1.
    
    vis_pix_list = []
    vis_rae_list = []
    
    num_steps = 8
    s_values = torch.linspace(0, 1, num_steps + 1)
    
    print(f"Visualizing forward path (s=0 [Noise] -> s=1 [Data]) with {num_steps} steps...")
    
    for s in s_values:
        s_val = s.item()
        
        # Map s -> t
        t_pix_val = get_t(s_val, time_dist_shift_pix)
        t_rae_val = get_t(s_val, time_dist_shift_rae)
        
        t_pix = torch.tensor([t_pix_val], device=device)
        t_rae = torch.tensor([t_rae_val], device=device)
        
        # Calculate x_t (Interpolate)
        # Pixel
        _, xt_pix, _ = transport_pix.path_sampler.plan(t_pix, eps_pix, x_pix)
        vis_pix_list.append(xt_pix.clamp(0, 1).cpu())
        
        # RAE
        _, xt_rae, _ = transport_rae.path_sampler.plan(t_rae, eps_rae, z_rae)
        decoded_rae = rae.decode(xt_rae) # Decode RAE latent to pixels
        vis_rae_list.append(decoded_rae.clamp(0, 1).cpu())
        
        print(f"s={s_val:.2f} | t_pix={t_pix_val:.3f} | t_rae={t_rae_val:.3f}")

    # Create Grid
    # Row 1: Pixel Stream
    # Row 2: RAE Stream
    row_pix = torch.cat(vis_pix_list, dim=0)
    row_rae = torch.cat(vis_rae_list, dim=0)
    all_imgs = torch.cat([row_pix, row_rae], dim=0) # (2*(steps+1), 3, H, W)
    
    out_path = args.output
    import time
    out_path = f"{out_path}_{int(time.time())}.png" # Add timestamp to avoid overwrite
    # nrow = number of columns = num_steps + 1
    save_image(all_imgs, out_path, nrow=num_steps+1, padding=2, normalize=False)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="~/RAE/configs/stage2/training/PixelDDT_test.yaml", help="Path to the config file.")
    parser.add_argument("--image-path", type=str, default="~/RAE/assets/pixabay_cat.png", help="Path to the input image.")
    parser.add_argument("--output", type=str, default="transport_vis.png", help="Output image path.")
    parser.add_argument("--image-size", type=int, default=256, help="Image size.")
    
    args = parser.parse_args()
    main(args)