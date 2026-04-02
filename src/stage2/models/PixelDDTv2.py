import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed
from .model_utils import (
    VisionRotaryEmbeddingFast, 
    RMSNorm, 
    LabelEmbedder, 
    get_2d_sincos_pos_embed, 
    GaussianFourierEmbedding,
    SwiGLUFFN,
    NormAttention
)
from .DDT import LightningDDTBlock, DDTFinalLayer, DiTwDDTHead, DDTGate, DDTModulate
from .lightningDiT import Mlp
from math import sqrt

class PixelLightningDDTBlock(nn.Module):
    """
    Pixel-level DiT Block with spatial downsampling, RAE latent injection,
    bottleneck Attention/MLP, upsampling, and residual connection.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        rae_hidden_size=None,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        **block_kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        rae_hidden_size = rae_hidden_size or hidden_size

        # 2. RAE Latent Alignment Projection (if RAE dim differs from Pixel dim)
        self.rae_proj = nn.Linear(rae_hidden_size, hidden_size) if rae_hidden_size != hidden_size else nn.Identity()

        # 3. Standard DDT Components (Operating at the bottleneck)
        norm_layer = RMSNorm if use_rmsnorm else lambda dim: nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm1 = norm_layer(hidden_size)
        self.norm2 = norm_layer(hidden_size)

        self.attn = NormAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True, 
            qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm, **block_kwargs
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)

        
        # AdaLN Modulation
        shift_multiplier = 4 if wo_shift else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, shift_multiplier * hidden_size, bias=True)
        )
        self.wo_shift = wo_shift

    def forward(self, x, c, rae_latent, feat_rope=None, H=None, W=None):
        """
        x: Pixel latent [B, L, C]
        c: Conditioning [B, C]
        rae_latent: Pre-computed DINO/RAE features [B, L/4, C_rae]
        H, W: Spatial dimensions of the input pixel sequence. If None, assumes square.
        """
        B, L, C = x.shape
        if H is None or W is None:
            H = W = int(sqrt(L))
            

        # --- INJECT RAE LATENT ---
        # Add the structural prior directly into the downsampled pixel stream
        x = x + self.rae_proj(rae_latent)

        # --- DDT BLOCK (At Bottleneck Resolution) ---
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
            
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
            shift_msa = shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        # Attention
        x = x + DDTGate(self.attn(DDTModulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope), gate_msa)
        # MLP (Heavy Lifting done here at L/4 sequence length)
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)

        return x


class RPFuse(nn.Module):
    """
    Fusion of RAE and Pix
    """
    def __init__(self, rae_hidden_size, pixel_hidden_size):
        super().__init__()
        self.scale = nn.Linear(rae_hidden_size, pixel_hidden_size)
        self.gate = nn.Linear(rae_hidden_size, pixel_hidden_size)
        self.bias = nn.Linear(rae_hidden_size, pixel_hidden_size)

    def forward(self, rae_latent, pixel):
        # Normalize each channel of the RAE latent
        rae_latent = F.normalize(rae_latent, dim=-1)
        gate = torch.sigmoid(self.gate(rae_latent))
        gated_rae = rae_latent * gate
        return pixel * self.scale(gated_rae) + self.bias(gated_rae)


class PixelLightningDDTBlockv2(nn.Module):
    """
    Pixel-level DiT Block with spatial downsampling, RAE latent injection,
    bottleneck Attention/MLP, upsampling, and residual connection.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        rae_hidden_size=None,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        **block_kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        rae_hidden_size = rae_hidden_size or hidden_size

        # 2. RAE Latent Alignment Projection
        self.rp_fuse = RPFuse(rae_hidden_size, hidden_size)

        # 3. Standard DDT Components (Operating at the bottleneck)
        norm_layer = RMSNorm if use_rmsnorm else lambda dim: nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm1 = norm_layer(hidden_size)
        self.norm2 = norm_layer(hidden_size)

        self.attn = NormAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True, 
            qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm, **block_kwargs
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)

        
        # AdaLN Modulation
        shift_multiplier = 4 if wo_shift else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, shift_multiplier * hidden_size, bias=True)
        )
        self.wo_shift = wo_shift

    def forward(self, x, c, rae_latent, feat_rope=None, H=None, W=None):
        """
        x: Pixel latent [B, L, C]
        c: Conditioning [B, C]
        rae_latent: Pre-computed DINO/RAE features [B, L/4, C_rae]
        H, W: Spatial dimensions of the input pixel sequence. If None, assumes square.
        """
        B, L, C = x.shape
        if H is None or W is None:
            H = W = int(sqrt(L))
            

        # --- INJECT RAE LATENT ---
        # Add the structural prior directly into the downsampled pixel stream
        x = self.rp_fuse(rae_latent, x)

        # --- DDT BLOCK (At Bottleneck Resolution) ---
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
            
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
            shift_msa = shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        # Attention
        x = x + DDTGate(self.attn(DDTModulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope), gate_msa)
        # MLP (Heavy Lifting done here at L/4 sequence length)
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)

        return x
class PixelConvLastLayer(nn.Module):
    """
    The final layer of DDT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)  # no gate
        x = self.linear(x)
        return x


class PixelDDT(nn.Module):
    """
    Spatially-Aligned Shallow Pixel DiT (SA-SPD) based on DDT.
    
    Contains the full pretrained RAE-DDT model to process DINO features.
    Uses LightningDDTBlock for pixel generation.
    """
    def __init__(
        self,
        # Pixel DiT Config
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1152,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        # Pretrained DDT Config
        stage2_config: dict = None,
        stage2_ckpt: str = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.depth = depth
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # 1. Load Pretrained DDT (Frozen)
        if stage2_config is None:
            raise ValueError("stage2_config must be provided for SA_SPD")
        
        print(f"Loading pretrained DDT model from {stage2_ckpt}...")
        self.ddt = DiTwDDTHead(**stage2_config)
        
        # 2. Pixel Embedding
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        # 3. Timestep and Class Embedding (for Pixel DiT)
        self.t_embedder = GaussianFourierEmbedding(self.ddt.encoder_hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        # 5. Pixel DiT Blocks (DDT Style)
        # We reuse LightningDDTBlock
        self.blocks = nn.ModuleList([
            LightningDDTBlock(
                hidden_size, 
                num_heads, 
                mlp_ratio=mlp_ratio, 
                use_qknorm=use_qknorm, 
                use_swiglu=use_swiglu, 
                use_rmsnorm=use_rmsnorm,
                wo_shift=wo_shift,
            ) for i in range(self.depth)
        ])
        
        # 6. Final Layer
        self.final_layer = DDTFinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        
        # 7. s_projector
        self.s_projector = nn.Linear(
            self.ddt.encoder_hidden_size, self.hidden_size)
        
        # RoPE
        if use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None
            
        self.initialize_weights()
        
        if stage2_ckpt:
            state_dict = torch.load(stage2_ckpt, map_location='cpu')
            if 'ema' in state_dict:
                state_dict = state_dict['ema']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            msg = self.ddt.load_state_dict(new_state_dict, strict=False)
            print(f"DDT load msg: {msg}")
            if msg.missing_keys:
                print(f"WARNING: DDT has missing keys: {msg.missing_keys}")
        else:
            raise ValueError("stage2_ckpt must be provided to load pretrained DDT weights.")
            
        # Freeze DDT
        for p in self.ddt.parameters():
            p.requires_grad = False
        self.ddt.eval()
        

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-init final layer (Crucial tip for stability)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * out_channels)
        imgs: (N, out_channels, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x_pix, x_rae, t_pix, t_rae, y):
        """
        x_pix: Noisy pixel input [B, in_channels, H_pix, W_pix]
        t: Timesteps
        y: Class labels
        x_rae: Corresponding noisy DINO latents [B, in_channels_ddt, H_rae, W_rae]. 
               Must be provided to guide the Pixel DiT and to output the next latent step.
        """
        if x_rae is None:
            raise ValueError("x_rae must be provided to guide the Pixel DiT and compute the RAE step.")

        # ==========================================
        # 1. Run Frozen RAE (DDT) & Complete its Forward
        # ==========================================
        with torch.no_grad():
            t_ddt = self.ddt.t_embedder(t_rae)
            y_ddt = self.ddt.y_embedder(y, self.training)
            c_ddt = nn.functional.silu(t_ddt + y_ddt)
            
            # Embed RAE latents
            s = self.ddt.s_embedder(x_rae)
            if self.ddt.use_pos_embed:
                s = s + self.ddt.pos_embed
            
            rae_latents = []
            
            # Encoder blocks
            for i in range(self.ddt.num_encoder_blocks):
                s = self.ddt.blocks[i](s, c_ddt, feat_rope=self.ddt.enc_feat_rope)
                rae_latents.append(s)
                
            # Bottleneck Transition
            t_broadcast = t_ddt.unsqueeze(1).repeat(1, s.shape[1], 1)
            s_trans = nn.functional.silu(t_broadcast + s)
            s_trans = self.ddt.s_projector(s_trans)
            
            # Decoder blocks
            x_ddt = self.ddt.x_embedder(x_rae)
            if self.ddt.use_pos_embed and self.ddt.x_pos_embed is not None:
                x_ddt = x_ddt + self.ddt.x_pos_embed
                
            for i in range(self.ddt.num_encoder_blocks, self.ddt.num_blocks):
                x_ddt = self.ddt.blocks[i](x_ddt, s_trans, feat_rope=self.ddt.dec_feat_rope)
                rae_latents.append(x_ddt)

            # --- COMPLETE RAE FORWARD TO GET NEXT STEP PREDICTION ---
            x_ddt_final = self.ddt.final_layer(x_ddt, s_trans)
            x_rae_out = self.ddt.unpatchify(x_ddt_final)

        # ==========================================
        # 2. Run Pixel DDT
        # ==========================================
        x_pix_emb = self.x_embedder(x_pix) + self.pos_embed
        
        t_pix = self.t_embedder(t_pix)
        # y_emb = self.y_embedder(y, self.training)
        # y_emb = y_ddt #Reuse y_emb from DDT for dropout consistency
        
        t_pix_broadcast = t_pix.unsqueeze(1).repeat(1, s.shape[1], 1)
        s_trans = nn.functional.silu(t_pix_broadcast + s)
        s_trans = self.s_projector(s_trans)
        # s_trans = self.s_projector(s)
        # t_pix_broadcast = t_pix.unsqueeze(1).repeat(1, s_trans.shape[1], 1)
        # s_trans = nn.functional.silu(t_pix_broadcast + s_trans)
        
        # c = nn.functional.silu(t_pix + y_emb)
        for i in range(self.depth): # range(self.ddt.num_decoder_blocks)
            x_pix_emb = self.blocks[i](x_pix_emb, s_trans, feat_rope=self.feat_rope)
        
            
        x_pix_emb = self.final_layer(x_pix_emb, s_trans)
        imgs_out = self.unpatchify(x_pix_emb)
        
        # Return both the pixel noise prediction and the RAE noise prediction
        return imgs_out, x_rae_out

    def forward_with_cfg(self, x_pix, x_rae, t_pix, t_rae, y, cfg_scale, cfg_interval=(0, 1)):
        """
        Forward pass with Classifier-Free Guidance applied to BOTH pixel and RAE streams.
        """
        # Duplicate both inputs for conditional/unconditional batching
        half_pix = x_pix[: len(x_pix) // 2]
        combined_pix = torch.cat([half_pix, half_pix], dim=0)
        
        half_rae = x_rae[: len(x_rae) // 2]
        combined_rae = torch.cat([half_rae, half_rae], dim=0)
        
        imgs_out, rae_out = self.forward(combined_pix, combined_rae, t_pix, t_rae, y)
        
        # --- Pixel CFG Split ---
        eps_pix, rest_pix = imgs_out[:, :self.in_channels], imgs_out[:, self.in_channels:]
        cond_eps_pix, uncond_eps_pix = torch.split(eps_pix, len(eps_pix) // 2, dim=0)
        
        # --- RAE CFG Split ---
        eps_rae, rest_rae = rae_out[:, :self.ddt.in_channels], rae_out[:, self.ddt.in_channels:]
        cond_eps_rae, uncond_eps_rae = torch.split(eps_rae, len(eps_rae) // 2, dim=0)
        
        # --- Apply CFG Interval ---
        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        t_half = t_pix[: len(t_pix) // 2]
        
        mask = ((t_half >= guid_t_min) & (t_half <= guid_t_max))
        
        # Resolve Pixel Guidance
        mask_pix = mask.view(-1, *[1] * (len(cond_eps_pix.shape) - 1))
        half_eps_pix = torch.where(
            mask_pix, 
            uncond_eps_pix + cfg_scale * (cond_eps_pix - uncond_eps_pix), 
            cond_eps_pix
        )
        eps_pix = torch.cat([half_eps_pix, half_eps_pix], dim=0)
        final_imgs = torch.cat([eps_pix, rest_pix], dim=1) if rest_pix.shape[1] > 0 else eps_pix
        
        # Resolve RAE Guidance
        mask_rae = mask.view(-1, *[1] * (len(cond_eps_rae.shape) - 1))
        half_eps_rae = torch.where(
            mask_rae, 
            uncond_eps_rae + cfg_scale * (cond_eps_rae - uncond_eps_rae), 
            cond_eps_rae
        )
        eps_rae = torch.cat([half_eps_rae, half_eps_rae], dim=0)
        final_rae = torch.cat([eps_rae, rest_rae], dim=1) if rest_rae.shape[1] > 0 else eps_rae

        return final_imgs, final_rae

    def forward_with_autoguidance(self, x_pix, x_rae, t_pix, t_rae, y, cfg_scale, additional_model_forward, cfg_interval=(0, 1)):
        """
        Forward pass applying external model guidance to BOTH pixel and RAE streams.
        Assumes `additional_model_forward` returns the identical tuple format: (ag_imgs_out, ag_rae_out).
        """
        # Original forward
        imgs_out, rae_out = self.forward(x_pix, x_rae, t_pix, t_rae, y)
        
        # Additional Model Guidance
        ag_imgs_out, ag_rae_out = additional_model_forward(x_pix, x_rae, t_pix, t_rae, y)
        
        eps_pix = imgs_out[:, :self.in_channels]
        ag_eps_pix = ag_imgs_out[:, :self.in_channels]
        
        eps_rae = rae_out[:, :self.ddt.in_channels]
        ag_eps_rae = ag_rae_out[:, :self.ddt.in_channels]

        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        mask = ((t_pix >= guid_t_min) & (t_pix <= guid_t_max))
        
        # Interpolate Pixel Guidance
        mask_pix = mask.view(-1, *[1] * (len(eps_pix.shape) - 1))
        eps_pix = torch.where(
            mask_pix, 
            ag_eps_pix + cfg_scale * (eps_pix - ag_eps_pix), 
            eps_pix
        )
        rest_pix = imgs_out[:, self.in_channels:]
        final_imgs = torch.cat([eps_pix, rest_pix], dim=1) if rest_pix.shape[1] > 0 else eps_pix

        # Interpolate RAE Guidance
        mask_rae = mask.view(-1, *[1] * (len(eps_rae.shape) - 1))
        eps_rae = torch.where(
            mask_rae, 
            ag_eps_rae + cfg_scale * (eps_rae - ag_eps_rae), 
            eps_rae
        )
        rest_rae = rae_out[:, self.ddt.in_channels:]
        final_rae = torch.cat([eps_rae, rest_rae], dim=1) if rest_rae.shape[1] > 0 else eps_rae

        return final_imgs, final_rae