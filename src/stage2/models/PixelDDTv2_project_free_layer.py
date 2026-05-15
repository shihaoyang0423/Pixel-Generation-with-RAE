"""
PixelDDTv2_project_free_layer.py — Free-layer injection version of PixelDDT (project).

Extends PixelDDTv2_project.py with configurable per-layer RAE injection:
  - rae_inject_layers: list | None
      None  → all blocks receive RAE conditioning (identical to _project)
      []    → no blocks receive RAE conditioning
      [0,4,8,12] → only those block indices receive RAE conditioning
  - rae_inject_final_layer: bool (default True)
      Controls whether DDTFinalLayer also receives RAE conditioning,
      independently of rae_inject_layers.

For blocks NOT in rae_inject_layers, conditioning falls back to t_pix + y_embed
(s_trans_t_only = s_projector(silu(t_pix_broadcast + y_broadcast))), consistent
with standard DiT which always conditions on both time and class label.
For blocks IN rae_inject_layers (or when rae_inject_layers is None), conditioning
uses full RAE features (s_trans_rae = s_projector(silu(t_pix_broadcast + s_clean)));
s_clean is assumed to already encode class-level semantics so y is not added separately.

All other architecture, training, and inference logic is identical to
PixelDDTv2_project.py.
"""

import contextlib
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
    NormAttention,
)
from .DDT import LightningDDTBlock, DDTFinalLayer, DiTwDDTHead, DDTGate, DDTModulate
from .lightningDiT import Mlp
from math import sqrt


# ---------------------------------------------------------------------------
# Unchanged helper blocks (copied verbatim from PixelDDTv2_project.py)
# ---------------------------------------------------------------------------

class PixelLightningDDTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, rae_hidden_size=None, mlp_ratio=4.0,
                 use_qknorm=False, use_swiglu=True, use_rmsnorm=True, wo_shift=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        rae_hidden_size = rae_hidden_size or hidden_size
        self.rae_proj = nn.Linear(rae_hidden_size, hidden_size) if rae_hidden_size != hidden_size else nn.Identity()
        norm_layer = RMSNorm if use_rmsnorm else lambda dim: nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm1 = norm_layer(hidden_size)
        self.norm2 = norm_layer(hidden_size)
        self.attn = NormAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                  qk_norm=use_qknorm, use_rmsnorm=use_rmsnorm, **block_kwargs)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                           act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        shift_multiplier = 4 if wo_shift else 6
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, shift_multiplier * hidden_size, bias=True))
        self.wo_shift = wo_shift

    def forward(self, x, c, rae_latent, feat_rope=None, H=None, W=None):
        B, L, C = x.shape
        if H is None or W is None:
            H = W = int(sqrt(L))
        x = x + self.rae_proj(rae_latent)
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
            shift_msa = shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + DDTGate(self.attn(DDTModulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope), gate_msa)
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)
        return x


class RPFuse(nn.Module):
    def __init__(self, rae_hidden_size, pixel_hidden_size):
        super().__init__()
        self.scale = nn.Linear(rae_hidden_size, pixel_hidden_size)
        self.gate = nn.Linear(rae_hidden_size, pixel_hidden_size)
        self.bias = nn.Linear(rae_hidden_size, pixel_hidden_size)

    def forward(self, rae_latent, pixel):
        rae_latent = F.normalize(rae_latent, dim=-1)
        gate = torch.sigmoid(self.gate(rae_latent))
        gated_rae = rae_latent * gate
        return pixel * self.scale(gated_rae) + self.bias(gated_rae)


class PixelConvLastLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class PixelDDT(nn.Module):
    """
    Free-layer injection version of PixelDDT (project sequential denoising).

    At inference:
      1. Phase 1 – run RAE ODE fully (forward_rae × num_steps).
      2. Phase 2 – extract fixed conditioning features (extract_clean_s once).
      3. Phase 3 – run Pixel ODE with fixed s_clean (forward_pix × num_steps).

    At training:
      forward() receives x1_rae (clean) alongside noisy xt_rae/xt_pix and
      computes both RAE loss and Pixel loss with clean conditioning.

    New parameters vs PixelDDTv2_project.py:
      rae_inject_layers      : list | None
          Which of the depth pixel blocks receive RAE conditioning.
          None → all blocks (default, identical to _project behaviour).
          []   → no blocks.
          [0,4,8,12] → only those indices.
      rae_inject_final_layer : bool (default True)
          Whether DDTFinalLayer also receives RAE conditioning.
    """

    def __init__(
        self,
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
        stage2_config: dict = None,
        stage2_ckpt: str = None,
        use_lora: bool = False,
        lora_rank: int = 4,
        mask_channel: str = False,
        rae_inject_layers=None,         # None = all blocks; [] = none; [0,4,8,12] = those indices
        rae_inject_final_layer: bool = True,
        use_direct_rae_injection: bool = False,
        direct_rae_in_channels: int = 768,   # channel dim of x1_rae (DINOv2-base = 768)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.depth = depth
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.mask_channel = mask_channel

        # Free-layer injection config
        self.rae_inject_final_layer = rae_inject_final_layer
        if rae_inject_layers is None:
            self._inject_set = None          # all layers
        else:
            self._inject_set = set(rae_inject_layers)
        self.use_direct_rae_injection = use_direct_rae_injection
        print(f"[PixelDDT free_layer] rae_inject_layers={rae_inject_layers}, "
              f"rae_inject_final_layer={rae_inject_final_layer}, "
              f"use_direct_rae_injection={use_direct_rae_injection}, "
              f"pixel_only={self._inject_set is not None and len(self._inject_set) == 0 and not rae_inject_final_layer}")

        if stage2_config is None:
            raise ValueError("stage2_config must be provided for PixelDDT")

        print(f"Loading pretrained DDT model from {stage2_ckpt}...")
        self.ddt = DiTwDDTHead(**stage2_config)

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.t_embedder = GaussianFourierEmbedding(self.ddt.encoder_hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, self.ddt.encoder_hidden_size, class_dropout_prob)

        self.blocks = nn.ModuleList([
            LightningDDTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm, use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm, wo_shift=wo_shift,
            ) for _ in range(self.depth)
        ])

        self.final_layer = DDTFinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)

        self.s_projector = nn.Linear(self.ddt.encoder_hidden_size, self.hidden_size)
        if use_direct_rae_injection:
            self.s_raw_projector = nn.Linear(direct_rae_in_channels, self.ddt.encoder_hidden_size)

        if use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len)
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

        for p in self.ddt.parameters():
            p.requires_grad = False

        self.use_lora = use_lora
        if use_lora:
            print(f"Injecting LoRA into RAE DDT (rank={lora_rank})...")
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                target_modules=["qkv", "proj", "fc1", "fc2", "linear"],
                lora_dropout=0.0,
                bias="none",
            )
            self.ddt = get_peft_model(self.ddt, lora_config)
            self.ddt.print_trainable_parameters()

        self.ddt.eval()

    # ------------------------------------------------------------------
    # Weight initialisation (identical to PixelDDTv2_project.py)
    # ------------------------------------------------------------------

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # ------------------------------------------------------------------
    # Pixel-only mode flag
    # ------------------------------------------------------------------

    @property
    def pixel_only(self) -> bool:
        """True when no RAE conditioning is injected anywhere (inject_set empty + final_layer off)."""
        return (self._inject_set is not None
                and len(self._inject_set) == 0
                and not self.rae_inject_final_layer)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    # ------------------------------------------------------------------
    # Internal: run DDT encoder and return raw bottleneck s
    # ------------------------------------------------------------------

    def _run_ddt_encoder(self, x_rae, t_rae, y):
        """
        Run DDT embedding + encoder blocks for x_rae at timestep t_rae.
        Returns (s, t_ddt, y_ddt) where s is the encoder bottleneck [B, L_enc, C_enc].
        """
        t_ddt = self.ddt.t_embedder(t_rae)
        y_ddt = self.ddt.y_embedder(y, self.training)
        c_ddt = nn.functional.silu(t_ddt + y_ddt)

        s = self.ddt.s_embedder(x_rae)
        if self.ddt.use_pos_embed:
            s = s + self.ddt.pos_embed
        for i in range(self.ddt.num_encoder_blocks):
            s = self.ddt.blocks[i](s, c_ddt, feat_rope=self.ddt.enc_feat_rope)
        return s, t_ddt, y_ddt

    def _run_ddt_decoder(self, x_rae, s_trans, t_ddt):
        """
        Run DDT decoder blocks + final layer to produce velocity prediction.
        Returns x_rae_out (velocity prediction, same spatial shape as x_rae input).
        """
        x_ddt = self.ddt.x_embedder(x_rae)
        if self.ddt.use_pos_embed and self.ddt.x_pos_embed is not None:
            x_ddt = x_ddt + self.ddt.x_pos_embed
        for i in range(self.ddt.num_encoder_blocks, self.ddt.num_blocks):
            x_ddt = self.ddt.blocks[i](x_ddt, s_trans, feat_rope=self.ddt.dec_feat_rope)
        x_ddt_final = self.ddt.final_layer(x_ddt, s_trans)
        return self.ddt.unpatchify(x_ddt_final)

    # ------------------------------------------------------------------
    # New method: extract clean encoder bottleneck from x1_rae at t=0
    # ------------------------------------------------------------------

    def _extract_clean_s(self, x1_rae, y):
        """
        Extract conditioning features from clean RAE latent x1_rae.
        Returns s [B, L_enc, encoder_hidden_size].

        Two modes (controlled by use_direct_rae_injection):

        Bottleneck mode (default, use_direct_rae_injection=False):
          Run DDT encoder (12 Transformer blocks) on x1_rae at t=0.
          Features are globally context-aware via self-attention.

        Direct mode (use_direct_rae_injection=True):
          Reshape x1_rae [B,C,H,W] → [B,H*W,C] and apply s_raw_projector
          (single linear layer C→encoder_hidden_size).
          Features remain spatially local — no global attention mixing.
          Significantly cheaper (no DDT forward pass in Phase 2).
        """
        if self.use_direct_rae_injection:
            B, C, H, W = x1_rae.shape
            x1_flat = x1_rae.flatten(2).permute(0, 2, 1)   # [B, H*W, C]
            return self.s_raw_projector(x1_flat)            # [B, H*W, encoder_hidden_size]
        else:
            B = x1_rae.shape[0]
            t_zero = torch.zeros(B, device=x1_rae.device, dtype=x1_rae.dtype)
            ddt_ctx = contextlib.nullcontext() if self.use_lora else torch.no_grad()
            with ddt_ctx:
                s, _, _ = self._run_ddt_encoder(x1_rae, t_zero, y)
            return s

    def extract_clean_s(self, x_rae_clean, y):
        """Public wrapper around _extract_clean_s (used by samplers)."""
        return self._extract_clean_s(x_rae_clean, y)

    # ------------------------------------------------------------------
    # New method: RAE-only forward (used during Phase 1 of inference ODE)
    # ------------------------------------------------------------------

    def forward_rae(self, x_rae, t_rae, y):
        """
        Run a full DDT forward (encoder + bottleneck + decoder) on noisy x_rae
        at timestep t_rae and return the velocity prediction x_rae_out.

        Called at every step of the RAE ODE during inference (Phase 1).
        """
        ddt_ctx = contextlib.nullcontext() if self.use_lora else torch.no_grad()
        with ddt_ctx:
            s, t_ddt, _ = self._run_ddt_encoder(x_rae, t_rae, y)

            t_broadcast = t_ddt.unsqueeze(1).repeat(1, s.shape[1], 1)
            s_trans = nn.functional.silu(t_broadcast + s)
            s_trans = self.ddt.s_projector(s_trans)

            x_rae_out = self._run_ddt_decoder(x_rae, s_trans, t_ddt)
        return x_rae_out

    # ------------------------------------------------------------------
    # Pixel-only forward with configurable per-layer RAE injection
    # ------------------------------------------------------------------

    def forward_pix(self, x_pix, t_pix, y, s_clean):
        """
        Run the Pixel DDT head conditioned on pre-extracted clean bottleneck s_clean.

        s_clean: [B, L_enc, encoder_hidden_size] — output of extract_clean_s,
                 FIXED across all pixel ODE steps.

        Conditioning per block is selected by rae_inject_layers:
          - block i in _inject_set (or _inject_set is None): uses s_trans_rae
          - otherwise: uses s_trans_t_only (t_pix conditioning only, no RAE)
        final_layer conditioning is controlled by rae_inject_final_layer.

        Returns pixel velocity prediction imgs_out.
        """
        # Mask channel augmentation is training-only; skip at inference (training=False).
        if self.mask_channel and self.training:
            p = 0.5
            p_two_channels = 0.5
            device = x_pix.device
            b, c, h, w = x_pix.shape
            apply_mask = torch.rand(b, device=device) < p
            batch_idx = torch.where(apply_mask)[0]
            if batch_idx.numel() > 0:
                n_mask = batch_idx.numel()
                c1 = torch.randint(0, c, (n_mask,), device=device)
                if c > 1:
                    mask_second = torch.rand(n_mask, device=device) < p_two_channels
                    offset = torch.randint(1, c, (n_mask,), device=device)
                    c2 = torch.where(mask_second, (c1 + offset) % c, c1)
                    x_pix[batch_idx, c2, :, :] = 0.0
                x_pix[batch_idx, c1, :, :] = 0.0

        x_pix_emb = self.x_embedder(x_pix) + self.pos_embed

        t_pix_emb = self.t_embedder(t_pix)                          # [B, enc_hidden]
        y_emb     = self.y_embedder(y, self.training)                # [B, enc_hidden]

        # pixel_only fast path: no RAE conditioning, use t+y like standard DiT
        if s_clean is None:
            c_ty = self.s_projector(nn.functional.silu(t_pix_emb + y_emb))  # [B, hidden_size]
            for i in range(self.depth):
                x_pix_emb = self.blocks[i](x_pix_emb, c_ty, feat_rope=self.feat_rope)
            x_pix_emb = self.final_layer(x_pix_emb, c_ty)
            return self.unpatchify(x_pix_emb)

        # Pre-compute both conditioning variants.
        t_pix_broadcast = t_pix_emb.unsqueeze(1).repeat(1, s_clean.shape[1], 1)
        # Full RAE injection: t_pix + s_clean (s already encodes class info)
        s_trans_rae    = self.s_projector(nn.functional.silu(t_pix_broadcast + s_clean))
        # t+y fallback for non-injected layers: standard DiT-style conditioning
        ty_broadcast   = (t_pix_emb + y_emb).unsqueeze(1).repeat(1, s_clean.shape[1], 1)
        s_trans_t_only = self.s_projector(nn.functional.silu(ty_broadcast))

        for i in range(self.depth):
            if self._inject_set is None or i in self._inject_set:
                c = s_trans_rae
            else:
                c = s_trans_t_only
            x_pix_emb = self.blocks[i](x_pix_emb, c, feat_rope=self.feat_rope)

        c_final = s_trans_rae if self.rae_inject_final_layer else s_trans_t_only
        x_pix_emb = self.final_layer(x_pix_emb, c_final)
        return self.unpatchify(x_pix_emb)

    # ------------------------------------------------------------------
    # Training forward — both streams, clean conditioning for pixel
    # ------------------------------------------------------------------

    def forward(self, x_pix, x_rae=None, t_pix=None, t_rae=None, y=None, x1_rae=None):
        """
        Training forward pass.

        x_pix  : noisy pixel at t_pix  [B, 3, H, W]
        x_rae  : noisy RAE   at t_rae  [B, C_rae, H_rae, W_rae]  (unused when pixel_only)
        t_pix  : pixel timestep        [B]
        t_rae  : RAE timestep          [B]                         (unused when pixel_only)
        y      : class labels          [B]
        x1_rae : CLEAN RAE latent      [B, C_rae, H_rae, W_rae]  (unused when pixel_only)
                 Used to extract conditioning features at t=0 (Phase 2 simulation).

        Returns (imgs_out, x_rae_out):
          imgs_out  — pixel velocity prediction (for pixel loss)
          x_rae_out — RAE  velocity prediction  (for RAE loss; zeros tensor when pixel_only)
        """
        # pixel_only fast path: skip RAE encoder and denoising entirely
        if self.pixel_only:
            imgs_out = self.forward_pix(x_pix, t_pix, y, s_clean=None)
            return imgs_out, torch.zeros(1, device=x_pix.device, dtype=x_pix.dtype)

        # 1. Extract clean conditioning features from x1_rae at t=0
        s_clean = self._extract_clean_s(x1_rae, y)

        # 2. RAE denoising branch — full DDT forward with noisy x_rae (for RAE loss)
        ddt_ctx = contextlib.nullcontext() if self.use_lora else torch.no_grad()
        with ddt_ctx:
            s_rae, t_ddt_rae, _ = self._run_ddt_encoder(x_rae, t_rae, y)
            t_broadcast_rae = t_ddt_rae.unsqueeze(1).repeat(1, s_rae.shape[1], 1)
            s_trans_rae = nn.functional.silu(t_broadcast_rae + s_rae)
            s_trans_rae = self.ddt.s_projector(s_trans_rae)
            x_rae_out = self._run_ddt_decoder(x_rae, s_trans_rae, t_ddt_rae)

        # 3. Pixel head conditioned on clean features (with per-layer injection config)
        imgs_out = self.forward_pix(x_pix, t_pix, y, s_clean)

        return imgs_out, x_rae_out

    # ------------------------------------------------------------------
    # CFG / autoguidance variants (kept for interface compatibility)
    # ------------------------------------------------------------------

    def forward_rae_with_cfg(self, x_rae, t_rae, y, cfg_scale, cfg_interval=(0, 1)):
        """
        CFG-guided RAE forward for use during Phase 1 of inference.
        Expects x_rae and y already doubled [cond; uncond].
        """
        rae_out = self.forward_rae(x_rae, t_rae, y)

        eps_rae = rae_out[:, :self.ddt.in_channels]
        rest_rae = rae_out[:, self.ddt.in_channels:]
        cond_eps, uncond_eps = torch.split(eps_rae, len(eps_rae) // 2, dim=0)

        t_half = t_rae[: len(t_rae) // 2]
        guid_t_min, guid_t_max = cfg_interval
        mask = ((t_half >= guid_t_min) & (t_half <= guid_t_max))
        mask = mask.view(-1, *([1] * (len(cond_eps.shape) - 1)))

        half_eps = torch.where(mask, uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_eps)
        eps_rae_out = torch.cat([half_eps, half_eps], dim=0)
        final_rae = torch.cat([eps_rae_out, rest_rae], dim=1) if rest_rae.shape[1] > 0 else eps_rae_out
        return final_rae

    def forward_pix_with_cfg(self, x_pix, t_pix, y, s_clean, cfg_scale, cfg_interval=(0, 1)):
        """
        CFG-guided Pixel forward for use during Phase 3 of inference.
        Expects x_pix, t_pix, y, s_clean already doubled [cond; uncond].
        """
        pix_out = self.forward_pix(x_pix, t_pix, y, s_clean)

        eps_pix = pix_out[:, :self.in_channels]
        rest_pix = pix_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps_pix, len(eps_pix) // 2, dim=0)

        t_half = t_pix[: len(t_pix) // 2]
        guid_t_min, guid_t_max = cfg_interval
        mask = ((t_half >= guid_t_min) & (t_half <= guid_t_max))
        mask = mask.view(-1, *([1] * (len(cond_eps.shape) - 1)))

        half_eps = torch.where(mask, uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_eps)
        eps_pix_out = torch.cat([half_eps, half_eps], dim=0)
        final_pix = torch.cat([eps_pix_out, rest_pix], dim=1) if rest_pix.shape[1] > 0 else eps_pix_out
        return final_pix

    def forward_with_cfg(self, x_pix, x_rae, t_pix, t_rae, y, cfg_scale, cfg_interval=(0, 1), x1_rae=None):
        """
        Kept for interface compatibility with train_stage3 visual sampling.
        Applies CFG to the joint training forward (both streams).
        In practice cfg_scale=1.0 so this is a pass-through.
        """
        half_pix = x_pix[: len(x_pix) // 2]
        combined_pix = torch.cat([half_pix, half_pix], dim=0)
        half_rae = x_rae[: len(x_rae) // 2]
        combined_rae = torch.cat([half_rae, half_rae], dim=0)
        if x1_rae is None:
            x1_rae_combined = combined_rae
        else:
            half_x1rae = x1_rae[: len(x1_rae) // 2]
            x1_rae_combined = torch.cat([half_x1rae, half_x1rae], dim=0)

        imgs_out, rae_out = self.forward(combined_pix, combined_rae, t_pix, t_rae, y, x1_rae_combined)

        eps_pix = imgs_out[:, :self.in_channels]
        rest_pix = imgs_out[:, self.in_channels:]
        cond_eps_pix, uncond_eps_pix = torch.split(eps_pix, len(eps_pix) // 2, dim=0)

        eps_rae = rae_out[:, :self.ddt.in_channels]
        rest_rae = rae_out[:, self.ddt.in_channels:]
        cond_eps_rae, uncond_eps_rae = torch.split(eps_rae, len(eps_rae) // 2, dim=0)

        guid_t_min, guid_t_max = cfg_interval
        t_half = t_pix[: len(t_pix) // 2]
        mask = ((t_half >= guid_t_min) & (t_half <= guid_t_max))

        mask_pix = mask.view(-1, *[1] * (len(cond_eps_pix.shape) - 1))
        half_eps_pix = torch.where(mask_pix, uncond_eps_pix + cfg_scale * (cond_eps_pix - uncond_eps_pix), cond_eps_pix)
        eps_pix = torch.cat([half_eps_pix, half_eps_pix], dim=0)
        final_imgs = torch.cat([eps_pix, rest_pix], dim=1) if rest_pix.shape[1] > 0 else eps_pix

        mask_rae = mask.view(-1, *[1] * (len(cond_eps_rae.shape) - 1))
        half_eps_rae = torch.where(mask_rae, uncond_eps_rae + cfg_scale * (cond_eps_rae - uncond_eps_rae), cond_eps_rae)
        eps_rae = torch.cat([half_eps_rae, half_eps_rae], dim=0)
        final_rae = torch.cat([eps_rae, rest_rae], dim=1) if rest_rae.shape[1] > 0 else eps_rae

        return final_imgs, final_rae

    def forward_with_autoguidance(self, x_pix, x_rae, t_pix, t_rae, y, cfg_scale,
                                  additional_model_forward, cfg_interval=(0, 1), x1_rae=None):
        """Autoguidance variant — kept for interface compatibility."""
        if x1_rae is None:
            x1_rae = x_rae
        imgs_out, rae_out = self.forward(x_pix, x_rae, t_pix, t_rae, y, x1_rae)
        ag_imgs_out, ag_rae_out = additional_model_forward(x_pix, x_rae, t_pix, t_rae, y, x1_rae)

        eps_pix = imgs_out[:, :self.in_channels]
        ag_eps_pix = ag_imgs_out[:, :self.in_channels]
        eps_rae = rae_out[:, :self.ddt.in_channels]
        ag_eps_rae = ag_rae_out[:, :self.ddt.in_channels]

        guid_t_min, guid_t_max = cfg_interval
        mask = ((t_pix >= guid_t_min) & (t_pix <= guid_t_max))

        mask_pix = mask.view(-1, *[1] * (len(eps_pix.shape) - 1))
        eps_pix = torch.where(mask_pix, ag_eps_pix + cfg_scale * (eps_pix - ag_eps_pix), eps_pix)
        rest_pix = imgs_out[:, self.in_channels:]
        final_imgs = torch.cat([eps_pix, rest_pix], dim=1) if rest_pix.shape[1] > 0 else eps_pix

        mask_rae = mask.view(-1, *[1] * (len(eps_rae.shape) - 1))
        eps_rae = torch.where(mask_rae, ag_eps_rae + cfg_scale * (eps_rae - ag_eps_rae), eps_rae)
        rest_rae = rae_out[:, self.ddt.in_channels:]
        final_rae = torch.cat([eps_rae, rest_rae], dim=1) if rest_rae.shape[1] > 0 else eps_rae

        return final_imgs, final_rae
