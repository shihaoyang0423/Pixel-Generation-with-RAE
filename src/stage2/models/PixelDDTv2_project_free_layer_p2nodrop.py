"""
PixelDDTv2_project_free_layer_p2nodrop.py — Phase-2-no-dropout variant.

Subclass of PixelDDT (from PixelDDTv2_project_free_layer) with one training-time
change: Phase 2 (extract_clean_s) never applies class label dropout to the DDT
encoder's y_embedder, regardless of model training state.

Background
----------
In the base class, _extract_clean_s calls _run_ddt_encoder, which passes
self.training to ddt.y_embedder.  During training (self.training=True) this
applies 10% class dropout independently from the Phase-3 Pixel Head dropout,
producing four possible per-sample conditioning states:

  Phase2 cond  + Phase3 cond  : 81%   (both see real label)
  Phase2 uncond + Phase3 cond : 9%    (Phase-2 drop fires, Phase-3 doesn't)
  Phase2 cond  + Phase3 uncond: 9%    (Phase-3 drop fires, Phase-2 doesn't)
  Phase2 uncond + Phase3 uncond: 1%   (both drop)

At CFG inference (current sampler code) Phase 2 always runs with the real class
label and Phase 3 splits into cond (y_real) and uncond (y_null).  So inference
only ever hits the 9% "Phase-2 cond, Phase-3 uncond" case for the uncond path
— the 10% of training where Phase 2 saw y_null is wasted, never matched at
inference.

This variant removes Phase-2 dropout so training distribution aligns with
CFG inference exactly:

  Training:   Phase2 = y_cond (always)  Phase3 = y_cond (90%) or y_null (10%)
  Inference:  Phase2 = y_cond (always)  Phase3 = y_cond (cond) or y_null (uncond)

Phase 1 (forward_rae) and Phase 3 (forward_pix) are fully inherited and
unchanged.  At inference (model.eval(), self.training=False) this class and
the base class behave identically — the difference is purely training-time.

Usage in YAML
-------------
  stage_2:
    target: stage2.models.PixelDDTv2_project_free_layer_p2nodrop.PixelDDT
    params: ...   # identical to base class params; no new parameters needed
"""

import contextlib
import torch
import torch.nn.functional as F

from .PixelDDTv2_project_free_layer import PixelDDT as _BasePixelDDT


class PixelDDT(_BasePixelDDT):
    """
    Phase-2-no-dropout variant of PixelDDT.

    All constructor parameters, weights, and inference behaviour are identical
    to PixelDDTv2_project_free_layer.PixelDDT.  Only _extract_clean_s is
    overridden: the DDT encoder's y_embedder is always called with train=False
    so class label dropout never fires in Phase 2, even during training.
    """

    def _extract_clean_s(self, x1_rae, y):
        """
        Extract Phase-2 conditioning from clean RAE latent x1_rae.

        Identical to the base-class implementation except that ddt.y_embedder
        is called with train=False, disabling class dropout regardless of
        whether the model is currently in training mode.

        This matches CFG inference, where Phase 2 always uses the real class
        label to extract s_clean.
        """
        if self.use_direct_rae_injection:
            B, C, H, W = x1_rae.shape
            x1_flat = x1_rae.flatten(2).permute(0, 2, 1)   # [B, H*W, C]
            return self.s_raw_projector(x1_flat)             # [B, H*W, encoder_hidden_size]

        B = x1_rae.shape[0]
        t_zero = torch.zeros(B, device=x1_rae.device, dtype=x1_rae.dtype)
        ddt_ctx = contextlib.nullcontext() if self.use_lora else torch.no_grad()
        with ddt_ctx:
            t_ddt = self.ddt.t_embedder(t_zero)
            y_ddt = self.ddt.y_embedder(y, False)   # ← Phase-2 dropout disabled
            c_ddt = F.silu(t_ddt + y_ddt)
            s = self.ddt.s_embedder(x1_rae)
            if self.ddt.use_pos_embed:
                s = s + self.ddt.pos_embed
            for i in range(self.ddt.num_encoder_blocks):
                s = self.ddt.blocks[i](s, c_ddt, feat_rope=self.ddt.enc_feat_rope)
        return s
