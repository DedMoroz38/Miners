"""Track A: SegFormer (MiT-B2/B3, ADE20k-pretrained encoder).

Transformer context is decisive for talc — a single dark pixel is
indistinguishable from matrix without its neighbourhood — and MiT is robust to
the field→panorama domain shift and tiles cleanly. A tiny random-init variant
(cfg.tiny) runs offline for CPU smoke tests.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelCfg
from . import register_model

_CKPT = {
    "segformer_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer_b3": "nvidia/segformer-b3-finetuned-ade-512-512",
}

# Explicit MiT architecture (matches the nvidia checkpoints) so inference can
# rebuild the exact graph OFFLINE (pretrained=False) before loading fine-tuned
# weights — no network round-trip at deploy time.
_ARCH = {
    "segformer_b2": dict(depths=[3, 4, 6, 3]),
    "segformer_b3": dict(depths=[3, 4, 18, 3]),
}
_MIT_COMMON = dict(
    hidden_sizes=[64, 128, 320, 512],
    decoder_hidden_size=768,
    num_attention_heads=[1, 2, 5, 8],
    patch_sizes=[7, 3, 3, 3],
    strides=[4, 2, 2, 2],
    sr_ratios=[8, 4, 2, 1],
    mlp_ratios=[4, 4, 4, 4],
)


class SegFormerWrap(nn.Module):
    """Wraps HF SegformerForSemanticSegmentation to emit full-res logits."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        logits = self.net(pixel_values=pixel_values).logits  # [B,C,H/4,W/4]
        return F.interpolate(logits, size=pixel_values.shape[-2:],
                             mode="bilinear", align_corners=False)


def _build(cfg: ModelCfg, track: str) -> nn.Module:
    from transformers import (SegformerConfig,
                              SegformerForSemanticSegmentation)
    if cfg.tiny:
        hf = SegformerConfig(
            num_labels=cfg.n_classes, depths=[1, 1, 1, 1],
            hidden_sizes=[8, 16, 32, 64], decoder_hidden_size=32,
            num_attention_heads=[1, 1, 1, 1])
        return SegFormerWrap(SegformerForSemanticSegmentation(hf))
    if cfg.pretrained:
        net = SegformerForSemanticSegmentation.from_pretrained(
            _CKPT[track], num_labels=cfg.n_classes, ignore_mismatched_sizes=True)
        return SegFormerWrap(net)
    # offline reconstruction (inference): explicit arch, no network
    hf = SegformerConfig(num_labels=cfg.n_classes, **_MIT_COMMON, **_ARCH[track])
    return SegFormerWrap(SegformerForSemanticSegmentation(hf))


@register_model("segformer_b2")
def _b2(cfg: ModelCfg) -> nn.Module:
    return _build(cfg, "segformer_b2")


@register_model("segformer_b3")
def _b3(cfg: ModelCfg) -> nn.Module:
    return _build(cfg, "segformer_b3")
