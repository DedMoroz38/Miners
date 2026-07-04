"""SegFormer wrapper for 5-class phase segmentation.

Uses an ADE20k-pretrained MiT-B2 encoder (transfer learning is the main
data-efficiency lever). A tiny random-init variant is available for offline
smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .constants import NUM_CLASSES


@dataclass(frozen=True)
class SegmenterConfig:
    checkpoint: str = "nvidia/segformer-b2-finetuned-ade-512-512"
    n_classes: int = NUM_CLASSES
    pretrained: bool = True
    tiny: bool = False  # offline random-init for tests


def build_segmenter(cfg: SegmenterConfig = SegmenterConfig()) -> nn.Module:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    if cfg.tiny:
        hf_cfg = SegformerConfig(
            num_labels=cfg.n_classes,
            depths=[1, 1, 1, 1],
            hidden_sizes=[8, 16, 32, 64],
            decoder_hidden_size=32,
            num_attention_heads=[1, 1, 1, 1],
        )
        return SegformerForSemanticSegmentation(hf_cfg)

    if cfg.pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(
            cfg.checkpoint,
            num_labels=cfg.n_classes,
            ignore_mismatched_sizes=True,
        )
    hf_cfg = SegformerConfig.from_pretrained(cfg.checkpoint, num_labels=cfg.n_classes)
    return SegformerForSemanticSegmentation(hf_cfg)


def forward_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return logits upsampled to the input resolution [B, C, H, W]."""
    logits = model(pixel_values=images).logits  # [B, C, H/4, W/4]
    return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
