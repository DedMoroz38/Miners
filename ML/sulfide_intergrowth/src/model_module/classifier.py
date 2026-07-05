"""ConvNeXt tile classifier for intergrowth type (normal vs fine)."""
import logging

import torch
import torch.nn as nn
from torchvision import models

from . import register_model

logger = logging.getLogger(__name__)

_BACKBONES = {
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1),
    "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
}


@register_model("convnext_tile")
class ConvNeXtTile(nn.Module):
    """Texture classifier on sulfide-bearing tiles. cfg node: model.classifier."""

    def __init__(self, cfg) -> None:
        super().__init__()
        name = str(cfg.backbone)
        if name not in _BACKBONES:
            raise KeyError(f"Unsupported backbone '{name}'")
        ctor, weights_enum = _BACKBONES[name]
        weights = weights_enum if bool(cfg.pretrained) else None
        self.backbone = ctor(weights=weights)
        in_feats = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(in_feats, int(cfg.num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def param_groups(self, lr: float, backbone_lr_mult: float) -> list[dict]:
        """Discriminative LRs: pretrained trunk slower than the fresh head."""
        head_params = list(self.backbone.classifier[2].parameters())
        head_ids = {id(p) for p in head_params}
        trunk = [p for p in self.parameters() if id(p) not in head_ids]
        return [
            {"params": trunk, "lr": lr * backbone_lr_mult},
            {"params": head_params, "lr": lr},
        ]
