"""Backbone for tile texture classification (fine vs normal intergrowth).

ConvNeXt-Small (ImageNet) by default — a strong texture backbone that fits a
V100-32GB comfortably. ConvNeXt-Tiny / ResNet-50 as lighter fallbacks. All fall
back to random init if weights can't be fetched offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "convnext_small"  # convnext_small | convnext_tiny | resnet50
    n_classes: int = 2
    pretrained: bool = True
    drop_rate: float = 0.2


def build_model(cfg: ModelConfig = ModelConfig()) -> nn.Module:
    import torchvision.models as tvm

    def _try(fn, weights_enum):
        try:
            return fn(weights=weights_enum if cfg.pretrained else None)
        except Exception as exc:  # noqa: BLE001 - offline fallback
            print(f"{cfg.backbone} weights unavailable ({exc}); random init")
            return fn(weights=None)

    if cfg.backbone.startswith("convnext"):
        fn = tvm.convnext_small if cfg.backbone == "convnext_small" else tvm.convnext_tiny
        enum = (tvm.ConvNeXt_Small_Weights.IMAGENET1K_V1 if cfg.backbone == "convnext_small"
                else tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model = _try(fn, enum)
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(nn.Dropout(cfg.drop_rate), nn.Linear(in_f, cfg.n_classes))
        return model

    model = _try(tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(nn.Dropout(cfg.drop_rate), nn.Linear(model.fc.in_features, cfg.n_classes))
    return model
