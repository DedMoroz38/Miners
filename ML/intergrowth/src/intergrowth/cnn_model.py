"""Method 2 model: an ImageNet-pretrained CNN texture classifier.

No public weights exist for ore intergrowth texture; the "ready weights" here are
a torchvision ImageNet backbone fine-tuned on the folder labels (standard transfer
learning, as used in the learned-texture paradigm of Pérez-Barnuevo et al. 2018 and
the MISIS/LumenStone work, Korshunov et al. 2025). ConvNeXt-Tiny by default with a
ResNet-18 fallback; both fall back to random init if weights can't be downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .constants import NUM_CLASSES


@dataclass(frozen=True)
class CnnConfig:
    backbone: str = "convnext_tiny"  # or "resnet18"
    n_classes: int = NUM_CLASSES
    pretrained: bool = True


def build_cnn(cfg: CnnConfig = CnnConfig()) -> nn.Module:
    """Build the backbone with an ImageNet head swapped for n_classes."""
    import torchvision.models as tvm

    if cfg.backbone == "convnext_tiny":
        try:
            weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            model = tvm.convnext_tiny(weights=weights)
        except Exception as exc:  # noqa: BLE001 - offline fallback
            print(f"convnext weights unavailable ({exc}); random init")
            model = tvm.convnext_tiny(weights=None)
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_f, cfg.n_classes)
        return model

    try:
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        model = tvm.resnet18(weights=weights)
    except Exception as exc:  # noqa: BLE001
        print(f"resnet weights unavailable ({exc}); random init")
        model = tvm.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, cfg.n_classes)
    return model
