"""Pretrained backbone factory for 3-class ore classification.

Why these models (and not YOLO/U-Net):
  * The task is *whole-image* classification (one label per thin section), so a
    segmentation net (U-Net) or a detector (YOLO) is the wrong tool — we need an
    image encoder + classifier head, fine-tuned from ImageNet.
  * efficientnet_v2_s: best accuracy/params trade-off, strong transfer on small,
    imbalanced datasets -> chosen as the default.
  * convnext_tiny / resnet50: solid modern / classic alternatives for comparison.
"""
import torch.nn as nn
from torchvision import models

SUPPORTED = ["efficientnet_v2_s", "convnext_tiny", "resnet50"]


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "efficientnet_v2_s":
        w = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_v2_s(weights=w)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
    elif name == "convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.convnext_tiny(weights=w)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, num_classes)
    elif name == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model '{name}'. Supported: {SUPPORTED}")
    return m
