"""U-Net with a ResNet encoder (torchvision, ImageNet-pretrained).

Compact hand-rolled implementation to avoid extra dependencies; equivalent in
spirit to segmentation_models_pytorch Unet(resnet34).
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from . import register_model

logger = logging.getLogger(__name__)

_ENCODERS = {
    "resnet18": (models.resnet18, [64, 64, 128, 256, 512]),
    "resnet34": (models.resnet34, [64, 64, 128, 256, 512]),
    "resnet50": (models.resnet50, [64, 256, 512, 1024, 2048]),
}


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(x)), inplace=True)


@register_model("resnet_unet")
class ResNetUNet(nn.Module):
    """Binary sulfide segmenter. cfg node: model.segmenter."""

    def __init__(self, cfg) -> None:
        super().__init__()
        encoder_name = str(cfg.encoder)
        if encoder_name not in _ENCODERS:
            raise KeyError(f"Unsupported encoder '{encoder_name}'")
        ctor, chs = _ENCODERS[encoder_name]
        weights = "IMAGENET1K_V1" if bool(cfg.pretrained) else None
        backbone = ctor(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        dec = [256, 128, 64, 32]
        self.dec4 = _DecoderBlock(chs[4], chs[3], dec[0])
        self.dec3 = _DecoderBlock(dec[0], chs[2], dec[1])
        self.dec2 = _DecoderBlock(dec[1], chs[1], dec[2])
        self.dec1 = _DecoderBlock(dec[2], chs[0], dec[3])
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(dec[3], 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, int(cfg.out_channels), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)                 # 1/2
        s1 = self.layer1(self.pool(s0))   # 1/4
        s2 = self.layer2(s1)              # 1/8
        s3 = self.layer3(s2)              # 1/16
        s4 = self.layer4(s3)              # 1/32
        d = self.dec4(s4, s3)
        d = self.dec3(d, s2)
        d = self.dec2(d, s1)
        d = self.dec1(d, s0)
        return self.head(d)               # logits, full resolution
