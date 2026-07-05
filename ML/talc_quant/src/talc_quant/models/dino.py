"""Track B: frozen DINOv2 ViT-B/14 + a light conv decoder.

The self-supervised backbone is frozen, so only the small decoder trains — it
cannot overfit 42 images, its texture features are strong, and it is more robust
to colour shift than a supervised backbone. A tiny random-init variant (cfg.tiny)
keeps CPU smoke tests offline.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelCfg
from . import register_model

_DINO_HUB = "facebookresearch/dinov2"
_DINO_NAME = "dinov2_vitb14"
_PATCH = 14
_EMBED = 768


class ConvDecoder(nn.Module):
    """Patch-token grid -> per-class logits, upsampled to input resolution."""

    def __init__(self, embed: int, n_classes: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(embed, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, n_classes, 1),
        )

    def forward(self, grid: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        x = self.head(grid)
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)


class DinoSeg(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.tiny = cfg.tiny
        self.embed = 32 if cfg.tiny else _EMBED
        if cfg.tiny:
            # offline stand-in for the frozen ViT: a strided conv "tokenizer"
            self.backbone = nn.Conv2d(3, self.embed, _PATCH, stride=_PATCH)
        else:
            self.backbone = torch.hub.load(_DINO_HUB, _DINO_NAME)
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.decoder = ConvDecoder(self.embed, cfg.n_classes)

    def _tokens_grid(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        B, _, H, W = x.shape
        gh, gw = H // _PATCH, W // _PATCH
        if self.tiny:
            grid = self.backbone(x)               # [B, embed, gh, gw]
            return grid, gh, gw
        feats = self.backbone.forward_features(x)
        tok = feats["x_norm_patchtokens"]         # [B, gh*gw, embed]
        grid = tok.transpose(1, 2).reshape(B, self.embed, gh, gw)
        return grid, gh, gw

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        H, W = pixel_values.shape[-2:]
        # DINOv2 needs input divisible by patch size; pad then crop logits back.
        ph, pw = (-H) % _PATCH, (-W) % _PATCH
        x = F.pad(pixel_values, (0, pw, 0, ph), mode="reflect") if (ph or pw) else pixel_values
        grid, _, _ = self._tokens_grid(x)
        logits = self.decoder(grid, x.shape[-2:])
        return logits[..., :H, :W]


@register_model("dino_vitb14")
def _dino(cfg: ModelCfg) -> nn.Module:
    return DinoSeg(cfg)
