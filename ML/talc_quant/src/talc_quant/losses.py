"""Segmentation loss for the rare talc class + an area-consistency term.

Base = weighted CE + Dice + Focal-Tversky (the nnU-Net-style Dice+CE recipe with
Focal-Tversky punishing missed talc harder than false alarms; Abraham & Khan
2019). Ported from orenet.losses but adds:

  area term:  area_w * |Σ p_talc − Σ y_talc| / N_valid

which directly penalizes the QUANTITY the product is graded on (talc area
fraction), pulling the model toward an unbiased soft count. The CE weight is kept
≥ the region terms so soft-Dice volume bias (Bertels 2022) does not dominate;
the residual bias is removed later by post-hoc calibration.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import LossCfg
from .constants import CLASS_TALC, IGNORE_INDEX, NUM_CLASSES


def class_weights(cfg: LossCfg, n: int = NUM_CLASSES) -> torch.Tensor:
    """Uniform weights with an extra boost on the talc channel."""
    w = torch.ones(n)
    w[CLASS_TALC] = cfg.talc_class_weight
    return w


class SegLoss(nn.Module):
    def __init__(self, cfg: LossCfg, n_classes: int = NUM_CLASSES,
                 ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.cfg = cfg
        self.n = n_classes
        self.ignore = ignore_index
        self.register_buffer("w", class_weights(cfg, n_classes))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore
        if valid.sum() == 0:
            return logits.sum() * 0.0
        probs = F.softmax(logits, dim=1)
        tgt = target.clone()
        tgt[~valid] = 0
        onehot = F.one_hot(tgt, self.n).permute(0, 3, 1, 2).float()
        mask = valid.unsqueeze(1).float()
        probs_m = probs * mask
        onehot_m = onehot * mask

        dims = (0, 2, 3)
        tp = (probs_m * onehot_m).sum(dims)
        fp = (probs_m * (1 - onehot_m)).sum(dims)
        fn = ((1 - probs_m) * onehot_m).sum(dims)

        dice = (2 * tp + 1) / (2 * tp + fp + fn + 1)
        tversky = (tp + 1) / (
            tp + self.cfg.tversky_alpha * fp + self.cfg.tversky_beta * fn + 1)
        focal_tversky = (1 - tversky).pow(self.cfg.tversky_gamma)

        w = self.w.to(logits.device)
        dice_loss = ((1 - dice) * w).sum() / w.sum()
        ft_loss = (focal_tversky * w).sum() / w.sum()
        ce_loss = F.cross_entropy(logits, target, weight=w,
                                  ignore_index=self.ignore)

        # area-consistency on the talc channel (soft count vs GT count)
        n_valid = mask.sum().clamp(min=1.0)
        soft_area = probs_m[:, CLASS_TALC].sum()
        gt_area = onehot_m[:, CLASS_TALC].sum()
        area_loss = (soft_area - gt_area).abs() / n_valid

        return (self.cfg.ce_w * ce_loss
                + self.cfg.dice_w * dice_loss
                + self.cfg.tversky_w * ft_loss
                + self.cfg.area_w * area_loss)
