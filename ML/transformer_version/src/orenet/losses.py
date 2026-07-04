"""Segmentation loss for imbalanced phases: Dice + Focal-Tversky + CE.

Focal-Tversky (Salehi 2017; Abraham 2019) penalises missing the rare talc class
harder than false alarms. Region losses alone give a weak, batch-smoothed
gradient that stalls early training, so we add a weighted cross-entropy term for
a strong per-pixel signal (the Dice+CE recipe of nnU-Net, Isensee 2021).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .constants import IGNORE_INDEX, NUM_CLASSES


class DiceFocalTversky(nn.Module):
    def __init__(
        self,
        n_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.3,
        dice_w: float = 0.5,
        tversky_w: float = 0.5,
        ce_w: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.n = n_classes
        self.ignore = ignore_index
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.dice_w, self.tversky_w, self.ce_w = dice_w, tversky_w, ce_w
        self.register_buffer(
            "w", class_weights if class_weights is not None else torch.ones(n_classes)
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore
        if valid.sum() == 0:
            return logits.sum() * 0.0
        probs = F.softmax(logits, dim=1)
        tgt = target.clone()
        tgt[~valid] = 0
        onehot = F.one_hot(tgt, self.n).permute(0, 3, 1, 2).float()
        mask = valid.unsqueeze(1).float()
        probs = probs * mask
        onehot = onehot * mask

        dims = (0, 2, 3)
        tp = (probs * onehot).sum(dims)
        fp = (probs * (1 - onehot)).sum(dims)
        fn = ((1 - probs) * onehot).sum(dims)

        dice = (2 * tp + 1) / (2 * tp + fp + fn + 1)
        tversky = (tp + 1) / (tp + self.alpha * fp + self.beta * fn + 1)
        focal_tversky = (1 - tversky).pow(self.gamma)

        w = self.w.to(logits.device)
        dice_loss = ((1 - dice) * w).sum() / w.sum()
        ft_loss = (focal_tversky * w).sum() / w.sum()
        ce_loss = F.cross_entropy(logits, target, weight=w, ignore_index=self.ignore)
        return self.dice_w * dice_loss + self.tversky_w * ft_loss + self.ce_w * ce_loss
