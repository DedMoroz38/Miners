"""Inference: TTA tile scoring, image-level probability, and region heatmaps.

The per-tile fine-probability, painted back onto the image, IS the predicted
"область срастаний" (red = тонкие, green = обычные). The image label is the mean
tile probability (soft voting), optionally ensembled over folds and TTA.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .dataset import IMAGENET_MEAN, IMAGENET_STD


def _tile_grid(path, tile: int, stride: int):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None, [], (0, 0)
    img = cv2.copyMakeBorder(img, 0, max(0, tile - img.shape[0]), 0,
                             max(0, tile - img.shape[1]), cv2.BORDER_REFLECT)
    h, w = img.shape[:2]
    coords = [(y, x) for y in range(0, max(1, h - tile + 1), stride)
              for x in range(0, max(1, w - tile + 1), stride)]
    return img, coords, (h, w)


def _batch_tensor(img, coords, tile):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    out = [rgb[y:y + tile, x:x + tile].transpose(2, 0, 1) for y, x in coords]
    return torch.from_numpy(np.stack(out))


@torch.no_grad()
def _tta_logits(models, batch, device, tta: bool = True):
    """Mean softmax over models x {identity, hflip, vflip} TTA (or identity only)."""
    flips = (None, (-1,), (-2,)) if tta else (None,)
    prob = 0.0
    n = 0
    for m in models:
        for flip in flips:
            x = torch.flip(batch, dims=flip) if flip else batch
            prob = prob + F.softmax(m(x.to(device)), dim=1).cpu()
            n += 1
    return prob / n


@torch.no_grad()
def predict_image(models, path, device, tile: int = 448, stride: int = 224,
                  batch: int = 32, tta: bool = True):
    """Return (label, p_fine_image, heatmap_coords, per_tile_p_fine, hw)."""
    for m in models:
        m.eval()
    img, coords, hw = _tile_grid(path, tile, stride)
    if img is None or not coords:
        return 0, 0.0, [], np.zeros(0), (0, 0)
    tens = _batch_tensor(img, coords, tile)
    probs = []
    for i in range(0, len(tens), batch):
        probs.append(_tta_logits(models, tens[i:i + batch], device, tta=tta))
    p = torch.cat(probs)
    p_fine = p[:, 1].numpy()
    return int(p_fine.mean() >= 0.5), float(p_fine.mean()), coords, p_fine, hw


def region_overlay(path, coords, per_tile_p_fine, hw, tile: int = 448):
    """Paint a red(fine)/green(normal) heatmap of intergrowth regions over the photo."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = hw
    heat = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for (y, x), pf in zip(coords, per_tile_p_fine):
        heat[y:y + tile, x:x + tile] += pf
        cnt[y:y + tile, x:x + tile] += 1.0
    heat = heat / np.maximum(cnt, 1e-6)
    heat = heat[: img.shape[0], : img.shape[1]]
    color = np.zeros_like(img)
    color[..., 2] = (heat * 255).astype(np.uint8)          # red  = тонкие
    color[..., 1] = ((1 - heat) * 255).astype(np.uint8)    # green = обычные
    return cv2.addWeighted(img, 0.6, color, 0.4, 0)
