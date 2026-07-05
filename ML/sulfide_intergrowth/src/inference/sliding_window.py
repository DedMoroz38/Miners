"""Sliding-window inference helpers for arbitrarily large panoramas."""
import logging
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..data_module.augmentation import to_tensor_chw

logger = logging.getLogger(__name__)


def _grid(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets covering [0, length) with the last tile flush to the edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _cosine_window(tile: int) -> np.ndarray:
    """2D raised-cosine weights to blend overlapping tiles seamlessly."""
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * (np.arange(tile) + 0.5) / tile)
    win = np.outer(w, w).astype(np.float32)
    return np.clip(win, 1e-3, None)


def _batched(items: list, batch_size: int) -> Iterator[list]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


@torch.no_grad()
def predict_prob_map(model: nn.Module, bgr: np.ndarray, tile: int, stride: int,
                     batch_size: int, device: torch.device) -> np.ndarray:
    """Dense sigmoid probability map for a binary segmenter.

    Args:
        model: segmentation model returning 1-channel logits.
        bgr: preprocessed uint8 BGR image (any size).
        tile: window size fed to the model.
        stride: window stride (< tile for overlap).
        batch_size: tiles per forward pass.
        device: torch device.

    Returns:
        float32 HxW probability map in [0, 1].
    """
    model.eval()
    h, w = bgr.shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    win = _cosine_window(tile)
    coords = [(y, x) for y in _grid(h, tile, stride) for x in _grid(w, tile, stride)]
    logger.info("segmenter sliding window: %d tiles (%dpx / stride %d)", len(coords), tile, stride)
    for batch in _batched(coords, batch_size):
        tensors = []
        for y, x in batch:
            patch = bgr[y:y + tile, x:x + tile]
            if patch.shape[0] != tile or patch.shape[1] != tile:
                patch = cv2.copyMakeBorder(patch, 0, tile - patch.shape[0],
                                           0, tile - patch.shape[1], cv2.BORDER_REFLECT)
            tensors.append(to_tensor_chw(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
        x_t = torch.from_numpy(np.stack(tensors)).to(device)
        prob = torch.sigmoid(model(x_t))[:, 0].float().cpu().numpy()
        for (y, x), p in zip(batch, prob):
            ph = min(tile, h - y)
            pw = min(tile, w - x)
            acc[y:y + ph, x:x + pw] += p[:ph, :pw] * win[:ph, :pw]
            weight[y:y + ph, x:x + pw] += win[:ph, :pw]
    return acc / np.maximum(weight, 1e-6)


@torch.no_grad()
def predict_class_grid(models: list[nn.Module], bgr: np.ndarray, sulfide_mask: np.ndarray,
                       tile: int, stride: int, batch_size: int, device: torch.device,
                       min_frac: float = 0.01, tta: bool = True) -> np.ndarray:
    """Coarse P(fine) grid over the panorama from an ensemble of tile classifiers.

    Cells whose tile holds < min_frac sulfide are NaN (no evidence there).

    Returns:
        float32 (n_gy, n_gx) grid aligned with tile top-left offsets.
    """
    for m in models:
        m.eval()
    h, w = bgr.shape[:2]
    ys, xs = _grid(h, tile, stride), _grid(w, tile, stride)
    grid = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
    integral = cv2.integral((sulfide_mask > 0).astype(np.uint8))
    coords = []
    for gy, y in enumerate(ys):
        for gx, x in enumerate(xs):
            y1, x1 = min(y + tile, h), min(x + tile, w)
            s = integral[y1, x1] - integral[y, x1] - integral[y1, x] + integral[y, x]
            if s / float((y1 - y) * (x1 - x)) >= min_frac:
                coords.append((gy, gx, y, x))
    logger.info("classifier sliding window: %d/%d tiles carry sulfide",
                len(coords), len(ys) * len(xs))
    for batch in _batched(coords, batch_size):
        tensors = []
        for _, _, y, x in batch:
            patch = bgr[y:y + tile, x:x + tile]
            if patch.shape[0] != tile or patch.shape[1] != tile:
                patch = cv2.copyMakeBorder(patch, 0, tile - patch.shape[0],
                                           0, tile - patch.shape[1], cv2.BORDER_REFLECT)
            tensors.append(to_tensor_chw(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
        x_t = torch.from_numpy(np.stack(tensors)).to(device)
        prob = torch.zeros(len(batch), device=device)
        n_views = 0
        for m in models:
            prob += torch.softmax(m(x_t), dim=1)[:, 1]
            n_views += 1
            if tta:
                prob += torch.softmax(m(torch.flip(x_t, dims=[3])), dim=1)[:, 1]
                n_views += 1
        prob = (prob / n_views).cpu().numpy()
        for (gy, gx, _, _), p in zip(batch, prob):
            grid[gy, gx] = p
    return grid
