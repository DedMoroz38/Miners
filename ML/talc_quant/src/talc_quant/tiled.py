"""Panorama tiling with cosine-window PROBABILITY blending.

Grid geometry mirrors ML/talc_seg_normalized/tiling.py (last tile flush to the
edge, reflect-pad edge tiles). The key difference from the existing talc_infer:
we blend per-tile SOFTMAX PROBABILITIES with a 2-D raised-cosine window instead
of OR-ing binary masks — logical OR double-counts the 50%-overlap seams and
systematically inflates the area, which directly breaks the ±3% budget.
"""
from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


def grid(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets covering [0, length); last tile flush to the far edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def iter_tiles(h: int, w: int, tile: int, stride: int) -> Iterator[tuple[int, int]]:
    for y0 in grid(h, tile, stride):
        for x0 in grid(w, tile, stride):
            yield y0, x0


def crop_padded(img: np.ndarray, y0: int, x0: int, tile: int) -> np.ndarray:
    """Cut a tile; reflect-pad edge tiles up to tile×tile."""
    patch = img[y0:y0 + tile, x0:x0 + tile]
    ph, pw = patch.shape[:2]
    if ph != tile or pw != tile:
        patch = cv2.copyMakeBorder(patch, 0, tile - ph, 0, tile - pw,
                                   cv2.BORDER_REFLECT)
    return patch


def cosine_window(tile: int) -> np.ndarray:
    """2-D raised-cosine (Hann) weight, strictly > 0 so every pixel is covered."""
    w1 = np.hanning(tile + 2)[1:-1]          # drop the zero endpoints
    w1 = np.clip(w1, 1e-3, None).astype(np.float32)
    return np.outer(w1, w1)


class ProbAccumulator:
    """Weighted running sum of per-class probabilities over a full canvas."""

    def __init__(self, h: int, w: int, n_classes: int) -> None:
        self.acc = np.zeros((n_classes, h, w), np.float32)
        self.wsum = np.zeros((h, w), np.float32)
        self.h, self.w = h, w

    def add(self, probs: np.ndarray, y0: int, x0: int, weight: np.ndarray) -> None:
        """probs: [C, tile, tile] softmax; clipped to the real (unpadded) extent."""
        vh = min(probs.shape[1], self.h - y0)
        vw = min(probs.shape[2], self.w - x0)
        p = probs[:, :vh, :vw]
        wt = weight[:vh, :vw]
        self.acc[:, y0:y0 + vh, x0:x0 + vw] += p * wt
        self.wsum[y0:y0 + vh, x0:x0 + vw] += wt

    def result(self) -> np.ndarray:
        """Normalized [C, H, W] probability map (weights guaranteed > 0)."""
        return self.acc / np.clip(self.wsum, 1e-6, None)[None]
