"""Illumination / colour-cast normalisation for OM images.

part1 has a yellow-green cast, part2 is neutral gray, panoramas are dark.
We normalise all of them in Lab space so the model does not learn
"yellow == talc-bearing" from the acquisition domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    l_low_pct: float = 1.0
    l_high_pct: float = 99.0
    clahe_clip: float = 2.0
    clahe_grid: int = 16
    cast_shift_limit: int = 25  # max a/b shift to avoid recolouring real phases


def _matrix_mask(l_channel: np.ndarray) -> np.ndarray:
    """Boolean mask of the dark non-ore background (below median L)."""
    return l_channel < float(np.median(l_channel))


def preprocess(image_bgr: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """Normalise a uint8 BGR image -> uint8 BGR, illumination/cast corrected."""
    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # 1) percentile stretch of L
    lo, hi = np.percentile(l, [cfg.l_low_pct, cfg.l_high_pct])
    if hi > lo:
        l = np.clip((l - lo) / (hi - lo) * 255.0, 0, 255)

    # 2) neutralise colour cast measured on the dark matrix background
    bg = _matrix_mask(l)
    if bg.sum() > 0:
        da = float(np.clip(np.mean(a[bg]) - 128.0, -cfg.cast_shift_limit, cfg.cast_shift_limit))
        db = float(np.clip(np.mean(b[bg]) - 128.0, -cfg.cast_shift_limit, cfg.cast_shift_limit))
        a = np.clip(a - da, 0, 255)
        b = np.clip(b - db, 0, 255)

    # 3) CLAHE on L for local contrast
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
    l = clahe.apply(l.astype(np.uint8)).astype(np.float32)

    out = np.stack([l, a, b], axis=-1).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
