"""Geologist-facing overlay: green=normal, red=fine, blue=talc."""

from __future__ import annotations

import cv2
import numpy as np

from .classify import grain_type
from .constants import CLASS_TALC, COLOR_FINE, COLOR_NORMAL, COLOR_TALC
from .grains import Grain


def make_overlay(
    image_bgr: np.ndarray,
    class_map: np.ndarray,
    grains: list[Grain],
    alpha: float = 0.45,
) -> np.ndarray:
    """Return an RGB overlay image (talc + per-grain intergrowth colours)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if class_map.shape != rgb.shape[:2]:
        class_map = cv2.resize(class_map, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(rgb)

    color[class_map == CLASS_TALC] = COLOR_TALC

    grain_id_map = np.zeros(class_map.shape, np.int32)
    from scipy import ndimage as ndi  # local import: keep module import cheap

    from .constants import CLASS_SULFIDE
    from .grains import GrainParams
    from skimage.morphology import binary_closing, disk

    agg = binary_closing(class_map == CLASS_SULFIDE, disk(GrainParams().aggregate_px))
    grain_id_map, _ = ndi.label(agg)

    for g in grains:
        col = COLOR_FINE if grain_type(g) == "fine" else COLOR_NORMAL
        color[grain_id_map == g.label] = col

    mask = color.any(axis=2, keepdims=True)
    blended = np.where(mask, (alpha * color + (1 - alpha) * rgb).astype(np.uint8), rgb)
    return blended


def colorize_uncertainty(uncertainty: np.ndarray) -> np.ndarray:
    u = np.clip(uncertainty, 0, 1)
    heat = np.stack([u, np.zeros_like(u), 1 - u], axis=-1)
    return (heat * 255).astype(np.uint8)
