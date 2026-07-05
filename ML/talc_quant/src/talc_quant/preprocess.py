"""Canonical photometric preprocessing — one profile for train AND inference.

Composition of two EXISTING, proven implementations (spec §5; reuse mandate):

  1. Anchor L-normalization — the sulfide pipeline's `normalize_exposure_anchor`
     (ML/sulfide_intergrowth/src/data_module/preprocessing.py), loaded standalone
     the same way ML/talc_seg_normalized/normalize.py does it. Fixes exposure
     without erasing absolute reflectance (so sulfides stay bright, matrix dark).

  2. a/b colour-cast neutralization — ported from
     ML/transformer_version/src/orenet/preprocess.py (the matrix-anchored a/b
     shift). The anchor step touches only L, so part1's olive cast and the
     panoramas' grey survive it (verified visually). This step removes the hue
     so the model cannot learn "olive == talc-bearing" from the acquisition
     domain. We deliberately DROP orenet's percentile-stretch + CLAHE to keep
     absolute L levels intact.

Applied PER TILE / PER FIELD (anchors are computed from the passed image) — never
on a whole panorama at once.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np

# Max a/b shift, mirrors orenet.preprocess.cast_shift_limit (avoid recolouring
# genuine phases if a field is legitimately tinted).
CAST_SHIFT_LIMIT = 25.0


def _load_sulfide_anchor(sulfide_preprocess_path: Path):
    """Import the sulfide preprocessing FILE standalone and return its
    `normalize_exposure_anchor` (single source of truth for exposure)."""
    if not sulfide_preprocess_path.is_file():
        raise FileNotFoundError(
            f"Sulfide preprocessing not found at {sulfide_preprocess_path}. "
            "The canonical L-anchor is shared with the sulfide pipeline."
        )
    spec = importlib.util.spec_from_file_location(
        "sulfide_preprocessing_tq", str(sulfide_preprocess_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.normalize_exposure_anchor


def neutralize_cast(bgr: np.ndarray, shift_limit: float = CAST_SHIFT_LIMIT) -> np.ndarray:
    """Center a/b of the dark non-ore matrix to neutral gray (ported from
    orenet.preprocess step 2). Cast measured on below-median-L pixels so
    sulfide-heavy fields do not drag the estimate."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    matrix = l < float(np.median(l))
    if matrix.sum() > 0:
        da = float(np.clip(np.mean(a[matrix]) - 128.0, -shift_limit, shift_limit))
        db = float(np.clip(np.mean(b[matrix]) - 128.0, -shift_limit, shift_limit))
        lab[..., 1] = np.clip(a - da, 0, 255)
        lab[..., 2] = np.clip(b - db, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def background_mask(bgr: np.ndarray, dark_l: int = 14, min_frac: float = 0.002) -> np.ndarray:
    """Boolean mask of mounting resin / field background: near-black pixels that
    are connected to the image border (flood from the frame edge through the
    dark threshold). Interior dark matrix is NOT flagged because it is enclosed
    by brighter phases. Used to exclude resin from the talc-fraction denominator.

    Returns all-False if the flooded area is negligible (< min_frac of the frame)
    — most field shots have no resin border.
    """
    lab_l = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0]
    dark = (lab_l < dark_l).astype(np.uint8)
    h, w = dark.shape
    # flood-fill from a 1px border ring through connected dark pixels
    ff = np.zeros((h + 2, w + 2), np.uint8)
    filled = dark.copy()
    for x in range(w):
        if dark[0, x]:
            cv2.floodFill(filled, ff, (x, 0), 2)
        if dark[h - 1, x]:
            cv2.floodFill(filled, ff, (x, h - 1), 2)
    for y in range(h):
        if dark[y, 0]:
            cv2.floodFill(filled, ff, (0, y), 2)
        if dark[y, w - 1]:
            cv2.floodFill(filled, ff, (w - 1, y), 2)
    bg = filled == 2
    if bg.sum() < min_frac * h * w:
        return np.zeros((h, w), dtype=bool)
    return bg


class CanonicalPreprocessor:
    """Callable that applies anchor-L then a/b cast neutralization.

    Holds the sulfide anchor fn (loaded once). Build one per process and reuse.
    """

    def __init__(self, sulfide_preprocess_path: Path,
                 shift_limit: float = CAST_SHIFT_LIMIT) -> None:
        self._anchor = _load_sulfide_anchor(sulfide_preprocess_path)
        self._shift = shift_limit

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """uint8 BGR -> uint8 BGR in the canonical profile (same shape)."""
        out = self._anchor(bgr)           # sulfide exposure anchor (L only)
        out = neutralize_cast(out, self._shift)
        return out
