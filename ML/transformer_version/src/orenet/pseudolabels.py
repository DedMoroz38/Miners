"""GMM pseudo-labels for sulfide / gray-ore / matrix from the L channel.

Sulfides (bright), magnetite (mid-gray) and matrix (dark) are separable by
reflectance. We fit a 1-D GMM on L and keep only high-confidence pixels; the
rest become IGNORE_INDEX so they never corrupt the loss. Talc is NOT pseudo-
labelled here — it only comes from the 42 hand masks.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

from .constants import CLASS_GRAY, CLASS_MATRIX, CLASS_SULFIDE, IGNORE_INDEX


def pseudo_label(
    image_bgr: np.ndarray,
    confidence: float = 0.95,
    sulfide_min_l: int = 140,
    max_pixels: int = 200_000,
) -> np.ndarray:
    """Return a uint8 label map with values {0,1,2,255}."""
    l = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    flat = l.reshape(-1, 1)

    # subsample for a fast, stable fit
    if flat.shape[0] > max_pixels:
        idx = np.random.default_rng(0).choice(flat.shape[0], max_pixels, replace=False)
        sample = flat[idx]
    else:
        sample = flat

    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0)
    gmm.fit(sample)

    order = np.argsort(gmm.means_.ravel())  # dark -> mid -> bright
    comp_to_class = {
        int(order[0]): CLASS_MATRIX,
        int(order[1]): CLASS_GRAY,
        int(order[2]): CLASS_SULFIDE,
    }

    proba = gmm.predict_proba(flat)
    conf = proba.max(axis=1)
    comp = proba.argmax(axis=1)

    out = np.full(flat.shape[0], IGNORE_INDEX, dtype=np.uint8)
    keep = conf >= confidence
    mapped = np.array([comp_to_class[int(c)] for c in comp], dtype=np.uint8)
    out[keep] = mapped[keep]

    out = out.reshape(l.shape)
    # guard: a "sulfide" pixel must be absolutely bright, else drop to ignore
    out[(out == CLASS_SULFIDE) & (l < sulfide_min_l)] = IGNORE_INDEX
    return out
