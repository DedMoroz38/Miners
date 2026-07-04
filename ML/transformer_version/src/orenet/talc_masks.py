"""Extract talc pixel masks from blue-contour annotations.

Each annotated image (in `Области оталькования`) is a copy of an original with
talc regions outlined by a hand-drawn blue line. We detect that line, close the
contours (including where they run off the frame edge), and fill the enclosed
dark regions to obtain a binary talc mask.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk


def detect_blue_line(annotated_bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of the blue annotation stroke."""
    b, g, r = (annotated_bgr[..., i].astype(np.int32) for i in range(3))
    return (b > 120) & (b - np.maximum(r, g) > 50)


def line_to_regions(
    line: np.ndarray,
    image_lab: np.ndarray,
    dilate_px: int = 7,
    min_area_px: int = 500,
) -> np.ndarray:
    """Turn an (possibly open) blue contour into filled talc regions (bool)."""
    wall = cv2.dilate(line.astype(np.uint8), disk(dilate_px)) > 0
    walled = wall.copy()
    walled[0, :] = walled[-1, :] = walled[:, 0] = walled[:, -1] = True

    labels, n = ndi.label(~walled)
    l_median = float(np.median(image_lab[..., 0]))
    frame_area = line.shape[0] * line.shape[1]

    out = np.zeros(line.shape, dtype=bool)
    for lab_id in range(1, n + 1):
        comp = labels == lab_id
        area = int(comp.sum())
        if area < min_area_px or area > 0.6 * frame_area:
            continue
        # fraction of the region's outer ring that is the contour wall:
        # ~1 for a region enclosed by the loop, low for the open background
        ring = ndi.binary_dilation(comp, iterations=3) & ~comp
        wall_frac = (ring & wall).sum() / max(int(ring.sum()), 1)
        if wall_frac < 0.5:
            continue
        # reject clearly bright regions (loose; talc is matrix-like, not bright)
        if float(image_lab[..., 0][comp].mean()) >= l_median * 1.25:
            continue
        out |= comp

    return ndi.binary_fill_holes(out)


def extract_pair(original: Path, annotated: Path) -> np.ndarray:
    """Load an (original, annotated) pair -> uint8 {0,1} talc mask."""
    orig = cv2.imread(str(original), cv2.IMREAD_COLOR)
    ann = cv2.imread(str(annotated), cv2.IMREAD_COLOR)
    if orig is None or ann is None:
        raise FileNotFoundError(f"cannot read {original} / {annotated}")
    if ann.shape != orig.shape:
        ann = cv2.resize(ann, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    lab = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB)
    mask = line_to_regions(detect_blue_line(ann), lab)
    return mask.astype(np.uint8)
