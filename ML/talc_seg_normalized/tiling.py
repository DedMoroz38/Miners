"""Shared panorama tiling geometry (grid + reflect-pad + paste-back clipping).

The offsets logic is ported verbatim from the sulfide pipeline
(ML/sulfide_intergrowth/src/inference/sliding_window.py::_grid) so both
pipelines cut panoramas on the same kind of grid. Only GEOMETRY lives here —
no photometry (see normalize.py), no model calls, no blending (the talc mask is
stitched with logical OR by the caller; the sulfide pipeline keeps its own
cosine-blended prob-map path).

Deps: numpy + opencv only — importable from any ML venv.
"""
from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


def grid(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets covering [0, length) with the last tile flush to the edge.

    Identical to the sulfide `_grid`: for length <= tile a single offset 0;
    otherwise strided starts plus a final start at length - tile so no pixels
    are missed on the far edge.
    """
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def iter_tiles(h: int, w: int, tile: int, stride: int) -> Iterator[tuple[int, int]]:
    """Yield (y0, x0) top-left offsets of every tile over an HxW canvas."""
    for y0 in grid(h, tile, stride):
        for x0 in grid(w, tile, stride):
            yield y0, x0


def crop_padded(img: np.ndarray, y0: int, x0: int, tile: int) -> np.ndarray:
    """Cut a tile at (y0, x0); reflect-pad edge tiles up to tile x tile.

    Mirrors the sulfide sliding-window edge handling (BORDER_REFLECT), so a
    model always sees a full-size tile even at the right/bottom edges.
    """
    patch = img[y0:y0 + tile, x0:x0 + tile]
    ph, pw = patch.shape[:2]
    if ph != tile or pw != tile:
        patch = cv2.copyMakeBorder(patch, 0, tile - ph, 0, tile - pw,
                                   cv2.BORDER_REFLECT)
    return patch


def valid_extent(h: int, w: int, y0: int, x0: int, tile: int) -> tuple[int, int]:
    """Real (unpadded) tile height/width inside the canvas — for paste-back.

    Use to clip a model output before stitching: only the first (vh, vw) pixels
    of the tile correspond to real panorama pixels; the rest is reflect padding.
    """
    return min(tile, h - y0), min(tile, w - x0)
