"""Tiled blending: grid coverage, positive weights, correct prob reconstruction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talc_quant.tiled import (ProbAccumulator, cosine_window, grid,  # noqa: E402
                              iter_tiles)


def test_grid_covers_far_edge():
    g = grid(1000, 256, 179)
    assert g[0] == 0
    assert g[-1] == 1000 - 256          # last tile flush to edge
    assert all(0 <= s <= 1000 - 256 for s in g)


def test_grid_small_length():
    assert grid(100, 256, 128) == [0]


def test_cosine_window_positive():
    w = cosine_window(64)
    assert w.shape == (64, 64)
    assert (w > 0).all()                # every pixel gets nonzero weight


def test_accumulator_reconstructs_constant_field():
    """A constant probability field must survive tiled blending unchanged."""
    h, w, c, tile, stride = 300, 400, 5, 128, 64
    true = np.zeros((c, h, w), np.float32)
    true[3] = 0.42                       # constant talc prob everywhere
    true[0] = 0.58
    acc = ProbAccumulator(h, w, c)
    win = cosine_window(tile)
    for y0, x0 in iter_tiles(h, w, tile, stride):
        # each tile "sees" the same constant probs (pad region irrelevant here)
        p = np.zeros((c, tile, tile), np.float32)
        p[3] = 0.42
        p[0] = 0.58
        acc.add(p, y0, x0, win)
    out = acc.result()
    assert np.allclose(out[3], 0.42, atol=1e-4)
    assert np.allclose(out[0], 0.58, atol=1e-4)


def test_accumulator_weights_cover_everywhere():
    h, w, tile, stride = 200, 250, 128, 64
    acc = ProbAccumulator(h, w, 5)
    win = cosine_window(tile)
    for y0, x0 in iter_tiles(h, w, tile, stride):
        acc.add(np.ones((5, tile, tile), np.float32) * 0.2, y0, x0, win)
    assert (acc.wsum > 0).all()          # no uncovered pixel
