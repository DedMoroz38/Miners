"""FDA blend: preserves shape/dtype, changes content, beta=0 is near-identity."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talc_quant.fda import fda_blend  # noqa: E402


def _rand(h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), np.uint8)


def test_shape_dtype_preserved():
    src, tgt = _rand(seed=1), _rand(seed=2)
    out = fda_blend(src, tgt, beta=0.05)
    assert out.shape == src.shape and out.dtype == np.uint8


def test_beta_zero_near_identity():
    src, tgt = _rand(seed=1), _rand(seed=2)
    # beta*min(h,w) < 1 -> b clamps to 1 (a single DC+low bin swap); still small.
    out = fda_blend(src, tgt, beta=0.0)
    # DC term swap changes mean but structure (phase) is intact -> high correlation
    a = src.astype(np.float32).ravel() - src.mean()
    b = out.astype(np.float32).ravel() - out.mean()
    corr = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    assert corr > 0.95


def test_larger_beta_changes_more():
    src, tgt = _rand(seed=1), _rand(seed=2)
    small = fda_blend(src, tgt, 0.02).astype(np.float32)
    large = fda_blend(src, tgt, 0.25).astype(np.float32)
    assert np.abs(large - src).mean() > np.abs(small - src).mean()
