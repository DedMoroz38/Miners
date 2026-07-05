"""Talc polygon fill + label composition semantics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talc_quant.config import load_config  # noqa: E402
from talc_quant.constants import CLASS_BACKGROUND, CLASS_TALC  # noqa: E402
from talc_quant.gt_build import compose_label, polygons_to_talc  # noqa: E402


def test_polygon_fill_solid(tmp_path):
    txt = tmp_path / "p.txt"
    # a square covering the top-left quarter (normalized coords)
    txt.write_text("0 0.0 0.0 0.5 0.0 0.5 0.5 0.0 0.5\n")
    mask = polygons_to_talc(txt, 100, 100, min_px=6)
    assert mask[10, 10]           # inside
    assert not mask[90, 90]       # outside
    assert 0.20 < mask.mean() < 0.30  # ~quarter


def test_polygon_missing_file_empty(tmp_path):
    mask = polygons_to_talc(tmp_path / "none.txt", 50, 50, 6)
    assert mask.sum() == 0


def test_talc_overrides_gmm():
    cfg = load_config()
    # bright image so GMM would call most pixels sulfide/gray, not talc
    bgr = np.full((64, 64, 3), 200, np.uint8)
    talc = np.zeros((64, 64), bool)
    talc[16:48, 16:48] = True
    label = compose_label(bgr, talc, cfg)
    assert (label[24, 24] == CLASS_TALC)          # talc wins inside polygon
    assert (label == CLASS_TALC).sum() == 32 * 32


def test_border_black_is_background():
    cfg = load_config()
    bgr = np.zeros((80, 80, 3), np.uint8)   # all black -> border flood = background
    bgr[30:50, 30:50] = 180                 # a bright island
    label = compose_label(bgr, np.zeros((80, 80), bool), cfg)
    assert label[0, 0] == CLASS_BACKGROUND
    assert label[40, 40] != CLASS_BACKGROUND
