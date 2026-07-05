"""Magnification parsing + canonical resample factor."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talc_quant.config import ScaleCfg  # noqa: E402
from talc_quant.scale import parse_magnification, scale_factor  # noqa: E402

CFG = ScaleCfg(target_mag=10.0, panorama_mag=10.0, jitter=0.2)


def test_parse_latin_and_cyrillic():
    assert parse_magnification("2550376-1 5x.jpg") == 5.0
    assert parse_magnification("2550374-2 10х.JPG") == 10.0  # Cyrillic х
    assert parse_magnification("foo ×20 bar.png") == 20.0


def test_parse_missing_returns_none():
    assert parse_magnification("DSCN3048.jpg") is None
    assert parse_magnification("-1.jpg") is None


def test_scale_factor_upsamples_low_mag():
    # ×5 image, target ×10 -> factor 2.0
    assert scale_factor("x 5x.jpg", CFG) == 2.0


def test_scale_factor_identity_for_target():
    assert scale_factor("x 10x.jpg", CFG) == 1.0


def test_scale_factor_unknown_is_identity():
    assert scale_factor("DSCN3048.jpg", CFG) == 1.0


def test_panorama_uses_config_mag():
    cfg = ScaleCfg(target_mag=10.0, panorama_mag=5.0, jitter=0.2)
    assert scale_factor("panorama_4.jpg", cfg, is_panorama=True) == 2.0
