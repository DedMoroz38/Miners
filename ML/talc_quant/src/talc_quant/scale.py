"""Magnification parsing -> resample factor to the canonical (×10) scale.

Field filenames encode magnification (e.g. "2550374-2 10х.JPG", "...5x.jpg").
We resample every field to a common µm/px so a talc grain has a consistent
pixel size regardless of the shot's zoom. Panoramas have no in-name magnification
(see spec open question) — we fall back to cfg.scale.panorama_mag.
"""
from __future__ import annotations

import re

from .config import ScaleCfg

# Magnification token: digits adjacent to x/х/× on either side.
# "10x", "10х" (Cyrillic х), "5 x" (number-first) or "×20" (symbol-first).
_MAG_RE = re.compile(r"(?:(\d+)\s*[xх×])|(?:[xх×]\s*(\d+))", re.IGNORECASE)


def parse_magnification(filename: str) -> float | None:
    """Return the magnification integer parsed from a filename, or None."""
    m = _MAG_RE.search(filename)
    if m:
        val = int(m.group(1) or m.group(2))
        return float(val) if val > 0 else None
    return None


def scale_factor(filename: str, cfg: ScaleCfg, is_panorama: bool = False) -> float:
    """Resize factor mapping this image to the canonical target magnification.

    factor > 1 upsamples (image was lower mag than target), < 1 downsamples.
    A ×5 image at target ×10 -> factor 10/5 = 2.0 (grains doubled to match ×10).
    """
    mag = cfg.panorama_mag if is_panorama else parse_magnification(filename)
    if mag is None or mag <= 0:
        mag = cfg.target_mag  # unknown -> assume already canonical (factor 1)
    return cfg.target_mag / mag
