"""Fourier Domain Adaptation (Yang & Soatto, 2020).

Swap the low-frequency amplitude of a source patch with that of a random
panorama tile from the FDA bank, keeping the source phase (and thus its content
and labels). This cheaply drags training patches toward the panorama's dark,
low-contrast "look" without any panorama labels — closing the field→panorama gap
that the spec flags as the dominant risk.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class FDABank:
    """Loads panorama style tiles once; `apply` blends a source patch toward one."""

    def __init__(self, bank_dir: Path, beta: float = 0.03, seed: int = 0) -> None:
        self.beta = beta
        self._rng = np.random.default_rng(seed)
        self._tiles = [p for p in sorted(Path(bank_dir).glob("*.jpg"))]

    def __len__(self) -> int:
        return len(self._tiles)

    def _random_style(self, h: int, w: int) -> np.ndarray | None:
        if not self._tiles:
            return None
        p = self._tiles[self._rng.integers(len(self._tiles))]
        tile = cv2.imread(str(p))
        if tile is None:
            return None
        return cv2.resize(tile, (w, h))

    def apply(self, src_bgr: np.ndarray) -> np.ndarray:
        """Return src with its low-freq amplitude replaced by a bank tile's."""
        style = self._random_style(*src_bgr.shape[:2])
        if style is None:
            return src_bgr
        return fda_blend(src_bgr, style, self.beta)


def fda_blend(src_bgr: np.ndarray, tgt_bgr: np.ndarray, beta: float) -> np.ndarray:
    """Per-channel low-frequency amplitude swap. beta = half-window as a fraction
    of the smaller side (0 -> no change, larger -> more style transferred)."""
    src = src_bgr.astype(np.float32)
    tgt = tgt_bgr.astype(np.float32)
    h, w = src.shape[:2]
    b = max(1, int(beta * min(h, w)))
    out = np.empty_like(src)
    for c in range(3):
        fs = np.fft.fft2(src[..., c])
        ft = np.fft.fft2(tgt[..., c])
        amp_s, pha_s = np.abs(fs), np.angle(fs)
        amp_t = np.abs(ft)
        amp_s = np.fft.fftshift(amp_s)
        amp_t = np.fft.fftshift(amp_t)
        cy, cx = h // 2, w // 2
        amp_s[cy - b:cy + b + 1, cx - b:cx + b + 1] = \
            amp_t[cy - b:cy + b + 1, cx - b:cx + b + 1]
        amp_s = np.fft.ifftshift(amp_s)
        rec = amp_s * np.exp(1j * pha_s)
        out[..., c] = np.real(np.fft.ifft2(rec))
    return np.clip(out, 0, 255).astype(np.uint8)
