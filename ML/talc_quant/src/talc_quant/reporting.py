"""Render the geologist-facing artifacts and the machine record.

Outputs (contract compatible with ML/merge/merge_phases.py consumers):
  talc_mask.png   uint8 HxW, 255 where talc (binary mask, area-matched threshold)
  overlay.png     blue semi-transparent talc over the original
  heatmap.png     local talc-fraction grid (neighbourhood inhomogeneity)
  entropy.png     per-pixel softmax entropy (model uncertainty)
  talc.json       fraction (raw + calibrated), area px/µm², flags, grid
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .quantify import Calib, soft_fraction


def area_matched_threshold(talc_prob: np.ndarray, valid: np.ndarray,
                           target_fraction: float) -> float:
    """Threshold on talc_prob whose binary area over `valid` equals the
    calibrated fraction — so the drawn mask never contradicts the reported %."""
    vals = talc_prob[valid]
    if vals.size == 0 or target_fraction <= 0:
        return 1.01
    q = 1.0 - float(np.clip(target_fraction, 0.0, 1.0))
    return float(np.quantile(vals, q))


def binary_mask(talc_prob: np.ndarray, valid: np.ndarray, thr: float) -> np.ndarray:
    m = (talc_prob >= thr) & valid
    return (m.astype(np.uint8) * 255)


def render_overlay(image_bgr: np.ndarray, mask255: np.ndarray, cfg: Config) -> np.ndarray:
    out = image_bgr.copy()
    sel = mask255 > 0
    color = np.array(cfg.report.talc_bgr, np.float32)
    a = cfg.report.overlay_alpha
    out[sel] = ((1 - a) * out[sel] + a * color).astype(np.uint8)
    return out


def local_fraction_grid(talc_prob: np.ndarray, valid: np.ndarray, cells: int) -> list[list[float]]:
    """cells×cells map of talc fraction per block (−1 where a block is all resin)."""
    h, w = talc_prob.shape
    gh, gw = max(1, h // cells), max(1, w // cells)
    grid: list[list[float]] = []
    for i in range(cells):
        row: list[float] = []
        for j in range(cells):
            y0, x0 = i * gh, j * gw
            y1 = h if i == cells - 1 else (i + 1) * gh
            x1 = w if j == cells - 1 else (j + 1) * gw
            v = valid[y0:y1, x0:x1]
            n = int(v.sum())
            row.append(round(float(talc_prob[y0:y1, x0:x1][v].sum()) / n, 4)
                       if n > 0 else -1.0)
        grid.append(row)
    return grid


def heatmap_image(grid: list[list[float]], h: int, w: int) -> np.ndarray:
    g = np.array(grid, np.float32)
    g[g < 0] = 0
    g = np.clip(g / max(g.max(), 1e-6), 0, 1)
    small = (g * 255).astype(np.uint8)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(big, cv2.COLORMAP_INFERNO)


def entropy_image(entropy: np.ndarray) -> np.ndarray:
    e = entropy / max(float(entropy.max()), 1e-6)
    return cv2.applyColorMap((e * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)


def write_outputs(out_dir: Path, image_bgr: np.ndarray, result, calib: Calib,
                  cfg: Config, um_per_px: float | None = None) -> dict:
    """Compute the calibrated fraction, render all artifacts, write talc.json.
    `result` is an inference.InferResult. Returns the record dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    talc_prob, valid = result.talc_prob, result.valid

    raw = soft_fraction(talc_prob, valid, calib.temperature)
    final = calib.apply(raw)

    thr = area_matched_threshold(talc_prob, valid, final)
    mask255 = binary_mask(talc_prob, valid, thr)
    grid = local_fraction_grid(talc_prob, valid, cfg.report.grid_cells)

    cv2.imwrite(str(out_dir / "talc_mask.png"), mask255)
    cv2.imwrite(str(out_dir / "overlay.png"),
                render_overlay(image_bgr, mask255, cfg))
    cv2.imwrite(str(out_dir / "heatmap.png"),
                heatmap_image(grid, *talc_prob.shape))
    cv2.imwrite(str(out_dir / "entropy.png"), entropy_image(result.entropy))

    n_valid = int(valid.sum())
    band = cfg.calibrate.flag_band
    high_entropy_frac = float((result.entropy > 1.0).mean())
    record = {
        "width": int(image_bgr.shape[1]),
        "height": int(image_bgr.shape[0]),
        "valid_px": n_valid,
        "talc_fraction_raw": round(raw, 5),
        "talc_fraction_final": round(final, 5),
        "calibration": {"method": calib.method, "temperature": calib.temperature,
                        "slope": calib.slope, "intercept": calib.intercept},
        "talc_area_px": int(mask255.sum() // 255),
        "talc_area_um2": (round((mask255.sum() // 255) * (um_per_px ** 2), 2)
                          if um_per_px else None),
        "is_talcose": bool(final > cfg.infer.talc_line),
        "needs_review": bool(band[0] <= final <= band[1] or high_entropy_frac > 0.30),
        "high_entropy_frac": round(high_entropy_frac, 4),
        "local_fraction_grid": grid,
    }
    (out_dir / "talc.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record
