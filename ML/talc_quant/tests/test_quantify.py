"""Quantification: soft fraction, Theil–Sen recovery, calibration apply, stats."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talc_quant.quantify import (Calib, fit_calibration, fraction_stats,  # noqa: E402
                                 soft_fraction, theil_sen)


def test_soft_fraction_basic():
    prob = np.full((10, 10), 0.3, np.float32)
    valid = np.ones((10, 10), bool)
    assert abs(soft_fraction(prob, valid) - 0.3) < 1e-6


def test_soft_fraction_respects_valid():
    prob = np.zeros((10, 10), np.float32)
    prob[:5] = 1.0
    valid = np.ones((10, 10), bool)
    valid[:5] = False               # exclude the talc half
    assert abs(soft_fraction(prob, valid) - 0.0) < 1e-6


def test_theil_sen_recovers_line():
    rng = np.random.default_rng(0)
    pred = np.linspace(0.0, 0.5, 40)
    gt = 1.3 * pred + 0.02 + rng.normal(0, 1e-4, pred.shape)
    s, b = theil_sen(pred, gt)
    assert abs(s - 1.3) < 0.05
    assert abs(b - 0.02) < 0.01


def test_theil_sen_robust_to_outliers():
    pred = np.linspace(0.0, 0.5, 40)
    gt = 1.0 * pred + 0.0
    gt[5] += 0.5                    # one gross outlier
    gt[20] -= 0.4
    s, b = theil_sen(pred, gt)
    assert abs(s - 1.0) < 0.1       # median slope shrugs off outliers


def test_calibration_apply_clips():
    c = Calib(temperature=1.0, slope=2.0, intercept=0.0, method="theilsen")
    assert c.apply(0.8) == 1.0      # clipped to [0,1]
    assert c.apply(0.1) == 0.2


def test_fit_calibration_theilsen_reduces_bias():
    pred = np.linspace(0.05, 0.45, 30)
    gt = 1.25 * pred + 0.03         # model under-predicts
    calib = fit_calibration(pred, gt, method="theilsen")
    corrected = np.array([calib.apply(p) for p in pred])
    assert np.abs(corrected - gt).mean() < np.abs(pred - gt).mean()


def test_calib_roundtrip(tmp_path):
    c = Calib(1.2, 1.1, -0.01, "theilsen")
    p = tmp_path / "c.json"
    c.save(p)
    assert Calib.load(p) == c


def test_fraction_stats():
    errs = np.array([0.01, -0.02, 0.03, -0.005])
    s = fraction_stats(errs)
    assert s["n"] == 4
    assert abs(s["mae"] - np.abs(errs).mean()) < 1e-9
    assert s["max"] == 0.03
