"""Calibration + the acceptance table on the held-out TEST split (spec §3, §8).

Loads the trained checkpoint, predicts the talc fraction on every TEST talc image
(images the model never trained on), pairs with GT, fits a ≤3-param calibrator
(temperature + robust Theil–Sen), and reports MAE / median / P90 / max / bias
before and after calibration. Writes calibration.json.

Note: with a single train/test split the calibrator is fit and reported on the
same held-out set, so the ≤3-param fit is mildly optimistic — kept minimal by
design. Add a val split later for a fully independent calibration.

    python scripts/calibrate.py --track segformer_b2
    python scripts/calibrate.py --track segformer_b2 calibrate.method=adjusted_count
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from talc_quant.config import load_config
from talc_quant.constants import CLASS_TALC, VALID_CLASSES
from talc_quant.dataset import items_for_split, load_manifest
from talc_quant.engine import resolve_device
from talc_quant.inference import load_models, predict
from talc_quant.quantify import (Calib, fit_calibration, fraction_stats,
                                 soft_fraction)


def _gt_fraction(build: Path, rec: dict) -> float:
    lab = cv2.imread(str(build / rec["label"]), cv2.IMREAD_GRAYSCALE)
    valid = np.isin(lab, VALID_CLASSES)
    return float((lab == CLASS_TALC).sum()) / max(int(valid.sum()), 1)


def _test_predictions(cfg, device, temperature: float):
    """Return (preds, gts) for every TEST talc image via the trained model."""
    manifest = load_manifest(cfg.paths.build_dir)
    ckpt = cfg.paths.runs_dir / cfg.model.track / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt} — train first")
    models = load_models([ckpt], device)
    test_items = [r for r in items_for_split(manifest, "test") if r["has_talc"]]
    preds, gts = [], []
    for rec in test_items:
        bgr = cv2.imread(str(cfg.paths.build_dir / rec["image"]))
        if bgr is None:
            continue
        res = predict(bgr, models, cfg, device, is_panorama=False,
                      preprocess=False)              # built img already canonical
        preds.append(soft_fraction(res.talc_prob, res.valid, temperature))
        gts.append(_gt_fraction(cfg.paths.build_dir, rec))
    return np.array(preds), np.array(gts)


def _print_table(name: str, errs: np.ndarray) -> None:
    s = fraction_stats(errs)
    print(f"  {name:16s} MAE {s['mae']*100:5.2f}  med {s['median']*100:5.2f}  "
          f"P90 {s['p90']*100:5.2f}  max {s['max']*100:5.2f}  "
          f"bias {s['bias']*100:+5.2f}  n {s['n']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--track", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    overrides = list(args.overrides)
    if args.track:
        overrides.append(f"model.track={args.track}")
    cfg = load_config(overrides=overrides)
    device = resolve_device(cfg.train.device)

    temp = cfg.calibrate.temperature_init
    preds, gts = _test_predictions(cfg, device, temp)
    if len(preds) == 0:
        raise SystemExit("no TEST talc predictions — check the split / train first")

    calib = fit_calibration(preds, gts, cfg.calibrate.method, temp)
    corrected = np.array([calib.apply(p) for p in preds])

    print(f"\n=== {cfg.model.track}: TEST talc-fraction error "
          f"(calibration: {calib.method}, T={temp}) ===")
    print("RAW (uncalibrated):")
    _print_table("test", preds - gts)
    print("CALIBRATED:")
    _print_table("test", corrected - gts)

    out = cfg.paths.runs_dir / cfg.model.track / "calibration.json"
    calib.save(out)
    s = fraction_stats(corrected - gts)
    verdict = "PASS" if s["mae"] <= 0.03 and abs(s["bias"]) <= 0.005 else "FAIL"
    print(f"\ncalibration -> {out}")
    print(f"ACCEPTANCE (MAE<=3%%, |bias|<=0.5%%): {verdict} "
          f"(MAE {s['mae']*100:.2f}%, bias {s['bias']*100:+.2f}%)")


if __name__ == "__main__":
    main()
