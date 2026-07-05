"""Out-of-fold calibration + the honest acceptance table (spec §3, §8).

For each fold f: load fold f's checkpoint, predict the talc fraction on fold f's
VAL talc images (images that model never trained on), pair with GT. Pool the OOF
pairs, fit a ≤3-param calibrator, and report MAE / median / P90 / max / bias
before and after calibration — per fold and pooled. Writes calibration.json.

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
import torch

from talc_quant.config import load_config
from talc_quant.constants import CLASS_TALC, VALID_CLASSES
from talc_quant.dataset import items_for_folds, load_manifest
from talc_quant.engine import resolve_device
from talc_quant.inference import load_models, predict
from talc_quant.quantify import (Calib, fit_calibration, fraction_stats,
                                 soft_fraction)


def _gt_fraction(build: Path, rec: dict) -> float:
    lab = cv2.imread(str(build / rec["label"]), cv2.IMREAD_GRAYSCALE)
    valid = np.isin(lab, VALID_CLASSES)
    return float((lab == CLASS_TALC).sum()) / max(int(valid.sum()), 1)


def _oof_predictions(cfg, device, temperature: float):
    """Return (preds, gts, folds) for every val talc image via its own fold model."""
    manifest = load_manifest(cfg.paths.build_dir)
    run_dir = cfg.paths.runs_dir / cfg.model.track
    preds, gts, fold_ids = [], [], []
    for f in range(cfg.train.folds):
        ckpt = run_dir / f"fold{f}_best.pt"
        if not ckpt.exists():
            logging.warning("missing %s, skipping fold %d", ckpt, f)
            continue
        models = load_models([ckpt], device)
        val_items = [r for r in items_for_folds(manifest, {f}) if r["has_talc"]]
        for rec in val_items:
            bgr = cv2.imread(str(cfg.paths.build_dir / rec["image"]))
            if bgr is None:
                continue
            res = predict(bgr, models, cfg, device, is_panorama=False,
                          preprocess=False)          # built img already canonical
            raw = soft_fraction(res.talc_prob, res.valid, temperature)
            preds.append(raw)
            gts.append(_gt_fraction(cfg.paths.build_dir, rec))
            fold_ids.append(f)
    return np.array(preds), np.array(gts), np.array(fold_ids)


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
    preds, gts, folds = _oof_predictions(cfg, device, temp)
    if len(preds) == 0:
        raise SystemExit("no OOF predictions — train folds first")

    calib = fit_calibration(preds, gts, cfg.calibrate.method, temp)
    corrected = np.array([calib.apply(p) for p in preds])

    print(f"\n=== {cfg.model.track}: OOF talc-fraction error "
          f"(calibration: {calib.method}, T={temp}) ===")
    print("RAW (uncalibrated):")
    _print_table("pooled", preds - gts)
    print("CALIBRATED:")
    _print_table("pooled", corrected - gts)
    for f in sorted(set(folds.tolist())):
        m = folds == f
        _print_table(f"fold{f}", corrected[m] - gts[m])

    out = cfg.paths.runs_dir / cfg.model.track / "calibration.json"
    calib.save(out)
    s = fraction_stats(corrected - gts)
    verdict = "PASS" if s["mae"] <= 0.03 and abs(s["bias"]) <= 0.005 else "FAIL"
    print(f"\ncalibration -> {out}")
    print(f"ACCEPTANCE (MAE<=3%%, |bias|<=0.5%%): {verdict} "
          f"(MAE {s['mae']*100:.2f}%, bias {s['bias']*100:+.2f}%)")


if __name__ == "__main__":
    main()
