"""False-positive control on non-talc sorts (spec §3.4).

The ±3% target is meaningless if the model paints talc on ordinary/refractory
ores. This predicts the talc fraction on the negative images in the corpus (their
folds' own models, OOF) and reports the distribution — it should sit near 0 with
P95 ≤ 3%.

    python scripts/fp_control.py --track segformer_b2
"""
from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from talc_quant.config import load_config
from talc_quant.dataset import items_for_folds, load_manifest
from talc_quant.engine import resolve_device
from talc_quant.inference import load_models, predict
from talc_quant.quantify import Calib, soft_fraction


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
    run_dir = cfg.paths.runs_dir / cfg.model.track
    calib_path = run_dir / "calibration.json"
    calib = Calib.load(calib_path) if calib_path.exists() else Calib()

    manifest = load_manifest(cfg.paths.build_dir)
    fracs_by_sort: dict[str, list[float]] = {}
    for f in range(cfg.train.folds):
        ckpt = run_dir / f"fold{f}_best.pt"
        if not ckpt.exists():
            continue
        models = load_models([ckpt], device)
        negs = [r for r in items_for_folds(manifest, {f}) if not r["has_talc"]]
        for rec in negs:
            bgr = cv2.imread(str(cfg.paths.build_dir / rec["image"]))
            if bgr is None:
                continue
            res = predict(bgr, models, cfg, device, is_panorama=False,
                          preprocess=False)
            raw = soft_fraction(res.talc_prob, res.valid, calib.temperature)
            fracs_by_sort.setdefault(rec["sort"], []).append(calib.apply(raw))

    print(f"\n=== {cfg.model.track}: talc fraction on NON-talc sorts (should ~0) ===")
    all_fr: list[float] = []
    for sort, fr in sorted(fracs_by_sort.items()):
        a = np.array(fr)
        all_fr += fr
        print(f"  {sort:10s} n={len(a):3d}  median {np.median(a)*100:5.2f}%  "
              f"mean {a.mean()*100:5.2f}%  P95 {np.percentile(a,95)*100:5.2f}%  "
              f"max {a.max()*100:5.2f}%")
    if all_fr:
        a = np.array(all_fr)
        p95 = float(np.percentile(a, 95))
        verdict = "PASS" if p95 <= 0.03 else "FAIL"
        print(f"  {'OVERALL':10s} n={len(a):3d}  P95 {p95*100:5.2f}%  -> {verdict}")


if __name__ == "__main__":
    main()
