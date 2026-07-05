"""Run the talc quantifier on one image (field or panorama) -> artifacts.

    python scripts/infer.py --image /abs/panorama.jpg --out job/talc --track segformer_b2
    python scripts/infer.py --image /abs/field.jpg --out job/talc --um-per-px 0.5

Loads every foldN_best.pt for the track (ensemble) + calibration.json (if present)
and writes talc_mask.png, overlay.png, heatmap.png, entropy.png, talc.json.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2

from talc_quant.config import load_config
from talc_quant.engine import resolve_device
from talc_quant.inference import load_models, predict
from talc_quant.quantify import Calib
from talc_quant.reporting import write_outputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--track", default=None)
    ap.add_argument("--um-per-px", type=float, default=None)
    ap.add_argument("--panorama", action="store_true", help="force tiled path")
    ap.add_argument("--field", action="store_true", help="force single-pass path")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    overrides = list(args.overrides)
    if args.track:
        overrides.append(f"model.track={args.track}")
    cfg = load_config(overrides=overrides)
    device = resolve_device(cfg.train.device)

    run_dir = cfg.paths.runs_dir / cfg.model.track
    ckpts = sorted(run_dir.glob("fold*_best.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints in {run_dir}; train first")
    models = load_models(ckpts, device)

    calib_path = run_dir / "calibration.json"
    calib = Calib.load(calib_path) if calib_path.exists() else Calib()
    if not calib_path.exists():
        logging.warning("no calibration.json — using identity calibration")

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")
    is_pano = True if args.panorama else (False if args.field else None)
    res = predict(bgr, models, cfg, device, is_panorama=is_pano, preprocess=True)
    rec = write_outputs(args.out, bgr, res, calib, cfg, um_per_px=args.um_per_px)

    print(f"{Path(args.image).name}: talc {rec['talc_fraction_raw']*100:.1f}% -> "
          f"{rec['talc_fraction_final']*100:.1f}% (calibrated) | "
          f"talcose={rec['is_talcose']} review={rec['needs_review']} -> {args.out}")


if __name__ == "__main__":
    main()
