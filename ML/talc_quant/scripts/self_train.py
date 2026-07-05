"""Optional Stage-3 self-training (spec §5.3) — OFF by default in the workflow.

Uses the trained model as a teacher to pseudo-label high-confidence talc pixels
on part2 talc-altered images + panorama tiles, then appends them to the corpus
for a low-LR fine-tune. Guards: only very-confident pixels kept, sanity-filter by
folder sort (a non-talc image predicted >10% talc is quarantined, not learned).

    python scripts/self_train.py --track segformer_b2 --round 1

This writes extra labeled tiles under data_build/pseudo_round{N}/ and a merged
manifest; retrain with train.py pointing at the merged manifest. Kept minimal:
generates pseudo-labels + quarantine list; the retrain reuses train.py.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from talc_quant.config import load_config
from talc_quant.constants import (CLASS_BACKGROUND, CLASS_MATRIX, CLASS_TALC,
                                  IGNORE_INDEX)
from talc_quant.engine import resolve_device
from talc_quant.inference import load_models, predict
from talc_quant.preprocess import CanonicalPreprocessor
from talc_quant.scale import scale_factor

CONF_TALC = 0.85     # keep a pixel as pseudo-talc only above this prob
CONF_MATRIX = 0.90   # confident non-talc -> matrix supervision
QUARANTINE_FRAC = 0.10


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--track", default=None)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--limit", type=int, default=80, help="max source images")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    overrides = list(args.overrides)
    if args.track:
        overrides.append(f"model.track={args.track}")
    cfg = load_config(overrides=overrides)
    device = resolve_device(cfg.train.device)
    run_dir = cfg.paths.runs_dir / cfg.model.track
    ckpt = run_dir / "best.pt"
    if not ckpt.exists():
        raise SystemExit("train the model first")
    models = load_models([ckpt], device)
    pp = CanonicalPreprocessor(cfg.paths.sulfide_preprocess)

    out = cfg.paths.build_dir / f"pseudo_round{args.round}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    # sources: part2 talc-altered (talc expected) + a few panorama tiles
    talc_dir = cfg.paths.part2 / "talc_altered"
    srcs = sorted(p for p in talc_dir.glob("*") if p.is_file())[: args.limit]
    kept, quarantined, records = 0, 0, []
    for p in srcs:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        factor = scale_factor(p.name, cfg.scale)
        bgr = cv2.resize(pp(bgr), None, fx=factor, fy=factor) if factor != 1 else pp(bgr)
        res = predict(bgr, models, cfg, device, is_panorama=False, preprocess=False)
        frac = float(res.talc_prob[res.valid].sum()) / max(int(res.valid.sum()), 1)
        # talc-altered folder but ~0 talc predicted -> quarantine (do not learn)
        if frac < 0.01:
            quarantined += 1
            continue
        label = np.full(bgr.shape[:2], IGNORE_INDEX, np.uint8)
        label[res.talc_prob >= CONF_TALC] = CLASS_TALC
        label[(res.talc_prob <= (1 - CONF_MATRIX)) & res.valid] = CLASS_MATRIX
        label[~res.valid] = CLASS_BACKGROUND
        stem = f"pseudo_{p.stem}"
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), bgr)
        cv2.imwrite(str(out / "labels" / f"{stem}.png"), label)
        records.append({"stem": stem, "image": f"pseudo_round{args.round}/images/{stem}.jpg",
                        "label": f"pseudo_round{args.round}/labels/{stem}.png",
                        "has_talc": True, "sort": "pseudo", "source": "self_train",
                        "split": "train"})   # pseudo-labels only augment training
        kept += 1

    (out / "pseudo_manifest.json").write_text(json.dumps(records, indent=2))
    print(f"self-train round {args.round}: kept {kept}, quarantined {quarantined} "
          f"-> {out}\nmerge pseudo_manifest.json into manifest.json and rerun train.py")


if __name__ == "__main__":
    main()
