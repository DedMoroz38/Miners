"""Train one or all folds for a track.

    # V100, SegFormer-B2, all 5 folds:
    python scripts/train.py --track segformer_b2 --fold all model.pretrained=true

    # single fold, SegFormer-B3 at 768:
    python scripts/train.py --track segformer_b3 --fold 0 data.crop=768 train.batch=8

    # track B (frozen DINOv2, cheap):
    python scripts/train.py --track dino_vitb14 --fold all

Checkpoints -> runs/<track>/foldN_best.pt. Run build_dataset.py first.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

from talc_quant.config import load_config
from talc_quant.engine import train_fold
from talc_quant.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--track", default=None, help="override model.track")
    ap.add_argument("--fold", default="all", help="'all' or an int fold index")
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    overrides = list(args.overrides)
    if args.track:
        overrides.append(f"model.track={args.track}")
    cfg = load_config(overrides=overrides)
    set_seed(cfg.train.seed)

    out_dir = cfg.paths.runs_dir / cfg.model.track
    folds = range(cfg.train.folds) if args.fold == "all" else [int(args.fold)]
    for f in folds:
        set_seed(cfg.train.seed + f)
        ckpt = train_fold(cfg, f, out_dir)
        print(f"fold {f} -> {ckpt}")


if __name__ == "__main__":
    main()
