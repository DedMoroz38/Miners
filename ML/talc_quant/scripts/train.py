"""Train a track on the TRAIN split, early-stopping on the TEST split.

    # V100, SegFormer-B2:
    python scripts/train.py --track segformer_b2 model.pretrained=true

    # SegFormer-B3 at 768:
    python scripts/train.py --track segformer_b3 data.crop=768 train.batch=8

    # track B (frozen DINOv2, cheap):
    python scripts/train.py --track dino_vitb14

Checkpoint -> runs/<track>/best.pt. Run build_dataset.py first.
"""
from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

from talc_quant.config import load_config
from talc_quant.engine import train_model
from talc_quant.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--track", default=None, help="override model.track")
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    overrides = list(args.overrides)
    if args.track:
        overrides.append(f"model.track={args.track}")
    cfg = load_config(overrides=overrides)
    set_seed(cfg.train.seed)

    out_dir = cfg.paths.runs_dir / cfg.model.track
    ckpt = train_model(cfg, out_dir)
    print(f"best checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
