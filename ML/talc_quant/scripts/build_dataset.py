"""Build the canonical-profile training corpus (labels + FDA bank + folds).

    python scripts/build_dataset.py
    python scripts/build_dataset.py data.neg_per_folder=20   # smaller negatives

Reads: first_labling_attempt/yolo_seg (talc polygons), part1/part2 (negatives),
panoramas (FDA bank). Writes: talc_quant/data_build/.
"""
from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

from talc_quant.config import load_config
from talc_quant.gt_build import build_dataset
from talc_quant.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(overrides=args.overrides)
    set_seed(cfg.train.seed)
    records = build_dataset(cfg)
    n_talc = sum(r.has_talc for r in records)
    print(f"done: {len(records)} items ({n_talc} talc, {len(records) - n_talc} neg) "
          f"-> {cfg.paths.build_dir}")


if __name__ == "__main__":
    main()
