#!/usr/bin/env python3
"""Populate yolo_seg/images/{train,val} from a raw image folder using the labels.

The YOLO-seg labels (yolo_seg/labels/{train,val}/*.txt) and filename_map.json
travel through git; the images do not. This script re-materializes the image
half of the dataset from a raw source folder (e.g. data/.../talc_altered_ores)
so build_dataset.py can run — no Label Studio / re-export needed.

For every label file it:
  * takes the stem (e.g. 2550374-2_10),
  * finds the sanitized image name in filename_map.json (stem + .jpg),
  * looks up the ORIGINAL basename it maps to (e.g. "2550374-2 10х.JPG"),
  * finds that basename in --src (top level only, skips nested dirs),
  * copies it to yolo_seg/images/<split>/<sanitized_name>.

Usage:
    python populate_images.py --src ../../data/ore_photos_by_grade_part1/talc_altered_ores
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
YOLO = HERE / "yolo_seg"
SPLITS = ("train", "val")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="folder with the raw original images")
    ap.add_argument("--yolo", default=str(YOLO), help="yolo_seg dataset dir")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    yolo = Path(args.yolo).resolve()

    fmap_path = yolo / "filename_map.json"
    fmap = json.loads(fmap_path.read_text())  # sanitized_name -> "to_label/ORIG.JPG"

    # Index the source folder by original basename (top level only, to avoid
    # picking a duplicate from a nested subfolder like talcification_zones/).
    by_name = {p.name: p for p in src.iterdir() if p.is_file()}

    total, copied, missing = 0, 0, []
    for split in SPLITS:
        lbl_dir = yolo / "labels" / split
        img_dir = yolo / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        for lbl in sorted(lbl_dir.glob("*.txt")):
            total += 1
            sanitized = lbl.stem + ".jpg"  # export always wrote .jpg
            orig_rel = fmap.get(sanitized)
            if orig_rel is None:
                missing.append((split, lbl.name, "no filename_map entry"))
                continue
            orig_name = os.path.basename(orig_rel)  # "2550374-2 10х.JPG"
            src_file = by_name.get(orig_name)
            if src_file is None:
                missing.append((split, lbl.name, f"source image not found: {orig_name}"))
                continue
            shutil.copy2(src_file, img_dir / sanitized)
            copied += 1

    print(f"labels: {total} | copied images: {copied} | missing: {len(missing)}")
    for split, name, why in missing:
        print(f"  MISSING [{split}] {name}: {why}")
    if missing:
        raise SystemExit(1)
    print(f"done -> {yolo / 'images'}")


if __name__ == "__main__":
    main()
