"""Grouped k-fold cross-validation over the full 42-image talc set.

The stock val split is 8 images, so single-split mask mAP moves ~±12% when one
image flips — too noisy to compare training recipes. K-fold puts every image in
val exactly once and reports mean ± std across folds.

Grouping: `2550374-2_10.jpg` etc. are multiple shots of the same drill specimen
(the 7-digit prefix); those are kept in the same fold so the model is never
validated on a specimen it trained on. DSCN* files are treated as singletons —
if you know some are shots of the same sample, extend group_key().

    python kfold.py                                      # yolo11n-seg, local
    python kfold.py --model yolo11l-seg.pt --imgsz 768 --device 0   # remote GPU
"""
from __future__ import annotations

import argparse
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import yaml
from ultralytics import YOLO

from train import HERE, SMALL_DATA_OVERRIDES, YOLO_SEG, CLASS_NAMES

KFOLD_DIR = HERE / "runs" / "kfold"
REPORT_KEYS = [  # results_dict keys worth aggregating
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]


def group_key(img: Path) -> str:
    """Same drill specimen -> same fold (no train/val leakage within a fold)."""
    m = re.match(r"(\d{7})-", img.stem)
    return m.group(1) if m else img.stem


def make_folds(k: int, seed: int = 42) -> list[list[Path]]:
    imgs = sorted(p for p in (YOLO_SEG / "images").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imgs:
        raise SystemExit(f"No images under {YOLO_SEG / 'images'} — regenerate the dataset first.")
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in imgs:
        groups[group_key(p)].append(p)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    folds: list[list[Path]] = [[] for _ in range(k)]
    for i, key in enumerate(keys):  # round-robin keeps fold sizes balanced
        folds[i % k].extend(groups[key])
    return folds


def write_fold_yaml(i: int, folds: list[list[Path]]) -> Path:
    """Emit train/val image lists + data yaml for fold i (absolute paths)."""
    KFOLD_DIR.mkdir(parents=True, exist_ok=True)
    val = folds[i]
    train = [p for j, f in enumerate(folds) if j != i for p in f]
    train_txt = KFOLD_DIR / f"fold{i}_train.txt"
    val_txt = KFOLD_DIR / f"fold{i}_val.txt"
    train_txt.write_text("\n".join(str(p) for p in train) + "\n")
    val_txt.write_text("\n".join(str(p) for p in val) + "\n")
    data = {"path": str(YOLO_SEG), "train": str(train_txt), "val": str(val_txt), "names": CLASS_NAMES}
    out = KFOLD_DIR / f"fold{i}_data.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    folds = make_folds(args.folds)
    for i, f in enumerate(folds):
        print(f"fold {i}: {len(f)} val images")

    per_fold: list[dict[str, float]] = []
    for i in range(args.folds):
        data = write_fold_yaml(i, folds)
        print(f"\n=== fold {i}/{args.folds - 1} ===")
        model = YOLO(args.model)
        model.train(
            data=str(data),
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(HERE / "runs"),
            name=f"kfold_{i}",
            exist_ok=True,
            seed=42,
            plots=False,
            **SMALL_DATA_OVERRIDES,
        )
        best = YOLO(HERE / "runs" / f"kfold_{i}" / "weights" / "best.pt")
        metrics = best.val(
            data=str(data),
            imgsz=args.imgsz,
            device=args.device,
            workers=0,
            project=str(HERE / "runs"),
            name=f"kfold_{i}_val",
            exist_ok=True,
        )
        per_fold.append({k: float(v) for k, v in metrics.results_dict.items()})

    print(f"\n=== {args.folds}-fold CV summary ({args.model}, imgsz {args.imgsz}) ===")
    for key in REPORT_KEYS:
        vals = [m[key] for m in per_fold]
        print(f"  {key:<24} {statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}   folds: "
              + " ".join(f"{v:.3f}" for v in vals))


if __name__ == "__main__":
    main()
