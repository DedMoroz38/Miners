"""Fine-tune Ultralytics YOLO-seg on the hand-labelled talc dataset.

Why YOLO-seg: our labels are already native YOLO-seg polygons
(../first_labling_attempt/yolo_seg), so there's zero conversion, we get
COCO-pretrained transfer (essential with only 42 images), and mask mAP for free.

Model choice:
  * local test / CPU / MPS : yolo11n-seg  (nano, fast, default)
  * remote GPU (V100)      : yolo11l-seg  (bigger backbone, better masks)

Small-data recipe: the first 100-epoch run overfit hard (train loss kept
falling while val loss rose; mask mAP50 plateaued ~0.30 by epoch 60). With 34
train images the lever is augmentation + early stopping, not capacity — see
SMALL_DATA_OVERRIDES below.

    python train.py --smoke                              # quick CPU sanity run
    python train.py                                      # local, yolo11n-seg
    python train.py --model yolo11l-seg.pt --imgsz 768 --device 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
YOLO_SEG = (HERE.parent / "first_labling_attempt" / "yolo_seg").resolve()
CLASS_NAMES = {0: "talc"}

# Anti-overfit settings for the 42-image dataset. Micrographs have no canonical
# orientation, so rotation + vertical flip are "free" extra data.
# First attempt (degrees=180 + copy_paste=0.5) was too aggressive for 34 train
# images: val metrics whipsawed and early-stop latched onto a degenerate epoch-8
# "best" (conf never above 0.03). Keep it gentle.
# Shared with kfold.py so CV measures exactly what train.py trains.
SMALL_DATA_OVERRIDES = dict(
    degrees=90.0,     # gentle rotation (180 destabilized training)
    flipud=0.5,       # same reasoning as fliplr (already 0.5 by default)
    cos_lr=True,      # smooth decay to lrf instead of step plateaus
    close_mosaic=30,  # last N epochs on clean (un-mosaicked) images
)


def write_data_yaml() -> Path:
    """Write a portable data.yaml pointing at the absolute yolo_seg dir.

    Regenerated each run so the committed (machine-specific) absolute path never
    bites us on another host.
    """
    if not (YOLO_SEG / "images" / "train").is_dir():
        raise SystemExit(
            f"Dataset not found at {YOLO_SEG}.\n"
            "Regenerate it: cd ../first_labling_attempt && "
            "python export_to_yoloseg.py --from-api"
        )
    data = {
        "path": str(YOLO_SEG),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }
    out = HERE / "data.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n-seg.pt", help="pretrained seg checkpoint")
    ap.add_argument("--epochs", type=int, default=300,
                    help="upper bound; patience stops the run once val mAP stalls")
    ap.add_argument("--patience", type=int, default=80,
                    help="early-stop after this many epochs without val improvement")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="None=auto, 'cpu', or '0' for GPU")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--freeze", type=int, default=0,
                    help="freeze first N layers (11 = whole backbone) to cut capacity")
    ap.add_argument("--name", default=None,
                    help="run dir under runs/; defaults per recipe: talc_seg (baseline), "
                         "talc_seg_v2 (small-data), smoke (--smoke) — so runs never "
                         "overwrite each other's best.pt")
    ap.add_argument("--baseline", action="store_true",
                    help="reproduce the original run: stock Ultralytics augmentation "
                         "(no SMALL_DATA_OVERRIDES), 100 epochs, no early stop")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU run to verify the pipeline (3 epochs, imgsz 320)")
    args = ap.parse_args()

    if args.smoke:
        args.model, args.epochs, args.imgsz = "yolo11n-seg.pt", 3, 320
        args.batch, args.device, args.workers = 2, "cpu", 0

    # --baseline reproduces the pre-augmentation reference run (mask mAP50 ≈ 0.30);
    # everything else stays on the small-data recipe so runs are comparable.
    overrides = {} if args.baseline else SMALL_DATA_OVERRIDES
    if args.baseline and args.epochs == ap.get_default("epochs"):
        args.epochs, args.patience = 100, 100  # match the original: full 100 epochs, no early stop

    # Per-recipe run names: a smoke test or a v2 run must never clobber
    # runs/talc_seg/weights/best.pt (that accident already cost us one baseline).
    if args.name is None:
        args.name = "smoke" if args.smoke else ("talc_seg" if args.baseline else "talc_seg_v2")

    data = write_data_yaml()
    recipe = "baseline (stock aug)" if args.baseline else "small-data (augmented)"
    print(f"dataset: {YOLO_SEG} | model: {args.model} | device: {args.device or 'auto'} | recipe: {recipe}")

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        freeze=args.freeze or None,
        project=str(HERE / "runs"),
        name=args.name,
        exist_ok=True,
        seed=42,
        plots=not args.smoke,
        **overrides,
    )
    best = HERE / "runs" / args.name / "weights" / "best.pt"
    print(f"done. best weights: {best}")


if __name__ == "__main__":
    main()
