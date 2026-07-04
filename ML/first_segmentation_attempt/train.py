"""Fine-tune Ultralytics YOLO-seg on the hand-labelled talc dataset.

Why YOLO-seg: our labels are already native YOLO-seg polygons
(../first_labling_attempt/yolo_seg), so there's zero conversion, we get
COCO-pretrained transfer (essential with only 42 images), and mask mAP for free.

Model choice:
  * local test / CPU / MPS : yolo11n-seg  (nano, fast, default)
  * remote GPU (V100)      : yolo11l-seg  (bigger backbone, better masks)

    python train.py --smoke                              # quick CPU sanity run
    python train.py                                      # local, yolo11n-seg
    python train.py --model yolo11l-seg.pt --epochs 200 --imgsz 768 --device 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
YOLO_SEG = (HERE.parent / "first_labling_attempt" / "yolo_seg").resolve()
CLASS_NAMES = {0: "talc"}


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
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="None=auto, 'cpu', or '0' for GPU")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default="talc_seg")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU run to verify the pipeline (3 epochs, imgsz 320)")
    args = ap.parse_args()

    if args.smoke:
        args.model, args.epochs, args.imgsz = "yolo11n-seg.pt", 3, 320
        args.batch, args.device, args.workers = 2, "cpu", 0

    data = write_data_yaml()
    print(f"dataset: {YOLO_SEG} | model: {args.model} | device: {args.device or 'auto'}")

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(HERE / "runs"),
        name=args.name,
        exist_ok=True,
        seed=42,
        plots=not args.smoke,
    )
    best = HERE / "runs" / args.name / "weights" / "best.pt"
    print(f"done. best weights: {best}")


if __name__ == "__main__":
    main()
