"""Evaluate a trained YOLO-seg checkpoint on the held-out val split.

Reports mask (and box) mAP, and can dump predicted-mask overlays for eyeballing.

    python evaluate.py                                   # uses runs/talc_seg/weights/best.pt
    python evaluate.py --weights <path> --predict        # + save overlays
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from train import HERE, YOLO_SEG, write_data_yaml

DEFAULT_WEIGHTS = HERE / "runs" / "talc_seg" / "weights" / "best.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu", help="'cpu' or '0' for GPU")
    ap.add_argument("--predict", action="store_true", help="save predicted-mask overlays")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"No weights at {args.weights}. Train first: python train.py --smoke")

    data = write_data_yaml()
    model = YOLO(args.weights)

    metrics = model.val(
        data=str(data),
        imgsz=args.imgsz,
        device=args.device,
        workers=0,
        project=str(HERE / "runs"),
        name="val",
        exist_ok=True,
    )

    # results_dict is stable across ultralytics versions; keys include mask + box mAP
    rd = metrics.results_dict
    print("\n=== VAL metrics (talc, 8 images) ===")
    for k, v in rd.items():
        print(f"  {k:<28} {v:.4f}")

    if args.predict:
        val_imgs = (YOLO_SEG / "images" / "val").resolve()
        model.predict(
            source=str(val_imgs),
            save=True,
            imgsz=args.imgsz,
            device=args.device,
            project=str(HERE / "runs"),
            name="predict",
            exist_ok=True,
        )
        print(f"overlays saved under {HERE / 'runs' / 'predict'}")


if __name__ == "__main__":
    main()
