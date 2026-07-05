"""Evaluate a trained YOLO-seg checkpoint on the held-out val split.

Reports mask (and box) mAP, and saves predicted-mask overlays (one image per
val sample, mask + box drawn) to runs/predict/ for eyeballing.

    python evaluate.py                                   # uses runs/talc_seg/weights/best.pt
    python evaluate.py --weights <path> --conf 0.1        # lower threshold for a weak/early checkpoint
    python evaluate.py --no-predict                       # metrics only, skip overlays
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
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold for overlay predictions")
    ap.add_argument("--hide-boxes", action="store_true",
                    help="draw only the segmentation masks (no boxes/labels) so the segments are unambiguous")
    ap.add_argument("--no-predict", action="store_true", help="skip saving predicted-mask overlays")
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

    if not args.no_predict:
        val_imgs = (YOLO_SEG / "images" / "val").resolve()
        results = model.predict(
            source=str(val_imgs),
            save=True,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            show_boxes=not args.hide_boxes,  # hide boxes/labels -> only the pixel masks remain
            project=str(HERE / "runs"),
            name="predict",
            exist_ok=True,
        )
        n_dets = sum(len(r.boxes) for r in results)
        print(f"\n{n_dets} detections across {len(results)} images at conf={args.conf}")
        print(f"overlays saved under {HERE / 'runs' / 'predict'}")

        if n_dets == 0:
            # Weak/early checkpoints can have zero detections at a normal operating
            # threshold even though val() reports nonzero recall — val sweeps confidence
            # down near 0 to build the mAP curve, which a fixed --conf does not. Step
            # down until something shows up, so there's always *something* to look at.
            for fallback_conf in (0.1, 0.05, 0.01, 0.001):
                if fallback_conf >= args.conf:
                    continue
                debug = model.predict(
                    source=str(val_imgs),
                    save=True,
                    conf=fallback_conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    project=str(HERE / "runs"),
                    name="predict_lowconf",
                    exist_ok=True,
                    verbose=False,
                )
                n_debug = sum(len(r.boxes) for r in debug)
                if n_debug > 0:
                    print(f"  no detections above conf={args.conf} — {n_debug} detections found at "
                          f"conf={fallback_conf} (debug only, NOT a usable threshold)")
                    print(f"  debug overlays saved under {HERE / 'runs' / 'predict_lowconf'}")
                    break
            else:
                print(f"  no detections even down to conf=0.001 — checkpoint appears untrained/degenerate")


if __name__ == "__main__":
    main()
