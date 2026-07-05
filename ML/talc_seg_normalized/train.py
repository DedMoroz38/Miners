"""Fine-tune YOLO-seg for talc on the anchor-NORMALIZED dataset (variant B).

Step 2. Trains on the profile-normalized tiles built by build_dataset.py, so the
weights expect the same canonical profile the panorama tiler feeds at inference
(normalize.preprocess_talc, applied per fragment). Only the input profile differs
from ML/first_segmentation_attempt/train.py — the augmentation recipe
(SMALL_DATA_OVERRIDES) is kept identical so results stay comparable to the raw
baseline.

Runs on CUDA, Apple MPS, or CPU. `--device auto` (default) picks CUDA if present,
else MPS, else CPU; override with `--device 0` / `mps` / `cpu`.

    python build_dataset.py          # once, produces ./dataset_norm/data.yaml
    python train.py                  # auto device, yolo11n-seg
    python train.py --device 0 --model yolo11l-seg.pt --imgsz 768   # V100
    python train.py --smoke          # 3-epoch CPU sanity check

Weights are loaded SEPARATELY (same convention as the sulfide pipeline): training
writes runs/<name>/weights/best.pt; copy the chosen checkpoint into ./weights/
(see weights/README.md) and point the talc inference at it. This script never
touches the inference weights dir.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "dataset_norm" / "data.yaml"

# Identical to ML/first_segmentation_attempt/train.py so the normalized run and
# the raw baseline differ ONLY in the input colour profile.
SMALL_DATA_OVERRIDES = dict(
    degrees=90.0,
    flipud=0.5,
    cos_lr=True,
    close_mosaic=30,
)


def resolve_device(arg: str) -> str:
    """Map 'auto' to CUDA -> MPS -> CPU; pass explicit choices through."""
    if arg and arg != "auto":
        return arg
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 — torch import/probe failure -> CPU
        pass
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="normalized dataset data.yaml (from build_dataset.py)")
    ap.add_argument("--model", default="yolo11n-seg.pt",
                    help="pretrained seg checkpoint (yolo11l-seg.pt on a big GPU)")
    ap.add_argument("--epochs", type=int, default=300,
                    help="upper bound; --patience stops once val mAP stalls")
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | mps | '0' (CUDA index)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default="talc_seg_norm", help="run dir under runs/")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU run to verify the pipeline (3 epochs, imgsz 320)")
    args = ap.parse_args()

    if args.smoke:
        args.model, args.epochs, args.imgsz = "yolo11n-seg.pt", 3, 320
        args.batch, args.device, args.workers = 2, "cpu", 0
        args.name = "smoke"

    data = Path(args.data)
    if not data.is_file():
        raise SystemExit(
            f"Normalized dataset not found: {data}\n"
            "Build it first: python build_dataset.py")

    device = resolve_device(args.device)
    print(f"data: {data} | model: {args.model} | device: {device} | imgsz: {args.imgsz}")

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(HERE / "runs"),
        name=args.name,
        exist_ok=True,
        seed=42,
        plots=not args.smoke,
        **({} if args.smoke else SMALL_DATA_OVERRIDES),
    )

    best = HERE / "runs" / args.name / "weights" / "best.pt"
    print(f"done. best weights: {best}")
    print(f"next: cp '{best}' '{HERE / 'weights' / 'talc_seg_best.pt'}'  "
          "# load separately at inference (see weights/README.md)")


if __name__ == "__main__":
    main()
