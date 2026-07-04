"""Standalone talc inference (classifier-gated YOLO-seg fusion) for the backend.

NEW, self-contained entry point the backend calls. It does NOT modify the model
training folders — it imports the fusion primitives from ML/fusion/fuse.py
(read-only) and installs a compatibility shim (ultra_compat) so the YOLO-seg
checkpoint (saved with custom loss classes) loads under stock ultralytics.

Fusion logic:
  * YOLO-seg predicts talc instances; talc-vs-talc overlaps are joined by taking
    the UNION of kept masks (a single talc mask, no self-overlap).
  * If the tile classifier is available, it gates the segmentation: when the
    segmenter says talcose (union fraction > --talc-frac) but the classifier is
    confidently a low-talc class (P >= --p-thresh), the lowest-confidence
    segments are pruned until the fraction drops back under the threshold.
  * The classifier is OPTIONAL: if its checkpoint is missing / fails to load, we
    fall back to segmentation-only (union, no gating) so talc still works.

Outputs (exactly what ML/merge/merge_phases.py consumes):
  <out>/talc.json      per-image record: class/P, raw/final talc fraction, every
                       KEPT/pruned segment with confidence + normalized polygon.
  <out>/talc_mask.png  uint8 HxW, 255 where talc — the UNION of KEPT masks.

Example:
    python infer.py --image /abs/img.jpg --out job/talc
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageFile

HERE = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)                       # ultra_compat
sys.path.insert(0, os.path.join(ML_DIR, "fusion"))  # fuse.py (read-only reuse)

from ultra_compat import install_loss_stubs  # noqa: E402
install_loss_stubs()                          # must precede YOLO(weights)

from ultralytics import YOLO  # noqa: E402
from fuse import (  # noqa: E402
    CLASS_RU,
    DEFAULT_CLS_CKPT,
    DEFAULT_SEG_WEIGHTS,
    LOW_TALC_CLASSES,
    classify,
    get_device,
    load_classifier,
    prune_to_fraction,
    seg_record,
    segment,
    union_frac,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _try_load_classifier(ckpt, device):
    """Return the classifier pack, or None if weights are absent / fail to load."""
    if not ckpt or not os.path.isfile(ckpt):
        print(f"[talc_infer] classifier ckpt not found ({ckpt}); "
              f"running segmentation-only (no gating).")
        return None
    try:
        return load_classifier(ckpt, device)
    except Exception as exc:  # noqa: BLE001 — degrade gracefully to seg-only
        print(f"[talc_infer] classifier load failed ({exc}); segmentation-only.")
        return None


def run_one(path, cls_pack, seg_model, args, device):
    """Fusion on ONE image -> (record dict, union mask uint8)."""
    img_pil = Image.open(path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # classifier is optional
    if cls_pack is not None:
        model, tf, classes, _ = cls_pack
        cls, p, probs = classify(model, tf, classes, img_pil, device, tta=not args.no_tta)
    else:
        cls, p, probs = "unknown", 0.0, {}

    segs, shape = segment(seg_model, img_bgr, args.seg_conf, args.imgsz, device)
    f_raw = union_frac(segs, shape)

    rule_fired = False
    note = "no classifier — segmentation only" if cls_pack is None else "no conflict"
    kept, pruned = segs, []
    if cls_pack is not None and f_raw > args.talc_frac and cls in LOW_TALC_CLASSES:
        if p >= args.p_thresh:
            kept, pruned = prune_to_fraction(segs, shape, args.talc_frac)
            rule_fired = True
            note = (f"seg said {f_raw:.1%} talc but classifier is confident it is "
                    f"'{cls}' (<{args.talc_frac:.0%}): pruned {len(pruned)} segment(s)")
        else:
            note = (f"seg said {f_raw:.1%} talc, classifier says '{cls}' but "
                    f"P={p:.2f} < {args.p_thresh}: segmentation kept as-is")
    f_final = union_frac(kept, shape)

    h, w = shape
    union = np.zeros((h, w), dtype=np.uint8)
    for s in kept:
        union[s["mask"]] = 255

    record = {
        "image": os.path.basename(path),
        "width": w, "height": h,
        "class": cls,
        "class_ru": CLASS_RU.get(cls, cls),
        "class_confidence": round(p, 4),
        "class_probs": {c: round(v, 4) for c, v in probs.items()},
        "talc_fraction_raw": round(f_raw, 5),
        "talc_fraction_final": round(f_final, 5),
        "is_talcose_final": f_final > args.talc_frac or cls == "talc",
        "rule_fired": rule_fired,
        "note": note,
        "segments": ([seg_record(s, shape, True) for s in kept] +
                     [seg_record(s, shape, False) for s in pruned]),
    }
    return record, union


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--cls-ckpt", default=DEFAULT_CLS_CKPT)
    ap.add_argument("--seg-weights", default=DEFAULT_SEG_WEIGHTS)
    ap.add_argument("--p-thresh", type=float, default=0.9)
    ap.add_argument("--seg-conf", type=float, default=0.42)
    ap.add_argument("--talc-frac", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-classifier", action="store_true",
                    help="force segmentation-only (skip classifier gating)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"--image not found: {args.image}")
    if not os.path.isfile(args.seg_weights):
        sys.exit(f"--seg-weights not found: {args.seg_weights}")
    os.makedirs(args.out, exist_ok=True)

    device = get_device(args.device)
    cls_pack = None if args.no_classifier else _try_load_classifier(args.cls_ckpt, device)
    seg_model = YOLO(args.seg_weights)

    record, union = run_one(args.image, cls_pack, seg_model, args, device)

    with open(os.path.join(args.out, "talc.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    cv2.imwrite(os.path.join(args.out, "talc_mask.png"), union)

    print(f"{record['image']}: {record['class']} P={record['class_confidence']:.2f} | "
          f"talc {record['talc_fraction_raw']:.1%} -> {record['talc_fraction_final']:.1%} "
          f"({len(record['segments'])} seg) -> {args.out}")


if __name__ == "__main__":
    main()
