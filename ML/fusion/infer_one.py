"""Single-image talc inference for the merge pipeline.

Thin wrapper around fuse.py that runs the classifier-gated YOLO-seg talc pipeline on
ONE image and writes exactly what ML/merge/merge_phases.py needs:

  <out>/talc.json      the fuse.py per-image record (class, P, talc fraction raw/final,
                       and every KEPT/pruned segment with confidence + normalized polygon).
  <out>/talc_mask.png  uint8 HxW, 255 where talc — the UNION of KEPT segment masks
                       (talc-vs-talc overlaps already joined; pruned segments excluded).

Reuses fuse.py so thresholds/rule stay identical to the batch tool.

Example:
    python infer_one.py --image /path/img.jpg --out job/talc
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageFile
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuse import (  # noqa: E402  (needs the sys.path insert above)
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


def run_one(path, cls_pack, seg_model, args, device):
    """Mirror fuse.process_image but also return the union mask of kept segments."""
    model, tf, classes, _ = cls_pack
    img_pil = Image.open(path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cls, p, probs = classify(model, tf, classes, img_pil, device, tta=not args.no_tta)
    segs, shape = segment(seg_model, img_bgr, args.seg_conf, args.imgsz, device)
    f_raw = union_frac(segs, shape)

    rule_fired = False
    note = "no conflict"
    kept, pruned = segs, []
    if f_raw > args.talc_frac and cls in LOW_TALC_CLASSES:
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
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"--image not found: {args.image}")
    os.makedirs(args.out, exist_ok=True)

    device = get_device(args.device)
    cls_pack = load_classifier(args.cls_ckpt, device)
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
