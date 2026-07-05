"""Standalone talc inference (classifier-gated YOLO-seg fusion) for the backend.

NEW, self-contained entry point the backend calls. It does NOT modify the model
training folders — it imports the fusion primitives from ML/fusion/fuse.py
(read-only) and installs a compatibility shim (ultra_compat) so the YOLO-seg
checkpoint (saved with custom loss classes) loads under stock ultralytics.

Two inference paths:

  * SINGLE FIELD (run_one) — images at the training field scale (~2272x1704).
    Classic fusion: YOLO-seg union + optional classifier gating (prune the
    lowest-confidence segments when a confident low-talc classifier disagrees).
  * PANORAMA (run_panorama) — stitched mosaics much larger than one field.
    The image is cut into field-scale tiles (shared grid geometry from
    ML/talc_seg_normalized/tiling.py), each tile is optionally brought to the
    canonical (sulfide) colour profile (preprocess_talc, PER TILE) and passed
    to YOLO at the training imgsz; tile masks are stitched into one full-res
    union with logical OR. The classifier gate is OFF here (it was trained on
    single fields; the global talc >10% verdict is the merge step's job).
    Chosen automatically when max(H, W) > --tile-threshold x tile, or forced
    with --tile.

Weights are loaded SEPARATELY (sulfide convention). Resolution order:
  --seg-weights > env TALC_SEG_WEIGHTS > ML/talc_seg_normalized/weights/
  talc_seg_best.pt (variant-B, canonical profile) > raw baseline
  (ML/first_segmentation_attempt/runs/talc_seg_v2/weights/best.pt).

PROFILE COUPLING (do not break): weights trained on the normalized dataset MUST
see normalized tiles; the raw baseline MUST see raw tiles. --profile auto infers
this from the weights path (anything under talc_seg_normalized/ => norm).

Outputs (exactly what ML/merge/merge_phases.py consumes):
  <out>/talc.json      per-image record: class/P, raw/final talc fraction, every
                       KEPT/pruned segment with confidence + normalized polygon.
  <out>/talc_mask.png  uint8 HxW, 255 where talc — the UNION of KEPT masks.

Example:
    python infer.py --image /abs/img.jpg --out job/talc
    python infer.py --image /abs/panorama.jpg --out job/talc --tile
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
sys.path.insert(0, os.path.join(ML_DIR, "talc_seg_normalized"))  # tiling + normalize

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
    mask_polygons,
    prune_to_fraction,
    seg_record,
    segment,
    union_frac,
)
import tiling  # noqa: E402 — shared grid geometry (talc_seg_normalized)
from normalize import preprocess_talc  # noqa: E402 — canonical (sulfide) profile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Variant-B weights (trained on the canonical profile) — preferred when present.
NORM_WEIGHTS_DIR = os.path.join(ML_DIR, "talc_seg_normalized", "weights")
NORM_WEIGHTS = os.path.join(NORM_WEIGHTS_DIR, "talc_seg_best.pt")

# Field scale the talc model was trained at (long side of one micrograph).
TRAIN_FIELD_LONG_SIDE = 2272


def resolve_seg_weights(cli_arg):
    """--seg-weights > $TALC_SEG_WEIGHTS > normalized best > raw baseline."""
    if cli_arg:
        return cli_arg
    env = os.environ.get("TALC_SEG_WEIGHTS")
    if env:
        return env
    if os.path.isfile(NORM_WEIGHTS):
        return NORM_WEIGHTS
    return DEFAULT_SEG_WEIGHTS


def resolve_profile(profile_arg, seg_weights):
    """Return True if tiles must be normalized (canonical profile weights).

    'auto' couples the profile to the weights location: any checkpoint under
    talc_seg_normalized/ was trained on preprocess_talc-normalized tiles.
    """
    if profile_arg == "norm":
        return True
    if profile_arg == "raw":
        return False
    return os.path.abspath(seg_weights).startswith(
        os.path.abspath(os.path.join(ML_DIR, "talc_seg_normalized")) + os.sep)


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


def run_one(path, cls_pack, seg_model, args, device, normalize_input=False):
    """Fusion on ONE field-scale image -> (record dict, union mask uint8)."""
    img_pil = Image.open(path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    if normalize_input:
        img_bgr = preprocess_talc(img_bgr)

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


def run_panorama(path, seg_model, args, device, normalize_input):
    """Tiled fusion on a stitched panorama -> (record dict, union mask uint8).

    Field-scale tiles, per-tile canonical-profile normalization (variant B),
    logical-OR stitching. No classifier gate (see module docstring). The output
    record keeps the exact run_one contract so merge_phases.py needs no changes.
    """
    # cv2.imread ignores EXIF orientation — same reader semantics as the sulfide
    # pipeline, so talc_mask and the sulfide classmap always share (W, H).
    pano = cv2.imread(path, cv2.IMREAD_COLOR)
    if pano is None:
        raise FileNotFoundError(f"Cannot read {path}")
    h, w = pano.shape[:2]
    tile, stride = args.tile_size, args.tile_stride or int(args.tile_size * 0.7)

    union = np.zeros((h, w), dtype=np.uint8)
    seg_records = []
    coords = list(tiling.iter_tiles(h, w, tile, stride))
    print(f"[talc_infer] panorama {w}x{h}: {len(coords)} tiles "
          f"({tile}px / stride {stride}) | profile: "
          f"{'canonical (normalized per tile)' if normalize_input else 'raw'}")

    for y0, x0 in coords:
        crop = tiling.crop_padded(pano, y0, x0, tile)
        if normalize_input:
            crop = preprocess_talc(crop)                 # PER TILE anchors
        segs, _ = segment(seg_model, crop, args.seg_conf, args.imgsz, device)
        if not segs:
            continue
        vh, vw = tiling.valid_extent(h, w, y0, x0, tile)
        for s in segs:
            m = s["mask"][:vh, :vw]                      # drop reflect padding
            if not m.any():
                continue                                  # lived only in padding
            union[y0:y0 + vh, x0:x0 + vw] |= m.astype(np.uint8)
            # polygons from the clipped mask -> global px -> normalized
            polys_px = []
            for ring in mask_polygons(m):
                g = ring.astype(np.int64)
                g[:, 0] += x0
                g[:, 1] += y0
                np.clip(g[:, 0], 0, w - 1, out=g[:, 0])
                np.clip(g[:, 1], 0, h - 1, out=g[:, 1])
                polys_px.append(g)
            if not polys_px:
                continue
            area = int(m.sum())
            seg_records.append({
                "confidence": round(float(s["conf"]), 4),
                "kept": True,
                "area_px": area,
                "area_frac": round(area / float(h * w), 5),
                "polygons_px": [[[int(x), int(y)] for x, y in g] for g in polys_px],
                "polygons_norm": [[[round(float(x) / w, 5), round(float(y) / h, 5)]
                                   for x, y in g] for g in polys_px],
            })

    union *= 255
    f_union = float((union > 0).sum()) / (h * w)
    record = {
        "image": os.path.basename(path),
        "width": w, "height": h,
        "class": "unknown",
        "class_ru": CLASS_RU.get("unknown", "unknown"),
        "class_confidence": 0.0,
        "class_probs": {},
        "talc_fraction_raw": round(f_union, 5),
        "talc_fraction_final": round(f_union, 5),
        "is_talcose_final": f_union > args.talc_frac,
        "rule_fired": False,
        "note": (f"panorama mode: {len(coords)} tiles @ {tile}px, "
                 f"{'canonical profile' if normalize_input else 'raw profile'}, "
                 "no classifier gate (verdict belongs to the merge step)"),
        "segments": seg_records,
    }
    return record, union


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--cls-ckpt", default=DEFAULT_CLS_CKPT)
    ap.add_argument("--seg-weights", default=None,
                    help="YOLO-seg checkpoint; default: $TALC_SEG_WEIGHTS, then "
                         "talc_seg_normalized/weights/talc_seg_best.pt, then raw baseline")
    ap.add_argument("--profile", choices=["auto", "norm", "raw"], default="auto",
                    help="input colour profile; auto = infer from weights path "
                         "(normalized weights <=> normalized tiles)")
    ap.add_argument("--p-thresh", type=float, default=0.9)
    ap.add_argument("--seg-conf", type=float, default=0.42)
    ap.add_argument("--talc-frac", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-classifier", action="store_true",
                    help="force segmentation-only (skip classifier gating)")
    ap.add_argument("--tile", action="store_true",
                    help="force the tiled panorama path")
    ap.add_argument("--no-tile", action="store_true",
                    help="force the single-image path even for huge inputs")
    ap.add_argument("--tile-size", type=int, default=TRAIN_FIELD_LONG_SIDE,
                    help="tile side, px (default = training field long side)")
    ap.add_argument("--tile-stride", type=int, default=None,
                    help="tile stride, px (default = 70%% of tile => ~30%% overlap)")
    ap.add_argument("--tile-threshold", type=float, default=1.5,
                    help="auto-tile when max(H,W) > threshold * tile-size")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"--image not found: {args.image}")
    seg_weights = resolve_seg_weights(args.seg_weights)
    if not os.path.isfile(seg_weights):
        sys.exit(f"seg weights not found: {seg_weights}")
    normalize_input = resolve_profile(args.profile, seg_weights)
    os.makedirs(args.out, exist_ok=True)

    device = get_device(args.device)
    seg_model = YOLO(seg_weights)
    print(f"[talc_infer] weights: {seg_weights} | "
          f"profile: {'canonical/normalized' if normalize_input else 'raw'}")

    # panorama vs single field
    with Image.open(args.image) as im:
        w0, h0 = im.size
    use_tiles = args.tile or (not args.no_tile and
                              max(w0, h0) > args.tile_threshold * args.tile_size)

    if use_tiles:
        record, union = run_panorama(args.image, seg_model, args, device,
                                     normalize_input)
    else:
        cls_pack = None if args.no_classifier else _try_load_classifier(args.cls_ckpt, device)
        record, union = run_one(args.image, cls_pack, seg_model, args, device,
                                normalize_input=normalize_input)

    with open(os.path.join(args.out, "talc.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    cv2.imwrite(os.path.join(args.out, "talc_mask.png"), union)

    print(f"{record['image']}: {record['class']} P={record['class_confidence']:.2f} | "
          f"talc {record['talc_fraction_raw']:.1%} -> {record['talc_fraction_final']:.1%} "
          f"({len(record['segments'])} seg) -> {args.out}")


if __name__ == "__main__":
    main()
