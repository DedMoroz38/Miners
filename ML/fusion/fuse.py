"""Two-model talc pipeline: classifier-gated YOLO-seg segmentation.

Takes a folder of ore thin-section images and runs both models on every image:

  1. EfficientNetV2-S classifier (ordinary / hard / talc) -> class + confidence P.
     8-view D4 test-time augmentation by default (label-safe for micrographs).
  2. YOLO11-seg talc segmenter -> instances with confidences. Talc area-fraction
     f = UNION of mask pixels / image pixels (union, not sum: instance masks
     overlap, and summing per-instance areas can exceed 100%).

  Conflict rule (seg says talcose, classifier disagrees):
     if f > --talc-frac AND class is non-talc (ordinary/hard imply <10% talc):
       P >= --p-thresh  -> trust the classifier: drop lowest-confidence segments
                           until f <= --talc-frac (dropped ones are still
                           reported, with kept=false).
       P <  --p-thresh  -> classifier not confident enough; keep segmentation.

Outputs per run:
  <out>/results.json          machine-readable (for the frontend): per image the
                              class, P, f before/after, and every segment with
                              confidence + polygon in pixel AND normalized coords.
  <out>/overlays/<name>.jpg   kept segments green (filled), pruned ones gray
                              (outline only), header with the verdict.

Threshold defaults (see README for the measurement):
  --seg-conf 0.42   val-split mask F1 peaks at 0.516 and is within 95% of peak
                    over [0.415, 0.550]; 0.42 takes the recall-leaning end since
                    the fusion rule can only remove segments, never add.
  --p-thresh 0.9    measured by tune_thresholds.py on the 177-image val split
                    (TTA): accuracy among P>=tau predictions first reaches 95%
                    at tau=0.90 (0.957, 53% coverage); at 0.80 it is only 0.941.

Example:
    python fuse.py --images /path/to/folder --out runs/fusion
"""
import argparse
import importlib.util
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import transforms
from ultralytics import YOLO

ImageFile.LOAD_TRUNCATED_IMAGES = True  # some hackathon JPEGs are truncated

FUSION_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(FUSION_DIR)
AUG_SRC = os.path.join(ML_DIR, "first_augmentation_attempt", "src")
DEFAULT_CLS_CKPT = os.path.join(ML_DIR, "first_augmentation_attempt", "runs",
                                "best_efficientnet_v2_s.pt")
FALLBACK_CLS_CKPT = os.path.join(ML_DIR, "first_classification_attempt", "runs",
                                 "best_efficientnet_v2_s.pt")
DEFAULT_SEG_WEIGHTS = os.path.join(ML_DIR, "first_segmentation_attempt", "runs",
                                   "talc_seg_v2", "weights", "best.pt")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Classifier classes -> Russian names + the <10%-talc mapping. Both non-talc
# classes imply "<10% talc" for the conflict rule (confirmed with the domain
# owner): only "оталькованные" means >=10%.
CLASS_RU = {"ordinary": "рядовые", "hard": "труднообогатимые", "talc": "оталькованные"}
LOW_TALC_CLASSES = {"ordinary", "hard"}


def load_module(path, name):
    """Import a module by file path without touching sys.path (both sibling
    experiments have generically-named model.py/data.py that would clash)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_device(arg):
    if arg != "auto":
        return torch.device(arg)
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --- Classifier ---------------------------------------------------------------
def load_classifier(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        if ckpt_path == DEFAULT_CLS_CKPT and os.path.exists(FALLBACK_CLS_CKPT):
            print(f"WARNING: augmented classifier ckpt not found at\n  {ckpt_path}\n"
                  f"  falling back to the BASELINE classifier:\n  {FALLBACK_CLS_CKPT}\n"
                  f"  (train the augmented one, then re-run for the better model)")
            ckpt_path = FALLBACK_CLS_CKPT
        else:
            raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_mod = load_module(os.path.join(AUG_SRC, "model.py"), "fusion_cls_model")
    model = model_mod.build_model(ckpt["model"], len(ckpt["classes"]),
                                  pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    img_size = ckpt.get("img_size", 256)
    # Same deterministic eval transform as first_augmentation_attempt/src/data.py.
    tf = transforms.Compose([
        transforms.Resize(int(round(img_size * 1.14))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return model, tf, list(ckpt["classes"]), ckpt_path


def d4_views(x):
    """The 8 dihedral-group views of a square image batch (label-safe TTA)."""
    for k in range(4):
        r = torch.rot90(x, k, dims=(-2, -1))
        yield r
        yield torch.flip(r, dims=(-1,))


@torch.no_grad()
def classify(model, tf, classes, img_pil, device, tta=True):
    x = tf(img_pil).unsqueeze(0).to(device)
    if tta:
        views = torch.cat(list(d4_views(x)), dim=0)      # (8, C, H, W)
        probs = F.softmax(model(views), dim=1).mean(0)
    else:
        probs = F.softmax(model(x), dim=1)[0]
    probs = probs.cpu().numpy()
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx]), {c: float(p) for c, p in zip(classes, probs)}


# --- Segmentation --------------------------------------------------------------
def mask_polygons(mask, min_area_px=32, eps_frac=0.01):
    """One clean polygon per connected blob of the mask (external contours).

    ultralytics' masks.xy joins disjoint blobs of one instance into a single
    self-intersecting polygon with thin bridge lines — bad for frontend drawing.

    Contours are simplified with Douglas–Peucker (approxPolyDP) at
    eps = eps_frac * perimeter so jagged pixel boundaries collapse to a handful
    of vertices. Mirrors ML/merge/merge_phases.py:mask_polygons.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if len(c) < 3 or cv2.contourArea(c) < min_area_px:
            continue
        eps = eps_frac * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True) if eps > 0 else c
        if len(approx) < 3:  # over-collapsed a valid blob — keep the raw ring
            approx = c
        out.append(approx.reshape(-1, 2))
    return out


def segment(seg_model, img_bgr, conf, imgsz, device):
    """Run YOLO-seg on a BGR array; return list of dicts with mask + polygons + conf."""
    r = seg_model.predict(source=img_bgr, conf=conf, imgsz=imgsz,
                          retina_masks=True,  # masks at original resolution
                          device=device, verbose=False)[0]
    h, w = r.orig_shape
    segs = []
    if r.masks is not None:
        confs = r.boxes.conf.cpu().numpy()
        masks = r.masks.data.cpu().numpy() > 0.5          # (N, h, w) bool
        for i in range(len(confs)):
            segs.append({
                "conf": float(confs[i]),
                "mask": masks[i],
                "polygons": mask_polygons(masks[i]),      # list of (K, 2) int arrays
            })
    return segs, (h, w)


def union_frac(segs, shape):
    """Talc area-fraction from the UNION of masks (overlaps counted once)."""
    if not segs:
        return 0.0
    u = np.zeros(shape, dtype=bool)
    for s in segs:
        u |= s["mask"]
    return float(u.sum()) / (shape[0] * shape[1])


def prune_to_fraction(segs, shape, target_frac):
    """Drop lowest-confidence segments until union fraction <= target.

    Returns (kept, pruned). Greedy by confidence, recomputing the union after
    each drop because overlapping instances share pixels.
    """
    kept = sorted(segs, key=lambda s: s["conf"], reverse=True)
    pruned = []
    while kept and union_frac(kept, shape) > target_frac:
        pruned.append(kept.pop())          # lowest-confidence of the remainder
    return kept, pruned


# --- Output --------------------------------------------------------------------
def seg_record(s, shape, kept):
    h, w = shape
    return {
        "confidence": round(s["conf"], 4),
        "kept": kept,
        "area_px": int(s["mask"].sum()),
        "area_frac": round(float(s["mask"].sum()) / (h * w), 5),
        # One polygon per connected blob of the instance mask.
        "polygons_px": [[[int(x), int(y)] for x, y in poly]
                        for poly in s["polygons"]],
        "polygons_norm": [[[round(float(x) / w, 5), round(float(y) / h, 5)]
                           for x, y in poly] for poly in s["polygons"]],
    }


def draw_overlay(img_bgr, kept, pruned, header_lines):
    out = img_bgr.copy()
    # Kept: green filled + outline. Pruned: gray outline only.
    if kept:
        fill = out.copy()
        for s in kept:
            fill[s["mask"]] = (0, 200, 0)
        out = cv2.addWeighted(fill, 0.35, out, 0.65, 0)
    scale = max(out.shape[1] / 1600, 0.7)
    line_h = int(34 * scale)
    band_h = line_h * len(header_lines) + int(12 * scale)
    for segs, color, tag in ((kept, (0, 220, 0), ""), (pruned, (128, 128, 128), " pruned")):
        for s in segs:
            if not s["polygons"]:
                continue
            polys = [p.astype(np.int32).reshape(-1, 1, 2) for p in s["polygons"]]
            cv2.polylines(out, polys, True, color, max(2, int(2 * scale)))
            # label at the largest blob, kept clear of the header band below
            big = max(polys, key=cv2.contourArea)
            x, y = big[:, 0, 0].min(), big[:, 0, 1].min()
            cv2.putText(out, f"{s['conf']:.2f}{tag}",
                        (int(x), max(band_h + line_h // 2, int(y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, color,
                        max(1, int(2 * scale)), cv2.LINE_AA)
    # Header band (ASCII only: cv2.putText cannot render Cyrillic).
    band = out[:band_h].copy()
    band[:] = (0, 0, 0)
    out[:band_h] = cv2.addWeighted(band, 0.55, out[:band_h], 0.45, 0)
    for i, line in enumerate(header_lines):
        cv2.putText(out, line, (int(10 * scale), line_h * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (255, 255, 255),
                    max(1, int(2 * scale)), cv2.LINE_AA)
    return out


# --- Main ----------------------------------------------------------------------
def process_image(path, cls_pack, seg_model, args, device):
    model, tf, classes, _ = cls_pack
    # Load once via PIL (tolerates the dataset's truncated JPEGs and unicode
    # paths, which cv2.imread does not) and share the array with both models.
    img_pil = Image.open(path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cls, p, probs = classify(model, tf, classes, img_pil, device,
                             tta=not args.no_tta)
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
                    f"'{cls}' (<{args.talc_frac:.0%}): pruned {len(pruned)} "
                    f"lowest-confidence segment(s)")
        else:
            note = (f"seg said {f_raw:.1%} talc, classifier says '{cls}' but "
                    f"P={p:.2f} < {args.p_thresh}: segmentation kept as-is")
    f_final = union_frac(kept, shape)

    record = {
        "image": os.path.basename(path),
        "width": shape[1], "height": shape[0],
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

    header = [
        f"class: {cls} P={p:.2f}" + ("  [TTA]" if not args.no_tta else ""),
        f"talc area: {f_raw:.1%} -> {f_final:.1%}"
        + (f"  (pruned {len(pruned)})" if rule_fired else ""),
    ]
    overlay = draw_overlay(img_bgr, kept, pruned, header)
    return record, overlay


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", required=True, help="folder of images (or one image)")
    ap.add_argument("--out", default=os.path.join(FUSION_DIR, "runs", "fusion"))
    ap.add_argument("--cls-ckpt", default=DEFAULT_CLS_CKPT)
    ap.add_argument("--seg-weights", default=DEFAULT_SEG_WEIGHTS)
    ap.add_argument("--p-thresh", type=float, default=0.9,
                    help="classifier trust threshold tau")
    ap.add_argument("--seg-conf", type=float, default=0.42,
                    help="YOLO-seg detection confidence")
    ap.add_argument("--talc-frac", type=float, default=0.10,
                    help="talcose decision line (area fraction)")
    ap.add_argument("--imgsz", type=int, default=640, help="seg inference size")
    ap.add_argument("--no-tta", action="store_true", help="disable 8-view TTA")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    cls_pack = load_classifier(args.cls_ckpt, device)
    seg_model = YOLO(args.seg_weights)
    print(f"Device: {device}\nClassifier: {cls_pack[3]}\nSegmenter: {args.seg_weights}\n"
          f"tau={args.p_thresh}  seg-conf={args.seg_conf}  talc-line={args.talc_frac:.0%}")

    if not os.path.exists(args.images):
        sys.exit(f"--images path does not exist: {args.images}")
    if os.path.isdir(args.images):
        paths = sorted(os.path.join(args.images, n) for n in os.listdir(args.images)
                       if os.path.splitext(n)[1].lower() in IMG_EXTS)
    else:
        paths = [args.images]
    if not paths:
        sys.exit(f"No images found in {args.images}")

    overlay_dir = os.path.join(args.out, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    records = []
    for i, path in enumerate(paths, 1):
        # One bad file must not kill a long batch: log the failure and move on.
        try:
            record, overlay = process_image(path, cls_pack, seg_model, args, device)
        except Exception as e:
            print(f"[{i}/{len(paths)}] {os.path.basename(path)}: FAILED ({e})")
            records.append({"image": os.path.basename(path), "error": str(e)})
            continue
        records.append(record)
        out_img = os.path.join(
            overlay_dir, os.path.splitext(os.path.basename(path))[0] + ".jpg")
        cv2.imwrite(out_img, overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        flag = " <- rule fired" if record["rule_fired"] else ""
        print(f"[{i}/{len(paths)}] {record['image']}: {record['class']} "
              f"P={record['class_confidence']:.2f} | talc "
              f"{record['talc_fraction_raw']:.1%} -> {record['talc_fraction_final']:.1%} "
              f"({len(record['segments'])} seg){flag}", flush=True)

    payload = {
        "config": {
            "classifier_ckpt": cls_pack[3],
            "seg_weights": args.seg_weights,
            "p_thresh": args.p_thresh,
            "seg_conf": args.seg_conf,
            "talc_frac": args.talc_frac,
            "tta": not args.no_tta,
        },
        "images": records,
    }
    out_json = os.path.join(args.out, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    fired = sum(r.get("rule_fired", False) for r in records)
    failed = sum("error" in r for r in records)
    print(f"\nDone: {len(records)} images, rule fired on {fired}, failed {failed}.")
    print(f"JSON     -> {out_json}")
    print(f"Overlays -> {overlay_dir}")


if __name__ == "__main__":
    main()
