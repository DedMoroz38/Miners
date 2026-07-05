"""Merge the two phase segmentations into one 4-class map + frontend payload.

Inputs (all at the SAME original image resolution):
  - sulfide class map  : uint8 HxW PNG, 0=bg / 1=обычные срастания / 2=тонкие срастания
                         (from ML/sulfide_intergrowth infer_panorama.py -> *_classmap.png)
  - talc union mask    : uint8 HxW PNG, >0 where talc (union of KEPT talc segments,
                         talc-vs-talc overlaps already joined upstream by fuse.py)
                         (from ML/fusion/infer_one.py -> talc_mask.png)
  - talc json          : fuse-style record, used only for per-segment talc confidence
  - sulfide segments   : *_segments.csv, used only for per-grain sulfide confidence

Overlap rules (the whole point of this module):
  1. talc-vs-talc  : already unioned upstream (single talc mask, no self-overlap).
  2. talc-vs-sulfide: SULFIDE WINS. Talc only fills pixels the sulfide map left as bg,
     so any talc that overlapped a sulfide grain is overridden -> "trust sulfide".

Output: analysis.json matching the frontend AnalysisResult contract, with every
segment as NORMALIZED polygon rings ([x/W, y/H] in 0..1) so the web layer is
resolution-independent. Optional --debug-overlay writes a green/red/blue PNG.

Deps: numpy + opencv + stdlib only (runs in any of the ML venvs).

Example:
    python merge_phases.py \
        --sulfide-classmap job/sulfide/img_classmap.png \
        --talc-mask job/talc/talc_mask.png \
        --talc-json job/talc/talc.json \
        --sulfide-segments job/sulfide/img_segments.csv \
        --image job/img.jpg \
        --out job/analysis.json --debug-overlay job/overlay.png
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

# Class map values in the merged phase map.
BG, COMMON, THIN, TALC = 0, 1, 2, 3
PHASE_NAME = {COMMON: "common", THIN: "thin", TALC: "talc"}

# Debug-overlay colours (BGR) — match the frontend PHASE_META:
#   common #22C55E green, thin #EF4444 red, talc #3B82F6 blue.
COLOR_BGR = {COMMON: (94, 197, 34), THIN: (68, 68, 239), TALC: (246, 130, 59)}


# --- geometry ----------------------------------------------------------------
def mask_polygons(mask, min_area_px=32, eps_frac=0.01):
    """One external-contour polygon per connected blob of `mask` (bool/uint8).

    Contours are simplified with Douglas–Peucker (approxPolyDP) at
    eps = eps_frac * perimeter, so a jagged pixel-boundary grain collapses from
    hundreds of vertices to a handful. On a panorama with thousands of sulfide
    grains this is what keeps the web payload small and the SVG viewer (and its
    per-vertex edit handles) responsive — the shape stays recognizable.
    Mirrors ML/fusion/fuse.py:mask_polygons so talc/sulfide vectorize identically.
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


# --- confidence sources ------------------------------------------------------
def _talc_records(talc_json):
    """Return the list of talc segment dicts (handles both flat and batch JSON)."""
    if talc_json is None:
        return []
    data = talc_json
    if isinstance(data, dict) and "images" in data:
        imgs = data["images"]
        data = imgs[0] if imgs else {}
    return data.get("segments", []) if isinstance(data, dict) else []


def build_talc_conf_map(talc_json, h, w):
    """Per-pixel talc confidence from fuse.py polygons (max over overlapping segs).

    Only KEPT segments contribute (pruned ones were dropped from the union mask).
    """
    conf = np.zeros((h, w), dtype=np.float32)
    for seg in _talc_records(talc_json):
        if not seg.get("kept", True):
            continue
        c = float(seg.get("confidence", 1.0))
        for ring in seg.get("polygons_norm", []):
            pts = np.array([[x * w, y * h] for x, y in ring], dtype=np.float32)
            if len(pts) < 3:
                continue
            layer = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(layer, [pts.astype(np.int32)], 1)
            hit = layer.astype(bool)
            conf[hit] = np.maximum(conf[hit], c)
    return conf


def build_sulfide_conf_map(segments_csv, labels_sulfide, class_map):
    """Per-pixel sulfide confidence by mapping each CSV grain centroid to its
    connected component of the sulfide mask and painting that component."""
    h, w = class_map.shape
    conf = np.full((h, w), 0.9, dtype=np.float32)  # sensible fallback
    if segments_csv is None or not os.path.isfile(segments_csv):
        return conf
    comp_conf = {}
    with open(segments_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                cx, cy = int(float(row["cx"])), int(float(row["cy"]))
                c = float(row["confidence"])
            except (KeyError, ValueError):
                continue
            if 0 <= cy < h and 0 <= cx < w:
                lbl = int(labels_sulfide[cy, cx])
                if lbl > 0:
                    comp_conf[lbl] = c
    for lbl, c in comp_conf.items():
        conf[labels_sulfide == lbl] = c
    return conf


# --- merge core --------------------------------------------------------------
def merge_maps(class_map, talc_mask):
    """merged = sulfide classmap, then talc fills ONLY background pixels."""
    merged = class_map.copy()
    if talc_mask is not None:
        if talc_mask.shape != merged.shape:
            talc_mask = cv2.resize(talc_mask, (merged.shape[1], merged.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        merged[(talc_mask > 0) & (merged == BG)] = TALC
    return merged


def compute_metrics(merged):
    total = int(merged.size)
    common = int((merged == COMMON).sum())
    thin = int((merged == THIN).sum())
    talc = int((merged == TALC).sum())
    sulfide = common + thin
    pct = lambda x, base: (100.0 * x / base) if base > 0 else 0.0
    return {
        "sulfideShare": round(pct(sulfide, total), 3),
        "commonShare": round(pct(common, sulfide), 3),
        "thinShare": round(pct(thin, sulfide), 3),
        "talcShare": round(pct(talc, total), 3),
    }


def classify(m):
    """Verdict + Russian conclusion. Ported from front/.../analyze.ts (single source)."""
    talc, common, thin = m["talcShare"], m["commonShare"], m["thinShare"]
    if talc > 10:
        verdict = "оталькованная"
        dom = "тонких" if thin > common else "обычных"
        conclusion = (f"Руда классифицирована как оталькованная: содержание талька — "
                      f"{talc:.0f}%, преобладание {dom} срастаний — {max(thin, common):.0f}%.")
    elif common >= thin:
        verdict = "рядовая"
        conclusion = (f"Руда классифицирована как рядовая: тальк — {talc:.0f}% (≤10%), "
                      f"преобладание обычных срастаний — {common:.0f}%.")
    else:
        verdict = "труднообогатимая"
        conclusion = (f"Руда классифицирована как труднообогатимая: тальк — {talc:.0f}% "
                      f"(≤10%), преобладание тонких срастаний — {thin:.0f}%.")
    return verdict, conclusion


def extract_segments(merged, talc_conf, sulfide_conf):
    """One Segment per connected component of each phase, normalized polygons."""
    h, w = merged.shape
    total = float(h * w)
    conf_by_phase = {COMMON: sulfide_conf, THIN: sulfide_conf, TALC: talc_conf}
    segments = []
    for val, name in PHASE_NAME.items():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (merged == val).astype(np.uint8), connectivity=8)
        conf_map = conf_by_phase[val]
        for cid in range(1, n):
            comp = labels == cid
            polys = mask_polygons(comp)
            if not polys:
                continue
            area = int(stats[cid, cv2.CC_STAT_AREA])
            conf = float(conf_map[comp].mean()) if comp.any() else 0.0
            segments.append({
                "id": f"{name}-{cid}",
                "phase": name,
                "confidence": round(conf, 4),
                "areaFrac": round(area / total, 6),
                "polygons": [[[round(float(x) / w, 5), round(float(y) / h, 5)]
                              for x, y in ring] for ring in polys],
            })
    return segments


def build_payload(class_map, talc_mask, talc_json, segments_csv, f1=0.0):
    h, w = class_map.shape
    merged = merge_maps(class_map, talc_mask)

    # sulfide grain labels (both classes) for the CSV centroid mapping
    _, labels_sulf, _, _ = cv2.connectedComponentsWithStats(
        (class_map > 0).astype(np.uint8), connectivity=8)
    talc_conf = build_talc_conf_map(talc_json, h, w)
    sulfide_conf = build_sulfide_conf_map(segments_csv, labels_sulf, class_map)

    metrics = compute_metrics(merged)
    verdict, conclusion = classify(metrics)
    segments = extract_segments(merged, talc_conf, sulfide_conf)

    payload = {
        **metrics,
        "verdict": verdict,
        "conclusion": conclusion,
        "f1": f1,
        "imageWidth": w,
        "imageHeight": h,
        "segments": segments,
    }
    return payload, merged


def make_debug_overlay(image_path, merged, alpha=0.45):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR) if image_path else None
    if bgr is None:
        bgr = np.full((*merged.shape, 3), 20, dtype=np.uint8)
    if bgr.shape[:2] != merged.shape:
        bgr = cv2.resize(bgr, (merged.shape[1], merged.shape[0]))
    color = np.zeros_like(bgr)
    for val, c in COLOR_BGR.items():
        color[merged == val] = c
    hit = merged > 0
    out = bgr.copy()
    out[hit] = cv2.addWeighted(bgr, 1.0 - alpha, color, alpha, 0.0)[hit]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sulfide-classmap", required=True)
    ap.add_argument("--talc-mask", default=None)
    ap.add_argument("--talc-json", default=None)
    ap.add_argument("--sulfide-segments", default=None)
    ap.add_argument("--decision-json", default=None,
                    help="sulfide weights/decision.json — source of the reported f1")
    ap.add_argument("--image", default=None, help="only for --debug-overlay")
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug-overlay", default=None)
    args = ap.parse_args()

    class_map = cv2.imread(args.sulfide_classmap, cv2.IMREAD_GRAYSCALE)
    if class_map is None:
        raise FileNotFoundError(f"Cannot read sulfide class map: {args.sulfide_classmap}")

    talc_mask = None
    if args.talc_mask and os.path.isfile(args.talc_mask):
        talc_mask = cv2.imread(args.talc_mask, cv2.IMREAD_GRAYSCALE)

    talc_json = None
    if args.talc_json and os.path.isfile(args.talc_json):
        with open(args.talc_json, encoding="utf-8") as f:
            talc_json = json.load(f)

    f1 = 0.0
    if args.decision_json and os.path.isfile(args.decision_json):
        with open(args.decision_json, encoding="utf-8") as f:
            d = json.load(f)
        f1 = float(d.get("val_f1") or d.get("holdout_f1") or d.get("f1") or 0.0)

    payload, merged = build_payload(class_map, talc_mask, talc_json,
                                    args.sulfide_segments, f1=f1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.debug_overlay:
        cv2.imwrite(args.debug_overlay, make_debug_overlay(args.image, merged))

    print(f"merged {merged.shape[1]}x{merged.shape[0]} | "
          f"sulfide {payload['sulfideShare']:.1f}% (common {payload['commonShare']:.1f}%, "
          f"thin {payload['thinShare']:.1f}%) talc {payload['talcShare']:.1f}% | "
          f"{len(payload['segments'])} segments | {payload['verdict']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
