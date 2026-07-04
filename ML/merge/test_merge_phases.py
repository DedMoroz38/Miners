"""Synthetic verification of merge_phases.py — the overlap-rule crux.

Runs without any weights/venv: fabricates a sulfide classmap, a talc mask that
OVERLAPS a sulfide grain, a fuse-style talc.json and a sulfide segments.csv,
then asserts the merge obeys the spec:

  * talc-vs-sulfide : SULFIDE WINS (talc only fills background pixels)
  * talc-vs-talc    : single union mask, pruned (kept=False) segments excluded
  * resolution mismatch talc_mask vs classmap is handled (NEAREST resize)
  * metrics / verdict / f1 passthrough / normalized polygons are correct

Run:  python ML/merge/test_merge_phases.py   (needs numpy + opencv only)
"""
import csv
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import merge_phases as M

H, W = 200, 300
BG, COMMON, THIN, TALC = 0, 1, 2, 3
TOL = 1e-3  # merge rounds shares to 3 decimals -> max error 5e-4


def _norm_ring(x0, y0, x1, y1):
    return [[x0 / W, y0 / H], [x1 / W, y0 / H], [x1 / W, y1 / H], [x0 / W, y1 / H]]


def build_inputs():
    cm = np.zeros((H, W), np.uint8)
    cm[40:80, 40:120] = COMMON       # common grain (green)
    cm[120:170, 200:260] = THIN      # thin grain (red)

    talc = np.zeros((H, W), np.uint8)
    talc[50:90, 90:180] = 255        # overlaps common (90..120) + bg (120..180)
    talc[100:150, 20:80] = 255       # fully in bg -> survives as talc

    talc_json = {
        "image": "synthetic.png", "width": W, "height": H,
        "class": "talc", "class_confidence": 0.95,
        "segments": [
            {"confidence": 0.80, "kept": True, "polygons_norm": [_norm_ring(90, 50, 180, 90)]},
            {"confidence": 0.40, "kept": True, "polygons_norm": [_norm_ring(20, 100, 80, 150)]},
            {"confidence": 0.10, "kept": False, "polygons_norm": [_norm_ring(0, 0, 5, 5)]},
        ],
    }
    rows = [
        {"confidence": 0.88, "cx": 80, "cy": 60},     # common grain centroid
        {"confidence": 0.77, "cx": 230, "cy": 145},   # thin grain centroid
    ]
    return cm, talc, talc_json, rows


def run():
    cm, talc, talc_json, rows = build_inputs()
    tmp = tempfile.mkdtemp()
    seg_csv = os.path.join(tmp, "segments.csv")
    with open(seg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["confidence", "cx", "cy"])
        w.writeheader()
        w.writerows(rows)

    payload, merged = M.build_payload(cm, talc, talc_json, seg_csv, f1=0.31)
    failures = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    # --- overlap rules (the crux) ---
    overlap = (cm > 0) & (talc > 0)
    check("sulfide wins on overlap", not np.any(merged[overlap] == TALC))
    check("overlap pixels kept as sulfide", np.all(merged[overlap] == cm[overlap]))
    check("talc == (talc_mask AND bg)", np.array_equal(merged == TALC, (talc > 0) & (cm == BG)))
    check("common grain intact", int((merged == COMMON).sum()) == int((cm == COMMON).sum()))
    check("thin grain intact", int((merged == THIN).sum()) == int((cm == THIN).sum()))

    # --- metrics ---
    total = H * W
    c, t, tl = ((merged == COMMON).sum(), (merged == THIN).sum(), (merged == TALC).sum())
    sulf = int(c + t)
    check("sulfideShare", abs(payload["sulfideShare"] - 100 * sulf / total) < TOL)
    check("talcShare", abs(payload["talcShare"] - 100 * int(tl) / total) < TOL)
    check("commonShare", abs(payload["commonShare"] - 100 * int(c) / sulf) < TOL)
    check("thinShare", abs(payload["thinShare"] - 100 * int(t) / sulf) < TOL)
    check("f1 passthrough", abs(payload["f1"] - 0.31) < 1e-9)
    check("imageWidth/Height", payload["imageWidth"] == W and payload["imageHeight"] == H)

    # --- polygons + confidence provenance ---
    pts = [c for s in payload["segments"] for ring in s["polygons"] for c in ring]
    check("polygon coords in [0,1]", all(0 <= x <= 1 and 0 <= y <= 1 for x, y in pts))
    talc_segs = [s for s in payload["segments"] if s["phase"] == "talc"]
    check("talc conf from kept segs only (not pruned 0.10)",
          bool(talc_segs) and all(0.35 <= s["confidence"] <= 0.85 for s in talc_segs))
    common = [s for s in payload["segments"] if s["phase"] == "common"]
    thin = [s for s in payload["segments"] if s["phase"] == "thin"]
    check("common conf ~0.88", bool(common) and abs(common[0]["confidence"] - 0.88) < 1e-3)
    check("thin conf ~0.77", bool(thin) and abs(thin[0]["confidence"] - 0.77) < 1e-3)

    # --- resolution mismatch ---
    talc_half = cv2.resize(talc, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
    p2, merged2 = M.build_payload(cm, talc_half, talc_json, seg_csv)
    check("res-mismatch: no talc over sulfide", not np.any(merged2[cm > 0] == TALC))
    check("res-mismatch: output at classmap resolution",
          p2["imageWidth"] == W and p2["imageHeight"] == H)

    print(f"\n  verdict={payload['verdict']!r} "
          f"talc={payload['talcShare']:.2f}% common={payload['commonShare']:.2f}% "
          f"thin={payload['thinShare']:.2f}%")

    # --- degenerate inputs (real pipelines can emit these) ---
    print("\n  edge cases:")

    # (a) no talc at all (talc_mask=None, talc_json=None, no csv)
    p_a, m_a = M.build_payload(cm, None, None, None)
    check("no-talc: zero talc share", p_a["talcShare"] == 0.0)
    check("no-talc: sulfide preserved", int((m_a > 0).sum()) == int((cm > 0).sum()))
    check("no-talc: sulfide fallback conf 0.9",
          all(abs(s["confidence"] - 0.9) < 1e-6 for s in p_a["segments"]))

    # (b) no sulfide grains, talc only -> talcShare>0, sulfideShare 0, no div-by-zero
    empty_cm = np.zeros((H, W), np.uint8)
    p_b, _ = M.build_payload(empty_cm, talc, talc_json, seg_csv)
    check("no-sulfide: sulfideShare 0", p_b["sulfideShare"] == 0.0)
    check("no-sulfide: commonShare/thinShare 0 (no div-by-zero)",
          p_b["commonShare"] == 0.0 and p_b["thinShare"] == 0.0)
    check("no-sulfide: talcShare > 0", p_b["talcShare"] > 0.0)

    # (c) talc ENTIRELY inside a sulfide grain -> override wipes it to 0 talc
    talc_inside = np.zeros((H, W), np.uint8)
    talc_inside[50:70, 50:100] = 255  # fully within the common grain [40:80,40:120]
    p_c, m_c = M.build_payload(cm, talc_inside, None, None)
    check("talc-inside-sulfide: overridden to 0 talc", p_c["talcShare"] == 0.0)
    check("talc-inside-sulfide: no TALC pixels", not np.any(m_c == TALC))

    # (d) fully talcose image -> verdict оталькованная
    big_talc = np.zeros((H, W), np.uint8)
    big_talc[:, :] = 255
    p_d, _ = M.build_payload(empty_cm, big_talc, None, None)
    check("fully-talc: verdict оталькованная", p_d["verdict"] == "оталькованная")
    check("fully-talc: talcShare ~100", abs(p_d["talcShare"] - 100.0) < TOL)

    return failures


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Cyrillic verdict on Windows console
    except Exception:
        pass
    fails = run()
    print("\nRESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}")
    sys.exit(1 if fails else 0)
