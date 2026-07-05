"""Phase-0 audit: triptychs (original | expert blue line | built talc mask) so a
human can confirm/repair the frozen GT (spec §3.1).

    python scripts/make_audit.py                 # -> data_build/audit/*.jpg

Requires build_dataset.py to have run (reads data_build/labels + images) and the
original blue-line copies in paths.talc_zones. Emits one triptych per talc image
plus a printed table of the built talc fraction per image.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np

from talc_quant.config import load_config
from talc_quant.constants import CLASS_TALC, VALID_CLASSES


def _talc_line_overlay(zones_dir: Path, orig_name: str) -> np.ndarray | None:
    p = zones_dir / orig_name
    return cv2.imread(str(p)) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--height", type=int, default=520)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(overrides=args.overrides)
    build = cfg.paths.build_dir
    audit = build / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((build / "manifest.json").read_text())

    fmap_path = cfg.paths.yolo_seg / "filename_map.json"
    fmap = json.loads(fmap_path.read_text()) if fmap_path.exists() else {}

    print(f"{'stem':28s} {'talc%':>7s}")
    for rec in manifest:
        if not rec["has_talc"]:
            continue
        img = cv2.imread(str(build / rec["image"]))
        label = cv2.imread(str(build / rec["label"]), cv2.IMREAD_GRAYSCALE)
        if img is None or label is None:
            continue
        valid = np.isin(label, VALID_CLASSES)
        talc = label == CLASS_TALC
        frac = float(talc.sum()) / max(int(valid.sum()), 1)

        mask_vis = img.copy()
        mask_vis[talc] = (0.5 * mask_vis[talc] +
                          0.5 * np.array([255, 80, 0])).astype(np.uint8)  # blue

        orig_name = fmap.get(rec["stem"] + ".jpg", "").replace("to_label/", "")
        line = _talc_line_overlay(cfg.paths.talc_zones, orig_name)
        panels = [img]
        panels.append(line if line is not None else np.zeros_like(img))
        panels.append(mask_vis)

        H = args.height
        resized = [cv2.resize(p, (int(p.shape[1] * H / p.shape[0]), H)) for p in panels]
        trip = np.hstack(resized)
        cv2.imwrite(str(audit / f"{rec['stem']}.jpg"), trip)
        print(f"{rec['stem']:28s} {frac * 100:7.2f}")

    print(f"\ntriptychs -> {audit}")


if __name__ == "__main__":
    main()
