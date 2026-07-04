#!/usr/bin/env python3
"""Convert a Label Studio polygon export into a YOLO-seg dataset.

Talc-only, single class (index 0). Reads a Label Studio JSON export (polygon
`points` are percentages of the image) and writes an Ultralytics-ready dataset:

    yolo_seg/
      images/train/*.jpg   labels/train/<stem>.txt
      images/val/*.jpg     labels/val/<stem>.txt
      data.yaml
      filename_map.json

Each label line is one talc polygon:  0 x1 y1 x2 y2 ... xn yn   (coords 0-1).

Usage:
    # 1) In the UI: Export -> JSON  (save to exports/annotations.json), then:
    python export_to_yoloseg.py
    # or pull the export live from the running server:
    python export_to_yoloseg.py --from-api
    # optional sanity overlays:
    python export_to_yoloseg.py --visualize 3

Stdlib only (Pillow is used only for --visualize, and it's already installed).
"""
import argparse
import json
import os
import random
import re
import shutil
import sqlite3
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES_ROOT = os.path.join(HERE, "images")            # LOCAL_FILES_DOCUMENT_ROOT
DB = os.path.join(HERE, ".label_studio", "label_studio.sqlite3")
DEFAULT_INPUT = os.path.join(HERE, "exports", "annotations.json")
DEFAULT_OUT = os.path.join(HERE, "yolo_seg")
CLASS_NAME = "talc"


# --- input loading -----------------------------------------------------------
def db_token():
    if not os.path.exists(DB):
        return None
    con = sqlite3.connect(DB)
    try:
        # LS 1.23 disables legacy `Token` auth by default; re-enable it (idempotent).
        try:
            con.execute("update jwt_auth_jwtsettings set legacy_api_tokens_enabled=1")
            con.commit()
        except sqlite3.OperationalError:
            pass
        row = con.execute(
            "select key from authtoken_token order by created limit 1"
        ).fetchone()
    finally:
        con.close()
    return row[0] if row else None


def load_from_api(url, project, token):
    token = token or db_token()
    if not token:
        raise SystemExit("No API token for --from-api. Log in once, or pass --token.")
    # download_all_tasks=true so un-annotated tasks come through too (needed for
    # --include-empty); without it LS exports only annotated tasks.
    req = urllib.request.Request(
        f"{url}/api/projects/{project}/export?exportType=JSON&download_all_tasks=true",
        headers={"Authorization": f"Token {token}"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Export API failed: HTTP {e.code} {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Could not reach {url} — is ./run.sh running? ({e})")


def load_tasks(args):
    if args.from_api:
        return load_from_api(args.url, args.project, args.token)
    if not os.path.exists(args.input):
        raise SystemExit(
            f"Export file not found: {args.input}\n"
            "Export from the UI (Export -> JSON) or use --from-api."
        )
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("tasks", data)


# --- parsing -----------------------------------------------------------------
def image_relpath(task):
    """Resolve the file path (relative to images/) from the task's image URL."""
    data = task.get("data", task)
    url = data.get("image") or data.get("$image") or ""
    # Local files look like: /data/local-files/?d=to_label/NAME.JPG
    if "?d=" in url:
        rel = url.split("?d=", 1)[1]
    else:
        rel = url  # fall back to whatever is there
    rel = urllib.parse.unquote(rel.split("?")[0]).lstrip("/")
    return rel


def polygons_of(task):
    """Yield lists of [x, y] points (percentages) for every polygon in a task."""
    # Full JSON export: task['annotations'][i]['result'][j]
    anns = task.get("annotations")
    if anns:
        for ann in anns:
            if ann.get("was_cancelled"):
                continue
            for r in ann.get("result", []):
                if r.get("type") == "polygonlabels":
                    pts = r.get("value", {}).get("points")
                    if pts:
                        yield pts
            return  # only the first non-cancelled annotation
    # JSON-MIN export: task['zone'] = [ {points, polygonlabels, ...}, ... ]
    for v in task.values():
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and "points" in item and "polygonlabels" in item:
                    yield item["points"]


def poly_to_line(points):
    """One YOLO-seg line for a polygon: '0 x1 y1 ...' with coords clamped to [0,1]."""
    if not points or len(points) < 3:
        return None
    coords = []
    for x, y in points:
        coords.append(f"{min(max(x / 100.0, 0.0), 1.0):.6f}")
        coords.append(f"{min(max(y / 100.0, 0.0), 1.0):.6f}")
    return "0 " + " ".join(coords)


# --- filename sanitizing -----------------------------------------------------
def slugify(name):
    stem, ext = os.path.splitext(os.path.basename(name))
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "img"
    ext = ext.lower() or ".jpg"
    if ext in (".jpeg", ".jpe"):
        ext = ".jpg"
    return stem + ext


# --- writing -----------------------------------------------------------------
def fresh_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Label Studio JSON export")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-empty", action="store_true",
                    help="include images with no polygons as background negatives")
    ap.add_argument("--from-api", action="store_true", help="pull export from the server")
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--project", type=int, default=1)
    ap.add_argument("--token", default=None)
    ap.add_argument("--visualize", type=int, default=0, metavar="N",
                    help="render N overlays into <out>/_preview/ (needs Pillow)")
    args = ap.parse_args()

    tasks = load_tasks(args)

    samples, name_map, n_poly, skipped_missing = [], {}, 0, 0
    used_stems = set()
    for task in tasks:
        rel = image_relpath(task)
        src = os.path.join(IMAGES_ROOT, rel)
        if not os.path.exists(src):
            skipped_missing += 1
            continue
        lines = [ln for pts in polygons_of(task) if (ln := poly_to_line(pts))]
        if not lines and not args.include_empty:
            continue
        # unique, ascii-safe stem shared by image + label
        stem = slugify(rel)
        base, ext = os.path.splitext(stem)
        i = 1
        while stem in used_stems:
            stem = f"{base}_{i}{ext}"
            i += 1
        used_stems.add(stem)
        name_map[stem] = rel
        n_poly += len(lines)
        samples.append((stem, src, lines))

    if not samples:
        raise SystemExit(
            "No usable samples. Have you labeled any talc polygons yet? "
            "(Use --include-empty to export background-only images.)"
        )

    # split
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    n_val = int(round(len(samples) * args.val_frac))
    val = set(id(s) for s in samples[:n_val])

    fresh_dir(args.out)
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split))
        os.makedirs(os.path.join(args.out, "labels", split))

    n_train = 0
    for s in samples:
        stem, src, lines = s
        split = "val" if id(s) in val else "train"
        n_train += split == "train"
        shutil.copy2(src, os.path.join(args.out, "images", split, stem))
        label_path = os.path.join(args.out, "labels", split, os.path.splitext(stem)[0] + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(
            f"path: {os.path.abspath(args.out)}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n  0: {CLASS_NAME}\n"
        )
    with open(os.path.join(args.out, "filename_map.json"), "w", encoding="utf-8") as f:
        json.dump(name_map, f, ensure_ascii=False, indent=2)

    print(f"Tasks read           : {len(tasks)}")
    print(f"Usable images        : {len(samples)}  (train {n_train} / val {len(samples)-n_train})")
    print(f"Talc polygons        : {n_poly}")
    if skipped_missing:
        print(f"Skipped (no source)  : {skipped_missing}")
    print(f"Dataset written to   : {os.path.abspath(args.out)}")
    print(f"Train a model with   : yolo segment train data={os.path.join(args.out,'data.yaml')} model=yolo11n-seg.pt")

    if args.visualize:
        visualize(args.out, args.visualize)


def visualize(out, n):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("(visualize skipped: Pillow not installed)")
        return
    prev = os.path.join(out, "_preview")
    os.makedirs(prev, exist_ok=True)
    img_dir = os.path.join(out, "images", "train")
    lbl_dir = os.path.join(out, "labels", "train")
    count = 0
    for name in sorted(os.listdir(img_dir)):
        if count >= n:
            break
        lbl = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
        if not os.path.exists(lbl):
            continue
        im = Image.open(os.path.join(img_dir, name)).convert("RGB")
        W, H = im.size
        dr = ImageDraw.Draw(im)
        with open(lbl) as f:
            for line in f:
                nums = line.split()
                if len(nums) < 7:
                    continue
                xy = [float(v) for v in nums[1:]]
                pts = [(xy[i] * W, xy[i + 1] * H) for i in range(0, len(xy), 2)]
                dr.line(pts + [pts[0]], fill=(31, 111, 255), width=4)
        im.save(os.path.join(prev, name))
        count += 1
    print(f"Wrote {count} overlay(s) to {prev}")


if __name__ == "__main__":
    main()
