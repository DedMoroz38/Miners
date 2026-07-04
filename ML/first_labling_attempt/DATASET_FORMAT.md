# Talc segmentation dataset — format (YOLO-seg)

Output of `export_to_yoloseg.py`, ready for Ultralytics `yolo segment`. Single
class, **talc = index 0**; everything not outlined is background.

## Current snapshot

Verified live from the running Label Studio project (all checks passed):

| | |
|---|---|
| Images labeled | **42 / 42** (0 unlabeled) |
| Talc polygons total | **205** |
| Train / val split | **34 / 8** (val-frac 0.2, seed 42) |
| Image ↔ label parity | 42 / 42 ✓ |
| Invalid lines | 0 |
| Points per polygon | 4 – 238 |

## Directory layout

```
yolo_seg/
├── images/
│   ├── train/<stem>.jpg      34 images
│   └── val/<stem>.jpg         8 images
├── labels/
│   ├── train/<stem>.txt      one .txt per image (same stem)
│   └── val/<stem>.txt
├── data.yaml                 dataset descriptor
└── filename_map.json         sanitized stem → original filename
```

Ultralytics pairs an image with its label by **identical stem** and by swapping
`images/` → `labels/` in the path. Every image has a matching `.txt` (empty file =
no talc in that image).

## Label file format

Each `labels/**/<stem>.txt` holds **one line per talc polygon**:

```
<class> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

- `<class>` is always `0` (talc).
- Coordinates are **normalized to [0, 1]**: `x/image_width`, `y/image_height`,
  **top-left origin**. Not pixels, not percentages.
- A polygon needs ≥ 3 points (≥ 6 numbers after the class). Point count is
  variable per polygon (here 4–238).
- Multiple polygons in one image → multiple lines in that image's file.

Example (one 14-point polygon, truncated):

```
0 0.368119 0.319572 0.305046 0.333333 0.285550 0.342508 … 0.351200 0.300400
```

## `data.yaml`

```yaml
path: <abs>/yolo_seg     # dataset root (absolute)
train: images/train      # relative to path
val: images/val
names:
  0: talc
```

## `filename_map.json`

Maps each sanitized output stem back to its original Label Studio filename, e.g.:

```json
{ "2550374-2_10.jpg": "to_label/2550374-2 10х.JPG" }
```

Filenames are sanitized (spaces + Cyrillic → ascii, lowercased `.jpg`) because
Ultralytics is fragile with spaces/non-ascii in paths. Image and label stems are
kept identical after sanitizing; collisions get a `_1`, `_2` suffix.

## Provenance (how it maps from Label Studio)

Label Studio stores each polygon as `value.points` — a list of `[x, y]` pairs in
**percent (0–100)** of the image. The converter divides by 100 (→ [0, 1]), clamps,
flattens to `x1 y1 x2 y2 …`, and prepends class `0`. One image with N polygons
becomes N lines. A verification overlay (`--visualize`) confirmed the exported
coordinates redraw exactly onto the painted regions (no flip/scale error).

## Regenerate / train

```bash
# refresh from the live project (pulls all annotators' work, no manual export):
python export_to_yoloseg.py --from-api

# then, later, in a training env with ultralytics installed:
yolo segment train data=yolo_seg/data.yaml model=yolo11n-seg.pt imgsz=640 epochs=100
```

The output dir is rebuilt cleanly on each run. `--val-frac` / `--seed` control the
split; `--include-empty` would add background-only images (not needed here — every
image has talc).
