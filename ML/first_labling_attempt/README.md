# Talc-zone segmentation labeling — ore thin-sections (Label Studio)

Outline the exact **talc region(s)** in each thin-section image with polygons, and
export them as a **YOLO-seg dataset** so a segmentation model (Ultralytics
`yolo segment`) can later predict *where* the talc is in a new image.

Single class: **talc** (index `0`). Anything you don't outline is background.

```
first_labling_attempt/
├── images/
│   └── to_label/          ← PUT IMAGES TO LABEL HERE (.jpg/.png/…)
├── label_config.xml       polygon labeling interface (talc)
├── apply_label_config.py  push label_config.xml into the running project (API)
├── export_to_yoloseg.py   Label Studio JSON  →  YOLO-seg dataset
├── run.sh                 starts Label Studio, self-contained in this folder
├── requirements.txt       label-studio
├── yolo_seg/              generated dataset (git-ignored)
└── .label_studio/         local DB + media, created on first run (git-ignored)
```

## 1. Install (once)

```bash
cd ML/first_labling_attempt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Put images in the folder

Copy images into **`images/to_label/`** (a subfolder is required — Label Studio
refuses to serve its document root directly):

```bash
cp /path/to/new/photos/*.JPG images/to_label/
```

## 3. Start Label Studio

```bash
./run.sh
```

Serves **http://localhost:8080** with preset login **`admin@miners.local` /
`MinersLabel2026`** (see `run.sh`). All state lives under `.label_studio/`.

## 4. Create the project + apply the polygon config

1. **Create Project** (e.g. `talc-seg`).
2. Apply the labeling interface — easiest is the helper (project id 1, server up):
   ```bash
   python apply_label_config.py
   ```
   *Fallback:* Settings → Labeling Interface → **Code** → paste `label_config.xml`
   → **Save**.

## 5. Connect the images folder (Local Storage)

1. **Settings → Cloud Storage → Add Source Storage** → type **Local files**.
2. **Absolute local path**: the `to_label` path `run.sh` printed
   (`.../images/to_label`) — must be a **subfolder** of the document root.
3. Turn **ON** “Treat every bucket object as a source file”.
4. **Check Connection → Save → Sync Storage**.

Re-**Sync** whenever you add files to `images/to_label/`.

> If you synced *before* applying the config, tasks may carry a `$undefined$` data
> key and images won't render. Fix: apply the config (step 4), then delete the
> tasks and Sync again so they're recreated with the `image` key.

## 6. Label the talc zones

1. Open the project → **Label All Tasks**.
2. Select the **talc** label, then **click point-by-point around a talc zone** and
   **close the polygon** by clicking the first point again.
3. Draw **one polygon per talc region** — multiple per image is fine.
4. **Submit** (Cmd/Ctrl+Enter) → next image. Zoom/pan to trace boundaries.

Prefer painting to clicking? Swap `PolygonLabels` → `BrushLabels` in
`label_config.xml` (heavier, but pixel-perfect) and re-apply.

## 7. Export → YOLO-seg dataset

Export the annotations, then convert:

```bash
# In the UI: Export -> JSON  →  save as exports/annotations.json, then:
python export_to_yoloseg.py

# …or pull the export straight from the running server (no manual download):
python export_to_yoloseg.py --from-api

# optional sanity overlays into yolo_seg/_preview/
python export_to_yoloseg.py --from-api --visualize 3
```

This writes:

```
yolo_seg/
├── images/train/*.jpg   labels/train/<stem>.txt
├── images/val/*.jpg     labels/val/<stem>.txt
├── data.yaml            (names: {0: talc})
└── filename_map.json    sanitized-stem → original filename
```

Each label line is one talc polygon: `0 x1 y1 x2 y2 … xn yn` (coords normalized
0–1). Useful flags: `--val-frac 0.2`, `--seed 42`, `--include-empty` (write
un-labeled images as background negatives).

The result is directly trainable later:

```bash
yolo segment train data=yolo_seg/data.yaml model=yolo11n-seg.pt
```

*(Training lives in a future `first_segmentation_attempt/` folder — not built yet.)*

---

### Troubleshooting

- **Sync fails / 0 tasks** → enable “Treat every bucket object as a source file”
  on the storage (otherwise LS tries to parse JPEGs as JSON).
- **Broken thumbnails** → start via `./run.sh` (sets the local-files env vars) and
  use a **subfolder** of the served root as the storage path.
- **`export_to_yoloseg.py` says “No usable samples”** → you haven't labeled any
  polygons yet (or export was empty). Label a few, or pass `--include-empty`.
- **`label-studio: command not found`** → `source .venv/bin/activate`.
- **Reset everything** → delete `.label_studio/` (wipes projects/annotations, not
  your `images/`).
