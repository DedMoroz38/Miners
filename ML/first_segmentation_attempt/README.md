# Talc instance segmentation — Ultralytics YOLO-seg

Fine-tunes a YOLO-seg model to segment **talc** regions in ore thin-sections,
using the hand-labelled polygons from
[`../first_labling_attempt/yolo_seg`](../first_labling_attempt).

## Why YOLO-seg (the model decision)

The labels are already **native YOLO-seg polygons**, so this is the lowest-risk,
highest-leverage choice:

- **Zero conversion** — trains directly on `yolo_seg/{images,labels}`.
- **COCO-pretrained transfer** — decisive with only **42 labelled images / 205
  talc polygons**; the backbone already knows edges/shapes.
- **Mask mAP for free** — Ultralytics reports mask precision/recall/mAP out of the box.

Two model sizes, one per environment (swap with `--model`):

| Environment | Model | Why |
|---|---|---|
| Local test / CPU / MPS | `yolo11n-seg` (default) | nano, fast, verifies the pipeline |
| Remote GPU (V100) | `yolo11l-seg` | larger backbone, better masks on the real run |

(SegFormer semantic segmentation lives separately in `../transformer_version`; it
needs the full multi-class dataset + GMM pseudo-labels. For a focused "where is
the talc" task, YOLO-seg on these polygons is the cleaner fit.)

## Setup

```bash
cd ML/first_segmentation_attempt
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # pulls torch + ultralytics
```

The dataset comes from `../first_labling_attempt/yolo_seg` (labels are committed;
**images are git-ignored**). If the images aren't present (fresh clone),
regenerate the whole dataset from Label Studio:

```bash
cd ../first_labling_attempt && python export_to_yoloseg.py --from-api
```

## Train

```bash
python train.py --smoke        # quick CPU sanity run (3 epochs, imgsz 320)
python train.py                # local, yolo11n-seg, ≤300 epochs w/ early stop
# remote GPU:
python train.py --model yolo11l-seg.pt --imgsz 768 --batch 16 --device 0
```

The first 100-epoch baseline overfit (train loss fell while val loss rose;
mask mAP50 flat after epoch 60), so `train.py` bakes in a small-data recipe
(`SMALL_DATA_OVERRIDES`): gentle rotation (`degrees=90`) + vertical flip
(micrographs have no canonical orientation), cosine LR, `close_mosaic=30`,
and early stopping (`--patience 80`). A first, harsher recipe
(`degrees=180` + `copy_paste=0.5`) destabilized training on 34 images —
val metrics whipsawed and early-stop picked a degenerate epoch-8 "best" —
hence the gentle settings. `--freeze 11` freezes the backbone if the bigger
models still overfit.

Runs get per-recipe names so they never overwrite each other's `best.pt`:
`--baseline` → `runs/talc_seg/`, small-data recipe → `runs/talc_seg_v2/`,
`--smoke` → `runs/smoke/`. Override with `--name`.

`train.py` writes a portable `data.yaml` each run (absolute path to `yolo_seg`, so
the machine-specific committed path never bites). Outputs go to
`runs/<name>/` (git-ignored); best weights at `runs/<name>/weights/best.pt`.

## Evaluate

```bash
python evaluate.py                       # mask + box mAP + per-image overlays -> runs/predict/
python evaluate.py --conf 0.1            # lower threshold (weak/early checkpoints show 0 dets at 0.25)
python evaluate.py --no-predict          # metrics only, skip overlays
python evaluate.py --weights runs/talc_seg/weights/best.pt --device 0
```

Overlays are saved every run by default (one image per val sample, mask + box
drawn) to `runs/predict/`. If a checkpoint has zero detections at `--conf`
(common for early/undertrained checkpoints — `val()` sweeps confidence near 0
to build the mAP curve, which a fixed threshold does not), evaluate.py
auto-retries at conf=0.05 and saves debug-only overlays to
`runs/predict_lowconf/` so you can see the raw, unfiltered output.

The 8-image val split moves ~±12% per image, so for any recipe comparison use
grouped k-fold CV instead (every image serves as val once; shots of the same
drill specimen never straddle train/val):

```bash
python kfold.py                                        # 5-fold, yolo11n-seg
python kfold.py --model yolo11l-seg.pt --imgsz 768 --device 0   # remote GPU
```

## Status

Pipeline verified end-to-end on CPU (Apple M1 Pro):

- **train.py --smoke** → trains 3 epochs, saves `best.pt`, no errors.
- **evaluate.py** → runs `val`, prints Box/Mask mAP, no errors.

The smoke metrics are ~0 (3 epochs, nano) — that only proves the plumbing. Run the
full `--device 0` training on the V100 for real numbers. Val is only 8 images, so
treat mask mAP as indicative; k-fold CV is the honest next step for a firm number.

## Files

- `train.py` — fine-tune YOLO-seg (small-data recipe baked in); `--smoke` for CPU verification, `--device 0` for GPU
- `evaluate.py` — mask/box mAP on val + optional overlays
- `kfold.py` — grouped 5-fold CV; the trustworthy number for recipe comparisons
- `requirements.txt` — `ultralytics`
- `data.yaml` — generated at runtime (git-ignored)
