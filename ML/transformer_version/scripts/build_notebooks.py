"""Generate experiment.ipynb and inference_panorama.ipynb from cell sources.

Run:  python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path("notebooks")
NB_DIR.mkdir(exist_ok=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------- experiment
EXPERIMENT = [
    md(
        "# Ore phase segmentation + intergrowth classification — experiment\n\n"
        "Running this notebook top-to-bottom on a CUDA machine (V100) will:\n"
        "1. build 5-class labels (GMM pseudo-labels + 42 hand talc masks),\n"
        "2. train a SegFormer-B2 phase segmenter,\n"
        "3. report **talc-fraction MAE** and **intergrowth-type accuracy**,\n"
        "4. show colour overlays (green=обычные, red=тонкие, blue=тальк)."
    ),
    code(
        "import sys, pathlib, torch\n"
        "sys.path.insert(0, str(pathlib.Path.cwd().parent / 'src'))\n"
        "sys.path.insert(0, str(pathlib.Path.cwd() / 'src'))  # if run from repo root\n"
        "from orenet import constants as C\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "print('device:', device, '| classes:', C.CLASS_NAMES)\n\n"
        "# --- experiment config (tune here) ---\n"
        "EPOCHS = 12\n"
        "STEPS_PER_EPOCH = 250\n"
        "CROP = 512\n"
        "BATCH = 8\n"
        "MAX_TRAIN_IMAGES = 500   # cap non-talc images cached for the first run; None = all\n"
        "CACHE_DIR = pathlib.Path('data/derived/cache')"
    ),
    md("## 1. Index dataset and split by slide (no leakage)"),
    code(
        "from orenet.data_index import build_index\n"
        "from sklearn.model_selection import GroupShuffleSplit\n\n"
        "records = build_index()\n"
        "print('total records:', len(records))\n\n"
        "def gsplit(recs, test_size):\n"
        "    if len(recs) < 5:\n"
        "        return recs, recs\n"
        "    groups = [r.slide_id for r in recs]\n"
        "    tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=42).split(recs, groups=groups))\n"
        "    return [recs[i] for i in tr], [recs[i] for i in te]\n\n"
        "talc = [r for r in records if r.talc_label is not None]\n"
        "rest = [r for r in records if r.talc_label is None]\n"
        "assert talc, 'No talc labels found. Run first_labling_attempt/export_to_yoloseg.py first.'\n"
        "# honour the hand YOLO-seg train/val split so val talc is truly held out\n"
        "talc_tr = [r for r in talc if r.talc_split == 'train']\n"
        "talc_ev = [r for r in talc if r.talc_split == 'val']\n"
        "if not talc_ev:\n"
        "    talc_tr, talc_ev = gsplit(talc, 0.2)\n"
        "rest_tr, rest_ev = gsplit(rest, 0.15)\n"
        "if MAX_TRAIN_IMAGES is not None:\n"
        "    rest_tr = rest_tr[:MAX_TRAIN_IMAGES]\n"
        "train_recs = talc_tr + rest_tr\n"
        "eval_recs = talc_ev + rest_ev\n"
        "print(f'train={len(train_recs)} (talc {len(talc_tr)})  eval={len(eval_recs)} (talc {len(talc_ev)})')"
    ),
    md("## 2. Build the label cache (GMM pseudo-labels + talc masks) — runs once"),
    code(
        "from orenet.cache import build_cache\n"
        "train_items = build_cache(train_recs, CACHE_DIR / 'train')\n"
        "eval_items = build_cache(eval_recs, CACHE_DIR / 'eval')\n"
        "print('cached train:', len(train_items), '| eval:', len(eval_items),\n"
        "      '| talc in eval:', sum(it.has_talc for it in eval_items))"
    ),
    md("## 3. Data loaders (patch crops, rare-class sampling, talc copy-paste)"),
    code(
        "from torch.utils.data import DataLoader\n"
        "from orenet.dataset import PatchDataset\n\n"
        "train_ds = PatchDataset(train_items, crop=CROP, length=STEPS_PER_EPOCH * BATCH, augment=True)\n"
        "val_ds = PatchDataset(eval_items, crop=CROP, length=200, augment=False, copy_paste_p=0.0)\n"
        "train_loader = DataLoader(train_ds, batch_size=BATCH, num_workers=4, drop_last=True)\n"
        "val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=2)\n"
        "b = next(iter(train_loader))\n"
        "print('batch image', tuple(b['image'].shape), 'label', tuple(b['label'].shape))"
    ),
    md("## 4. Model — SegFormer-B2 (ADE20k pretrained encoder)"),
    code(
        "from orenet.model import SegmenterConfig, build_segmenter\n"
        "model = build_segmenter(SegmenterConfig(pretrained=True, n_classes=C.NUM_CLASSES))\n"
        "n_params = sum(p.numel() for p in model.parameters()) / 1e6\n"
        "print(f'{n_params:.1f}M params')"
    ),
    md("## 5. Train"),
    code(
        "from orenet.engine import TrainConfig, train\n"
        "cfg = TrainConfig(epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)\n"
        "history = train(model, train_loader, val_loader, device, cfg)"
    ),
    code(
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))\n"
        "ax[0].plot(history['loss']); ax[0].set_title('train loss'); ax[0].set_xlabel('epoch')\n"
        "ax[1].plot(history['miou'], label='mIoU'); ax[1].plot(history['talc_iou'], label='talc IoU')\n"
        "ax[1].set_title('val IoU'); ax[1].set_xlabel('epoch'); ax[1].legend(); plt.tight_layout(); plt.show()"
    ),
    md("## 6. Required metrics: talc-fraction MAE + intergrowth-type accuracy"),
    code(
        "from orenet.metrics import evaluate\n"
        "res = evaluate(model, eval_items, device, C.NUM_CLASSES)\n"
        "print(f'Talc-fraction MAE     : {res.talc_mae*100:.2f}%  (n={res.talc_n})')\n"
        "print(f'Intergrowth accuracy  : {res.intergrowth_acc*100:.1f}%  (n={res.intergrowth_n})')"
    ),
    md("## 7. Qualitative overlays (green=обычные, red=тонкие, blue=тальк)"),
    code(
        "import cv2, numpy as np, matplotlib.pyplot as plt\n"
        "from orenet.inference import predict\n"
        "from orenet.grains import extract_grains\n"
        "from orenet.viz import make_overlay\n\n"
        "samples = [it for it in eval_items if it.has_talc][:2] + \\\n"
        "          [it for it in eval_items if it.sort == 'fine'][:1] + \\\n"
        "          [it for it in eval_items if it.sort == 'normal'][:1]\n"
        "fig, axes = plt.subplots(len(samples), 2, figsize=(11, 4 * len(samples)))\n"
        "for row, it in enumerate(samples):\n"
        "    img = cv2.imread(str(it.image_path), cv2.IMREAD_COLOR)\n"
        "    cmap, _ = predict(model, img, device, C.NUM_CLASSES)\n"
        "    grains = extract_grains(cmap)\n"
        "    overlay = make_overlay(img, cmap, grains)\n"
        "    axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[row, 0].set_title(f'{it.sort}: original'); axes[row, 0].axis('off')\n"
        "    axes[row, 1].imshow(overlay); axes[row, 1].set_title('overlay'); axes[row, 1].axis('off')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## 8. Save checkpoint"),
    code(
        "ckpt = pathlib.Path('outputs'); ckpt.mkdir(exist_ok=True)\n"
        "torch.save(model.state_dict(), ckpt / 'segformer_phases.pt')\n"
        "print('saved', ckpt / 'segformer_phases.pt')"
    ),
]

# ---------------------------------------------------------------- panorama
PANORAMA = [
    md(
        "# Panorama inference\n\n"
        "Loads the trained SegFormer and runs tiled inference on a full panorama, "
        "then renders the intergrowth overlay and prints the ore-sort verdict."
    ),
    code(
        "import sys, pathlib, cv2, torch, numpy as np, matplotlib.pyplot as plt\n"
        "sys.path.insert(0, str(pathlib.Path.cwd().parent / 'src'))\n"
        "sys.path.insert(0, str(pathlib.Path.cwd() / 'src'))\n"
        "from orenet import constants as C\n"
        "from orenet.model import SegmenterConfig, build_segmenter\n"
        "from orenet.inference import predict\n"
        "from orenet.grains import extract_grains\n"
        "from orenet.classify import classify_image\n"
        "from orenet.viz import make_overlay, colorize_uncertainty\n\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "model = build_segmenter(SegmenterConfig(pretrained=False, n_classes=C.NUM_CLASSES))\n"
        "model.load_state_dict(torch.load('outputs/segformer_phases.pt', map_location=device, weights_only=True))\n"
        "model.to(device).eval()\n"
        "print('loaded model on', device)"
    ),
    code(
        "pano_paths = sorted((C.PANORAMS).glob('*.jpg'))\n"
        "print('panoramas:', [p.name for p in pano_paths])\n"
        "pano = cv2.imread(str(pano_paths[0]), cv2.IMREAD_COLOR)\n"
        "# downscale very large panoramas for a quick demo pass\n"
        "if max(pano.shape[:2]) > 8000:\n"
        "    scale = 8000 / max(pano.shape[:2])\n"
        "    pano = cv2.resize(pano, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)\n"
        "print('panorama shape:', pano.shape)"
    ),
    code(
        "import time\n"
        "t0 = time.time()\n"
        "class_map, uncertainty = predict(model, pano, device, C.NUM_CLASSES, tile=1024, overlap=0.25)\n"
        "print(f'inference {time.time()-t0:.1f}s, class_map {class_map.shape}')\n"
        "grains = extract_grains(class_map)\n"
        "res = classify_image(class_map, grains)\n"
        "print(f\"VERDICT: {res.ore_class} | talc={res.talc_frac*100:.1f}% \"\n"
        "      f\"fine={res.fine_frac*100:.1f}% sulfide={res.sulfide_frac*100:.1f}% grains={len(grains)}\")"
    ),
    code(
        "overlay = make_overlay(pano, class_map, grains)\n"
        "fig, ax = plt.subplots(1, 3, figsize=(18, 7))\n"
        "ax[0].imshow(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)); ax[0].set_title('panorama'); ax[0].axis('off')\n"
        "ax[1].imshow(overlay); ax[1].set_title('overlay (green/red/blue)'); ax[1].axis('off')\n"
        "ax[2].imshow(colorize_uncertainty(uncertainty)); ax[2].set_title('uncertainty'); ax[2].axis('off')\n"
        "plt.tight_layout(); plt.show()\n"
        "cv2.imwrite('outputs/panorama_overlay.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))"
    ),
]


def main() -> None:
    (NB_DIR / "experiment.ipynb").write_text(json.dumps(notebook(EXPERIMENT), indent=1))
    (NB_DIR / "inference_panorama.ipynb").write_text(json.dumps(notebook(PANORAMA), indent=1))
    print("wrote notebooks/experiment.ipynb and notebooks/inference_panorama.ipynb")


if __name__ == "__main__":
    main()
