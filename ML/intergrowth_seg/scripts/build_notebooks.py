"""Generate notebooks/train_regions.ipynb — train, report F1, draw region maps.

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
import pathlib

NB_DIR = pathlib.Path(__file__).resolve().parent.parent / "notebooks"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# Intergrowth regions — train (V100), report F1, draw region maps

Localises **обычные (green) / тонкие (red)** intergrowth regions on each photo and
decides рядовая vs труднообогатимая. Honest metric = **image-level F1** (folder sort
is the only ground truth). Levers: ConvNeXt-Small (ImageNet) + strong aug +
balanced sampling + 5-fold ensemble + TTA + threshold tuning.

Reuses the transformer_version preprocessed cache automatically when present.
"""),
    code("""
import os, sys, pathlib, torch
p = pathlib.Path.cwd()
if p.name == 'notebooks':
    os.chdir(p.parent)
sys.path.insert(0, 'src')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cwd:', pathlib.Path.cwd(), '| device:', device)

from iseg.sources import build_items, split_items
items = build_items(prefer_cache=True)
cached = sum(it.cached for it in items)
from collections import Counter
print('images:', len(items), '| cached-preproc reused:', cached,
      '| by class:', dict(Counter(it.label for it in items)))
trainval, holdout = split_items(items, test_size=0.15, seed=42)
print('trainval:', len(trainval), '| holdout:', len(holdout))
assert items, 'No images. Symlink data (see README) or edit intergrowth/constants.DATA_ROOTS.'
"""),
    md("## Train 5-fold ensemble (OOF F1 = honest held-out estimate)"),
    code("""
from iseg.train import train_kfold, TrainConfig, evaluate
cfg = TrainConfig(folds=5, epochs=12, backbone='convnext_small', batch=24)
if device.type != 'cuda':
    cfg = TrainConfig(folds=2, epochs=1, backbone='convnext_tiny', batch=4, num_workers=0)  # CPU smoke
models, oof, oof_p, oof_y = train_kfold(trainval, device, cfg)
print(f'\\n>>> OOF F1 = {oof.f1*100:.1f}%  (P {oof.precision*100:.1f} / R {oof.recall*100:.1f}) @thr={oof.threshold:.2f}')
"""),
    md("## Held-out F1 (ensemble + TTA) at the tuned threshold"),
    code("""
res, p_hold, y_hold = evaluate(models, holdout, device, thr=oof.threshold, tile=cfg.tile)
print(f'HOLD-OUT F1 = {res.f1*100:.1f}%   P={res.precision*100:.1f}  R={res.recall*100:.1f}')
print(f'  tp={res.tp} fp={res.fp} fn={res.fn} tn={res.tn}')
if res.f1 >= 0.90:
    print('\\n✅ target F1>=90 reached on hold-out')
else:
    print('\\n⚠️ below 90 — bump epochs/tiles_per_image, try convnext_base, or add folds')
"""),
    md("## Region maps — красим предсказанные области срастаний"),
    code("""
import matplotlib.pyplot as plt, cv2
from iseg.infer import predict_image, region_overlay
show = holdout[:6]
fig, ax = plt.subplots(len(show), 2, figsize=(11, 4*len(show)))
for r, it in enumerate(show):
    lbl, pf, coords, per_tile, hw = predict_image(models, it.path, device, tile=cfg.tile)
    ov = region_overlay(it.path, coords, per_tile, hw, tile=cfg.tile)
    raw = cv2.cvtColor(cv2.imread(str(it.path)), cv2.COLOR_BGR2RGB)
    ax[r,0].imshow(raw); ax[r,0].set_title(f'true={\"fine\" if it.label else \"normal\"}'); ax[r,0].axis('off')
    ax[r,1].imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
    ax[r,1].set_title(f'pred p_fine={pf:.2f} (red=тонкие, green=обычные)'); ax[r,1].axis('off')
plt.tight_layout(); plt.show()
"""),
    md("## Save the ensemble"),
    code("""
import torch, pathlib
out = pathlib.Path('outputs'); out.mkdir(exist_ok=True)
torch.save({'state_dicts': [m.state_dict() for m in models],
            'threshold': oof.threshold, 'cfg': cfg.__dict__}, out/'iseg_ensemble.pt')
print('saved', out/'iseg_ensemble.pt')
"""),
]


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = NB_DIR / "train_regions.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
