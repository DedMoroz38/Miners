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
    md("## Train (plain train/val, no k-fold) — metrics printed every epoch"),
    code("""
from iseg.train import train_single, TrainConfig, evaluate
# single model, fast to iterate. Watch val_acc / val_F1 climb per epoch.
cfg = TrainConfig(epochs=15, backbone='convnext_small', batch=16, tile=384)
if device.type != 'cuda':
    cfg = TrainConfig(epochs=1, backbone='convnext_tiny', batch=4, tile=224, num_workers=0)  # CPU smoke
model, history, val_p, val_y = train_single(trainval, holdout, device, cfg)
"""),
    md("## Final hold-out F1 (single model + TTA + tuned threshold)"),
    code("""
from iseg.metrics import best_threshold
# tune threshold on the val predictions, report TTA-refined F1
tuned = best_threshold(val_p, val_y)
res, p_hold, y_hold = evaluate([model], holdout, device, thr=tuned.threshold, tile=cfg.tile)
print(f'HOLD-OUT F1 = {res.f1*100:.1f}%   P={res.precision*100:.1f}  R={res.recall*100:.1f}  @thr={tuned.threshold:.2f}')
print(f'  acc={(res.tp+res.tn)/max(len(y_hold),1)*100:.1f}%  tp={res.tp} fp={res.fp} fn={res.fn} tn={res.tn}')
if res.f1 >= 0.90:
    print('\\n✅ target F1>=90 reached on hold-out')
else:
    print('\\n⚠️ below 90 — bump epochs/tiles_per_image, try convnext_base, or add 5-fold (train_kfold)')
"""),
    md("## Metric curves per epoch"),
    code("""
import matplotlib.pyplot as plt
ep = range(1, len(history)+1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ep, [h['loss'] for h in history], 'o-'); ax[0].set_title('train loss'); ax[0].set_xlabel('epoch')
ax[1].plot(ep, [h['val_acc'] for h in history], 'o-', label='val acc')
ax[1].plot(ep, [h['val_f1'] for h in history], 's-', label='val F1@0.5')
ax[1].plot(ep, [h['val_f1_best'] for h in history], '^-', label='val F1 (tuned thr)')
ax[1].axhline(0.90, ls='--', c='gray'); ax[1].set_ylim(0,1); ax[1].legend(); ax[1].set_title('val metrics'); ax[1].set_xlabel('epoch')
plt.tight_layout(); plt.show()
"""),
    md("## Region maps — красим предсказанные области срастаний"),
    code("""
import matplotlib.pyplot as plt, cv2
from iseg.infer import predict_image, region_overlay
show = holdout[:6]
fig, ax = plt.subplots(len(show), 2, figsize=(11, 4*len(show)))
for r, it in enumerate(show):
    lbl, pf, coords, per_tile, hw = predict_image([model], it.path, device, tile=cfg.tile)
    ov = region_overlay(it.path, coords, per_tile, hw, tile=cfg.tile)
    raw = cv2.cvtColor(cv2.imread(str(it.path)), cv2.COLOR_BGR2RGB)
    ax[r,0].imshow(raw); ax[r,0].set_title(f'true={\"fine\" if it.label else \"normal\"}'); ax[r,0].axis('off')
    ax[r,1].imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
    ax[r,1].set_title(f'pred p_fine={pf:.2f} (red=тонкие, green=обычные)'); ax[r,1].axis('off')
plt.tight_layout(); plt.show()
"""),
    md("## Save the model"),
    code("""
import torch, pathlib
out = pathlib.Path('outputs'); out.mkdir(exist_ok=True)
torch.save({'state_dict': model.state_dict(),
            'threshold': tuned.threshold, 'cfg': cfg.__dict__}, out/'iseg_model.pt')
print('saved', out/'iseg_model.pt')
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
