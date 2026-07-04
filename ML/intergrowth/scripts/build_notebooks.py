"""Generate notebooks/compare.ipynb — runs BOTH methods on a held-out test split.

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
NB_DIR = HERE.parent / "notebooks"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# Intergrowth type: обычные vs тонкие срастания — two methods compared

Runs on a slide-grouped train/test split (no leakage):
- **Method 1** — minerallurgical indices + LDA (Pérez-Barnuevo et al. 2013, reimplemented).
- **Method 2** — ImageNet-pretrained CNN texture classifier (Pérez-Barnuevo 2018 / MISIS 2025 paradigm).

Both trained on your folder labels (рядовая=NORMAL, труднообогатимая=FINE). Talc folders excluded.
"""),
    code("""
import os, sys, pathlib, torch
# run from the intergrowth/ root so relative data paths resolve
p = pathlib.Path.cwd()
if p.name == 'notebooks':
    os.chdir(p.parent)
sys.path.insert(0, 'src')
print('cwd:', pathlib.Path.cwd())

from intergrowth.data import build_samples, group_split
from intergrowth.constants import CLASS_NAMES
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

samples = build_samples()
train, test = group_split(samples, test_size=0.2, seed=42)
from collections import Counter
print('total:', len(samples), '| classes:', {CLASS_NAMES[k]: v for k, v in Counter(s.label for s in samples).items()})
print('train:', len(train), '| test:', len(test))
assert samples, 'No images found. Symlink data or edit intergrowth/constants.DATA_ROOTS.'
"""),
    md("## Method 1 — minerallurgical indices + LDA (Pérez-Barnuevo 2013)"),
    code("""
from intergrowth.lda_model import build_feature_matrix, LdaClassifier
from intergrowth.evaluate import eval_method1, format_result

Xtr, ytr = build_feature_matrix(train)   # GMM phases -> grains -> indices
clf = LdaClassifier.fit(Xtr, ytr)
print('top discriminant features:')
for name, w in clf.coef_table():
    print(f'  {name:>22}: {w:+.3f}')

res1 = eval_method1(clf, test)
print('\\n' + format_result(res1))
"""),
    md("## Method 2 — ImageNet-pretrained CNN texture classifier"),
    code("""
from torch.utils.data import DataLoader
from intergrowth.tiles import TileDataset
from intergrowth.cnn_model import build_cnn, CnnConfig
from intergrowth.train_cnn import train_cnn, class_weights, CnnTrainConfig
from intergrowth.evaluate import eval_method2, format_result

BATCH = 16 if device.type == 'cuda' else 4
EPOCHS = 8 if device.type == 'cuda' else 1

ds = TileDataset(train, augment=True)
loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4 if device.type=='cuda' else 0, drop_last=True)
model = build_cnn(CnnConfig(backbone='convnext_tiny', pretrained=True))
w = class_weights(train, 2, device)
train_cnn(model, loader, device, w, CnnTrainConfig(epochs=EPOCHS, amp=device.type=='cuda'))

res2 = eval_method2(model, test, device)
print('\\n' + format_result(res2))
"""),
    md("## Side-by-side"),
    code("""
for r in (res1, res2):
    print(f'{r.method:<34} acc={r.accuracy*100:5.1f}%  balanced={r.balanced_accuracy*100:5.1f}%')
print('\\nNOTE: Method 1 is interpretable (grain indices); Method 2 is data-driven texture.')
print('Labels are image-level (folder sort) — this is the honest ceiling of both.')
"""),
]


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": CELLS,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = NB_DIR / "compare.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
