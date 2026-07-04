# intergrowth — обычные vs тонкие срастания (рядовая / труднообогатимая)

Separate model(s) that classify NON-talc ore by intergrowth type, comparing two
paradigms from process mineralogy on a held-out test split.

| | Method 1 | Method 2 |
|---|---|---|
| Paper | Pérez-Barnuevo, Pirard, Castroviejo 2013 (Min. Eng. 52:136–142) | Pérez-Barnuevo et al. 2018 (Min. Eng. 118:87–96); MISIS/LumenStone, Korshunov 2025 |
| Idea | minerallurgical indices per grain + discriminant analysis | ImageNet-pretrained CNN texture classifier on tiles |
| Signal | grain replacement / morphology (interpretable) | learned texture (data-driven) |
| Depends on | GMM phase map + grain extraction (from `orenet`) | raw tiles only |

## ⚠️ Honesty note

Neither paper released code, and their datasets are **not public** (proprietary MLA
data from Kansanshi / Mont-Wright). These are **faithful reimplementations from the
papers**, not the original code, trained on **your** folder-labelled data
(`Рядовые руды`→NORMAL, `Труднообогатимые руды`→FINE; talc folders excluded).
There is **no public pretrained weight** for this task — Method 2's "ready weights"
are a torchvision ImageNet backbone that we fine-tune (standard transfer learning).

The supervision is **image-level** (folder sort), so both methods share the same
honest ceiling: a "рядовая" slide contributes tiles/grains that are mostly, not
exclusively, обычные срастания.

## Layout

```
intergrowth/
├── src/intergrowth/
│   ├── constants.py     # 2 classes, folder→label map, paths
│   ├── data.py          # index images, slide-grouped split (no leakage)
│   ├── indices.py       # Method 1: PB2013 minerallurgical indices per image
│   ├── lda_model.py     # Method 1: StandardScaler + LDA
│   ├── tiles.py         # Method 2: tile dataset (tile inherits slide label)
│   ├── cnn_model.py     # Method 2: ConvNeXt-Tiny / ResNet-18 (ImageNet)
│   ├── train_cnn.py     # Method 2: train loop + tile-vote image inference
│   └── evaluate.py      # both methods → accuracy + confusion
├── scripts/build_notebooks.py
└── notebooks/compare.ipynb   # runs BOTH on the test set
```

## Run

```bash
cd ML/intergrowth
# 1) make the photos reachable (edit constants.DATA_ROOTS or symlink)
#    default expects ../first_classification_attempt/data/part1|part2
# 2) deps: torch torchvision scikit-learn scikit-image scipy opencv-python-headless tqdm
python scripts/build_notebooks.py        # (re)generate the comparison notebook
jupyter lab                              # open notebooks/compare.ipynb → Run All
```

Method 1 reuses `../transformer_version/src/orenet` (preprocess + GMM phases +
grains) via `constants.ORENET_SRC` — no trained segmenter needed.

## Caveats

- **Method 1 is slow**: the per-image GMM phase map costs ~3–5 s/image on CPU, so
  the full train set (~850 images) is ~1 h. It's a one-pass feature extraction.
- Method 2 wants a GPU for the real run (`convnext_tiny`, ImageNet weights); on CPU
  the notebook drops to 1 epoch / ResNet as a smoke run.
- External data (LumenStone S2 — Cu-Ni sulfides, Norilsk-like) can be added later;
  it strengthens phase segmentation, not the intergrowth labels.
