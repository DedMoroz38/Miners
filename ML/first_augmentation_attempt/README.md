# Augmentation experiment — ore thin-sections (3 classes)

Second iteration on the [`first_classification_attempt`](../first_classification_attempt)
baseline. **Same model, same data, same test set** — the *only* thing that
changes is the augmentation / sampling recipe, so any macro-F1 difference is
attributable to augmentation, not luck.

> **Why this experiment:** the baseline topped out at **test macro-F1 0.849**,
> dragged down almost entirely by the tiny **talc** class (F1 **0.732**, 91 train
> / 19 test images), plus a clear overfit gap (train F1 0.93 vs test 0.85). See
> [`APPROACH.md`](APPROACH.md) for the full diagnosis and the design rationale.

## What's new vs. the baseline

| Lever | Flag | Attacks |
|-------|------|---------|
| Strong per-image aug (TrivialAugmentWide + flips + RandomErasing) | `--aug strong` | overfit gap; starved talc class |
| Class-balancing sampler (oversamples talc) | `--sampler weighted` | talc's incomplete decision region |
| MixUp / CutMix (soft labels) | `--mix mixup_cutmix` | ordinary↔hard confusion |
| Drop stacked class weights when sampler is on | `--class-weights auto` | talc over-prediction (precision) |
| 8-view D4 test-time augmentation | `evaluate --tta` | noisy 19-image talc metric |

Everything is a toggle so you can A/B any single lever.

## Setup

```bash
pip install -r requirements.txt   # torchvision>=0.17 for transforms.v2 MixUp/CutMix
```

Data and the persisted split are **read from the sibling
`first_classification_attempt/` folder** (nothing to copy).

## Train

```bash
cd ML/first_augmentation_attempt

# quick sanity check (1 epoch, tiny subset, no weight download)
python src/train.py --smoke

# full recipe (auto-selects MPS / CUDA / CPU)
python src/train.py

# reproduce the baseline inside this folder (for a controlled A/B)
python src/train.py --aug basic --sampler none --mix none --class-weights on
```

Best checkpoint (by val macro-F1) → `runs/best_efficientnet_v2_s.pt`.

## Evaluate

```bash
# held-out test set, same 177 images as the baseline
python src/evaluate.py --ckpt runs/best_efficientnet_v2_s.pt

# with 8-view test-time augmentation
python src/evaluate.py --ckpt runs/best_efficientnet_v2_s.pt --tta
```

Prints accuracy, macro-F1, per-class precision/recall/F1 and a confusion matrix;
saves metrics to `runs/eval_test[_tta]_efficientnet_v2_s.json`. Compare the talc
row and the macro-F1 against the baseline's **0.849 / talc 0.732**.

## Files

- `src/common.py` — paths (point at sibling data + shared split), class map, device/seed
- `src/data.py` — aug policies, WeightedRandomSampler, MixUp/CutMix factory
- `src/model.py` — pretrained backbone factory (unchanged from baseline)
- `src/train.py` — training loop with sampler + MixUp/CutMix + soft-label loss
- `src/evaluate.py` — held-out eval + D4 test-time augmentation
- `APPROACH.md` — **design rationale**: why each augmentation, and its expected effect
