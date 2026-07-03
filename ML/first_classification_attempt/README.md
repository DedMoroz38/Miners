# First classification model — ore thin-sections (3 classes)

Fine-tunes an ImageNet-pretrained backbone to classify polished-thin-section
microphotographs into three ore types (Nornickel "Скажи мне кто твой шлиф" task).

## Classes (part1 + part2 merged)

| class      | folders merged                                        | images |
|------------|-------------------------------------------------------|--------|
| `ordinary` | `part1/Рядовые руды`, `part2/рядовые`                 | 565    |
| `hard`     | `part1/Труднообогатимые руды`, `part2/тонкие`         | 486    |
| `talc`     | `part1/Оталькованные руды`, `part2/оталькованные`     | 129    |

- `part2/тонкие` (thin intergrowths) is the same class as `part1/Труднообогатимые
  руды` (hard-to-process ore) per the task's geological logic → merged as `hard`.
- The `Области оталькования` sub-folder is **ignored** (it holds annotated copies
  of the talc images, not new samples).
- Stratified **70 / 15 / 15** train / val / test split, persisted to `splits.json`
  so training and evaluation always use the same test set.

## Why this model

The task is **whole-image classification** (one label per thin section):
- **U-Net** = pixel segmentation, **YOLO** = object detection → both solve a
  different problem (needed for the *full* pipeline's mask stage, not this label).
- Correct tool = an ImageNet encoder + classifier head, fine-tuned.
- Default **EfficientNetV2-S**: best accuracy/params trade-off for a small,
  imbalanced dataset. `convnext_tiny` and `resnet50` are selectable via `--model`.
- Class imbalance (talc is ~11%) handled with inverse-frequency **class-weighted
  cross-entropy** + label smoothing; validation tracked by **macro-F1**
  (the task's target metric, ≥0.90 F1).

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
# quick sanity check (1 epoch, tiny subset, no weight download)
python src/train.py --smoke

# full training (auto-selects MPS on Apple Silicon / CUDA / CPU)
python src/train.py --model efficientnet_v2_s --epochs 30 --img-size 256
```

Best checkpoint (by val macro-F1) → `runs/best_<model>.pt`.

## Test

```bash
python src/evaluate.py --ckpt runs/best_efficientnet_v2_s.pt
```

Prints accuracy, macro-F1, per-class precision/recall/F1 and a confusion matrix;
saves metrics to `runs/eval_test_<model>.json`.

## Files

- `src/common.py` — paths, class map, device/seed helpers
- `src/data.py` — merge + stratified split + transforms + loaders
- `src/model.py` — pretrained backbone factory
- `src/train.py` — training loop (class-weighted loss, cosine LR, early stop)
- `src/evaluate.py` — held-out test evaluation + report
