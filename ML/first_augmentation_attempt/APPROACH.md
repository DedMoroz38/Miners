# Augmentation approach — design rationale

This document explains **what** was implemented and **why**, tracing every choice
back to a concrete weakness measured on the baseline. It is the reasoning half of
the experiment; `README.md` is the how-to-run half.

---

## 1. The problem we are fixing

The baseline (`first_classification_attempt`, EfficientNetV2-S, full fine-tune)
scored, on the held-out test set (177 images):

```
Accuracy 0.887 | Macro-F1 0.849

class      precision  recall   f1     support
ordinary     0.936    0.859   0.896     85
hard         0.896    0.945   0.920     73
talc         0.682    0.789   0.732     19      <-- the bottleneck

confusion (rows=true, cols=pred)
          ordinary  hard  talc
ordinary     73       8     4
hard          1      69     3
talc          4       0    15
```

Three findings drive the whole design:

1. **Talc is the ceiling.** ordinary/hard are already ~0.90–0.92; macro-F1 (equal
   weight per class) is pulled down almost entirely by talc's **0.732**.
2. **Talc fails in both directions.** Recall 0.789 → its decision region is
   *incomplete* (4 real talc called ordinary). Precision 0.682 → the region is
   also *too loose* (7 non-talc grabbed as talc). The over-prediction is partly a
   side effect of the baseline's inverse-frequency **class weights**.
3. **Overfitting.** Train F1 ~0.93 vs test 0.85, and val macro-F1 plateaued at
   epoch 4 — the model memorized 826 specific images rather than generalizing.

Root cause behind all three: **talc has only 91 train / 19 test images**, and the
augmentation was mild, so the model never saw enough talc variation.

The honest, zero-new-data lever is **augmentation** — increase the *effective*
diversity of what the model trains on. Four coordinated changes follow.

---

## 2. The four levers

### 2.1 Stronger per-image augmentation — `--aug strong`

`data.py :: get_transforms(policy="strong")`:

```
RandomResizedCrop(256, scale=0.6–1.0)   # scale/position invariance
RandomHorizontalFlip + RandomVerticalFlip
TrivialAugmentWide()                     # one random op/image, random magnitude
ToTensor + Normalize
RandomErasing(p=0.25, value="random")    # occlude a region -> force texture cues
```

**Why TrivialAugmentWide** (over RandAugment / AutoAugment): it is
**parameter-free** — it applies exactly one randomly chosen operation per image at
a random magnitude. On a dataset this small there is no budget to tune
RandAugment's `(N, M)`; TrivialAugmentWide matches or beats it out of the box and
removes a hyperparameter search. Because only one op fires per image and its
magnitude is random, individual destructive ops stay rare and mild.

**Why flips are kept explicit and always-on:** thin-section microphotographs have
**no canonical orientation** — a flipped/rotated field of view is an equally valid
image of the same ore. Flips are therefore *label-safe* and add geometric variety
on top of TrivialAugmentWide's single op.

**Why RandomErasing:** the real signal in a thin section is *local mineral
texture*, not a single dominant region. Erasing a random patch stops the model
leaning on one blob and pushes it toward texture that generalizes.

**Expected effect:** closes the train/test gap and lets training run past the
epoch-4 plateau; the **starved talc class benefits most** because augmentation
multiplies the variety of its 91 images the hardest.

> **Microscopy caveat (deliberate):** TrivialAugmentWide includes colour ops
> (solarize/posterize/equalize/etc.). Under some microscopy modes colour is
> diagnostic, so if colour distortion ever *hurts*, `--aug basic` restores the
> milder baseline pipeline (moderate ColorJitter only). Kept as a switch rather
> than hard-removed so the trade-off can be measured, not assumed.

### 2.2 Class-balancing sampler — `--sampler weighted`

`data.py :: make_sampler()` builds a `WeightedRandomSampler` with per-sample
weight `1 / count(class)`, so every **class** gets equal expected mass each epoch.
Talc (91 images) is therefore drawn ~4× as often as ordinary (395).

**Why this and not just class-weighted loss:** up-weighting the *loss* tells the
model "talc mistakes cost more" but it still *sees* few distinct talc images.
Oversampling makes it actually *look at* talc more. **The synergy with §2.1 is the
key idea:** because every talc draw passes through strong augmentation, the model
sees many *different* talc views, not 91 identical duplicates — genuinely more
effective talc data. Directly targets talc's incomplete decision region (recall).

Epoch length is kept at `len(records)` so timing stays comparable to the baseline.

### 2.3 MixUp / CutMix — `--mix mixup_cutmix`

`data.py :: build_mixups()` + the train loop in `train.py`. Per batch, with
probability `--mix-prob` (default 0.5), we apply either **MixUp** (linear blend of
two images and their labels) or **CutMix** (paste a rectangular patch from one
image into another; labels mixed by area). The integer label becomes a **soft
(probability) target**.

**Why:** MixUp/CutMix are strong regularizers for small datasets and they smooth
decision boundaries between confusable classes — aimed at the **ordinary↔hard**
leak (8 images) and at general overfitting. `mixup_alpha=0.2` follows the
EfficientNetV2 recipe: the Beta(0.2, 0.2) distribution is U-shaped, so the mix
coefficient is usually near 0 or 1 — most images stay close to a single original,
keeping labels mostly clean while still regularizing.

**Why only 50% of batches:** the sampler (§2.2) is working to *sharpen* talc;
MixUp *blends* images and can dilute a rare-class signal. Applying it to half the
batches keeps its regularization benefit without constantly diluting talc.

### 2.4 Don't double-correct the imbalance — `--class-weights auto`

The baseline used inverse-frequency **loss weights**. If we *also* add a balancing
sampler (§2.2), the imbalance gets corrected **twice**, over-emphasizing talc —
which is exactly what already hurt talc **precision** (0.682, too many false
talc). So:

```
--class-weights auto  ->  loss weights OFF when --sampler weighted, ON otherwise
```

(`on` / `off` force it either way.) The sampler balances *exposure*; stacking loss
weights on top would re-introduce the over-prediction we are trying to fix.
Verified live in the smoke run: `loss class-weights: False` when the sampler is on.

**Loss handling.** Un-mixed batches use `CrossEntropyLoss(weight, label_smoothing
=0.05)`. Mixed batches use a hand-rolled **weighted soft cross-entropy**
(`train.py :: soft_cross_entropy`) so soft targets + optional class weights behave
identically across torch versions. Training accuracy/F1 are always scored against
the **original** hard labels, so the logged train metrics stay interpretable even
when a batch was mixed.

### 2.5 Test-time augmentation — `evaluate.py --tta`

At evaluation, each image is passed through the **8 D4 views** (4 rotations × with/
without a flip); the softmax probabilities are averaged before `argmax`. Since
flips/rotations are label-safe for microscopy (§2.1), this is a near-free accuracy
gain and it **stabilizes the noisy 19-image talc metric** (averaging over views
reduces variance from any single ambiguous crop). Off by default; enable with
`--tta`.

---

## 3. Supporting training changes

- **`--epochs 40`, `--patience 12`** (baseline: 30 / 8). Stronger augmentation
  reduces overfitting, so the useful-learning plateau should arrive *later* — the
  model needs more room before early stopping is meaningful. Cosine LR `T_max`
  tracks `--epochs`. Early stopping is **kept** (it still saves only the best
  checkpoint, so it costs nothing and guards against late overfitting — see the
  reasoning in the baseline discussion).
- **`drop_last=True`** on the train loader: EfficientNetV2 uses BatchNorm, and a
  trailing batch of size 1 breaks BN in train mode; dropping it also keeps MixUp
  batches well-formed.
- **Everything else identical** to the baseline (backbone, optimizer AdamW
  1e-4/wd 1e-4, img 256, seed 42, label smoothing 0.05) so the comparison is
  clean.

---

## 4. What this fixes vs. what it can't

**Should improve:** talc recall (oversampling + varied augmentation fills the
decision region), talc precision (dropping the stacked class weights), the
ordinary↔hard confusion (MixUp/CutMix), and the overall train→test gap — pushing
macro-F1 up from 0.849 toward the task's 0.90 target.

**Cannot fix with augmentation alone:**
- Augmentation cannot invent talc *morphology* absent from the 91 originals —
  there is a genuine ceiling; collecting more talc images is still the highest-
  value move.
- The **19-image talc test set stays noisy** (~±0.05 macro-F1 per misclassified
  image). Augmentation improves the *model*, not the *measurement*. A trustworthy
  estimate needs **stratified k-fold cross-validation** — the natural next
  experiment after this one.

---

## 5. How to read the results

Train, then evaluate on the **same 177 test images** as the baseline:

```bash
python src/train.py
python src/evaluate.py --ckpt runs/best_efficientnet_v2_s.pt --tta
```

Compare against the baseline anchors: **macro-F1 0.849**, **talc F1 0.732 (recall
0.789 / precision 0.682)**. The talc row and the macro-F1 are the numbers that
matter. For a controlled attribution, re-run with individual levers turned off
(`--aug basic`, `--sampler none`, `--mix none`) to see how much each contributes.
