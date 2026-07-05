# Two-model talc estimation: classifier + segmentation fusion

*Notes on the idea of combining an image-level class model with the YOLO-seg
talc segmenter to make a more robust "is this specimen talcose?" decision.*

---

## The idea, restated precisely

So we're on the same page:

1. **Classifier** — predict the image class
   (`оталькованные` / `рядовые` / `труднообогатимые`) with confidence `P`.
   Rule: `P > τ` → trust it; `P < τ` → don't. (`τ` ≈ 0.8, maybe 0.9.)
2. **Segmenter** — the existing YOLO-seg model. Predict talc segments, each with a
   confidence. Compute total talc **area-fraction** `f = talc pixels / image pixels`.
3. **Conflict resolution** — if `f > 10%` (segmenter says *talcose*) but the
   classifier says *< 10%* (i.e. `рядовые`):
   - **a)** `P > τ` → trust the classifier: drop the lowest-confidence segments
     until `f` falls back under 10%.
   - **b)** `P < τ` → leave the segmentation as-is.

The instinct — *use a second, independent view to sanity-check the first* — is
sound. But the specific mechanism has a few problems I'd fix before building it.

---

## What's right about it

- **Anchoring on the 10% decision, not on IoU.** The deliverable is a
  *decision* (talcose vs not), so optimizing the thing near 10% is the correct
  target. Mask mAP is a proxy; the 10% line is the product.
- **Two different inductive biases can be complementary.** A whole-image
  classifier reads global texture/context; the segmenter localizes. When they
  disagree, that disagreement carries real information.
- **Gating trust on confidence** is the right shape of idea — *when* to defer to
  which model should depend on how sure each one is.

## Where it breaks (ranked by how much it matters)

### 1. The three classes aren't the same quantity as the 10% line ⚠️
`оталькованные / рядовые / труднообогатимые` is not a clean binary around 10%
talc. `труднообогатимые` ("hard-to-enrich") is arguably a *different axis*
(enrichability), which may be uncorrelated with raw talc %. Before any fusion
logic, we need one written-down definition: **which classes map to "≥10% talc"
and which to "<10%"?** If that mapping isn't 1:1, the whole "classifier says
<10%" branch is comparing two different measurements. This is the first thing to
pin down — it's conceptual, not code.

### 2. "Prune low-confidence segments until f < 10%" fits the answer to the prior
This step is outcome-driven: it deletes plausibly-real detections purely to make
the number match the classifier. Three problems:
- **It's biased by construction.** Whenever the classifier is confidently-low,
  you force `f` down — even when the segmentation was *right*. A confidently
  wrong classifier now silently corrupts a correct measurement.
- **YOLO-seg "confidence" is a detection score, not P(really-talc).** Low-conf ≠
  false positive. Small, faint, but real talc grains often score low. Pruning by
  confidence to hit a target throws away true positives preferentially.
- **Where do you stop — exactly 9.9%?** Then the reported number is an artifact
  of the threshold, not a measurement of the rock.

### 3. Two hard thresholds (τ and 10%) create cliff-edges exactly where it matters
`P = 0.79` vs `0.81`, or `f = 9.9%` vs `10.1%`, flip the entire regime. The
output is most unstable right at the decision boundary — the one place you need
it to be stable. Bumping τ to 0.9 doesn't fix this; it just moves the cliff.

### 4. The rule is one-directional
It only fires for *seg-high / class-low*. It says nothing about **seg-low /
class-talcose** — segmenter misses talc that the classifier's global view
catches. In ore processing a **missed** contamination (false negative) is
usually costlier than a false alarm, so ignoring that direction is backwards.
Decide which error is more expensive and handle both directions accordingly.

### 5. Confidence isn't calibrated — and can't be, on 42 images
"`P > 0.8` = trust" assumes the softmax is calibrated. Neural nets are
systematically overconfident, and a net trained on ~42 images will be wildly so —
a raw 0.8 might mean 55% real accuracy. Worse, with 42 images you can't even
*measure* calibration reliably. Any fixed τ is guesswork until there's a
reliability curve behind it.

### 6. The data bottleneck is labels, not model count
The project's own status says it plainly: the segmenter already overfits
(mask mAP50 ≈ 0.30) because there are only 42 images. A second deep model trained
on the *same* 42 images will overfit *just as hard*, and its confidence will be
noise. **Ensembling two overfit models doesn't make a strong one** — their errors
are correlated (both keyed off the same talc-texture signal), so they agree and
disagree *together* and the "second opinion" adds little. Two weak, correlated
models ≠ one good model.

*(Also, a mechanical bug to note for whoever implements `f`: summing segment
areas double-counts overlaps and can exceed 100%. Use the **union** of mask
pixels, not the sum of per-instance areas.)*

---

## What I'd build instead

Keep the two-model instinct; drop the hard gates and the pruning hack.

### A. Turn fusion into one calibrated score (stacking), not an override
Instead of one model editing the other, feed both into a tiny meta-model:

```
features per image = [ f_seg,  P_class,  n_segments,  mean_seg_conf ]
target             = talcose?  (0/1, from the definition in issue #1)
meta-model         = logistic regression   # 2–4 weights, not a deep net
```

A 2–4-feature logistic regression **is** estimable on 42 images via the k-fold CV
we already have (`kfold.py`) — unlike a second CNN. It *learns* the weighting
between the two models from data instead of us hand-guessing τ = 0.8, and it
handles both disagreement directions automatically. This is the standard,
defensible way to combine two models.

### B. If you'd rather not train a meta-model: soft, threshold-free fusion
Replace "delete segments until <10%" with a confidence-weighted area:

```
f = Σ (area_i · conf_i) / total_pixels
```

Monotone, no cliff, nothing deleted. Then optionally shrink `f` toward the
classifier's implied value by a fixed weight `w`:
`f_final = (1−w)·f + w·f_class`. Still needs calibrated confidences (see C), but
it removes the arbitrary "prune to 9.9%" behavior.

### C. Calibrate before trusting any number
Extend `kfold.py` to emit out-of-fold predictions, temperature-scale the
classifier, and plot a reliability curve. *Then* a threshold means something.
Until then, treat every `P` as ordinal, not probability.

### D. The genuinely best use of two models: **abstain, don't fabricate**
The most honest and most valuable thing disagreement buys you is a
**"needs human review"** band, not an auto-edited answer:

> If the two models disagree, **or** the combined score sits in a margin around
> 10% (say 8–12%), flag the specimen for a human instead of silently overriding.

In a mineral-processing QA setting, a calibrated *triage* system ("here are the
15% of specimens the models are unsure about") is worth far more than a pipeline
that quietly deletes segments to manufacture a clean number. This reframes the
whole idea from "model A overrules model B" to "two models raise their hand when
unsure" — which is both safer and easier to defend to whoever consumes the result.

---

## Recommended order of work

1. **Write the class → 10% mapping** (issue #1). Blocks everything; costs an
   afternoon and a conversation, not code.
2. **Check label availability.** Do the 42 images already carry an image-level
   class label (ideally a lab-assay talc %)? If yes, the classifier target — and
   the fusion ground truth — is nearly free. If no, that's a labeling task, and
   see #4.
3. **Get out-of-fold predictions + calibration** from the existing k-fold, for
   the *segmenter alone* first. Establish the single-model 10%-decision accuracy
   as the honest baseline.
4. **Only then** add the classifier, and fuse with **stacking (A)** + an
   **abstain band (D)** — measured against that baseline via the same CV.
5. Given the project's stated bottleneck, seriously weigh spending the same
   effort on **more labeled images** first. A second model is cleverness; more
   data is the thing the current numbers are actually starved for. If assay
   talc-% labels exist per specimen, the classifier is worth it *as a decision
   ground truth*; if it needs fresh hand-labeling, more segmentation labels
   likely beat it.

**Bottom line:** the two-model instinct is good, and I'd keep it — but as a
*calibrated soft vote with an abstain band*, not as a confident classifier that
deletes segments to force agreement. The override-and-prune version will look
fine on the val images and quietly bias the answer toward whichever model happens
to be confident, right at the 10% line where being wrong costs the most.
