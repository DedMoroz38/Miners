# Two-model fusion — classifier-gated talc segmentation

Combines the two existing models into one inference pipeline:

- **Classifier** — EfficientNetV2-S from
  [`../first_augmentation_attempt`](../first_augmentation_attempt)
  (3 classes: рядовые / труднообогатимые / оталькованные), run with 8-view D4
  test-time augmentation. It is the *stronger* model (test acc ≈ 0.91 with TTA).
- **Segmenter** — YOLO11-seg from
  [`../first_segmentation_attempt`](../first_segmentation_attempt)
  (`runs/talc_seg_v2/weights/best.pt`), the *weaker* model
  (val mask mAP50 ≈ 0.37) but the one that produces the deliverable: talc masks.

## The fusion rule

Per image:

1. Classify → class + confidence `P` (mean softmax over 8 D4 views).
2. Segment at `--seg-conf` → instances with confidences. Talc area-fraction
   `f` = **union** of mask pixels / image pixels (union, not sum — instances
   overlap, a sum can exceed 100%).
3. **Conflict**: `f > 10%` (segmenter says *talcose*) while the classifier says
   a low-talc class (both рядовые *and* труднообогатимые imply <10% talc):
   - `P ≥ τ` → trust the classifier: drop lowest-confidence segments (greedy,
     union recomputed after each drop) until `f ≤ 10%`. Dropped segments stay
     in the JSON with `"kept": false` — nothing is silently deleted.
   - `P < τ` → classifier not confident enough; segmentation kept as-is
     (the `note` field records the unresolved disagreement).

The reverse direction (classifier says оталькованные, segmenter finds <10%) is
deliberately not handled — decided with the domain owner.

## Thresholds (measured, not guessed)

| Flag | Default | Where the number comes from |
|---|---|---|
| `--seg-conf` | **0.42** | Val-split mask F1 peaks at **0.516** and stays ≥95% of the peak over **[0.415, 0.550]** (`tune_thresholds.py`). 0.42 takes the recall-leaning end of that band on purpose: the fusion rule can only *remove* segments, so err toward catching candidates. |
| `--p-thresh` (τ) | **0.9** | Measured by `tune_thresholds.py` on the augmented classifier (val split, 177 images, TTA): accuracy among trusted (`P ≥ τ`) predictions first reaches 95% at **τ=0.90** (0.957 acc, 53% coverage). At the hand-picked 0.8 it is only 0.941 — the softmax is overconfident, as expected. At 0.95 accuracy *drops* (0.933) — don't push τ higher. |
| `--talc-frac` | **0.10** | The product definition of "оталькованные". |

## Run

```bash
# env: needs ultralytics + torchvision; first_labling_attempt/.venv has both
VENV=../first_labling_attempt/.venv/bin/python

$VENV fuse.py --images /path/to/folder
$VENV fuse.py --images /path/to/folder --out runs/my_batch --p-thresh 0.85
$VENV tune_thresholds.py            # measure both thresholds
$VENV tune_thresholds.py --skip-cls # seg sweep only (no classifier ckpt yet)
```

If the augmented classifier checkpoint
(`../first_augmentation_attempt/runs/best_efficientnet_v2_s.pt`) is missing,
`fuse.py` warns and falls back to the baseline
(`../first_classification_attempt/runs/best_efficientnet_v2_s.pt`).

## Output

- `runs/fusion/results.json` — everything the frontend needs:

```jsonc
{
  "config": { "p_thresh": 0.8, "seg_conf": 0.42, "talc_frac": 0.1, ... },
  "images": [{
    "image": "DSCN3042.jpg",
    "width": 2048, "height": 1536,
    "class": "ordinary",                  // ordinary | hard | talc
    "class_ru": "рядовые",
    "class_confidence": 0.93,
    "class_probs": { "ordinary": 0.93, "hard": 0.05, "talc": 0.02 },
    "talc_fraction_raw": 0.14,            // before the rule
    "talc_fraction_final": 0.09,          // after the rule
    "is_talcose_final": false,
    "rule_fired": true,
    "note": "seg said 14.0% talc but classifier is confident...",
    "segments": [{
      "confidence": 0.71,
      "kept": true,                       // false = pruned by the rule
      "area_px": 51234,
      "area_frac": 0.016,
      // one polygon per connected blob of the instance mask:
      "polygons_px":   [[[x, y], ...], ...],  // pixel coords (ints)
      "polygons_norm": [[[x, y], ...], ...]   // 0–1 relative coords for drawing
    }]
  }]
}
```

- `runs/fusion/overlays/<image>` — visual check: kept segments green (filled),
  pruned segments gray (outline only), header with class / P / talc % before→after.
  (Overlay text is English/ASCII — cv2 can't render Cyrillic; the JSON has `class_ru`.)

## Files

- `fuse.py` — the pipeline (folder in → overlays + `results.json` out)
- `tune_thresholds.py` — measures `--seg-conf` (mask-F1 curve) and `--p-thresh`
  (τ sweep on the classifier val split)

## Known limits (by design, documented in `../../two-model-talc-fusion.md`)

- Pruning is outcome-driven: a confidently *wrong* classifier will delete real
  talc segments. τ controls how often that happens; measure it before trusting.
- YOLO confidence is a detection score, not P(really talc) — pruning by it
  preferentially removes small/faint but real grains.
- Both hard thresholds (τ, 10%) are cliff-edges; images near the boundary flip
  regimes on tiny changes. The `note` + `rule_fired` fields exist so a human
  can audit exactly those cases.
