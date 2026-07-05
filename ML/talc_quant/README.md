# talc_quant — calibrated talc-fraction segmentation (±3% target)

Self-contained module that takes a polished-section OM image (field **or**
gigapixel panorama) and returns a **blue talc mask** + a **calibrated talc area
fraction**. It does **not** emit sulfide artifacts — sulfides are handled by the
separate sulfide model; here the sulfide/gray channels are internal auxiliary
supervision only.

Design: `docs/superpowers/specs/2026-07-05-talc-quantification-design.md`.
Plan: `docs/superpowers/plans/2026-07-05-talc-quant.md`.

## Why this works where the YOLO baseline didn't

Talc is a *stuff* class (diffuse dark phase), not objects — so this is **semantic
segmentation + a calibrated area estimator**, not instance detection. The ±3%
budget is met by three things the detector lacked:

1. **Probabilistic classify-and-count** — the fraction is `Σ p(talc)/N_valid`
   (a calibrated soft count), not a pixel argmax.
2. **Out-of-fold calibration** (temperature + robust Theil–Sen), ≤3 params, so it
   cannot overfit the 42 labeled images.
3. **Cosine-blended tiling** for panoramas — replaces the OR-union that
   double-counts 50%-overlap seams and inflates area.

Two interchangeable model tracks (pick by grouped-CV MAE): `segformer_b2/b3`
(track A) and `dino_vitb14` (frozen DINOv2 + light decoder, track B).

## Install

```bash
cd ML/talc_quant
python -m pip install -r requirements.txt          # torch: use the CUDA index on V100
```

## Run order

```bash
# 1) Build the corpus (talc masks from polygons + GMM aux labels + FDA bank + folds)
python scripts/build_dataset.py                    # -> data_build/

# 2) Phase-0 audit: confirm/repair the frozen GT (triptychs original|line|mask)
python scripts/make_audit.py                        # -> data_build/audit/*.jpg

# 3) Train all folds for a track (V100)
python scripts/train.py --track segformer_b2 --fold all model.pretrained=true
#   bigger backbone / larger crop:
python scripts/train.py --track segformer_b3 --fold all data.crop=768 train.batch=8
#   cheap second track:
python scripts/train.py --track dino_vitb14 --fold all

# 4) Calibrate + print the honest acceptance table (MAE/median/P90/max/bias)
python scripts/calibrate.py --track segformer_b2    # -> runs/segformer_b2/calibration.json

# 5) False-positive control on non-talc sorts (must sit near 0, P95<=3%)
python scripts/fp_control.py --track segformer_b2

# 6) Inference on a field or panorama
python scripts/infer.py --image /abs/panorama.jpg --out job/talc --track segformer_b2
```

Every script accepts OmegaConf dotlist overrides, e.g. `data.crop=768`,
`train.batch=16`, `calibrate.method=adjusted_count`, `infer.tta_scales=[0.8,1.0,1.25]`.

## Outputs (per image, contract-compatible with merge_phases.py)

| File | Meaning |
|---|---|
| `talc_mask.png` | uint8 0/255, area-matched to the reported fraction |
| `overlay.png` | blue semi-transparent talc over the original |
| `heatmap.png` | local talc-fraction grid (inhomogeneity) |
| `entropy.png` | per-pixel model uncertainty |
| `talc.json` | `talc_fraction_raw/final`, area px/µm², `is_talcose`, `needs_review`, grid |

## Acceptance (measured by `calibrate.py` + `fp_control.py`)

- talc-fraction **MAE ≤ 3%**, |bias| ≤ 0.5% on grouped-CV OOF predictions;
- non-talc sorts P95 ≤ 3%;
- all panoramas pass the geologist triptych/overlay check.

## Compute (V100 32 GB)

| Stage | Approx |
|---|---|
| build_dataset | ~3 min (GMM per image) |
| train B2 @768, 1 fold | 6–10 h → run folds overnight ×2 |
| dino_vitb14, 1 fold | ~2 h |
| calibrate / fp_control | minutes |
| panorama inference (148 Mp, ensemble+TTA) | 1–3 min |

## Layout

```
conf/default.yaml        all hyperparameters (OmegaConf)
src/talc_quant/          config, preprocess, scale, folds, gt_build, fda,
                         dataset, losses, models/{segformer,dino}, engine,
                         tiled, inference, quantify, reporting
scripts/                 build_dataset, make_audit, train, calibrate,
                         fp_control, infer, self_train
tests/                   pytest smoke (offline, tiny tensors)
data_build/  runs/       generated (git-ignored)
```

## Open question (blocks field-scale accuracy, not panorama)

Panorama **µm/px is unknown** (`scale.panorama_mag` is a placeholder). Confirm the
panorama magnification with the data owner; until then the module hedges with
scale jitter in training and multiscale TTA (`infer.tta_scales`).

## Notes on reuse

- `preprocess.py` composes the **sulfide** `normalize_exposure_anchor` (loaded
  standalone, the same trick as `talc_seg_normalized/normalize.py`) with the
  a/b cast neutralization ported from `transformer_version/orenet/preprocess.py`.
- `tiled.py` grid geometry mirrors `talc_seg_normalized/tiling.py`; the blending
  is upgraded from OR-union to cosine probability blending.
