# Talc Quantification Module (`ML/talc_quant`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) — user requested immediate implementation, no subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Self-contained `ML/talc_quant/` module: build dataset → train SegFormer (track A) and frozen-DINOv2 (track B) on grouped folds → calibrate area estimator → run panorama/field inference producing blue overlay + calibrated talc fraction. Spec: `docs/superpowers/specs/2026-07-05-talc-quantification-design.md`.

**Architecture:** 5-class semantic segmentation (talc supervised from 42 hand masks; matrix/sulfide/gray from GMM pseudo-labels as *internal aux only* — no sulfide outputs), probabilistic classify-and-count with fold-OOF calibration, cosine-blended tiled inference.

**Tech Stack:** PyTorch + HF transformers (SegFormer, DINOv2), OpenCV, scikit-learn (GMM), scipy (Theil–Sen), OmegaConf configs, pytest smoke tests. Runs CUDA (V100) / MPS / CPU-smoke.

## Global Constraints

- No sulfide artifacts in outputs (user has a separate sulfide model). Aux classes internal only.
- Reuse existing preprocessing: sulfide `preprocess_image` loaded standalone (same trick as `talc_seg_normalized/normalize.py`) + a/b cast neutralization ported from `orenet/preprocess.py` (provenance comments).
- Output contract kept: `talc_mask.png` (uint8 0/255), `talc.json` with `talc_fraction_raw/final`, plus `overlay.png`.
- Files 200–400 lines, type hints, frozen dataclass configs, `logging` not `print` (CLI argparse output allowed), seeds fixed (`set_seed(42)`).
- Model registry: `MODEL_FACTORY` + `@register_model(name)`; models built from cfg only.
- Grouped folds by specimen — one specimen never straddles train/val.
- Local smoke via `ML/transformer_version/.venv` (torch/cv2/transformers present); V100 uses `requirements.txt`.

## File Structure

```
ML/talc_quant/
├── README.md                  # run order: build → audit → train → calibrate → infer
├── requirements.txt
├── conf/default.yaml          # OmegaConf; all hyperparams
├── src/talc_quant/
│   ├── __init__.py            # __all__
│   ├── config.py              # frozen dataclasses + load_config(yaml, overrides)
│   ├── constants.py           # class ids (orenet contract), palette, paths
│   ├── seed.py                # set_seed
│   ├── preprocess.py          # canonical chain: anchor L (sulfide impl) + a/b cast fix
│   ├── scale.py               # magnification parse → resize factor to ×10-equivalent
│   ├── folds.py               # specimen_id(), assign_folds()
│   ├── gt_build.py            # polygons→talc mask; GMM pseudo-labels outside; label PNGs
│   ├── fda.py                 # Fourier amplitude swap w/ panorama tile bank
│   ├── dataset.py             # PatchDataset (rare-class crops, copy-paste, augs, FDA)
│   ├── losses.py              # DiceFocalTverskyCE + area-consistency term
│   ├── models/__init__.py     # registry + factory
│   ├── models/segformer.py    # track A
│   ├── models/dino.py         # track B (frozen ViT + conv decoder)
│   ├── engine.py              # train loop (AMP, cosine, val MAE early stop)
│   ├── tiled.py               # tile grid (port of tiling.py) + cosine-window prob blending
│   ├── inference.py           # predict_probs(image, models, cfg, tta); valid_mask()
│   ├── quantify.py            # soft fraction, temperature, Theil–Sen, calibration.json IO
│   └── reporting.py           # overlay/heatmap/entropy renders, talc.json writer
├── scripts/
│   ├── build_dataset.py       # labels + FDA bank + folds.json
│   ├── make_audit.py          # Phase-0 triptychs (original | blue line | mask)
│   ├── train.py               # --track a|b --fold N|all
│   ├── calibrate.py           # OOF inference → fit calib → CV metrics table
│   ├── infer.py               # panorama/field → overlay + talc.json + talc_mask.png
│   ├── fp_control.py          # non-talc folders FP distribution
│   └── self_train.py          # optional stage-3 pseudo-labels (off by default)
└── tests/                     # pytest smoke (no downloads, tiny tensors)
    ├── test_folds.py  test_gt_build.py  test_fda.py  test_tiled.py
    ├── test_losses.py test_quantify.py  test_scale.py
```

### Task 1: Skeleton + config + constants + seed + folds
**Produces:** `load_config() -> Config` (frozen dataclasses mirroring conf/default.yaml); `specimen_id(name) -> str` (2550374-2→"2550374"; DSCNnnnn→cluster by gap>10); `assign_folds(names, k=5, seed) -> dict[name,int]` deterministic.
- [ ] config/constants/seed/folds + tests (`test_folds.py`, `test_scale.py`) green.

### Task 2: Preprocess + scale
**Produces:** `preprocess_canonical(bgr) -> bgr` (anchor L via standalone-loaded sulfide file + matrix-anchored a/b shift, no CLAHE); `scale_factor(filename, cfg) -> float` (10/mag; panorama→cfg.panorama_scale).
- [ ] visual sanity: field + pano tile through chain; cast gone (a/b means ~128±3).

### Task 3: GT build + audit triptychs
**Produces:** `build_labels(cfg)` writing `data_build/labels/*.png` (uint8 ids, 255 ignore), `data_build/images/*.jpg` (canonical-scale, preprocessed), `folds.json`, FDA bank `data_build/fda_bank/*.jpg`; `make_audit.py` triptych JPEGs. Talc from yolo_seg polygons (zone convention); outside polygons GMM(L,3comp, conf≥0.95) → matrix/gray/sulfide else ignore; bright guard inside talc stays talc.
- [ ] run on real data: 42 labeled + non-talc part1/part2 subsample; spot-check 3 label overlays.

### Task 4: Dataset + FDA + losses
**Produces:** `PatchDataset(items, cfg, fda_bank)` → dict(image f32 CHW imagenet-norm, label i64); rare-class crop bias, talc copy-paste, Lab a/b jitter, gray-drop p=0.3, FDA p=0.3 (β≈0.03); `SegLoss(cfg)` = wCE+Dice+FocalTversky+λ·|softA−gtA|/N.
- [ ] `test_fda.py`, `test_losses.py` (loss finite, grads flow, area term zero when probs==onehot) green.

### Task 5: Models (registry) + engine
**Produces:** `MODEL_FACTORY`; `build_model(cfg) -> nn.Module` for `segformer_b2|b3` (HF, ade-pretrained, 5 labels) and `dino_vitb14` (frozen torch.hub/HF DINOv2 + 3-conv decoder to /4 + bilinear); `train_fold(cfg, fold) -> ckpt path` (AMP, AdamW, cosine+warmup, val talc-MAE early stop, checkpoint best).
- [ ] smoke train 20 iters CPU on tiny random-init segformer config; loss drops.

### Task 6: Tiled inference + quantify + reporting
**Produces:** `blend_probs(image, models, cfg)` full-res prob map (1024 tiles, 50% overlap, cosine window, per-tile preprocess, flips TTA, fold ensemble); `valid_mask(bgr, probs) -> bool[]` (border-connected dark ∪ predicted background); `soft_fraction(probs, valid) -> float`; `fit_calibration(pairs) -> Calib(temp, slope, intercept)`; `apply_calibration`; `render_overlay` (blue, alpha .5, area-matched threshold), heatmap, entropy, `talc.json`.
- [ ] `test_tiled.py` (cosine weights sum≈1 everywhere), `test_quantify.py` (Theil–Sen recovers synthetic slope) green.

### Task 7: CLI scripts + README + fp_control + self_train
**Produces:** all `scripts/*` runnable end-to-end; README with V100 commands; `calibrate.py` prints per-fold + pooled MAE/median/P90/max/bias table and writes `calibration.json`.
- [ ] end-to-end smoke on 3 images with tiny model; README accurate.

### Task 8: Verification pass
- [ ] pytest all green in transformer venv; ruff-clean; file sizes ≤400; no sulfide outputs anywhere; commit plan+code? (ask user re commit).
