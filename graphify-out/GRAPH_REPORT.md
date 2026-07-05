# Graph Report - miners  (2026-07-05)

## Corpus Check
- 144 files · ~160,665,485 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 784 nodes · 1263 edges · 40 communities detected
- Extraction: 74% EXTRACTED · 26% INFERRED · 0% AMBIGUOUS · INFERRED: 334 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]

## God Nodes (most connected - your core abstractions)
1. `Sample` - 24 edges
2. `Item` - 17 edges
3. `main()` - 15 edges
4. `train()` - 15 edges
5. `ClsTileDataset` - 15 edges
6. `TileDataset` - 14 edges
7. `PreprocessConfig` - 14 edges
8. `PatchDataset` - 13 edges
9. `SegTileDataset` - 13 edges
10. `PseudoLabelConfig` - 13 edges

## Surprising Connections (you probably didn't know these)
- `real_merge_payload()` --calls--> `build_payload()`  [INFERRED]
  back/test_api_contract.py → ML/merge/merge_phases.py
- `main()` --calls--> `sleep()`  [INFERRED]
  back/test_api_contract.py → front/src/features/analyze-sample/lib/analyze.ts
- `run_one()` --calls--> `classify()`  [INFERRED]
  ML/fusion/infer_one.py → front/src/features/analyze-sample/lib/metrics.ts
- `run_one()` --calls--> `classify()`  [INFERRED]
  ML/talc_infer/infer.py → front/src/features/analyze-sample/lib/metrics.ts
- `Run YOLO-seg; return list of dicts with bool mask + polygons + conf.` --rationale_for--> `segment()`  [EXTRACTED]
  fusion/fuse.py → ML/fusion/fuse.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.04
Nodes (69): _estimate_flat_field(), from_cfg(), luminance(), normalize_exposure_anchor(), normalize_illumination(), preprocess_image(), PreprocessConfig, Photometric preprocessing for reflected-light OM images.  Goals: flatten uneven (+61 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (55): classify(), d4_views(), draw_overlay(), get_device(), load_classifier(), load_module(), main(), mask_polygons() (+47 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (44): _ensure_orenet(), image_features(), Method 1 — minerallurgical indices (reimplemented from the paper).  Pérez-Barnue, Return the PB2013-style intergrowth feature vector for one image., build_cache(), CachedItem, One-off cache of preprocessed images + label maps to disk.  GMM pseudo-labelling, Materialise (image, label) PNG pairs; return the cached index. (+36 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (39): build_cnn(), CnnConfig, Method 2 model: an ImageNet-pretrained CNN texture classifier.  No public weight, Build the backbone with an ImageNet head swapped for n_classes., Constants for the intergrowth-type classifier (обычные vs тонкие срастания).  Tw, build_samples(), group_split(), Index the ore photos by folder label and split by slide (no leakage).  Only NON- (+31 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (40): Dataset, get_device(), Shared utilities: device selection, seeding, paths., set_seed(), build_index(), build_mixups(), class_counts(), class_weights() (+32 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (34): predict_image(), Image label = argmax of tile-probability mean (soft voting). Returns (label, p_f, _alb_pipeline(), balanced_sampler(), Tile dataset with strong augmentation and class-balanced sampling.  Tiles inheri, Inverse-frequency sampling so both ore types are seen equally., TileDataset, _to_tensor() (+26 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (34): main(), Evaluate a trained YOLO-seg checkpoint on the held-out val split.  Reports mask, group_key(), main(), make_folds(), Grouped k-fold cross-validation over the full 42-image talc set.  The stock val, Same drill specimen -> same fold (no train/val leakage within a fold)., Emit train/val image lists + data yaml for fold i (absolute paths). (+26 more)

### Community 7 - "Community 7"
Cohesion: 0.07
Nodes (25): ClsTileDataset, make_balanced_sampler(), Tile dataset for the intergrowth-type classifier.  Every tile inherits the IMAGE, Deterministic evaluation tiles: up to `cap` highest-sulfide tiles per image., Classification tiles indexed by a tiles table (image_id, x, y, frac)., Args:             tiles: rows (image_id, x, y, sulfide_frac) from make_pseudo_ma, Sampler balancing both classes and per-image tile counts.      Weight of a tile, top_tiles_per_image() (+17 more)

### Community 8 - "Community 8"
Cohesion: 0.08
Nodes (30): cls_train_transform(), _FallbackFlip, Albumentations pipelines for both training stages.  Falls back to light numpy-on, Geometric + photometric augmentation for segmenter training., Strong augmentation for the tile classifier., Minimal flip-only augmentation used when albumentations is absent., uint8 HWC RGB -> float32 CHW, ImageNet-normalized., seg_train_transform() (+22 more)

### Community 9 - "Community 9"
Cohesion: 0.12
Nodes (31): analyze(), analyze_status(), get_sample_image(), Статус задания анализа (опрашивается фронтендом до status='done')., Размеры изображения из заголовка (без полной декодировки)., Приём панорамного шлифа (multipart/form-data, поле `file`)., Отдаёт ранее загруженный снимок образца., Ставит анализ образца в очередь и возвращает id фонового задания. (+23 more)

### Community 10 - "Community 10"
Cohesion: 0.11
Nodes (17): classify(), computeShares(), recomputeResult(), segmentAreaFrac(), clientToNorm(), commitSelectedPolys(), deleteSelected(), deleteVertex() (+9 more)

### Community 11 - "Community 11"
Cohesion: 0.12
Nodes (23): build_payload(), build_sulfide_conf_map(), build_talc_conf_map(), classify(), compute_metrics(), extract_segments(), main(), make_debug_overlay() (+15 more)

### Community 12 - "Community 12"
Cohesion: 0.18
Nodes (18): classify_image(), grain_type(), ImageResult, Per-grain intergrowth type + image-level ore-sort decision (explicit rule)., fine' (тонкое) or 'normal' (обычное) intergrowth., RuleThresholds, Global constants: class ids, palette, dataset paths.  Phase classes are the fixe, extract_grains() (+10 more)

### Community 13 - "Community 13"
Cohesion: 0.12
Nodes (17): asJson(), getJob(), startAnalysis(), uploadSample(), analyzeSample(), analyzeViaApi(), sleep(), sampleFromFile() (+9 more)

### Community 14 - "Community 14"
Cohesion: 0.24
Nodes (13): db_token(), fresh_dir(), image_relpath(), load_from_api(), load_tasks(), main(), poly_to_line(), polygons_of() (+5 more)

### Community 15 - "Community 15"
Cohesion: 0.19
Nodes (12): _assign_holdout(), build_index(), _class_of(), load_index(), Dataset indexing: scan class folders, derive slide groups, holdout split.  Label, Group-aware stratified holdout, preferring strongly-grouped slides., Persist the index as CSV., Load a previously built index.      Raises:         FileNotFoundError: when the (+4 more)

### Community 16 - "Community 16"
Cohesion: 0.27
Nodes (9): _batch_tensor(), predict_image(), Inference: TTA tile scoring, image-level probability, and region heatmaps.  The, Mean softmax over models x {identity, hflip, vflip} TTA (or identity only)., Return (label, p_fine_image, heatmap_coords, per_tile_p_fine, hw)., Paint a red(fine)/green(normal) heatmap of intergrowth regions over the photo., region_overlay(), _tile_grid() (+1 more)

### Community 17 - "Community 17"
Cohesion: 0.25
Nodes (6): Сульфидный шаг запускается только если есть его веса (иначе — talc-only).      Т, sulfide_available(), analyze_sample(), Ore analysis: run the ML pipeline(s) + merge, return the frontend payload.  Orch, Run talc (+ sulfide if available) + merge; return the AnalysisResult dict., _run()

### Community 18 - "Community 18"
Cohesion: 0.28
Nodes (4): _DecoderBlock, U-Net with a ResNet encoder (torchvision, ImageNet-pretrained).  Compact hand-ro, Binary sulfide segmenter. cfg node: model.segmenter., ResNetUNet

### Community 19 - "Community 19"
Cohesion: 0.43
Nodes (7): d4_views(), get_device(), load_module(), main(), Measure, not guess, the two fusion thresholds.  1. --seg-conf : run YOLO-seg val, sweep_seg_conf(), sweep_tau()

### Community 20 - "Community 20"
Cohesion: 0.5
Nodes (5): code(), main(), md(), notebook(), Generate notebooks/compare.ipynb — runs BOTH methods on a held-out test split.

### Community 21 - "Community 21"
Cohesion: 0.32
Nodes (7): detect_blue_line(), extract_pair(), line_to_regions(), Extract talc pixel masks from blue-contour annotations.  Each annotated image (i, Boolean mask of the blue annotation stroke., Turn an (possibly open) blue contour into filled talc regions (bool)., Load an (original, annotated) pair -> uint8 {0,1} talc mask.

### Community 22 - "Community 22"
Cohesion: 0.4
Nodes (3): install_loss_stubs(), Compatibility shim so the talc YOLO-seg checkpoint loads under stock ultralytics, Patch ultralytics.utils.loss to resolve any missing loss class to a stub.

### Community 23 - "Community 23"
Cohesion: 0.67
Nodes (3): db_token(), main(), Return the first API token, enabling legacy-token auth (LS 1.23+ default-off).

### Community 24 - "Community 24"
Cohesion: 0.67
Nodes (2): downloadBlob(), exportCsv()

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): orenet: OM polished-section ore phase segmentation and intergrowth analysis.

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Make `src` importable when pipeline scripts run from any CWD.

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Build from an OmegaConf node (cfg.data.pseudo).

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Build from an OmegaConf node (cfg.data.preprocess).

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): Сульфидный шаг запускается только если есть его веса (иначе — talc-only).      Т

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): Return the classifier pack, or None if weights are absent / fail to load.

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (1): Fusion on ONE image -> (record dict, union mask uint8).

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): Two-point photometric standardization across differently exposed shots.      Map

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (1): Full canonical talc transform: exposure anchor (+ light median denoise).      Ar

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (1): Import a module by file path without touching sys.path (both sibling     experim

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (1): The 8 dihedral-group views of a square image batch (label-safe TTA).

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (1): Run YOLO-seg; return list of dicts with bool mask + polygon + conf.

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (1): Talc area-fraction from the UNION of masks (overlaps counted once).

### Community 77 - "Community 77"
Cohesion: 1.0
Nodes (1): Drop lowest-confidence segments until union fraction <= target.      Returns (ke

### Community 78 - "Community 78"
Cohesion: 1.0
Nodes (1): Write a portable data.yaml pointing at the absolute yolo_seg dir.      Regenerat

## Knowledge Gaps
- **183 isolated node(s):** `Сульфидный шаг запускается только если есть его веса (иначе — talc-only).      Т`, `Загруженный образец. Контракт под фронтенд (entities/sample).`, `Сегмент фазовой маски. Полигоны в НОРМАЛИЗОВАННЫХ координатах (0..1).`, `Результат анализа — контракт под фронтенд (entities/analysis: AnalysisResult).`, `Ответ на запуск анализа: идентификатор фонового задания.` (+178 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 24`** (4 nodes): `export.ts`, `downloadBlob()`, `exportCsv()`, `exportReport()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (2 nodes): `__init__.py`, `orenet: OM polished-section ore phase segmentation and intergrowth analysis.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (2 nodes): `_bootstrap.py`, `Make `src` importable when pipeline scripts run from any CWD.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Build from an OmegaConf node (cfg.data.pseudo).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Build from an OmegaConf node (cfg.data.preprocess).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `Сульфидный шаг запускается только если есть его веса (иначе — talc-only).      Т`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `Return the classifier pack, or None if weights are absent / fail to load.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `Fusion on ONE image -> (record dict, union mask uint8).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `Two-point photometric standardization across differently exposed shots.      Map`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `Full canonical talc transform: exposure anchor (+ light median denoise).      Ar`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (1 nodes): `Import a module by file path without touching sys.path (both sibling     experim`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `The 8 dihedral-group views of a square image batch (label-safe TTA).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `Run YOLO-seg; return list of dicts with bool mask + polygon + conf.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (1 nodes): `Talc area-fraction from the UNION of masks (overlaps counted once).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 77`** (1 nodes): `Drop lowest-confidence segments until union fraction <= target.      Returns (ke`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 78`** (1 nodes): `Write a portable data.yaml pointing at the absolute yolo_seg dir.      Regenerat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_panorama()` connect `Community 1` to `Community 11`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **Why does `get_device()` connect `Community 0` to `Community 1`, `Community 4`?**
  _High betweenness centrality (0.121) - this node is a cross-community bridge._
- **Why does `segment()` connect `Community 1` to `Community 3`, `Community 4`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Are the 22 inferred relationships involving `Sample` (e.g. with `Item` and `Sample source with optional reuse of the transformer_version preprocessed cache.`) actually correct?**
  _`Sample` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `Item` (e.g. with `iseg — region-level intergrowth prediction (обычные vs тонкие срастания).  A til` and `TileDataset`) actually correct?**
  _`Item` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `main()` (e.g. with `set_seed()` and `get_device()`) actually correct?**
  _`main()` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `train()` (e.g. with `_train_one()` and `train_single()`) actually correct?**
  _`train()` has 12 INFERRED edges - model-reasoned connections that need verification._