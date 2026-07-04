# Graph Report - miners  (2026-07-04)

## Corpus Check
- 17 files · ~77,609,395 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 122 nodes · 184 edges · 15 communities detected
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 18 edges (avg confidence: 0.8)
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

## God Nodes (most connected - your core abstractions)
1. `main()` - 12 edges
2. `process_image()` - 8 edges
3. `make_splits()` - 8 edges
4. `main()` - 8 edges
5. `main()` - 7 edges
6. `OreDataset` - 7 edges
7. `build_model()` - 6 edges
8. `predict()` - 6 edges
9. `class_weights()` - 6 edges
10. `make_loader()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `load_classifier()` --calls--> `build_model()`  [INFERRED]
  fusion/fuse.py → first_classification_attempt/src/model.py
- `segment()` --calls--> `predict()`  [INFERRED]
  fusion/fuse.py → first_classification_attempt/src/evaluate.py
- `predict()` --calls--> `main()`  [INFERRED]
  first_classification_attempt/src/evaluate.py → first_segmentation_attempt/evaluate.py
- `sweep_tau()` --calls--> `build_model()`  [INFERRED]
  fusion/tune_thresholds.py → first_classification_attempt/src/model.py
- `main()` --calls--> `build_mixups()`  [INFERRED]
  first_classification_attempt/src/train.py → first_augmentation_attempt/src/data.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.12
Nodes (23): Dataset, build_index(), class_counts(), class_weights(), get_or_make_splits(), get_transforms(), load_splits(), make_loader() (+15 more)

### Community 1 - "Community 1"
Cohesion: 0.16
Nodes (20): classify(), d4_views(), draw_overlay(), get_device(), load_classifier(), load_module(), main(), mask_polygons() (+12 more)

### Community 2 - "Community 2"
Cohesion: 0.22
Nodes (10): get_device(), Shared utilities: device selection, seeding, paths., set_seed(), build_mixups(), Return a (mixup, cutmix) pair of v2 batch transforms, or (None, None).      Appl, main(), Fine-tune a pretrained backbone on the merged 3-class ore dataset.  Example:, Weighted cross-entropy for soft (probability) targets from MixUp/CutMix.      Ha (+2 more)

### Community 3 - "Community 3"
Cohesion: 0.24
Nodes (13): db_token(), fresh_dir(), image_relpath(), load_from_api(), load_tasks(), main(), poly_to_line(), polygons_of() (+5 more)

### Community 4 - "Community 4"
Cohesion: 0.26
Nodes (9): d4_views(), get_device(), load_module(), main(), Measure, not guess, the two fusion thresholds.  1. --seg-conf : run YOLO-seg val, sweep_seg_conf(), sweep_tau(), build_model() (+1 more)

### Community 5 - "Community 5"
Cohesion: 0.36
Nodes (7): group_key(), main(), make_folds(), Grouped k-fold cross-validation over the full 42-image talc set.  The stock val, Same drill specimen -> same fold (no train/val leakage within a fold)., Emit train/val image lists + data yaml for fold i (absolute paths)., write_fold_yaml()

### Community 6 - "Community 6"
Cohesion: 0.29
Nodes (6): main(), Evaluate a trained YOLO-seg checkpoint on the held-out val split.  Reports mask, main(), Fine-tune Ultralytics YOLO-seg on the hand-labelled talc dataset.  Why YOLO-seg:, Write a portable data.yaml pointing at the absolute yolo_seg dir.      Regenerat, write_data_yaml()

### Community 7 - "Community 7"
Cohesion: 0.48
Nodes (5): d4_views(), main(), predict(), Evaluate a trained checkpoint on the held-out test set.  Example:     python src, Yield the 8 dihedral-group views of a square image batch (B,C,H,W).

### Community 8 - "Community 8"
Cohesion: 0.67
Nodes (3): db_token(), main(), Return the first API token, enabling legacy-token auth (LS 1.23+ default-off).

### Community 9 - "Community 9"
Cohesion: 1.0
Nodes (1): Import a module by file path without touching sys.path (both sibling     experim

### Community 10 - "Community 10"
Cohesion: 1.0
Nodes (1): The 8 dihedral-group views of a square image batch (label-safe TTA).

### Community 11 - "Community 11"
Cohesion: 1.0
Nodes (1): Run YOLO-seg; return list of dicts with bool mask + polygon + conf.

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (1): Talc area-fraction from the UNION of masks (overlaps counted once).

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (1): Drop lowest-confidence segments until union fraction <= target.      Returns (ke

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (1): Write a portable data.yaml pointing at the absolute yolo_seg dir.      Regenerat

## Knowledge Gaps
- **37 isolated node(s):** `Measure, not guess, the two fusion thresholds.  1. --seg-conf : run YOLO-seg val`, `Two-model talc pipeline: classifier-gated YOLO-seg segmentation.  Takes a folder`, `Import a module by file path without touching sys.path (both sibling     experim`, `The 8 dihedral-group views of a square image batch (label-safe TTA).`, `One clean polygon per connected blob of the mask (external contours).      ultra` (+32 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 9`** (1 nodes): `Import a module by file path without touching sys.path (both sibling     experim`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 10`** (1 nodes): `The 8 dihedral-group views of a square image batch (label-safe TTA).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 11`** (1 nodes): `Run YOLO-seg; return list of dicts with bool mask + polygon + conf.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 12`** (1 nodes): `Talc area-fraction from the UNION of masks (overlaps counted once).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (1 nodes): `Drop lowest-confidence segments until union fraction <= target.      Returns (ke`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `Write a portable data.yaml pointing at the absolute yolo_seg dir.      Regenerat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_model()` connect `Community 4` to `Community 1`, `Community 2`, `Community 7`?**
  _High betweenness centrality (0.224) - this node is a cross-community bridge._
- **Why does `main()` connect `Community 2` to `Community 0`, `Community 4`?**
  _High betweenness centrality (0.205) - this node is a cross-community bridge._
- **Why does `main()` connect `Community 7` to `Community 0`, `Community 2`, `Community 4`?**
  _High betweenness centrality (0.165) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `main()` (e.g. with `set_seed()` and `get_device()`) actually correct?**
  _`main()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `main()` (e.g. with `get_device()` and `build_model()`) actually correct?**
  _`main()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Measure, not guess, the two fusion thresholds.  1. --seg-conf : run YOLO-seg val`, `Two-model talc pipeline: classifier-gated YOLO-seg segmentation.  Takes a folder`, `Import a module by file path without touching sys.path (both sibling     experim` to the rest of the system?**
  _37 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._