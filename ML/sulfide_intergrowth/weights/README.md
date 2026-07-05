# Sulfide-intergrowth inference weights — drop your trained files here

`infer_panorama.py` loads its weights from this directory (`paths.weights=weights`,
overridable). Put the following files here, then the merge pipeline can run:

| File | What | Format |
|---|---|---|
| `segmenter_best.pt` | U-Net sulfide segmenter (ResNet34 encoder) | `torch.save({"model_state_dict": ..., "best_metric": <float>})` — loaded in `src/inference/panorama.py:_load_segmenter` |
| `classifier_best.pt` **or** `classifier_fold*.pt` | ConvNeXt-small tile classifier (normal vs fine). One file, or an ensemble of `classifier_fold0.pt`, `classifier_fold1.pt`, … | `torch.save({"model_state_dict": ...})` — loaded in `_load_classifiers` |
| `decision.json` | fine-vs-normal decision threshold + reported F1 | `{"threshold": <float 0..1>, "val_f1": <float>, "holdout_f1": <float>}` — **required**, read in `PanoramaPipeline.__init__` |

Notes:
- The model architecture is fixed by `run/conf/model/default.yaml` (`resnet_unet` /
  `convnext_tile`, `num_classes: 2`). Weights must match that architecture.
- `decision.json["threshold"]` is mandatory — inference raises `FileNotFoundError`
  without it. `val_f1`/`holdout_f1` are optional and surface as the `f1` metric.
- Class map convention emitted to `*_classmap.png`: `0=bg / 1=обычные срастания (green) /
  2=тонкие срастания (red)`. Talc is added later by the merge step (blue).
- To run a single image (as the backend does):
  ```
  python run/pipeline/infer_panorama.py +input=/abs/img.jpg \
      paths.weights=/abs/ML/sulfide_intergrowth/weights \
      paths.outputs=/abs/job_dir
  ```
  Output: `<job_dir>/reports/<stem>_classmap.png` (+ `_segments.csv`, `_metrics.csv`, `_overlay.jpg`).
