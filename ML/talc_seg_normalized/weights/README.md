# Talc segmenter weights (loaded separately)

Same convention as the sulfide pipeline (`ML/sulfide_intergrowth/weights/`):
trained weights are **not** produced here and **not** committed — drop them in
and the talc inference loads them from this directory.

| File | What | Produced by |
|---|---|---|
| `talc_seg_best.pt` | YOLO11-seg talc segmenter, trained on the **anchor-normalized** dataset (variant B) | `train.py` → `runs/talc_seg_norm/weights/best.pt`, copied here |

After training:

```bash
python build_dataset.py            # -> dataset_norm/
python train.py                    # -> runs/talc_seg_norm/weights/best.pt
cp runs/talc_seg_norm/weights/best.pt weights/talc_seg_best.pt
```

## Contract with inference

- The checkpoint here is trained on tiles passed through
  `normalize.preprocess_talc`. The panorama tiler in `ML/talc_infer/` **must**
  apply the *same* `preprocess_talc` per fragment before feeding YOLO —
  otherwise the profiles diverge (the mismatch variant B exists to remove).
- Point the talc inference at this file via its `--seg-weights` argument (or the
  `TALC_SEG_WEIGHTS` env var once wired). Absent → inference should fall back to
  the raw baseline weights in
  `ML/first_segmentation_attempt/runs/talc_seg_v2/weights/best.pt` **only if**
  it also skips normalization (raw weights ⇄ raw input).
