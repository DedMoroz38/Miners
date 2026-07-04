"""iseg — region-level intergrowth prediction (обычные vs тонкие срастания).

A tile classifier localises intergrowth regions on the photo (red = тонкие,
green = обычные) and aggregates to an image-level ore-sort decision. F1 is
measured at image level (folder sort is the only ground truth) and pushed with a
pretrained ConvNeXt backbone, strong augmentation, class-balanced sampling,
group k-fold ensembling, TTA, and threshold tuning.
"""

from __future__ import annotations

from .sources import FINE, NORMAL, Item, build_items, split_items

__all__ = ["NORMAL", "FINE", "Item", "build_items", "split_items"]
