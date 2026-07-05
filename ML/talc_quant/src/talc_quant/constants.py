"""Global constants: class ids, palette, image extensions.

Class ids MATCH the orenet contract (transformer_version/src/orenet/constants.py)
so labels are interchangeable, but this module emits ONLY the talc artifact — the
sulfide / gray channels are internal auxiliary supervision and never surfaced.
"""
from __future__ import annotations

# --- phase class ids (segmentation output channels) ---
CLASS_MATRIX = 0      # non-ore silicate/oxide matrix (dark)
CLASS_SULFIDE = 1     # sulfides (bright) — internal aux only, never output
CLASS_GRAY = 2        # gray ore phase, e.g. magnetite (mid-gray) — internal aux
CLASS_TALC = 3        # talc: disseminated dark phase in non-ore matrix (the target)
CLASS_BACKGROUND = 4  # mounting resin / field background / defects
IGNORE_INDEX = 255    # pixels excluded from the loss

NUM_CLASSES = 5

CLASS_NAMES: dict[int, str] = {
    CLASS_MATRIX: "matrix",
    CLASS_SULFIDE: "sulfide",
    CLASS_GRAY: "gray_ore",
    CLASS_TALC: "talc",
    CLASS_BACKGROUND: "background",
}

# Classes that count toward the VALID area (denominator of the talc fraction).
# Background/resin is excluded so the fraction is "% of the actual thin section".
VALID_CLASSES = (CLASS_MATRIX, CLASS_SULFIDE, CLASS_GRAY, CLASS_TALC)

# ImageNet stats for backbone normalization.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
