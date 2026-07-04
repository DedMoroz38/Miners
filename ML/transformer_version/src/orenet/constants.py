"""Global constants: class ids, palette, dataset paths.

Phase classes are the fixed contract shared by every module.
"""

from __future__ import annotations

from pathlib import Path

# --- phase class ids (segmentation output channels) ---
CLASS_MATRIX = 0  # non-ore silicate/oxide matrix (dark)
CLASS_SULFIDE = 1  # sulfides (bright)
CLASS_GRAY = 2  # gray ore phase, e.g. magnetite (mid-gray)
CLASS_TALC = 3  # talc: disseminated dark phase in non-ore matrix
CLASS_BACKGROUND = 4  # mounting resin / field background / defects
IGNORE_INDEX = 255  # pixels excluded from the loss

NUM_CLASSES = 5
CLASS_NAMES: dict[int, str] = {
    CLASS_MATRIX: "matrix",
    CLASS_SULFIDE: "sulfide",
    CLASS_GRAY: "gray_ore",
    CLASS_TALC: "talc",
    CLASS_BACKGROUND: "background",
}

# RGB palette for phase visualisation (matplotlib expects RGB 0..255)
PALETTE: dict[int, tuple[int, int, int]] = {
    CLASS_MATRIX: (60, 60, 60),
    CLASS_SULFIDE: (0, 200, 0),
    CLASS_GRAY: (150, 150, 150),
    CLASS_TALC: (0, 80, 255),
    CLASS_BACKGROUND: (0, 0, 0),
}

# Intergrowth overlay colours (RGB) — the geologist-facing result
COLOR_NORMAL = (0, 200, 0)  # green  = обычные срастания
COLOR_FINE = (220, 30, 30)  # red    = тонкие срастания
COLOR_TALC = (0, 80, 255)  # blue   = тальк

# --- dataset layout ---
DATA_ROOT = Path("data")
PART1 = DATA_ROOT / "ore_photos_part1"
PART2 = DATA_ROOT / "ore_photos_part2"
PANORAMS = DATA_ROOT / "panorams"

# talc pixel supervision: hand-labelled YOLO-seg export (class 0 = talc).
# Produced by ../first_labling_attempt/export_to_yoloseg.py; override in the notebook.
YOLO_SEG_DIR = Path("../first_labling_attempt/yolo_seg")

# legacy blue-contour annotations (kept for reference; no longer used for talc)
TALC_ANNOT_DIR = PART1 / "Оталькованные руды" / "Области оталькования"
TALC_ORIG_DIR = PART1 / "Оталькованные руды"

# folder -> ore sort label (weak, image-level supervision for intergrowth eval)
# "normal" = обычные срастания dominate, "fine" = тонкие, "talc" = оталькованная
SORT_FOLDERS: dict[str, str] = {
    "Оталькованные руды": "talc",
    "Рядовые руды": "normal",
    "Труднообогатимые руды": "fine",
    "оталькованные": "talc",
    "рядовые": "normal",
    "тонкие": "fine",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
