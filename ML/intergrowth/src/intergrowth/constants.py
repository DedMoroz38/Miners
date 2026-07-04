"""Constants for the intergrowth-type classifier (обычные vs тонкие срастания).

Two classes only; this module is for NON-talc ore, where the geologist's rule is
    обычные срастания dominate  -> рядовая руда        (NORMAL)
    тонкие  срастания dominate  -> труднообогатимая     (FINE)

Talc-bearing folders are excluded here (talc is handled by the orenet segmenter).
"""

from __future__ import annotations

from pathlib import Path

# --- class ids ---
NORMAL = 0  # обычные срастания -> рядовая руда
FINE = 1    # тонкие срастания -> труднообогатимая руда
NUM_CLASSES = 2
CLASS_NAMES: dict[int, str] = {NORMAL: "normal", FINE: "fine"}
RU_NAMES: dict[int, str] = {NORMAL: "рядовая", FINE: "труднообогатимая"}

# --- dataset layout (relative to ML/intergrowth/) ---
# Photos live in the shared classification dataset; symlink or edit as needed.
DATA_ROOTS = (
    Path("../first_classification_attempt/data/part1"),
    Path("../first_classification_attempt/data/part2"),
)

# folder name -> class label (Russian folders as they exist on disk).
# Talc folders are intentionally absent so they never enter this task.
FOLDER_TO_LABEL: dict[str, int] = {
    "Рядовые руды": NORMAL,
    "Труднообогатимые руды": FINE,
    "рядовые": NORMAL,
    "тонкие": FINE,
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# --- Method 2 (tile CNN) ---
TILE = 512
TILES_PER_IMAGE = 6  # random tiles drawn per image per epoch

# reuse the orenet package (preprocess / GMM phases / grains) for Method 1
ORENET_SRC = Path("../transformer_version/src")
