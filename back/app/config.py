import os
from pathlib import Path

# Каталог для загруженных образцов (лежит рядом с пакетом app).
STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"

# Рабочие каталоги ML-заданий (talc/sulfide/merge на одно задание).
JOBS_DIR = Path(__file__).resolve().parent.parent / "jobs"

# Форматы панорамных снимков, которые принимает лаборатория.
ALLOWED_CONTENT_TYPES = {"image/tiff", "image/png", "image/jpeg"}
ALLOWED_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Панорамные шлифы бывают гигапиксельными — держим большой потолок.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024  # 512 МБ

# Локальный фронтенд Next.js.
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

# --- ML-конвейеры (каждый в своём venv; бэкенд вызывает их как subprocess) ----
# Корень с ML-кодом (back/ и ML/ лежат в корне репозитория).
ML_ROOT = Path(os.environ.get("ML_ROOT") or (Path(__file__).resolve().parents[2] / "ML"))

# Python-интерпретаторы venv-ов конвейеров (переопределяются переменными окружения).
TALC_PYTHON = os.environ.get("TALC_PYTHON") or str(
    ML_ROOT / "first_labling_attempt" / ".venv" / "bin" / "python")
SULFIDE_PYTHON = os.environ.get("SULFIDE_PYTHON") or str(
    ML_ROOT / "sulfide_intergrowth" / ".venv" / "bin" / "python")
# merge требует только numpy+opencv — по умолчанию берём venv talc.
MERGE_PYTHON = os.environ.get("MERGE_PYTHON") or TALC_PYTHON

# Точки входа конвейеров.
TALC_SCRIPT = str(ML_ROOT / "fusion" / "infer_one.py")
SULFIDE_SCRIPT = str(ML_ROOT / "sulfide_intergrowth" / "run" / "pipeline" / "infer_panorama.py")
SULFIDE_CWD = str(ML_ROOT / "sulfide_intergrowth")
MERGE_SCRIPT = str(ML_ROOT / "merge" / "merge_phases.py")

# Каталог с загруженными весами сульфидного конвейера (см. weights/README.md).
SULFIDE_WEIGHTS_DIR = os.environ.get("SULFIDE_WEIGHTS_DIR") or str(
    ML_ROOT / "sulfide_intergrowth" / "weights")

# Таймаут одного шага инференса (сек).
ML_STEP_TIMEOUT = int(os.environ.get("ML_STEP_TIMEOUT", "1800"))

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
