import os
import sys
from pathlib import Path

# Каталог для загруженных образцов (лежит рядом с пакетом app).
STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"

# Рабочие каталоги ML-заданий (talc/sulfide/merge на одно задание).
JOBS_DIR = Path(__file__).resolve().parent.parent / "jobs"

# Каталог с DZI-пирамидами тайлов для OpenSeadragon (по одной на образец).
TILES_DIR = Path(__file__).resolve().parent.parent / "tiles"

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

# Python-интерпретаторы конвейеров. Приоритет: переменная окружения ->
# venv конвейера (если существует) -> текущий интерпретатор бэкенда (fallback,
# удобно когда torch/ultralytics установлены в тот же python, что и API).
def _pipeline_python(env_name: str, venv_python: Path) -> str:
    override = os.environ.get(env_name)
    if override:
        return override
    return str(venv_python) if venv_python.is_file() else sys.executable


TALC_PYTHON = _pipeline_python(
    "TALC_PYTHON", ML_ROOT / "first_labling_attempt" / ".venv" / "bin" / "python")
SULFIDE_PYTHON = _pipeline_python(
    "SULFIDE_PYTHON", ML_ROOT / "sulfide_intergrowth" / ".venv" / "bin" / "python")
# merge требует только numpy+opencv — по умолчанию берём тот же python, что и talc.
MERGE_PYTHON = os.environ.get("MERGE_PYTHON") or TALC_PYTHON

# Точки входа конвейеров.
# TALC: новый самодостаточный модуль (не трогает папки с обучением моделей;
# импортирует ML/fusion/fuse.py и ставит shim совместимости ultralytics).
TALC_SCRIPT = str(ML_ROOT / "talc_infer" / "infer.py")
SULFIDE_SCRIPT = str(ML_ROOT / "sulfide_intergrowth" / "run" / "pipeline" / "infer_panorama.py")
SULFIDE_CWD = str(ML_ROOT / "sulfide_intergrowth")
MERGE_SCRIPT = str(ML_ROOT / "merge" / "merge_phases.py")

# Каталог с загруженными весами сульфидного конвейера (см. weights/README.md).
SULFIDE_WEIGHTS_DIR = os.environ.get("SULFIDE_WEIGHTS_DIR") or str(
    ML_ROOT / "sulfide_intergrowth" / "weights")

# Веса талько-сегментера (вариант B — обучен в каноническом сульфидном профиле,
# см. ML/talc_seg_normalized/weights/README.md). Приоритет: env -> файл весов
# варианта B (если выложен) -> None (тогда talc_infer сам берёт сырой бейзлайн
# и, по связке профиль<->веса, НЕ нормализует вход).
_TALC_NORM_WEIGHTS = ML_ROOT / "talc_seg_normalized" / "weights" / "talc_seg_best.pt"
_talc_env = os.environ.get("TALC_SEG_WEIGHTS")
# Передаём --seg-weights только если файл реально существует: иначе talc_infer
# аварийно выйдет, а без аргумента он мягко откатится на сырой бейзлайн.
TALC_SEG_WEIGHTS = next(
    (str(p) for p in (_talc_env, _TALC_NORM_WEIGHTS) if p and Path(p).is_file()),
    None)


def sulfide_available() -> bool:
    """Сульфидный шаг запускается только если есть его веса (иначе — talc-only).

    Требуются segmenter-веса и обязательный decision.json (см. weights/README.md).
    Пока весов нет — конвейер работает в режиме только-тальк.
    """
    if os.environ.get("SKIP_SULFIDE") == "1":
        return False
    d = Path(SULFIDE_WEIGHTS_DIR)
    has_segmenter = any(d.glob("segmenter*.pt")) if d.is_dir() else False
    has_decision = (d / "decision.json").is_file()
    return has_segmenter and has_decision

# Таймаут одного шага инференса (сек).
ML_STEP_TIMEOUT = int(os.environ.get("ML_STEP_TIMEOUT", "1800"))

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
TILES_DIR.mkdir(parents=True, exist_ok=True)
