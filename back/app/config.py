from pathlib import Path

# Каталог для загруженных образцов (лежит рядом с пакетом app).
STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"

# Форматы панорамных снимков, которые принимает лаборатория.
ALLOWED_CONTENT_TYPES = {"image/tiff", "image/png", "image/jpeg"}
ALLOWED_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Панорамные шлифы бывают гигапиксельными — держим большой потолок.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024  # 512 МБ

# Локальный фронтенд Next.js.
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
