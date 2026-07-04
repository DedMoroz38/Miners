import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from app import config
from app.schemas import AnalysisResultOut, AnalyzeJobOut, JobStatusOut, SampleOut
from app.services import jobs
from app.services.analysis import analyze_sample

# Гигапиксельные шлифы — снимаем защиту Pillow от «бомбы» (только для чтения размеров).
Image.MAX_IMAGE_PIXELS = None

router = APIRouter(prefix="/api/samples", tags=["samples"])


def _read_dimensions(path: Path) -> tuple[int | None, int | None]:
    """Размеры изображения из заголовка (без полной декодировки)."""
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None, None


def _stored_file(sample_id: str) -> Path | None:
    matches = sorted(config.STORAGE_DIR.glob(f"{sample_id}.*"))
    return matches[0] if matches else None


@router.post("", response_model=SampleOut, status_code=201)
async def upload_sample(request: Request, file: UploadFile = File(...)) -> SampleOut:
    """Приём панорамного шлифа (multipart/form-data, поле `file`)."""
    ext = Path(file.filename or "").suffix.lower()
    if file.content_type not in config.ALLOWED_CONTENT_TYPES and ext not in config.ALLOWED_EXT:
        raise HTTPException(
            status_code=415,
            detail="Неподдерживаемый формат. Ожидается TIFF, PNG или JPEG.",
        )

    sample_id = uuid.uuid4().hex[:12]
    dest = config.STORAGE_DIR / f"{sample_id}{ext or '.img'}"

    # Пишем потоково, обрывая слишком большие файлы.
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > config.MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Файл слишком большой.")
            out.write(chunk)

    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Пустой файл.")

    width, height = _read_dimensions(dest)

    size_mb = size / (1024 * 1024)
    ext_label = (ext.lstrip(".") or "img").upper()
    meta = f"Загружен · {ext_label} · {size_mb:.1f} МБ"
    if width and height:
        meta += f" · {width}×{height} px"

    image_url = str(request.url_for("get_sample_image", sample_id=sample_id))

    return SampleOut(
        id=sample_id,
        name=Path(file.filename or sample_id).stem,
        meta=meta,
        size_bytes=size,
        content_type=file.content_type or "application/octet-stream",
        width=width,
        height=height,
        image_url=image_url,
    )


@router.get("/{sample_id}/image", name="get_sample_image")
def get_sample_image(sample_id: str) -> FileResponse:
    """Отдаёт ранее загруженный снимок образца."""
    path = _stored_file(sample_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Образец не найден.")
    return FileResponse(path)


@router.post("/{sample_id}/analyze", response_model=AnalyzeJobOut, status_code=202)
def analyze(sample_id: str) -> AnalyzeJobOut:
    """Ставит анализ образца в очередь и возвращает id фонового задания."""
    path = _stored_file(sample_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Образец не найден.")
    job_id = jobs.create_job()
    jobs.submit(job_id, analyze_sample, path)
    return AnalyzeJobOut(job_id=job_id, status="pending")


@router.get("/{sample_id}/analyze/{job_id}", response_model=JobStatusOut)
def analyze_status(sample_id: str, job_id: str) -> JobStatusOut:
    """Статус задания анализа (опрашивается фронтендом до status='done')."""
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задание не найдено.")
    result = AnalysisResultOut(**job["result"]) if job.get("result") else None
    return JobStatusOut(job_id=job_id, status=job["status"], result=result,
                        error=job.get("error"))
