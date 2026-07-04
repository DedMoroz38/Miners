from pydantic import BaseModel


class SampleOut(BaseModel):
    """Загруженный образец. Контракт под фронтенд (entities/sample)."""

    id: str
    name: str
    meta: str
    size_bytes: int
    content_type: str
    width: int | None = None
    height: int | None = None
    image_url: str


class AnalyzeOut(BaseModel):
    """Результат анализа образца."""

    id: str
    result: str
