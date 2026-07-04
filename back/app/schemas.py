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


class SegmentOut(BaseModel):
    """Сегмент фазовой маски. Полигоны в НОРМАЛИЗОВАННЫХ координатах (0..1)."""

    id: str
    phase: str  # "common" | "thin" | "talc"
    confidence: float
    areaFrac: float
    polygons: list[list[list[float]]]  # список колец: [ [ [x,y], ... ], ... ]


class AnalysisResultOut(BaseModel):
    """Результат анализа — контракт под фронтенд (entities/analysis: AnalysisResult)."""

    sulfideShare: float  # % площади (обычные + тонкие)
    commonShare: float   # % от сульфидов
    thinShare: float     # % от сульфидов
    talcShare: float     # % площади всего кадра
    verdict: str
    conclusion: str
    f1: float
    imageWidth: int
    imageHeight: int
    segments: list[SegmentOut]


class AnalyzeJobOut(BaseModel):
    """Ответ на запуск анализа: идентификатор фонового задания."""

    job_id: str
    status: str  # pending | running | done | error


class JobStatusOut(BaseModel):
    """Статус задания анализа (для опроса фронтендом)."""

    job_id: str
    status: str  # pending | running | done | error
    result: AnalysisResultOut | None = None
    error: str | None = None
