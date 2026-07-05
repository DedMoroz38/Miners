from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import config
from app.api.samples import router as samples_router

app = FastAPI(title="ШЛИФ · Lab Console API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(samples_router)

# DZI-тайлы для OpenSeadragon: <base>/tiles/<id>/img.dzi + img_files/... .
# OpenSeadragon сам достраивает URL тайлов относительно .dzi-дескриптора.
app.mount("/tiles", StaticFiles(directory=str(config.TILES_DIR)), name="tiles")


@app.get("/api/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}
