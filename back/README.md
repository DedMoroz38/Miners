# ШЛИФ · Lab Console — Backend (FastAPI)

Бэкенд лабораторной консоли: приём панорамных шлифов, (далее) инференс и метрики.

## Запуск (локально)

```bash
cd back
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

## Запуск (Docker Compose, весь стек)

Из корня репозитория:

```bash
docker compose up --build      # поднимает back (:8000) + front (:3000)
```

Бэкенд-образ (`back/Dockerfile`) обслуживает только API. Тяжёлые ML-конвейеры
запускаются как подпроцессы в СВОИХ venv-ах (см. `app/config.py`), поэтому
`docker-compose.yaml` монтирует дерево `ML/` и передаёт пути к venv/весам через
переменные окружения (`ML_ROOT`, `TALC_PYTHON`, `SULFIDE_PYTHON`, `MERGE_PYTHON`,
`SULFIDE_WEIGHTS_DIR`). venv-ы и веса **не** лежат в git — их нужно подготовить на
хосте заранее (см. `ML/sulfide_intergrowth/weights/README.md`).

**Graceful degradation:** без venv/весов API стартует и загрузка образцов
работает; падает только шаг `/analyze` — ошибка возвращается в статусе задания
(`status: "error"`), сервис остаётся живым.

## Эндпоинты

| Метод | Путь | Описание |
|------|------|----------|
| `GET` | `/api/health` | Проверка живости |
| `POST` | `/api/samples` | Загрузка образца (multipart, поле `file`: TIFF/PNG/JPEG) |
| `GET` | `/api/samples/{id}/image` | Отдать загруженный снимок |
| `POST` | `/api/samples/{id}/analyze` | Запустить анализ (talc + sulfide + merge) → `job_id` |
| `GET` | `/api/samples/{id}/analyze/{job_id}` | Статус задания; при `done` — `AnalysisResult` |

Анализ асинхронный: `POST .../analyze` ставит фоновое задание (202 + `job_id`),
фронтенд опрашивает `GET .../analyze/{job_id}` до `status: "done"` и получает
`AnalysisResult` (доли фаз, вердикт, заключение, сегменты в нормализованных
координатах). Контракт совпадает с `entities/analysis` на фронтенде.

## Тесты

```bash
python ML/merge/test_merge_phases.py    # правила слияния фаз (перекрытия, метрики) — без весов
cd back && python test_api_contract.py  # контракт upload→analyze→poll→result — без весов
```

## Известные мелочи

- В диапазоне талька 9.5–9.99% текстовый вывод показывает «тальк — 10% (≤10%)»
  (форматирование `{:.0f}` округляет 9.5→10). Правило одно и то же в
  `ML/merge/merge_phases.py:classify` и `front/.../metrics.ts:classify` — при
  правке менять обе стороны, чтобы строковый контракт совпадал.

### Пример загрузки

```bash
curl -F "file=@sample.png" http://localhost:8000/api/samples
```

Ответ (`SampleOut`):

```json
{
  "id": "a1b2c3d4e5f6",
  "name": "sample",
  "meta": "Загружен · PNG · 3.2 МБ · 8000×6000 px",
  "size_bytes": 3355443,
  "content_type": "image/png",
  "width": 8000,
  "height": 6000,
  "image_url": "http://localhost:8000/api/samples/a1b2c3d4e5f6/image"
}
```

Файлы складываются в `storage/` (в git не попадает).
