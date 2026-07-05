# ШЛИФ · Lab Console

Полный стек: FastAPI-бэкенд (`back/`) + Next.js-фронтенд (`front/`) для анализа
панорамных шлифов (talc + sulfide + merge ML-конвейеры лежат в `ML/`).

## Запуск локально (без Docker)

Нужны два терминала — бэкенд и фронтенд запускаются отдельно.

**1. Backend (FastAPI, порт 8000):**

```bash
cd back
python -m venv .venv            # если venv ещё нет
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

**2. Frontend (Next.js, порт 3000):**

```bash
cd front
npm install                     # если node_modules ещё нет
npm run dev
```

Приложение: http://localhost:3000

Фронтенд обращается к бэкенду по `NEXT_PUBLIC_API_URL` (по умолчанию
`http://localhost:8000`, см. `front/.env*` при необходимости переопределить).

Без подготовленных ML venv-ов/весов (см. ниже) API всё равно поднимается и
загрузка образцов работает — упадёт только шаг `/analyze` (ошибка вернётся в
статусе задания, сервис остаётся живым).

## Запуск через Docker Compose (весь стек)

Из корня репозитория:

```bash
docker compose up --build       # back (:8000) + front (:3000)
```

Перед этим на хосте (Linux) должны быть подготовлены venv-ы и веса ML-конвейеров
(они не лежат в git — слишком тяжёлые):

- `ML/first_labling_attempt/.venv` — talc: torch + ultralytics + classifier
- `ML/sulfide_intergrowth/.venv` — sulfide: torch + segmentation-models-pytorch
- веса — см. `ML/sulfide_intergrowth/weights/README.md`

Подробности про переменные окружения (`ML_ROOT`, `TALC_PYTHON`, `SULFIDE_PYTHON`,
`MERGE_PYTHON`, `SULFIDE_WEIGHTS_DIR`, `TALC_SEG_WEIGHTS`) — в `docker-compose.yaml`
и `back/README.md`.

## Дальше

- API-эндпоинты и контракт анализа — `back/README.md`
- ML-конвейеры (talc/sulfide/merge/fusion) — `ML/`
