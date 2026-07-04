# ШЛИФ · Lab Console — Backend (FastAPI)

Бэкенд лабораторной консоли: приём панорамных шлифов, (далее) инференс и метрики.

## Запуск

```bash
cd back
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

## Эндпоинты

| Метод | Путь | Описание |
|------|------|----------|
| `GET` | `/api/health` | Проверка живости |
| `POST` | `/api/samples` | Загрузка образца (multipart, поле `file`: TIFF/PNG/JPEG) |
| `GET` | `/api/samples/{id}/image` | Отдать загруженный снимок |

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
