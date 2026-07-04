# FastAPI backend for ШЛИФ · Lab Console.
#
# This image serves the API only. The heavy ML pipelines run as subprocesses in
# their OWN venvs (see app/config.py) — those venvs + weights live under the ML/
# tree, which is mounted in at runtime (see docker-compose.yaml). Baking multi-GB
# torch/ultralytics venvs into the API image is impractical, so we keep them out.
FROM python:3.12-slim

WORKDIR /app

# Pillow reads gigapixel panoramas; no build tools needed (wheels only).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# storage/ and jobs/ are created at import by app.config (relative to /app).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
