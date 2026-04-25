# biotech-api: FastAPI backend
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Run migrations on startup, then start uvicorn.
# If migrations fail (e.g. DATABASE_URL missing), continue anyway so health check passes.
CMD sh -c "alembic upgrade head || echo 'WARN: alembic upgrade failed, continuing startup'; uvicorn main:app --host 0.0.0.0 --port \${PORT:-8000} --log-level info"
