# biotech-api: FastAPI backend
FROM python:3.11-slim
LABEL BUILD_TS_BUST=1777152679

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Chromium for Playwright-based paywalled-news fetchers (SA, STAT+).
# --with-deps installs the OS libs Chromium needs (libnss3, libasound2, etc).
# Adds ~250MB to the image but is required for fetch_sa_logged_in / news_stat_plus.
RUN playwright install --with-deps chromium

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Run migrations on startup, then start uvicorn.
# If migrations fail (e.g. DATABASE_URL missing), continue anyway so health check passes.
CMD sh -c "alembic upgrade head || echo 'WARN: alembic upgrade failed, continuing startup'; uvicorn main:app --host 0.0.0.0 --port \${PORT:-8000} --log-level info"
