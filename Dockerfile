# biotech-api: FastAPI backend + background worker
# Stateless HTTP JSON API. No Streamlit. No WebSockets.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# supervisord runs both uvicorn (API) and the background worker
COPY supervisord.conf /etc/supervisor/conf.d/biotech.conf
RUN chmod +x /app/start.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Use sh explicitly so even without exec bit it runs
CMD ["sh", "/app/start.sh"]
