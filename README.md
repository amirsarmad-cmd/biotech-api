# biotech-api

FastAPI backend for the biotech stock screener. Replaces the Streamlit app with a stateless JSON REST API + Redis-backed job queue for long-running LLM work.

## Architecture

- **FastAPI** on uvicorn — `main.py` + `routes/*.py`
- **Background worker** — `jobs/worker.py` consumes Redis queue for LLM jobs
- **Postgres** — shared with aegra, schema `screener_*`
- **Redis** — job queue + caching
- **No WebSockets** — proven unstable on Railway. SSE for streaming (future).

## Service modules (ported from old Streamlit app)

- `services/npv_model.py` — NPV calculation
- `services/news_npv_impact.py` — Section 2C news × NPV analysis (multi-provider LLM)
- `services/ai_pipeline.py` — 3-model consensus synthesis
- `services/strategy.py` — buy/sell/hedge strategies
- `services/risk_factors.py` — Section 2B adverse risk discount
- `services/universe.py` — 70-stock FDA catalyst seeder
- `services/fetcher*.py` — Finnhub, NewsAPI, yfinance data
- `services/social_sources.py` — StockTwits + Reddit
- `services/authenticated_sources.py` — TipRanks + yfinance analyst
- `services/database.py` — Postgres DAO

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Service info + endpoint list |
| GET | `/health` | Health check (DB + Redis status) |
| GET | `/stocks` | Universe list (filters: `high_prob_only`, `min_probability`, `sort`, `limit`) |
| GET | `/stocks/{ticker}` | Full detail: catalysts, NPV, scores |
| GET | `/stocks/{ticker}/news` | Recent news with sentiment |
| GET | `/stocks/{ticker}/social` | StockTwits + Reddit |
| GET | `/stocks/{ticker}/analyst` | TipRanks + yfinance consensus |
| POST | `/analyze/npv` | Sync NPV calc |
| POST | `/analyze/news-impact` | Async Section 2C — returns `job_id` |
| POST | `/analyze/consensus` | Async 3-model consensus — returns `job_id` |
| GET | `/jobs/{job_id}` | Poll job status + result |
| GET | `/strategies/{ticker}` | Buy/sell/hedge strategies |
| POST | `/admin/universe/refresh` | Re-seed 70-stock universe |

## Deploy

Railway auto-detects the Dockerfile via `railway.toml`. Required env vars:

```
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
FINNHUB_API_KEY=...
NEWSAPI_KEY=...
```

## Local dev

```bash
pip install -r requirements.txt
export DATABASE_URL=... REDIS_URL=...
uvicorn main:app --reload
# separate terminal:
python -m jobs.worker
```
