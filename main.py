"""
Biotech Stock Screener API — FastAPI backend.
Replaces the Streamlit app with JSON REST + Redis-backed async jobs.
"""
import os, logging, asyncio, json, traceback, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routes import stocks, analyze, jobs as jobs_router, strategies, admin, shortlist, universe, catalyst_risks, post_catalyst, llm_usage

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("biotech-api")
VERSION = os.getenv("VERSION", "1.0.0")


async def _background_worker():
    """Single-task worker that consumes Redis job queue and runs LLM jobs."""
    import redis as _redis
    url = os.getenv("REDIS_URL")
    if not url:
        logger.warning("REDIS_URL not set — worker idle")
        return
    r = _redis.from_url(url, decode_responses=True)
    logger.info("background worker started")
    while True:
        try:
            # BRPOP blocks; run in thread to not block the event loop
            item = await asyncio.to_thread(r.brpop, "biotech-api:job-queue", 5)
            if not item:
                continue
            _, job_id = item
            await asyncio.to_thread(_process_job, r, job_id)
        except asyncio.CancelledError:
            logger.info("background worker cancelled")
            break
        except Exception as e:
            logger.exception(f"worker loop: {e}")
            await asyncio.sleep(2)


def _process_job(r, job_id):
    key = f"biotech-api:job:{job_id}"
    data = r.hgetall(key)
    if not data: return
    job_type = data.get("type")
    r.hset(key, mapping={"status": "running", "started": time.time()})
    logger.info(f"[{job_id}] running {job_type}")
    try:
        payload = json.loads(data.get("payload","{}"))
        if job_type == "news-impact":
            from services.news_npv_impact import analyze_news_npv_impact
            result = analyze_news_npv_impact(**payload)
        elif job_type == "consensus":
            from services.ai_pipeline import run_parallel_only, compute_consensus
            # Build text context for ai_pipeline
            cat = payload.get("catalyst_info", {}) or {}
            drug = payload.get("drug_info", {}) or {}
            sources = payload.get("sources", []) or []
            base_pct = int(round((cat.get("probability") or drug.get("commercial_prob") or 0.5) * 100))
            ctx_lines = [
                f"TICKER: {payload.get('ticker','')}",
                f"COMPANY: {payload.get('company_name','')}",
                f"PROBABILITY: {base_pct}%",
                "",
                f"CATALYST: {cat.get('type','')} on {cat.get('date','')}",
                f"  Description: {cat.get('description','')}",
                "",
                f"DRUG ECONOMICS: peak ${drug.get('peak_sales_b','?')}B × {drug.get('multiple','?')}x | p_commercial {drug.get('commercial_prob','?')}",
                f"  Peak sales rationale: {drug.get('peak_sales_rationale','')}",
                f"  Multiple rationale: {drug.get('multiple_rationale','')}",
                f"  Commercial rationale: {drug.get('commercial_rationale','')}",
            ]
            if sources:
                ctx_lines.append("\nRECENT NEWS (top 8):")
                for s in sources[:8]:
                    ctx_lines.append(f"  [{s.get('source','?')} {s.get('date','?')}] {(s.get('title') or '')[:140]}")
            context = "\n".join(ctx_lines)
            parallel = run_parallel_only(context=context, question=payload.get("question",""))
            cons = compute_consensus(parallel)
            result = {"parallel": parallel, "consensus": cons}
        else:
            raise ValueError(f"unknown job type: {job_type}")
        r.hset(key, mapping={
            "status": "completed",
            "completed": time.time(),
            "result": json.dumps(result, default=str),
        })
        r.expire(key, 3600)
        logger.info(f"[{job_id}] completed")
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:400]}"
        logger.error(f"[{job_id}] failed: {err}")
        r.hset(key, mapping={"status": "failed", "completed": time.time(), "error": err})
        r.expire(key, 3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"biotech-api {VERSION} starting")
    from services.database import BiotechDatabase
    try:
        db = BiotechDatabase()
        db.ensure_schema()
        logger.info("DB schema ensured")
    except Exception as e:
        logger.error(f"DB init failed: {e}")
    # Spawn background worker
    worker_task = asyncio.create_task(_background_worker())
    try:
        yield
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        logger.info("biotech-api shutting down")


app = FastAPI(title="Biotech Stock Screener API", version=VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "biotech-api", "version": VERSION, "docs": "/docs",
        "endpoints": [
            "GET /health", "GET /stocks", "GET /stocks/{ticker}",
            "GET /stocks/{ticker}/news", "GET /stocks/{ticker}/social",
            "GET /stocks/{ticker}/analyst",
            "POST /analyze/npv", "POST /analyze/news-impact", "POST /analyze/consensus",
            "GET /jobs/{job_id}",
            "GET /strategies/{ticker}",
            "POST /admin/universe/refresh",
        ],
    }


@app.get("/health")
async def health():
    from services.database import BiotechDatabase
    db_ok = False; redis_ok = False
    try:
        db = BiotechDatabase()
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1"); db_ok = cur.fetchone() is not None
    except Exception as e:
        logger.warning(f"DB health: {e}")
    try:
        import redis
        url = os.getenv("REDIS_URL")
        if url:
            r = redis.from_url(url, socket_connect_timeout=3); r.ping(); redis_ok = True
    except Exception as e:
        logger.warning(f"Redis health: {e}")
    return {
        "status": "healthy" if db_ok else "degraded",
        "version": VERSION,
        "db": "connected" if db_ok else "disconnected",
        "redis": "connected" if redis_ok else "disconnected",
    }


app.include_router(stocks.router, prefix="/stocks", tags=["stocks"])
app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])
app.include_router(jobs_router.router, prefix="/jobs", tags=["jobs"])
app.include_router(strategies.router, prefix="/strategies", tags=["strategies"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(shortlist.router, prefix="/shortlist", tags=["shortlist"])
app.include_router(universe.router, prefix="/universe", tags=["universe"])
app.include_router(catalyst_risks.router, tags=["catalyst_risks"])
app.include_router(post_catalyst.router, tags=["post_catalyst"])
app.include_router(llm_usage.router, tags=["llm_usage"])


@app.exception_handler(Exception)
async def generic_error(request: Request, exc: Exception):
    logger.exception(f"Unhandled on {request.url.path}: {exc}")
    return JSONResponse(status_code=500,
        content={"error": type(exc).__name__, "message": str(exc)[:300]})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
