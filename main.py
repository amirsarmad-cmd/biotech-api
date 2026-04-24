"""
Biotech Stock Screener API — FastAPI backend.
Replaces the Streamlit app with a proper JSON REST API + SSE for streaming LLM.
"""
import os, logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routes import stocks, analyze, jobs as jobs_router, strategies, admin

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("biotech-api")

VERSION = os.getenv("VERSION", "1.0.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm DB connection, ensure schema
    logger.info(f"biotech-api {VERSION} starting")
    from services.database import BiotechDatabase
    try:
        db = BiotechDatabase()
        db.ensure_schema()
        logger.info("DB schema ensured")
    except Exception as e:
        logger.error(f"DB init failed: {e}")
    yield
    # Shutdown
    logger.info("biotech-api shutting down")


app = FastAPI(
    title="Biotech Stock Screener API",
    description="FDA catalyst screener with LLM-driven NPV analysis and consensus AI",
    version=VERSION,
    lifespan=lifespan,
)

# CORS: allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "biotech-api",
        "version": VERSION,
        "docs": "/docs",
        "endpoints": [
            "GET /health",
            "GET /stocks",
            "GET /stocks/{ticker}",
            "GET /stocks/{ticker}/news",
            "GET /stocks/{ticker}/social",
            "GET /stocks/{ticker}/analyst",
            "POST /analyze/npv",
            "POST /analyze/news-impact",
            "POST /analyze/consensus",
            "GET /jobs/{job_id}",
            "GET /strategies/{ticker}",
            "POST /admin/universe/refresh",
        ],
    }


@app.get("/health")
async def health():
    from services.database import BiotechDatabase
    db_ok = False
    redis_ok = False
    try:
        db = BiotechDatabase()
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                db_ok = cur.fetchone() is not None
    except Exception as e:
        logger.warning(f"DB health check failed: {e}")
    try:
        import redis
        url = os.getenv("REDIS_URL")
        if url:
            r = redis.from_url(url, socket_connect_timeout=3)
            r.ping()
            redis_ok = True
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
    return {
        "status": "healthy" if db_ok else "degraded",
        "version": VERSION,
        "db": "connected" if db_ok else "disconnected",
        "redis": "connected" if redis_ok else "disconnected",
    }


# Routes
app.include_router(stocks.router, prefix="/stocks", tags=["stocks"])
app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])
app.include_router(jobs_router.router, prefix="/jobs", tags=["jobs"])
app.include_router(strategies.router, prefix="/strategies", tags=["strategies"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])


# Catch-all error handler so stack traces don't leak
@app.exception_handler(Exception)
async def generic_error(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "message": str(exc)[:300]},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
