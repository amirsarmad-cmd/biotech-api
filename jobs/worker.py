"""
Background worker — pops job IDs off Redis queue, runs LLM work, writes results.
Runs as separate process via supervisord (or can be single-shot-tested with `python -m jobs.worker`).
"""
import os, json, time, logging, traceback
import redis

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s worker: %(message)s')
logger = logging.getLogger("worker")

REDIS_URL = os.getenv("REDIS_URL")
QUEUE_KEY = "biotech-api:job-queue"


def _job_key(job_id): return f"biotech-api:job:{job_id}"


def run_news_impact(payload: dict) -> dict:
    from services.news_npv_impact import analyze_news_npv_impact
    return analyze_news_npv_impact(**payload)


def run_consensus(payload: dict) -> dict:
    from services.ai_pipeline import run_parallel_only, compute_consensus
    # Run all 3 models in parallel
    parallel = run_parallel_only(
        ticker=payload.get("ticker", ""),
        company_name=payload.get("company_name", ""),
        catalyst_info=payload.get("catalyst_info", {}),
        drug_info=payload.get("drug_info", {}),
        sources=payload.get("sources", []),
    )
    # Synthesize consensus
    cons = compute_consensus(parallel)
    return {"parallel": parallel, "consensus": cons}


HANDLERS = {
    "news-impact": run_news_impact,
    "consensus": run_consensus,
}


def process_one(r: redis.Redis, job_id: str):
    key = _job_key(job_id)
    data = r.hgetall(key)
    if not data:
        logger.warning(f"[{job_id}] job expired before processing")
        return
    job_type = data.get("type")
    handler = HANDLERS.get(job_type)
    if not handler:
        r.hset(key, mapping={"status": "failed", "error": f"unknown job type: {job_type}"})
        return
    
    r.hset(key, mapping={"status": "running", "started": time.time()})
    logger.info(f"[{job_id}] running {job_type}")
    
    try:
        payload = json.loads(data.get("payload", "{}"))
        result = handler(payload)
        r.hset(key, mapping={
            "status": "completed",
            "completed": time.time(),
            "result": json.dumps(result, default=str),
        })
        r.expire(key, 3600)
        logger.info(f"[{job_id}] ✓ completed")
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:400]}"
        tb = traceback.format_exc()[:800]
        logger.error(f"[{job_id}] ✗ {err}\n{tb}")
        r.hset(key, mapping={
            "status": "failed",
            "completed": time.time(),
            "error": err,
        })
        r.expire(key, 3600)


def main():
    if not REDIS_URL:
        logger.error("REDIS_URL not set — worker cannot start")
        return
    r = redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("worker started, consuming from %s", QUEUE_KEY)
    while True:
        try:
            # Block up to 5s waiting for a job
            item = r.brpop(QUEUE_KEY, timeout=5)
            if not item:
                continue
            _, job_id = item
            process_one(r, job_id)
        except KeyboardInterrupt:
            logger.info("shutting down")
            break
        except Exception as e:
            logger.exception(f"worker loop error: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()
