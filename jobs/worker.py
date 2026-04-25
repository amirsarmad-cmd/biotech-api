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


def _build_context_string(payload: dict) -> str:
    """ai_pipeline.run_parallel_only expects a single formatted text 'context' string,
    not separate kwargs. Build it from our consensus payload."""
    ticker = payload.get("ticker", "")
    company = payload.get("company_name", ticker)
    cat = payload.get("catalyst_info", {}) or {}
    drug = payload.get("drug_info", {}) or {}
    sources = payload.get("sources", []) or []
    base_prob = cat.get("probability") or drug.get("commercial_prob") or 0.5
    base_pct = int(round(base_prob * 100))

    lines = [
        f"TICKER: {ticker}",
        f"COMPANY: {company}",
        f"PROBABILITY: {base_pct}%   (current base estimate)",
        "",
        "CATALYST",
        f"  Type: {cat.get('type','')}",
        f"  Date: {cat.get('date','')}",
        f"  Description: {cat.get('description','')}",
        "",
        "DRUG ECONOMICS",
        f"  Peak sales: ${drug.get('peak_sales_b','?')}B",
        f"  Multiple: {drug.get('multiple','?')}x",
        f"  Commercial prob (if approved): {drug.get('commercial_prob','?')}",
        f"  Peak sales rationale: {drug.get('peak_sales_rationale','')}",
        f"  Multiple rationale: {drug.get('multiple_rationale','')}",
        f"  Commercial rationale: {drug.get('commercial_rationale','')}",
    ]
    if sources:
        lines.append("")
        lines.append("RECENT NEWS / SOURCES (top 8)")
        for s in sources[:8]:
            title = (s.get("title") or "")[:140]
            src = s.get("source") or "?"
            date = s.get("date") or "?"
            lines.append(f"  [{src} {date}] {title}")
    return "\n".join(lines)


def run_consensus(payload: dict) -> dict:
    from services.ai_pipeline import run_parallel_only, compute_consensus
    context = _build_context_string(payload)
    parallel = run_parallel_only(context=context, question=payload.get("question", ""))
    cons = compute_consensus(parallel)
    return {"parallel": parallel, "consensus": cons, "context_used": context[:1500]}


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
