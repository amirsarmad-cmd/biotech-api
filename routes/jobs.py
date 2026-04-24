"""
/jobs/{id} — poll async job status.
Frontend polls every ~2s until status in ('completed','failed').
"""
import logging, json, os
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()

import redis as _redis_mod

_redis = None
def redis():
    global _redis
    if _redis is None:
        url = os.getenv("REDIS_URL")
        if not url:
            raise RuntimeError("REDIS_URL not set")
        _redis = _redis_mod.from_url(url, decode_responses=True)
    return _redis


def _job_key(job_id): return f"biotech-api:job:{job_id}"


@router.get("/{job_id}")
async def get_job(job_id: str):
    try:
        data = redis().hgetall(_job_key(job_id))
    except Exception as e:
        logger.exception("redis get failed")
        raise HTTPException(500, f"queue error: {e}")
    
    if not data:
        raise HTTPException(404, "job not found (may have expired)")
    
    result = {}
    if data.get("result"):
        try:
            result = json.loads(data["result"])
        except Exception:
            result = {"raw": data["result"]}
    
    return {
        "job_id": job_id,
        "type": data.get("type"),
        "status": data.get("status", "unknown"),
        "created": float(data.get("created") or 0),
        "started": float(data.get("started") or 0) if data.get("started") else None,
        "completed": float(data.get("completed") or 0) if data.get("completed") else None,
        "result": result if data.get("status") == "completed" else None,
        "error": data.get("error") or None,
    }
