"""
/analyze — NPV calc, news-impact (Section 2C), consensus AI.
Long-running operations return a job_id; poll /jobs/{id} for results.
"""
import logging, uuid, time, json, os
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# ─── Simple Redis-backed job queue ──────────────────────────────────────────
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


def enqueue_job(job_type: str, payload: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "type": job_type,
        "status": "queued",
        "created": time.time(),
        "payload": json.dumps(payload),
        "result": "",
        "error": "",
    }
    try:
        redis().hset(_job_key(job_id), mapping=job)
        redis().expire(_job_key(job_id), 3600)  # 1hr
        # Push to processing list (worker will consume)
        redis().lpush("biotech-api:job-queue", job_id)
    except Exception as e:
        logger.exception("enqueue failed")
        raise HTTPException(500, f"queue error: {e}")
    return job_id


# ─── Request models ─────────────────────────────────────────────────────────

class NPVRequest(BaseModel):
    ticker: str
    catalyst_type: str = "FDA Decision"
    peak_sales_b: float = 3.0
    multiple: float = 3.5
    p_commercial: float = 0.6
    market_cap_m: float = 0.0
    cogs_pct: Optional[float] = 0.15
    opex_pct: Optional[float] = 0.25
    tax_rate: Optional[float] = 0.21
    discount_rate: Optional[float] = 0.12
    time_to_peak_years: Optional[float] = 5.0
    loe_years: Optional[float] = 10.0


class NewsImpactRequest(BaseModel):
    ticker: str
    company_name: str = ""
    catalyst_type: str = "FDA Decision"
    catalyst_date: str = ""
    current_price: float
    market_cap_b: float
    drug_npv_b: float
    peak_sales_b: float
    multiple: float
    p_commercial: float
    fundamental_impact_pct: float
    implied_move_pct: float = 0.0
    baseline_days: int = 30
    articles: List[Dict[str, Any]] = Field(default_factory=list)


class ConsensusRequest(BaseModel):
    ticker: str
    company_name: str = ""
    catalyst_info: Dict[str, Any] = Field(default_factory=dict)
    drug_info: Dict[str, Any] = Field(default_factory=dict)
    sources: List[Dict[str, Any]] = Field(default_factory=list)


# ─── Routes ─────────────────────────────────────────────────────────────────

@router.post("/npv")
async def analyze_npv(req: NPVRequest):
    """
    NPV calc. Two-step: 1) estimate drug economics via LLM, 2) compute NPV from economics + market data.
    Heavy — takes 10-30s via Claude. Consider caching.
    """
    try:
        from services.npv_model import compute_npv_estimate, estimate_drug_economics, get_baseline_price
        from services.database import BiotechDatabase
        
        db = BiotechDatabase()
        rows = db.get_stock(req.ticker)
        if not rows:
            raise HTTPException(404, f"Ticker {req.ticker} not found")
        primary = rows[0]
        
        # Step 1: estimate drug economics via Claude (peak sales, multiple, commercial prob)
        economics = estimate_drug_economics(
            ticker=req.ticker,
            company_name=primary.get("company_name", req.ticker),
            catalyst_type=req.catalyst_type,
            catalyst_date=primary.get("catalyst_date", ""),
            description=primary.get("description", ""),
            market_cap_m=req.market_cap_m or float(primary.get("market_cap") or 0),
        )
        
        # Step 2: get current price + baseline — use market_cap as proxy if no price available
        current_price = 50.0  # sensible default; frontend may override
        baseline_price = get_baseline_price(req.ticker) or current_price
        
        # Step 3: compute NPV
        result = compute_npv_estimate(
            ticker=req.ticker,
            current_price=current_price,
            market_cap_m=req.market_cap_m or float(primary.get("market_cap") or 0),
            p_approval=req.p_commercial or (primary.get("probability") or 0.5),
            economics=economics,
            baseline_price=baseline_price,
            info={"catalyst_type": req.catalyst_type,
                  "catalyst_date": primary.get("catalyst_date", ""),
                  "description": primary.get("description", "")},
        )
        return {"ticker": req.ticker, "economics": economics, "npv": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("NPV compute failed")
        raise HTTPException(500, f"NPV error: {e}")


@router.post("/news-impact")
async def analyze_news_impact(req: NewsImpactRequest):
    """
    Section 2C — news × NPV analysis.
    Long-running (multi-provider LLM), so this returns a job_id.
    Poll /jobs/{job_id}.
    """
    job_id = enqueue_job("news-impact", req.model_dump())
    return {"job_id": job_id, "poll_url": f"/jobs/{job_id}"}


@router.post("/consensus")
async def analyze_consensus(req: ConsensusRequest):
    """
    Three-provider AI consensus analysis.
    Long-running — returns job_id.
    """
    job_id = enqueue_job("consensus", req.model_dump())
    return {"job_id": job_id, "poll_url": f"/jobs/{job_id}"}


# Fallback: synchronous Section 2C for quick testing (will block request)
@router.post("/news-impact-sync")
async def analyze_news_impact_sync(req: NewsImpactRequest):
    """Synchronous Section 2C — blocks for ~30-60s. Use the async version in prod."""
    try:
        from services.news_npv_impact import analyze_news_npv_impact
        result = analyze_news_npv_impact(**req.model_dump())
        return {"ticker": req.ticker, "result": result}
    except Exception as e:
        logger.exception("news-impact sync failed")
        raise HTTPException(500, f"news-impact error: {e}")
