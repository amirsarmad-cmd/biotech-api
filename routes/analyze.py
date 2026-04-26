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
    # V2 controls
    force_refresh: bool = False  # bypass drug_economics_cache + catalyst_npv_cache
    drug_name_override: Optional[str] = None  # override heuristic drug-name extraction
    description_override: Optional[str] = None  # override description (useful when caller has more context)
    # Capital structure (per-share NPV)
    dilution_assumed_pct: Optional[float] = None  # 0-75, default 0 (no dilution)
    shares_outstanding_m_override: Optional[float] = None  # if caller has authoritative count


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
    NPV calc — V2 with structured drug economics + true rNPV.
    
    Pipeline:
    1. Look up `catalyst_npv_cache` first (fast path) — return if fresh
    2. Pull/compute structured drug economics (population, pricing, penetration,
       LOE) via `fetch_or_compute_drug_economics_v2`. Reads/writes
       `drug_economics_cache`.
    3. Run BOTH legacy multiple-based NPV AND new year-by-year rNPV.
    4. Persist final payload to `catalyst_npv_cache`.
    
    Response shape (additive, backward compatible):
      { ticker, economics: <legacy>, npv: <legacy>,
        economics_v2: <structured fields + cache flag>,
        rnpv: <year-by-year discounted CF>,
        from_cache: bool }
    """
    try:
        from services.npv_model import (
            compute_npv_estimate, estimate_drug_economics, get_baseline_price,
            fetch_or_compute_drug_economics_v2, compute_rnpv_full,
            load_npv_defaults_from_db, _params_hash, get_npv_cached, write_npv_cached,
        )
        from services.database import BiotechDatabase
        
        db = BiotechDatabase()
        rows = db.get_stock(req.ticker)
        if not rows:
            raise HTTPException(404, f"Ticker {req.ticker} not found")
        primary = rows[0]
        company_name = primary.get("company_name", req.ticker)
        catalyst_date = primary.get("catalyst_date", "") or ""
        description = req.description_override or (primary.get("description", "") or "")
        market_cap_m = req.market_cap_m or float(primary.get("market_cap") or 0)
        
        # Try to extract drug name from description (best-effort)
        drug_name = req.drug_name_override or primary.get("drug_name") or ""
        if not drug_name and description:
            # Heuristic: first parenthetical generic name, else first capitalized word
            import re as _re
            m = _re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', description[:200])
            drug_name = m.group(1) if m else f"{req.ticker}_{req.catalyst_type}"
        
        # ---- Cache key based on inputs ----
        cache_payload = {
            "ticker": req.ticker, "catalyst_type": req.catalyst_type,
            "catalyst_date": catalyst_date, "drug_name": drug_name,
            "discount_rate": req.discount_rate, "tax_rate": req.tax_rate,
            "cogs_pct": req.cogs_pct, "p_approval": req.p_commercial,
            "time_to_peak_years": req.time_to_peak_years,
            "loe_dropoff_pct": None,  # filled after econ_v2
            # Methodology audit fields — different values MUST produce different cache keys
            "dilution_assumed_pct": req.dilution_assumed_pct,
            "shares_outstanding_m_override": req.shares_outstanding_m_override,
            "drug_name_override": req.drug_name_override,
            "description_override": req.description_override,
        }
        params_hash_val = _params_hash(cache_payload)
        
        if not req.force_refresh:
            cached = get_npv_cached(req.ticker, None, params_hash_val)
            if cached:
                cached["from_cache"] = True
                return cached
        
        # ---- Step 1: structured drug economics (V2) ----
        econ_v2 = fetch_or_compute_drug_economics_v2(
            ticker=req.ticker, company_name=company_name, drug_name=drug_name,
            catalyst_type=req.catalyst_type, catalyst_date=catalyst_date,
            description=description, market_cap_m=market_cap_m,
            force_refresh=req.force_refresh,
        )
        
        # ---- Step 2: legacy NPV (kept for back-compat / sanity check) ----
        # Build legacy economics from V2 if possible, else call legacy LLM
        if econ_v2.get("peak_sales_usd_b"):
            legacy_economics = {
                "peak_sales_usd_b": econ_v2.get("peak_sales_usd_b"),
                "peak_sales_year": econ_v2.get("peak_sales_year"),
                "peak_sales_rationale": (econ_v2.get("llm_rationale", "") or "")[:300],
                "multiple": 3.5,  # legacy default; rNPV doesn't use this
                "multiple_rationale": "Default — V2 uses year-by-year DCF instead",
                "commercial_success_prob": econ_v2.get("commercial_success_prob", 0.6),
                "commercial_success_rationale": "From structured V2 estimate",
                "first_in_class": econ_v2.get("first_in_class", False),
                "competitive_intensity": econ_v2.get("competitive_intensity", "medium"),
                "error": econ_v2.get("error"),
            }
        else:
            # V2 produced no peak sales — fall back to legacy LLM call
            legacy_economics = estimate_drug_economics(
                ticker=req.ticker, company_name=company_name,
                catalyst_type=req.catalyst_type, catalyst_date=catalyst_date,
                description=description, market_cap_m=market_cap_m,
            )
        
        # ---- Step 3: compute legacy NPV envelope ----
        current_price = float(primary.get("current_price") or 50.0)
        baseline_price = get_baseline_price(req.ticker) or current_price
        p_approval = req.p_commercial or float(primary.get("probability") or 0.5)
        
        legacy_npv = compute_npv_estimate(
            ticker=req.ticker, current_price=current_price,
            market_cap_m=market_cap_m, p_approval=p_approval,
            economics=legacy_economics, baseline_price=baseline_price,
            info={"catalyst_type": req.catalyst_type,
                  "catalyst_date": catalyst_date,
                  "description": description},
        )
        
        # ---- Step 4: true rNPV (V2) — year-by-year ----
        weights_override = {}
        if req.discount_rate is not None:
            weights_override["discount_rate"] = req.discount_rate
        if req.tax_rate is not None:
            weights_override["tax_rate"] = req.tax_rate
        if req.cogs_pct is not None:
            weights_override["cogs_pct"] = req.cogs_pct
        if req.dilution_assumed_pct is not None:
            weights_override["dilution_assumed_pct"] = req.dilution_assumed_pct

        # Capture shares_outstanding for per-share NPV. Override > yfinance.
        shares_outstanding_m = None
        if req.shares_outstanding_m_override:
            shares_outstanding_m = float(req.shares_outstanding_m_override)
        else:
            try:
                import yfinance as yf
                tkr_yf = yf.Ticker(req.ticker)
                yfi = tkr_yf.info or {}
                so = yfi.get("sharesOutstanding")
                if so:
                    shares_outstanding_m = float(so) / 1e6  # convert to millions
            except Exception as e:
                logger.info(f"shares_outstanding lookup failed for {req.ticker}: {e}")
        if shares_outstanding_m and isinstance(econ_v2, dict):
            econ_v2["shares_outstanding_m"] = shares_outstanding_m

        rnpv = compute_rnpv_full(
            econ_v2=econ_v2, p_approval=p_approval,
            market_cap_m=market_cap_m,
            weights=weights_override or None,
        )
        
        # ---- Step 5: persist to cache ----
        result_payload = {
            "ticker": req.ticker,
            "drug_name": drug_name,
            "economics": legacy_economics,
            "npv": legacy_npv,
            "economics_v2": econ_v2,
            "rnpv": rnpv,
            "from_cache": False,
        }
        try:
            write_npv_cached(req.ticker, None, params_hash_val, result_payload, ttl_days=1)
        except Exception as e:
            logger.warning(f"NPV cache write failed (non-fatal): {e}")
        
        return result_payload
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
