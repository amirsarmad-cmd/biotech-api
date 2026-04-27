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
    # Probability fields — IMPORTANT: these are different probabilities and
    # should not be conflated. Names match the V2 economic model:
    #   - p_approval        = P(catalyst event resolves favorably): for FDA Decision,
    #                         P(approved | event occurs); for Phase 3, P(positive readout).
    #                         Most analogous to BioMedTracker LBT score.
    #   - p_commercial      = P(strong commercial uptake | favorable outcome):
    #                         answers "even if it works, will it sell?"
    # rNPV math: NPV × p_approval × p_commercial (separate haircuts).
    p_approval: Optional[float] = None        # primary — preferred input
    p_commercial: Optional[float] = None      # secondary — defaults to LLM econ_v2.commercial_success_prob
    # DEPRECATED: legacy field that confused the two. If only p_commercial_legacy
    # is supplied (e.g., older clients), it's treated as p_approval for backwards
    # compatibility — but a deprecation warning is logged.
    p_commercial_legacy: Optional[float] = None  # deprecated alias
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
        # IMPORTANT: bump _schema_version whenever the response shape adds
        # new fields. Old cached entries will then be ignored (they lack the
        # new fields) and the next call writes a fresh response.
        cache_payload = {
            "_schema_version": "v4_confidence_breakdown",  # bumped 2026-04-27
            "ticker": req.ticker, "catalyst_type": req.catalyst_type,
            "catalyst_date": catalyst_date, "drug_name": drug_name,
            "discount_rate": req.discount_rate, "tax_rate": req.tax_rate,
            "cogs_pct": req.cogs_pct,
            # Use the explicit names — separate cache entries per (p_approval, p_commercial) pair
            "p_approval": req.p_approval if req.p_approval is not None else req.p_commercial_legacy,
            "p_commercial": req.p_commercial,
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

        # ---- Step 1.5: Enforce source precedence (ChatGPT pass-3 critique #3) ─
        # FDA / SEC / ClinicalTrials / Orange Book facts MUST override LLM
        # inference on the same field. Without this, the prompt receives
        # verified_facts but the LLM's structured output can still 'win' —
        # patent_expiry_date='2030-05-15' from LLM guess vs Orange Book
        # actual = '2026-08-21' is a serious gap. This step rewrites
        # specific fields with verified-source values + records audit.
        precedence_audit = None
        try:
            # Pre-fetch SEC capital structure so precedence can use shares + runway.
            # Best-effort — failures here don't break NPV computation.
            try:
                from services.sec_financials import fetch_capital_structure
                cap_pre = fetch_capital_structure(req.ticker)
                if cap_pre and not cap_pre.get("_error"):
                    econ_v2["_sec_capital_structure"] = cap_pre
            except Exception:
                pass
            from services.source_precedence import enforce_source_precedence
            econ_v2, precedence_audit = enforce_source_precedence(econ_v2, ticker=req.ticker)
        except Exception as e:
            logger.info(f"source precedence enforcement failed (non-fatal): {e}")

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
        # Resolve P(approval) — separate from P(commercial success).
        #   1. explicit req.p_approval (preferred)
        #   2. req.p_commercial_legacy (deprecated — emit warning)
        #   3. catalyst_universe.p_positive_outcome (preferred over legacy probability)
        #   4. catalyst_universe.probability (legacy)
        #   5. default 0.5
        p_approval = None
        p_approval_source = None
        if req.p_approval is not None:
            p_approval = float(req.p_approval)
            p_approval_source = "user_override"
        elif req.p_commercial_legacy is not None:
            logger.warning(
                f"NPVRequest using deprecated p_commercial_legacy={req.p_commercial_legacy} as p_approval. "
                f"Update caller to use p_approval explicitly. ticker={req.ticker}"
            )
            p_approval = float(req.p_commercial_legacy)
            p_approval_source = "legacy_alias"
        elif primary.get("p_positive_outcome") is not None:
            # Prefer the split-probability field over legacy 'probability'.
            # alembic 004 backfilled this from confidence_score so it should be present.
            p_approval = float(primary["p_positive_outcome"])
            p_approval_source = "catalyst_p_positive_outcome"
        else:
            p_approval = float(primary.get("probability") or 0.5)
            p_approval_source = "catalyst_probability_legacy"
        p_approval = max(0.0, min(1.0, p_approval))

        # Resolve P(commercial success | approval) — separate from p_approval.
        # Caller can override; otherwise inherit from V2 econ estimate.
        p_commercial_resolved = None
        if req.p_commercial is not None:
            p_commercial_resolved = float(req.p_commercial)
        elif econ_v2 and econ_v2.get("commercial_success_prob") is not None:
            p_commercial_resolved = float(econ_v2["commercial_success_prob"])
        else:
            p_commercial_resolved = 0.6  # legacy default
        p_commercial_resolved = max(0.0, min(1.0, p_commercial_resolved))
        # If V2 econ has a value, propagate the resolved one back so downstream
        # uses the user-overridden value (instead of the LLM estimate).
        if isinstance(econ_v2, dict):
            econ_v2["commercial_success_prob"] = p_commercial_resolved

            # CRITICAL: compute_rnpv_full reads econ_v2.p_event_occurs and
            # econ_v2.p_positive_outcome when present, falling back to
            # combined p_approval × p_commercial only when both are missing.
            # When the user supplies an explicit p_approval, we MUST push it
            # into econ_v2.p_positive_outcome — otherwise the rNPV math would
            # silently use the LLM's estimate instead of the user override.
            if req.p_approval is not None:
                econ_v2["p_positive_outcome"] = p_approval
                # Don't override p_event_occurs unless the LLM didn't supply one;
                # event timing certainty is a separate axis from approval likelihood.
                if econ_v2.get("p_event_occurs") is None:
                    # Default to 1.0 (event will happen on schedule) if LLM didn't
                    # provide one — this preserves backward compatibility where
                    # a single user p_approval was treated as the full haircut.
                    econ_v2["p_event_occurs"] = 1.0
                logger.info(
                    f"NPV {req.ticker}: pushed user p_approval={p_approval:.3f} into "
                    f"econ_v2.p_positive_outcome (was {econ_v2.get('p_positive_outcome')})"
                )

            # Push catalyst-table split probabilities into econ_v2 if LLM
            # didn't already populate them. This way historical events with
            # backfilled p_event/p_positive can flow through the rNPV math.
            if econ_v2.get("p_event_occurs") is None and primary.get("p_event_occurs") is not None:
                econ_v2["p_event_occurs"] = float(primary["p_event_occurs"])
            if econ_v2.get("p_positive_outcome") is None and primary.get("p_positive_outcome") is not None and req.p_approval is None:
                econ_v2["p_positive_outcome"] = float(primary["p_positive_outcome"])

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
        
        # ---- Step 4.5: Move estimates (4 distinct types per ChatGPT critique) ----
        # Don't collapse expected-value, options-implied, scenario, and reference
        # into one number — they answer different questions.
        try:
            from services.post_catalyst_tracker import compute_move_estimates
            from services.options_implied import get_implied_move_for_catalyst
            options_implied_pct = None
            try:
                opt = get_implied_move_for_catalyst(req.ticker, catalyst_date)
                options_implied_pct = (opt or {}).get("implied_move_pct")
            except Exception as e:
                logger.info(f"options_implied lookup failed for {req.ticker}: {e}")
            # Fundamental impact = how big is the catalyst vs the company.
            # Prefer rNPV / market_cap (most accurate), fall back to legacy.
            fund_impact = None
            try:
                rnpv_m = float((rnpv or {}).get("rnpv_m") or 0)
                if rnpv_m > 0 and market_cap_m and market_cap_m > 0:
                    # rNPV is risk-adjusted asset value at this p_approval.
                    # Express as % of current market cap. If rNPV is $11B and
                    # mkt cap is $1.6B, fundamental_impact = 700% (huge re-rating
                    # potential if positive).
                    fund_impact = (rnpv_m / market_cap_m) * 100
                else:
                    fund_impact = float((legacy_npv or {}).get("fundamental_impact_pct") or 0)
            except Exception:
                pass
            move_estimates = compute_move_estimates(
                catalyst_type=req.catalyst_type,
                p_approval=p_approval,
                options_implied_pct=options_implied_pct,
                fundamental_impact_pct=fund_impact,
                sentiment_adj_factor=1.0,
            )
        except Exception as e:
            logger.warning(f"compute_move_estimates failed (non-fatal): {e}")
            move_estimates = None

        # ---- Step 4.6: Equity value with cash/debt/dilution adjustment ----
        # ChatGPT critique #5: "Add cash/debt/dilution-adjusted equity value,
        # not only asset rNPV divided by market cap."
        # Pull SEC EDGAR balance sheet → compute per-share value, dilution risk.
        equity_value = None
        cap_structure = None
        try:
            from services.sec_financials import fetch_capital_structure, compute_equity_value
            cap_structure = fetch_capital_structure(req.ticker)
            if cap_structure and not cap_structure.get("_error"):
                rnpv_m = float((rnpv or {}).get("rnpv_m") or 0)
                if rnpv_m > 0:
                    equity_value = compute_equity_value(
                        rnpv_m=rnpv_m,
                        cap_struct=cap_structure,
                        dilution_assumed_pct=req.dilution_assumed_pct,
                    )
        except Exception as e:
            logger.info(f"equity_value computation failed (non-fatal): {e}")

        # ---- Step 4.7: Narrative dilution capacity (ATM / shelf / warrants) ----
        # ChatGPT pass-3 critique #1: SEC XBRL gives current cash/debt/shares,
        # but ATM facilities + shelf registrations + warrants live in narrative
        # filings (S-3, 424B5, 8-K). For micro-caps, the CAPACITY to dilute
        # often matters more than current count.
        #
        # PERFORMANCE: This step fetches up to N filings + makes one LLM call
        # per filing — typical latency 20-60s on cache miss. To prevent the
        # /analyze/npv endpoint from hanging the UI, we:
        #   1. Cap max_filings_to_parse to 2 (down from 4) — most relevant
        #      filings are S-3 + most-recent 424B5
        #   2. Wrap the whole call in asyncio.wait_for with a 25s budget
        #   3. On timeout/failure, return None — caller can fetch later via
        #      /admin/sec/dilution-capacity?ticker=X for the full version
        dilution_capacity = None
        try:
            import asyncio
            from services.sec_dilution import fetch_dilution_capacity
            try:
                # Run blocking call in threadpool with hard timeout
                loop = asyncio.get_event_loop()
                dilution_capacity = await asyncio.wait_for(
                    loop.run_in_executor(None, fetch_dilution_capacity, req.ticker, 2),
                    timeout=25.0,
                )
            except asyncio.TimeoutError:
                logger.info(f"dilution_capacity timeout for {req.ticker} (>25s) — returning null, fetch via /admin")
                dilution_capacity = {"_status": "timeout_in_npv_endpoint",
                                       "_message": "Use /admin/sec/dilution-capacity for full extraction"}
        except Exception as e:
            logger.info(f"dilution_capacity lookup failed (non-fatal): {e}")

        # ---- Step 5: persist to cache ----
        result_payload = {
            "ticker": req.ticker,
            "drug_name": drug_name,
            "economics": legacy_economics,
            "npv": legacy_npv,
            "economics_v2": econ_v2,
            "rnpv": rnpv,
            "from_cache": False,
            # Probability resolution — surface what was actually used so UI
            # can show users which value drove the rNPV calculation.
            "probability_resolution": {
                "p_approval_used": p_approval,
                "p_approval_source": p_approval_source,
                "p_commercial_used": p_commercial_resolved,
                "p_event_occurs_used": (econ_v2 or {}).get("p_event_occurs"),
                "p_positive_outcome_used": (econ_v2 or {}).get("p_positive_outcome"),
                "rnpv_method": (rnpv or {}).get("assumptions_used", {}).get("rnpv_method"),
            },
            # 4 distinct move types — UI should show all separately
            "move_estimates": move_estimates,
            # Capital-structure-aware equity value — replaces naive rNPV/market_cap
            "equity_value": equity_value,
            "capital_structure": cap_structure,
            # Narrative dilution: ATM, shelf, warrants from S-3/424B5/8-K
            "dilution_capacity": dilution_capacity,
            # Source precedence audit — what verified facts overrode LLM inference
            "source_precedence_audit": precedence_audit,
        }

        # Recompute confidence_breakdown with the FINAL enriched provenance.
        # source_precedence adds sec_edgar entries (shares_outstanding_m,
        # cash_runway_months) AFTER estimate_drug_economics_v2 returned.
        # Without this recompute, the dilution category shows 0/5 populated
        # even though those fields exist with high confidence.
        try:
            from services.npv_model import _compute_confidence_breakdown
            final_prov = (econ_v2 or {}).get("provenance") or {}
            if final_prov:
                if "economics_v2" in result_payload and isinstance(result_payload["economics_v2"], dict):
                    result_payload["economics_v2"]["confidence_breakdown"] = _compute_confidence_breakdown(final_prov)
        except Exception as e:
            logger.info(f"confidence_breakdown recompute failed (non-fatal): {e}")

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
