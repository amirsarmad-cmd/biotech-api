"""
/v2/post-catalyst routes — historical catalyst outcome tracker.

User-facing:
  GET  /v2/post-catalyst/history/{ticker}     — past catalysts + actual moves
  GET  /v2/post-catalyst/accuracy             — system-wide prediction accuracy

Admin:
  POST /admin/post-catalyst/backfill          — run backfill batch
  POST /admin/post-catalyst/backfill-one      — backfill specific catalyst
  GET  /admin/post-catalyst/due               — preview what's due
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from services.post_catalyst_tracker import (
    find_due_catalysts, get_outcomes_for_ticker, get_aggregate_accuracy,
    backfill_one, backfill_batch,
)
from services.scenario_algo import backtest_scenario_algo

logger = logging.getLogger(__name__)
router = APIRouter()


# ---- User-facing ----

@router.get("/v2/post-catalyst/history/{ticker}")
async def post_catalyst_history(ticker: str, limit: int = Query(50, ge=1, le=200)):
    """Return all post_catalyst_outcomes rows for ticker, newest first.
    
    Each row includes: catalyst type/date/drug, pre/post prices, % moves at 1d/7d/30d,
    inferred outcome (approved/rejected/mixed/delayed/unknown), prediction error.
    """
    ticker = ticker.upper().strip()
    rows = get_outcomes_for_ticker(ticker, limit=limit)
    # Convert any decimal/datetime to JSON-friendly
    cleaned = []
    for r in rows:
        d = {}
        for k, v in r.items():
            if v is None:
                d[k] = None
            elif hasattr(v, 'isoformat'):
                d[k] = v.isoformat()
            elif hasattr(v, '__float__') and not isinstance(v, bool):
                try: d[k] = float(v)
                except Exception: d[k] = v
            else:
                d[k] = v
        cleaned.append(d)
    return {"ticker": ticker, "count": len(cleaned), "outcomes": cleaned}


@router.get("/v2/post-catalyst/accuracy")
async def post_catalyst_accuracy():
    """System-wide prediction accuracy: direction hit rate, avg error %, outcome breakdown."""
    return get_aggregate_accuracy()


@router.get("/v2/stocks/{ticker}/drug-programs")
async def stock_drug_programs(ticker: str):
    """Group all catalysts for the ticker by drug program (with substring-
    aware aliasing so NTLA-2001 / nex-z / nexiguran ziclumeran collapse
    into one program), sequence them by milestone (Phase 1 → Phase 2 →
    Phase 3 → Submission → FDA Decision), compute % completion, and
    surface per-event price action + v2 prediction columns.

    Replaces the user-flagged problem of "too much individual dates and
    sub-catalysts" with a drug-centric tree view.
    """
    from services.drug_programs import get_drug_programs_for_ticker
    return get_drug_programs_for_ticker(ticker)


# ---- Admin ----

@router.post("/admin/post-catalyst/backfill")
async def admin_backfill_batch(limit: int = Query(25, ge=1, le=200),
                                min_age_days: int = Query(7, ge=0, le=365)):
    """Find catalysts whose date has passed (>= min_age_days ago) without an outcome
    row, and backfill up to `limit` of them. Returns batch summary."""
    try:
        return backfill_batch(limit=limit, min_age_days=min_age_days)
    except Exception as e:
        logger.exception("backfill batch failed")
        raise HTTPException(500, f"backfill error: {e}")


@router.post("/admin/post-catalyst/backfill-one")
async def admin_backfill_one(ticker: str, catalyst_type: str, catalyst_date: str,
                              catalyst_id: Optional[int] = None,
                              drug_name: Optional[str] = None,
                              indication: Optional[str] = None,
                              probability: Optional[float] = None):
    """Backfill a specific catalyst by (ticker, type, date). Useful for manual recovery."""
    try:
        cat = {
            "id": catalyst_id, "ticker": ticker.upper(),
            "catalyst_type": catalyst_type, "catalyst_date": catalyst_date,
            "drug_name": drug_name, "indication": indication,
            "confidence_score": probability,
        }
        return backfill_one(cat)
    except Exception as e:
        logger.exception("backfill_one failed")
        raise HTTPException(500, f"backfill error: {e}")


@router.get("/admin/post-catalyst/due")
async def admin_due_preview(limit: int = Query(50, ge=1, le=200),
                             min_age_days: int = Query(7, ge=0, le=365)):
    """Preview which catalysts are queued for backfill, without running it."""
    due = find_due_catalysts(limit=limit, min_age_days=min_age_days)
    return {"count": len(due), "min_age_days": min_age_days, "catalysts": due}


@router.get("/admin/post-catalyst/scenario-algo-backtest")
async def admin_scenario_algo_backtest(
    only_labeled: bool = Query(True, description="Only score rows where Gemini has labeled the outcome"),
    min_n_per_bucket: int = Query(10, ge=1, le=200, description="Buckets with fewer events flagged low_confidence"),
    limit: int = Query(5000, ge=10, le=20000, description="Max post_catalyst_outcomes rows to score"),
):
    """Phase 1 backtest of the proposed V2 scenario-range algo against
    historical post_catalyst_outcomes. Reports per-bucket direction-hit %,
    MAE, and bias so we can validate the algo BEFORE shipping it to prod
    (currently, services/post_catalyst_tracker.py still uses the old 80%-clamp
    formula). Includes caveat list (lookback bias on market_cap, sample size,
    survivorship from incomplete label backfill).

    Decision gate: ship to prod when totals.decision_gate_passed=True. Else
    move to Phase 2 (refit capture rates per catalyst_type from this dataset).
    """
    try:
        return backtest_scenario_algo(
            only_labeled=only_labeled,
            min_n_per_bucket=min_n_per_bucket,
            limit=limit,
        )
    except Exception as e:
        logger.exception("scenario-algo-backtest failed")
        raise HTTPException(500, f"backtest error: {e}")
