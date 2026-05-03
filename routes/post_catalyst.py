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
from services.feature_store import (
    backfill_features_batch, compute_event_features, get_coverage_report,
)

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


# ─── Feature store (catalyst_event_features) ────────────────

def _apply_migration_module(filename: str):
    """Run a single alembic migration file's upgrade() inline. Bypasses the
    alembic Operations proxy (which requires env loading) by running the raw
    SQL via BiotechDatabase. Idempotent — relies on each migration using
    IF NOT EXISTS / IF EXISTS guards."""
    import importlib.util, pathlib, re
    path = pathlib.Path(__file__).resolve().parent.parent / "alembic" / "versions" / filename
    if not path.exists():
        raise HTTPException(404, f"migration file {filename} not found")
    src = path.read_text(encoding="utf-8")
    # Extract every op.execute("""...""") block via regex (cheap parser)
    blocks = re.findall(r'op\.execute\(\s*"""(.*?)"""\s*\)', src, flags=re.DOTALL)
    if not blocks:
        raise HTTPException(500, f"no op.execute() blocks in {filename}")
    from services.database import BiotechDatabase
    db = BiotechDatabase()
    with db.get_conn() as conn:
        cur = conn.cursor()
        for sql in blocks:
            # Skip blocks that look like DROP-only (downgrade)
            if "DROP" in sql.upper() and "CREATE" not in sql.upper() and "ADD COLUMN" not in sql.upper():
                continue
            cur.execute(sql)
        conn.commit()


@router.post("/admin/features/apply-migration-021")
async def admin_apply_migration_021():
    """Create the catalyst_event_features table (idempotent). Runs the raw
    DDL from alembic/versions/021_catalyst_event_features.py. Returns row count."""
    _apply_migration_module("021_catalyst_event_features.py")
    from services.database import BiotechDatabase
    db = BiotechDatabase()
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM catalyst_event_features")
        n = cur.fetchone()[0]
    return {"ok": True, "rows_in_table": int(n)}


@router.post("/admin/features/apply-migration-022")
async def admin_apply_migration_022():
    """Add the FAERS / ct.gov / Finviz / PubMed / USPTO / NIH RePORTER
    columns to catalyst_event_features (idempotent ALTER TABLE ADD IF
    NOT EXISTS)."""
    _apply_migration_module("022_event_features_extended.py")
    return {"ok": True}


@router.post("/admin/features/backfill-batch")
async def admin_features_backfill_batch(
    limit: int = Query(100, ge=1, le=2000),
    only_labeled: bool = Query(False),
    refresh: bool = Query(False, description="Recompute existing rows"),
):
    """Backfill catalyst_event_features for catalysts that don't yet have a
    row (or refresh existing rows). Idempotent. Returns batch summary +
    error samples.
    """
    try:
        return backfill_features_batch(
            limit=limit, only_labeled=only_labeled, refresh=refresh,
        )
    except Exception as e:
        logger.exception("features backfill failed")
        raise HTTPException(500, f"backfill error: {e}")


@router.post("/admin/features/event/{catalyst_id}")
async def admin_features_compute_one(catalyst_id: int, refresh: bool = Query(True)):
    """Compute / refresh feature row for a single catalyst event."""
    try:
        return compute_event_features(catalyst_id, refresh=refresh)
    except Exception as e:
        logger.exception("compute_event_features(%s) failed", catalyst_id)
        raise HTTPException(500, f"compute error: {e}")


@router.get("/admin/features/coverage")
async def admin_features_coverage():
    """Population stats: total rows, % vs universe + labeled, per-column %."""
    try:
        return get_coverage_report()
    except Exception as e:
        logger.exception("coverage report failed")
        raise HTTPException(500, f"coverage error: {e}")


@router.get("/admin/features/per-ticker-coverage")
async def admin_features_per_ticker_coverage(
    limit: int = Query(0, ge=0, le=5000, description="0 = no limit"),
):
    """Per-ticker boolean coverage matrix for the Data Control Center.

    Collapses the 8762-event feature store to ~1500 ticker rows; each row
    has BOOL_OR(col IS NOT NULL) flags per source plus labeled / pending /
    error event counts. On-demand single SQL pass — expected <300ms.
    """
    from services.database import BiotechDatabase
    db = BiotechDatabase()
    sql = """
        SELECT
          cef.ticker,
          COUNT(*)                                               AS events,
          BOOL_OR(cef.runup_pct_30d         IS NOT NULL)         AS has_yfinance,
          BOOL_OR(cef.xbi_runup_30d         IS NOT NULL)         AS has_peers,
          BOOL_OR(cef.short_interest_pct_at_date IS NOT NULL)    AS has_microstructure,
          BOOL_OR(cef.cash_at_date_m        IS NOT NULL)         AS has_sec_capital,
          BOOL_OR(cef.atm_iv_at_date        IS NOT NULL)         AS has_massive_options,
          BOOL_OR(cef.analyst_recommendation_avg IS NOT NULL OR
                  cef.analyst_target_price_usd  IS NOT NULL OR
                  cef.finviz_perf_ytd_pct       IS NOT NULL)     AS has_finviz,
          BOOL_OR(cef.trial_count_active_at_date IS NOT NULL OR
                  cef.trial_total_enrollment     IS NOT NULL)    AS has_ctgov,
          BOOL_OR(cef.adverse_event_count_90d_pre  IS NOT NULL OR
                  cef.adverse_event_count_365d_pre IS NOT NULL)  AS has_faers,
          BOOL_OR(cef.outcome_label_gemini  IS NOT NULL)         AS has_label_gemini,
          BOOL_OR(cef.outcome_label_price_proxy IS NOT NULL)     AS has_label_proxy,
          BOOL_OR(cef.drug_npv_b_at_date    IS NOT NULL)         AS has_drug_npv,
          COUNT(*) FILTER (
            WHERE pco.outcome_label_class IS NOT NULL
          )                                                      AS labeled_events,
          COUNT(*) FILTER (
            WHERE pco.outcome_labeled_at IS NULL
              AND COALESCE(pco.outcome_label_attempts, 0) < 3
              AND cef.catalyst_date IS NOT NULL
          )                                                      AS pending_events,
          COUNT(*) FILTER (
            WHERE pco.outcome_labeled_at IS NULL
              AND COALESCE(pco.outcome_label_attempts, 0) >= 3
          )                                                      AS error_events,
          MAX(cef.backfilled_at)                                 AS last_backfilled_at,
          MAX(cef.llm_enriched_at)                               AS last_enriched_at
        FROM catalyst_event_features cef
        LEFT JOIN post_catalyst_outcomes pco ON pco.catalyst_id = cef.catalyst_id
        GROUP BY cef.ticker
        ORDER BY cef.ticker
    """
    if limit > 0:
        sql += f"\n        LIMIT {int(limit)}"
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = []
            for raw in cur.fetchall():
                row = dict(zip(cols, raw))
                # JSON-friendly: ints/bools/None pass through; timestamps → ISO
                for k in ("last_backfilled_at", "last_enriched_at"):
                    if row.get(k) is not None and hasattr(row[k], "isoformat"):
                        row[k] = row[k].isoformat()
                rows.append(row)
        return {"count": len(rows), "rows": rows}
    except Exception as e:
        logger.exception("per-ticker coverage failed")
        raise HTTPException(500, f"per-ticker coverage error: {e}")


@router.get("/admin/features/data-feed-debug")
async def admin_data_feed_debug():
    """Probe massive.com (options) + Finviz Elite inside the running
    container. Compares `os.getenv` to actual API responses + char-level
    key inspection so trailing whitespace / hidden chars are visible.

    Renamed from /polygon-debug because the vendor is massive.com
    (Polygon-compatible REST). MASSIVE_API_KEY is canonical;
    POLYGON_API_KEY is a legacy alias (read as fallback only)."""
    import os, requests
    out: Dict[str, object] = {}
    # massive.com
    raw = os.getenv("MASSIVE_API_KEY", "") or os.getenv("POLYGON_API_KEY", "")
    key = raw.strip()
    out["massive"] = {
        "key_source": "MASSIVE_API_KEY" if os.getenv("MASSIVE_API_KEY") else "POLYGON_API_KEY (legacy)",
        "raw_repr": repr(raw),
        "stripped_len": len(key),
        "tests": {},
    }
    try:
        r = requests.get("https://api.massive.com/v3/snapshot/options/NTLA",
                         params={"limit": 3, "apiKey": key}, timeout=15)
        out["massive"]["tests"]["current_snapshot"] = {"status": r.status_code, "body_head": r.text[:300]}
    except Exception as e:
        out["massive"]["tests"]["current_snapshot"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    try:
        r = requests.get(
            "https://api.massive.com/v3/reference/options/contracts",
            params={"underlying_ticker": "NTLA", "as_of": "2025-01-15", "limit": 3, "apiKey": key},
            timeout=15,
        )
        out["massive"]["tests"]["historical_contracts"] = {"status": r.status_code, "body_head": r.text[:200]}
    except Exception as e:
        out["massive"]["tests"]["historical_contracts"] = {"error": str(e)[:120]}

    # Finviz Elite
    fv_raw = os.getenv("FINVIZ_API_KEY", "")
    fv_key = fv_raw.strip()
    out["finviz"] = {"raw_repr": repr(fv_raw), "stripped_len": len(fv_key)}
    try:
        # Match _fill_finviz: v=111 with the documented column codes that
        # actually return Recom (64) and Target Price (65).
        url = (
            "https://elite.finviz.com/export.ashx"
            f"?v=111&t=NTLA&c=1,6,25,26,27,28,29,30,31,47,48,50,60,61,64,65&auth={fv_key}"
        )
        r = requests.get(url, timeout=20)
        out["finviz"]["status"] = r.status_code
        out["finviz"]["body_head"] = r.text[:800]
    except Exception as e:
        out["finviz"]["error"] = str(e)[:120]
    return out
