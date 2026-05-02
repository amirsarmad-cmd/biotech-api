"""
/admin — universe refresh, DB stats, migration tools.
"""
import logging, os, json
from datetime import datetime
from typing import Optional, Dict
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/universe/refresh")
async def refresh_universe():
    """Re-seed the 70-stock universe from Finnhub + FDA calendar."""
    try:
        from services.universe import seed_universe
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        stocks = seed_universe(expand=True)
        added = 0
        for s in stocks:
            if db.add_stock(s):
                added += 1
        return {"status": "ok", "fetched": len(stocks), "upserted": added}
    except Exception as e:
        logger.exception("universe refresh failed")
        raise HTTPException(500, f"universe refresh error: {e}")


@router.get("/db/stats")
async def db_stats():
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        stats = db.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(500, f"stats error: {e}")


# ─── Migration inspection & repair ─────────────────────────────────────────

def _pg_conn():
    import psycopg2
    url = os.getenv("DATABASE_URL", "")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(url)


@router.get("/db/inspect")
async def inspect_db():
    """Show alembic_version state + list of public tables."""
    try:
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                # alembic versions — both shared table and our isolated table
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = 'alembic_version'
                    )
                """)
                has_alembic = cur.fetchone()[0]
                versions = []
                if has_alembic:
                    cur.execute("SELECT version_num FROM alembic_version")
                    versions = [r[0] for r in cur.fetchall()]
                
                # Our isolated alembic table
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = 'alembic_version_biotech'
                    )
                """)
                has_biotech_alembic = cur.fetchone()[0]
                biotech_versions = []
                if has_biotech_alembic:
                    cur.execute("SELECT version_num FROM alembic_version_biotech")
                    biotech_versions = [r[0] for r in cur.fetchall()]

                # all public tables
                cur.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema='public' ORDER BY table_name
                """)
                tables = [r[0] for r in cur.fetchall()]

                # check which v2 tables exist
                v2_tables = [
                    "catalyst_universe", "earnings_dates", "catalyst_npv_cache",
                    "ai_analysis_cache", "stock_risk_factors", "drug_economics_cache",
                    "historical_catalyst_moves", "stock_scores", "npv_defaults", "cron_runs"
                ]
                v2_present = [t for t in v2_tables if t in tables]
                v2_missing = [t for t in v2_tables if t not in tables]

                # NPV defaults singleton check
                npv_defaults_row = None
                if "npv_defaults" in tables:
                    cur.execute("SELECT scope, discount_rate, tax_rate, cogs_pct, default_penetration_pct, default_time_to_peak_years, rating_weights FROM npv_defaults WHERE scope='global'")
                    row = cur.fetchone()
                    if row:
                        npv_defaults_row = {
                            "scope": row[0],
                            "discount_rate": float(row[1]) if row[1] else None,
                            "tax_rate": float(row[2]) if row[2] else None,
                            "cogs_pct": float(row[3]) if row[3] else None,
                            "default_penetration_pct": float(row[4]) if row[4] else None,
                            "default_time_to_peak_years": row[5],
                            "rating_weights": row[6],
                        }

                return {
                    "alembic_version_table_exists": has_alembic,
                    "alembic_versions_shared": versions,
                    "alembic_version_biotech_exists": has_biotech_alembic,
                    "alembic_versions_biotech": biotech_versions,
                    "all_public_tables": tables,
                    "v2_tables_present": v2_present,
                    "v2_tables_missing": v2_missing,
                    "npv_defaults_singleton": npv_defaults_row,
                }
    except Exception as e:
        logger.exception("inspect_db")
        raise HTTPException(500, f"inspect error: {e}")


class StampRequest(BaseModel):
    revision: str
    confirm: bool = False


@router.post("/db/alembic-clear")
async def alembic_clear(req: StampRequest):
    """DESTRUCTIVE: clear the alembic_version table. Requires confirm=True."""
    if not req.confirm:
        raise HTTPException(400, "Must pass confirm=True")
    try:
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM alembic_version")
                conn.commit()
                cur.execute("SELECT count(*) FROM alembic_version")
                remaining = cur.fetchone()[0]
                return {"cleared": True, "rows_remaining": remaining}
    except Exception as e:
        logger.exception("alembic_clear")
        raise HTTPException(500, f"clear error: {e}")


@router.post("/db/migrate-now")
async def migrate_now():
    """Run alembic upgrade head right now from the API container."""
    import subprocess
    try:
        r = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True, text=True, timeout=120, cwd="/app"
        )
        return {
            "rc": r.returncode,
            "stdout": r.stdout[-3000:],
            "stderr": r.stderr[-3000:],
        }
    except Exception as e:
        logger.exception("migrate_now")
        raise HTTPException(500, f"migrate error: {e}")


# ─── Universe seeder (Phase B) ────────────────────────────────────────────

class SeedRequest(BaseModel):
    max_tickers: int | None = None  # cap for safety; None = use env default


@router.post("/universe/v2-seed")
async def seed_universe_v2(req: SeedRequest):
    """Run the Phase B universe seeder synchronously. Use only for small max_tickers (≤10)."""
    try:
        from services.universe_seeder import run_universe_seed
        return run_universe_seed(max_tickers=req.max_tickers)
    except Exception as e:
        logger.exception("v2-seed")
        raise HTTPException(500, f"seed error: {e}")


import threading
import uuid as _uuid
from datetime import datetime as _dt

# In-memory tracker for background seed jobs
_bg_seed_jobs: dict = {}


def _run_seed_background(job_id: str, max_tickers: int | None, start_idx: int):
    """Background thread runner for the seeder.
    
    Passes start_idx natively (no monkeypatch) so concurrent runs don't race.
    """
    from services.universe_seeder import run_universe_seed
    
    _bg_seed_jobs[job_id] = {"job_id": job_id, "status": "running", "started_at": _dt.utcnow().isoformat()+"Z", "max_tickers": max_tickers, "start_idx": start_idx}
    try:
        result = run_universe_seed(max_tickers=max_tickers, start_idx=start_idx)
        _bg_seed_jobs[job_id]["status"] = "completed"
        _bg_seed_jobs[job_id]["result"] = result
        _bg_seed_jobs[job_id]["completed_at"] = _dt.utcnow().isoformat()+"Z"
    except Exception as e:
        _bg_seed_jobs[job_id]["status"] = "failed"
        _bg_seed_jobs[job_id]["error"] = f"{type(e).__name__}: {e}"
        _bg_seed_jobs[job_id]["completed_at"] = _dt.utcnow().isoformat()+"Z"


class AsyncSeedRequest(BaseModel):
    max_tickers: int | None = None
    start_idx: int = 0
    force: bool = False  # bypass mutex (use only for stuck-state recovery)


@router.post("/universe/v2-seed-async")
async def seed_universe_v2_async(req: AsyncSeedRequest):
    """Kick off background seed run. Returns job_id immediately.

    Mutex: refuses to start if any job already in 'running' state, unless
    force=True is passed. This prevents concurrent runs from racing on the
    same tickers + budget.
    """
    if not req.force:
        running = [j for j in _bg_seed_jobs.values() if j.get("status") == "running"]
        if running:
            return {
                "rejected": True,
                "reason": "Another seed job is already running. Pass force=true to override.",
                "running_job_ids": [j.get("job_id") for j in running],
                "running_count": len(running),
            }
    job_id = _uuid.uuid4().hex[:12]
    t = threading.Thread(target=_run_seed_background, args=(job_id, req.max_tickers, req.start_idx), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "queued", "start_idx": req.start_idx, "max_tickers": req.max_tickers}


@router.post("/universe/test-tier3-anthropic")
async def test_anthropic_tier3(ticker: str, company_name: str = ""):
    """Direct test of the tier-3 Anthropic+web_search extractor.

    Bypasses Gemini and OpenAI; calls _call_anthropic_extract directly so
    we can verify the wiring without waiting for both upstream tiers to
    return empty. Use sparingly — each call is ~$0.012 + 30-90s latency.

    Returns: {"catalysts": [...], "n": int, "ticker": str}
    """
    from services.universe_seeder import _call_anthropic_extract, ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured")
    try:
        cats = _call_anthropic_extract(ticker, company_name or ticker)
        return {"ticker": ticker, "n": len(cats), "catalysts": cats}
    except Exception as e:
        raise HTTPException(500, f"anthropic extract failed: {type(e).__name__}: {e}")


@router.post("/universe/v2-seed-mark-stale")
async def mark_stale_jobs(stale_minutes: int = 10):
    """Mark in-memory bg jobs that have been 'running' for >stale_minutes as 'stale'.
    Also marks corresponding cron_runs rows as 'killed'.
    Returns how many were marked.
    """
    from datetime import datetime as _ndt, timedelta
    cutoff = _ndt.utcnow() - timedelta(minutes=stale_minutes)
    marked = []
    for job in list(_bg_seed_jobs.values()):
        if job.get("status") != "running":
            continue
        try:
            started = _ndt.fromisoformat(job["started_at"].rstrip("Z"))
        except Exception:
            continue
        if started < cutoff:
            job["status"] = "stale"
            job["completed_at"] = _ndt.utcnow().isoformat() + "Z"
            job["error"] = f"marked stale (running > {stale_minutes} min)"
            marked.append(job["job_id"])

    # Also clean up cron_runs entries
    cron_killed = 0
    try:
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE cron_runs
                    SET status='killed', completed_at=NOW(),
                        errors=COALESCE(errors, '[]'::jsonb) || jsonb_build_array(%s)
                    WHERE status='running'
                      AND started_at < NOW() - INTERVAL '%s minutes'
                    RETURNING id
                """, (f"marked killed by /v2-seed-mark-stale (>{stale_minutes}min)", stale_minutes))
                cron_killed = len(cur.fetchall())
                conn.commit()
    except Exception as e:
        logger.warning(f"cron_runs killed update failed: {e}")

    return {"bg_jobs_marked_stale": marked, "cron_runs_killed": cron_killed}


@router.get("/universe/v2-seed-async/{job_id}")
async def seed_status(job_id: str):
    if job_id not in _bg_seed_jobs:
        raise HTTPException(404, f"job {job_id} not found")
    return _bg_seed_jobs[job_id]


@router.get("/universe/v2-seed-async")
async def list_seed_jobs():
    """List all in-memory background seed jobs."""
    return {"jobs": list(_bg_seed_jobs.values())}


@router.get("/universe/v2-spend")
async def universe_spend():
    """Inspect current LLM spend + budget for the universe seeder."""
    try:
        from services.universe_seeder import get_daily_spend
        return get_daily_spend()
    except Exception as e:
        raise HTTPException(500, f"spend error: {e}")


@router.get("/universe/v2-debug-gemini/{ticker}")
async def debug_gemini(ticker: str):
    """Show raw Gemini response for one ticker — for debugging extraction."""
    import os
    from google import genai
    from google.genai import types
    from datetime import date, timedelta
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    if not GOOGLE_API_KEY:
        raise HTTPException(500, "GOOGLE_API_KEY not set")
    
    today = date.today()
    six_months = today + timedelta(days=183)
    prompt = f"""You are a biotech catalyst analyst. Extract REAL upcoming non-earnings catalysts for {ticker} expected between {today.isoformat()} and {six_months.isoformat()}.

CRITICAL: Use ONLY real factual data. No placeholders.

For each catalyst:
- catalyst_type: "FDA Decision" | "AdComm" | "Phase 3 Readout" | "Phase 2 Readout" | "Phase 1 Readout" | "Clinical Trial" | "Partnership" | "BLA submission" | "NDA submission"
- catalyst_date: ISO date
- date_precision: "exact" | "quarter" | "half" | "year"
- description: 1-sentence factual description
- drug_name: real drug name (no placeholders)
- indication: specific medical condition
- phase: "Phase 1" | "Phase 2" | "Phase 3" | "BLA" | "NDA" | null
- confidence_score: 0.0-1.0

Return ONLY a JSON object with format: {{"catalysts": [...]}}
"""
    
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        config = types.GenerateContentConfig(
            max_output_tokens=2000,
            temperature=0.1,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )
        raw = response.text or ""
        # also try to expose grounding metadata
        metadata = {}
        try:
            if response.candidates and response.candidates[0].grounding_metadata:
                gm = response.candidates[0].grounding_metadata
                metadata["grounding_chunks"] = len(gm.grounding_chunks or [])
                metadata["search_queries"] = list(gm.web_search_queries or [])
        except Exception:
            pass
        return {
            "ticker": ticker,
            "raw_response": raw,
            "raw_length": len(raw),
            "metadata": metadata,
        }
    except Exception as e:
        return {"ticker": ticker, "error": f"{type(e).__name__}: {e}"}


@router.delete("/universe/v2-mock-clear")
async def clear_mock_catalysts(confirm: bool = False, source: str = "mock"):
    """Delete catalyst rows with given source. Pass confirm=true to execute. source='all' wipes everything."""
    if not confirm:
        raise HTTPException(400, "Pass ?confirm=true to delete rows")
    try:
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                if source == "all":
                    cur.execute("DELETE FROM catalyst_universe RETURNING id")
                else:
                    cur.execute("DELETE FROM catalyst_universe WHERE source=%s RETURNING id", (source,))
                deleted = len(cur.fetchall())
                conn.commit()
                return {"deleted": deleted, "source_filter": source}
    except Exception as e:
        raise HTTPException(500, f"clear error: {e}")


@router.get("/universe/v2-cron-runs")
async def universe_cron_runs(limit: int = 20):
    """Show recent cron_runs entries."""
    try:
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, job_name, started_at, completed_at, status,
                           records_processed, records_added, records_updated, errors
                    FROM cron_runs ORDER BY started_at DESC LIMIT %s
                """, (limit,))
                rows = cur.fetchall()
                cols = ["id", "job_name", "started_at", "completed_at", "status",
                        "records_processed", "records_added", "records_updated", "errors"]
                items = []
                for r in rows:
                    item = {}
                    for c, v in zip(cols, r):
                        item[c] = v.isoformat() if hasattr(v, "isoformat") else v
                    items.append(item)
                return {"runs": items}
    except Exception as e:
        raise HTTPException(500, f"cron error: {e}")



@router.delete("/universe/v2-clean-junk")
async def clean_junk_catalysts(confirm: bool = False):
    """Delete catalyst_universe rows where both drug_name AND indication are NULL.
    These are partial extractions from the LLM that aren't useful for downstream NPV/risk analysis.
    """
    if not confirm:
        raise HTTPException(400, "Pass ?confirm=true to delete junk rows")
    try:
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM catalyst_universe 
                    WHERE drug_name IS NULL AND indication IS NULL 
                    RETURNING id, ticker, catalyst_type
                """)
                deleted = cur.fetchall()
                conn.commit()
                return {
                    "deleted": len(deleted),
                    "samples": [{"id": r[0], "ticker": r[1], "type": r[2]} for r in deleted[:5]]
                }
    except Exception as e:
        raise HTTPException(500, f"clean error: {e}")


# ============================================================
# Market cap backfill — yfinance batch fetch for V2 universe tickers
# ============================================================

@router.post("/marketcap/fix-units")
async def fix_market_cap_units(dry_run: bool = True):
    """One-time fix: earlier versions of /admin/marketcap/backfill divided
    market_cap_m by 1000, storing values 1000× too small. This identifies and
    fixes those rows by multiplying market_cap by 1000 where the row was
    written by yfinance backfill (description LIKE 'yfinance backfill%').
    
    Skips rows where market_cap is already > 100 (likely already fixed or
    legitimate $M value)."""
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            # Find candidates: backfill marker rows with suspiciously small market_cap
            cur.execute("""
                SELECT ticker, market_cap, description
                FROM screener_stocks
                WHERE description LIKE 'yfinance backfill%'
                  AND market_cap IS NOT NULL
                  AND market_cap > 0
                  AND market_cap < 1000  -- if real, would be a $1B company written as $1; could be legit but rare for biotech
                ORDER BY ticker
            """)
            candidates = [{"ticker": r[0], "current_market_cap": float(r[1]),
                            "would_become": float(r[1]) * 1000} for r in cur.fetchall()]

            updated = 0
            if not dry_run:
                cur.execute("""
                    UPDATE screener_stocks
                    SET market_cap = market_cap * 1000
                    WHERE description LIKE 'yfinance backfill%'
                      AND market_cap IS NOT NULL
                      AND market_cap > 0
                      AND market_cap < 1000
                """)
                updated = cur.rowcount
                conn.commit()

            return {
                "dry_run": dry_run,
                "candidates_count": len(candidates),
                "updated": updated,
                "candidates": candidates[:20],
            }
    except Exception as e:
        logger.exception("fix_market_cap_units failed")
        raise HTTPException(500, f"fix error: {e}")


@router.post("/marketcap/backfill")
async def backfill_market_cap(
    limit: int = 100,
    only_missing: bool = True,
    dry_run: bool = False,
):
    """Pull market cap (and current price) from yfinance for V2 catalyst_universe
    tickers and UPSERT into screener_stocks.market_cap.
    
    Args:
      limit: max tickers per call (default 100). yfinance is rate-limited; batches.
      only_missing: only fetch tickers where market_cap is NULL or 0 (default True)
      dry_run: report what would change but don't write
    
    Returns: {fetched, written, errors, results: [{ticker, market_cap, current_price}]}
    """
    try:
        import yfinance as yf
        from services.database import BiotechDatabase
        db = BiotechDatabase()

        # 1. Find target tickers
        with db.get_conn() as conn:
            cur = conn.cursor()
            if only_missing:
                # Skip tickers we tried in the last 24h that came back $0 — likely
                # delisted / acquired / inaccessible. We mark these by writing a
                # marker row with description='yfinance: no data ...' and last_updated.
                # NOTE: % must be escaped as %% in psycopg2 LIKE patterns when the
                # query also has %s placeholders.
                cur.execute("""
                    SELECT DISTINCT cu.ticker
                    FROM catalyst_universe cu
                    LEFT JOIN screener_stocks ss ON ss.ticker = cu.ticker
                    WHERE (ss.market_cap IS NULL OR ss.market_cap = 0)
                      AND cu.ticker IS NOT NULL
                      AND cu.ticker != ''
                      AND (
                        ss.last_updated IS NULL
                        OR ss.description IS NULL
                        OR ss.description NOT LIKE 'yfinance: no data%%'
                        OR ss.last_updated < (NOW() - INTERVAL '24 hours')::text
                      )
                    ORDER BY cu.ticker
                    LIMIT %s
                """, (limit,))
            else:
                cur.execute("""
                    SELECT DISTINCT ticker FROM catalyst_universe
                    WHERE ticker IS NOT NULL AND ticker != ''
                    ORDER BY ticker
                    LIMIT %s
                """, (limit,))
            tickers = [r[0] for r in cur.fetchall()]

        if not tickers:
            return {"fetched": 0, "written": 0, "errors": 0, "results": [],
                    "note": "no tickers needing backfill"}

        # 2. Batch yfinance fetch
        results = []
        errors = 0
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                info = t.info or {}
                mcap = info.get("marketCap") or 0  # in USD
                price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
                company = info.get("longName") or info.get("shortName") or ticker
                industry = info.get("industry") or "Biotechnology"
                if mcap == 0:
                    # Try fast_info as fallback
                    try:
                        fi = getattr(t, "fast_info", None)
                        if fi:
                            mcap = (fi.get("marketCap") if hasattr(fi, "get") else getattr(fi, "market_cap", 0)) or 0
                            price = price or (fi.get("lastPrice") if hasattr(fi, "get") else getattr(fi, "last_price", 0))
                    except Exception:
                        pass
                results.append({
                    "ticker": ticker,
                    "market_cap_m": round(mcap / 1_000_000, 2) if mcap else 0,
                    "market_cap_b": round(mcap / 1_000_000_000, 3) if mcap else 0,
                    "current_price": price,
                    "company_name": company,
                    "industry": industry,
                })
            except Exception as e:
                errors += 1
                results.append({"ticker": ticker, "error": str(e)[:100]})

        # 3. UPSERT into screener_stocks
        written = 0
        marked_dead = 0
        if not dry_run:
            with db.get_conn() as conn:
                cur = conn.cursor()
                for r in results:
                    if "error" in r:
                        continue
                    mc = r.get("market_cap_m") or 0
                    if not mc:
                        # Write a marker row so we skip this ticker for 24h
                        try:
                            cur.execute("""
                                INSERT INTO screener_stocks
                                    (ticker, company_name, industry, market_cap,
                                     catalyst_type, catalyst_date, probability,
                                     description, last_updated)
                                VALUES (%s, %s, %s, 0, %s, %s, %s, %s, NOW()::text)
                                ON CONFLICT (ticker, catalyst_type, catalyst_date) DO UPDATE SET
                                    description = EXCLUDED.description,
                                    last_updated = NOW()::text
                            """, (
                                r["ticker"], r["ticker"], "Biotechnology",
                                "", "", 0.5,
                                f"yfinance: no data {datetime.now().strftime('%Y-%m-%d')}",
                            ))
                            marked_dead += 1
                        except Exception as e:
                            errors += 1
                            r["mark_dead_error"] = str(e)[:100]
                        continue
                    try:
                        # screener_stocks.market_cap is in $M (NOT $B — the
                        # /stocks endpoint maps it directly to market_cap_m).
                        # Earlier versions of this code divided by 1000, which
                        # produced row values 1000× too small. Write market_cap_m
                        # directly.
                        cur.execute("""
                            INSERT INTO screener_stocks
                                (ticker, company_name, industry, market_cap,
                                 catalyst_type, catalyst_date, probability,
                                 description, last_updated)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW()::text)
                            ON CONFLICT (ticker, catalyst_type, catalyst_date) DO UPDATE SET
                                company_name = EXCLUDED.company_name,
                                industry = EXCLUDED.industry,
                                market_cap = EXCLUDED.market_cap,
                                description = EXCLUDED.description,
                                last_updated = NOW()::text
                        """, (
                            r["ticker"], r.get("company_name", r["ticker"]),
                            r.get("industry", "Biotechnology"),
                            r["market_cap_m"],  # store as $M, no division
                            "", "", 0.5,
                            f"yfinance backfill — current_price ${r.get('current_price', 0)}",
                        ))
                        written += 1
                    except Exception as e:
                        errors += 1
                        r["upsert_error"] = str(e)[:100]
                conn.commit()

        return {
            "fetched": len(tickers),
            "written": written if not dry_run else 0,
            "marked_dead": marked_dead if not dry_run else 0,
            "errors": errors,
            "dry_run": dry_run,
            "results": results,
        }
    except Exception as e:
        logger.exception("market cap backfill failed")
        raise HTTPException(500, f"backfill error: {e}")


# ============================================================
# Dead-marker cleanup — for tickers that yfinance returns no data on
# (delisted, acquired, ticker-changed). These rows pollute screener
# listings with $0 market_cap entries.
# ============================================================

@router.post("/marketcap/cleanup-dead")
async def cleanup_dead_markers(dry_run: bool = True, only_with_universe: bool = False):
    """Identify and remove screener_stocks rows that are pure dead-marker
    entries: empty catalyst_type AND empty catalyst_date AND market_cap = 0
    AND description starts with 'yfinance: no data'.

    Args:
      dry_run: report candidates without deleting (default True)
      only_with_universe: only delete if the ticker still has rows in
        catalyst_universe (in case we want to keep dead-marker for orphan
        tickers as a 'tried this ticker, no data' record). Default False —
        delete all dead markers regardless.

    Returns: {dry_run, candidates_count, deleted, candidates: [...]}
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            # Find candidates
            sql = """
                SELECT ss.ticker, ss.last_updated, ss.description
                FROM screener_stocks ss
                WHERE (ss.catalyst_type IS NULL OR ss.catalyst_type = '')
                  AND (ss.catalyst_date IS NULL OR ss.catalyst_date = '')
                  AND (ss.market_cap IS NULL OR ss.market_cap = 0)
                  AND ss.description LIKE 'yfinance: no data%'
            """
            if only_with_universe:
                sql += """
                  AND ss.ticker IN (SELECT DISTINCT ticker FROM catalyst_universe)
                """
            sql += " ORDER BY ss.ticker LIMIT 500"
            cur.execute(sql)
            candidates = [{"ticker": r[0], "last_updated": str(r[1])[:19] if r[1] else None,
                           "description": r[2]} for r in cur.fetchall()]

            deleted = 0
            if not dry_run:
                del_sql = """
                    DELETE FROM screener_stocks
                    WHERE (catalyst_type IS NULL OR catalyst_type = '')
                      AND (catalyst_date IS NULL OR catalyst_date = '')
                      AND (market_cap IS NULL OR market_cap = 0)
                      AND description LIKE 'yfinance: no data%'
                """
                if only_with_universe:
                    del_sql += " AND ticker IN (SELECT DISTINCT ticker FROM catalyst_universe)"
                cur.execute(del_sql)
                deleted = cur.rowcount
                conn.commit()

            return {
                "dry_run": dry_run,
                "only_with_universe": only_with_universe,
                "candidates_count": len(candidates),
                "deleted": deleted,
                "candidates": candidates[:50],
            }
    except Exception as e:
        logger.exception("cleanup_dead_markers failed")
        raise HTTPException(500, f"cleanup error: {e}")


@router.post("/marketcap/mark-superseded-by-universe")
async def mark_screener_rows_superseded():
    """For tickers where catalyst_universe has at least one active catalyst,
    flag the legacy screener_stocks rows so /stocks endpoint can prefer V2
    data without showing duplicates.

    This is informational — current /stocks endpoint already merges + filters,
    but having a status field helps with bulk queries and screener listings.

    Currently a no-op marker for visibility; will become a real status update
    once a screener_stocks.status column exists. Returns count of tickers
    with V2 catalysts.
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(DISTINCT ticker) FROM catalyst_universe WHERE status='active'
            """)
            v2_tickers = cur.fetchone()[0]
            cur.execute("""
                SELECT COUNT(DISTINCT ticker) FROM screener_stocks
            """)
            legacy_tickers = cur.fetchone()[0]
            cur.execute("""
                SELECT COUNT(DISTINCT ss.ticker)
                FROM screener_stocks ss
                INNER JOIN catalyst_universe cu ON cu.ticker = ss.ticker
                WHERE cu.status = 'active'
            """)
            overlap = cur.fetchone()[0]
            return {
                "v2_active_tickers": v2_tickers,
                "legacy_screener_tickers": legacy_tickers,
                "overlap_tickers": overlap,
                "v2_only_tickers": v2_tickers - overlap,
                "legacy_only_tickers": legacy_tickers - overlap,
                "note": "/stocks endpoint already merges V2 + legacy and filters dead markers",
            }
    except Exception as e:
        logger.exception("mark_screener_rows_superseded failed")
        raise HTTPException(500, f"error: {e}")


# ============================================================
# Phase 3A backfill scheduler — runs in-process if SCHEDULER_ENABLED=1
# ============================================================

_post_catalyst_scheduler_state: dict = {
    "started": False,
    "last_run_at": None,
    "last_result": None,
    "runs_total": 0,
}


def _start_post_catalyst_scheduler_once() -> None:
    """Start a background asyncio task that runs the Phase 3A backfill on
    a schedule. Idempotent — multiple calls are no-ops. Only runs if env
    var POST_CATALYST_SCHEDULER_ENABLED=1."""
    if _post_catalyst_scheduler_state.get("started"):
        return
    if os.getenv("POST_CATALYST_SCHEDULER_ENABLED", "0") != "1":
        return
    _post_catalyst_scheduler_state["started"] = True

    import asyncio
    interval_hours = float(os.getenv("POST_CATALYST_SCHEDULER_HOURS", "6"))
    batch_limit = int(os.getenv("POST_CATALYST_SCHEDULER_LIMIT", "25"))
    min_age_days = int(os.getenv("POST_CATALYST_SCHEDULER_MIN_AGE_DAYS", "7"))

    async def _loop():
        # Wait 60s on startup so the app finishes initialization
        await asyncio.sleep(60)
        from datetime import datetime as _dt
        from services.post_catalyst_tracker import backfill_batch
        while True:
            try:
                logger.info(f"[post-catalyst-scheduler] running batch limit={batch_limit} min_age={min_age_days}")
                result = backfill_batch(limit=batch_limit, min_age_days=min_age_days)
                _post_catalyst_scheduler_state["last_run_at"] = _dt.utcnow().isoformat()
                _post_catalyst_scheduler_state["last_result"] = {
                    "processed": result.get("processed"),
                    "created": result.get("created"),
                    "failed": result.get("failed"),
                    "skipped": result.get("skipped"),
                }
                _post_catalyst_scheduler_state["runs_total"] += 1
                logger.info(f"[post-catalyst-scheduler] done: {result.get('created')} created, {result.get('failed')} failed, {result.get('skipped')} skipped")
            except Exception as e:
                logger.warning(f"[post-catalyst-scheduler] error: {e}")
                _post_catalyst_scheduler_state["last_result"] = {"error": str(e)[:200]}
            await asyncio.sleep(interval_hours * 3600)

    try:
        loop = asyncio.get_event_loop()
        loop.create_task(_loop())
        logger.info(f"[post-catalyst-scheduler] STARTED interval={interval_hours}h limit={batch_limit} min_age={min_age_days}d")
    except Exception as e:
        logger.warning(f"[post-catalyst-scheduler] failed to start: {e}")
        _post_catalyst_scheduler_state["started"] = False


@router.get("/post-catalyst/scheduler-status")
async def post_catalyst_scheduler_status():
    """Status of the Phase 3A backfill scheduler. Set env var
    POST_CATALYST_SCHEDULER_ENABLED=1 to enable.
    Tunable via:
      POST_CATALYST_SCHEDULER_HOURS (default 6)
      POST_CATALYST_SCHEDULER_LIMIT (default 25)
      POST_CATALYST_SCHEDULER_MIN_AGE_DAYS (default 7)"""
    return {
        "enabled": os.getenv("POST_CATALYST_SCHEDULER_ENABLED", "0") == "1",
        "interval_hours": float(os.getenv("POST_CATALYST_SCHEDULER_HOURS", "6")),
        "batch_limit": int(os.getenv("POST_CATALYST_SCHEDULER_LIMIT", "25")),
        "min_age_days": int(os.getenv("POST_CATALYST_SCHEDULER_MIN_AGE_DAYS", "7")),
        **_post_catalyst_scheduler_state,
    }


@router.post("/post-catalyst/scheduler-trigger-now")
async def post_catalyst_scheduler_trigger():
    """Manually trigger one round of the Phase 3A backfill, bypassing the
    scheduler. Equivalent to POST /admin/post-catalyst/backfill but uses
    the env-var defaults so it matches what the scheduler would do."""
    from services.post_catalyst_tracker import backfill_batch
    batch_limit = int(os.getenv("POST_CATALYST_SCHEDULER_LIMIT", "25"))
    min_age_days = int(os.getenv("POST_CATALYST_SCHEDULER_MIN_AGE_DAYS", "7"))
    return backfill_batch(limit=batch_limit, min_age_days=min_age_days)


# ============================================================
# Universe seeder scheduler — keeps catalyst_universe fresh
# ============================================================

_seeder_scheduler_state: dict = {
    "started": False,
    "last_run_at": None,
    "last_result": None,
    "runs_total": 0,
}


def _start_seeder_scheduler_once() -> None:
    """Start a background asyncio task that runs the V2 universe seeder
    on a schedule. Idempotent — multiple calls are no-ops. Only runs if
    env var SEEDER_SCHEDULER_ENABLED=1."""
    if _seeder_scheduler_state.get("started"):
        return
    if os.getenv("SEEDER_SCHEDULER_ENABLED", "0") != "1":
        return
    _seeder_scheduler_state["started"] = True

    import asyncio
    interval_hours = float(os.getenv("SEEDER_SCHEDULER_HOURS", "24"))
    max_tickers = os.getenv("SEEDER_SCHEDULER_MAX_TICKERS")  # None = full universe
    max_tickers_int = int(max_tickers) if max_tickers else None

    async def _loop():
        # Wait 5min on startup so the app finishes initialization and other
        # one-time work (alembic, cache warm) doesn't compete for resources.
        await asyncio.sleep(300)
        from datetime import datetime as _dt
        from services.universe_seeder import run_universe_seed
        while True:
            try:
                # Check that no other seed is running already (mutex from
                # the ad-hoc /v2-seed-async path)
                running = [j for j in _bg_seed_jobs.values() if j.get("status") == "running"]
                if running:
                    logger.info(f"[seeder-scheduler] skipping run — {len(running)} other job(s) running")
                    _seeder_scheduler_state["last_result"] = {"skipped": "other_job_running"}
                else:
                    # Use a thread so the asyncio loop isn't blocked by the
                    # synchronous seeder (which calls Gemini, yfinance, etc.)
                    job_id = "scheduler-" + _dt.utcnow().strftime("%Y%m%d-%H%M%S")
                    _bg_seed_jobs[job_id] = {
                        "job_id": job_id, "status": "running",
                        "started_at": _dt.utcnow().isoformat()+"Z",
                        "max_tickers": max_tickers_int, "start_idx": 0,
                        "source": "scheduler",
                    }
                    logger.info(f"[seeder-scheduler] starting run job_id={job_id} max_tickers={max_tickers_int}")
                    try:
                        # Run the seed in the executor so we don't block the loop
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, lambda: run_universe_seed(max_tickers=max_tickers_int, start_idx=0)
                        )
                        _bg_seed_jobs[job_id]["status"] = "completed"
                        _bg_seed_jobs[job_id]["result"] = result
                        _bg_seed_jobs[job_id]["completed_at"] = _dt.utcnow().isoformat()+"Z"
                        _seeder_scheduler_state["last_result"] = {
                            "tickers_processed": result.get("tickers_processed") if isinstance(result, dict) else None,
                            "catalysts_added": result.get("catalysts_added") if isinstance(result, dict) else None,
                            "job_id": job_id,
                        }
                        logger.info(f"[seeder-scheduler] done job_id={job_id}")
                    except Exception as e:
                        _bg_seed_jobs[job_id]["status"] = "failed"
                        _bg_seed_jobs[job_id]["error"] = f"{type(e).__name__}: {e}"
                        _bg_seed_jobs[job_id]["completed_at"] = _dt.utcnow().isoformat()+"Z"
                        _seeder_scheduler_state["last_result"] = {"error": str(e)[:200], "job_id": job_id}
                        logger.warning(f"[seeder-scheduler] error: {e}")

                _seeder_scheduler_state["last_run_at"] = _dt.utcnow().isoformat()
                _seeder_scheduler_state["runs_total"] += 1
            except Exception as e:
                logger.warning(f"[seeder-scheduler] loop error: {e}")
                _seeder_scheduler_state["last_result"] = {"error": str(e)[:200]}
            await asyncio.sleep(interval_hours * 3600)

    try:
        loop = asyncio.get_event_loop()
        loop.create_task(_loop())
        logger.info(f"[seeder-scheduler] STARTED interval={interval_hours}h max_tickers={max_tickers_int}")
    except Exception as e:
        logger.warning(f"[seeder-scheduler] failed to start: {e}")
        _seeder_scheduler_state["started"] = False


@router.get("/universe/seeder-scheduler-status")
async def seeder_scheduler_status():
    """Status of the universe seeder scheduler. Set env var
    SEEDER_SCHEDULER_ENABLED=1 to enable.
    Tunable via:
      SEEDER_SCHEDULER_HOURS (default 24)
      SEEDER_SCHEDULER_MAX_TICKERS (default unset = full universe)"""
    return {
        "enabled": os.getenv("SEEDER_SCHEDULER_ENABLED", "0") == "1",
        "interval_hours": float(os.getenv("SEEDER_SCHEDULER_HOURS", "24")),
        "max_tickers": int(os.getenv("SEEDER_SCHEDULER_MAX_TICKERS", "0")) or None,
        **_seeder_scheduler_state,
    }


# ============================================================
# news_count refresher — periodic background job
# Updates screener_stocks.news_count + sentiment_score for V2 tickers
# whose row was created by the marketcap backfill (news_count = 0).
# This makes the screener listing's "news_count" column meaningful.
# ============================================================

_news_refresher_state: dict = {
    "started": False,
    "last_run_at": None,
    "last_result": None,
    "runs_total": 0,
}


def _start_news_refresher_once() -> None:
    """Background task that updates news_count + sentiment_score in
    screener_stocks for tickers with stale or missing values. Idempotent.
    Only runs if NEWS_REFRESHER_ENABLED=1."""
    if _news_refresher_state.get("started"):
        return
    if os.getenv("NEWS_REFRESHER_ENABLED", "0") != "1":
        return
    _news_refresher_state["started"] = True

    import asyncio
    interval_hours = float(os.getenv("NEWS_REFRESHER_HOURS", "12"))
    batch_limit = int(os.getenv("NEWS_REFRESHER_BATCH", "50"))

    async def _loop():
        await asyncio.sleep(180)  # 3min startup delay
        from datetime import datetime as _dt
        while True:
            try:
                logger.info(f"[news-refresher] starting batch limit={batch_limit}")
                # Run sync work in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: _refresh_news_counts(batch_limit))
                _news_refresher_state["last_run_at"] = _dt.utcnow().isoformat()
                _news_refresher_state["last_result"] = result
                _news_refresher_state["runs_total"] += 1
                logger.info(f"[news-refresher] done: {result.get('updated',0)} updated, {result.get('errors',0)} errors")
            except Exception as e:
                logger.warning(f"[news-refresher] error: {e}")
                _news_refresher_state["last_result"] = {"error": str(e)[:200]}
            await asyncio.sleep(interval_hours * 3600)

    try:
        loop = asyncio.get_event_loop()
        loop.create_task(_loop())
        logger.info(f"[news-refresher] STARTED interval={interval_hours}h batch={batch_limit}")
    except Exception as e:
        logger.warning(f"[news-refresher] failed to start: {e}")
        _news_refresher_state["started"] = False


def _refresh_news_counts(limit: int = 50) -> dict:
    """Refresh news_count + sentiment_score in screener_stocks for V2 tickers.
    Picks tickers that are in catalyst_universe (active) but have stale or zero
    news_count in screener_stocks. Updates by calling fetcher.fetch_news_sentiment.
    Returns: {processed, updated, errors, sample_results}.
    """
    from services.database import BiotechDatabase
    from services.fetcher import BiotechDataFetcher
    db = BiotechDatabase()
    fetcher = BiotechDataFetcher(
        finnhub_api_key=os.getenv("FINNHUB_API_KEY", ""),
        newsapi_key=os.getenv("NEWSAPI_KEY", ""),
    )
    
    # Pick tickers with stale or 0 news_count, prioritizing those with
    # near-term catalysts (next 90 days). Use a subquery so SELECT DISTINCT
    # doesn't conflict with the ORDER BY clause (Postgres requires ORDER BY
    # columns to appear in SELECT list when DISTINCT is present).
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT ticker FROM (
                SELECT DISTINCT ON (cu.ticker)
                    cu.ticker,
                    cu.catalyst_date
                FROM catalyst_universe cu
                LEFT JOIN screener_stocks ss ON ss.ticker = cu.ticker
                WHERE cu.status = 'active'
                  AND cu.catalyst_date IS NOT NULL
                  AND cu.catalyst_date::date <= (CURRENT_DATE + INTERVAL '120 days')
                  AND (ss.news_count IS NULL OR ss.news_count = 0
                       OR ss.last_updated IS NULL
                       OR ss.last_updated < (NOW() - INTERVAL '24 hours')::text)
                ORDER BY cu.ticker, cu.catalyst_date ASC
            ) t
            ORDER BY catalyst_date ASC
            LIMIT %s
        """, (limit,))
        tickers = [r[0] for r in cur.fetchall()]
    
    if not tickers:
        return {"processed": 0, "updated": 0, "errors": 0, "note": "no tickers needing refresh"}
    
    updated = 0
    errors = 0
    results = []
    with db.get_conn() as conn:
        cur = conn.cursor()
        for ticker in tickers:
            try:
                ns = fetcher.fetch_news_sentiment(ticker, days_back=30)
                news_count = int(ns.get("news_count", 0))
                sentiment = float(ns.get("sentiment_score", 0))
                # UPDATE the most recent row for this ticker (or all rows)
                cur.execute("""
                    UPDATE screener_stocks
                    SET news_count = %s,
                        sentiment_score = %s,
                        last_updated = NOW()::text
                    WHERE ticker = %s
                """, (news_count, sentiment, ticker))
                updated += cur.rowcount
                results.append({"ticker": ticker, "news_count": news_count,
                                "sentiment_score": round(sentiment, 3),
                                "rows_updated": cur.rowcount})
            except Exception as e:
                errors += 1
                results.append({"ticker": ticker, "error": str(e)[:100]})
        conn.commit()
    
    return {
        "processed": len(tickers),
        "updated": updated,
        "errors": errors,
        "sample_results": results[:20],
    }


@router.get("/news-refresher/status")
async def news_refresher_status():
    """Status of the news_count refresher. Set NEWS_REFRESHER_ENABLED=1 to enable.
    Tunable via NEWS_REFRESHER_HOURS (default 12), NEWS_REFRESHER_BATCH (default 50)."""
    return {
        "enabled": os.getenv("NEWS_REFRESHER_ENABLED", "0") == "1",
        "interval_hours": float(os.getenv("NEWS_REFRESHER_HOURS", "12")),
        "batch_limit": int(os.getenv("NEWS_REFRESHER_BATCH", "50")),
        **_news_refresher_state,
    }


@router.post("/news-refresher/trigger-now")
async def news_refresher_trigger():
    """Manually trigger one round of news_count refresh."""
    batch = int(os.getenv("NEWS_REFRESHER_BATCH", "50"))
    return _refresh_news_counts(limit=batch)


# ============================================================
# Historical post-catalyst seeder
# Backfills 1-2 historical catalysts per V2 ticker so we have a real
# accuracy dataset, not just one Mounjaro datapoint.
# ============================================================

@router.post("/post-catalyst/seed-historical")
async def seed_historical_catalysts(
    per_ticker: int = 2,
    max_tickers: int = 50,
    dry_run: bool = True,
    use_llm: bool = True,
):
    """For each V2 ticker, ask the LLM for {per_ticker} notable historical
    catalysts (FDA decisions, Phase 3 readouts) from the last 5 years and
    backfill outcomes for them.
    
    This builds a real accuracy dataset for Phase 3A. Without it we have
    only forward-dated V2 catalysts and the single Mounjaro test row.
    
    Args:
      per_ticker: how many historical catalysts to ask LLM for per ticker (default 2)
      max_tickers: cap total tickers processed per call (default 50; cost control)
      dry_run: only fetch LLM suggestions, don't actually backfill
      use_llm: must be True (placeholder for future heuristic version)
    
    Returns: {tickers_processed, catalysts_seeded, llm_failures, results}
    """
    if not use_llm:
        raise HTTPException(400, "use_llm=false not yet implemented")
    
    try:
        from services.database import BiotechDatabase
        from services.llm_helper import call_llm_json
        from services.post_catalyst_tracker import backfill_one
        
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            # Pick tickers in catalyst_universe that haven't been seeded or marked.
            # Use LEFT JOIN to screener_stocks so V2-only tickers (which never
            # made it into screener_stocks via the seeder pipeline) still
            # participate. The seeder will fail gracefully on truly delisted
            # tickers (no yfinance data → backfill_one returns failed status).
            #
            # Order by screener_stocks.market_cap DESC NULLS LAST so the LIVE
            # well-documented names get tried first; V2-only catch up after.
            cur.execute("""
                SELECT t.ticker, t.company_name FROM (
                    SELECT DISTINCT ON (cu.ticker)
                        cu.ticker, cu.company_name, ss.market_cap
                    FROM catalyst_universe cu
                    LEFT JOIN screener_stocks ss ON ss.ticker = cu.ticker
                    WHERE cu.status = 'active'
                      AND cu.ticker IS NOT NULL
                      AND (
                        ss.ticker IS NULL  -- V2-only tickers, never in screener_stocks
                        OR (
                          ss.market_cap IS NOT NULL AND ss.market_cap > 0
                          -- Note: yfinance backfill rows DO have real market_cap data
                          -- written by /admin/marketcap/backfill -- only exclude the
                          -- "no data" marker rows for truly delisted tickers.
                          AND COALESCE(ss.description, '') NOT LIKE 'yfinance: no data%%'
                        )
                      )
                      AND cu.ticker NOT IN (
                        SELECT DISTINCT ticker FROM post_catalyst_outcomes
                        WHERE outcome IS NOT NULL
                          -- Exclude any ticker that has any non-null outcome row,
                          -- including 'unknown' stubs (means yfinance already failed
                          -- for this ticker — no point retrying every batch)
                      )
                    ORDER BY cu.ticker, ss.market_cap DESC NULLS LAST
                ) t
                ORDER BY t.market_cap DESC NULLS LAST
                LIMIT %s
            """, (max_tickers,))
            tickers = [(r[0], r[1] or r[0]) for r in cur.fetchall()]
        
        if not tickers:
            return {"tickers_processed": 0, "catalysts_seeded": 0,
                    "note": "all V2 tickers already have outcome rows"}
        
        results = []
        catalysts_seeded = 0
        llm_failures = 0
        
        for ticker, company_name in tickers:
            prompt = f"""You are populating a learning-loop dataset for biotech catalyst predictions.

For {ticker} ({company_name}), provide {per_ticker} notable HISTORICAL catalysts from the last 5 years (2020-2025) where you have high confidence about both the date and the outcome.

Prioritize:
1. FDA approvals/rejections (PDUFA decisions)
2. Phase 3 readouts (top-line results)
3. Major regulatory milestones (CRL, AdComm)

EXCLUDE: forward-dated events, ambiguous announcements, partnership news, earnings.

Respond with ONLY a JSON object:
{{
  "catalysts": [
    {{
      "catalyst_date": "YYYY-MM-DD",
      "catalyst_type": "FDA Decision" | "Phase 3 Readout" | "AdComm" | "Phase 2 Readout",
      "drug_name": "exact drug name (with code if applicable, e.g., 'risdiplam (Evrysdi)')",
      "indication": "specific indication treated",
      "outcome": "approved" | "rejected" | "delayed" | "mixed",
      "probability_at_time": 0.0-1.0,
      "rationale": "1-sentence factual citation"
    }}
  ]
}}

If you don't have high-confidence knowledge of {per_ticker} historical catalysts for {ticker}, return fewer (or empty array). Quality over quantity. Do NOT make up dates or outcomes."""
            
            try:
                result, err = call_llm_json(
                    prompt, max_tokens=800, temperature=0.1,
                    feature="historical_seed", ticker=ticker,
                )
                if not result or err:
                    llm_failures += 1
                    results.append({"ticker": ticker, "error": err or "no result"})
                    continue
                
                catalysts = result.get("catalysts", []) or []
                if not catalysts:
                    # Write a sentinel row so subsequent batches skip this ticker.
                    # We use catalyst_type='_no_history_known' + a fixed dummy date
                    # so it goes into post_catalyst_outcomes and the query above
                    # excludes it via 'outcome IS NOT NULL AND outcome != unknown'.
                    if not dry_run:
                        try:
                            with db.get_conn() as conn2:
                                cur2 = conn2.cursor()
                                cur2.execute("""
                                    INSERT INTO post_catalyst_outcomes
                                      (catalyst_id, ticker, catalyst_type, catalyst_date,
                                       outcome, outcome_confidence, outcome_source, outcome_notes,
                                       computed_at, last_updated, backfill_attempts)
                                    VALUES (NULL, %s, '_no_history_known', '1900-01-01',
                                            'no_history_known', 0.0, 'historical_seed_skip',
                                            'LLM returned no historical catalysts',
                                            NOW(), NOW(), 1)
                                    ON CONFLICT (ticker, catalyst_type, catalyst_date) DO UPDATE SET
                                        last_updated = NOW(),
                                        backfill_attempts = post_catalyst_outcomes.backfill_attempts + 1
                                """, (ticker,))
                                conn2.commit()
                        except Exception as e:
                            logger.info(f"skip-marker write failed for {ticker}: {e}")
                    results.append({"ticker": ticker, "catalysts_returned": 0,
                                     "note": "LLM returned no historical catalysts"})
                    continue
                
                ticker_results = []
                for cat in catalysts[:per_ticker]:
                    cat_date = cat.get("catalyst_date")
                    cat_type = cat.get("catalyst_type")
                    if not cat_date or not cat_type:
                        continue
                    
                    if dry_run:
                        ticker_results.append({
                            "date": cat_date, "type": cat_type,
                            "drug": cat.get("drug_name"),
                            "outcome_known": cat.get("outcome"),
                            "would_seed": True,
                        })
                    else:
                        # Backfill via the tracker — it'll fetch yfinance prices
                        # and run the full classifier pipeline
                        try:
                            br = backfill_one({
                                "id": None,
                                "ticker": ticker,
                                "catalyst_type": cat_type,
                                "catalyst_date": cat_date,
                                "drug_name": cat.get("drug_name"),
                                "indication": cat.get("indication"),
                                "confidence_score": cat.get("probability_at_time", 0.7),
                            })
                            if br.get("status") == "created":
                                catalysts_seeded += 1
                            ticker_results.append({
                                "date": cat_date, "type": cat_type,
                                "drug": cat.get("drug_name"),
                                "llm_outcome": cat.get("outcome"),
                                "actual_outcome": br.get("outcome"),
                                "actual_1d_pct": br.get("actual_1d_pct"),
                                "status": br.get("status"),
                            })
                        except Exception as e:
                            ticker_results.append({
                                "date": cat_date, "type": cat_type,
                                "error": str(e)[:100],
                            })
                
                results.append({"ticker": ticker, "catalysts": ticker_results})
            except Exception as e:
                llm_failures += 1
                results.append({"ticker": ticker, "error": str(e)[:100]})
        
        return {
            "tickers_processed": len(tickers),
            "catalysts_seeded": catalysts_seeded,
            "llm_failures": llm_failures,
            "dry_run": dry_run,
            "results": results,
        }
    except Exception as e:
        logger.exception("seed_historical_catalysts failed")
        raise HTTPException(500, f"error: {e}")


@router.get("/post-catalyst/seed-historical-diag")
async def seed_historical_diag():
    """Diagnostic: shows the candidate pool size at each stage of the seeder query."""
    from services.database import BiotechDatabase
    db = BiotechDatabase()
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT ticker) FROM catalyst_universe WHERE status='active'")
        active = cur.fetchone()[0]
        cur.execute("""
            SELECT COUNT(DISTINCT cu.ticker)
            FROM catalyst_universe cu
            INNER JOIN screener_stocks ss ON ss.ticker = cu.ticker
            WHERE cu.status = 'active'
              AND ss.market_cap IS NOT NULL AND ss.market_cap > 0
              -- 'yfinance backfill' rows have real market_cap, keep them
              AND COALESCE(ss.description, '') NOT LIKE 'yfinance: no data%%'
        """)
        live = cur.fetchone()[0]
        cur.execute("""
            SELECT COUNT(DISTINCT ticker) FROM post_catalyst_outcomes
            WHERE outcome IS NOT NULL
        """)
        seeded_or_marked = cur.fetchone()[0]
        cur.execute("""
            SELECT outcome, COUNT(*) FROM post_catalyst_outcomes
            WHERE outcome IS NOT NULL
            GROUP BY outcome ORDER BY 2 DESC
        """)
        by_outcome = {r[0]: r[1] for r in cur.fetchall()}
        cur.execute("""
            SELECT COUNT(*) FROM (
                SELECT DISTINCT ON (cu.ticker) cu.ticker
                FROM catalyst_universe cu
                LEFT JOIN screener_stocks ss ON ss.ticker = cu.ticker
                WHERE cu.status = 'active'
                  AND cu.ticker IS NOT NULL
                  AND (
                    ss.ticker IS NULL
                    OR (
                      ss.market_cap IS NOT NULL AND ss.market_cap > 0
                      -- 'yfinance backfill' rows have real market_cap, keep them
                      AND COALESCE(ss.description, '') NOT LIKE 'yfinance: no data%%'
                    )
                  )
                  AND cu.ticker NOT IN (
                    SELECT DISTINCT ticker FROM post_catalyst_outcomes
                    WHERE outcome IS NOT NULL
                  )
                ORDER BY cu.ticker, ss.market_cap DESC NULLS LAST
            ) t
        """)
        candidate_pool = cur.fetchone()[0]
        # Also: sample 5 candidates
        cur.execute("""
            SELECT t.ticker, t.market_cap FROM (
                SELECT DISTINCT ON (cu.ticker) cu.ticker, ss.market_cap
                FROM catalyst_universe cu
                LEFT JOIN screener_stocks ss ON ss.ticker = cu.ticker
                WHERE cu.status = 'active'
                  AND cu.ticker IS NOT NULL
                  AND (
                    ss.ticker IS NULL
                    OR (
                      ss.market_cap IS NOT NULL AND ss.market_cap > 0
                      -- 'yfinance backfill' rows have real market_cap, keep them
                      AND COALESCE(ss.description, '') NOT LIKE 'yfinance: no data%%'
                    )
                  )
                  AND cu.ticker NOT IN (
                    SELECT DISTINCT ticker FROM post_catalyst_outcomes
                    WHERE outcome IS NOT NULL
                  )
                ORDER BY cu.ticker, ss.market_cap DESC NULLS LAST
            ) t
            ORDER BY t.market_cap DESC NULLS LAST
            LIMIT 8
        """)
        sample = [{"ticker": r[0], "market_cap_m": float(r[1]) if r[1] else 0} for r in cur.fetchall()]
    return {
        "active_v2_tickers": active,
        "live_tickers_in_screener": live,
        "tickers_with_seeded_or_marked": seeded_or_marked,
        "candidate_pool_remaining": candidate_pool,
        "by_outcome_count": by_outcome,
        "next_8_candidates": sample,
    }


@router.get("/post-catalyst/move-stats")
async def move_stats():
    """Distribution of actual_move_pct_1d grouped by catalyst_type + outcome.
    Used to recalibrate REFERENCE_MOVES in services/post_catalyst_tracker.py."""
    from services.database import BiotechDatabase
    db = BiotechDatabase()
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                catalyst_type,
                outcome,
                COUNT(*) AS n,
                ROUND(AVG(actual_move_pct_1d)::numeric, 2) AS avg_1d,
                ROUND(STDDEV(actual_move_pct_1d)::numeric, 2) AS std_1d,
                ROUND(MIN(actual_move_pct_1d)::numeric, 2) AS min_1d,
                ROUND(MAX(actual_move_pct_1d)::numeric, 2) AS max_1d,
                ROUND(percentile_cont(0.25) WITHIN GROUP (ORDER BY actual_move_pct_1d)::numeric, 2) AS p25,
                ROUND(percentile_cont(0.50) WITHIN GROUP (ORDER BY actual_move_pct_1d)::numeric, 2) AS median,
                ROUND(percentile_cont(0.75) WITHIN GROUP (ORDER BY actual_move_pct_1d)::numeric, 2) AS p75
            FROM post_catalyst_outcomes
            WHERE actual_move_pct_1d IS NOT NULL
              AND outcome NOT IN ('no_history_known', 'unknown')
            GROUP BY catalyst_type, outcome
            ORDER BY catalyst_type, outcome
        """)
        rows = cur.fetchall()
    return {
        "total_outcomes_with_moves": sum(r[2] for r in rows),
        "by_catalyst_outcome": [
            {
                "catalyst_type": r[0],
                "outcome": r[1],
                "n": r[2],
                "avg_1d_pct": float(r[3]) if r[3] else None,
                "std_1d_pct": float(r[4]) if r[4] else None,
                "min_1d_pct": float(r[5]) if r[5] else None,
                "max_1d_pct": float(r[6]) if r[6] else None,
                "p25_1d_pct": float(r[7]) if r[7] else None,
                "median_1d_pct": float(r[8]) if r[8] else None,
                "p75_1d_pct": float(r[9]) if r[9] else None,
            } for r in rows
        ],
    }


@router.post("/post-catalyst/recompute-predictions")
async def recompute_predictions(limit: int = 1000):
    """Recompute predicted_move_pct for existing post_catalyst_outcomes rows
    using the current REF_MOVES table in services/post_catalyst_tracker.py.
    
    Why: REF_MOVES was recalibrated (commit 559e018). Existing rows have
    predictions frozen from when they were first seeded with the old aggressive
    table — accuracy metrics still reflect that. Run this once to bring
    historical predictions in line with the calibrated values."""
    from services.database import BiotechDatabase
    
    # Inline the calibrated table here so this endpoint doesn't depend on
    # post_catalyst_tracker exporting it.
    REF_MOVES = {
        "FDA Decision": (4, -5),
        "PDUFA Decision": (4, -5),
        "Regulatory Decision": (4, -5),
        "AdComm": (8, -10),
        "Advisory Committee": (8, -10),
        "Phase 3 Readout": (3, -5),
        "Phase 3": (3, -5),
        "Phase 2 Readout": (4, -2),
        "Phase 2": (4, -2),
        "Phase 1/2 Readout": (8, -6),
        "Phase 1 Readout": (10, -6),
        "Phase 1": (10, -6),
        "Clinical Trial Readout": (5, -3),
        "Clinical Trial": (5, -3),
        "NDA submission": (2, -2),
        "BLA submission": (2, -2),
        "Partnership": (5, -2),
        "Earnings": (3, -3),
        "Product Launch": (4, -4),
        "Commercial Launch": (4, -4),
    }
    DEFAULT = (4, -4)
    
    db = BiotechDatabase()
    updated = 0
    skipped = 0
    error_changed_total = 0.0
    
    with db.get_conn() as conn:
        cur = conn.cursor()
        # Pull all outcome rows with predicted_prob set
        cur.execute("""
            SELECT id, catalyst_type, predicted_prob, actual_move_pct_30d, actual_move_pct_1d, predicted_move_pct
            FROM post_catalyst_outcomes
            WHERE predicted_prob IS NOT NULL
              AND outcome IS NOT NULL
              AND outcome NOT IN ('no_history_known')
            ORDER BY computed_at DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
        
        for row_id, ctype, prob, actual_30d, actual_1d, old_pred in rows:
            up, down = REF_MOVES.get(ctype or "", DEFAULT)
            p = float(prob) if prob is not None else 0.5
            new_pred = p * up + (1 - p) * down
            
            # Recompute error metrics
            actual_for_error = actual_30d if actual_30d is not None else actual_1d
            if actual_for_error is None:
                skipped += 1
                continue
            new_err_abs = abs(new_pred - float(actual_for_error))
            new_err_signed = new_pred - float(actual_for_error)
            new_dir_correct = (
                (new_pred > 0 and actual_for_error > 0) or
                (new_pred < 0 and actual_for_error < 0) or
                (abs(new_pred) < 1 and abs(actual_for_error) < 5)
            )
            
            cur.execute("""
                UPDATE post_catalyst_outcomes
                SET predicted_move_pct = %s,
                    error_abs_pct = %s,
                    error_signed_pct = %s,
                    direction_correct = %s,
                    last_updated = NOW()
                WHERE id = %s
            """, (new_pred, new_err_abs, new_err_signed, new_dir_correct, row_id))
            updated += 1
            error_changed_total += abs(float(old_pred or 0) - new_pred)
        
        conn.commit()
    
    return {
        "updated": updated,
        "skipped": skipped,
        "avg_prediction_change_pts": round(error_changed_total / max(updated, 1), 2),
        "note": "Predictions recomputed using calibrated REF_MOVES. /v2/post-catalyst/accuracy will reflect changes immediately.",
    }


@router.get("/fda-sources/lookup")
@router.get("/fda-sources/lookup")
async def fda_sources_lookup(drug_name: str, indication: Optional[str] = None):
    """Spot-check the FDA + ClinicalTrials.gov facts our system has for a drug.
    
    Used to:
    - Validate the fda_sources integration is working before NPV calls
    - Inspect what's available before manually pinning provenance
    - Compare LLM-claimed numbers vs. official records
    
    Example: GET /admin/fda-sources/lookup?drug_name=Repatha&indication=hyperlipidemia
    """
    try:
        from services.fda_sources import gather_verified_facts, format_verified_facts_for_prompt
        facts = gather_verified_facts(ticker="?", drug_name=drug_name, indication=indication)
        # Also include the prompt-block render so user can see what the LLM
        # would receive
        prompt_block = format_verified_facts_for_prompt(facts)
        return {
            "drug_name": drug_name,
            "indication": indication,
            "facts": facts,
            "prompt_block_preview": prompt_block[:2000],
        }
    except Exception as e:
        logger.exception("fda_sources_lookup failed")
        raise HTTPException(500, f"fda_sources error: {e}")


# ────────────────────────────────────────────────────────────
# Research corpus — URL ingestion + retrieval
# ────────────────────────────────────────────────────────────

class ResearchIngestRequest(BaseModel):
    url: str
    ticker_hint: Optional[str] = None
    cookies: Optional[str] = None  # "k1=v1; k2=v2" — passed per-request, never stored


@router.post("/research/ingest")
async def research_ingest(req: ResearchIngestRequest):
    """Ingest a URL: fetch → extract → LLM summarize → embed → store.

    Solves the SA-scraper question by inverting it. User pastes any URL
    (Seeking Alpha, Substack, IR page, transcript) and we extract structured
    insights into research_corpus for retrieval at NPV time.

    Cookies are passed per-request and never stored — caller can supply
    SA Premium auth cookies for paywalled articles.
    """
    try:
        from services.research_ingestor import ingest_url
        result = ingest_url(
            url=req.url,
            ticker_hint=(req.ticker_hint or "").upper().strip() or None,
            cookies=req.cookies,
        )
        return result
    except Exception as e:
        logger.exception("research_ingest failed")
        raise HTTPException(500, f"ingest error: {e}")


@router.get("/research/list")
async def research_list(ticker: Optional[str] = None, limit: int = 50):
    """List ingested articles. Filter by ticker if given."""
    try:
        from services.research_ingestor import list_corpus
        items = list_corpus(ticker=(ticker or "").upper().strip() or None, limit=min(limit, 200))
        return {"count": len(items), "items": items}
    except Exception as e:
        logger.exception("research_list failed")
        raise HTTPException(500, f"list error: {e}")


@router.get("/research/relevant")
async def research_relevant(ticker: str, indication: Optional[str] = None,
                             query: Optional[str] = None, limit: int = 5):
    """Return relevant research_corpus entries for a ticker / indication.

    Uses pgvector cosine similarity on embeddings. Direct ticker_hint matches
    are ranked first. Used by V2 NPV at analysis time as Layer 5 user research.
    """
    try:
        from services.research_ingestor import find_relevant_research
        items = find_relevant_research(
            ticker=ticker.upper(),
            indication=indication,
            query_text=query,
            limit=min(limit, 20),
        )
        return {"ticker": ticker.upper(), "count": len(items), "items": items}
    except Exception as e:
        logger.exception("research_relevant failed")
        raise HTTPException(500, f"relevant error: {e}")


@router.get("/orange-book/status")
async def orange_book_status(force_refresh: bool = False):
    """Diagnose Orange Book download/parse pipeline.
    Use force_refresh=true to bypass Redis cache."""
    try:
        from services.fda_sources import _get_orange_book_lookup, _OB_CACHE_KEY, _redis_client
        r = _redis_client()
        cache_present = False
        cache_meta = None
        neg_cache = None
        if r:
            try:
                raw = r.get(_OB_CACHE_KEY)
                cache_present = raw is not None
                if cache_present:
                    import json
                    parsed = json.loads(raw)
                    cache_meta = parsed.get("_meta")
                neg = r.get(_OB_CACHE_KEY + ":neg")
                if neg:
                    import json
                    neg_cache = json.loads(neg)
            except Exception as e:
                pass

        # If forced or no cache, attempt download
        if force_refresh or (not cache_present and not neg_cache):
            lookup = _get_orange_book_lookup(force_refresh=True)
            if lookup and lookup.get("_download_failed"):
                return {
                    "status": "download_failed",
                    "attempts": lookup.get("_attempts"),
                    "zip_error": lookup.get("_zip_error"),
                    "buf_size": lookup.get("_buf_size"),
                }
            elif lookup:
                return {
                    "status": "ok",
                    "fresh_download": True,
                    "meta": lookup.get("_meta"),
                }
        return {
            "status": "ok" if cache_present else ("download_failed_recently" if neg_cache else "no_cache"),
            "cache_present": cache_present,
            "cache_meta": cache_meta,
            "neg_cache": neg_cache,
        }
    except Exception as e:
        logger.exception("orange_book_status failed")
        raise HTTPException(500, f"orange_book_status error: {e}")


# ────────────────────────────────────────────────────────────
# Polygon — historical options + news
# ────────────────────────────────────────────────────────────

@router.get("/polygon/status")
async def polygon_status():
    """Diagnose Polygon API connectivity + plan tier coverage."""
    try:
        from services.polygon_data import diagnostic
        return diagnostic()
    except Exception as e:
        logger.exception("polygon_status failed")
        raise HTTPException(500, f"polygon_status error: {e}")


@router.get("/polygon/test-implied-move")
async def polygon_test_implied_move(ticker: str, target_date: str):
    """Direct test of Polygon current-snapshot implied move computation.

    Bypasses the yfinance-fallback layer in services.options_implied so we
    can verify Polygon is the source. Returns the polygon-only result or
    {available: false} if Polygon failed or returned empty.
    """
    try:
        from services.polygon_data import get_implied_move_polygon
        result = get_implied_move_polygon(ticker=ticker, target_date=target_date)
        if result is None:
            return {"available": False, "ticker": ticker, "target_date": target_date}
        return {"available": True, **result}
    except Exception as e:
        logger.exception("polygon test_implied_move failed")
        raise HTTPException(500, f"polygon test_implied_move error: {e}")


@router.get("/polygon/options-chain")
async def polygon_options_chain(ticker: str, as_of: str):
    """Fetch historical options chain for a ticker on a specific date.
    Used to verify the chain we'll use for backfill matches reality."""
    try:
        from services.polygon_data import fetch_historical_options_chain, compute_implied_move_from_chain
        chain = fetch_historical_options_chain(ticker=ticker.upper(), as_of_date=as_of)
        if not chain or chain.get("_status") == "not_available":
            return {"status": "not_available", "ticker": ticker, "as_of": as_of,
                    "_polygon_status": chain.get("_polygon_status") if chain else None}
        move = compute_implied_move_from_chain(chain)
        return {
            "status": "ok",
            "ticker": ticker.upper(),
            "as_of": as_of,
            "chain_size": chain.get("n_contracts"),
            "underlying_price": chain.get("underlying_price"),
            "implied_move": move,
            "from_cache": chain.get("_from_cache", False),
        }
    except Exception as e:
        logger.exception("polygon_options_chain failed")
        raise HTTPException(500, f"options_chain error: {e}")


class OptionsBackfillRequest(BaseModel):
    max_outcomes: int = 50
    skip_with_existing: bool = True  # don't re-fetch outcomes that already have options_implied_move_pct
    target_dte_min: int = 7
    target_dte_max: int = 60


@router.post("/polygon/backfill-options-implied")
async def polygon_backfill_options_implied(req: OptionsBackfillRequest):
    """Backfill options_implied_move_pct on post_catalyst_outcomes using
    Polygon historical chains (T-1 from catalyst date).

    This replaces the current implementation which captures TODAY's chain
    at backfill time — meaningless for events that happened months ago.
    """
    try:
        from services.polygon_data import is_configured, backfill_historical_implied_move
        from services.database import BiotechDatabase

        if not is_configured():
            raise HTTPException(400, "POLYGON_API_KEY not configured (check for trailing-space typo on Railway)")

        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            # Pick outcomes ordered by recency; skip ones that already have data
            # if requested.
            where_clause = ""
            if req.skip_with_existing:
                where_clause = "AND (options_implied_move_pct IS NULL OR options_implied_move_pct = 0)"
            cur.execute(f"""
                SELECT id, ticker, catalyst_date, catalyst_type, outcome_classification
                FROM post_catalyst_outcomes
                WHERE catalyst_date IS NOT NULL
                  {where_clause}
                ORDER BY catalyst_date DESC
                LIMIT %s
            """, (req.max_outcomes,))
            rows = cur.fetchall()

        results = {"attempted": 0, "succeeded": 0, "no_chain": 0, "no_straddle": 0,
                   "errors": [], "samples": []}
        for row in rows:
            results["attempted"] += 1
            outcome_id, ticker, cat_date, cat_type, _ = row
            try:
                cat_str = cat_date.strftime("%Y-%m-%d") if hasattr(cat_date, "strftime") else str(cat_date)[:10]
                bf = backfill_historical_implied_move(
                    ticker=ticker, catalyst_date=cat_str,
                    target_dte_min=req.target_dte_min, target_dte_max=req.target_dte_max,
                )
                if not bf:
                    # Could be no chain available, or no straddle pricing
                    results["no_chain"] += 1
                    continue
                imp = bf.get("implied_move_pct")
                if imp is None:
                    results["no_straddle"] += 1
                    continue
                # Update DB
                with db.get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE post_catalyst_outcomes
                        SET options_implied_move_pct = %s,
                            options_implied_meta = %s::jsonb
                        WHERE id = %s
                    """, (imp, json.dumps({
                        "method": "polygon_historical_chain",
                        "as_of": bf.get("as_of_date"),
                        "expiration": bf.get("expiration"),
                        "atm_strike": bf.get("atm_strike"),
                        "dte": bf.get("dte"),
                        "underlying_price": bf.get("underlying_price"),
                        "computed_at": __import__("datetime").datetime.utcnow().isoformat(),
                    }), outcome_id))
                    conn.commit()
                results["succeeded"] += 1
                if len(results["samples"]) < 5:
                    results["samples"].append({
                        "id": outcome_id, "ticker": ticker, "catalyst_date": cat_str,
                        "implied_move_pct": imp, "expiration": bf.get("expiration"),
                    })
            except Exception as e:
                if len(results["errors"]) < 10:
                    results["errors"].append({"id": outcome_id, "ticker": ticker, "error": str(e)[:100]})
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("polygon backfill failed")
        raise HTTPException(500, f"backfill error: {e}")


@router.get("/polygon/news")
async def polygon_news(ticker: str, since: Optional[str] = None, limit: int = 20):
    """Fetch Polygon news for a ticker. Optional `since=YYYY-MM-DD`."""
    try:
        from services.polygon_data import fetch_news
        result = fetch_news(
            ticker=ticker.upper(),
            published_utc_gte=since,
            limit=min(limit, 100),
        )
        if not result:
            raise HTTPException(503, "polygon news fetch failed")
        # Strip the full body in summary view
        return {
            "ticker": ticker.upper(),
            "count": result.get("count"),
            "from_cache": result.get("_from_cache", False),
            "articles": [
                {
                    "title": a.get("title"),
                    "publisher": (a.get("publisher") or {}).get("name"),
                    "published_utc": a.get("published_utc"),
                    "article_url": a.get("article_url"),
                    "description": (a.get("description") or "")[:300],
                    "insights": a.get("insights") or [],
                    "tickers": a.get("tickers") or [],
                }
                for a in (result.get("articles") or [])[:limit]
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("polygon_news failed")
        raise HTTPException(500, f"news error: {e}")


class PolygonNewsIngestRequest(BaseModel):
    ticker: str
    since_days: int = 30
    max_articles: int = 20


@router.post("/polygon/news-ingest")
async def polygon_news_ingest(req: PolygonNewsIngestRequest):
    """Pull recent Polygon news for a ticker, push each article URL through
    research_ingestor → research_corpus. This populates Layer 5 with
    professionally-aggregated article bodies.

    Cost: ~$0.001-0.005 per article (LLM extraction + embedding).
    """
    try:
        from services.polygon_data import fetch_news, is_configured
        from services.research_ingestor import ingest_url

        if not is_configured():
            raise HTTPException(400, "POLYGON_API_KEY not configured")

        ticker = req.ticker.upper()
        since = (datetime.utcnow() - __import__("datetime").timedelta(days=req.since_days)).strftime("%Y-%m-%dT00:00:00Z")
        result = fetch_news(ticker=ticker, published_utc_gte=since,
                            limit=min(req.max_articles, 100))
        if not result or not result.get("articles"):
            return {"ticker": ticker, "ingested": 0, "skipped": 0, "errors": [],
                    "_note": "no articles returned by Polygon"}

        ingested = 0
        skipped = 0
        errors = []
        samples = []
        for a in (result.get("articles") or [])[:req.max_articles]:
            url = a.get("article_url")
            if not url:
                skipped += 1
                continue
            try:
                r = ingest_url(url=url, ticker_hint=ticker)
                if r.get("status") == "ok":
                    ingested += 1
                    if len(samples) < 5:
                        samples.append({
                            "id": r.get("id"),
                            "title": (r.get("title") or "")[:80],
                            "url_domain": r.get("url_domain"),
                        })
                else:
                    skipped += 1
                    if len(errors) < 5:
                        errors.append({"url": url[:80], "error": r.get("error")})
            except Exception as e:
                if len(errors) < 5:
                    errors.append({"url": url[:80], "error": str(e)[:100]})

        return {"ticker": ticker, "ingested": ingested, "skipped": skipped,
                "errors": errors, "samples": samples}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("polygon_news_ingest failed")
        raise HTTPException(500, f"news_ingest error: {e}")


@router.get("/orange-book/discover-url")
async def orange_book_discover_url():
    """Discover the current Orange Book download URL by scraping FDA's
    data-files page (works from inside Railway, not sandbox).

    This solves the problem that the historical /media/76860/download URL
    has been returning 404 — FDA periodically renames bundle endpoints
    when they release new versions.

    Returns: list of all candidate ZIP/download URLs found on the page,
             with sizes if reachable.
    """
    import re, urllib.request
    page_url = "https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files"
    try:
        req = urllib.request.Request(page_url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; biotech-screener/1.0)",
            "Accept": "*/*",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return {"status": "page_fetch_failed", "error": str(e)[:200], "page_url": page_url}

    # Find candidate URLs
    candidates = set()
    for m in re.finditer(r'href="([^"]+)"', html):
        href = m.group(1)
        if any(k in href.lower() for k in ["orange", "download", ".zip"]):
            if href.startswith("/"):
                href = "https://www.fda.gov" + href
            if href.startswith("http"):
                candidates.add(href[:300])

    # Probe each — HEAD request to get size + status
    probes = []
    for url in sorted(candidates)[:20]:
        try:
            req = urllib.request.Request(url, method="HEAD", headers={
                "User-Agent": "Mozilla/5.0 (compatible; biotech-screener/1.0)",
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                probes.append({
                    "url": url,
                    "status": resp.status,
                    "content_length": resp.headers.get("Content-Length"),
                    "content_type": resp.headers.get("Content-Type"),
                    "last_modified": resp.headers.get("Last-Modified"),
                })
        except Exception as e:
            probes.append({"url": url, "status": "err", "error": str(e)[:100]})

    # Highlight the most likely OB ZIP (largest content-length, .zip in URL)
    likely = None
    for p in probes:
        if (p.get("status") == 200 and
            (p.get("content_length") and int(p.get("content_length")) > 1_000_000) and
            (".zip" in p.get("url", "").lower() or
             (p.get("content_type") or "").startswith("application/zip"))):
            if likely is None or int(p.get("content_length", 0)) > int(likely.get("content_length", 0)):
                likely = p

    return {
        "status": "ok",
        "page_url": page_url,
        "page_size": len(html),
        "n_candidates": len(candidates),
        "probes": probes,
        "likely_orange_book_url": likely,
    }


@router.get("/polygon/options-chain-raw")
async def polygon_options_chain_raw(ticker: str, as_of: str, sample: int = 5):
    """Dump raw sample contracts from a chain — for debugging compute_implied_move."""
    try:
        from services.polygon_data import fetch_historical_options_chain
        chain = fetch_historical_options_chain(ticker=ticker.upper(), as_of_date=as_of)
        if not chain or chain.get("_status") == "not_available":
            return {"status": "not_available"}
        contracts = chain.get("contracts") or []
        # Sample diverse contracts: 5 calls and 5 puts at varying strikes
        calls = [c for c in contracts if c.get("type") == "call"][:sample]
        puts = [c for c in contracts if c.get("type") == "put"][:sample]
        # Group by expiration to see what's available
        exps = sorted(set((c.get("expiration"), c.get("type")) for c in contracts if c.get("expiration")))
        exp_summary = {}
        for exp, ct in exps:
            exp_summary.setdefault(exp, {"calls": 0, "puts": 0})
            exp_summary[exp][ct + "s"] += 1
        return {
            "ticker": ticker, "as_of": as_of,
            "underlying_price": chain.get("underlying_price"),
            "n_contracts": chain.get("n_contracts"),
            "expirations": exp_summary,
            "sample_calls": calls,
            "sample_puts": puts,
        }
    except Exception as e:
        logger.exception("polygon_options_chain_raw failed")
        raise HTTPException(500, f"raw chain error: {e}")


@router.get("/polygon/aggs-probe")
async def polygon_aggs_probe(contract_ticker: str, as_of: str):
    """Single-call probe of /v2/aggs/ticker/{contract}/range/1/day/{as_of}/{as_of}.
    Used to verify whether per-option historical OHLC actually returns data
    (some Polygon plans may not include this). FAST — one HTTP call.
    
    Example: contract_ticker=O:AAPL260116C00200000, as_of=2026-01-16
    """
    try:
        from services.polygon_data import _http_get, POLYGON_BASE
        url = f"{POLYGON_BASE}/v2/aggs/ticker/{contract_ticker}/range/1/day/{as_of}/{as_of}"
        data = _http_get(url, params={"adjusted": "true"}, timeout=8)
        return {
            "contract": contract_ticker, "as_of": as_of, "url": url,
            "raw_response": data,
        }
    except Exception as e:
        logger.exception("polygon_aggs_probe failed")
        raise HTTPException(500, f"probe error: {e}")


@router.get("/polygon/contracts-probe")
async def polygon_contracts_probe(ticker: str, as_of: str,
                                    expired: str = "true"):
    """Single-call probe of /v3/reference/options/contracts.
    Returns first 5 contracts so we can see what's available."""
    try:
        from services.polygon_data import _http_get, POLYGON_BASE
        params = {
            "underlying_ticker": ticker, "as_of": as_of, "expired": expired,
            "limit": 5, "order": "asc", "sort": "expiration_date",
            "expiration_date.gte": as_of,
        }
        data = _http_get(f"{POLYGON_BASE}/v3/reference/options/contracts", params=params)
        return {"ticker": ticker, "as_of": as_of, "expired": expired,
                "raw_response": data}
    except Exception as e:
        logger.exception("polygon_contracts_probe failed")
        raise HTTPException(500, f"probe error: {e}")


@router.get("/system-state")
async def system_state():
    """One-shot comprehensive diagnostic. Replaces 10+ separate admin calls.

    Returns the full state of the biotech-V2 system in a single response —
    designed so external diagnostics can succeed in one egress-proxy round trip
    instead of being killed by intermittent flapping.

    Sections:
      - alembic_versions      (which migrations applied)
      - backtest              (N=358 outcomes summary, accuracy, options coverage)
      - drug_economics        (count, avg confidence, FDA-anchored count)
      - research_corpus       (count by domain, ticker)
      - polygon               (key configured, plan checks pass)
      - fda_sources           (orange book cache state)
      - recent_npv_runs       (last 5 NPV computations)
    """
    from datetime import datetime
    out = {"as_of": datetime.utcnow().isoformat()}

    # 1. Alembic
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT version_num FROM alembic_version_biotech")
            out["alembic_versions"] = [r[0] for r in cur.fetchall()]
    except Exception as e:
        out["alembic_versions"] = {"_error": str(e)[:120]}

    # 2. Backtest summary
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT count(*) total,
                       count(*) FILTER (WHERE error_abs_pct IS NOT NULL) with_error,
                       avg(error_abs_pct)::numeric(10,2) avg_abs_err,
                       avg(error_signed_pct)::numeric(10,2) avg_signed_err,
                       avg(CASE WHEN direction_correct THEN 1.0 ELSE 0.0 END)::numeric(10,3) dir_acc,
                       count(*) FILTER (WHERE options_implied_move_pct IS NOT NULL AND options_implied_move_pct > 0) with_options
                FROM post_catalyst_outcomes
            """)
            r = cur.fetchone()
            out["backtest"] = {
                "total_outcomes": r[0],
                "with_error": r[1],
                "avg_abs_error_pct": float(r[2]) if r[2] else None,
                "avg_signed_error_pct": float(r[3]) if r[3] else None,
                "direction_accuracy": float(r[4]) if r[4] else None,
                "with_options_implied": r[5],
                "options_coverage_pct": round(100.0 * r[5] / r[0], 1) if r[0] else 0,
            }
    except Exception as e:
        out["backtest"] = {"_error": str(e)[:120]}

    # 3. Drug economics cache
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT count(*) total,
                       count(confidence_score) with_confidence,
                       avg(confidence_score)::numeric(10,3) avg_confidence,
                       count(*) FILTER (WHERE annual_cost_us_net_usd IS NOT NULL) with_us_net,
                       count(*) FILTER (WHERE provenance IS NOT NULL) with_provenance
                FROM drug_economics_cache
            """)
            r = cur.fetchone()
            out["drug_economics"] = {
                "total": r[0],
                "with_confidence_score": r[1],
                "avg_confidence_score": float(r[2]) if r[2] else None,
                "with_us_net_pricing": r[3],
                "with_provenance": r[4],
            }
    except Exception as e:
        out["drug_economics"] = {"_error": str(e)[:120]}

    # 4. Research corpus
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT count(*), count(DISTINCT ticker_hint), count(DISTINCT url_domain) FROM research_corpus")
            r = cur.fetchone()
            cur.execute("""SELECT url_domain, count(*)
                           FROM research_corpus
                           GROUP BY url_domain ORDER BY count(*) DESC LIMIT 10""")
            domains = {row[0]: row[1] for row in cur.fetchall()}
            cur.execute("""SELECT ticker_hint, count(*)
                           FROM research_corpus
                           WHERE ticker_hint IS NOT NULL
                           GROUP BY ticker_hint ORDER BY count(*) DESC LIMIT 10""")
            tickers = {row[0]: row[1] for row in cur.fetchall()}
            out["research_corpus"] = {
                "total": r[0],
                "distinct_tickers": r[1],
                "distinct_domains": r[2],
                "top_domains": domains,
                "top_tickers": tickers,
            }
    except Exception as e:
        out["research_corpus"] = {"_error": str(e)[:120]}

    # 5. Polygon
    try:
        from services.polygon_data import is_configured, _get_api_key
        if is_configured():
            out["polygon"] = {
                "configured": True,
                "key_prefix": _get_api_key()[:6] + "...",
            }
        else:
            out["polygon"] = {"configured": False}
    except Exception as e:
        out["polygon"] = {"_error": str(e)[:120]}

    # 6. FDA sources
    try:
        from services.fda_sources import _redis_client, _OB_CACHE_KEY
        r = _redis_client()
        ob_present = False
        ob_meta = None
        if r:
            try:
                raw = r.get(_OB_CACHE_KEY)
                if raw:
                    import json
                    ob_present = True
                    parsed = json.loads(raw)
                    ob_meta = parsed.get("_meta")
            except Exception:
                pass
        out["fda_sources"] = {
            "orange_book_cached": ob_present,
            "orange_book_meta": ob_meta,
        }
    except Exception as e:
        out["fda_sources"] = {"_error": str(e)[:120]}

    # 7. Recent NPV runs
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT ticker, drug_npv_b, p_approval, p_commercial, computed_at
                FROM catalyst_npv_cache
                ORDER BY computed_at DESC
                LIMIT 5
            """)
            out["recent_npv_runs"] = [
                {"ticker": r[0], "drug_npv_b": float(r[1]) if r[1] else None,
                 "p_approval": float(r[2]) if r[2] else None,
                 "p_commercial": float(r[3]) if r[3] else None,
                 "computed_at": str(r[4]) if r[4] else None}
                for r in cur.fetchall()
            ]
    except Exception as e:
        out["recent_npv_runs"] = {"_error": str(e)[:120]}

    return out


@router.get("/sec/capital-structure")
async def sec_capital_structure(ticker: str):
    """Inspect SEC EDGAR capital structure for a ticker — cash, debt, shares,
    burn rate, runway. Used to spot-check the data feeding equity_value
    computation in /analyze/npv.
    """
    try:
        from services.sec_financials import fetch_capital_structure
        result = fetch_capital_structure(ticker.upper())
        if not result:
            raise HTTPException(404, "no SEC data for ticker")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("sec capital_structure failed")
        raise HTTPException(500, f"sec_capital_structure error: {e}")


@router.get("/post-catalyst/inspect-stubs")
async def inspect_unfilled_stubs(limit: int = 30):
    """List the post_catalyst_outcomes rows that have no actual_move_pct
    populated. Helps explain why the retry-with-polygon endpoint reports
    0 succeeded — usually these are old dates beyond yfinance/Polygon
    history, or delisted tickers.
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, ticker, catalyst_type, catalyst_date,
                       outcome, last_error, backfill_attempts, actual_move_pct_1d
                FROM post_catalyst_outcomes
                WHERE actual_move_pct_1d IS NULL
                  AND catalyst_date IS NOT NULL
                ORDER BY catalyst_date DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
            cur.execute(
                "SELECT COUNT(*) FROM post_catalyst_outcomes WHERE actual_move_pct_1d IS NULL"
            )
            total_null = cur.fetchone()[0]
            # Histogram of catalyst_date age. catalyst_date is TEXT (ISO), cast
            # to date for the bucketing.
            cur.execute("""
                SELECT
                  CASE
                    WHEN catalyst_date::date >= CURRENT_DATE - INTERVAL '90 days' THEN '0-90d'
                    WHEN catalyst_date::date >= CURRENT_DATE - INTERVAL '1 year' THEN '90d-1y'
                    WHEN catalyst_date::date >= CURRENT_DATE - INTERVAL '3 years' THEN '1-3y'
                    WHEN catalyst_date::date >= CURRENT_DATE - INTERVAL '10 years' THEN '3-10y'
                    ELSE '10y+'
                  END AS age_bucket,
                  COUNT(*)
                FROM post_catalyst_outcomes
                WHERE actual_move_pct_1d IS NULL
                  AND catalyst_date IS NOT NULL
                GROUP BY age_bucket
                ORDER BY age_bucket
            """)
            by_age = [{"bucket": r[0], "n": r[1]} for r in cur.fetchall()]

        return {
            "total_null_actual_move_1d": total_null,
            "by_catalyst_date_age": by_age,
            "sample": [
                {
                    "id": r[0],
                    "ticker": r[1],
                    "catalyst_type": r[2],
                    "catalyst_date": str(r[3])[:10],
                    "outcome": r[4],
                    "last_error": r[5],
                    "backfill_attempts": r[6],
                }
                for r in rows
            ],
        }
    except Exception as e:
        logger.exception("inspect_stubs failed")
        raise HTTPException(500, f"inspect_stubs error: {e}")


@router.get("/post-catalyst/sector-coverage")
async def post_catalyst_sector_coverage():
    """Distribution of sector_basket values across post_catalyst_outcomes.
    Used to debug abnormal-returns recompute coverage.
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    COALESCE(sector_basket, 'NULL') AS basket,
                    COUNT(*) AS n,
                    SUM(CASE WHEN actual_move_pct_1d IS NOT NULL THEN 1 ELSE 0 END) AS with_actual_1d,
                    SUM(CASE WHEN abnormal_move_pct_1d IS NOT NULL THEN 1 ELSE 0 END) AS with_abnormal_1d,
                    SUM(CASE WHEN abnormal_move_pct_30d IS NOT NULL THEN 1 ELSE 0 END) AS with_abnormal_30d
                FROM post_catalyst_outcomes
                GROUP BY 1
                ORDER BY n DESC
            """)
            rows = cur.fetchall()
            cur.execute("SELECT COUNT(*) FROM post_catalyst_outcomes")
            total = cur.fetchone()[0]
        return {
            "total_outcomes": total,
            "by_sector_basket": [
                {
                    "basket": r[0],
                    "n": r[1],
                    "with_actual_1d": r[2],
                    "with_abnormal_1d": r[3],
                    "with_abnormal_30d": r[4],
                }
                for r in rows
            ],
        }
    except Exception as e:
        logger.exception("sector_coverage failed")
        raise HTTPException(500, f"sector_coverage error: {e}")


@router.post("/post-catalyst/retry-stubs-with-polygon")
async def retry_stub_rows_with_polygon(max_rows: int = 50):
    """Re-run backfill for post_catalyst_outcomes rows that have no
    actual_move_pct (yfinance failed at original backfill). Now that
    _fetch_price_window has Polygon as a fallback, those tickers may be
    fetchable via Polygon's historical aggs endpoint.

    Selects rows where actual_move_pct_1d IS NULL AND last_error IS NULL
    (i.e. never attempted, OR previously attempted and didn't get marked).
    Re-runs _fetch_price_window which now tries yfinance first, then Polygon.
    Updates the row in place if successful; on failure, increments
    backfill_attempts and sets last_error so the row doesn't re-qualify.

    Returns: {attempted, succeeded, still_failed, by_source, errors}
    """
    try:
        from services.post_catalyst_tracker import _fetch_price_window
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, ticker, catalyst_type, catalyst_date
                FROM post_catalyst_outcomes
                WHERE actual_move_pct_1d IS NULL
                  AND catalyst_date IS NOT NULL
                  AND (last_error IS NULL OR last_error NOT LIKE %s)
                ORDER BY catalyst_date DESC
                LIMIT %s
            """, ('no_data:%', max_rows,))
            rows = cur.fetchall()

        results = {"attempted": 0, "succeeded": 0, "still_failed": 0, "by_source": {"yfinance": 0, "polygon": 0}, "errors": []}
        for outcome_id, ticker, cat_type, cat_date in rows:
            results["attempted"] += 1
            try:
                cat_str = cat_date.strftime("%Y-%m-%d") if hasattr(cat_date, "strftime") else str(cat_date)[:10]
                pw = _fetch_price_window(ticker, cat_str)
                if not pw or not pw.get("_data_present"):
                    # Mark as no-data so this row doesn't re-qualify forever.
                    # Common cause: ticker delisted or catalyst_date outside
                    # both yfinance and Polygon archive coverage.
                    with db.get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute(
                            """UPDATE post_catalyst_outcomes
                               SET last_error = 'no_data: yfinance+polygon both empty',
                                   backfill_attempts = COALESCE(backfill_attempts, 0) + 1
                               WHERE id = %s""",
                            (outcome_id,),
                        )
                        conn.commit()
                    results["still_failed"] += 1
                    continue
                pre = pw["pre_event_price"]
                m1 = ((pw["day1_price"] - pre) / pre * 100.0) if (pw.get("day1_price") and pre) else None
                m7 = ((pw["day7_price"] - pre) / pre * 100.0) if (pw.get("day7_price") and pre) else None
                m30 = ((pw["day30_price"] - pre) / pre * 100.0) if (pw.get("day30_price") and pre) else None
                source = pw.get("_source", "unknown")
                results["by_source"][source] = results["by_source"].get(source, 0) + 1
                with db.get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE post_catalyst_outcomes
                        SET actual_move_pct_1d = %s,
                            actual_move_pct_7d = %s,
                            actual_move_pct_30d = %s,
                            preevent_avg_volume_30d = COALESCE(preevent_avg_volume_30d, %s),
                            postevent_volume_1d = COALESCE(postevent_volume_1d, %s),
                            postevent_max_intraday_move_pct = COALESCE(postevent_max_intraday_move_pct, %s)
                        WHERE id = %s
                    """, (
                        m1, m7, m30,
                        pw.get("preevent_avg_volume_30d"),
                        pw.get("postevent_volume_1d"),
                        pw.get("postevent_max_intraday_move_pct"),
                        outcome_id,
                    ))
                    conn.commit()
                results["succeeded"] += 1
            except Exception as e:
                if len(results["errors"]) < 10:
                    results["errors"].append({"id": outcome_id, "ticker": ticker, "error": str(e)[:80]})
        return results
    except Exception as e:
        logger.exception("retry_stubs_with_polygon failed")
        raise HTTPException(500, f"retry_stubs error: {e}")


@router.post("/post-catalyst/recompute-abnormal")
async def recompute_abnormal(max_rows: int = 100):
    """Backfill sector_move + abnormal_move columns on existing post_catalyst
    outcomes that don't yet have them. Per ChatGPT critique #4: 'Compare
    stock_move - XBI_move (sector basket), not raw stock move.'

    Uses XBI as the default basket (small-cap weighted, closer to our universe).
    Updates rows where sector_basket IS NULL (never attempted).

    On success: sector_basket='XBI', sector_move_pct_*, abnormal_move_pct_* set.
    On no-data (e.g. catalyst_date outside available XBI history):
      sector_basket='no_data' so the row is marked attempted and excluded
      from future runs. Otherwise the same 50 rows would re-qualify forever.
    """
    try:
        from services.post_catalyst_tracker import _compute_sector_moves
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, ticker, catalyst_type, catalyst_date,
                       actual_move_pct_1d, actual_move_pct_7d, actual_move_pct_30d
                FROM post_catalyst_outcomes
                WHERE catalyst_date IS NOT NULL
                  AND sector_basket IS NULL
                ORDER BY catalyst_date DESC
                LIMIT %s
            """, (max_rows,))
            rows = cur.fetchall()

        results = {"attempted": 0, "succeeded": 0, "no_sector_data": 0, "errors": []}
        for r in rows:
            outcome_id, ticker, cat_type, cat_date, m1, m7, m30 = r
            results["attempted"] += 1
            try:
                cat_str = cat_date.strftime("%Y-%m-%d") if hasattr(cat_date, "strftime") else str(cat_date)[:10]
                sector_moves = _compute_sector_moves(cat_str, basket="XBI")
                if not sector_moves:
                    # Mark as attempted-but-no-data so it doesn't re-qualify forever.
                    # Common cause: catalyst_date predates XBI inception or falls
                    # outside yfinance's available history window.
                    with db.get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute(
                            "UPDATE post_catalyst_outcomes SET sector_basket = 'no_data' WHERE id = %s",
                            (outcome_id,),
                        )
                        conn.commit()
                    results["no_sector_data"] += 1
                    continue
                s1 = sector_moves.get("day1_pct")
                s7 = sector_moves.get("day7_pct")
                s30 = sector_moves.get("day30_pct")
                ab1 = float(m1) - s1 if (m1 is not None and s1 is not None) else None
                ab7 = float(m7) - s7 if (m7 is not None and s7 is not None) else None
                ab30 = float(m30) - s30 if (m30 is not None and s30 is not None) else None
                with db.get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE post_catalyst_outcomes
                        SET sector_basket = 'XBI',
                            sector_move_pct_1d = %s,
                            sector_move_pct_7d = %s,
                            sector_move_pct_30d = %s,
                            abnormal_move_pct_1d = %s,
                            abnormal_move_pct_7d = %s,
                            abnormal_move_pct_30d = %s
                        WHERE id = %s
                    """, (s1, s7, s30, ab1, ab7, ab30, outcome_id))
                    conn.commit()
                results["succeeded"] += 1
            except Exception as e:
                if len(results["errors"]) < 10:
                    results["errors"].append({"id": outcome_id, "ticker": ticker, "error": str(e)[:80]})
        return results
    except Exception as e:
        logger.exception("recompute_abnormal failed")
        raise HTTPException(500, f"recompute_abnormal error: {e}")


@router.get("/post-catalyst/move-stats-abnormal")
async def move_stats_abnormal():
    """Aggregate accuracy stats keyed on ABNORMAL returns (stock - XBI).
    
    Compares against the existing /move-stats endpoint which uses raw moves
    so we can show the accuracy difference. Per ChatGPT critique: alpha
    accuracy is the right metric, not raw-return accuracy.
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            # Stats by catalyst type + outcome on abnormal returns
            cur.execute("""
                SELECT catalyst_type, outcome, count(*),
                       avg(abnormal_move_pct_1d)::numeric(10,2)  avg_ab_1d,
                       avg(abnormal_move_pct_30d)::numeric(10,2) avg_ab_30d,
                       avg(actual_move_pct_30d)::numeric(10,2)   avg_raw_30d,
                       avg(sector_move_pct_30d)::numeric(10,2)   avg_sector_30d
                FROM post_catalyst_outcomes
                WHERE outcome IN ('approved','rejected','positive','negative','beat','miss')
                  AND abnormal_move_pct_30d IS NOT NULL
                GROUP BY catalyst_type, outcome
                ORDER BY catalyst_type, outcome
            """)
            by_type_outcome = [
                {"catalyst_type": r[0], "outcome": r[1], "n": r[2],
                 "avg_abnormal_1d": float(r[3]) if r[3] else None,
                 "avg_abnormal_30d": float(r[4]) if r[4] else None,
                 "avg_raw_30d": float(r[5]) if r[5] else None,
                 "avg_sector_30d": float(r[6]) if r[6] else None}
                for r in cur.fetchall()
            ]
            # Coverage stats
            cur.execute("""
                SELECT count(*) total,
                       count(abnormal_move_pct_30d) with_abnormal,
                       count(sector_move_pct_30d) with_sector
                FROM post_catalyst_outcomes
            """)
            cov = cur.fetchone()
            return {
                "coverage": {
                    "total_outcomes": cov[0],
                    "with_abnormal_returns": cov[1],
                    "abnormal_coverage_pct": round(100.0 * cov[1] / cov[0], 1) if cov[0] else 0,
                },
                "by_catalyst_outcome": by_type_outcome,
                "interpretation": (
                    "abnormal = stock_move - sector_move (XBI). Positive abnormal "
                    "for 'approved' outcomes = real catalyst alpha. Compare to "
                    "avg_raw_30d to see how much was sector drift."
                ),
            }
    except Exception as e:
        logger.exception("move_stats_abnormal failed")
        raise HTTPException(500, f"move_stats_abnormal error: {e}")


@router.get("/sec/dilution-capacity")
async def sec_dilution_capacity(ticker: str, max_filings: int = 4):
    """Spot-check ATM/shelf/warrant extraction from SEC narrative filings.

    Per ChatGPT pass-3 critique #1: SEC XBRL extraction gives cash/debt/shares
    but not the dilution CAPACITY hidden in S-3, 424B5, 8-K narrative.
    This endpoint runs services/sec_dilution.py and returns structured facts:
    ATM facilities, shelf registrations, outstanding warrants, convertibles,
    recent issuances.

    Latency: ~10-20s (fetches up to N filings + LLM-extracts each).
    Result cached 12h in Redis.
    """
    try:
        from services.sec_dilution import fetch_dilution_capacity
        result = fetch_dilution_capacity(ticker.upper(), max_filings_to_parse=max_filings)
        if not result:
            raise HTTPException(404, "no SEC dilution data")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("sec_dilution_capacity failed")
        raise HTTPException(500, f"sec_dilution_capacity error: {e}")


# ────────────────────────────────────────────────────────────
# Manual catalyst override + ingestion health
# ────────────────────────────────────────────────────────────
# Per user proposal: "Build a simple admin form allowing manual catalyst
# entry (drug name, phase, date, estimated PoS, indication) that overrides
# auto-parsed data. Display parse health status on detail page."

class ManualCatalystPayload(BaseModel):
    ticker: str
    catalyst_type: str        # 'FDA Decision' | 'Phase 3 Readout' | etc.
    catalyst_date: str        # YYYY-MM-DD
    drug_name: Optional[str] = None
    indication: Optional[str] = None
    phase: Optional[str] = None
    description: Optional[str] = None
    probability: Optional[float] = None    # P(approval). Stored as confidence_score.
    source_url: Optional[str] = None       # if user has a press release / clinicaltrials link


@router.post("/catalysts/manual")
async def add_manual_catalyst(payload: ManualCatalystPayload):
    """Insert a manual catalyst override. The is_manual_override flag
    keeps the seeder from overwriting it on next refresh.

    The UNIQUE constraint on catalyst_universe is on canonical_drug_name
    (not drug_name) — different formattings of the same drug should
    upsert into the same row. We compute canonical_drug_name with the
    same function the seeder uses.

    Returns the inserted row ID.
    """
    try:
        from services.database import BiotechDatabase
        from services.universe_seeder import _canonicalize_drug
        canonical = _canonicalize_drug(payload.drug_name)
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            # The unique index is partial: WHERE canonical_drug_name IS NOT NULL
            # AND status='active'. So ON CONFLICT only fires when canonical
            # is non-null. For NULL canonical (company-level catalysts), we
            # check for an existing row first and either UPDATE or INSERT.
            if canonical:
                cur.execute("""
                    INSERT INTO catalyst_universe (
                        ticker, company_name, catalyst_type, catalyst_date,
                        description, drug_name, canonical_drug_name,
                        indication, phase,
                        source, source_url, confidence_score, verified, status,
                        is_manual_override, last_updated, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        'manual', %s, %s, TRUE, 'active',
                        TRUE, NOW(), NOW()
                    )
                    ON CONFLICT (ticker, catalyst_type, catalyst_date, canonical_drug_name)
                    WHERE canonical_drug_name IS NOT NULL AND status = 'active'
                    DO UPDATE SET
                        description = EXCLUDED.description,
                        drug_name = EXCLUDED.drug_name,
                        indication = EXCLUDED.indication,
                        phase = EXCLUDED.phase,
                        source_url = EXCLUDED.source_url,
                        confidence_score = EXCLUDED.confidence_score,
                        is_manual_override = TRUE,
                        source = 'manual',
                        verified = TRUE,
                        status = 'active',
                        last_updated = NOW()
                    RETURNING id
                """, (
                    payload.ticker.upper().strip(),
                    None,  # company_name will be filled by next seeder run
                    payload.catalyst_type, payload.catalyst_date,
                    payload.description, payload.drug_name, canonical,
                    payload.indication, payload.phase,
                    payload.source_url, payload.probability,
                ))
            else:
                # NULL canonical → no unique constraint. Look for existing
                # company-level row at this (ticker, type, date) and update
                # or insert manually.
                cur.execute("""
                    SELECT id FROM catalyst_universe
                    WHERE ticker = %s AND catalyst_type = %s
                      AND catalyst_date = %s AND canonical_drug_name IS NULL
                      AND status = 'active'
                    LIMIT 1
                """, (payload.ticker.upper().strip(), payload.catalyst_type,
                      payload.catalyst_date))
                existing = cur.fetchone()
                if existing:
                    cur.execute("""
                        UPDATE catalyst_universe
                        SET description = %s, indication = %s, phase = %s,
                            source_url = %s, confidence_score = %s,
                            is_manual_override = TRUE,
                            source = 'manual', verified = TRUE,
                            last_updated = NOW()
                        WHERE id = %s
                        RETURNING id
                    """, (payload.description, payload.indication, payload.phase,
                          payload.source_url, payload.probability,
                          existing[0]))
                else:
                    cur.execute("""
                        INSERT INTO catalyst_universe (
                            ticker, catalyst_type, catalyst_date, description,
                            drug_name, canonical_drug_name, indication, phase,
                            source, source_url, confidence_score, verified,
                            status, is_manual_override, last_updated, created_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, NULL, %s, %s,
                            'manual', %s, %s, TRUE, 'active', TRUE, NOW(), NOW()
                        )
                        RETURNING id
                    """, (
                        payload.ticker.upper().strip(),
                        payload.catalyst_type, payload.catalyst_date,
                        payload.description, payload.drug_name,
                        payload.indication, payload.phase,
                        payload.source_url, payload.probability,
                    ))
            new_id = cur.fetchone()[0]
            cur.execute("""
                INSERT INTO catalyst_ingestion_log
                  (ticker, source, status, catalysts_found)
                VALUES (%s, 'manual', 'success', 1)
            """, (payload.ticker.upper().strip(),))
            conn.commit()
        return {"id": new_id, "ticker": payload.ticker.upper().strip(),
                "is_manual_override": True}
    except Exception as e:
        logger.exception("add_manual_catalyst failed")
        raise HTTPException(500, f"add_manual_catalyst error: {e}")


@router.patch("/catalysts/{catalyst_id}")
async def edit_catalyst(catalyst_id: int, payload: ManualCatalystPayload):
    """Edit any catalyst row. Marks as is_manual_override=TRUE so the
    seeder won't revert the change."""
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE catalyst_universe
                SET catalyst_type = %s,
                    catalyst_date = %s,
                    description = COALESCE(%s, description),
                    drug_name = COALESCE(%s, drug_name),
                    indication = COALESCE(%s, indication),
                    phase = COALESCE(%s, phase),
                    confidence_score = COALESCE(%s, confidence_score),
                    source_url = COALESCE(%s, source_url),
                    is_manual_override = TRUE,
                    last_updated = NOW()
                WHERE id = %s
                RETURNING id, ticker, catalyst_type, catalyst_date
            """, (
                payload.catalyst_type, payload.catalyst_date,
                payload.description, payload.drug_name,
                payload.indication, payload.phase,
                payload.probability, payload.source_url,
                catalyst_id,
            ))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, f"catalyst id {catalyst_id} not found")
            conn.commit()
        return {"id": row[0], "ticker": row[1],
                "catalyst_type": row[2],
                "catalyst_date": str(row[3])[:10] if row[3] else None,
                "is_manual_override": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("edit_catalyst failed")
        raise HTTPException(500, f"edit_catalyst error: {e}")


@router.delete("/catalysts/{catalyst_id}")
async def soft_delete_catalyst(catalyst_id: int):
    """Soft-delete by setting status='invalid'. Preserves audit trail."""
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE catalyst_universe
                SET status = 'invalid',
                    last_updated = NOW()
                WHERE id = %s
                RETURNING id, ticker
            """, (catalyst_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, f"catalyst id {catalyst_id} not found")
            conn.commit()
        return {"id": row[0], "ticker": row[1], "status": "invalid"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("delete_catalyst failed")
        raise HTTPException(500, f"delete_catalyst error: {e}")


@router.get("/catalysts/{ticker}/ingestion-log")
async def catalyst_ingestion_log(ticker: str, limit: int = 20):
    """Show the last N ingestion attempts for a ticker. Lets the user
    see WHY parsing failed (if it did) and diagnose stale/missing data.
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT attempt_at, source, status, catalysts_found,
                       error_class, error_message, duration_ms
                FROM catalyst_ingestion_log
                WHERE ticker = %s
                ORDER BY attempt_at DESC
                LIMIT %s
            """, (ticker.upper().strip(), limit))
            rows = cur.fetchall()
        return {
            "ticker": ticker.upper().strip(),
            "attempts": [
                {
                    "attempt_at": str(r[0])[:19] if r[0] else None,
                    "source": r[1],
                    "status": r[2],
                    "catalysts_found": r[3],
                    "error_class": r[4],
                    "error_message": r[5],
                    "duration_ms": r[6],
                }
                for r in rows
            ],
        }
    except Exception as e:
        logger.exception("ingestion_log failed")
        raise HTTPException(500, f"ingestion_log error: {e}")


@router.get("/catalysts/ingestion-health")
async def catalyst_ingestion_health(hours: int = 24):
    """System-wide ingestion health over the last N hours. Returns
    success/failure breakdown by source so admin can spot a degraded
    parser before it impacts users.
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT source, status, COUNT(*) AS n
                FROM catalyst_ingestion_log
                WHERE attempt_at > NOW() - INTERVAL '%s hours'
                GROUP BY 1, 2
                ORDER BY 1, 2
            """, (hours,))
            rows = cur.fetchall()
            cur.execute("""
                SELECT COUNT(DISTINCT ticker)
                FROM catalyst_ingestion_log
                WHERE attempt_at > NOW() - INTERVAL '%s hours'
            """, (hours,))
            tickers_touched = cur.fetchone()[0]
        return {
            "window_hours": hours,
            "tickers_touched": tickers_touched,
            "by_source_status": [
                {"source": r[0], "status": r[1], "count": r[2]}
                for r in rows
            ],
        }
    except Exception as e:
        logger.exception("ingestion_health failed")
        raise HTTPException(500, f"ingestion_health error: {e}")


@router.post("/db/apply-migration-011")
async def apply_migration_011():
    """One-shot endpoint to run migration 011 on demand. Use when the
    Dockerfile's `alembic upgrade head` failed silently and the table
    doesn't exist. Idempotent — uses IF NOT EXISTS everywhere.

    Returns the new alembic version + list of tables affected.
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            # 1. Add is_manual_override column to catalyst_universe
            cur.execute("""
                ALTER TABLE catalyst_universe
                ADD COLUMN IF NOT EXISTS is_manual_override BOOLEAN DEFAULT FALSE
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_cu_manual
                ON catalyst_universe(is_manual_override)
                WHERE is_manual_override = TRUE
            """)
            # 2. catalyst_ingestion_log
            cur.execute("""
                CREATE TABLE IF NOT EXISTS catalyst_ingestion_log (
                    id SERIAL PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    attempt_at TIMESTAMPTZ DEFAULT NOW(),
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    catalysts_found INTEGER DEFAULT 0,
                    error_class TEXT,
                    error_message TEXT,
                    duration_ms INTEGER
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_cil_ticker_attempt
                ON catalyst_ingestion_log(ticker, attempt_at DESC)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_cil_status
                ON catalyst_ingestion_log(status, attempt_at DESC)
            """)
            # 3. Stamp the alembic version
            cur.execute("""
                UPDATE alembic_version_biotech
                SET version_num = '011_manual_override'
                WHERE version_num = '010_recanonicalize_dedup'
            """)
            stamped = cur.rowcount
            # 4. Verify
            cur.execute("SELECT version_num FROM alembic_version_biotech")
            new_v = cur.fetchone()[0]
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'catalyst_universe' AND column_name = 'is_manual_override'
            """)
            col_present = cur.fetchone() is not None
            cur.execute("""
                SELECT EXISTS (
                  SELECT 1 FROM information_schema.tables
                  WHERE table_name = 'catalyst_ingestion_log'
                )
            """)
            log_table = cur.fetchone()[0]
            conn.commit()
        return {
            "success": True,
            "new_alembic_version": new_v,
            "is_manual_override_column_present": col_present,
            "catalyst_ingestion_log_table_present": log_table,
            "stamped_rows": stamped,
        }
    except Exception as e:
        logger.exception("apply_migration_011 failed")
        raise HTTPException(500, f"apply_migration_011 error: {e}")


@router.post("/catalysts/{ticker}/refetch-now")
async def refetch_ticker_catalysts(ticker: str):
    """On-demand re-seed for a single ticker. Bypasses the scheduled
    seeder and triggers the LLM pipeline (Gemini → OpenAI → Anthropic+
    web_search) immediately.

    Honors is_manual_override — manual rows are NEVER touched. The
    seeder upsert's WHERE clause skips them.

    Returns: {ticker, attempted, catalysts_found, source, persisted: {...}}

    Use case: user sees a stale-data warning ('5 days ago · Gemini') on
    the freshness badge and wants to refresh without editing manually.
    """
    try:
        ticker = ticker.upper().strip()
        from services.universe_seeder import (
            extract_catalysts_for_ticker,
            write_catalysts_to_db,
        )
        from services.database import BiotechDatabase
        # Look up company name
        db = BiotechDatabase()
        company_name = ticker
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT company_name FROM catalyst_universe
                WHERE ticker = %s AND company_name IS NOT NULL
                LIMIT 1
            """, (ticker,))
            row = cur.fetchone()
            if row and row[0]:
                company_name = row[0]
        # Extract — this writes its own ingestion-log entry via the wrapper
        catalysts, meta = extract_catalysts_for_ticker(ticker, company_name)
        # Persist — write_catalysts_to_db already respects is_manual_override
        persisted = {"added": 0, "updated": 0, "skipped": 0, "errors": []}
        if catalysts:
            with db.get_conn() as conn:
                persisted = write_catalysts_to_db(catalysts, conn)
                conn.commit()
        return {
            "ticker": ticker,
            "company_name": company_name,
            "catalysts_found": len(catalysts),
            "source": meta.get("source"),
            "cost_usd": meta.get("cost_usd"),
            "error": meta.get("error"),
            "persisted": persisted,
        }
    except Exception as e:
        logger.exception("refetch_ticker_catalysts failed")
        raise HTTPException(500, f"refetch error: {e}")


@router.get("/db/alembic-preflight")
async def alembic_preflight():
    """Check alembic migrations for issues that could cause silent rollback.

    Most common pitfall: alembic_version_biotech.version_num is varchar(32).
    A revision ID >32 chars makes the post-migration version stamp fail,
    rolling back the entire migration. The Dockerfile's '|| echo WARN'
    swallows the error so the deploy looks healthy until you query the
    missing table. This endpoint surfaces those overflows before they bite.

    Run after adding any new migration file. Returns:
      current_version, all_revisions, max_id_length, overflows (≥32 chars)
    """
    try:
        import os
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT version_num FROM alembic_version_biotech LIMIT 1")
            row = cur.fetchone()
            current_version = row[0] if row else None
            # The varchar(32) limit
            cur.execute("""
                SELECT character_maximum_length FROM information_schema.columns
                WHERE table_name = 'alembic_version_biotech' AND column_name = 'version_num'
            """)
            limit_row = cur.fetchone()
            version_num_max = limit_row[0] if limit_row else None

        # Scan all migration files
        migrations_dir = os.path.join(os.path.dirname(__file__), "..", "alembic", "versions")
        revisions = []
        if os.path.isdir(migrations_dir):
            for fname in sorted(os.listdir(migrations_dir)):
                if not fname.endswith(".py"):
                    continue
                path = os.path.join(migrations_dir, fname)
                try:
                    with open(path) as f:
                        text = f.read()
                    # Parse revision = '...' line
                    import re
                    m = re.search(r"^revision\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
                    if m:
                        rev_id = m.group(1)
                        revisions.append({
                            "file": fname,
                            "revision": rev_id,
                            "length": len(rev_id),
                            "exceeds_limit": (
                                version_num_max is not None
                                and len(rev_id) > version_num_max
                            ),
                        })
                except Exception:
                    pass
        overflows = [r for r in revisions if r.get("exceeds_limit")]
        return {
            "current_version": current_version,
            "version_num_max_chars": version_num_max,
            "all_revisions": revisions,
            "max_id_length_seen": max((r["length"] for r in revisions), default=0),
            "overflows": overflows,
            "healthy": len(overflows) == 0,
        }
    except Exception as e:
        logger.exception("alembic_preflight failed")
        raise HTTPException(500, f"alembic_preflight error: {e}")


# ────────────────────────────────────────────────────────────
# Backtest abstention layer (migration 012)
# ────────────────────────────────────────────────────────────
# After ChatGPT critique that 52.5% direction accuracy across all 358
# catalysts is meaningless: 'chasing 70% across all events is overfitting,
# the right target is 70% on tradeable subset with abstention.'

@router.post("/post-catalyst/apply-migration-012")
async def apply_migration_012():
    """One-shot for migration 012 — abstention layer + 3-day actuals.
    Idempotent. Run if the Dockerfile-startup alembic upgrade swallowed
    the migration silently (same failure mode that bit 011)."""
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                ALTER TABLE post_catalyst_outcomes
                ADD COLUMN IF NOT EXISTS actual_move_pct_3d NUMERIC,
                ADD COLUMN IF NOT EXISTS abnormal_move_pct_3d NUMERIC,
                ADD COLUMN IF NOT EXISTS sector_move_pct_3d NUMERIC,
                ADD COLUMN IF NOT EXISTS day3_price NUMERIC,
                ADD COLUMN IF NOT EXISTS trade_signal TEXT,
                ADD COLUMN IF NOT EXISTS tradeable BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS error_abs_abnormal_3d_pct NUMERIC,
                ADD COLUMN IF NOT EXISTS direction_correct_3d BOOLEAN
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_pco_tradeable
                ON post_catalyst_outcomes(tradeable, trade_signal)
                WHERE tradeable = TRUE
            """)
            cur.execute("""
                UPDATE alembic_version_biotech
                SET version_num = '012_abstention_layer'
                WHERE version_num = '011_manual_override'
            """)
            stamped = cur.rowcount
            cur.execute("SELECT version_num FROM alembic_version_biotech")
            new_v = cur.fetchone()[0]
            conn.commit()
        return {"success": True, "new_alembic_version": new_v, "stamped_rows": stamped}
    except Exception as e:
        logger.exception("apply_migration_012 failed")
        raise HTTPException(500, f"apply_migration_012 error: {e}")


@router.post("/post-catalyst/backfill-3d-and-signals")
async def backfill_3d_and_signals(max_rows: int = 100, only_compute_signals: bool = False):
    """Two-phase backfill:

    Phase A (only_compute_signals=False, default):
      For up to max_rows post_catalyst_outcomes rows missing actual_move_pct_3d:
        1. Re-fetch the price window via _fetch_price_window (yfinance + Polygon
           fallback). Now also picks day3 + day3 sector basket.
        2. Compute actual_move_pct_3d, sector_move_pct_3d, abnormal_move_pct_3d.
        3. Persist day3_price, day3 moves, abnormal_3d, error metrics.

    Phase B (always runs after phase A):
      For ALL rows: classify trade_signal via services.catalyst_signal.
      Computes direction_correct_3d using sign(predicted) vs sign(abnormal_3d).
      Sets tradeable = (signal in LONG/SHORT).
      Sets error_abs_abnormal_3d_pct = |predicted - abnormal_3d| where data exists.

    Pass only_compute_signals=True to skip phase A (useful when you've
    already filled the 3d columns and just want to re-classify after
    tweaking signal thresholds).

    Returns: {phase_a_processed, phase_a_succeeded, phase_b_classified,
              tradeable_count, signal_distribution}
    """
    try:
        from services.database import BiotechDatabase
        from services.post_catalyst_tracker import (
            _fetch_price_window, _compute_sector_moves,
        )
        from services.catalyst_signal import classify_trade_signal, is_tradeable
        db = BiotechDatabase()
        results = {
            "phase_a_processed": 0,
            "phase_a_succeeded": 0,
            "phase_a_skipped_no_window": 0,
            "phase_b_classified": 0,
            "tradeable_count": 0,
            "signal_distribution": {},
            "errors": [],
        }

        # ── Phase A: backfill 3d actuals ──────────────────────────────
        if not only_compute_signals:
            with db.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT id, ticker, catalyst_date, pre_event_price
                    FROM post_catalyst_outcomes
                    WHERE actual_move_pct_3d IS NULL
                      AND catalyst_date IS NOT NULL
                      AND pre_event_price IS NOT NULL
                      AND (last_error IS NULL OR last_error NOT LIKE %s)
                    ORDER BY catalyst_date DESC
                    LIMIT %s
                """, ('no_data:%', max_rows))
                rows = cur.fetchall()

            for outcome_id, ticker, cat_date, pre_event_price in rows:
                results["phase_a_processed"] += 1
                try:
                    cat_str = cat_date.strftime("%Y-%m-%d") if hasattr(cat_date, "strftime") else str(cat_date)[:10]
                    pw = _fetch_price_window(ticker, cat_str)
                    if not pw or not pw.get("_data_present"):
                        results["phase_a_skipped_no_window"] += 1
                        continue
                    day3_price = pw.get("day3_price")
                    pre = float(pre_event_price)
                    move_3d = ((day3_price - pre) / pre * 100.0) if (day3_price and pre) else None
                    # Sector moves at 3d
                    sector_moves = _compute_sector_moves(cat_str, basket="XBI") or {}
                    sector_3d = sector_moves.get("day3_pct")
                    abnormal_3d = (
                        float(move_3d) - float(sector_3d)
                        if move_3d is not None and sector_3d is not None else None
                    )
                    with db.get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            UPDATE post_catalyst_outcomes
                            SET day3_price = %s,
                                actual_move_pct_3d = %s,
                                sector_move_pct_3d = %s,
                                abnormal_move_pct_3d = %s
                            WHERE id = %s
                        """, (day3_price, move_3d, sector_3d, abnormal_3d, outcome_id))
                        conn.commit()
                    results["phase_a_succeeded"] += 1
                except Exception as e:
                    if len(results["errors"]) < 10:
                        results["errors"].append({"id": outcome_id, "ticker": ticker, "phase": "A", "error": str(e)[:100]})

        # ── Phase B: classify signals (always runs) ──────────────────
        # Pull every row with a predicted_prob, classify via the new
        # probability-bias + scenario-magnitude logic. predicted_prob is
        # the catalyst's confidence_score at prediction time (used as
        # P(approval)). predicted_move_pct is the tiny EV that we
        # explicitly DON'T use as a directional signal — it collapses
        # 60/40 bets to ~0%, masking the model's actual conviction.
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT pco.id, pco.ticker, pco.catalyst_type, pco.catalyst_date,
                       pco.predicted_prob, pco.predicted_move_pct,
                       pco.abnormal_move_pct_3d,
                       cu.date_precision
                FROM post_catalyst_outcomes pco
                LEFT JOIN catalyst_universe cu
                  ON cu.ticker = pco.ticker
                 AND cu.catalyst_type = pco.catalyst_type
                 AND cu.catalyst_date::text = pco.catalyst_date::text
                 AND cu.status = 'active'
            """)
            rows = cur.fetchall()

        from services.catalyst_signal import (
            classify_trade_signal, is_tradeable,
            predicted_direction_from_probability,
        )
        sig_dist: Dict[str, int] = {}
        tradeable_n = 0
        for outcome_id, ticker, cat_type, cat_date, pred_prob, pred_ev, abnormal_3d, dprec in rows:
            try:
                # Direction signal driven by P(approval), not by EV.
                signal = classify_trade_signal(
                    probability=float(pred_prob) if pred_prob is not None else None,
                    catalyst_type=cat_type,
                    confidence_score=float(pred_prob) if pred_prob is not None else None,
                    date_precision=dprec,
                    options_implied_move_pct=None,  # historical rows lack live options
                )
                tradeable_flag = is_tradeable(signal)

                # Direction correctness on 3D abnormal target.
                # Predicted direction comes from probability bias (LONG if p>0.5,
                # SHORT if p<0.5). Actual direction comes from sign of 3D
                # abnormal return vs XBI, with a 3% deadband to avoid flipping
                # on near-zero noise.
                pred_dir = predicted_direction_from_probability(
                    float(pred_prob) if pred_prob is not None else None
                )
                dir_correct_3d = None
                err_3d = None
                if abnormal_3d is not None:
                    ab = float(abnormal_3d)
                    if pred_ev is not None:
                        err_3d = abs(ab - float(pred_ev))
                    if tradeable_flag and pred_dir is not None and abs(ab) >= 3.0:
                        actual_dir = 1 if ab > 0 else -1
                        dir_correct_3d = (pred_dir == actual_dir)
                with db.get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE post_catalyst_outcomes
                        SET trade_signal = %s,
                            tradeable = %s,
                            direction_correct_3d = %s,
                            error_abs_abnormal_3d_pct = %s
                        WHERE id = %s
                    """, (signal, tradeable_flag, dir_correct_3d, err_3d, outcome_id))
                    conn.commit()
                results["phase_b_classified"] += 1
                sig_dist[signal] = sig_dist.get(signal, 0) + 1
                if tradeable_flag:
                    tradeable_n += 1
            except Exception as e:
                if len(results["errors"]) < 10:
                    results["errors"].append({"id": outcome_id, "ticker": ticker, "phase": "B", "error": str(e)[:100]})

        results["signal_distribution"] = sig_dist
        results["tradeable_count"] = tradeable_n
        return results
    except Exception as e:
        logger.exception("backfill_3d_and_signals failed")
        raise HTTPException(500, f"backfill_3d_and_signals error: {e}")


@router.get("/post-catalyst/aggregate-v2")
async def post_catalyst_aggregate_v2():
    """Three-tier accuracy breakdown that addresses the user's pushback
    on the broken single 52.5% number:

      all_events      — the noise floor across 358 rows. Direction on raw 30D.
      tradeable_events — high-confidence subset (LONG/SHORT signals only).
                          Direction on 3D abnormal-vs-XBI. The number that matters.
      no_trade_events — subset by abstention reason. Lets you see WHY the
                          system is selective.

    A useful screener should target:
      tradeable direction ≥ 65-70%, coverage 25-40%.
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            # All-event metrics (legacy 30D-raw target)
            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE direction_correct) AS direction_hits,
                    AVG(error_abs_pct) AS avg_abs_error_raw_30d
                FROM post_catalyst_outcomes
                WHERE actual_move_pct_30d IS NOT NULL
                  AND predicted_move_pct IS NOT NULL
            """)
            r = cur.fetchone()
            total_all = r[0] or 0
            hits_all = r[1] or 0
            avg_err_30d = float(r[2]) if r[2] is not None else None

            # Tradeable-event metrics (3D abnormal-vs-XBI target)
            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE direction_correct_3d) AS direction_hits_3d,
                    AVG(error_abs_abnormal_3d_pct) AS avg_abs_error_abnormal_3d,
                    COUNT(*) FILTER (WHERE abnormal_move_pct_3d IS NOT NULL) AS with_3d_data
                FROM post_catalyst_outcomes
                WHERE tradeable = TRUE
            """)
            r = cur.fetchone()
            total_tradeable = r[0] or 0
            hits_tradeable = r[1] or 0
            avg_err_abnormal_3d = float(r[2]) if r[2] is not None else None
            with_3d = r[3] or 0

            # Signal distribution across all rows
            cur.execute("""
                SELECT trade_signal, COUNT(*) AS n
                FROM post_catalyst_outcomes
                WHERE trade_signal IS NOT NULL
                GROUP BY 1
                ORDER BY n DESC
            """)
            distribution = [{"signal": r[0], "count": r[1]} for r in cur.fetchall()]

        return {
            "all_events": {
                "count": total_all,
                "direction_hits": hits_all,
                "direction_accuracy_pct": (
                    round(100.0 * hits_all / total_all, 1) if total_all > 0 else None
                ),
                "avg_abs_error_pct": round(avg_err_30d, 1) if avg_err_30d is not None else None,
                "_target": "raw 30D move (noisy — sector + macro contaminated)",
            },
            "tradeable_events": {
                "count": total_tradeable,
                "direction_hits": hits_tradeable,
                "with_3d_data": with_3d,
                "direction_accuracy_pct": (
                    round(100.0 * hits_tradeable / total_tradeable, 1) if total_tradeable > 0 else None
                ),
                "coverage_pct": (
                    round(100.0 * total_tradeable / total_all, 1) if total_all > 0 else None
                ),
                "avg_abs_error_pct": round(avg_err_abnormal_3d, 1) if avg_err_abnormal_3d is not None else None,
                "_target": "3D abnormal return vs XBI (sector-adjusted)",
            },
            "signal_distribution": distribution,
            "interpretation": {
                "noise_floor": "all_events.direction_accuracy ~50% is biotech baseline",
                "actionable_target": "tradeable_events.direction_accuracy ≥ 65-70% with coverage 25-40%",
            },
        }
    except Exception as e:
        logger.exception("aggregate_v2 failed")
        raise HTTPException(500, f"aggregate_v2 error: {e}")


# ────────────────────────────────────────────────────────────
# V2 priced-in classifier (migration 013)
# ────────────────────────────────────────────────────────────
# After user observed tradeable accuracy = 31.7% (inverse 68.3%):
# 'old loose signal was anti-alpha. Build classifier around priced-in
# score, not just probability.'

@router.post("/post-catalyst/apply-migration-013")
async def apply_migration_013():
    """One-shot for migration 013 — priced-in features. Idempotent.
    Run if Dockerfile alembic startup didn't pick up 013."""
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                ALTER TABLE post_catalyst_outcomes
                ADD COLUMN IF NOT EXISTS price_30d_before_event NUMERIC,
                ADD COLUMN IF NOT EXISTS runup_pre_event_30d_pct NUMERIC,
                ADD COLUMN IF NOT EXISTS priced_in_score NUMERIC,
                ADD COLUMN IF NOT EXISTS signal_v2 TEXT,
                ADD COLUMN IF NOT EXISTS direction_correct_v2 BOOLEAN
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_pco_signal_v2
                ON post_catalyst_outcomes(signal_v2)
                WHERE signal_v2 IS NOT NULL
            """)
            cur.execute("""
                UPDATE alembic_version_biotech
                SET version_num = '013_priced_in_features'
                WHERE version_num = '012_abstention_layer'
            """)
            stamped = cur.rowcount
            cur.execute("SELECT version_num FROM alembic_version_biotech")
            new_v = cur.fetchone()[0]
            conn.commit()
        return {"success": True, "new_alembic_version": new_v, "stamped_rows": stamped}
    except Exception as e:
        logger.exception("apply_migration_013 failed")
        raise HTTPException(500, f"apply_migration_013 error: {e}")


@router.post("/post-catalyst/backfill-runup-and-classify-v2")
async def backfill_runup_and_classify_v2(max_rows: int = 100,
                                          only_compute_signals: bool = False):
    """Two-phase backfill for V2 priced-in classifier:

    Phase A: Re-fetch price windows for rows missing runup_pre_event_30d_pct.
             Computes runup from earliest pre-window close to pre_event_price.
             Persists price_30d_before_event + runup_pre_event_30d_pct.

    Phase B: For ALL rows with predicted_prob, classify via V2 logic.
             Stores signal_v2, priced_in_score, direction_correct_v2.

    Pass only_compute_signals=True to skip phase A. Useful for re-classifying
    after threshold tuning without re-fetching prices.

    Returns: {phase_a_processed, phase_a_succeeded, phase_b_classified,
              tradeable_v2, signal_v2_distribution}
    """
    try:
        from services.database import BiotechDatabase
        from services.post_catalyst_tracker import _fetch_price_window
        from services.catalyst_signal import (
            classify_trade_signal_v2, is_tradeable_v2,
            predicted_direction_v2,
        )
        db = BiotechDatabase()
        results = {
            "phase_a_processed": 0,
            "phase_a_succeeded": 0,
            "phase_a_skipped_no_window": 0,
            "phase_b_classified": 0,
            "tradeable_v2": 0,
            "signal_v2_distribution": {},
            "errors": [],
        }

        # ── Phase A: backfill runup_30d ──────────────────────────────
        if not only_compute_signals:
            with db.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT id, ticker, catalyst_date, pre_event_price
                    FROM post_catalyst_outcomes
                    WHERE runup_pre_event_30d_pct IS NULL
                      AND catalyst_date IS NOT NULL
                      AND pre_event_price IS NOT NULL
                    ORDER BY catalyst_date DESC
                    LIMIT %s
                """, (max_rows,))
                rows = cur.fetchall()

            for outcome_id, ticker, cat_date, pre_event_price in rows:
                results["phase_a_processed"] += 1
                try:
                    cat_str = cat_date.strftime("%Y-%m-%d") if hasattr(cat_date, "strftime") else str(cat_date)[:10]
                    pw = _fetch_price_window(ticker, cat_str)
                    if not pw or not pw.get("_data_present"):
                        results["phase_a_skipped_no_window"] += 1
                        continue
                    p30b = pw.get("price_30d_before_event")
                    pre = float(pre_event_price)
                    runup = ((pre - float(p30b)) / float(p30b) * 100.0) if (p30b and pre) else None
                    with db.get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            UPDATE post_catalyst_outcomes
                            SET price_30d_before_event = %s,
                                runup_pre_event_30d_pct = %s
                            WHERE id = %s
                        """, (p30b, runup, outcome_id))
                        conn.commit()
                    results["phase_a_succeeded"] += 1
                except Exception as e:
                    if len(results["errors"]) < 10:
                        results["errors"].append({"id": outcome_id, "ticker": ticker, "phase": "A", "error": str(e)[:100]})

        # ── Phase B: classify with V2 logic ──────────────────────────
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT pco.id, pco.ticker, pco.catalyst_type,
                       pco.predicted_prob, pco.runup_pre_event_30d_pct,
                       pco.abnormal_move_pct_3d,
                       cu.date_precision
                FROM post_catalyst_outcomes pco
                LEFT JOIN catalyst_universe cu
                  ON cu.ticker = pco.ticker
                 AND cu.catalyst_type = pco.catalyst_type
                 AND cu.catalyst_date::text = pco.catalyst_date::text
                 AND cu.status = 'active'
            """)
            rows = cur.fetchall()

        sig_dist: Dict[str, int] = {}
        tradeable_n = 0
        for outcome_id, ticker, cat_type, pred_prob, runup, abnormal_3d, dprec in rows:
            try:
                signal, priced_in = classify_trade_signal_v2(
                    probability=float(pred_prob) if pred_prob is not None else None,
                    runup_30d_pct=float(runup) if runup is not None else None,
                    catalyst_type=cat_type,
                    confidence_score=float(pred_prob) if pred_prob is not None else None,
                    date_precision=dprec,
                )
                tradeable_flag = is_tradeable_v2(signal)
                # Direction correctness on 3D abnormal target
                pred_dir = predicted_direction_v2(signal)
                dir_correct_v2 = None
                if abnormal_3d is not None and pred_dir is not None and tradeable_flag:
                    ab = float(abnormal_3d)
                    if abs(ab) >= 3.0:  # ±3% deadband on actuals
                        actual_dir = 1 if ab > 0 else -1
                        dir_correct_v2 = (pred_dir == actual_dir)
                with db.get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE post_catalyst_outcomes
                        SET signal_v2 = %s,
                            priced_in_score = %s,
                            direction_correct_v2 = %s
                        WHERE id = %s
                    """, (signal, priced_in, dir_correct_v2, outcome_id))
                    conn.commit()
                results["phase_b_classified"] += 1
                sig_dist[signal] = sig_dist.get(signal, 0) + 1
                if tradeable_flag:
                    tradeable_n += 1
            except Exception as e:
                if len(results["errors"]) < 10:
                    results["errors"].append({"id": outcome_id, "ticker": ticker, "phase": "B", "error": str(e)[:100]})

        results["signal_v2_distribution"] = sig_dist
        results["tradeable_v2"] = tradeable_n
        return results
    except Exception as e:
        logger.exception("backfill_runup_and_classify_v2 failed")
        raise HTTPException(500, f"backfill_runup_and_classify_v2 error: {e}")


@router.get("/post-catalyst/aggregate-v3")
async def post_catalyst_aggregate_v3(
    min_outcome_confidence: float = 0.0,
    catalyst_type: Optional[str] = None,
):
    """Multi-tier scoreboard with the corrected math + 95% Wilson CIs.

    Query params:
      - min_outcome_confidence (0.0-1.0, default 0.0): only count events whose
        outcome label confidence is at least this high. Use 0.7+ to exclude
        noisy LLM-derived labels and see "clean" accuracy.
      - catalyst_type: filter to a single catalyst type (e.g. "FDA Decision",
        "Phase 2 Readout"). Useful for per-type tuning.

    HISTORICAL CONTEXT: The original 31.7% / 'inverse 68.3%' analysis that
    motivated V2 was caused by a SQL denominator bug — direction_correct_3d
    is NULL on deadband rows (|abnormal_3d| < 3%), and including those NULLs
    in COUNT(*) deflated accuracy. After the fix, V1 is at 58.4% (modest
    edge, not anti-alpha). V2 adds a runup-based priced-in filter and lifts
    accuracy to 65.7% on a smaller subset.

    All numbers are IN-SAMPLE — the V2 thresholds were retuned against the
    same 459 rows we're scoring against. Out-of-sample validation pending.

    Wilson 95% CIs are reported alongside point estimates because at n=24-105
    judged rows, point estimates are not enough to claim production-grade
    accuracy. ChatGPT's critique was correct: the lower bound of V2's CI is
    ~56%, which means we cannot rule out coin-flip with statistical confidence.
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        # Build optional filter fragments. We ALWAYS pass them as separate
        # parameters to avoid SQL injection; empty string when unused.
        conf_filter = ""
        type_filter = ""
        params_extra = []
        if min_outcome_confidence > 0.0:
            conf_filter = " AND outcome_confidence >= %s "
            params_extra.append(min_outcome_confidence)
        if catalyst_type:
            type_filter = " AND catalyst_type = %s "
            params_extra.append(catalyst_type)
        with db.get_conn() as conn:
            cur = conn.cursor()
            # All-event metrics (legacy 30D-raw target)
            cur.execute(f"""
                SELECT COUNT(*),
                       COUNT(*) FILTER (WHERE direction_correct),
                       AVG(error_abs_pct)
                FROM post_catalyst_outcomes
                WHERE actual_move_pct_30d IS NOT NULL
                  AND predicted_move_pct IS NOT NULL
                  {conf_filter}
                  {type_filter}
            """, tuple(params_extra))
            r = cur.fetchone()
            total_all = r[0] or 0
            hits_all = r[1] or 0
            avg_err_30d = float(r[2]) if r[2] is not None else None

            # V1 tradeable (legacy LONG/SHORT) — denominator excludes deadband
            cur.execute(f"""
                SELECT
                    COUNT(*) FILTER (WHERE direction_correct_3d IS NOT NULL) AS judged,
                    COUNT(*) FILTER (WHERE direction_correct_3d) AS hits,
                    AVG(error_abs_abnormal_3d_pct) AS avg_err,
                    COUNT(*) AS total_in_bucket
                FROM post_catalyst_outcomes
                WHERE tradeable = TRUE
                  {conf_filter}
                  {type_filter}
            """, tuple(params_extra))
            r = cur.fetchone()
            v1_judged = r[0] or 0
            v1_total = r[3] or 0
            v1_hits = r[1] or 0
            v1_err = float(r[2]) if r[2] is not None else None

            # V2 tradeable
            cur.execute(f"""
                SELECT
                    COUNT(*) FILTER (WHERE direction_correct_v2 IS NOT NULL) AS judged,
                    COUNT(*) FILTER (WHERE direction_correct_v2) AS hits,
                    AVG(error_abs_abnormal_3d_pct) AS avg_err,
                    COUNT(*) AS total_in_bucket
                FROM post_catalyst_outcomes
                WHERE signal_v2 IN ('LONG_UNDERPRICED_POSITIVE',
                                    'SHORT_SELL_THE_NEWS',
                                    'SHORT_LOW_PROBABILITY',
                                    'LONG', 'SHORT')
                  {conf_filter}
                  {type_filter}
            """, tuple(params_extra))
            r = cur.fetchone()
            v2_judged = r[0] or 0
            v2_total = r[3] or 0
            v2_hits = r[1] or 0
            v2_err = float(r[2]) if r[2] is not None else None

            # Per-bucket V2 breakdown
            cur.execute(f"""
                SELECT signal_v2,
                       COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE direction_correct_v2 IS NOT NULL) AS judged,
                       COUNT(*) FILTER (WHERE direction_correct_v2 = TRUE) AS hits,
                       AVG(priced_in_score),
                       AVG(error_abs_abnormal_3d_pct)
                FROM post_catalyst_outcomes
                WHERE signal_v2 IS NOT NULL
                  {conf_filter}
                  {type_filter}
                GROUP BY signal_v2
                ORDER BY signal_v2
            """, tuple(params_extra))
            buckets = []
            for row in cur.fetchall():
                sig, total, judged, hits, avg_pi, avg_err = row
                acc = round(100.0 * hits / judged, 1) if judged > 0 else None
                ci = _wilson_ci_pct(hits, judged) if judged >= 5 else None
                # Mark buckets as 'production-ready' vs 'research only'
                # based on n>=50 AND CI lower bound > 55%
                production_ready = (
                    judged >= 50
                    and ci is not None
                    and ci.get("lower_pct") is not None
                    and ci["lower_pct"] > 55.0
                )
                buckets.append({
                    "signal": sig,
                    "count": total,
                    "judged": judged,
                    "hits": hits,
                    "deadband_excluded": total - judged,
                    "direction_accuracy_pct": acc,
                    "ci_95_pct": ci,  # {lower_pct, upper_pct} or None
                    "production_ready": production_ready,
                    "avg_priced_in_score": float(avg_pi) if avg_pi is not None else None,
                    "avg_abs_error_pct": float(avg_err) if avg_err is not None else None,
                })

        return {
            "all_events": {
                "count": total_all,
                "direction_hits": hits_all,
                "direction_accuracy_pct": (
                    round(100.0 * hits_all / total_all, 1) if total_all > 0 else None
                ),
                "ci_95_pct": _wilson_ci_pct(hits_all, total_all) if total_all >= 5 else None,
                "avg_abs_error_pct": round(avg_err_30d, 1) if avg_err_30d is not None else None,
                "_target": "raw 30D move (noisy — sector + macro contaminated)",
            },
            "tradeable_v1": {
                "count": v1_total,
                "judged": v1_judged,
                "deadband_excluded": v1_total - v1_judged,
                "direction_hits": v1_hits,
                "direction_accuracy_pct": (
                    round(100.0 * v1_hits / v1_judged, 1) if v1_judged > 0 else None
                ),
                "ci_95_pct": _wilson_ci_pct(v1_hits, v1_judged) if v1_judged >= 5 else None,
                "coverage_pct": (
                    round(100.0 * v1_total / total_all, 1) if total_all > 0 else None
                ),
                "avg_abs_error_pct": round(v1_err, 1) if v1_err is not None else None,
                "_target": "V1: probability-only classifier",
            },
            "tradeable_v2": {
                "count": v2_total,
                "judged": v2_judged,
                "deadband_excluded": v2_total - v2_judged,
                "direction_hits": v2_hits,
                "direction_accuracy_pct": (
                    round(100.0 * v2_hits / v2_judged, 1) if v2_judged > 0 else None
                ),
                "ci_95_pct": _wilson_ci_pct(v2_hits, v2_judged) if v2_judged >= 5 else None,
                "coverage_pct": (
                    round(100.0 * v2_total / total_all, 1) if total_all > 0 else None
                ),
                "avg_abs_error_pct": round(v2_err, 1) if v2_err is not None else None,
                "_target": "V2: priced-in-aware (uses 30d runup as priced-in proxy; thresholds 0.60/0.80 retuned in-sample)",
            },
            "v2_buckets": buckets,
            "interpretation": {
                "v2_methodology": "V1 uses probability bias (p>0.60 → LONG, p<0.40 → SHORT). V2 adds a runup-based priced-in filter: flat/clean setups (priced_in ≤ 0.60) → LONG_UNDERPRICED, strong-runup (priced_in ≥ 0.80) → SHORT_SELL_THE_NEWS, mid-runup (0.60-0.80) → NO_TRADE. This was retuned after empirical bucket data showed flat-runup events have ~78% V1 LONG accuracy and strong-runup events fade ~67%.",
                "denominator_note": "Direction accuracy excludes deadband rows (|abnormal_3d| < 3%) where there is no clear actual direction to score against. 'judged' is the denominator; 'count' is total in bucket including deadband.",
                "in_sample_warning": "V2 thresholds were tuned on the same 459 rows being scored. Numbers should be treated as in-sample diagnostics, not proven OOS performance. With n_judged in the 24-105 range, 95% CIs are wide; some buckets cannot be statistically distinguished from coin-flip yet.",
                "production_target": "tradeable hit rate ≥ 60% with CI lower bound > 55% AND n_judged ≥ 50 AND coverage 25-40%. SHORT_SELL_THE_NEWS does not currently meet n_judged threshold and is treated as research-only.",
            },
        }
    except Exception as e:
        logger.exception("aggregate_v3 failed")
        raise HTTPException(500, f"aggregate_v3 error: {e}")


def _wilson_ci_pct(hits: int, n: int, z: float = 1.96) -> Optional[Dict]:
    """Wilson score 95% confidence interval for a proportion.

    Returns {lower_pct, upper_pct, n, hits} or None if n is too small.
    Wilson is preferred over normal-approximation because it stays in
    [0,1] and behaves well for small n and extreme p.
    """
    if n < 1:
        return None
    p = hits / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    spread = (z / denom) * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return {
        "lower_pct": round(100.0 * lower, 1),
        "upper_pct": round(100.0 * upper, 1),
        "n": n,
        "hits": hits,
    }


@router.get("/post-catalyst/v1-vs-v2-same-row")
async def v1_vs_v2_same_row():
    """Same-row A/B: V1 vs V2 on the rows where BOTH classify as tradeable
    AND both have a clear actual direction (|abnormal_3d| >= 3%).

    Per ChatGPT's critique: comparing V1's 237-row subset to V2's 190-row
    subset is misleading because they're different row sets. The true V2
    lift is only meaningful on the intersection.

    Returns: common_judged, v1_hits, v2_hits, both_correct, both_wrong,
             v1_only_correct, v2_only_correct, lift_pct
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    COUNT(*) AS common_judged,
                    COUNT(*) FILTER (WHERE direction_correct_3d = TRUE) AS v1_hits,
                    COUNT(*) FILTER (WHERE direction_correct_v2 = TRUE) AS v2_hits,
                    COUNT(*) FILTER (
                        WHERE direction_correct_3d = TRUE
                          AND direction_correct_v2 = TRUE
                    ) AS both_correct,
                    COUNT(*) FILTER (
                        WHERE direction_correct_3d = FALSE
                          AND direction_correct_v2 = FALSE
                    ) AS both_wrong,
                    COUNT(*) FILTER (
                        WHERE direction_correct_3d = TRUE
                          AND direction_correct_v2 = FALSE
                    ) AS v1_only_correct,
                    COUNT(*) FILTER (
                        WHERE direction_correct_3d = FALSE
                          AND direction_correct_v2 = TRUE
                    ) AS v2_only_correct
                FROM post_catalyst_outcomes
                WHERE direction_correct_3d IS NOT NULL
                  AND direction_correct_v2 IS NOT NULL
            """)
            r = cur.fetchone()
            common = r[0] or 0
            v1_hits = r[1] or 0
            v2_hits = r[2] or 0
            both_correct = r[3] or 0
            both_wrong = r[4] or 0
            v1_only = r[5] or 0
            v2_only = r[6] or 0

        v1_acc = round(100.0 * v1_hits / common, 1) if common > 0 else None
        v2_acc = round(100.0 * v2_hits / common, 1) if common > 0 else None
        lift = round(v2_acc - v1_acc, 1) if (v1_acc is not None and v2_acc is not None) else None

        return {
            "common_judged": common,
            "v1": {
                "hits": v1_hits,
                "accuracy_pct": v1_acc,
                "ci_95_pct": _wilson_ci_pct(v1_hits, common) if common >= 5 else None,
            },
            "v2": {
                "hits": v2_hits,
                "accuracy_pct": v2_acc,
                "ci_95_pct": _wilson_ci_pct(v2_hits, common) if common >= 5 else None,
            },
            "agreement": {
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "v1_only_correct": v1_only,
                "v2_only_correct": v2_only,
            },
            "v2_lift_pp": lift,
            "interpretation": {
                "rule_of_thumb": "V2 is meaningfully better than V1 if v2_lift_pp ≥ 3 AND v2_only_correct > v1_only_correct AND CI ranges don't substantially overlap. If lift is small or negative, V2 is just being more selective, not more accurate.",
            },
        }
    except Exception as e:
        logger.exception("v1_vs_v2_same_row failed")
        raise HTTPException(500, f"v1_vs_v2_same_row error: {e}")
    except Exception as e:
        logger.exception("aggregate_v3 failed")
        raise HTTPException(500, f"aggregate_v3 error: {e}")


@router.get("/post-catalyst/runup-buckets")
async def runup_buckets():
    """Diagnostic per the user's request: bucket V1 LONG signals by
    runup_30d_pct and report direction accuracy in each bucket.

    Same denominator fix as aggregate-v3: 'judged' = rows where
    |abnormal_3d| ≥ 3% (deadband excluded). Without this fix, accuracy
    is artificially deflated because deadband rows are counted as
    'not hits' in a simple COUNT(*) FILTER division.

    Buckets:
      runup ≥ +20%    Stock has run, expect priced-in / sell-the-news
      runup +5..+20%  Mild runup
      runup -5..+5%   Flat
      runup ≤ -5%     Washed out, expect real long edge
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    CASE
                        WHEN runup_pre_event_30d_pct IS NULL THEN 'unknown'
                        WHEN runup_pre_event_30d_pct >= 20 THEN '4_strong_runup_>=20'
                        WHEN runup_pre_event_30d_pct >= 5  THEN '3_mild_runup_5_to_20'
                        WHEN runup_pre_event_30d_pct >= -5 THEN '2_flat_-5_to_5'
                        ELSE                                    '1_washed_out_<=-5'
                    END AS bucket,
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE direction_correct_3d IS NOT NULL) AS judged,
                    COUNT(*) FILTER (WHERE direction_correct_3d = TRUE) AS hits,
                    AVG(abnormal_move_pct_3d) AS avg_abnormal_3d,
                    AVG(runup_pre_event_30d_pct) AS avg_runup
                FROM post_catalyst_outcomes
                WHERE tradeable = TRUE
                  AND trade_signal IN ('LONG', 'SHORT')
                  AND abnormal_move_pct_3d IS NOT NULL
                GROUP BY bucket
                ORDER BY bucket
            """)
            rows = cur.fetchall()
        buckets = []
        for r in rows:
            bucket, total, judged, hits, avg_ab, avg_ru = r
            acc = round(100.0 * hits / judged, 1) if judged > 0 else None
            inv = round(100.0 - acc, 1) if acc is not None else None
            buckets.append({
                "bucket": bucket,
                "count": total,
                "judged": judged,
                "deadband_excluded": total - judged,
                "v1_hits": hits,
                "v1_direction_accuracy_pct": acc,
                "inverse_accuracy_pct": inv,
                "avg_abnormal_3d_pct": round(float(avg_ab), 1) if avg_ab is not None else None,
                "avg_runup_pct": round(float(avg_ru), 1) if avg_ru is not None else None,
            })
        return {
            "method": "Tradeable V1 signals (LONG/SHORT) bucketed by pre-event 30d runup. Direction scored against 3D abnormal-vs-XBI. 'judged' excludes deadband (|abnormal_3d| < 3%) rows where there's no clear actual direction.",
            "buckets": buckets,
            "interpretation": {
                "denominator_note": "v1_direction_accuracy_pct uses 'judged' as denominator. The user's original '31.7%' V1 accuracy used total-as-denominator which artificially deflates because deadband rows count as 'not-hits' rather than being excluded.",
            },
        }
    except Exception as e:
        logger.exception("runup_buckets failed")
        raise HTTPException(500, f"runup_buckets error: {e}")


# ────────────────────────────────────────────────────────────
# Sector-adjusted runup backfill (migration 014)
# ────────────────────────────────────────────────────────────

@router.post("/post-catalyst/apply-migration-014")
async def apply_migration_014():
    """One-shot for migration 014 — sector-adjusted runup. Idempotent."""
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                ALTER TABLE post_catalyst_outcomes
                ADD COLUMN IF NOT EXISTS sector_runup_30d_pct NUMERIC,
                ADD COLUMN IF NOT EXISTS runup_pre_event_30d_vs_xbi_pct NUMERIC
            """)
            cur.execute("""
                UPDATE alembic_version_biotech
                SET version_num = '014_sector_runup'
                WHERE version_num = '013_priced_in_features'
            """)
            stamped = cur.rowcount
            cur.execute("SELECT version_num FROM alembic_version_biotech")
            new_v = cur.fetchone()[0]
            conn.commit()
        return {"success": True, "new_alembic_version": new_v, "stamped_rows": stamped}
    except Exception as e:
        logger.exception("apply_migration_014 failed")
        raise HTTPException(500, f"apply_migration_014 error: {e}")


@router.post("/post-catalyst/backfill-sector-runup-and-reclassify")
async def backfill_sector_runup_and_reclassify(max_rows: int = 100,
                                                 only_compute_signals: bool = False):
    """Two-phase backfill for sector-adjusted runup:

    Phase A: Fetch XBI 30d-before close for rows missing
             sector_runup_30d_pct. Computes:
               sector_runup_30d_pct = XBI_30d_before → XBI_pre_event % move
               runup_30d_vs_xbi_pct = stock_runup - sector_runup
             Persists both. Skips rows where catalyst_date is missing or
             runup_pre_event_30d_pct is null (no stock baseline to subtract).

    Phase B: Re-classify ALL rows with V2 logic using sector-adjusted runup.
             Falls back to raw runup when sector data is missing.

    Pass only_compute_signals=True to skip phase A.

    Returns: {phase_a_*, phase_b_classified, signal_v2_distribution,
              with_sector_runup, fallback_to_raw_runup}
    """
    try:
        from services.database import BiotechDatabase
        from services.post_catalyst_tracker import _compute_sector_runup_30d
        from services.catalyst_signal import (
            classify_trade_signal_v2, is_tradeable_v2,
            predicted_direction_v2,
        )
        db = BiotechDatabase()
        results = {
            "phase_a_processed": 0,
            "phase_a_succeeded": 0,
            "phase_a_skipped_no_window": 0,
            "phase_b_classified": 0,
            "tradeable_v2": 0,
            "with_sector_runup": 0,
            "fallback_to_raw_runup": 0,
            "signal_v2_distribution": {},
            "errors": [],
        }

        # ── Phase A: backfill sector runup ──────────────────────────
        if not only_compute_signals:
            with db.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT id, ticker, catalyst_date, runup_pre_event_30d_pct
                    FROM post_catalyst_outcomes
                    WHERE sector_runup_30d_pct IS NULL
                      AND catalyst_date IS NOT NULL
                      AND runup_pre_event_30d_pct IS NOT NULL
                    ORDER BY catalyst_date DESC
                    LIMIT %s
                """, (max_rows,))
                rows = cur.fetchall()

            for outcome_id, ticker, cat_date, stock_runup in rows:
                results["phase_a_processed"] += 1
                try:
                    cat_str = cat_date.strftime("%Y-%m-%d") if hasattr(cat_date, "strftime") else str(cat_date)[:10]
                    sector_runup = _compute_sector_runup_30d(cat_str, basket="XBI")
                    if sector_runup is None:
                        results["phase_a_skipped_no_window"] += 1
                        continue
                    runup_vs_xbi = float(stock_runup) - float(sector_runup)
                    with db.get_conn() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            UPDATE post_catalyst_outcomes
                            SET sector_runup_30d_pct = %s,
                                runup_pre_event_30d_vs_xbi_pct = %s
                            WHERE id = %s
                        """, (sector_runup, runup_vs_xbi, outcome_id))
                        conn.commit()
                    results["phase_a_succeeded"] += 1
                except Exception as e:
                    if len(results["errors"]) < 10:
                        results["errors"].append({"id": outcome_id, "ticker": ticker, "phase": "A", "error": str(e)[:100]})

        # ── Phase B: re-classify with sector-adjusted runup ─────────
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT pco.id, pco.ticker, pco.catalyst_type,
                       pco.predicted_prob,
                       pco.runup_pre_event_30d_pct,
                       pco.runup_pre_event_30d_vs_xbi_pct,
                       pco.abnormal_move_pct_3d,
                       cu.date_precision
                FROM post_catalyst_outcomes pco
                LEFT JOIN catalyst_universe cu
                  ON cu.ticker = pco.ticker
                 AND cu.catalyst_type = pco.catalyst_type
                 AND cu.catalyst_date::text = pco.catalyst_date::text
                 AND cu.status = 'active'
            """)
            rows = cur.fetchall()

        sig_dist: Dict[str, int] = {}
        tradeable_n = 0
        for outcome_id, ticker, cat_type, pred_prob, runup_raw, runup_vs_xbi, abnormal_3d, dprec in rows:
            try:
                # Track which runup variant we're using
                if runup_vs_xbi is not None:
                    results["with_sector_runup"] += 1
                elif runup_raw is not None:
                    results["fallback_to_raw_runup"] += 1

                signal, priced_in = classify_trade_signal_v2(
                    probability=float(pred_prob) if pred_prob is not None else None,
                    runup_30d_pct=float(runup_raw) if runup_raw is not None else None,
                    runup_30d_vs_xbi_pct=float(runup_vs_xbi) if runup_vs_xbi is not None else None,
                    catalyst_type=cat_type,
                    confidence_score=float(pred_prob) if pred_prob is not None else None,
                    date_precision=dprec,
                )
                tradeable_flag = is_tradeable_v2(signal)
                pred_dir = predicted_direction_v2(signal)
                dir_correct_v2 = None
                if abnormal_3d is not None and pred_dir is not None and tradeable_flag:
                    ab = float(abnormal_3d)
                    if abs(ab) >= 3.0:
                        actual_dir = 1 if ab > 0 else -1
                        dir_correct_v2 = (pred_dir == actual_dir)
                with db.get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE post_catalyst_outcomes
                        SET signal_v2 = %s,
                            priced_in_score = %s,
                            direction_correct_v2 = %s
                        WHERE id = %s
                    """, (signal, priced_in, dir_correct_v2, outcome_id))
                    conn.commit()
                results["phase_b_classified"] += 1
                sig_dist[signal] = sig_dist.get(signal, 0) + 1
                if tradeable_flag:
                    tradeable_n += 1
            except Exception as e:
                if len(results["errors"]) < 10:
                    results["errors"].append({"id": outcome_id, "ticker": ticker, "phase": "B", "error": str(e)[:100]})

        results["signal_v2_distribution"] = sig_dist
        results["tradeable_v2"] = tradeable_n
        return results
    except Exception as e:
        logger.exception("backfill_sector_runup_and_reclassify failed")
        raise HTTPException(500, f"backfill_sector_runup_and_reclassify error: {e}")


# ────────────────────────────────────────────────────────────
# Precision-coverage curve (item 2 from next-phase plan)
# ────────────────────────────────────────────────────────────
# ChatGPT critique: "If 70% only happens at 5% coverage, it is probably
# not useful. You need a curve: coverage 10% → hit rate ?, coverage 20%
# → hit rate ?, etc."
#
# This sweeps min_confidence from 0.50 to 0.80 and reports for each
# threshold the resulting (coverage, hit rate, CI, n_judged). Lets us
# pick a threshold that actually lands in the 25-40% coverage band
# while preserving accuracy.

@router.get("/post-catalyst/precision-coverage-curve")
async def precision_coverage_curve():
    """Sweep min_confidence threshold and report (coverage, accuracy, CI)
    at each level. Uses V2 classifier with sector-adjusted runup when
    available.

    Reads existing rows; does NOT re-write signal_v2. This is read-only —
    a what-if curve, not a re-classify.

    Returns:
      total_events: total backtest rows
      points: [{min_confidence, n_tradeable, n_judged, coverage_pct,
                hits, accuracy_pct, ci_95_pct, classification_distribution}, ...]
      sweet_spot: heuristic recommendation (highest CI lower bound where
                  coverage stays >= 25%)
    """
    try:
        from services.database import BiotechDatabase
        from services.catalyst_signal import (
            classify_trade_signal_v2, is_tradeable_v2,
            predicted_direction_v2,
        )
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT pco.id, pco.predicted_prob, pco.catalyst_type,
                       pco.runup_pre_event_30d_pct,
                       pco.runup_pre_event_30d_vs_xbi_pct,
                       pco.abnormal_move_pct_3d,
                       cu.date_precision
                FROM post_catalyst_outcomes pco
                LEFT JOIN catalyst_universe cu
                  ON cu.ticker = pco.ticker
                 AND cu.catalyst_type = pco.catalyst_type
                 AND cu.catalyst_date::text = pco.catalyst_date::text
                 AND cu.status = 'active'
            """)
            rows = cur.fetchall()

        total_events = len(rows)
        if total_events == 0:
            return {"total_events": 0, "points": [], "sweet_spot": None}

        # Sweep thresholds
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        points = []

        for thresh in thresholds:
            n_tradeable = 0
            n_judged = 0
            hits = 0
            class_dist: Dict[str, int] = {}

            for r in rows:
                _, pred_prob, cat_type, runup_raw, runup_vs_xbi, abnormal_3d, dprec = r
                signal, _ = classify_trade_signal_v2(
                    probability=float(pred_prob) if pred_prob is not None else None,
                    runup_30d_pct=float(runup_raw) if runup_raw is not None else None,
                    runup_30d_vs_xbi_pct=float(runup_vs_xbi) if runup_vs_xbi is not None else None,
                    catalyst_type=cat_type,
                    confidence_score=float(pred_prob) if pred_prob is not None else None,
                    date_precision=dprec,
                    min_confidence=thresh,  # vary this
                )
                class_dist[signal] = class_dist.get(signal, 0) + 1
                if not is_tradeable_v2(signal):
                    continue
                n_tradeable += 1

                pred_dir = predicted_direction_v2(signal)
                if pred_dir is None or abnormal_3d is None:
                    continue
                ab = float(abnormal_3d)
                if abs(ab) < 3.0:  # deadband, NULL
                    continue
                n_judged += 1
                actual_dir = 1 if ab > 0 else -1
                if pred_dir == actual_dir:
                    hits += 1

            ci = _wilson_ci_pct(hits, n_judged) if n_judged >= 5 else None
            points.append({
                "min_confidence": thresh,
                "n_tradeable": n_tradeable,
                "n_judged": n_judged,
                "coverage_pct": round(100.0 * n_tradeable / total_events, 1),
                "hits": hits,
                "accuracy_pct": round(100.0 * hits / n_judged, 1) if n_judged > 0 else None,
                "ci_95_pct": ci,
                "classification_distribution": class_dist,
            })

        # Heuristic sweet spot: highest CI lower bound where:
        #   coverage_pct >= 25% AND n_judged >= 30 AND CI lower > 55
        eligible = [
            p for p in points
            if p["coverage_pct"] >= 25
               and p["n_judged"] >= 30
               and p["ci_95_pct"]
               and p["ci_95_pct"]["lower_pct"] > 55.0
        ]
        sweet_spot = max(eligible, key=lambda p: p["ci_95_pct"]["lower_pct"]) if eligible else None

        return {
            "total_events": total_events,
            "current_default_threshold": 0.55,
            "points": points,
            "sweet_spot": {
                "min_confidence": sweet_spot["min_confidence"],
                "coverage_pct": sweet_spot["coverage_pct"],
                "accuracy_pct": sweet_spot["accuracy_pct"],
                "ci_95_pct": sweet_spot["ci_95_pct"],
                "n_judged": sweet_spot["n_judged"],
                "rationale": "Highest CI lower bound among thresholds where coverage ≥ 25%, n_judged ≥ 30, and CI lower > 55%.",
            } if sweet_spot else None,
            "interpretation": {
                "method": "What-if sweep over min_confidence. For each value, classifies all backtest rows with V2, counts tradeable, scores direction on rows with |abnormal_3d| ≥ 3%.",
                "caveat": "All numbers are in-sample. Tightening the threshold reduces coverage but may not improve accuracy uniformly — the precision-coverage curve shows whether tightening helps or just shrinks sample.",
            },
        }
    except Exception as e:
        logger.exception("precision_coverage_curve failed")
        raise HTTPException(500, f"precision_coverage_curve error: {e}")


# ────────────────────────────────────────────────────────────
# Forward prediction snapshots (item 1)
# ────────────────────────────────────────────────────────────

@router.post("/post-catalyst/apply-migration-015")
async def apply_migration_015():
    """One-shot for migration 015 — prediction_snapshots table. Idempotent."""
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prediction_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    catalyst_id INTEGER,
                    catalyst_date DATE,
                    catalyst_type TEXT,
                    catalyst_outcome_id BIGINT,
                    prediction_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                    signal_version TEXT NOT NULL DEFAULT 'v2',
                    feature_version TEXT NOT NULL DEFAULT 'v1',
                    model_version TEXT,
                    signal TEXT NOT NULL,
                    predicted_prob NUMERIC,
                    priced_in_score NUMERIC,
                    predicted_direction INTEGER,
                    full_features_json JSONB,
                    evaluated_at TIMESTAMP WITH TIME ZONE,
                    actual_abnormal_3d_pct NUMERIC,
                    actual_dir_3d_vs_xbi INTEGER,
                    direction_correct BOOLEAN,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
                )
            """)
            for stmt in [
                "CREATE INDEX IF NOT EXISTS idx_pred_snap_prediction_time ON prediction_snapshots(prediction_time DESC)",
                "CREATE INDEX IF NOT EXISTS idx_pred_snap_outcome_id ON prediction_snapshots(catalyst_outcome_id) WHERE catalyst_outcome_id IS NOT NULL",
                "CREATE INDEX IF NOT EXISTS idx_pred_snap_signal_version_signal ON prediction_snapshots(signal_version, signal)",
                "CREATE INDEX IF NOT EXISTS idx_pred_snap_pending ON prediction_snapshots(catalyst_date) WHERE evaluated_at IS NULL",
                "CREATE INDEX IF NOT EXISTS idx_pred_snap_ticker_date ON prediction_snapshots(ticker, catalyst_date DESC)",
            ]:
                cur.execute(stmt)
            cur.execute("""
                UPDATE alembic_version_biotech
                SET version_num = '015_snapshot_table'
                WHERE version_num = '014_sector_runup'
            """)
            stamped = cur.rowcount
            cur.execute("SELECT version_num FROM alembic_version_biotech")
            new_v = cur.fetchone()[0]
            conn.commit()
        return {"success": True, "new_alembic_version": new_v, "stamped_rows": stamped}
    except Exception as e:
        logger.exception("apply_migration_015 failed")
        raise HTTPException(500, f"apply_migration_015 error: {e}")


@router.post("/post-catalyst/snapshot-current-classifications")
async def snapshot_current_classifications(only_new: bool = True):
    """One-shot: write a snapshot for every classified row in
    post_catalyst_outcomes. Captures the current V2 state as a baseline
    so we can diff future snapshots against it.

    NOTE: These snapshots are flagged with feature_version='v1_runup_vs_xbi'
    and model_version='v2_thresh_0.60_0.80_sector_adj' (current). These are
    NOT true OOS predictions — they were classified after outcomes were
    known. The OOS aggregate endpoint should filter by prediction_time >
    a cutoff date if pure OOS evaluation is desired.

    only_new: skip outcomes that already have a snapshot for the current
    feature_version + model_version (idempotent).
    """
    try:
        from services.database import BiotechDatabase
        from services.prediction_snapshots import (
            write_snapshot, ACTIVE_MODEL_VERSION, ACTIVE_FEATURE_VERSION,
        )
        from services.catalyst_signal import predicted_direction_v2
        db = BiotechDatabase()

        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT pco.id, pco.ticker, pco.catalyst_type, pco.catalyst_date,
                       pco.signal_v2, pco.predicted_prob, pco.priced_in_score,
                       pco.runup_pre_event_30d_pct,
                       pco.runup_pre_event_30d_vs_xbi_pct,
                       pco.sector_runup_30d_pct,
                       pco.options_implied_move_pct,
                       pco.abnormal_move_pct_3d,
                       cu.id, cu.date_precision
                FROM post_catalyst_outcomes pco
                LEFT JOIN catalyst_universe cu
                  ON cu.ticker = pco.ticker
                 AND cu.catalyst_type = pco.catalyst_type
                 AND cu.catalyst_date::text = pco.catalyst_date::text
                 AND cu.status = 'active'
                WHERE pco.signal_v2 IS NOT NULL
                {"AND NOT EXISTS (SELECT 1 FROM prediction_snapshots ps WHERE ps.catalyst_outcome_id = pco.id AND ps.feature_version = %s AND ps.model_version = %s)" if only_new else ""}
            """, (ACTIVE_FEATURE_VERSION, ACTIVE_MODEL_VERSION) if only_new else None)
            rows = cur.fetchall()

        results = {"scanned": 0, "written": 0, "errors": 0,
                   "active_model_version": ACTIVE_MODEL_VERSION,
                   "active_feature_version": ACTIVE_FEATURE_VERSION}

        for row in rows:
            (pco_id, ticker, cat_type, cat_date, signal, pred_prob,
             priced_in, runup_raw, runup_vs_xbi, sector_runup,
             opt_implied, abnormal_3d, cu_id, dprec) = row
            results["scanned"] += 1
            try:
                # Build features snapshot
                features = {
                    "predicted_prob": float(pred_prob) if pred_prob is not None else None,
                    "runup_30d_pct": float(runup_raw) if runup_raw is not None else None,
                    "runup_30d_vs_xbi_pct": float(runup_vs_xbi) if runup_vs_xbi is not None else None,
                    "sector_runup_30d_pct": float(sector_runup) if sector_runup is not None else None,
                    "options_implied_move_pct": float(opt_implied) if opt_implied is not None else None,
                    "date_precision": dprec,
                    "catalyst_type": cat_type,
                }
                pred_dir = predicted_direction_v2(signal)
                snap_id = write_snapshot(
                    db=db,
                    ticker=ticker,
                    catalyst_id=cu_id,
                    catalyst_date=cat_date.isoformat() if hasattr(cat_date, "isoformat") else str(cat_date)[:10] if cat_date else None,
                    catalyst_type=cat_type,
                    catalyst_outcome_id=pco_id,
                    signal=signal,
                    predicted_prob=float(pred_prob) if pred_prob is not None else None,
                    priced_in_score=float(priced_in) if priced_in is not None else None,
                    predicted_direction=pred_dir,
                    full_features=features,
                )
                if snap_id:
                    results["written"] += 1
                    # Immediately evaluate if outcome is known
                    if abnormal_3d is not None:
                        from services.prediction_snapshots import evaluate_snapshot
                        evaluate_snapshot(
                            db=db, snapshot_id=snap_id,
                            actual_abnormal_3d_pct=float(abnormal_3d),
                        )
                else:
                    results["errors"] += 1
            except Exception as e:
                results["errors"] += 1
                logger.exception(f"snapshot write failed for {ticker}: {e}")

        return results
    except Exception as e:
        logger.exception("snapshot_current_classifications failed")
        raise HTTPException(500, f"snapshot_current_classifications error: {e}")


@router.post("/post-catalyst/evaluate-pending-snapshots")
async def evaluate_pending_snapshots_endpoint(max_rows: int = 500):
    """For snapshots whose catalyst_outcome_id has abnormal_move_pct_3d
    populated, fill in evaluation fields. Designed to be called from the
    nightly cron."""
    try:
        from services.database import BiotechDatabase
        from services.prediction_snapshots import evaluate_pending_snapshots
        db = BiotechDatabase()
        return evaluate_pending_snapshots(db=db, max_rows=max_rows)
    except Exception as e:
        logger.exception("evaluate_pending_snapshots failed")
        raise HTTPException(500, f"evaluate_pending_snapshots error: {e}")


@router.get("/post-catalyst/oos-aggregate")
async def oos_aggregate(
    signal_version: str = "v2",
    pure_oos_only: bool = False,
):
    """OOS aggregate: prediction snapshots vs actual outcomes.

    pure_oos_only=True filters to snapshots whose prediction_time is >=
    the cutoff date (the date OOS data collection started). Anything
    snapshotted before that is in-sample and excluded.

    For now there's no cutoff — the table is fresh. As snapshots accumulate
    over weeks, this filter becomes meaningful.
    """
    try:
        from services.database import BiotechDatabase
        from services.prediction_snapshots import aggregate_oos
        db = BiotechDatabase()
        out = aggregate_oos(db=db, signal_version=signal_version)
        # Add Wilson CIs to top-level + buckets using the existing helper
        if out.get("judged", 0) >= 5:
            out["ci_95_pct"] = _wilson_ci_pct(out["hits"], out["judged"])
        for b in out.get("buckets", []):
            if b.get("judged", 0) >= 5:
                b["ci_95_pct"] = _wilson_ci_pct(b["hits"], b["judged"])
            else:
                b["ci_95_pct"] = None
        return out
    except Exception as e:
        logger.exception("oos_aggregate failed")
        raise HTTPException(500, f"oos_aggregate error: {e}")


# ────────────────────────────────────────────────────────────
# Per-catalyst-type accuracy breakdown (item 2)
# ────────────────────────────────────────────────────────────

@router.get("/post-catalyst/aggregate-by-catalyst-type")
async def aggregate_by_catalyst_type():
    """Per-catalyst-type V2 accuracy. With 459 rows we can probably
    distinguish FDA Decision (~80) from Phase 3 (~40) from AdCom (~15).
    Even at small samples it tells us *where* V2 actually works.

    Returns rows for each catalyst_type with V1 + V2 accuracy + CIs.
    Production-ready flag per type same as aggregate-v3 (n_judged ≥ 30
    AND CI lower > 55%).
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    catalyst_type,
                    COUNT(*) AS total,
                    -- V1 metrics
                    COUNT(*) FILTER (WHERE tradeable = TRUE) AS v1_tradeable,
                    COUNT(*) FILTER (WHERE tradeable = TRUE AND direction_correct_3d IS NOT NULL) AS v1_judged,
                    COUNT(*) FILTER (WHERE tradeable = TRUE AND direction_correct_3d = TRUE) AS v1_hits,
                    -- V2 metrics
                    COUNT(*) FILTER (
                        WHERE signal_v2 IN ('LONG_UNDERPRICED_POSITIVE',
                                            'SHORT_SELL_THE_NEWS',
                                            'SHORT_LOW_PROBABILITY',
                                            'LONG', 'SHORT')
                    ) AS v2_tradeable,
                    COUNT(*) FILTER (
                        WHERE signal_v2 IN ('LONG_UNDERPRICED_POSITIVE',
                                            'SHORT_SELL_THE_NEWS',
                                            'SHORT_LOW_PROBABILITY',
                                            'LONG', 'SHORT')
                          AND direction_correct_v2 IS NOT NULL
                    ) AS v2_judged,
                    COUNT(*) FILTER (
                        WHERE signal_v2 IN ('LONG_UNDERPRICED_POSITIVE',
                                            'SHORT_SELL_THE_NEWS',
                                            'SHORT_LOW_PROBABILITY',
                                            'LONG', 'SHORT')
                          AND direction_correct_v2 = TRUE
                    ) AS v2_hits
                FROM post_catalyst_outcomes
                WHERE catalyst_type IS NOT NULL
                GROUP BY catalyst_type
                ORDER BY total DESC
            """)
            rows = cur.fetchall()

        types = []
        for r in rows:
            (cat_type, total, v1_trade, v1_judged, v1_hits,
             v2_trade, v2_judged, v2_hits) = r

            v1_acc = round(100.0 * v1_hits / v1_judged, 1) if v1_judged > 0 else None
            v2_acc = round(100.0 * v2_hits / v2_judged, 1) if v2_judged > 0 else None
            v1_ci = _wilson_ci_pct(v1_hits, v1_judged) if v1_judged >= 5 else None
            v2_ci = _wilson_ci_pct(v2_hits, v2_judged) if v2_judged >= 5 else None

            v2_production_ready = (
                (v2_judged or 0) >= 30
                and v2_ci is not None
                and v2_ci.get("lower_pct") is not None
                and v2_ci["lower_pct"] > 55.0
            )

            types.append({
                "catalyst_type": cat_type,
                "total_events": total,
                "v1": {
                    "tradeable": v1_trade,
                    "judged": v1_judged,
                    "hits": v1_hits,
                    "accuracy_pct": v1_acc,
                    "ci_95_pct": v1_ci,
                },
                "v2": {
                    "tradeable": v2_trade,
                    "judged": v2_judged,
                    "hits": v2_hits,
                    "accuracy_pct": v2_acc,
                    "ci_95_pct": v2_ci,
                    "production_ready": v2_production_ready,
                },
                "v2_lift_pp": (
                    round(v2_acc - v1_acc, 1)
                    if (v1_acc is not None and v2_acc is not None) else None
                ),
            })

        return {
            "types": types,
            "interpretation": {
                "use_case": "Identifies which catalyst types V2 actually works on. With small samples per type (15-80), CIs are wider than aggregate. A type with v2_production_ready=true is a real candidate for live trading; types without it are research-only regardless of point accuracy.",
                "production_gate": "n_judged ≥ 30 AND CI lower > 55%",
            },
        }
    except Exception as e:
        logger.exception("aggregate_by_catalyst_type failed")
        raise HTTPException(500, f"aggregate_by_catalyst_type error: {e}")


# ────────────────────────────────────────────────────────────
# V2 reclassify + snapshot evaluation scheduler (item 4)
# ────────────────────────────────────────────────────────────
# Runs nightly to:
#   1. Re-classify outcomes whose abnormal_3d data has landed since last run
#      (only_compute_signals=true; no price re-fetch needed)
#   2. Evaluate any pending snapshots whose outcomes are now scored
#   3. Write fresh snapshots for newly classified rows (idempotent)
# Enabled via env var V2_RECLASSIFY_SCHEDULER_ENABLED=1.

_v2_reclassify_scheduler_state: dict = {
    "started": False,
    "last_run_at": None,
    "last_result": None,
    "runs_total": 0,
}


def _start_v2_reclassify_scheduler_once() -> None:
    """Start a background asyncio task that runs V2 reclassify + snapshot
    evaluation on a schedule. Idempotent — multiple calls are no-ops.
    Only runs if env var V2_RECLASSIFY_SCHEDULER_ENABLED=1."""
    if _v2_reclassify_scheduler_state.get("started"):
        return
    if os.getenv("V2_RECLASSIFY_SCHEDULER_ENABLED", "0") != "1":
        return
    _v2_reclassify_scheduler_state["started"] = True

    import asyncio
    interval_hours = float(os.getenv("V2_RECLASSIFY_SCHEDULER_HOURS", "24"))

    async def _loop():
        await asyncio.sleep(120)  # Wait 2min on startup
        from datetime import datetime as _dt
        from services.database import BiotechDatabase
        from services.prediction_snapshots import (
            evaluate_pending_snapshots,
        )
        while True:
            try:
                logger.info("[v2-reclassify-scheduler] running")
                results: dict = {}
                db = BiotechDatabase()

                # Step 1: Re-classify all rows with V2 (only_compute_signals,
                # no price fetches). This is fast (~10s for 459 rows) and
                # picks up any new outcomes that landed.
                from services.catalyst_signal import (
                    classify_trade_signal_v2, is_tradeable_v2,
                    predicted_direction_v2,
                )
                with db.get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT pco.id, pco.predicted_prob, pco.catalyst_type,
                               pco.runup_pre_event_30d_pct,
                               pco.runup_pre_event_30d_vs_xbi_pct,
                               pco.abnormal_move_pct_3d,
                               cu.date_precision
                        FROM post_catalyst_outcomes pco
                        LEFT JOIN catalyst_universe cu
                          ON cu.ticker = pco.ticker
                         AND cu.catalyst_type = pco.catalyst_type
                         AND cu.catalyst_date::text = pco.catalyst_date::text
                         AND cu.status = 'active'
                    """)
                    rows = cur.fetchall()

                reclass_count = 0
                for r in rows:
                    pco_id, pred_prob, cat_type, runup_raw, runup_vs_xbi, abnormal_3d, dprec = r
                    try:
                        signal, priced_in = classify_trade_signal_v2(
                            probability=float(pred_prob) if pred_prob is not None else None,
                            runup_30d_pct=float(runup_raw) if runup_raw is not None else None,
                            runup_30d_vs_xbi_pct=float(runup_vs_xbi) if runup_vs_xbi is not None else None,
                            catalyst_type=cat_type,
                            confidence_score=float(pred_prob) if pred_prob is not None else None,
                            date_precision=dprec,
                        )
                        pred_dir = predicted_direction_v2(signal)
                        dir_correct_v2 = None
                        if abnormal_3d is not None and pred_dir is not None and is_tradeable_v2(signal):
                            ab = float(abnormal_3d)
                            if abs(ab) >= 3.0:
                                actual_dir = 1 if ab > 0 else -1
                                dir_correct_v2 = (pred_dir == actual_dir)
                        with db.get_conn() as conn:
                            cur = conn.cursor()
                            cur.execute("""
                                UPDATE post_catalyst_outcomes
                                SET signal_v2 = %s,
                                    priced_in_score = %s,
                                    direction_correct_v2 = %s
                                WHERE id = %s
                            """, (signal, priced_in, dir_correct_v2, pco_id))
                            conn.commit()
                        reclass_count += 1
                    except Exception:
                        pass
                results["reclassified"] = reclass_count

                # Step 2: Evaluate pending snapshots whose outcomes are now scored
                eval_results = evaluate_pending_snapshots(db=db, max_rows=500)
                results["snapshot_evaluation"] = eval_results

                _v2_reclassify_scheduler_state["last_run_at"] = _dt.utcnow().isoformat()
                _v2_reclassify_scheduler_state["last_result"] = results
                _v2_reclassify_scheduler_state["runs_total"] += 1
                logger.info(f"[v2-reclassify-scheduler] done: {results}")
            except Exception as e:
                logger.warning(f"[v2-reclassify-scheduler] error: {e}")
                _v2_reclassify_scheduler_state["last_result"] = {"error": str(e)[:200]}
            await asyncio.sleep(interval_hours * 3600)

    try:
        loop = asyncio.get_event_loop()
        loop.create_task(_loop())
        logger.info(f"[v2-reclassify-scheduler] STARTED interval={interval_hours}h")
    except Exception as e:
        logger.warning(f"[v2-reclassify-scheduler] failed to start: {e}")
        _v2_reclassify_scheduler_state["started"] = False


@router.get("/post-catalyst/v2-reclassify-scheduler-status")
async def v2_reclassify_scheduler_status():
    """Status of the V2 reclassify + snapshot eval scheduler.
    Set env var V2_RECLASSIFY_SCHEDULER_ENABLED=1 to enable.
    Tunable via V2_RECLASSIFY_SCHEDULER_HOURS (default 24)."""
    return {
        "enabled": os.getenv("V2_RECLASSIFY_SCHEDULER_ENABLED", "0") == "1",
        "interval_hours": float(os.getenv("V2_RECLASSIFY_SCHEDULER_HOURS", "24")),
        **_v2_reclassify_scheduler_state,
    }


@router.post("/post-catalyst/v2-reclassify-trigger-now")
async def v2_reclassify_trigger_now():
    """Manually trigger one round of V2 reclassify + snapshot evaluation."""
    try:
        from services.database import BiotechDatabase
        from services.prediction_snapshots import evaluate_pending_snapshots
        from services.catalyst_signal import (
            classify_trade_signal_v2, is_tradeable_v2,
            predicted_direction_v2,
        )
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT pco.id, pco.predicted_prob, pco.catalyst_type,
                       pco.runup_pre_event_30d_pct,
                       pco.runup_pre_event_30d_vs_xbi_pct,
                       pco.abnormal_move_pct_3d,
                       cu.date_precision
                FROM post_catalyst_outcomes pco
                LEFT JOIN catalyst_universe cu
                  ON cu.ticker = pco.ticker
                 AND cu.catalyst_type = pco.catalyst_type
                 AND cu.catalyst_date::text = pco.catalyst_date::text
                 AND cu.status = 'active'
            """)
            rows = cur.fetchall()

        reclass_count = 0
        for r in rows:
            pco_id, pred_prob, cat_type, runup_raw, runup_vs_xbi, abnormal_3d, dprec = r
            try:
                signal, priced_in = classify_trade_signal_v2(
                    probability=float(pred_prob) if pred_prob is not None else None,
                    runup_30d_pct=float(runup_raw) if runup_raw is not None else None,
                    runup_30d_vs_xbi_pct=float(runup_vs_xbi) if runup_vs_xbi is not None else None,
                    catalyst_type=cat_type,
                    confidence_score=float(pred_prob) if pred_prob is not None else None,
                    date_precision=dprec,
                )
                pred_dir = predicted_direction_v2(signal)
                dir_correct_v2 = None
                if abnormal_3d is not None and pred_dir is not None and is_tradeable_v2(signal):
                    ab = float(abnormal_3d)
                    if abs(ab) >= 3.0:
                        actual_dir = 1 if ab > 0 else -1
                        dir_correct_v2 = (pred_dir == actual_dir)
                with db.get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE post_catalyst_outcomes
                        SET signal_v2 = %s,
                            priced_in_score = %s,
                            direction_correct_v2 = %s
                        WHERE id = %s
                    """, (signal, priced_in, dir_correct_v2, pco_id))
                    conn.commit()
                reclass_count += 1
            except Exception:
                pass

        eval_results = evaluate_pending_snapshots(db=db, max_rows=500)
        return {
            "reclassified": reclass_count,
            "snapshot_evaluation": eval_results,
        }
    except Exception as e:
        logger.exception("v2_reclassify_trigger_now failed")
        raise HTTPException(500, f"v2_reclassify_trigger_now error: {e}")


# ────────────────────────────────────────────────────────────
# Outcome labeler (item 3)
# ────────────────────────────────────────────────────────────

@router.post("/post-catalyst/apply-migration-016")
async def apply_migration_016():
    """One-shot for migration 016 — outcome label columns."""
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                ALTER TABLE post_catalyst_outcomes
                ADD COLUMN IF NOT EXISTS outcome_labeled_json JSONB,
                ADD COLUMN IF NOT EXISTS outcome_label_class TEXT,
                ADD COLUMN IF NOT EXISTS outcome_label_confidence NUMERIC,
                ADD COLUMN IF NOT EXISTS outcome_labeled_at TIMESTAMP WITH TIME ZONE
            """)
            for stmt in [
                "CREATE INDEX IF NOT EXISTS idx_pco_outcome_label_class ON post_catalyst_outcomes(outcome_label_class) WHERE outcome_label_class IS NOT NULL",
                "CREATE INDEX IF NOT EXISTS idx_pco_outcome_labeled_pending ON post_catalyst_outcomes(catalyst_date) WHERE outcome_labeled_at IS NULL AND catalyst_date IS NOT NULL",
            ]:
                cur.execute(stmt)
            cur.execute("""
                UPDATE alembic_version_biotech
                SET version_num = '016_outcome_labels'
                WHERE version_num = '015_snapshot_table'
            """)
            stamped = cur.rowcount
            cur.execute("SELECT version_num FROM alembic_version_biotech")
            new_v = cur.fetchone()[0]
            conn.commit()
        return {"success": True, "new_alembic_version": new_v, "stamped_rows": stamped}
    except Exception as e:
        logger.exception("apply_migration_016 failed")
        raise HTTPException(500, f"apply_migration_016 error: {e}")


@router.post("/post-catalyst/label-outcomes-batch")
async def label_outcomes_batch(max_rows: int = 20, only_new: bool = True):
    """Label catalyst outcomes via Gemini-grounded press release search.
    Cost: ~$0.0003 per row. Default batch is 20 rows = ~$0.006.

    only_new=True (default): skip rows that already have outcome_labeled_at
    set. To re-label, pass only_new=false.
    """
    try:
        from services.database import BiotechDatabase
        from services.outcome_labeler import label_outcome_for_db_row
        db = BiotechDatabase()

        with db.get_conn() as conn:
            cur = conn.cursor()
            where = "WHERE catalyst_date IS NOT NULL"
            if only_new:
                where += " AND outcome_labeled_at IS NULL"
            cur.execute(f"""
                SELECT id FROM post_catalyst_outcomes
                {where}
                ORDER BY catalyst_date DESC
                LIMIT %s
            """, (max_rows,))
            ids = [r[0] for r in cur.fetchall()]

        results = {
            "scanned": 0,
            "labeled": 0,
            "errors": 0,
            "estimated_cost_usd": round(len(ids) * 0.0003, 4),
            "class_distribution": {},
            "sample_outputs": [],
        }
        for outcome_id in ids:
            results["scanned"] += 1
            try:
                labeled = label_outcome_for_db_row(db=db, outcome_id=outcome_id)
                if labeled:
                    results["labeled"] += 1
                    cls = labeled.get("outcome_class", "UNKNOWN")
                    results["class_distribution"][cls] = results["class_distribution"].get(cls, 0) + 1
                    if len(results["sample_outputs"]) < 3:
                        results["sample_outputs"].append({
                            "outcome_id": outcome_id,
                            "class": cls,
                            "confidence": labeled.get("confidence"),
                            "evidence": (labeled.get("evidence") or "")[:120],
                            "source": labeled.get("primary_source_url"),
                        })
                else:
                    results["errors"] += 1
            except Exception as e:
                results["errors"] += 1
                logger.exception(f"label batch row {outcome_id}: {e}")
        return results
    except Exception as e:
        logger.exception("label_outcomes_batch failed")
        raise HTTPException(500, f"label_outcomes_batch error: {e}")


# ============================================================
# Background labeler — concurrent workers
# ============================================================

import threading as _threading  # noqa: E402 — keep near use site

_label_state: dict = {
    "running": False,
    "labeled": 0,
    "errors": 0,
    "started_at": None,
    "stopped_at": None,
    "stop_reason": None,
    "last_error": None,
    "estimated_cost_usd": 0.0,
    "class_distribution": {},
    # Rolling window of last N attempts: True=success, False=failure.
    # Used by the circuit breaker to detect "Gemini is broken, every
    # call returns None" without operator intervention.
    "recent_attempts": [],
    "circuit_breaker_tripped": False,
}
_label_in_progress: set = set()
_label_lock = _threading.Lock()
# Default worker count — overridable per /label-all-start call. Free-tier
# Gemini has ~10 RPM per project; with 5 rotating keys (each potentially
# in a separate project, but we don't trust that) we keep concurrency low
# and let the per-call sleep do the rate shaping.
_LABEL_WORKERS_DEFAULT = 2
# Default per-call sleep inside each worker. 6s × 2 workers = ~20 RPM
# best case, but real grounded-search latency (5-15s) usually keeps us
# well under free-tier 10 RPM.
_LABEL_CALL_INTERVAL_S_DEFAULT = 6.0
# After this many failed attempts on a single row, the row is locked
# out of the claim queue. Prevents one bad row from being retried
# forever and guarantees the queue eventually exhausts.
_LABEL_MAX_ATTEMPTS = 3
# Circuit breaker: if fewer than this many of the last
# _CIRCUIT_WINDOW attempts succeeded, the worker pool aborts. This
# is what would have caught the 2026-05-02 stall — Gemini went into
# instant-None mode and ran 400k+ wasted iterations before anyone
# noticed. We'd rather stop after 100.
_CIRCUIT_WINDOW = 100
_CIRCUIT_MIN_SUCCESS = 5


def _claim_unlabeled(limit: int) -> list:
    """Atomically grab N unlabeled outcome IDs, marking them in-process.

    Skips rows that have already been attempted _LABEL_MAX_ATTEMPTS
    times — those rows are dead-lettered until a future re-run
    explicitly resets the counter.
    """
    from services.database import BiotechDatabase
    with _label_lock:
        excluded = list(_label_in_progress)
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id FROM post_catalyst_outcomes
                WHERE outcome_labeled_at IS NULL
                  AND catalyst_date IS NOT NULL
                  AND COALESCE(outcome_label_attempts, 0) < %s
                  AND id != ALL(%s)
                ORDER BY catalyst_date DESC
                LIMIT %s
            """, (_LABEL_MAX_ATTEMPTS, excluded, limit))
            ids = [r[0] for r in cur.fetchall()]
        for i in ids:
            _label_in_progress.add(i)
    return ids


def _release_label_ids(ids):
    with _label_lock:
        for i in ids:
            _label_in_progress.discard(i)


def _bump_attempt(outcome_id: int, error_text: Optional[str] = None) -> None:
    """Increment the per-row attempt counter and record last_attempt_at.

    Called on EVERY attempt regardless of success — that is the
    invariant that makes the queue exhaust. On success the labeler
    itself also sets outcome_labeled_at, removing the row from the
    claim queue immediately. On failure the row stays unlabeled and
    will be re-claimed until attempts >= _LABEL_MAX_ATTEMPTS.
    """
    from services.database import BiotechDatabase
    db = BiotechDatabase()
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE post_catalyst_outcomes
                SET outcome_label_attempts = COALESCE(outcome_label_attempts, 0) + 1,
                    outcome_label_last_attempt_at = now(),
                    outcome_label_last_error = %s
                WHERE id = %s
            """, (error_text[:500] if error_text else None, outcome_id))
            conn.commit()
    except Exception as e:
        logger.warning(f"_bump_attempt id={outcome_id}: {e}")


def _record_attempt_result(success: bool) -> None:
    """Append to the rolling attempt window and trip the circuit
    breaker if the success rate falls below the threshold."""
    window = _label_state["recent_attempts"]
    window.append(success)
    if len(window) > _CIRCUIT_WINDOW:
        del window[: len(window) - _CIRCUIT_WINDOW]
    if (
        len(window) >= _CIRCUIT_WINDOW
        and sum(window) < _CIRCUIT_MIN_SUCCESS
        and not _label_state["circuit_breaker_tripped"]
    ):
        _label_state["circuit_breaker_tripped"] = True
        _label_state["running"] = False
        _label_state["stop_reason"] = (
            f"circuit breaker: only {sum(window)}/{_CIRCUIT_WINDOW} "
            "recent attempts succeeded — Gemini is likely rate-limited "
            "or the API key is exhausted"
        )
        _label_state["stopped_at"] = datetime.utcnow().isoformat()


@router.post("/post-catalyst/label-all-start")
async def label_all_start(
    claim_size: int = 3,
    workers: int = _LABEL_WORKERS_DEFAULT,
    call_interval_s: float = _LABEL_CALL_INTERVAL_S_DEFAULT,
):
    """Start a background task that labels all unlabeled outcomes via Gemini.

    workers + call_interval_s shape the throughput. Free-tier Gemini Flash
    is 10 RPM per project; if your N keys live in N projects the budget
    is N × 10 RPM but we don't depend on that. Default 2 workers × 6s
    interval ≈ 20 RPM theoretical, ~10-12 RPM realistic given grounded-
    search latency. Bump workers to 4-6 only if you've confirmed each
    key lives in its own project AND the project quota is generous.

    Cost ~$0.0003/event.
    """
    if _label_state["running"]:
        raise HTTPException(409, "label-all already running")
    workers = max(1, min(workers, 12))
    call_interval_s = max(0.0, min(call_interval_s, 60.0))

    import asyncio
    from services.database import BiotechDatabase
    from services.outcome_labeler import label_outcome_for_db_row

    _label_state.update({
        "running": True,
        "labeled": 0,
        "errors": 0,
        "started_at": datetime.utcnow().isoformat(),
        "stopped_at": None,
        "stop_reason": None,
        "last_error": None,
        "estimated_cost_usd": 0.0,
        "class_distribution": {},
        "recent_attempts": [],
        "circuit_breaker_tripped": False,
        "workers": workers,
        "claim_size": claim_size,
        "call_interval_s": call_interval_s,
    })

    async def _worker():
        db = BiotechDatabase()
        while _label_state["running"]:
            ids = await asyncio.to_thread(_claim_unlabeled, claim_size)
            if not ids:
                # Queue exhausted (no rows under the attempt cap). Worker
                # exits; the last worker out flips running=False in _run.
                break
            try:
                for outcome_id in ids:
                    if not _label_state["running"]:
                        break
                    err_text: Optional[str] = None
                    success = False
                    try:
                        labeled = await asyncio.to_thread(
                            lambda oid=outcome_id: label_outcome_for_db_row(db=db, outcome_id=oid)
                        )
                        if labeled:
                            success = True
                            _label_state["labeled"] += 1
                            _label_state["estimated_cost_usd"] = round(_label_state["labeled"] * 0.0003, 4)
                            cls = labeled.get("outcome_class", "UNKNOWN")
                            _label_state["class_distribution"][cls] = _label_state["class_distribution"].get(cls, 0) + 1
                        else:
                            err_text = "labeler returned None (no source found or transient API failure)"
                            _label_state["errors"] += 1
                    except Exception as e:
                        err_text = f"{type(e).__name__}: {e}"
                        _label_state["errors"] += 1
                        _label_state["last_error"] = err_text[:200]
                        logger.warning(f"label-all row {outcome_id}: {e}")
                    # Always bump the attempt counter — guarantees every
                    # row eventually drops out of the claim queue. On
                    # success the labeler also sets outcome_labeled_at
                    # which removes the row independently.
                    if not success:
                        await asyncio.to_thread(_bump_attempt, outcome_id, err_text)
                    _record_attempt_result(success)
                    if _label_state["circuit_breaker_tripped"]:
                        break
                    # Throttle: stay under per-project Gemini RPM. The
                    # per-key cooldown only catches keys that already 429'd;
                    # this prevents the 429 in the first place.
                    if call_interval_s > 0:
                        await asyncio.sleep(call_interval_s)
            finally:
                await asyncio.to_thread(_release_label_ids, ids)

    async def _run():
        try:
            await asyncio.gather(*[_worker() for _ in range(workers)])
        except Exception as e:
            logger.exception(f"[label-all] error: {e}")
            _label_state["last_error"] = str(e)[:200]
        finally:
            if _label_state["running"]:
                # Natural completion (queue exhausted, no breaker, no stop call)
                _label_state["stop_reason"] = "queue exhausted"
                _label_state["stopped_at"] = datetime.utcnow().isoformat()
            _label_state["running"] = False

    asyncio.create_task(_run())
    return {
        "ok": True,
        "status_url": "/admin/post-catalyst/label-all-status",
        "workers": workers,
        "claim_size": claim_size,
        "call_interval_s": call_interval_s,
    }


@router.get("/llm/status")
async def llm_status():
    """Universal LLM gateway health — every provider, every key,
    circuit-breaker state. The 'never stalls again' single source of
    truth: if this endpoint shows everything green, no LLM call in
    biotech-api should hang.
    """
    try:
        from services.llm_gateway import get_status
        return get_status()
    except Exception as e:
        logger.exception("llm_status failed")
        raise HTTPException(500, f"llm_status error: {e}")


@router.get("/labeler/key-status")
async def labeler_key_status():
    """Per-Gemini-key health for the outcome labeler. Surfaces success
    counts, last error, and whether a key is currently on cooldown
    (rate-limited / quota-exhausted / timeout / auth-failed). The
    labeler picks the LRU non-cooling key on every call.
    """
    try:
        from services.outcome_labeler import get_key_status
        return get_key_status()
    except Exception as e:
        logger.exception("labeler_key_status failed")
        raise HTTPException(500, f"labeler_key_status error: {e}")


@router.get("/post-catalyst/label-all-status")
async def label_all_status():
    """Live progress for background label-all task.

    Surfaces the rolling success ratio used by the circuit breaker so
    the operator can see "Gemini is degraded" before the breaker
    actually trips.
    """
    state = _label_state.copy()
    state["in_progress_count"] = len(_label_in_progress)
    window = state.pop("recent_attempts", [])
    state["recent_window_size"] = len(window)
    state["recent_success_pct"] = (
        round(100.0 * sum(window) / len(window), 1) if window else None
    )
    state["circuit_window"] = _CIRCUIT_WINDOW
    state["circuit_min_success"] = _CIRCUIT_MIN_SUCCESS
    state["max_attempts_per_row"] = _LABEL_MAX_ATTEMPTS
    return state


@router.post("/post-catalyst/label-all-stop")
async def label_all_stop():
    """Cooperative stop for the background labeler. Sets running=False;
    workers exit at their next loop check. Use when you need to halt
    a runaway labeler without redeploying the service.
    """
    if not _label_state["running"]:
        return {
            "ok": True,
            "already_stopped": True,
            "stop_reason": _label_state.get("stop_reason"),
        }
    _label_state["running"] = False
    _label_state["stop_reason"] = "stopped by /label-all-stop"
    _label_state["stopped_at"] = datetime.utcnow().isoformat()
    return {
        "ok": True,
        "stopped_at": _label_state["stopped_at"],
        "labeled": _label_state["labeled"],
        "errors": _label_state["errors"],
    }


@router.post("/post-catalyst/label-reset-attempts")
async def label_reset_attempts(only_errored: bool = True):
    """Zero out the per-row attempt counter so dead-lettered rows can
    be re-tried. Use this after a Gemini outage to give failed rows
    another shot — typically with `only_errored=true` to leave
    successfully-labeled rows alone.
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            if only_errored:
                cur.execute("""
                    UPDATE post_catalyst_outcomes
                    SET outcome_label_attempts = 0,
                        outcome_label_last_error = NULL
                    WHERE outcome_labeled_at IS NULL
                      AND COALESCE(outcome_label_attempts, 0) > 0
                """)
            else:
                cur.execute("""
                    UPDATE post_catalyst_outcomes
                    SET outcome_label_attempts = 0,
                        outcome_label_last_error = NULL
                """)
            reset_n = cur.rowcount
            conn.commit()
        return {"ok": True, "rows_reset": reset_n, "only_errored": only_errored}
    except Exception as e:
        logger.exception("label_reset_attempts failed")
        raise HTTPException(500, f"label_reset_attempts error: {e}")


@router.get("/post-catalyst/outcome-label-stats")
async def outcome_label_stats():
    """Coverage + class distribution of labeled outcomes, plus the
    per-row attempt-counter histogram (which rows have been tried 0,
    1, 2, 3+ times). The 3+ bucket is the dead-letter queue —
    rows that hit _LABEL_MAX_ATTEMPTS and won't be re-claimed until
    the operator calls /label-reset-attempts.
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE outcome_labeled_at IS NOT NULL) AS labeled,
                    COUNT(*) FILTER (WHERE outcome_label_class = 'APPROVED') AS approved,
                    COUNT(*) FILTER (WHERE outcome_label_class = 'REJECTED') AS rejected,
                    COUNT(*) FILTER (WHERE outcome_label_class = 'MET_ENDPOINT') AS met_endpoint,
                    COUNT(*) FILTER (WHERE outcome_label_class = 'MISSED_ENDPOINT') AS missed_endpoint,
                    COUNT(*) FILTER (WHERE outcome_label_class = 'DELAYED') AS delayed,
                    COUNT(*) FILTER (WHERE outcome_label_class = 'WITHDRAWN') AS withdrawn,
                    COUNT(*) FILTER (WHERE outcome_label_class = 'MIXED') AS mixed,
                    COUNT(*) FILTER (WHERE outcome_label_class = 'UNKNOWN') AS unknown,
                    AVG(outcome_label_confidence) FILTER (WHERE outcome_labeled_at IS NOT NULL) AS avg_conf,
                    AVG(outcome_label_confidence) FILTER (WHERE outcome_label_class != 'UNKNOWN') AS avg_conf_known,
                    COUNT(*) FILTER (
                        WHERE outcome_labeled_at IS NULL
                          AND COALESCE(outcome_label_attempts, 0) = 0
                    ) AS pending_untried,
                    COUNT(*) FILTER (
                        WHERE outcome_labeled_at IS NULL
                          AND outcome_label_attempts BETWEEN 1 AND 2
                    ) AS pending_retryable,
                    COUNT(*) FILTER (
                        WHERE outcome_labeled_at IS NULL
                          AND outcome_label_attempts >= 3
                    ) AS dead_lettered
                FROM post_catalyst_outcomes
            """)
            r = cur.fetchone()
        total = r[0] or 0
        labeled = r[1] or 0
        return {
            "total_outcomes": total,
            "labeled_outcomes": labeled,
            "labeled_pct": round(100.0 * labeled / total, 1) if total > 0 else 0,
            "class_distribution": {
                "APPROVED": r[2] or 0, "REJECTED": r[3] or 0,
                "MET_ENDPOINT": r[4] or 0, "MISSED_ENDPOINT": r[5] or 0,
                "DELAYED": r[6] or 0, "WITHDRAWN": r[7] or 0,
                "MIXED": r[8] or 0, "UNKNOWN": r[9] or 0,
            },
            "avg_confidence": round(float(r[10]), 2) if r[10] is not None else None,
            "avg_confidence_known": round(float(r[11]), 2) if r[11] is not None else None,
            "estimated_full_backfill_cost_usd": round((total - labeled) * 0.0003, 4),
            "attempt_buckets": {
                "pending_untried": r[12] or 0,
                "pending_retryable": r[13] or 0,
                "dead_lettered": r[14] or 0,
            },
        }
    except Exception as e:
        logger.exception("outcome_label_stats failed")
        raise HTTPException(500, f"outcome_label_stats error: {e}")


@router.post("/post-catalyst/apply-migration-019")
async def apply_migration_019():
    """One-shot for migration 019 — per-row attempt counter on the
    outcome labeler. Adds outcome_label_attempts (defaults 0),
    outcome_label_last_attempt_at, outcome_label_last_error. Index
    keeps the claim query fast.

    Required by the Gemini-stall workaround: the labeler now skips
    rows with attempts >= _LABEL_MAX_ATTEMPTS so a single bad row or
    a global Gemini outage cannot loop forever.

    Note: an earlier endpoint (apply-migration-018) was applied
    against production before the alembic name collision with
    018_lgbm_model was noticed. The DDL is idempotent so re-running
    via this endpoint is safe; only the alembic version_num bump
    differs.
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                ALTER TABLE post_catalyst_outcomes
                ADD COLUMN IF NOT EXISTS outcome_label_attempts INTEGER NOT NULL DEFAULT 0,
                ADD COLUMN IF NOT EXISTS outcome_label_last_attempt_at TIMESTAMP WITH TIME ZONE,
                ADD COLUMN IF NOT EXISTS outcome_label_last_error TEXT
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_pco_label_pending_attempts
                ON post_catalyst_outcomes(catalyst_date)
                WHERE outcome_labeled_at IS NULL
                  AND outcome_label_attempts < 3
                  AND catalyst_date IS NOT NULL
            """)
            cur.execute("""
                UPDATE alembic_version_biotech
                SET version_num = '019_label_attempts'
                WHERE version_num = '018_lgbm_model'
            """)
            stamped = cur.rowcount
            cur.execute("SELECT version_num FROM alembic_version_biotech")
            new_v = cur.fetchone()[0]
            conn.commit()
        return {"success": True, "new_alembic_version": new_v, "stamped_rows": stamped}
    except Exception as e:
        logger.exception("apply_migration_019 failed")
        raise HTTPException(500, f"apply_migration_019 error: {e}")


# ────────────────────────────────────────────────────────────
# EDGAR 8-K backfill (historical catalyst scraping)
# ────────────────────────────────────────────────────────────
# User selected: 2015-2025 deepest, source priority EDGAR > Finnhub > BPC,
# one-shot scrape. This backfills historical biotech catalysts from SEC
# 8-K filings, extracts events via Gemini, dedupes against catalyst_universe.

@router.get("/edgar/recon/{ticker}")
async def edgar_recon(ticker: str, start_date: str = "2015-01-01",
                      end_date: Optional[str] = None):
    """Quick recon: how many 8-Ks does a ticker have in the window, and
    how many look like catalysts? Read-only, no writes. ~1-2s per ticker."""
    try:
        from services.edgar_scraper import (
            resolve_cik, list_8k_filings, fetch_filing_text, is_catalyst_8k,
        )
        cik = resolve_cik(ticker)
        if not cik:
            return {"ticker": ticker.upper(), "cik": None, "error": "ticker not in SEC company_tickers"}
        filings = list_8k_filings(cik, start_date=start_date, end_date=end_date)

        # Classify a sample of up to 10 to see hit rate
        sample = filings[:10] if filings else []
        catalyst_hits = 0
        sampled = 0
        sample_results = []
        for f in sample:
            if not f.get("primary_doc"):
                continue
            sampled += 1
            text = fetch_filing_text(cik, f["accession"], f["primary_doc"])
            if not text:
                continue
            is_cat, kws = is_catalyst_8k(text, f.get("items", ""))
            if is_cat:
                catalyst_hits += 1
            sample_results.append({
                "filing_date": f["filing_date"],
                "items": f["items"],
                "is_catalyst_keyword_hit": is_cat,
                "keywords_matched": kws,
            })

        return {
            "ticker": ticker.upper(),
            "cik": cik,
            "total_8ks_in_range": len(filings),
            "sample_size": sampled,
            "catalyst_hits_in_sample": catalyst_hits,
            "estimated_catalyst_8ks": int(len(filings) * (catalyst_hits / sampled)) if sampled > 0 else None,
            "sample_results": sample_results,
        }
    except Exception as e:
        logger.exception("edgar_recon failed")
        raise HTTPException(500, f"edgar_recon error: {e}")


@router.post("/edgar/backfill-ticker")
async def edgar_backfill_ticker(
    ticker: str,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    max_filings: int = 50,
    extract: bool = True,
    dry_run: bool = False,
):
    """Scrape one ticker's 8-K archive, keyword-filter to catalysts,
    LLM-extract structured events, dedupe + write to catalyst_universe.

    extract=False stops after keyword filter (no LLM cost). Useful for
    dry runs to see how many catalysts will be processed.

    dry_run=True does everything except the final DB write.

    Returns counts + sample extracted events.
    """
    try:
        from services.edgar_scraper import (
            resolve_cik, list_8k_filings, fetch_filing_text,
            is_catalyst_8k, extract_catalyst_via_llm,
        )
        from services.universe_seeder import _canonicalize_drug
        from services.database import BiotechDatabase

        ticker_u = ticker.upper().strip()
        cik = resolve_cik(ticker_u)
        if not cik:
            return {"error": f"ticker {ticker_u} not in SEC company_tickers"}

        # Get company name from screener_stocks
        db = BiotechDatabase()
        company_name = ticker_u
        try:
            with db.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("SELECT company_name FROM screener_stocks WHERE ticker = %s", (ticker_u,))
                r = cur.fetchone()
                if r:
                    company_name = r[0] or ticker_u
        except Exception:
            pass

        filings = list_8k_filings(cik, start_date=start_date, end_date=end_date)
        if max_filings and len(filings) > max_filings:
            filings = filings[:max_filings]

        results = {
            "ticker": ticker_u,
            "cik": cik,
            "total_8ks_in_range": len(filings),
            "keyword_hits": 0,
            "llm_extracted": 0,
            "llm_skipped_not_catalyst": 0,
            "llm_failures": 0,
            "rows_added": 0,
            "rows_updated": 0,
            "rows_skipped_dup": 0,
            "estimated_llm_cost_usd": 0.0,
            "sample_events": [],
        }

        # Pass 1: keyword filter
        candidates = []
        for f in filings:
            if not f.get("primary_doc"):
                continue
            text = fetch_filing_text(cik, f["accession"], f["primary_doc"])
            if not text:
                continue
            is_cat, _ = is_catalyst_8k(text, f.get("items", ""))
            if is_cat:
                candidates.append({**f, "_text": text})
        results["keyword_hits"] = len(candidates)

        if not extract:
            return results

        # Pass 2: LLM extract
        results["estimated_llm_cost_usd"] = round(len(candidates) * 0.0003, 4)
        for c in candidates:
            extracted = extract_catalyst_via_llm(
                ticker=ticker_u, cik=cik,
                filing_date=c["filing_date"],
                text=c["_text"],
            )
            if not extracted:
                results["llm_failures"] += 1
                continue
            if not extracted.get("is_catalyst"):
                results["llm_skipped_not_catalyst"] += 1
                continue

            results["llm_extracted"] += 1
            # Build canonical row
            cat_type = extracted.get("catalyst_type")
            cat_date = extracted.get("catalyst_date_iso") or c["filing_date"]
            drug_name = extracted.get("drug_name")
            indication = extracted.get("indication")
            outcome_class = extracted.get("outcome_class")
            evidence = extracted.get("evidence")
            confidence = extracted.get("confidence", 0.5)

            if not cat_type:
                continue

            source_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{c['accession'].replace('-', '')}/{c['primary_doc']}"

            if dry_run:
                if len(results["sample_events"]) < 5:
                    results["sample_events"].append({
                        "filing_date": c["filing_date"],
                        "catalyst_type": cat_type,
                        "drug_name": drug_name,
                        "indication": indication,
                        "outcome_class": outcome_class,
                        "evidence": (evidence or "")[:120],
                        "source_url": source_url,
                    })
                continue

            # Write catalyst_universe row
            try:
                with db.get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO catalyst_universe
                            (ticker, company_name, catalyst_type, catalyst_date, date_precision,
                             description, drug_name, canonical_drug_name, indication, phase,
                             source, source_url, confidence_score, status, last_updated)
                        VALUES (%s, %s, %s, %s, 'exact', %s, %s, %s, %s, %s,
                                'edgar_8k', %s, %s, 'active', NOW())
                        ON CONFLICT (ticker, catalyst_type, catalyst_date, canonical_drug_name)
                        WHERE canonical_drug_name IS NOT NULL AND status = 'active'
                        DO UPDATE SET
                            description = EXCLUDED.description,
                            confidence_score = GREATEST(catalyst_universe.confidence_score,
                                                         EXCLUDED.confidence_score),
                            last_updated = NOW()
                        WHERE catalyst_universe.is_manual_override = FALSE
                           OR catalyst_universe.is_manual_override IS NULL
                        RETURNING (xmax = 0) AS inserted
                    """, (
                        ticker_u, company_name, cat_type, cat_date,
                        evidence,
                        drug_name, _canonicalize_drug(drug_name),
                        indication, None,
                        source_url, confidence,
                    ))
                    row = cur.fetchone()
                    conn.commit()

                if row is None:
                    results["rows_skipped_dup"] += 1
                else:
                    if row[0]:
                        results["rows_added"] += 1
                    else:
                        results["rows_updated"] += 1
                    if len(results["sample_events"]) < 5:
                        results["sample_events"].append({
                            "filing_date": c["filing_date"],
                            "catalyst_type": cat_type,
                            "drug_name": drug_name,
                            "outcome_class": outcome_class,
                            "evidence": (evidence or "")[:120],
                        })
            except Exception as e:
                logger.warning(f"[edgar] insert failed for {ticker_u}/{cat_date}/{cat_type}: {e}")

        return results
    except Exception as e:
        logger.exception("edgar_backfill_ticker failed")
        raise HTTPException(500, f"edgar_backfill_ticker error: {e}")


@router.post("/edgar/backfill-batch")
async def edgar_backfill_batch(
    max_tickers: int = 10,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    max_filings_per_ticker: int = 100,
    skip_done: bool = True,
    extract: bool = True,
    dry_run: bool = False,
):
    """Batch over multiple tickers in our universe. Use skip_done=True
    (default) to avoid re-scraping tickers that already have 'edgar_8k'-
    sourced rows in catalyst_universe (idempotent across runs).

    For the full 740-ticker / 10-year backfill, this should be called
    repeatedly (each call processes max_tickers tickers) until no more
    are returned.
    """
    try:
        from services.edgar_scraper import resolve_cik
        from services.database import BiotechDatabase
        db = BiotechDatabase()

        with db.get_conn() as conn:
            cur = conn.cursor()
            if skip_done:
                cur.execute("""
                    SELECT ticker FROM screener_stocks
                    WHERE ticker NOT IN (
                        SELECT DISTINCT ticker FROM catalyst_universe
                        WHERE source = 'edgar_8k'
                    )
                    ORDER BY market_cap DESC NULLS LAST
                    LIMIT %s
                """, (max_tickers,))
            else:
                cur.execute("""
                    SELECT ticker FROM screener_stocks
                    ORDER BY market_cap DESC NULLS LAST
                    LIMIT %s
                """, (max_tickers,))
            tickers = [r[0] for r in cur.fetchall()]

        batch_results = {
            "tickers_processed": 0,
            "tickers_no_cik": 0,
            "total_8ks_scanned": 0,
            "total_keyword_hits": 0,
            "total_llm_extracted": 0,
            "total_rows_added": 0,
            "total_rows_updated": 0,
            "total_estimated_cost_usd": 0.0,
            "errors": [],
            "per_ticker": [],
        }

        for ticker in tickers:
            try:
                cik = resolve_cik(ticker)
                if not cik:
                    batch_results["tickers_no_cik"] += 1
                    continue
                # Inline the per-ticker logic by calling the function directly
                r = await edgar_backfill_ticker(
                    ticker=ticker, start_date=start_date, end_date=end_date,
                    max_filings=max_filings_per_ticker,
                    extract=extract, dry_run=dry_run,
                )
                if isinstance(r, dict) and "error" not in r:
                    batch_results["tickers_processed"] += 1
                    batch_results["total_8ks_scanned"] += r.get("total_8ks_in_range", 0)
                    batch_results["total_keyword_hits"] += r.get("keyword_hits", 0)
                    batch_results["total_llm_extracted"] += r.get("llm_extracted", 0)
                    batch_results["total_rows_added"] += r.get("rows_added", 0)
                    batch_results["total_rows_updated"] += r.get("rows_updated", 0)
                    batch_results["total_estimated_cost_usd"] += r.get("estimated_llm_cost_usd", 0)
                    batch_results["per_ticker"].append({
                        "ticker": ticker,
                        "8ks": r.get("total_8ks_in_range"),
                        "keyword_hits": r.get("keyword_hits"),
                        "extracted": r.get("llm_extracted"),
                        "added": r.get("rows_added"),
                        "updated": r.get("rows_updated"),
                    })
            except Exception as e:
                batch_results["errors"].append({"ticker": ticker, "error": str(e)[:120]})

        batch_results["total_estimated_cost_usd"] = round(batch_results["total_estimated_cost_usd"], 4)
        return batch_results
    except Exception as e:
        logger.exception("edgar_backfill_batch failed")
        raise HTTPException(500, f"edgar_backfill_batch error: {e}")


@router.get("/edgar/diag")
async def edgar_diag(ticker: str = "NTLA"):
    """Diagnostic: test which SEC endpoints work from this server.
    SEC blocks some IPs/UAs; we need to know which paths are reachable."""
    import requests
    UA = "AEGRA Biotech Research [email protected]"
    H = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
    results = {}

    # Test 1: data.sec.gov submissions for known CIK
    try:
        r = requests.get(
            f"https://data.sec.gov/submissions/CIK0001652130.json",
            headers=H, timeout=10,
        )
        results["data_sec_submissions"] = {
            "status": r.status_code,
            "bytes": len(r.content),
            "ticker_match": "NTLA" in r.text if r.status_code == 200 else None,
        }
    except Exception as e:
        results["data_sec_submissions"] = {"error": str(e)[:100]}

    # Test 2: www.sec.gov/files company_tickers
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=H, timeout=10,
        )
        results["www_sec_company_tickers"] = {
            "status": r.status_code,
            "bytes": len(r.content),
            "is_blocked_html": "Undeclared Automated Tool" in r.text,
        }
    except Exception as e:
        results["www_sec_company_tickers"] = {"error": str(e)[:100]}

    # Test 3: www.sec.gov Archives (where 8-K HTML lives)
    try:
        r = requests.get(
            "https://www.sec.gov/Archives/edgar/data/1652130/000119312526179401/d138980d8k.htm",
            headers=H, timeout=10,
        )
        results["www_sec_archives_8k"] = {
            "status": r.status_code,
            "bytes": len(r.content),
            "is_blocked_html": "Undeclared Automated Tool" in r.text,
        }
    except Exception as e:
        results["www_sec_archives_8k"] = {"error": str(e)[:100]}

    return {"ticker_tested": ticker, "results": results, "user_agent": UA}


# ────────────────────────────────────────────────────────────
# Backfill staging + EDGAR scraper (one-shot)
# ────────────────────────────────────────────────────────────

@router.post("/post-catalyst/apply-migration-017")
async def apply_migration_017():
    """One-shot for migration 017 — backfill_staging table."""
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS catalyst_backfill_staging (
                    id BIGSERIAL PRIMARY KEY,
                    source TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    UNIQUE (source, source_id),
                    ticker TEXT, cik TEXT,
                    filing_date DATE, catalyst_date DATE,
                    date_precision TEXT DEFAULT 'unknown',
                    catalyst_type TEXT, drug_name TEXT, indication TEXT,
                    raw_title TEXT, raw_text_excerpt TEXT, source_url TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    reject_reason TEXT, catalyst_id INTEGER,
                    normalized_json JSONB,
                    scraped_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                    processed_at TIMESTAMP WITH TIME ZONE
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS catalyst_backfill_runs (
                    id BIGSERIAL PRIMARY KEY,
                    source TEXT NOT NULL,
                    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                    ended_at TIMESTAMP WITH TIME ZONE,
                    year_range TEXT,
                    params_json JSONB,
                    rows_scraped INTEGER DEFAULT 0,
                    rows_inserted INTEGER DEFAULT 0,
                    rows_skipped INTEGER DEFAULT 0,
                    rows_errored INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running',
                    error_message TEXT
                )
            """)
            for stmt in [
                "CREATE INDEX IF NOT EXISTS idx_bks_status_source ON catalyst_backfill_staging(status, source)",
                "CREATE INDEX IF NOT EXISTS idx_bks_ticker_filing_date ON catalyst_backfill_staging(ticker, filing_date) WHERE ticker IS NOT NULL",
                "CREATE INDEX IF NOT EXISTS idx_bks_cik ON catalyst_backfill_staging(cik) WHERE cik IS NOT NULL",
                "CREATE INDEX IF NOT EXISTS idx_bks_pending ON catalyst_backfill_staging(scraped_at) WHERE status = 'pending'",
            ]:
                cur.execute(stmt)
            cur.execute("""
                UPDATE alembic_version_biotech
                SET version_num = '017_backfill_staging'
                WHERE version_num = '016_outcome_labels'
            """)
            stamped = cur.rowcount
            cur.execute("SELECT version_num FROM alembic_version_biotech")
            new_v = cur.fetchone()[0]
            conn.commit()
        return {"success": True, "new_alembic_version": new_v, "stamped_rows": stamped}
    except Exception as e:
        logger.exception("apply_migration_017 failed")
        raise HTTPException(500, f"apply_migration_017 error: {e}")


# Module-level state for the background normalize-all task
_normalize_all_state: dict = {
    "running": False,
    "batches_done": 0,
    "rows_accepted": 0,
    "rows_rejected": 0,
    "rows_duplicate": 0,
    "rows_unclear": 0,
    "rows_errored": 0,
    "started_at": None,
    "last_error": None,
}


# Module-level state for tracking long-running EDGAR backfill
_edgar_backfill_state: dict = {
    "running": False,
    "current_run_id": None,
    "ciks_total": 0,
    "ciks_processed": 0,
    "filings_scraped": 0,
    "filings_inserted": 0,
    "current_ticker": None,
    "started_at": None,
    "last_error": None,
}


@router.post("/post-catalyst/edgar-backfill-start")
async def edgar_backfill_start(
    start_year: int = 2015,
    end_year: int = 2025,
    max_ciks: int = 0,  # 0 = no limit
    background_tasks: BackgroundTasks = None,
):
    """Kick off the EDGAR 10-year backfill in the background.
    One-shot — does not run on a schedule.

    Reads CIKs from catalyst_universe + screener_stocks, looks up each
    via SEC's company_tickers.json, then scans 8-K filings.

    max_ciks: cap for testing (0 = scan all)
    """
    if _edgar_backfill_state["running"]:
        raise HTTPException(409, "EDGAR backfill already running")

    import asyncio
    from services.database import BiotechDatabase
    from services.edgar_backfill import (
        fetch_biotech_ciks_from_universe, edgar_backfill_for_cik,
    )

    db = BiotechDatabase()

    # Create run record
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO catalyst_backfill_runs (
                    source, year_range, params_json, status
                ) VALUES (
                    'edgar', %s, %s, 'running'
                )
                RETURNING id
            """, (
                f"{start_year}-{end_year}",
                json.dumps({
                    "start_year": start_year, "end_year": end_year,
                    "max_ciks": max_ciks,
                }),
            ))
            run_id = cur.fetchone()[0]
            conn.commit()
    except Exception as e:
        raise HTTPException(500, f"failed to create run record: {e}")

    _edgar_backfill_state.update({
        "running": True,
        "current_run_id": run_id,
        "ciks_total": 0,
        "ciks_processed": 0,
        "filings_scraped": 0,
        "filings_inserted": 0,
        "current_ticker": None,
        "started_at": datetime.utcnow().isoformat(),
        "last_error": None,
    })

    async def _run():
        from datetime import datetime as _dt
        try:
            cik_list = fetch_biotech_ciks_from_universe(db)
            if max_ciks:
                cik_list = cik_list[:max_ciks]
            _edgar_backfill_state["ciks_total"] = len(cik_list)
            logger.info(f"[edgar-backfill] starting {len(cik_list)} CIKs, "
                        f"years {start_year}-{end_year}, run_id={run_id}")

            total_scraped = 0
            total_inserted = 0
            total_errored = 0

            for cik_padded, ticker, _name in cik_list:
                _edgar_backfill_state["current_ticker"] = ticker
                try:
                    counts = edgar_backfill_for_cik(
                        db, cik_padded, ticker, start_year, end_year, run_id,
                    )
                    total_scraped += counts.get("scraped", 0)
                    total_inserted += counts.get("inserted", 0)
                    total_errored += counts.get("errored", 0)
                    _edgar_backfill_state["ciks_processed"] += 1
                    _edgar_backfill_state["filings_scraped"] = total_scraped
                    _edgar_backfill_state["filings_inserted"] = total_inserted
                    # Periodic progress update
                    if _edgar_backfill_state["ciks_processed"] % 20 == 0:
                        with db.get_conn() as conn:
                            cur = conn.cursor()
                            cur.execute("""
                                UPDATE catalyst_backfill_runs
                                SET rows_scraped = %s,
                                    rows_inserted = %s,
                                    rows_errored = %s
                                WHERE id = %s
                            """, (total_scraped, total_inserted, total_errored, run_id))
                            conn.commit()
                except Exception as e:
                    total_errored += 1
                    _edgar_backfill_state["last_error"] = str(e)[:200]
                    logger.warning(f"[edgar-backfill] CIK {cik_padded} ({ticker}) failed: {e}")

            with db.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE catalyst_backfill_runs
                    SET ended_at = now(), status = 'completed',
                        rows_scraped = %s, rows_inserted = %s, rows_errored = %s
                    WHERE id = %s
                """, (total_scraped, total_inserted, total_errored, run_id))
                conn.commit()
            logger.info(f"[edgar-backfill] DONE: {total_inserted} inserted "
                        f"out of {total_scraped} scraped, {total_errored} errors")
        except Exception as e:
            logger.exception(f"[edgar-backfill] top-level failure: {e}")
            try:
                with db.get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE catalyst_backfill_runs
                        SET ended_at = now(), status = 'failed',
                            error_message = %s
                        WHERE id = %s
                    """, (str(e)[:500], run_id))
                    conn.commit()
            except Exception:
                pass
            _edgar_backfill_state["last_error"] = str(e)[:200]
        finally:
            _edgar_backfill_state["running"] = False
            _edgar_backfill_state["current_ticker"] = None

    asyncio.create_task(_run())
    return {
        "ok": True, "run_id": run_id,
        "status_url": "/admin/post-catalyst/edgar-backfill-status",
    }


@router.get("/post-catalyst/edgar-backfill-status")
async def edgar_backfill_status():
    """Live progress of the EDGAR backfill."""
    return _edgar_backfill_state.copy()


@router.get("/post-catalyst/backfill-staging-stats")
async def backfill_staging_stats():
    """Counts of staging rows by source × status."""
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT source, status, COUNT(*)
                FROM catalyst_backfill_staging
                GROUP BY source, status
                ORDER BY source, status
            """)
            rows = cur.fetchall()
            cur.execute("""
                SELECT id, source, started_at, ended_at, year_range,
                       rows_scraped, rows_inserted, rows_errored, status,
                       error_message
                FROM catalyst_backfill_runs
                ORDER BY started_at DESC
                LIMIT 10
            """)
            runs = cur.fetchall()
        by_source = {}
        for source, status, n in rows:
            by_source.setdefault(source, {})[status] = n
        return {
            "by_source_status": by_source,
            "recent_runs": [
                {
                    "id": r[0], "source": r[1],
                    "started_at": r[2].isoformat() if r[2] else None,
                    "ended_at": r[3].isoformat() if r[3] else None,
                    "year_range": r[4],
                    "rows_scraped": r[5], "rows_inserted": r[6],
                    "rows_errored": r[7], "status": r[8],
                    "error": (r[9] or "")[:200],
                }
                for r in runs
            ],
        }
    except Exception as e:
        logger.exception("backfill_staging_stats failed")
        raise HTTPException(500, f"backfill_staging_stats error: {e}")


@router.post("/post-catalyst/backfill-normalize-batch")
async def backfill_normalize_batch(batch_size: int = 50):
    """Run the LLM normalizer on a batch of pending staging rows.
    Promotes accepted rows into catalyst_universe.
    Cost: ~$0.0003 per row.
    """
    try:
        from services.database import BiotechDatabase
        from services.backfill_normalizer import normalize_pending_batch
        db = BiotechDatabase()
        return normalize_pending_batch(db=db, batch_size=batch_size)
    except Exception as e:
        logger.exception("backfill_normalize_batch failed")
        raise HTTPException(500, f"backfill_normalize_batch error: {e}")


_NORMALIZE_WORKERS = 4


@router.post("/post-catalyst/normalize-all-start")
async def normalize_all_start():
    """Start a background task that runs normalize_pending_batch() in a loop
    until pending=0. Returns immediately; poll /normalize-all-status for progress.
    Runs _NORMALIZE_WORKERS concurrent workers with SKIP LOCKED for throughput.
    Cost: ~$0.0003 per row.
    """
    if _normalize_all_state["running"]:
        raise HTTPException(409, "normalize-all already running")

    import asyncio
    from services.database import BiotechDatabase
    from services.backfill_normalizer import normalize_pending_batch

    # Reset any rows stuck in 'processing' state from a previous crashed run.
    try:
        db_reset = BiotechDatabase()
        with db_reset.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE catalyst_backfill_staging SET status = 'pending'
                WHERE status = 'processing'
            """)
            reset_count = cur.rowcount
            conn.commit()
        if reset_count:
            logger.info(f"[normalize-all] reset {reset_count} stuck 'processing' rows → 'pending'")
    except Exception as e:
        logger.warning(f"[normalize-all] failed to reset processing rows: {e}")

    _normalize_all_state.update({
        "running": True,
        "batches_done": 0,
        "rows_accepted": 0,
        "rows_rejected": 0,
        "rows_duplicate": 0,
        "rows_unclear": 0,
        "rows_errored": 0,
        "started_at": datetime.utcnow().isoformat(),
        "last_error": None,
    })

    async def _worker():
        db = BiotechDatabase()
        while True:
            result = await asyncio.to_thread(normalize_pending_batch, db=db, batch_size=50)
            if result.get("scanned", 0) == 0:
                break
            _normalize_all_state["batches_done"] += 1
            _normalize_all_state["rows_accepted"] += result.get("accepted", 0)
            _normalize_all_state["rows_rejected"] += result.get("rejected", 0)
            _normalize_all_state["rows_duplicate"] += result.get("duplicate", 0)
            _normalize_all_state["rows_unclear"] += result.get("unclear", 0)
            _normalize_all_state["rows_errored"] += result.get("errored", 0)
            await asyncio.sleep(0.1)

    async def _run():
        try:
            await asyncio.gather(*[_worker() for _ in range(_NORMALIZE_WORKERS)])
        except Exception as e:
            logger.exception(f"[normalize-all] error: {e}")
            _normalize_all_state["last_error"] = str(e)[:200]
        finally:
            _normalize_all_state["running"] = False

    asyncio.create_task(_run())
    return {"ok": True, "status_url": "/admin/post-catalyst/normalize-all-status"}


@router.get("/post-catalyst/normalize-all-status")
async def normalize_all_status():
    """Live progress for the background normalize-all task."""
    return _normalize_all_state.copy()


# ============================================================
# V3 LightGBM classifier — train + predict + info
# ============================================================

@router.post("/post-catalyst/train-v3-model")
async def train_v3_model(
    min_outcome_confidence: float = 0.7,
    notes: str = "",
):
    """Train a fresh V3 LightGBM model on currently-labeled events.
    Returns metrics; persists model to lgbm_models table (marks prior active=false).

    Walk-forward split: oldest 80% train, newest 20% test (catches era overfit).
    """
    try:
        from services.database import BiotechDatabase
        from services.lgbm_classifier import train_v3_lgbm, save_model_to_db
        db = BiotechDatabase()
        version = f"v3-lgbm-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        result = train_v3_lgbm(db=db, min_outcome_confidence=min_outcome_confidence)
        new_id = save_model_to_db(db=db, training_result=result, model_version=version, notes=notes)
        return {
            "ok": True,
            "model_id": new_id,
            "model_version": version,
            "train_n": result["train_n"],
            "test_n": result["test_n"],
            "train_accuracy": round(result["train_accuracy"], 4),
            "test_accuracy": round(result["test_accuracy"], 4),
            "test_ci_lower_pct": result["test_ci_lower_pct"],
            "test_ci_upper_pct": result["test_ci_upper_pct"],
            "high_conf_subset": result["metrics_json"].get("high_conf_subset"),
            "feature_importance": result["metrics_json"].get("feature_importance"),
        }
    except Exception as e:
        logger.exception("train_v3_model failed")
        raise HTTPException(500, f"train_v3_model error: {e}")


@router.get("/post-catalyst/v3-model-info")
async def v3_model_info():
    """Return the active V3 model metadata. Null if no model has been trained."""
    try:
        from services.database import BiotechDatabase
        from services.lgbm_classifier import load_active_model
        db = BiotechDatabase()
        info = load_active_model(db)
        if info is None:
            return {"active_model": None, "note": "No V3 model trained yet. POST /train-v3-model to create one."}
        return {"active_model": info}
    except Exception as e:
        logger.exception("v3_model_info failed")
        raise HTTPException(500, f"v3_model_info error: {e}")


# ============================================================
# Price-window backfill — concurrent background workers
# ============================================================

import threading as _threading

_backfill_price_state: dict = {
    "running": False,
    "events_processed": 0,
    "events_created": 0,
    "events_failed": 0,
    "events_skipped": 0,
    "started_at": None,
    "last_error": None,
}
_backfill_price_in_progress: set = set()
_backfill_price_lock = _threading.Lock()
_BACKFILL_PRICE_WORKERS = 8


def _claim_due_catalysts(limit: int, min_age_days: int):
    """Atomically claim N due catalysts in the in-process set so concurrent
    workers don't double-process the same rows. Returns list of catalyst dicts.
    """
    from datetime import date, timedelta
    from services.database import BiotechDatabase
    cutoff = (date.today() - timedelta(days=min_age_days)).isoformat()
    with _backfill_price_lock:
        excluded = list(_backfill_price_in_progress)
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT cu.id, cu.ticker, cu.catalyst_type, cu.catalyst_date,
                       cu.drug_name, cu.indication, cu.confidence_score,
                       cu.description, cu.phase
                FROM catalyst_universe cu
                LEFT JOIN post_catalyst_outcomes pco
                    ON pco.catalyst_id = cu.id
                    OR (pco.ticker = cu.ticker
                        AND pco.catalyst_type = cu.catalyst_type
                        AND pco.catalyst_date::text = cu.catalyst_date::text)
                WHERE cu.catalyst_date::text <= %s
                  AND cu.catalyst_date IS NOT NULL
                  AND cu.catalyst_date::text != ''
                  AND pco.id IS NULL
                  AND (cu.status IS NULL OR cu.status NOT IN ('superseded', 'invalid'))
                  AND cu.id != ALL(%s)
                ORDER BY cu.catalyst_date ASC
                LIMIT %s
            """, (cutoff, excluded, limit))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        catalysts = [dict(zip(cols, r)) for r in rows]
        for c in catalysts:
            _backfill_price_in_progress.add(c["id"])
    return catalysts


def _release_catalysts(ids):
    with _backfill_price_lock:
        for i in ids:
            _backfill_price_in_progress.discard(i)


@router.post("/post-catalyst/backfill-price-start")
async def backfill_price_start(min_age_days: int = 7, claim_size: int = 10):
    """Start a background task that runs price-window backfill across many
    concurrent workers. Each worker claims a small batch (claim_size) atomically
    via an in-process exclusion set, processes them, then claims the next batch.
    Returns immediately; poll /backfill-price-status for progress.
    """
    if _backfill_price_state["running"]:
        raise HTTPException(409, "backfill-price already running")

    import asyncio
    from services.post_catalyst_tracker import backfill_one

    _backfill_price_state.update({
        "running": True,
        "events_processed": 0,
        "events_created": 0,
        "events_failed": 0,
        "events_skipped": 0,
        "started_at": datetime.utcnow().isoformat(),
        "last_error": None,
    })

    async def _worker():
        while True:
            catalysts = await asyncio.to_thread(_claim_due_catalysts, claim_size, min_age_days)
            if not catalysts:
                break
            ids = [c["id"] for c in catalysts]
            try:
                for c in catalysts:
                    result = await asyncio.to_thread(backfill_one, c)
                    _backfill_price_state["events_processed"] += 1
                    status = (result or {}).get("status")
                    if status == "created":
                        _backfill_price_state["events_created"] += 1
                    elif status == "skipped":
                        _backfill_price_state["events_skipped"] += 1
                    else:
                        _backfill_price_state["events_failed"] += 1
            finally:
                await asyncio.to_thread(_release_catalysts, ids)
            await asyncio.sleep(0.05)

    async def _run():
        try:
            await asyncio.gather(*[_worker() for _ in range(_BACKFILL_PRICE_WORKERS)])
        except Exception as e:
            logger.exception(f"[backfill-price-all] error: {e}")
            _backfill_price_state["last_error"] = str(e)[:200]
        finally:
            _backfill_price_state["running"] = False

    asyncio.create_task(_run())
    return {"ok": True, "status_url": "/admin/post-catalyst/backfill-price-status",
            "workers": _BACKFILL_PRICE_WORKERS}


@router.get("/post-catalyst/backfill-price-status")
async def backfill_price_status():
    """Live progress for the background price-window backfill task."""
    state = _backfill_price_state.copy()
    state["in_progress_count"] = len(_backfill_price_in_progress)
    return state


@router.post("/post-catalyst/edgar-backfill-debug-ticker")
async def edgar_backfill_debug_ticker(
    ticker: str,
    start_year: int = 2024,
    end_year: int = 2025,
):
    """Single-ticker EDGAR test with skip-reason breakdown. Helps debug
    why filings are being filtered out.

    Synchronous (no background task). Use for troubleshooting only —
    don't call on a large set or it'll timeout.
    """
    try:
        from services.database import BiotechDatabase
        from services.edgar_backfill import (
            _fetch_sec_ticker_map, fetch_filings_for_cik,
            fetch_filing_text_excerpt, looks_like_clinical_catalyst,
            CLINICAL_KEYWORDS_RE, RELEVANT_ITEM_CODES_RE,
            edgar_backfill_for_cik,
        )

        db = BiotechDatabase()
        ticker_map = _fetch_sec_ticker_map()
        entry = ticker_map.get(ticker.upper())
        if not entry:
            return {"error": f"ticker {ticker} not in SEC company_tickers.json", "found_in_map": False}

        cik_padded = entry["cik_padded"]
        cik = entry["cik"]
        company = entry["title"]

        # Step 1: list filings (no body fetch)
        filings = fetch_filings_for_cik(cik_padded, start_year, end_year)
        result = {
            "ticker": ticker,
            "cik": cik,
            "cik_padded": cik_padded,
            "company": company,
            "year_range": f"{start_year}-{end_year}",
            "filings_listed": len(filings),
            "filings_sample": [],
            "step_results": {},
        }
        # Show first 3 filings raw
        for f in filings[:3]:
            result["filings_sample"].append({
                "filing_date": str(f.get("filing_date")),
                "accession": f.get("accession_no"),
                "items": f.get("items", "")[:80],
                "primary_doc": (f.get("primary_doc") or "")[:80],
            })

        # Run actual scraper for diagnostic counters
        # Use temp run_id=0 since we're not creating a run record
        counts = edgar_backfill_for_cik(db, cik_padded, ticker, start_year, end_year, run_id=0)
        result["step_results"] = counts

        # Show a couple sample excerpts (we have to re-fetch since the scraper
        # already inserted what passed; but we want to see what FAILED)
        for f in filings[:3]:
            excerpt = fetch_filing_text_excerpt(f, max_chars=400)
            has_keywords = looks_like_clinical_catalyst(excerpt) if excerpt else False
            items = f.get("items") or ""
            items_match = bool(RELEVANT_ITEM_CODES_RE.search(items)) if items else None  # None = no items field
            result["filings_sample"][filings.index(f)].update({
                "excerpt_first_400": (excerpt or "")[:400] + ("..." if excerpt and len(excerpt) > 400 else ""),
                "has_clinical_keywords": has_keywords,
                "items_field_present": bool(items),
                "items_matches_relevant": items_match,
            })

        return result
    except Exception as e:
        logger.exception("edgar_backfill_debug_ticker failed")
        raise HTTPException(500, f"edgar_backfill_debug_ticker error: {e}")


@router.get("/post-catalyst/backfill-staging-sample")
async def backfill_staging_sample(
    status: str = "rejected", source: str = "edgar", limit: int = 5,
):
    """Peek at sample rows in catalyst_backfill_staging by status, for debugging."""
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, ticker, filing_date, raw_title, reject_reason, normalized_json
                FROM catalyst_backfill_staging
                WHERE source = %s AND status = %s
                ORDER BY filing_date DESC
                LIMIT %s
            """, (source, status, limit))
            rows = cur.fetchall()
        out = []
        for r in rows:
            sid, ticker, fd, title, reject, nj = r
            normalized = nj if isinstance(nj, dict) else (json.loads(nj) if nj else None)
            out.append({
                "id": sid,
                "ticker": ticker,
                "filing_date": str(fd) if fd else None,
                "raw_title": (title or "")[:200],
                "reject_reason": (reject or "")[:300],
                "normalized": normalized,
            })
        return {"rows": out}
    except Exception as e:
        logger.exception("backfill_staging_sample failed")
        raise HTTPException(500, f"backfill_staging_sample error: {e}")


@router.post("/post-catalyst/backfill-reset-status")
async def backfill_reset_status(
    from_status: str = "unclear",
    source: str = "edgar",
    only_llm_failures: bool = True,
):
    """Reset rows from a given status back to 'pending' so they re-process.
    Useful after fixing a bug or softening the prompt.

    only_llm_failures=True (default for unclear): only resets rows whose
    reject_reason indicates an LLM call failure (vs legitimate low-confidence
    classification). For 'rejected' rows, set only_llm_failures=False to
    re-evaluate them.
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            if only_llm_failures and from_status == "unclear":
                cur.execute("""
                    UPDATE catalyst_backfill_staging
                    SET status = 'pending',
                        processed_at = NULL,
                        reject_reason = NULL,
                        normalized_json = NULL
                    WHERE source = %s AND status = %s
                      AND (reject_reason ILIKE '%%LLM call%%'
                           OR reject_reason ILIKE '%%LLM call failed%%'
                           OR reject_reason ILIKE '%%503%%'
                           OR reject_reason ILIKE '%%UNAVAILABLE%%'
                           OR reject_reason ILIKE '%%timeout%%')
                """, (source, from_status))
            else:
                cur.execute("""
                    UPDATE catalyst_backfill_staging
                    SET status = 'pending',
                        processed_at = NULL,
                        reject_reason = NULL,
                        normalized_json = NULL
                    WHERE source = %s AND status = %s
                """, (source, from_status))
            n = cur.rowcount
            conn.commit()
        return {"reset_count": n, "source": source, "from_status": from_status}
    except Exception as e:
        logger.exception("backfill_reset_status failed")
        raise HTTPException(500, f"backfill_reset_status error: {e}")


@router.post("/post-catalyst/backfill-wipe-source")
async def backfill_wipe_source(
    source: str,
    confirm: str = "",
):
    """Delete ALL staging rows for a given source. Destructive — requires
    confirm='yes'. Useful when a scraper bug invalidated a batch and we
    want a clean re-scrape.

    Does NOT delete from catalyst_universe (those are already promoted).
    """
    if confirm != "yes":
        raise HTTPException(400, "must pass confirm=yes")
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                DELETE FROM catalyst_backfill_staging
                WHERE source = %s
            """, (source,))
            n = cur.rowcount
            conn.commit()
        return {"deleted": n, "source": source}
    except Exception as e:
        logger.exception("backfill_wipe_source failed")
        raise HTTPException(500, f"backfill_wipe_source error: {e}")


@router.post("/post-catalyst/backfill-debug-insert")
async def backfill_debug_insert():
    """Debug: take ONE staging row marked unclear with valid normalized_json
    and try the INSERT manually, returning the actual exception text."""
    try:
        from services.database import BiotechDatabase
        from services.backfill_normalizer import insert_into_catalyst_universe
        db = BiotechDatabase()

        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, ticker, filing_date, catalyst_date, normalized_json
                FROM catalyst_backfill_staging
                WHERE status = 'unclear'
                  AND normalized_json IS NOT NULL
                  AND normalized_json->>'is_clinical_catalyst' = 'true'
                ORDER BY filing_date DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            if not row:
                return {"error": "no qualifying unclear row found"}
            sid, ticker, filing_date, cat_date, normalized = row

        # Show what we're about to attempt
        out = {
            "row_id": sid, "ticker": ticker,
            "filing_date": str(filing_date) if filing_date else None,
            "catalyst_date_staging": str(cat_date) if cat_date else None,
            "normalized": normalized if isinstance(normalized, dict) else json.loads(normalized) if normalized else None,
        }

        # Manual INSERT attempt with raw exception
        n = out['normalized'] or {}
        try:
            with db.get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO catalyst_universe (
                        ticker, catalyst_type, catalyst_date,
                        date_precision, drug_name, indication,
                        confidence, source, status,
                        created_at, last_updated
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        'edgar_backfill_test', 'inactive', now(), now()
                    )
                    RETURNING id
                """, (
                    ticker,
                    n.get('catalyst_type'),
                    n.get('extracted_catalyst_date') or cat_date,
                    n.get('date_precision') or 'day',
                    n.get('drug_name'),
                    n.get('indication'),
                    float(n.get('confidence') or 0.5),
                ))
                new_id = cur.fetchone()[0]
                conn.commit()
                # Now delete the test row
                cur.execute("DELETE FROM catalyst_universe WHERE id = %s AND source = 'edgar_backfill_test'", (new_id,))
                conn.commit()
                out['raw_insert'] = {'success': True, 'new_id': new_id, 'cleaned_up': True}
        except Exception as e:
            out['raw_insert'] = {'success': False, 'error': str(e)[:600], 'error_type': type(e).__name__}

        # Now try via the helper (for comparison)
        try:
            staging_dict = {
                "ticker": ticker, "catalyst_date": cat_date,
            }
            new_id = insert_into_catalyst_universe(
                db=db, normalized=n, staging=staging_dict,
            )
            out['helper_insert'] = {'success': new_id is not None, 'new_id': new_id}
            if new_id:
                with db.get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("DELETE FROM catalyst_universe WHERE id = %s AND source = 'edgar_backfill'", (new_id,))
                    conn.commit()
        except Exception as e:
            out['helper_insert'] = {'success': False, 'error': str(e)[:600], 'error_type': type(e).__name__}

        return out
    except Exception as e:
        logger.exception("backfill_debug_insert failed")
        raise HTTPException(500, f"backfill_debug_insert error: {e}")


@router.post("/post-catalyst/backfill-reset-unclear-to-pending")
async def backfill_reset_unclear_to_pending(
    source: str = "edgar",
    only_with_normalized_json: bool = True,
    confirm: str = "",
):
    """Reset status='unclear' rows back to 'pending' so they can be
    retried after normalizer bug fixes. By default, only resets rows
    that have valid normalized_json (i.e. LLM succeeded but INSERT failed
    or downstream had a bug) — these are the ones most likely to succeed
    on retry. Skips runaway-newline-truncation rows since those will
    fail again unless we re-call the LLM.

    To also reset rows without normalized_json (genuine LLM failures),
    pass only_with_normalized_json=false.

    Requires confirm=yes.
    """
    if confirm != "yes":
        raise HTTPException(400, "must pass confirm=yes")
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            if only_with_normalized_json:
                cur.execute("""
                    UPDATE catalyst_backfill_staging
                    SET status = 'pending',
                        reject_reason = NULL,
                        processed_at = NULL
                    WHERE source = %s
                      AND status = 'unclear'
                      AND normalized_json IS NOT NULL
                """, (source,))
            else:
                cur.execute("""
                    UPDATE catalyst_backfill_staging
                    SET status = 'pending',
                        reject_reason = NULL,
                        processed_at = NULL,
                        normalized_json = NULL
                    WHERE source = %s
                      AND status = 'unclear'
                """, (source,))
            n = cur.rowcount
            conn.commit()
        return {"reset": n, "source": source, "only_with_normalized_json": only_with_normalized_json}
    except Exception as e:
        logger.exception("backfill_reset_unclear_to_pending failed")
        raise HTTPException(500, f"reset error: {e}")


@router.get("/_deploy_sentinel_v1")
async def _deploy_sentinel_v1():
    """Sentinel endpoint to verify which commit is live."""
    return {"sentinel": "v1", "commit": "deploy-marker-2026-04-28-15-50"}
