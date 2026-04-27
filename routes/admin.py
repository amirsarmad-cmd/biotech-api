"""
/admin — universe refresh, DB stats, migration tools.
"""
import logging, os, json
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException
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

    Returns the inserted row ID.
    """
    try:
        from services.database import BiotechDatabase
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO catalyst_universe (
                    ticker, company_name, catalyst_type, catalyst_date,
                    description, drug_name, indication, phase,
                    source, source_url, confidence_score, verified, status,
                    is_manual_override, last_updated, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    'manual', %s, %s, TRUE, 'active',
                    TRUE, NOW(), NOW()
                )
                ON CONFLICT (ticker, catalyst_type, catalyst_date, drug_name)
                DO UPDATE SET
                    description = EXCLUDED.description,
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
                None,  # company_name will be filled by trigger or next seeder run
                payload.catalyst_type,
                payload.catalyst_date,
                payload.description,
                payload.drug_name,
                payload.indication,
                payload.phase,
                payload.source_url,
                payload.probability,
            ))
            new_id = cur.fetchone()[0]
            # Log the manual addition into the ingestion log
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
                SET version_num = '011_manual_override_and_ingestion_log'
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
