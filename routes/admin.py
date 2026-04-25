"""
/admin — universe refresh, DB stats, migration tools.
"""
import logging, os
from datetime import datetime
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
                        OR ss.description NOT LIKE 'yfinance: no data%'
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
                        # screener_stocks UNIQUE on (ticker, catalyst_type, catalyst_date)
                        # Use empty string defaults for the catalyst_type/date if needed
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
                            r["market_cap_m"] / 1000,  # screener_stocks.market_cap is in $B
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
