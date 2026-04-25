"""
/admin — universe refresh, DB stats, migration tools.
"""
import logging, os
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
    """Run the Phase B universe seeder. LLM cost gated by LLM_ENABLED env var."""
    try:
        from services.universe_seeder import run_universe_seed
        return run_universe_seed(max_tickers=req.max_tickers)
    except Exception as e:
        logger.exception("v2-seed")
        raise HTTPException(500, f"seed error: {e}")


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
