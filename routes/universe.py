"""
/universe — read endpoints for catalyst_universe table (Phase B v2).
Separate from old /stocks routes which still serve screener_stocks for backward compat.
"""
import logging
import os
from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter()


def _pg_conn():
    import psycopg2
    url = os.getenv("DATABASE_URL", "")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return psycopg2.connect(url)


def _row_to_dict(row, columns):
    """Convert psycopg2 row tuple → dict, normalizing dates/decimals."""
    import math
    from decimal import Decimal
    out = {}
    for col, val in zip(columns, row):
        if val is None:
            out[col] = None
        elif isinstance(val, Decimal):
            f = float(val)
            out[col] = None if (math.isnan(f) or math.isinf(f)) else f
        elif hasattr(val, "isoformat"):
            out[col] = val.isoformat()
        else:
            out[col] = val
    return out


@router.get("/catalysts")
async def list_catalysts(
    limit: int = Query(500, ge=1, le=5000),
    status: str = Query("active", description="'active' | 'all' | 'superseded'"),
    catalyst_type: Optional[str] = None,
    phase: Optional[str] = None,
    days_ahead: Optional[int] = Query(None, description="Only catalysts within N days from today"),
    ticker: Optional[str] = None,
):
    """Read catalyst_universe with filters."""
    where = []
    params: list = []
    
    if status != "all":
        where.append("status = %s")
        params.append(status)
    if catalyst_type:
        where.append("catalyst_type = %s")
        params.append(catalyst_type)
    if phase:
        where.append("phase = %s")
        params.append(phase)
    if days_ahead is not None:
        where.append("catalyst_date >= CURRENT_DATE AND catalyst_date <= CURRENT_DATE + INTERVAL '%s days'" % int(days_ahead))
    if ticker:
        where.append("ticker = %s")
        params.append(ticker.upper().strip())
    
    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    
    cols = [
        "id", "ticker", "company_name", "catalyst_type", "catalyst_date", "date_precision",
        "description", "drug_name", "canonical_drug_name", "indication", "phase",
        "source", "source_url", "confidence_score", "verified", "status",
        "superseded_by", "superseded_at", "last_updated", "created_at"
    ]
    
    sql = f"""
        SELECT {','.join(cols)} FROM catalyst_universe
        {where_clause}
        ORDER BY catalyst_date ASC NULLS LAST, ticker
        LIMIT %s
    """
    params.append(limit)
    
    try:
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                items = [_row_to_dict(r, cols) for r in rows]
                # total count for the filter
                count_sql = f"SELECT count(*) FROM catalyst_universe {where_clause}"
                cur.execute(count_sql, params[:-1])
                total = cur.fetchone()[0]
                return {"count": len(items), "total_matching": total, "items": items}
    except Exception as e:
        logger.exception("list_catalysts")
        raise HTTPException(500, f"list error: {e}")


@router.get("/catalysts/{catalyst_id}")
async def get_catalyst(catalyst_id: int):
    cols = [
        "id", "ticker", "company_name", "catalyst_type", "catalyst_date", "date_precision",
        "description", "drug_name", "canonical_drug_name", "indication", "phase",
        "source", "source_url", "confidence_score", "verified", "status",
        "superseded_by", "superseded_at", "last_updated", "created_at"
    ]
    try:
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT {','.join(cols)} FROM catalyst_universe WHERE id=%s", (catalyst_id,))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(404, f"catalyst {catalyst_id} not found")
                return _row_to_dict(row, cols)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"get error: {e}")


@router.get("/stats")
async def universe_stats():
    """Universe-wide counts grouped by status / type / phase."""
    try:
        with _pg_conn() as conn:
            with conn.cursor() as cur:
                stats = {}
                cur.execute("SELECT count(*) FROM catalyst_universe")
                stats["total_catalysts"] = cur.fetchone()[0]
                
                cur.execute("SELECT status, count(*) FROM catalyst_universe GROUP BY status")
                stats["by_status"] = {r[0]: r[1] for r in cur.fetchall()}
                
                cur.execute("""
                    SELECT catalyst_type, count(*) FROM catalyst_universe
                    WHERE status='active'
                    GROUP BY catalyst_type ORDER BY count(*) DESC
                """)
                stats["by_type"] = {r[0]: r[1] for r in cur.fetchall()}
                
                cur.execute("""
                    SELECT source, count(*) FROM catalyst_universe
                    WHERE status='active'
                    GROUP BY source ORDER BY count(*) DESC
                """)
                stats["by_source"] = {r[0]: r[1] for r in cur.fetchall()}
                
                cur.execute("SELECT count(DISTINCT ticker) FROM catalyst_universe WHERE status='active'")
                stats["unique_tickers"] = cur.fetchone()[0]
                
                cur.execute("""
                    SELECT count(*) FROM catalyst_universe
                    WHERE status='active' AND catalyst_date >= CURRENT_DATE
                    AND catalyst_date <= CURRENT_DATE + INTERVAL '30 days'
                """)
                stats["upcoming_30d"] = cur.fetchone()[0]
                
                cur.execute("""
                    SELECT count(*) FROM catalyst_universe
                    WHERE status='active' AND catalyst_date >= CURRENT_DATE
                    AND catalyst_date <= CURRENT_DATE + INTERVAL '180 days'
                """)
                stats["upcoming_6mo"] = cur.fetchone()[0]
                
                return stats
    except Exception as e:
        logger.exception("universe_stats")
        raise HTTPException(500, f"stats error: {e}")
