"""
/shortlist — CRUD for the user's shortlist (watchlist).
Backed by screener_shortlist table; full add/remove/list with stats.
"""
import logging, math
from typing import Any, List, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.database import BiotechDatabase

logger = logging.getLogger(__name__)
router = APIRouter()

_db = None
def db():
    global _db
    if _db is None:
        _db = BiotechDatabase()
    return _db


def _to_jsonable(obj):
    """Strip NaN/Infinity, convert numpy/pandas types."""
    try: import numpy as _np
    except ImportError: _np = None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj): return None
    if _np is not None:
        if isinstance(obj, (_np.integer,)): return int(obj)
        if isinstance(obj, (_np.floating,)):
            v = float(obj)
            if math.isnan(v) or math.isinf(v): return None
            return v
        if isinstance(obj, (_np.ndarray,)): return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict): return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_jsonable(x) for x in obj]
    if isinstance(obj, (bytes, bytearray)): return obj.decode("utf-8", errors="replace")
    if hasattr(obj, "isoformat"):
        try: return obj.isoformat()
        except Exception: pass
    if hasattr(obj, "item"):
        try: return obj.item()
        except Exception: pass
    try:
        import json as _j
        _j.dumps(obj); return obj
    except (TypeError, ValueError):
        return str(obj)


class AddRequest(BaseModel):
    ticker: str
    company_name: str = ""
    initial_price: float = 0
    catalyst_type: str = ""
    catalyst_date: str = ""
    initial_probability: float = 0
    initial_score: float = 0
    notes: str = ""


@router.get("")
async def list_shortlist():
    """List all shortlisted tickers, enriched with current catalyst data from the stocks universe."""
    try:
        rows = db().get_shortlist() or []
        enriched = []
        for r in rows:
            tkr = (r.get("ticker") or "").upper()
            try:
                stocks = db().get_stock(tkr) or []
                if stocks:
                    primary = sorted(stocks, key=lambda s: s.get("probability") or 0, reverse=True)[0]
                    r["catalyst_type"] = primary.get("catalyst_type")
                    r["catalyst_date"] = primary.get("catalyst_date")
                    r["current_probability"] = primary.get("probability")
                    r["current_score"] = primary.get("overall_score")
                    r["market_cap"] = primary.get("market_cap")
                    r["industry"] = primary.get("industry")
                    # Update company_name if it's still placeholder
                    if not r.get("company_name") or r.get("company_name") == tkr:
                        r["company_name"] = primary.get("company_name") or tkr
            except Exception as e:
                logger.warning(f"enrich {tkr}: {e}")
            enriched.append(r)
        return _to_jsonable({
            "count": len(enriched),
            "items": enriched,
        })
    except Exception as e:
        logger.exception("list_shortlist")
        raise HTTPException(500, f"shortlist error: {e}")


@router.post("")
async def add_to_shortlist(req: AddRequest):
    """Add a ticker to the shortlist. Idempotent — duplicate adds return existing record."""
    try:
        # Look up live data if not provided
        if not req.company_name or not req.catalyst_type:
            stocks = db().get_stock(req.ticker.upper().strip()) or []
            if stocks:
                primary = sorted(stocks, key=lambda r: r.get("probability") or 0, reverse=True)[0]
                req.company_name = req.company_name or primary.get("company_name") or ""
                req.catalyst_type = req.catalyst_type or primary.get("catalyst_type") or ""
                req.catalyst_date = req.catalyst_date or primary.get("catalyst_date") or ""
                req.initial_probability = req.initial_probability or float(primary.get("probability") or 0)
                req.initial_score = req.initial_score or float(primary.get("overall_score") or 0)

        ok = db().add_to_shortlist(
            ticker=req.ticker.upper().strip(),
            company_name=req.company_name,
            initial_price=req.initial_price,
            initial_score=req.initial_score,
            initial_sentiment=0,
        )
        if not ok:
            raise HTTPException(500, "Failed to add to shortlist")
        return {"ticker": req.ticker.upper(), "added": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("add_to_shortlist")
        raise HTTPException(500, f"add error: {e}")


@router.delete("/{ticker}")
async def remove_from_shortlist(ticker: str):
    """Remove a ticker from the shortlist."""
    ticker = ticker.upper().strip()
    try:
        ok = db().remove_from_shortlist(ticker)
        if not ok:
            raise HTTPException(404, f"Ticker {ticker} not in shortlist")
        return {"ticker": ticker, "removed": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("remove_from_shortlist")
        raise HTTPException(500, f"remove error: {e}")


@router.get("/{ticker}/check")
async def check_shortlist(ticker: str):
    """Whether the ticker is on the shortlist."""
    ticker = ticker.upper().strip()
    try:
        return {"ticker": ticker, "shortlisted": db().is_shortlisted(ticker)}
    except Exception as e:
        raise HTTPException(500, f"check error: {e}")


@router.get("/stats")
async def shortlist_stats():
    """Quick stats on the shortlist."""
    try:
        stats = db().get_stats() or {}
        return _to_jsonable(stats)
    except Exception as e:
        raise HTTPException(500, f"stats error: {e}")
