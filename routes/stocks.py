"""
/stocks routes — universe, detail, news, social, analyst.
"""
import logging
import json as _json
from typing import Optional, Any
from fastapi import APIRouter, HTTPException, Query

from services.database import BiotechDatabase

logger = logging.getLogger(__name__)
router = APIRouter()

_db = None
def db():
    global _db
    if _db is None:
        _db = BiotechDatabase()
    return _db


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas/other non-JSON types to plain Python."""
    try:
        import numpy as _np
    except ImportError: _np = None
    if _np is not None:
        if isinstance(obj, (_np.integer,)): return int(obj)
        if isinstance(obj, (_np.floating,)): return float(obj)
        if isinstance(obj, (_np.ndarray,)): return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict): return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_jsonable(x) for x in obj]
    if isinstance(obj, (bytes, bytearray)): return obj.decode("utf-8", errors="replace")
    if hasattr(obj, "item"): 
        try: return obj.item()
        except Exception: pass
    # Fallback — if it's not JSON-serializable, stringify
    try:
        _json.dumps(obj); return obj
    except (TypeError, ValueError):
        return str(obj)


def _get_live_price(ticker: str) -> Optional[float]:
    """Best-effort price fetch via yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        try:
            fi = t.fast_info
            p = getattr(fi, "last_price", None) or (fi.get("last_price") if hasattr(fi, "get") else None)
            if p: return float(p)
        except Exception: pass
        info = t.info
        for k in ("currentPrice", "regularMarketPrice", "previousClose"):
            v = info.get(k)
            if v: return float(v)
    except Exception as e:
        logger.warning(f"price {ticker}: {e}")
    return None


@router.get("")
async def list_stocks(
    high_prob_only: bool = Query(False),
    min_probability: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(200, ge=1, le=1000),
    sort: str = Query("overall_score", pattern="^(overall_score|probability|market_cap|ticker)$"),
):
    try:
        rows = db().get_all_stocks()
    except Exception as e:
        logger.exception("get_all_stocks failed")
        raise HTTPException(500, f"DB error: {e}")

    filtered = []
    for r in rows:
        p = r.get("probability") or 0
        if high_prob_only and p < 0.6: continue
        if p < min_probability: continue
        filtered.append(r)

    reverse = sort != "ticker"
    filtered.sort(key=lambda x: (x.get(sort) or 0) if sort != "ticker" else x.get("ticker",""), reverse=reverse)

    return _to_jsonable({
        "count": len(filtered),
        "universe_size": len(rows),
        "high_prob_count": sum(1 for r in rows if (r.get("probability") or 0) >= 0.6),
        "stocks": filtered[:limit],
    })


@router.get("/{ticker}")
async def get_stock_detail(ticker: str, with_npv: bool = Query(True)):
    """
    Detail page data. NPV is computed via LLM and takes 10-30s.
    Pass with_npv=false to skip the NPV call for faster fetches.
    """
    ticker = ticker.upper().strip()
    try:
        rows = db().get_stock(ticker)
    except Exception as e:
        logger.exception(f"get_stock({ticker}) failed")
        raise HTTPException(500, f"DB error: {e}")

    if not rows:
        raise HTTPException(404, f"Ticker {ticker} not found in universe")

    rows_sorted = sorted(rows, key=lambda r: r.get("probability") or 0, reverse=True)
    primary = rows_sorted[0]

    current_price = _get_live_price(ticker)

    # NPV (optional + error-wrapped)
    npv_result = None
    if with_npv:
        try:
            from services.npv_model import compute_npv_estimate, estimate_drug_economics, get_baseline_price
            economics = estimate_drug_economics(
                ticker=ticker,
                company_name=primary.get("company_name", ticker),
                catalyst_type=primary.get("catalyst_type") or "FDA Decision",
                catalyst_date=primary.get("catalyst_date", ""),
                description=primary.get("description", ""),
                market_cap_m=float(primary.get("market_cap") or 0),
            )
            baseline_price = get_baseline_price(ticker) or current_price or 50.0
            price_for_calc = current_price or baseline_price
            npv_result = compute_npv_estimate(
                ticker=ticker,
                current_price=price_for_calc,
                market_cap_m=float(primary.get("market_cap") or 0),
                p_approval=float(primary.get("probability") or 0.5),
                economics=economics,
                baseline_price=baseline_price,
                info={
                    "catalyst_type": primary.get("catalyst_type") or "",
                    "catalyst_date": primary.get("catalyst_date") or "",
                    "description": primary.get("description") or "",
                },
            )
        except Exception as e:
            logger.warning(f"NPV compute failed for {ticker}: {e}")
            npv_result = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    return _to_jsonable({
        "ticker": ticker,
        "company_name": primary.get("company_name"),
        "industry": primary.get("industry"),
        "current_price": current_price,
        "market_cap_m": primary.get("market_cap"),
        "primary_catalyst": {
            "type": primary.get("catalyst_type"),
            "date": primary.get("catalyst_date"),
            "probability": primary.get("probability"),
            "description": primary.get("description"),
        },
        "all_catalysts": [{
            "type": r.get("catalyst_type"),
            "date": r.get("catalyst_date"),
            "probability": r.get("probability"),
            "description": r.get("description"),
        } for r in rows_sorted],
        "npv": npv_result,
        "scores": {
            "overall": primary.get("overall_score"),
            "sentiment": primary.get("sentiment_score"),
            "news_count": primary.get("news_count"),
        },
        "last_updated": primary.get("last_updated"),
    })


@router.get("/{ticker}/news")
async def get_stock_news(ticker: str, limit: int = Query(20, ge=1, le=50)):
    ticker = ticker.upper().strip()
    try:
        from services.fetcher_news import fetch_all_sources
        sources = fetch_all_sources(ticker, days_back=30)
        return _to_jsonable({"ticker": ticker, "count": len(sources), "articles": sources[:limit]})
    except Exception as e:
        logger.exception(f"news fetch failed for {ticker}")
        raise HTTPException(500, f"news fetch error: {e}")


@router.get("/{ticker}/social")
async def get_stock_social(ticker: str):
    ticker = ticker.upper().strip()
    try:
        from services.social_sources import fetch_social_all
        data = fetch_social_all(ticker)
        return _to_jsonable({"ticker": ticker, "data": data})
    except Exception as e:
        logger.warning(f"social fetch failed for {ticker}: {e}")
        return {"ticker": ticker, "data": None, "error": str(e)[:200]}


@router.get("/{ticker}/analyst")
async def get_stock_analyst(ticker: str):
    ticker = ticker.upper().strip()
    try:
        from services.authenticated_sources import fetch_analyst_data_all_merged
        data = fetch_analyst_data_all_merged(ticker)
        return _to_jsonable({"ticker": ticker, "data": data})
    except Exception as e:
        logger.warning(f"analyst fetch failed for {ticker}: {e}")
        return {"ticker": ticker, "data": None, "error": str(e)[:200]}
