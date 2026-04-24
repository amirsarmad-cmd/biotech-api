"""
/stocks routes — universe, detail, news, social, analyst.
"""
import logging
from typing import Optional
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


def _get_live_price(ticker: str) -> Optional[float]:
    """Best-effort price fetch via yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        for k in ("currentPrice", "regularMarketPrice", "previousClose"):
            v = info.get(k)
            if v: return float(v)
        # Fallback: fast_info
        try:
            fi = t.fast_info
            p = fi.get("last_price") if hasattr(fi, "get") else getattr(fi, "last_price", None)
            if p: return float(p)
        except Exception: pass
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

    return {
        "count": len(filtered),
        "universe_size": len(rows),
        "high_prob_count": sum(1 for r in rows if (r.get("probability") or 0) >= 0.6),
        "stocks": filtered[:limit],
    }


@router.get("/{ticker}")
async def get_stock_detail(ticker: str):
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

    # NPV via LLM provider — may take up to 30s
    npv_result = None
    try:
        from services.npv_model import compute_npv
        inputs = dict(
            ticker=ticker,
            catalyst_type=primary.get("catalyst_type") or "FDA Decision",
            peak_sales_b=3.0,
            multiple=3.5,
            p_commercial=0.5 if (primary.get("probability") or 0.5) < 0.6 else 0.7,
            market_cap_m=float(primary.get("market_cap") or 0),
        )
        npv_result = compute_npv(**inputs)
    except Exception as e:
        logger.warning(f"NPV compute failed for {ticker}: {e}")
        npv_result = {"error": str(e)[:300]}

    return {
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
    }


@router.get("/{ticker}/news")
async def get_stock_news(ticker: str, limit: int = Query(20, ge=1, le=50)):
    ticker = ticker.upper().strip()
    try:
        from services.fetcher_news import fetch_all_sources
        sources = fetch_all_sources(ticker, days_back=30)
        return {"ticker": ticker, "count": len(sources), "articles": sources[:limit]}
    except Exception as e:
        logger.exception(f"news fetch failed for {ticker}")
        raise HTTPException(500, f"news fetch error: {e}")


@router.get("/{ticker}/social")
async def get_stock_social(ticker: str):
    ticker = ticker.upper().strip()
    try:
        from services.social_sources import fetch_social_all
        data = fetch_social_all(ticker)
        return {"ticker": ticker, "data": data}
    except Exception as e:
        logger.warning(f"social fetch failed for {ticker}: {e}")
        return {"ticker": ticker, "data": None, "error": str(e)[:200]}


@router.get("/{ticker}/analyst")
async def get_stock_analyst(ticker: str):
    ticker = ticker.upper().strip()
    try:
        from services.authenticated_sources import fetch_analyst_data_all_merged
        data = fetch_analyst_data_all_merged(ticker)
        return {"ticker": ticker, "data": data}
    except Exception as e:
        logger.warning(f"analyst fetch failed for {ticker}: {e}")
        return {"ticker": ticker, "data": None, "error": str(e)[:200]}
