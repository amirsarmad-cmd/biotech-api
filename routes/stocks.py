"""
/stocks routes — universe, detail, news, social, analyst.
Replaces the Streamlit screener home + detail view.
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


@router.get("")
async def list_stocks(
    high_prob_only: bool = Query(False, description="Filter to probability >= 0.6"),
    min_probability: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(200, ge=1, le=1000),
    sort: str = Query("overall_score", pattern="^(overall_score|probability|market_cap|ticker)$"),
):
    """Returns the universe of tracked biotech stocks with their next catalyst."""
    try:
        rows = db().get_all_stocks()
    except Exception as e:
        logger.exception("get_all_stocks failed")
        raise HTTPException(500, f"DB error: {e}")

    filtered = []
    for r in rows:
        p = r.get("probability") or 0
        if high_prob_only and p < 0.6:
            continue
        if p < min_probability:
            continue
        filtered.append(r)

    reverse = sort != "ticker"
    filtered.sort(key=lambda x: (x.get(sort) or 0) if sort != "ticker" else x.get("ticker", ""), reverse=reverse)

    return {
        "count": len(filtered),
        "universe_size": len(rows),
        "high_prob_count": sum(1 for r in rows if (r.get("probability") or 0) >= 0.6),
        "stocks": filtered[:limit],
    }


@router.get("/{ticker}")
async def get_stock_detail(ticker: str):
    """
    Full detail for a ticker: price estimates, NPV inputs, all catalysts.
    Heavy data (news, social, analyst) is on sub-endpoints.
    """
    ticker = ticker.upper().strip()
    try:
        rows = db().get_stock(ticker)
    except Exception as e:
        logger.exception(f"get_stock({ticker}) failed")
        raise HTTPException(500, f"DB error: {e}")

    if not rows:
        raise HTTPException(404, f"Ticker {ticker} not found in universe")

    # Use the primary catalyst (highest probability)
    rows_sorted = sorted(rows, key=lambda r: r.get("probability") or 0, reverse=True)
    primary = rows_sorted[0]

    # Live price via fetcher (yfinance)
    current_price = None
    try:
        from services.fetcher import create_fetcher
        fetcher = create_fetcher()
        p = fetcher.get_current_price(ticker) if hasattr(fetcher, "get_current_price") else None
        current_price = float(p) if p else None
    except Exception as e:
        logger.warning(f"price fetch failed for {ticker}: {e}")

    # NPV estimates
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
    """Recent news articles with sentiment."""
    ticker = ticker.upper().strip()
    try:
        from services.fetcher_news import fetch_all_sources
        # fetcher_news.fetch_all_sources(ticker, company, catalyst) signature
        # We don't have company/catalyst here so pass ticker for all 3 to get keyword-based search
        rows = db().get_stock(ticker)
        primary = rows[0] if rows else {}
        sources = fetch_all_sources(
            ticker,
            primary.get("company_name", ticker),
            primary.get("catalyst_type", "")
        )
        return {
            "ticker": ticker,
            "count": len(sources),
            "articles": sources[:limit],
        }
    except Exception as e:
        logger.exception(f"news fetch failed for {ticker}")
        raise HTTPException(500, f"news fetch error: {e}")


@router.get("/{ticker}/social")
async def get_stock_social(ticker: str):
    """StockTwits + Reddit sentiment."""
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
    """Analyst ratings — TipRanks + yfinance consensus."""
    ticker = ticker.upper().strip()
    try:
        from services.authenticated_sources import fetch_analyst_data_all_merged
        data = fetch_analyst_data_all_merged(ticker)
        return {"ticker": ticker, "data": data}
    except Exception as e:
        logger.warning(f"analyst fetch failed for {ticker}: {e}")
        return {"ticker": ticker, "data": None, "error": str(e)[:200]}
