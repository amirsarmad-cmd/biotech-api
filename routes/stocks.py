"""
/stocks routes — universe, detail, news, social, analyst.
"""
import logging
import os
import json as _json
from typing import Optional, Any, List, Dict
from fastapi import APIRouter, HTTPException, Query

from services.database import BiotechDatabase

logger = logging.getLogger(__name__)
router = APIRouter()

# Catalyst types that represent drug approval/clinical events (NPV applies)
DRUG_CATALYST_KEYWORDS = ("fda", "approval", "phase", "clinical", "pdufa", "readout", "trial", "nda", "bla")
# Catalyst types where NPV math doesn't apply
NON_DRUG_CATALYST_KEYWORDS = ("earnings", "dividend", "split", "analyst day", "investor day")

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
    except ImportError:
        _np = None
    if _np is not None:
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.ndarray,)):
            return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    try:
        _json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _is_drug_catalyst(catalyst_type: Optional[str]) -> bool:
    if not catalyst_type:
        return False
    t = catalyst_type.lower()
    if any(k in t for k in NON_DRUG_CATALYST_KEYWORDS):
        return False
    return any(k in t for k in DRUG_CATALYST_KEYWORDS)


def _pick_npv_catalyst(catalysts: List[Dict]) -> Optional[Dict]:
    """
    Select the best catalyst for NPV computation.
    Priority: FDA Decision > Phase 3 Readout > Clinical Trial > other drug catalysts.
    Among drug catalysts, pick highest probability.
    Returns None if no drug catalyst exists.
    """
    drug_cats = [c for c in catalysts if _is_drug_catalyst(c.get("catalyst_type"))]
    if not drug_cats:
        return None

    def rank(c):
        t = (c.get("catalyst_type") or "").lower()
        # Higher number = better priority
        if "fda" in t or "approval" in t or "pdufa" in t:
            tier = 4
        elif "phase 3" in t or "phase iii" in t:
            tier = 3
        elif "phase" in t or "clinical" in t or "readout" in t:
            tier = 2
        else:
            tier = 1
        return (tier, c.get("probability") or 0)

    drug_cats.sort(key=rank, reverse=True)
    return drug_cats[0]


def _get_live_price(ticker: str) -> Optional[float]:
    """
    Best-effort price fetch with 4 fallbacks, each with explicit logging.
    1. yfinance fast_info.last_price
    2. yfinance Ticker.info dict
    3. yfinance history() last close
    4. Finnhub /quote endpoint
    """
    # --- Attempt 1: yfinance fast_info ---
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        try:
            fi = t.fast_info
            p = None
            if hasattr(fi, "last_price"):
                p = fi.last_price
            if p and float(p) > 0:
                return float(p)
        except Exception as e:
            logger.warning(f"price {ticker}: fast_info failed: {type(e).__name__}: {e}")

        # --- Attempt 2: yfinance info ---
        try:
            info = t.info or {}
            for k in ("currentPrice", "regularMarketPrice", "previousClose", "regularMarketPreviousClose"):
                v = info.get(k)
                if v and float(v) > 0:
                    return float(v)
        except Exception as e:
            logger.warning(f"price {ticker}: info dict failed: {type(e).__name__}: {e}")

        # --- Attempt 3: yfinance history (usually never fails) ---
        try:
            hist = t.history(period="5d")
            if hist is not None and len(hist) > 0:
                close = hist["Close"].iloc[-1]
                if close and float(close) > 0:
                    return float(close)
        except Exception as e:
            logger.warning(f"price {ticker}: history failed: {type(e).__name__}: {e}")
    except Exception as e:
        logger.warning(f"price {ticker}: yfinance import/Ticker failed: {type(e).__name__}: {e}")

    # --- Attempt 4: Finnhub quote endpoint ---
    try:
        import urllib.request, json
        key = os.getenv("FINNHUB_API_KEY")
        if key:
            req = urllib.request.Request(
                f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={key}",
                headers={"User-Agent": "biotech-api/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                p = data.get("c")  # current price
                if p and float(p) > 0:
                    logger.info(f"price {ticker}: finnhub fallback succeeded (${p})")
                    return float(p)
    except Exception as e:
        logger.warning(f"price {ticker}: finnhub fallback failed: {type(e).__name__}: {e}")

    logger.warning(f"price {ticker}: ALL fallbacks returned no price")
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
        if high_prob_only and p < 0.6:
            continue
        if p < min_probability:
            continue
        filtered.append(r)

    reverse = sort != "ticker"
    filtered.sort(
        key=lambda x: (x.get(sort) or 0) if sort != "ticker" else x.get("ticker", ""),
        reverse=reverse,
    )

    return _to_jsonable({
        "count": len(filtered),
        "universe_size": len(rows),
        "high_prob_count": sum(1 for r in rows if (r.get("probability") or 0) >= 0.6),
        "stocks": filtered[:limit],
    })


@router.get("/{ticker}")
async def get_stock_detail(ticker: str, with_npv: bool = Query(True)):
    """
    Detail page data.
    - primary_catalyst = highest-probability catalyst (usually Earnings)
    - npv_catalyst = best DRUG catalyst for NPV math (FDA > Phase 3 > Clinical)
    - NPV is computed against npv_catalyst, not primary.
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

    # Pick the right catalyst for NPV (drug-related, not earnings)
    npv_catalyst = _pick_npv_catalyst(rows_sorted)

    current_price = _get_live_price(ticker)

    npv_result = None
    npv_catalyst_summary = None
    if with_npv:
        if npv_catalyst is None:
            npv_result = {
                "status": "skipped",
                "reason": "No drug catalyst (FDA Decision / Phase 3 / Clinical Trial) found for this ticker.",
            }
        else:
            npv_catalyst_summary = {
                "type": npv_catalyst.get("catalyst_type"),
                "date": npv_catalyst.get("catalyst_date"),
                "probability": npv_catalyst.get("probability"),
                "description": npv_catalyst.get("description"),
            }
            try:
                from services.npv_model import (
                    compute_npv_estimate,
                    estimate_drug_economics,
                    get_baseline_price,
                )
                economics = estimate_drug_economics(
                    ticker=ticker,
                    company_name=primary.get("company_name", ticker),
                    catalyst_type=npv_catalyst.get("catalyst_type") or "FDA Decision",
                    catalyst_date=npv_catalyst.get("catalyst_date", ""),
                    description=npv_catalyst.get("description", ""),
                    market_cap_m=float(primary.get("market_cap") or 0),
                )
                baseline_price = get_baseline_price(ticker) or current_price or 50.0
                price_for_calc = current_price or baseline_price
                npv_result = compute_npv_estimate(
                    ticker=ticker,
                    current_price=price_for_calc,
                    market_cap_m=float(primary.get("market_cap") or 0),
                    p_approval=float(npv_catalyst.get("probability") or 0.5),
                    economics=economics,
                    baseline_price=baseline_price,
                    info={
                        "catalyst_type": npv_catalyst.get("catalyst_type") or "",
                        "catalyst_date": npv_catalyst.get("catalyst_date") or "",
                        "description": npv_catalyst.get("description") or "",
                    },
                )
                # Fold economics into result for the frontend
                if isinstance(npv_result, dict):
                    npv_result["_economics"] = economics
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
        "npv_catalyst": npv_catalyst_summary,
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
