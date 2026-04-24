"""
/strategies/{ticker} — wraps the strategy module pipeline:
  compute_technicals → classify_setup → recommend_strategies → build_strategy
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{ticker}")
async def get_strategies(
    ticker: str,
    ai_prob: float = Query(0.7, ge=0.0, le=1.0, description="AI-estimated approval probability"),
    days_to_catalyst: int = Query(90, ge=1, le=730),
):
    """
    Full strategy pack for a ticker.
    Pipeline: fetch technicals → classify setup → recommend strategies.
    """
    ticker = ticker.upper().strip()
    try:
        from services import strategy as S
        from services.fetcher import create_fetcher
        fetcher = create_fetcher()

        # 1. Get price history (yfinance)
        hist = None
        try:
            if hasattr(fetcher, "get_price_history"):
                hist = fetcher.get_price_history(ticker, days=252)
            elif hasattr(fetcher, "get_stock_data"):
                hist = fetcher.get_stock_data(ticker)
        except Exception as e:
            logger.warning(f"price history for {ticker}: {e}")

        # 2. Technicals
        tech = {"available": False}
        try:
            if hist is not None and len(hist) > 0:
                tech = S.compute_technicals(hist)
        except Exception as e:
            logger.warning(f"compute_technicals({ticker}): {e}")

        # 3. Options chain
        chain = {}
        try:
            chain = S.get_options_chain(ticker, expiry_target_days=max(days_to_catalyst, 30))
        except Exception as e:
            logger.warning(f"options chain {ticker}: {e}")
            chain = {"error": str(e)[:200]}

        # 4. Setup classification
        setup = {}
        atm_iv = chain.get("atm_iv") if isinstance(chain, dict) else None
        try:
            setup = S.classify_setup(tech, ai_prob, days_to_catalyst, atm_iv)
        except Exception as e:
            logger.warning(f"classify_setup {ticker}: {e}")
            setup = {"error": str(e)[:200]}

        # 5. Recommendations
        recs = []
        try:
            recs = S.recommend_strategies(tech, ai_prob, days_to_catalyst, atm_iv)
        except Exception as e:
            logger.warning(f"recommend_strategies {ticker}: {e}")

        return {
            "ticker": ticker,
            "inputs": {
                "ai_prob": ai_prob,
                "days_to_catalyst": days_to_catalyst,
                "atm_iv": atm_iv,
            },
            "technicals": tech,
            "setup": setup,
            "options_chain": chain,
            "recommendations": recs,
        }
    except Exception as e:
        logger.exception(f"strategies {ticker} failed")
        raise HTTPException(500, f"strategies error: {type(e).__name__}: {str(e)[:300]}")


@router.get("/{ticker}/options")
async def get_options_chain(ticker: str, days: int = Query(30, ge=7, le=365)):
    """Just the raw options chain."""
    ticker = ticker.upper().strip()
    try:
        from services.strategy import get_options_chain
        chain = get_options_chain(ticker, expiry_target_days=days)
        return {"ticker": ticker, "chain": chain}
    except Exception as e:
        raise HTTPException(500, f"options error: {type(e).__name__}: {str(e)[:300]}")
