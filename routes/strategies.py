"""/strategies/{ticker} — full strategy pipeline with JSON-safe output."""
import logging, json as _json
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter()


def _to_jsonable(obj: Any) -> Any:
    try: import numpy as _np
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
    try:
        _json.dumps(obj); return obj
    except (TypeError, ValueError):
        return str(obj)


@router.get("/{ticker}")
async def get_strategies(
    ticker: str,
    ai_prob: float = Query(0.7, ge=0.0, le=1.0),
    days_to_catalyst: int = Query(90, ge=1, le=730),
):
    ticker = ticker.upper().strip()
    try:
        from services import strategy as S
        import yfinance as yf

        # 1. Price history (yfinance direct)
        hist = None
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1y")
        except Exception as e:
            logger.warning(f"price history {ticker}: {e}")

        # 2. Technicals
        tech = {"available": False}
        try:
            if hist is not None and len(hist) > 0:
                tech = S.compute_technicals(hist)
        except Exception as e:
            logger.warning(f"compute_technicals({ticker}): {e}")
            tech = {"available": False, "error": str(e)[:200]}

        # 3. Options chain
        chain = {}
        try:
            chain = S.get_options_chain(ticker, expiry_target_days=max(days_to_catalyst, 30))
        except Exception as e:
            logger.warning(f"options chain {ticker}: {e}")
            chain = {"available": False, "error": str(e)[:200]}

        atm_iv = chain.get("atm_iv") if isinstance(chain, dict) else None

        # 4. Setup
        setup = {}
        try:
            setup = S.classify_setup(tech, ai_prob, days_to_catalyst, atm_iv)
        except Exception as e:
            setup = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

        # 5. Recommendations
        recs = []
        try:
            recs = S.recommend_strategies(tech, ai_prob, days_to_catalyst, atm_iv)
        except Exception as e:
            logger.warning(f"recommend_strategies {ticker}: {e}")
            recs = []

        return _to_jsonable({
            "ticker": ticker,
            "inputs": {"ai_prob": ai_prob, "days_to_catalyst": days_to_catalyst, "atm_iv": atm_iv},
            "technicals": tech,
            "setup": setup,
            "options_chain": chain,
            "recommendations": recs,
        })
    except Exception as e:
        logger.exception(f"strategies {ticker} failed")
        raise HTTPException(500, f"strategies error: {type(e).__name__}: {str(e)[:300]}")


@router.get("/{ticker}/options")
async def get_options_chain(ticker: str, days: int = Query(30, ge=7, le=365)):
    ticker = ticker.upper().strip()
    try:
        from services.strategy import get_options_chain as goc
        return _to_jsonable({"ticker": ticker, "chain": goc(ticker, expiry_target_days=days)})
    except Exception as e:
        raise HTTPException(500, f"options error: {type(e).__name__}: {str(e)[:300]}")
