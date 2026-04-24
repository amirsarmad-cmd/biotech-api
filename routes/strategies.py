"""
/strategies/{ticker} — buy/sell/hedge recommendations + options chains.
"""
import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{ticker}")
async def get_strategies(ticker: str):
    """Full strategy pack — long/short/options/hedge."""
    ticker = ticker.upper().strip()
    try:
        from services.strategy import build_strategy_pack
        result = build_strategy_pack(ticker)
        return {"ticker": ticker, "strategies": result}
    except AttributeError:
        # Fallback if that function name is different
        try:
            from services.strategy import get_strategies as gs
            return {"ticker": ticker, "strategies": gs(ticker)}
        except Exception as e:
            logger.exception("strategies fetch failed")
            raise HTTPException(500, f"strategies error: {e}")
    except Exception as e:
        logger.exception("strategies fetch failed")
        raise HTTPException(500, f"strategies error: {e}")


@router.get("/{ticker}/setup-read")
async def get_setup_read(ticker: str):
    """Setup Read — top-3 strategies based on current entry window."""
    ticker = ticker.upper().strip()
    try:
        from services import strategy as strategy_mod
        # Look for the right function name
        if hasattr(strategy_mod, "build_setup_read"):
            result = strategy_mod.build_setup_read(ticker)
        elif hasattr(strategy_mod, "setup_read"):
            result = strategy_mod.setup_read(ticker)
        else:
            raise HTTPException(501, "setup-read function not yet ported")
        return {"ticker": ticker, "setup": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("setup-read failed")
        raise HTTPException(500, f"setup-read error: {e}")
