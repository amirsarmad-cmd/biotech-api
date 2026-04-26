"""
options_implied — pull straddle-based implied moves from yfinance options chain.

This is the standard pre-event indicator used by biotech analysts: the ATM
straddle premium (call_atm + put_atm) divided by stock price gives the
market's consensus expected absolute move through expiration.

Used in TWO places:
1. Forward-looking: shown on stock detail page as "market expects ±X%" for
   the upcoming catalyst, alongside our model's predicted_move_pct.
2. Post-event: captured at Phase 3A backfill time for accuracy comparison —
   how well did we predict relative to what the options market implied?
"""
import os
import logging
from datetime import datetime, date, timedelta
from typing import Dict, Optional, List
from functools import lru_cache

logger = logging.getLogger(__name__)


def get_implied_move(
    ticker: str,
    target_date: Optional[str] = None,
    use_cache: bool = True,
) -> Optional[Dict]:
    """Compute ATM-straddle implied move for a stock.
    
    Args:
      ticker: stock symbol
      target_date: catalyst date (YYYY-MM-DD). Picks the FIRST expiry
        on or after this date so we capture the catalyst event. If None,
        uses the nearest expiry (good for general "current implied vol" reads).
      use_cache: 5-minute LRU cache (set False for backtest accuracy)
    
    Returns dict {
      implied_move_pct: float,        # absolute % expected move
      expiry: str,                    # YYYY-MM-DD of the straddle expiry
      atm_strike: float,
      straddle_premium: float,
      stock_price: float,
      days_to_expiry: int,
      source: "yfinance",
      annualized_iv_pct: float,        # rough — straddle / S * sqrt(252/dte) * 100
    }
    Or None if options data unavailable.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed for options implied move")
        return None
    
    try:
        t = yf.Ticker(ticker)
        expirations = list(t.options or [])
        if not expirations:
            return None
        
        # Pick expiry: nearest >= target_date, else nearest absolute
        chosen_expiry = None
        if target_date:
            try:
                target_d = datetime.strptime(target_date[:10], "%Y-%m-%d").date()
                future = [e for e in expirations
                          if datetime.strptime(e, "%Y-%m-%d").date() >= target_d]
                chosen_expiry = future[0] if future else expirations[-1]
            except (ValueError, IndexError):
                chosen_expiry = expirations[0]
        else:
            chosen_expiry = expirations[0]
        
        # Pull stock price
        info = t.info or {}
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            try:
                fi = getattr(t, "fast_info", None)
                if fi:
                    price = (fi.get("lastPrice") if hasattr(fi, "get")
                             else getattr(fi, "last_price", None))
            except Exception:
                pass
        if not price or price <= 0:
            return None
        price = float(price)
        
        # Pull option chain
        chain = t.option_chain(chosen_expiry)
        calls = chain.calls
        puts = chain.puts
        if calls is None or calls.empty or puts is None or puts.empty:
            return None
        
        # Find ATM strike — the strike closest to current price with both call+put available
        all_strikes = set(calls["strike"].tolist()) & set(puts["strike"].tolist())
        if not all_strikes:
            return None
        atm_strike = min(all_strikes, key=lambda k: abs(k - price))
        
        # Get the call + put rows for that strike
        call_row = calls[calls["strike"] == atm_strike].iloc[0]
        put_row = puts[puts["strike"] == atm_strike].iloc[0]
        
        # Use mark = (bid + ask) / 2 if both > 0, else lastPrice fallback
        def _mark(row):
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            if bid > 0 and ask > 0 and ask > bid:
                return (bid + ask) / 2
            last = float(row.get("lastPrice", 0) or 0)
            return last if last > 0 else 0
        
        call_price = _mark(call_row)
        put_price = _mark(put_row)
        if call_price == 0 or put_price == 0:
            return None
        
        straddle = call_price + put_price
        # Adjust for moneyness: if ATM is slightly OTM, straddle understates
        # the implied move. Standard correction: straddle gives expected
        # absolute deviation; we use it directly as the % move estimator.
        implied_move_pct = (straddle / price) * 100.0
        
        # DTE
        try:
            exp_d = datetime.strptime(chosen_expiry, "%Y-%m-%d").date()
            dte = (exp_d - date.today()).days
        except ValueError:
            dte = 0
        
        # Rough annualized IV: straddle/S = sigma * sqrt(2/pi) * sqrt(T)
        # Simpler approach: scale by sqrt(252/dte) for an approximation
        import math
        annualized_iv = None
        if dte > 0:
            try:
                # straddle ≈ S × sigma × sqrt(2T/pi)
                # → sigma_annual ≈ (straddle/S) × sqrt(pi/(2T))
                T = dte / 365.0
                annualized_iv = (straddle / price) * math.sqrt(math.pi / (2 * T)) * 100
            except (ValueError, ZeroDivisionError):
                annualized_iv = None
        
        return {
            "implied_move_pct": round(implied_move_pct, 2),
            "expiry": chosen_expiry,
            "atm_strike": float(atm_strike),
            "straddle_premium": round(straddle, 2),
            "stock_price": price,
            "days_to_expiry": dte,
            "source": "yfinance_options",
            "annualized_iv_pct": round(annualized_iv, 1) if annualized_iv else None,
            "call_price": round(call_price, 2),
            "put_price": round(put_price, 2),
        }
    except Exception as e:
        logger.info(f"get_implied_move({ticker}) failed: {e}")
        return None


def get_implied_move_for_catalyst(
    ticker: str,
    catalyst_date: str,
) -> Optional[Dict]:
    """Convenience: pick the option expiry covering this catalyst date.
    Returns same shape as get_implied_move."""
    return get_implied_move(ticker=ticker, target_date=catalyst_date)
