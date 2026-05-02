"""Buy Strategy engine: stock + option strategies with technical + event-based entry/exit."""
import logging, re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# OPTION STRATEGIES — full catalog
# ============================================================
# bias:   'long', 'short', 'neutral', 'volatile'
# risk:   'conservative', 'moderate', 'aggressive'
# iv_pref: 'low', 'high', 'any'  (IV preference when entering)

def _safe_num(v, default=None):
    """Convert v to float, returning default if NaN/None/not-number."""
    try:
        import math
        f = float(v)
        if math.isnan(f) or math.isinf(f): return default
        return f
    except (TypeError, ValueError):
        return default


def _sanitize_option(opt):
    """Sanitize strike/mid/bid/ask/delta/gamma/iv in an option row.
    Returns None if essential fields (strike, mid) can't be made valid."""
    if not opt or not isinstance(opt, dict): return None
    strike = _safe_num(opt.get("strike"))
    mid = _safe_num(opt.get("mid"), default=0.0)
    bid = _safe_num(opt.get("bid"), default=0.0)
    ask = _safe_num(opt.get("ask"), default=0.0)
    if strike is None or strike <= 0: return None
    # If mid is missing/NaN, try to derive from bid/ask or fallback to either
    if mid == 0.0 or mid is None:
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
        elif ask > 0:
            mid = ask
        elif bid > 0:
            mid = bid
        else:
            return None  # truly no price info
    out = dict(opt)
    out["strike"] = strike
    out["mid"] = mid
    out["bid"] = bid
    out["ask"] = ask
    for k in ("delta","gamma","theta","vega","iv","volume","openInterest"):
        out[k] = _safe_num(opt.get(k), default=0.0)
    return out


OPTION_STRATEGIES = {
    # === Simple long options ===
    "long_call": {
        "name": "Long Call",
        "bias": "long", "risk": "aggressive", "iv_pref": "low",
        "description": "Buy a call option. Unlimited upside, limited downside to premium paid.",
        "max_loss": "100% of premium", "max_gain": "Unlimited",
    },
    "long_put": {
        "name": "Long Put",
        "bias": "short", "risk": "aggressive", "iv_pref": "low",
        "description": "Buy a put option. Profit when stock falls. Limited downside to premium paid.",
        "max_loss": "100% of premium", "max_gain": "Strike − premium (until $0)",
    },

    # === Covered/income ===
    "covered_call": {
        "name": "Covered Call",
        "bias": "neutral", "risk": "conservative", "iv_pref": "high",
        "description": "Own 100 shares + sell a call above current price. Collect premium; capped upside.",
        "max_loss": "Stock declines to $0 (minus premium)", "max_gain": "Strike − entry + premium",
    },
    "cash_secured_put": {
        "name": "Cash-Secured Put",
        "bias": "long", "risk": "conservative", "iv_pref": "high",
        "description": "Sell a put below current price; hold cash to buy shares if assigned. Collect premium.",
        "max_loss": "Strike − premium (stock to $0)", "max_gain": "Premium collected",
    },

    # === Vertical spreads ===
    "bull_call_spread": {
        "name": "Bull Call Spread",
        "bias": "long", "risk": "moderate", "iv_pref": "low",
        "description": "Buy lower-strike call + sell higher-strike call (same expiry). Defined risk/reward.",
        "max_loss": "Net debit paid", "max_gain": "Strike width − debit",
    },
    "bear_put_spread": {
        "name": "Bear Put Spread",
        "bias": "short", "risk": "moderate", "iv_pref": "low",
        "description": "Buy higher-strike put + sell lower-strike put. Defined bearish play.",
        "max_loss": "Net debit paid", "max_gain": "Strike width − debit",
    },

    # === Volatility plays ===
    "long_straddle": {
        "name": "Long Straddle",
        "bias": "volatile", "risk": "aggressive", "iv_pref": "low",
        "description": "Buy call + put at same strike/expiry. Profit if stock moves big in either direction.",
        "max_loss": "Sum of both premiums", "max_gain": "Unlimited either direction",
    },
    "long_strangle": {
        "name": "Long Strangle",
        "bias": "volatile", "risk": "aggressive", "iv_pref": "low",
        "description": "Buy OTM call + OTM put. Cheaper than straddle but needs bigger move.",
        "max_loss": "Sum of both premiums", "max_gain": "Unlimited either direction",
    },

    # === Defined-risk neutral ===
    "iron_condor": {
        "name": "Iron Condor",
        "bias": "neutral", "risk": "moderate", "iv_pref": "high",
        "description": "Sell OTM call spread + sell OTM put spread. Profits if stock stays in range.",
        "max_loss": "Strike width − net credit", "max_gain": "Net credit received",
    },
    "butterfly": {
        "name": "Butterfly (Long Call)",
        "bias": "neutral", "risk": "moderate", "iv_pref": "high",
        "description": "Buy 1 ITM + sell 2 ATM + buy 1 OTM calls. Max profit if stock lands at middle strike.",
        "max_loss": "Net debit paid", "max_gain": "Strike width − debit (at middle)",
    },

    # === Calendar / time-based ===
    "calendar_spread": {
        "name": "Calendar Spread",
        "bias": "neutral", "risk": "moderate", "iv_pref": "low",
        "description": "Sell near-term option + buy longer-term option at same strike. Profits from time decay difference.",
        "max_loss": "Net debit paid", "max_gain": "Variable (peaks near short-expiry at strike)",
    },
    "diagonal": {
        "name": "Diagonal Spread",
        "bias": "long", "risk": "moderate", "iv_pref": "low",
        "description": "Buy LEAPS call + sell near-term OTM call. Generates income while holding long position.",
        "max_loss": "Long leg cost − credits", "max_gain": "Variable by roll",
    },

    # === Leveraged/ratio ===
    "ratio_spread": {
        "name": "Ratio Spread (1×2)",
        "bias": "long", "risk": "aggressive", "iv_pref": "high",
        "description": "Buy 1 ITM call + sell 2 OTM calls. Profits if stock rises moderately; loses if it rockets.",
        "max_loss": "Unlimited above upper strike", "max_gain": "Spread width",
    },

    # === Stock + option combined ===
    "collar": {
        "name": "Collar",
        "bias": "neutral", "risk": "conservative", "iv_pref": "any",
        "description": "Own 100 shares + buy put + sell call. Protects downside, caps upside. Near-zero cost.",
        "max_loss": "Entry − put strike (bounded)", "max_gain": "Call strike − entry",
    },
    "synthetic_long": {
        "name": "Synthetic Long",
        "bias": "long", "risk": "aggressive", "iv_pref": "any",
        "description": "Buy call + sell put at same strike. Mimics stock with leverage.",
        "max_loss": "Very large if stock drops sharply", "max_gain": "Unlimited",
    },
    "synthetic_short": {
        "name": "Synthetic Short",
        "bias": "short", "risk": "aggressive", "iv_pref": "any",
        "description": "Sell call + buy put at same strike. Mimics shorting stock without borrowing.",
        "max_loss": "Unlimited", "max_gain": "Strike − premium (stock to $0)",
    },
}

RISK_PROFILES = {
    "conservative": ["covered_call","cash_secured_put","collar","bull_call_spread","bear_put_spread","iron_condor"],
    "moderate":     ["covered_call","cash_secured_put","collar","bull_call_spread","bear_put_spread",
                     "iron_condor","butterfly","calendar_spread","diagonal","long_straddle","long_strangle"],
    "aggressive":   list(OPTION_STRATEGIES.keys()),  # All 15+
}

# ============================================================
# TECHNICAL INDICATORS
# ============================================================
def compute_technicals(hist: pd.DataFrame) -> Dict:
    """Given yfinance history DataFrame, compute RSI, MACD, MAs, support/resistance."""
    if hist is None or hist.empty or len(hist) < 20:
        return {"available": False}

    try:
        close = hist["Close"]
        high = hist["High"]
        low = hist["Low"]
        volume = hist["Volume"]

        # Moving averages
        ma20  = close.rolling(20).mean().iloc[-1]
        ma50  = close.rolling(50).mean().iloc[-1] if len(close)>=50 else None
        ma200 = close.rolling(200).mean().iloc[-1] if len(close)>=200 else None

        # RSI (14)
        delta = close.diff()
        gain = delta.where(delta>0, 0).rolling(14).mean()
        loss = -delta.where(delta<0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs.iloc[-1])) if not loss.iloc[-1] == 0 else 50

        # MACD (12, 26, 9)
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd_line = (exp12 - exp26).iloc[-1]
        signal = (exp12 - exp26).ewm(span=9, adjust=False).mean().iloc[-1]

        # Support/resistance (simple: recent lows/highs in last 60 days)
        recent = hist.tail(60)
        support    = recent["Low"].min()
        resistance = recent["High"].max()

        # 52w position
        hi52 = close.tail(252).max() if len(close) >= 252 else close.max()
        lo52 = close.tail(252).min() if len(close) >= 252 else close.min()
        current = close.iloc[-1]
        pos_52w = (current - lo52) / (hi52 - lo52) if (hi52 > lo52) else 0.5

        # Historical volatility (60-day, annualized)
        log_returns = np.log(close / close.shift(1)).dropna()
        hv_60 = log_returns.tail(60).std() * np.sqrt(252) if len(log_returns) >= 60 else None

        return {
            "available": True,
            "current": float(current),
            "ma20": float(ma20) if ma20 else None,
            "ma50": float(ma50) if ma50 else None,
            "ma200": float(ma200) if ma200 else None,
            "rsi": float(rsi) if not pd.isna(rsi) else None,
            "macd": float(macd_line) if not pd.isna(macd_line) else None,
            "macd_signal": float(signal) if not pd.isna(signal) else None,
            "support": float(support),
            "resistance": float(resistance),
            "hi_52w": float(hi52),
            "lo_52w": float(lo52),
            "pos_52w": float(pos_52w),
            "hist_volatility": float(hv_60) if hv_60 else None,
        }
    except Exception as e:
        logger.error(f"compute_technicals error: {e}")
        return {"available": False, "error": str(e)}


# ============================================================
# ENTRY/EXIT LOGIC — picks stronger of technical vs event-based
# ============================================================
def suggest_entry_exit(tech: Dict, ai_prob: float, days_to_catalyst: int,
                       direction: str = "long") -> Dict:
    """Return: {entry, exit, stop_loss, reason, confidence, mode}.
    Combines technical signals and catalyst event-based logic, picks strongest."""

    if not tech.get("available"):
        return {"available": False, "reason": "Insufficient price history"}

    current = tech["current"]

    # === TECHNICAL SIGNAL ===
    tech_signals = []
    tech_confidence = 0.5

    # RSI
    rsi = tech.get("rsi") or 50
    if direction == "long":
        if rsi < 30: tech_signals.append(("oversold", 0.8, "RSI < 30"))
        elif rsi > 70: tech_signals.append(("overbought", 0.2, "RSI > 70 — bad entry for long"))
    elif direction == "short":
        if rsi > 70: tech_signals.append(("overbought", 0.8, "RSI > 70"))
        elif rsi < 30: tech_signals.append(("oversold", 0.2, "RSI < 30 — bad entry for short"))

    # MACD
    macd = tech.get("macd") or 0
    signal = tech.get("macd_signal") or 0
    if direction == "long" and macd > signal:
        tech_signals.append(("macd_bullish_cross", 0.7, f"MACD {macd:.2f} > signal {signal:.2f}"))
    elif direction == "short" and macd < signal:
        tech_signals.append(("macd_bearish_cross", 0.7, f"MACD {macd:.2f} < signal {signal:.2f}"))

    # Support/resistance
    support = tech.get("support") or current*0.9
    resistance = tech.get("resistance") or current*1.1
    dist_to_support = (current - support) / current
    dist_to_resistance = (resistance - current) / current

    if direction == "long":
        # Best entry: near support with buffer
        tech_entry = max(support * 1.02, current * 0.97)  # 2% above support, or -3% from current
        tech_exit = min(resistance * 0.98, current * 1.15)  # 2% below resistance, or +15%
        tech_stop = support * 0.97
    else:
        tech_entry = min(resistance * 0.98, current * 1.03)
        tech_exit = max(support * 1.02, current * 0.85)
        tech_stop = resistance * 1.03

    if tech_signals:
        tech_confidence = sum(s[1] for s in tech_signals) / len(tech_signals)

    # === EVENT-BASED SIGNAL ===
    event_signals = []

    # If catalyst is soon, enter ahead
    if days_to_catalyst <= 7:
        event_entry = current  # enter NOW if catalyst is imminent
        event_exit = None  # exit post-catalyst
        event_signals.append(("imminent_catalyst", 0.9, f"Catalyst in {days_to_catalyst}d — enter now, exit post-event"))
    elif days_to_catalyst <= 30:
        event_entry = current * (0.97 if direction == "long" else 1.03)  # slight dip/pop
        event_exit = None
        event_signals.append(("near_catalyst", 0.7, f"Catalyst in {days_to_catalyst}d — enter on small pullback"))
    elif days_to_catalyst <= 90:
        event_entry = current * (0.93 if direction == "long" else 1.07)  # wait for 7% move
        event_exit = None
        event_signals.append(("mid_catalyst", 0.5, f"Catalyst in {days_to_catalyst}d — accumulate on dips"))
    else:
        event_entry = current * (0.90 if direction == "long" else 1.10)  # bigger patience
        event_exit = None
        event_signals.append(("far_catalyst", 0.3, f"Catalyst in {days_to_catalyst}d — plenty of time"))

    # AI probability modifier
    if direction == "long" and ai_prob >= 0.70:
        event_signals.append(("high_conviction_long", 0.9, f"AI gives {ai_prob:.0%} → high conviction"))
    elif direction == "short" and ai_prob <= 0.30:
        event_signals.append(("high_conviction_short", 0.9, f"AI gives {ai_prob:.0%} → high short conviction"))
    elif 0.45 <= ai_prob <= 0.55:
        event_signals.append(("coin_flip", 0.2, f"AI gives {ai_prob:.0%} → low directional conviction, prefer volatility play"))

    event_confidence = sum(s[1] for s in event_signals) / len(event_signals) if event_signals else 0.5

    # === PICK THE STRONGER SIGNAL ===
    if tech_confidence >= event_confidence:
        chosen_mode = "technical"
        chosen_entry = tech_entry
        chosen_exit = tech_exit
        chosen_stop = tech_stop
        chosen_reasons = [f"{s[2]}" for s in tech_signals] or ["Technical levels (no strong signal, using S/R zones)"]
        confidence = tech_confidence
    else:
        chosen_mode = "event_based"
        chosen_entry = event_entry
        # Event-based exit: post-catalyst, estimated using expected price move
        if direction == "long":
            chosen_exit = current * 1.20  # assume 20% move up
        else:
            chosen_exit = current * 0.80
        chosen_stop = current * (0.92 if direction == "long" else 1.08)
        chosen_reasons = [s[2] for s in event_signals]
        confidence = event_confidence

    return {
        "available": True,
        "direction": direction,
        "entry": round(chosen_entry, 2),
        "exit": round(chosen_exit, 2),
        "stop_loss": round(chosen_stop, 2),
        "mode": chosen_mode,
        "confidence": round(confidence, 2),
        "reasons": chosen_reasons,
        "technical": {
            "entry": round(tech_entry, 2),
            "exit": round(tech_exit, 2),
            "stop": round(tech_stop, 2),
            "confidence": round(tech_confidence, 2),
            "signals": [s[2] for s in tech_signals],
        },
        "event_based": {
            "entry": round(event_entry, 2),
            "exit": round(chosen_exit if chosen_mode == "event_based" else (current * (1.20 if direction == "long" else 0.80)), 2),
            "stop": round(current * (0.92 if direction == "long" else 1.08), 2),
            "confidence": round(event_confidence, 2),
            "signals": [s[2] for s in event_signals],
        },
        "risk_reward": round(abs(chosen_exit - chosen_entry) / abs(chosen_entry - chosen_stop), 2)
            if chosen_stop != chosen_entry else 0,
    }


# ============================================================
# OPTIONS CHAIN — fetch live from yfinance
# ============================================================
def get_options_chain(ticker: str, expiry_target_days: int = 30) -> Dict:
    """Fetch live options chain from yfinance. Returns {calls, puts, expiry, iv_rank_est, atm_iv}."""
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        expiries = tk.options  # List of 'YYYY-MM-DD'
        if not expiries:
            return {"available": False, "error": "No options listed for this ticker"}

        # Find expiry closest to target days
        today = datetime.now().date()
        best_exp = None
        best_diff = 10000
        for e in expiries:
            try:
                edt = datetime.strptime(e, "%Y-%m-%d").date()
                days = (edt - today).days
                if days >= 1:
                    diff = abs(days - expiry_target_days)
                    if diff < best_diff:
                        best_diff = diff
                        best_exp = e
            except: continue

        if not best_exp:
            best_exp = expiries[0]

        chain = tk.option_chain(best_exp)
        calls = chain.calls
        puts = chain.puts

        # ATM IV estimate
        try:
            current_price = tk.history(period="1d")["Close"].iloc[-1]
            calls_atm = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]
            atm_iv = float(calls_atm["impliedVolatility"].iloc[0]) if not calls_atm.empty else None
        except:
            atm_iv = None
            current_price = None

        exp_dt = datetime.strptime(best_exp, "%Y-%m-%d").date()
        days_to_exp = (exp_dt - today).days

        return {
            "available": True,
            "expiry": best_exp,
            "days_to_expiry": days_to_exp,
            "all_expiries": list(expiries),
            "calls": calls,
            "puts": puts,
            "atm_iv": atm_iv,
            "current_price": current_price,
        }
    except Exception as e:
        logger.error(f"get_options_chain({ticker}): {e}")
        return {"available": False, "error": str(e)}


# ============================================================
# STRATEGY BUILDER — produces concrete legs per strategy
# ============================================================
def _closest_strike(chain_df: pd.DataFrame, target: float) -> Optional[Dict]:
    """Pick the option whose strike is closest to target.
    Handles NaN in strike/bid/ask/etc by returning None if the row is unusable.
    """
    import math
    if chain_df is None or chain_df.empty:
        return None
    try:
        # Filter to rows with valid (non-NaN) strikes
        valid = chain_df[chain_df["strike"].notna()]
        if valid.empty: return None
        idx = (valid["strike"] - target).abs().argsort()[:1]
        row = valid.iloc[idx.values[0]]
    except Exception:
        return None
    
    def _f(v, default=0.0):
        try:
            f = float(v) if v is not None else default
            if math.isnan(f) or math.isinf(f): return default
            return f
        except (TypeError, ValueError):
            return default
    
    strike = _f(row.get("strike"), default=None)
    if strike is None or strike <= 0: return None
    
    bid = _f(row.get("bid"))
    ask = _f(row.get("ask"))
    last = _f(row.get("lastPrice"))
    
    # Derive mid: prefer (bid+ask)/2, fall back to lastPrice, then ask, then bid
    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2
    elif last > 0:
        mid = last
    elif ask > 0:
        mid = ask
    elif bid > 0:
        mid = bid
    else:
        return None  # no usable price at all
    
    return {
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "last": last,
        "mid": mid,
        "iv": _f(row.get("impliedVolatility")),
        "volume": int(_f(row.get("volume"))),
        "open_interest": int(_f(row.get("openInterest"))),
        "delta": _f(row.get("delta")) if "delta" in chain_df.columns else None,
    }


def build_strategy(strategy_key: str, chain: Dict, position_value: float = 10000,
                   direction: str = "long") -> Dict:
    """Given an options chain and a strategy name, construct concrete legs and metrics."""

    if strategy_key not in OPTION_STRATEGIES:
        return {"error": f"Unknown strategy: {strategy_key}"}
    if not chain.get("available"):
        return {"error": "Options chain unavailable"}

    meta = OPTION_STRATEGIES[strategy_key]
    calls = chain["calls"]
    puts = chain["puts"]
    current = chain.get("current_price") or 0

    if current == 0 or calls is None or calls.empty:
        return {"error": "No usable chain data"}

    legs = []
    max_loss = max_gain = breakeven = 0
    prob_profit = 0.5
    capital = 0
    commentary = meta["description"]

    try:
        # === LONG CALL ===
        if strategy_key == "long_call":
            strike = current * 1.05  # slightly OTM
            c = _closest_strike(calls, strike)
            if not c: return {"error":"No suitable call"}
            contracts = max(1, int(position_value / (c["mid"] * 100))) if c["mid"] > 0 else 1
            legs = [{"action":"BUY","type":"CALL","strike":c["strike"],"qty":contracts,"price":c["mid"]}]
            max_loss = c["mid"] * 100 * contracts
            max_gain = float('inf')
            breakeven = c["strike"] + c["mid"]
            capital = max_loss
            prob_profit = 0.40 if c["strike"] > current else 0.55

        elif strategy_key == "long_put":
            strike = current * 0.95  # OTM put
            p = _closest_strike(puts, strike)
            if not p: return {"error":"No suitable put"}
            contracts = max(1, int(position_value / (p["mid"] * 100))) if p["mid"] > 0 else 1
            legs = [{"action":"BUY","type":"PUT","strike":p["strike"],"qty":contracts,"price":p["mid"]}]
            max_loss = p["mid"] * 100 * contracts
            max_gain = (p["strike"] - p["mid"]) * 100 * contracts  # if stock to $0
            breakeven = p["strike"] - p["mid"]
            capital = max_loss
            prob_profit = 0.40 if p["strike"] < current else 0.55

        # === COVERED CALL ===
        elif strategy_key == "covered_call":
            strike = current * 1.08  # OTM call
            c = _closest_strike(calls, strike)
            if not c: return {"error":"No suitable call"}
            shares = max(100, int(position_value / current / 100) * 100)
            contracts = shares // 100
            stock_cost = shares * current
            legs = [
                {"action":"BUY","type":"STOCK","qty":shares,"price":current},
                {"action":"SELL","type":"CALL","strike":c["strike"],"qty":contracts,"price":c["mid"]},
            ]
            max_loss = stock_cost - c["mid"]*100*contracts
            max_gain = (c["strike"] - current + c["mid"]) * shares
            breakeven = current - c["mid"]
            capital = stock_cost - c["mid"]*100*contracts
            prob_profit = 0.70

        elif strategy_key == "cash_secured_put":
            strike = current * 0.92
            p = _closest_strike(puts, strike)
            if not p: return {"error":"No suitable put"}
            if p is None or not p.get("strike"):
                return {"error": "Option data incomplete (NaN strike). Try a different expiry."}
            contracts = max(1, int(position_value / (p["strike"] * 100)))
            legs = [
                {"action":"SELL","type":"PUT","strike":p["strike"],"qty":contracts,"price":p["mid"]},
                {"action":"HOLD_CASH","amount":p["strike"]*100*contracts},
            ]
            max_loss = (p["strike"] - p["mid"]) * 100 * contracts
            max_gain = p["mid"] * 100 * contracts
            breakeven = p["strike"] - p["mid"]
            capital = p["strike"] * 100 * contracts
            prob_profit = 0.75

        # === VERTICAL SPREADS ===
        elif strategy_key == "bull_call_spread":
            long_strike = current * 1.02
            short_strike = current * 1.10
            lc = _closest_strike(calls, long_strike)
            sc = _closest_strike(calls, short_strike)
            if not lc or not sc: return {"error":"Missing legs"}
            net_debit = lc["mid"] - sc["mid"]
            contracts = max(1, int(position_value / (net_debit * 100))) if net_debit > 0 else 1
            legs = [
                {"action":"BUY","type":"CALL","strike":lc["strike"],"qty":contracts,"price":lc["mid"]},
                {"action":"SELL","type":"CALL","strike":sc["strike"],"qty":contracts,"price":sc["mid"]},
            ]
            max_loss = net_debit * 100 * contracts
            max_gain = ((sc["strike"] - lc["strike"]) - net_debit) * 100 * contracts
            breakeven = lc["strike"] + net_debit
            capital = max_loss
            prob_profit = 0.45

        elif strategy_key == "bear_put_spread":
            long_strike = current * 0.98
            short_strike = current * 0.90
            lp = _closest_strike(puts, long_strike)
            sp = _closest_strike(puts, short_strike)
            if not lp or not sp: return {"error":"Missing legs"}
            net_debit = lp["mid"] - sp["mid"]
            contracts = max(1, int(position_value / (net_debit * 100))) if net_debit > 0 else 1
            legs = [
                {"action":"BUY","type":"PUT","strike":lp["strike"],"qty":contracts,"price":lp["mid"]},
                {"action":"SELL","type":"PUT","strike":sp["strike"],"qty":contracts,"price":sp["mid"]},
            ]
            max_loss = net_debit * 100 * contracts
            max_gain = ((lp["strike"] - sp["strike"]) - net_debit) * 100 * contracts
            breakeven = lp["strike"] - net_debit
            capital = max_loss
            prob_profit = 0.45

        # === VOLATILITY PLAYS ===
        elif strategy_key == "long_straddle":
            c = _closest_strike(calls, current)
            p = _closest_strike(puts, current)
            if not c or not p: return {"error":"Missing legs"}
            total_premium = c["mid"] + p["mid"]
            contracts = max(1, int(position_value / (total_premium * 100))) if total_premium > 0 else 1
            legs = [
                {"action":"BUY","type":"CALL","strike":c["strike"],"qty":contracts,"price":c["mid"]},
                {"action":"BUY","type":"PUT","strike":p["strike"],"qty":contracts,"price":p["mid"]},
            ]
            max_loss = total_premium * 100 * contracts
            max_gain = float('inf')
            breakeven = f"{c['strike']+total_premium:.2f} up / {p['strike']-total_premium:.2f} down"
            capital = max_loss
            prob_profit = 0.50

        elif strategy_key == "long_strangle":
            c = _closest_strike(calls, current * 1.05)
            p = _closest_strike(puts, current * 0.95)
            if not c or not p: return {"error":"Missing legs"}
            total_premium = c["mid"] + p["mid"]
            contracts = max(1, int(position_value / (total_premium * 100))) if total_premium > 0 else 1
            legs = [
                {"action":"BUY","type":"CALL","strike":c["strike"],"qty":contracts,"price":c["mid"]},
                {"action":"BUY","type":"PUT","strike":p["strike"],"qty":contracts,"price":p["mid"]},
            ]
            max_loss = total_premium * 100 * contracts
            max_gain = float('inf')
            breakeven = f"{c['strike']+total_premium:.2f} up / {p['strike']-total_premium:.2f} down"
            capital = max_loss
            prob_profit = 0.40

        # === NEUTRAL DEFINED-RISK ===
        elif strategy_key == "iron_condor":
            # Sell OTM call spread + sell OTM put spread
            sc = _closest_strike(calls, current * 1.10)
            lc = _closest_strike(calls, current * 1.18)
            sp = _closest_strike(puts, current * 0.90)
            lp = _closest_strike(puts, current * 0.82)
            if not all([sc, lc, sp, lp]): return {"error":"Missing legs"}
            net_credit = (sc["mid"] - lc["mid"]) + (sp["mid"] - lp["mid"])
            strike_width = (lc["strike"] - sc["strike"])
            contracts = max(1, int(position_value / (strike_width * 100))) if strike_width > 0 else 1
            legs = [
                {"action":"SELL","type":"CALL","strike":sc["strike"],"qty":contracts,"price":sc["mid"]},
                {"action":"BUY", "type":"CALL","strike":lc["strike"],"qty":contracts,"price":lc["mid"]},
                {"action":"SELL","type":"PUT", "strike":sp["strike"],"qty":contracts,"price":sp["mid"]},
                {"action":"BUY", "type":"PUT", "strike":lp["strike"],"qty":contracts,"price":lp["mid"]},
            ]
            max_loss = (strike_width - net_credit) * 100 * contracts
            max_gain = net_credit * 100 * contracts
            breakeven = f"{sc['strike']+net_credit:.2f} up / {sp['strike']-net_credit:.2f} down"
            capital = max_loss  # Broker requires strike-width margin
            prob_profit = 0.65

        elif strategy_key == "butterfly":
            # Buy 1 ITM + Sell 2 ATM + Buy 1 OTM (call butterfly)
            low = _closest_strike(calls, current * 0.95)
            mid = _closest_strike(calls, current)
            high = _closest_strike(calls, current * 1.05)
            if not all([low, mid, high]): return {"error":"Missing legs"}
            net_debit = low["mid"] - 2*mid["mid"] + high["mid"]
            if net_debit <= 0:
                net_debit = abs(net_debit) + 0.5  # safety
            contracts = max(1, int(position_value / (net_debit * 100)))
            legs = [
                {"action":"BUY", "type":"CALL","strike":low["strike"], "qty":contracts,  "price":low["mid"]},
                {"action":"SELL","type":"CALL","strike":mid["strike"], "qty":contracts*2,"price":mid["mid"]},
                {"action":"BUY", "type":"CALL","strike":high["strike"],"qty":contracts,  "price":high["mid"]},
            ]
            max_loss = net_debit * 100 * contracts
            max_gain = ((mid["strike"] - low["strike"]) - net_debit) * 100 * contracts
            breakeven = f"{low['strike']+net_debit:.2f} up / {high['strike']-net_debit:.2f} down"
            capital = max_loss
            prob_profit = 0.35

        # === CALENDAR ===
        elif strategy_key == "calendar_spread":
            # Need multi-expiry — approximate with near-term short + longer-term long at same strike
            c_near = _closest_strike(calls, current)
            if not c_near: return {"error":"No calls available"}
            # Assume longer-term option costs 1.5x near-term (rough approximation)
            approx_long_price = c_near["mid"] * 1.5
            net_debit = approx_long_price - c_near["mid"]
            contracts = max(1, int(position_value / (net_debit * 100))) if net_debit > 0 else 1
            legs = [
                {"action":"SELL","type":"CALL","strike":c_near["strike"],"qty":contracts,"price":c_near["mid"],"expiry":"near"},
                {"action":"BUY","type":"CALL","strike":c_near["strike"],"qty":contracts,"price":approx_long_price,"expiry":"far"},
            ]
            max_loss = net_debit * 100 * contracts
            max_gain = c_near["mid"] * 100 * contracts * 1.5  # approximation
            breakeven = f"~{c_near['strike']:.2f} (at near-term expiry)"
            capital = max_loss
            prob_profit = 0.55
            commentary += "  Note: requires second expiry — pick longer-dated option manually."

        elif strategy_key == "diagonal":
            # Long LEAPS + short near-term OTM
            long_strike = current * 0.92  # deep ITM call (LEAPS)
            short_strike = current * 1.05
            lc = _closest_strike(calls, long_strike)
            sc = _closest_strike(calls, short_strike)
            if not lc or not sc: return {"error":"Missing legs"}
            approx_leaps = lc["mid"] * 2.5  # LEAPS much pricier
            net_debit = approx_leaps - sc["mid"]
            contracts = max(1, int(position_value / (net_debit * 100))) if net_debit > 0 else 1
            legs = [
                {"action":"BUY","type":"CALL","strike":lc["strike"],"qty":contracts,"price":approx_leaps,"expiry":"LEAPS"},
                {"action":"SELL","type":"CALL","strike":sc["strike"],"qty":contracts,"price":sc["mid"],"expiry":"near"},
            ]
            max_loss = approx_leaps * 100 * contracts
            max_gain = (sc["strike"] - lc["strike"]) * 100 * contracts + sc["mid"]*100*contracts*3  # multiple rolls
            breakeven = lc["strike"] + net_debit
            capital = max_loss
            prob_profit = 0.55

        # === LEVERAGED/RATIO ===
        elif strategy_key == "ratio_spread":
            long_strike = current * 1.02
            short_strike = current * 1.12
            lc = _closest_strike(calls, long_strike)
            sc = _closest_strike(calls, short_strike)
            if not lc or not sc: return {"error":"Missing legs"}
            net_credit = 2*sc["mid"] - lc["mid"]  # 1x2 ratio
            contracts_ratio = max(1, int(position_value / (lc["mid"] * 100))) if lc["mid"] > 0 else 1
            legs = [
                {"action":"BUY","type":"CALL","strike":lc["strike"],"qty":contracts_ratio,"price":lc["mid"]},
                {"action":"SELL","type":"CALL","strike":sc["strike"],"qty":contracts_ratio*2,"price":sc["mid"]},
            ]
            max_loss = float('inf')  # unlimited above upper strike
            max_gain = (sc["strike"] - lc["strike"] + net_credit) * 100 * contracts_ratio
            breakeven = sc["strike"] + (sc["strike"] - lc["strike"] + net_credit)
            capital = lc["mid"] * 100 * contracts_ratio  # approximate margin
            prob_profit = 0.55

        # === STOCK + OPTION COMBOS ===
        elif strategy_key == "collar":
            shares = max(100, int(position_value / current / 100) * 100)
            contracts = shares // 100
            put_strike = current * 0.92  # protective
            call_strike = current * 1.08
            pp = _closest_strike(puts, put_strike)
            cc = _closest_strike(calls, call_strike)
            if not pp or not cc: return {"error":"Missing legs"}
            net_cost = pp["mid"] - cc["mid"]  # ~zero
            legs = [
                {"action":"BUY","type":"STOCK","qty":shares,"price":current},
                {"action":"BUY","type":"PUT","strike":pp["strike"],"qty":contracts,"price":pp["mid"]},
                {"action":"SELL","type":"CALL","strike":cc["strike"],"qty":contracts,"price":cc["mid"]},
            ]
            max_loss = (current - pp["strike"] + net_cost) * shares
            max_gain = (cc["strike"] - current - net_cost) * shares
            breakeven = current + net_cost
            capital = shares * current + net_cost*100*contracts
            prob_profit = 0.60

        elif strategy_key == "synthetic_long":
            c = _closest_strike(calls, current)
            p = _closest_strike(puts, current)
            if not c or not p: return {"error":"Missing legs"}
            net_cost = c["mid"] - p["mid"]  # small debit or credit
            contracts = max(1, int(position_value / (current * 100)))  # margin-based sizing
            legs = [
                {"action":"BUY","type":"CALL","strike":c["strike"],"qty":contracts,"price":c["mid"]},
                {"action":"SELL","type":"PUT","strike":p["strike"],"qty":contracts,"price":p["mid"]},
            ]
            max_loss = c["strike"] * 100 * contracts  # stock goes to $0
            max_gain = float('inf')
            breakeven = c["strike"] + net_cost
            capital = c["strike"] * 100 * contracts * 0.20  # ~20% margin
            prob_profit = 0.50

        elif strategy_key == "synthetic_short":
            c = _closest_strike(calls, current)
            p = _closest_strike(puts, current)
            if not c or not p: return {"error":"Missing legs"}
            net_credit = c["mid"] - p["mid"]
            contracts = max(1, int(position_value / (current * 100)))
            legs = [
                {"action":"SELL","type":"CALL","strike":c["strike"],"qty":contracts,"price":c["mid"]},
                {"action":"BUY","type":"PUT","strike":p["strike"],"qty":contracts,"price":p["mid"]},
            ]
            max_loss = float('inf')
            max_gain = c["strike"] * 100 * contracts  # stock to $0
            breakeven = c["strike"] + net_credit
            capital = c["strike"] * 100 * contracts * 0.30
            prob_profit = 0.50

        else:
            return {"error": f"Strategy {strategy_key} build not implemented"}

        # Format infinities
        def fmt_money(v):
            if v == float('inf'): return "Unlimited"
            if v == -float('inf'): return "Unlimited"
            return f"${v:,.0f}"

        return {
            "strategy": strategy_key,
            "name": meta["name"],
            "description": meta["description"],
            "bias": meta["bias"],
            "risk": meta["risk"],
            "legs": legs,
            "max_loss": max_loss,
            "max_gain": max_gain,
            "max_loss_fmt": fmt_money(max_loss),
            "max_gain_fmt": fmt_money(max_gain),
            "breakeven": breakeven,
            "prob_profit": prob_profit,
            "capital_required": capital,
            "capital_fmt": fmt_money(capital),
            "commentary": commentary,
            "expiry": chain["expiry"],
            "days_to_expiry": chain["days_to_expiry"],
            "atm_iv": chain.get("atm_iv"),
        }

    except Exception as e:
        logger.error(f"build_strategy({strategy_key}): {e}", exc_info=True)
        return {"error": f"Build failed: {e}"}


# ============================================================
# AI COMMENTARY for a strategy
# ============================================================
def get_ai_strategy_commentary(ticker: str, strategy_result: Dict,
                                stock_context: str = "") -> str:
    """LLM-generated commentary on strategy fit for the catalyst.
    Routes through the universal gateway: Anthropic → OpenAI → Google
    fallback with key rotation."""
    prompt = f"""You are an options strategist. Analyze this strategy for {ticker}:

Strategy: {strategy_result['name']}
Direction bias: {strategy_result['bias']}
Risk: {strategy_result['risk']}
Legs: {strategy_result['legs']}
Max loss: {strategy_result['max_loss_fmt']}
Max gain: {strategy_result['max_gain_fmt']}
Breakeven: {strategy_result['breakeven']}
Expected prob of profit: {strategy_result['prob_profit']:.0%}
Days to expiry: {strategy_result['days_to_expiry']}
ATM IV: {strategy_result.get('atm_iv', 'N/A')}

Stock context:
{stock_context[:1500]}

In 120 words or less, explain:
1. Why this strategy fits (or doesn't fit) the catalyst
2. Primary risk to watch
3. One adjustment to consider if the trade moves against you

Be concrete and direct."""
    try:
        from services.llm_gateway import llm_call, LLMAllProvidersFailed
        result = llm_call(
            capability="text_freeform",
            feature="strategy_commentary",
            ticker=ticker,
            prompt=prompt,
            max_tokens=400,
            temperature=0.3,
            timeout_s=35.0,
        )
        return result.text
    except LLMAllProvidersFailed as e:
        return f"**Commentary unavailable:** {len(e.attempts)} provider attempts failed"
    except Exception as e:
        return f"**Commentary unavailable:** {e}"


# ============================================================
# DEPLOY 7: TREND-AWARE STRATEGY RECOMMENDATIONS
# Rule-based ranking based on technicals + probability + IV.
# No AI call needed — instant.
# ============================================================

def _classify_trend(tech: Dict) -> str:
    """Returns 'uptrend' | 'downtrend' | 'sideways' from MA50 vs MA200 + current."""
    current = tech.get("current")
    ma50 = tech.get("ma50")
    ma200 = tech.get("ma200")
    if not current or not ma50 or not ma200:
        return "unknown"
    
    # Classic: MA50 > MA200 = uptrend, MA50 < MA200 = downtrend
    # Also require current above/below both for strong signal
    above_both = current > ma50 and current > ma200
    below_both = current < ma50 and current < ma200
    ma_up = ma50 > ma200
    
    if ma_up and above_both: return "uptrend"
    if (not ma_up) and below_both: return "downtrend"
    return "sideways"


def _classify_rsi(rsi: float) -> str:
    """Returns 'oversold' | 'neutral' | 'overbought'."""
    if rsi is None: return "neutral"
    if rsi < 30: return "oversold"
    if rsi > 70: return "overbought"
    return "neutral"


def _classify_macd(tech: Dict) -> str:
    """Returns 'bullish_cross' | 'bearish_cross' | 'neutral'."""
    macd = tech.get("macd")
    signal = tech.get("macd_signal")
    if macd is None or signal is None: return "neutral"
    if macd > signal and macd > 0: return "bullish_cross"
    if macd < signal and macd < 0: return "bearish_cross"
    return "neutral"


def _classify_iv(atm_iv: float, hist_vol: float) -> str:
    """Returns 'cheap' | 'fair' | 'expensive'.
    ATM IV vs historical vol. IV >> HV = expensive (market pricing more future move than has occurred)."""
    if atm_iv is None or hist_vol is None or hist_vol < 0.01:
        return "fair"
    ratio = atm_iv / hist_vol
    if ratio < 0.85: return "cheap"
    if ratio > 1.30: return "expensive"
    return "fair"


def recommend_strategies(tech: Dict, ai_prob: float, days_to_catalyst: int,
                         atm_iv: Optional[float], risk_profile: str = "moderate",
                         direction_hint: str = None) -> List[Dict]:
    """
    Returns top 3 ranked strategies for the current setup, each with reasoning.
    
    Output: [
        {
            "strategy_key": "bull_call_spread",
            "name": "Bull Call Spread",
            "rank": 1,
            "fit_score": 85,  # 0-100
            "reasoning": "Uptrend + RSI 54 + 81% approval prob + IV 42% (fair). Defined risk...",
            "why_chosen": ["Uptrend confirmed", "Moderate IV", "High probability catalyst"],
        }, ...
    ]
    """
    if not tech or not tech.get("available"):
        return []
    
    # Classify the environment
    trend = _classify_trend(tech)
    rsi_regime = _classify_rsi(tech.get("rsi"))
    macd_signal = _classify_macd(tech)
    iv_regime = _classify_iv(atm_iv, tech.get("hist_volatility"))
    
    # Auto-derive direction if not hinted
    if not direction_hint:
        if ai_prob >= 0.65: direction_hint = "bullish"
        elif ai_prob <= 0.40: direction_hint = "bearish"
        elif days_to_catalyst and days_to_catalyst < 45: direction_hint = "volatile"  # binary event close
        else: direction_hint = "neutral"
    
    # Score every strategy against this environment
    scored = []
    allowed = RISK_PROFILES.get(risk_profile, RISK_PROFILES["moderate"])
    
    for key, meta in OPTION_STRATEGIES.items():
        if key not in allowed:
            continue
        score, reasons = _score_strategy(key, meta, trend, rsi_regime, macd_signal,
                                         iv_regime, ai_prob, days_to_catalyst, direction_hint)
        if score > 0:
            scored.append({
                "strategy_key": key,
                "name": meta["name"],
                "fit_score": score,
                "reasoning": _build_reasoning(meta["name"], trend, rsi_regime, macd_signal,
                                               iv_regime, ai_prob, direction_hint),
                "why_chosen": reasons,
                "bias": meta.get("bias", ""),
                "iv_pref": meta.get("iv_pref", "any"),
                "risk": meta.get("risk", ""),
            })
    
    # Sort by score and take top 3
    scored.sort(key=lambda x: -x["fit_score"])
    top3 = scored[:3]
    for i, s in enumerate(top3):
        s["rank"] = i + 1
    
    return top3


def _score_strategy(key: str, meta: Dict, trend: str, rsi: str, macd: str,
                    iv: str, ai_prob: float, days: int, direction: str) -> tuple:
    """Returns (fit_score 0-100, list of reasons).
    Scoring rules capture ~30+ combinations."""
    score = 50  # base
    reasons = []
    
    bias = meta.get("bias", "").lower()
    iv_pref = meta.get("iv_pref", "any").lower()
    
    # ---- Direction alignment ----
    is_bullish_strat = "bullish" in bias or key in ("long_call", "bull_call_spread",
                                                      "cash_secured_put", "synthetic_long")
    is_bearish_strat = "bearish" in bias or key in ("long_put", "bear_put_spread",
                                                      "synthetic_short")
    is_neutral_strat = "neutral" in bias or key in ("covered_call", "iron_condor",
                                                     "butterfly", "calendar_spread")
    is_volatile_strat = key in ("long_straddle", "long_strangle", "synthetic_long",
                                 "synthetic_short")  # benefit from big moves
    
    if direction == "bullish":
        if is_bullish_strat: score += 25; reasons.append("Bullish direction alignment")
        elif is_bearish_strat: score -= 40; reasons.append("Wrong direction (bearish strategy for bullish view)")
        elif is_volatile_strat: score += 5; reasons.append("Volatile play OK but directional preferred")
    elif direction == "bearish":
        if is_bearish_strat: score += 25; reasons.append("Bearish direction alignment")
        elif is_bullish_strat: score -= 40; reasons.append("Wrong direction")
        elif is_volatile_strat: score += 5; reasons.append("Volatile play OK")
    elif direction == "volatile":
        if is_volatile_strat: score += 30; reasons.append("Binary catalyst — volatility play ideal")
        elif is_neutral_strat: score -= 20; reasons.append("Range-bound strategy poor for binary event")
    elif direction == "neutral":
        if is_neutral_strat: score += 25; reasons.append("Sideways market suits range strategy")
        elif is_volatile_strat: score -= 10
    
    # ---- Trend alignment ----
    if trend == "uptrend":
        if is_bullish_strat: score += 10; reasons.append("Trending up (MA50 > MA200)")
        elif is_bearish_strat: score -= 15
    elif trend == "downtrend":
        if is_bearish_strat: score += 10; reasons.append("Trending down")
        elif is_bullish_strat: score -= 15
    
    # ---- MACD momentum ----
    if macd == "bullish_cross":
        if is_bullish_strat: score += 10; reasons.append("MACD bullish cross")
        elif is_bearish_strat: score -= 10
    elif macd == "bearish_cross":
        if is_bearish_strat: score += 10; reasons.append("MACD bearish cross")
    
    # ---- RSI regime ----
    if rsi == "oversold":
        if key in ("long_call", "bull_call_spread", "cash_secured_put"): 
            score += 15; reasons.append("RSI oversold — bounce likely")
        if key in ("long_put", "bear_put_spread"): score -= 15
    elif rsi == "overbought":
        if key in ("long_put", "bear_put_spread", "covered_call"):
            score += 15; reasons.append("RSI overbought — pullback likely")
        if key in ("long_call", "bull_call_spread"): score -= 15
    
    # ---- IV environment ----
    if iv == "cheap":
        if iv_pref == "low" or key in ("long_call", "long_put", "long_straddle", "long_strangle"):
            score += 15; reasons.append("IV cheap — buyers favored")
        if iv_pref == "high" or key in ("iron_condor", "covered_call", "cash_secured_put"):
            score -= 20; reasons.append("IV too cheap for premium-selling strategy")
    elif iv == "expensive":
        if iv_pref == "high" or key in ("iron_condor", "covered_call", "cash_secured_put", "butterfly"):
            score += 15; reasons.append("IV expensive — premium-sellers favored")
        if iv_pref == "low" or key in ("long_call", "long_put", "long_straddle"):
            score -= 20; reasons.append("IV too expensive for premium-buying strategy")
    
    # ---- Probability edge ----
    prob_edge = abs(ai_prob - 0.5) * 2  # 0 to 1
    if prob_edge > 0.6:  # strong conviction
        if is_bullish_strat or is_bearish_strat:
            score += 10; reasons.append(f"High probability conviction ({ai_prob:.0%}) supports directional trade")
        if is_volatile_strat:
            score -= 10  # if you have strong view, direction trumps volatility
    elif prob_edge < 0.2:  # no conviction
        if is_volatile_strat:
            score += 10; reasons.append("Low directional conviction — volatility play makes sense")
        if is_bullish_strat or is_bearish_strat:
            score -= 5
    
    # ---- Days to catalyst ----
    if days and days < 30:
        if key in ("long_straddle", "long_strangle"):
            score += 10; reasons.append(f"Close to catalyst ({days}d) — buy vol")
        if key == "calendar_spread":
            score -= 20  # calendars need more time
    elif days and days > 90:
        if key in ("long_straddle", "long_strangle"):
            score -= 5  # theta decay too long
        if key in ("covered_call", "cash_secured_put"):
            score += 5; reasons.append("Long timeframe — theta collection works")
    
    return max(0, min(100, score)), reasons[:4]  # keep top 4 reasons


def _build_reasoning(strategy_name: str, trend: str, rsi: str, macd: str,
                     iv: str, ai_prob: float, direction: str) -> str:
    """Build a concise 1-sentence reasoning summary."""
    parts = []
    if trend != "unknown" and trend != "sideways":
        parts.append(f"{trend}")
    parts.append(f"RSI {rsi}")
    if macd != "neutral":
        parts.append(f"MACD {macd.replace('_', ' ')}")
    parts.append(f"IV {iv}")
    parts.append(f"{ai_prob:.0%} prob ({direction})")
    return " + ".join(parts)


# ============================================================
# DEPLOY 10: ENRICHED RECOMMENDATIONS WITH OUTCOME PROJECTIONS
# Extends recommend_strategies with:
#   - Explicit direction + volatility classification
#   - Exit timing recommendation (when to close)
#   - Anticipated P/L at approval/rejection/expected using NPV-based prices
# ============================================================

def classify_setup(tech: Dict, ai_prob: float, days_to_catalyst: int,
                   atm_iv: Optional[float]) -> Dict:
    """Produces explicit classifications the UI can show:
    - direction: bullish | bearish | volatile | stable  (what the setup predicts)
    - volatility_regime: low | moderate | high | extreme  (how much swing to expect)
    - convexity: directional | symmetric (does this need to pick a side?)
    Returns a dict with explanation strings.
    """
    if not tech or not tech.get("available"):
        return {"direction": "unknown", "volatility_regime": "unknown",
                "direction_explanation": "Technical data unavailable",
                "volatility_explanation": "Volatility data unavailable",
                "convexity": "unknown"}
    
    trend = _classify_trend(tech)
    rsi_regime = _classify_rsi(tech.get("rsi"))
    
    # ---- Direction ----
    if ai_prob >= 0.70:
        direction = "bullish"
        dexp = f"High approval probability ({ai_prob:.0%}) points to favorable outcome"
    elif ai_prob <= 0.40:
        direction = "bearish"
        dexp = f"Low approval probability ({ai_prob:.0%}) points to unfavorable outcome"
    elif days_to_catalyst and days_to_catalyst <= 45:
        direction = "volatile"
        dexp = f"Uncertain probability ({ai_prob:.0%}) + close catalyst ({days_to_catalyst}d) — binary event risk"
    elif trend == "uptrend":
        direction = "bullish"
        dexp = f"Trend-following: MA50>MA200, stock above both — bias long"
    elif trend == "downtrend":
        direction = "bearish"
        dexp = f"Trend-following: MA50<MA200, stock below both — bias short"
    else:
        direction = "stable"
        dexp = f"No clear catalyst edge, trend sideways — range-bound expectation"
    
    # ---- Volatility regime ----
    hv = tech.get("hist_volatility", 0) or 0
    iv = atm_iv or 0
    
    # Use max(IV, HV) — either implies the trading range to expect
    effective_vol = max(iv, hv)
    if effective_vol < 0.25:
        vol_regime = "low"
        vexp = f"Quiet stock (HV {hv*100:.0f}%, IV {iv*100:.0f}%) — premiums cheap"
    elif effective_vol < 0.50:
        vol_regime = "moderate"
        vexp = f"Normal biotech vol (HV {hv*100:.0f}%, IV {iv*100:.0f}%)"
    elif effective_vol < 0.85:
        vol_regime = "high"
        vexp = f"Elevated vol (HV {hv*100:.0f}%, IV {iv*100:.0f}%) — expect big swings"
    else:
        vol_regime = "extreme"
        vexp = f"Very high vol (HV {hv*100:.0f}%, IV {iv*100:.0f}%) — binary event priced in"
    
    # ---- Convexity (does this need a direction call or can it be symmetric?) ----
    convexity = "symmetric" if direction in ("volatile", "stable") else "directional"
    
    return {
        "direction": direction,
        "direction_explanation": dexp,
        "volatility_regime": vol_regime,
        "volatility_explanation": vexp,
        "convexity": convexity,
        "trend": trend,
        "rsi_regime": rsi_regime,
    }


def recommend_exit_timing(strategy_key: str, days_to_catalyst: int,
                          cat_type: str = "FDA") -> Dict:
    """Returns timing guidance for when to close the position.
    
    Logic:
    - Binary event plays (straddle/strangle): hold through event, close 1-2 days after
    - Directional plays with catalyst in <45d: hold through catalyst, close on IV crush
    - Premium-selling (IC, CSP, covered call): hold to expiry if ITM
    - Credit spreads: close at 50-70% of max profit
    - Long calls/puts: close at 2x profit or IV crush + target hit
    """
    is_close = days_to_catalyst is not None and days_to_catalyst <= 45
    
    guidance = {
        "long_call": {
            "exit_trigger": "Approval announcement or +100% profit, whichever first",
            "hold_duration": f"Through catalyst ({days_to_catalyst}d) then close within 24-48h",
            "profit_target_pct": 100,
            "stop_loss_pct": -50,
            "close_on_iv_crush": True,
        },
        "long_put": {
            "exit_trigger": "CRL/trial miss confirmed or +100% profit",
            "hold_duration": f"Through catalyst ({days_to_catalyst}d) then close within 24-48h",
            "profit_target_pct": 100,
            "stop_loss_pct": -50,
            "close_on_iv_crush": True,
        },
        "bull_call_spread": {
            "exit_trigger": "70% of max profit (at/near upper strike) OR approval confirmed",
            "hold_duration": "Hold through catalyst; close same day as announcement",
            "profit_target_pct": 70,  # of max
            "stop_loss_pct": -60,
            "close_on_iv_crush": False,
        },
        "bear_put_spread": {
            "exit_trigger": "70% of max profit OR CRL/miss confirmed",
            "hold_duration": "Hold through catalyst; close same day as announcement",
            "profit_target_pct": 70,
            "stop_loss_pct": -60,
            "close_on_iv_crush": False,
        },
        "long_straddle": {
            "exit_trigger": "Day after catalyst — close WHATEVER side is profitable, regardless of direction",
            "hold_duration": f"Hold until 1-2 days before catalyst, then hold through event",
            "profit_target_pct": 50,  # of either leg
            "stop_loss_pct": -40,
            "close_on_iv_crush": True,  # CRITICAL — IV crash hurts straddles hard
        },
        "long_strangle": {
            "exit_trigger": "Day after catalyst — close profitable side",
            "hold_duration": "Through catalyst event",
            "profit_target_pct": 50,
            "stop_loss_pct": -40,
            "close_on_iv_crush": True,
        },
        "iron_condor": {
            "exit_trigger": "50% of max credit collected, OR stock tests short strikes",
            "hold_duration": "Close 21 days before expiry regardless of P/L",
            "profit_target_pct": 50,
            "stop_loss_pct": -100,  # defined max loss
            "close_on_iv_crush": False,
        },
        "covered_call": {
            "exit_trigger": "Short call assigned (= shares called away) OR buy-back at 20% of credit",
            "hold_duration": "Through expiry, roll if stock flat",
            "profit_target_pct": 80,
            "stop_loss_pct": None,  # long stock backs it
            "close_on_iv_crush": False,
        },
        "cash_secured_put": {
            "exit_trigger": "Short put assigned (= take delivery at strike) OR buy-back at 50% of credit",
            "hold_duration": "Through expiry",
            "profit_target_pct": 50,
            "stop_loss_pct": None,
            "close_on_iv_crush": False,
        },
        "butterfly": {
            "exit_trigger": "Stock pins middle strike at expiry (max profit)",
            "hold_duration": "Hold to expiry; close early if stock moves far from center",
            "profit_target_pct": 60,
            "stop_loss_pct": -100,
            "close_on_iv_crush": False,
        },
        "calendar_spread": {
            "exit_trigger": "Front month expires near short strike — rollable",
            "hold_duration": "Hold to front-month expiry",
            "profit_target_pct": 40,
            "stop_loss_pct": -50,
            "close_on_iv_crush": False,
        },
        "diagonal_spread": {
            "exit_trigger": "Front month expires near short strike, roll short",
            "hold_duration": "Hold to front-month expiry, then re-evaluate",
            "profit_target_pct": 40,
            "stop_loss_pct": -50,
            "close_on_iv_crush": False,
        },
        "ratio_spread": {
            "exit_trigger": "Stock reaches target, or defined stop",
            "hold_duration": "Through catalyst if directional",
            "profit_target_pct": 60,
            "stop_loss_pct": -80,
            "close_on_iv_crush": False,
        },
        "collar": {
            "exit_trigger": "Long stock hits target — protective put expires",
            "hold_duration": "Long-term position protection",
            "profit_target_pct": 30,
            "stop_loss_pct": None,
            "close_on_iv_crush": False,
        },
        "synthetic_long": {
            "exit_trigger": "Target hit on stock, OR max loss triggered",
            "hold_duration": "Through expiry or target",
            "profit_target_pct": 100,
            "stop_loss_pct": -100,
            "close_on_iv_crush": False,
        },
        "synthetic_short": {
            "exit_trigger": "Target hit on stock, OR max loss triggered",
            "hold_duration": "Through expiry or target",
            "profit_target_pct": 100,
            "stop_loss_pct": -100,
            "close_on_iv_crush": False,
        },
    }
    
    result = guidance.get(strategy_key, {
        "exit_trigger": "Close when profit target hit or stop loss",
        "hold_duration": f"Through catalyst ({days_to_catalyst}d)",
        "profit_target_pct": 50,
        "stop_loss_pct": -50,
        "close_on_iv_crush": False,
    })
    
    # Add close-by-date for binary events
    if is_close and "straddle" in strategy_key or "strangle" in strategy_key:
        result["close_by_days"] = max(1, days_to_catalyst + 2)
    elif is_close:
        result["close_by_days"] = max(2, days_to_catalyst + 3)
    else:
        result["close_by_days"] = None
    
    return result


def project_strategy_pnl(strategy_result: Dict, stock_prices: Dict,
                         position_value: float = 10000) -> Dict:
    """Given a built strategy and projected stock prices (from NPV model),
    estimate P/L at each outcome.
    
    Args:
        strategy_result: output of build_strategy() — has max_loss, max_gain, breakeven, legs
        stock_prices: {"approval": X, "rejection": Y, "expected": Z, "current": C}
        position_value: dollars in the position
    
    Returns dict:
    {
        "approval": {"pnl_dollars": +1234, "pnl_pct": +50.2, "outcome": "max_profit_zone"},
        "rejection": {...},
        "expected": {...},
        "current": {...},
    }
    """
    try:
        max_loss = float(strategy_result.get("max_loss", 0) or 0)
        max_gain_raw = strategy_result.get("max_gain")
        # max_gain can be "Unlimited" or a float
        max_gain = float(max_gain_raw) if (max_gain_raw not in (None, "Unlimited", "unlimited", float("inf"))) else None
        breakeven = strategy_result.get("breakeven")
        strategy_key = strategy_result.get("strategy_key", "")
        
        # Scale by position size — build_strategy returns unit-contract values;
        # position_value ÷ (initial cost/credit) = contracts, scale accordingly
        initial_cost = abs(float(strategy_result.get("initial_cost", 0) or 0))
        contracts = max(1.0, position_value / max(100.0, initial_cost))  # 100 shares per contract
        
        max_loss_scaled = max_loss * contracts
        max_gain_scaled = max_gain * contracts if max_gain is not None else None
        
        current = float(stock_prices.get("current", 0) or 0)
        
        def estimate_pnl_at_price(target_price: float) -> Dict:
            """Piecewise-linear P/L estimate for each strategy type at expiry."""
            if target_price <= 0 or current <= 0:
                return {"pnl_dollars": 0, "pnl_pct": 0, "outcome": "n/a"}
            
            # Move % from current
            move_pct = (target_price - current) / current
            
            # Piecewise estimates by strategy family
            if "long_call" in strategy_key or strategy_key == "bull_call_spread":
                # Bullish: gain on up-moves
                if move_pct > 0.15:
                    return {"pnl_dollars": max_gain_scaled or (max_loss_scaled * 2),
                            "pnl_pct": ((max_gain_scaled or max_loss_scaled*2) / position_value) * 100,
                            "outcome": "max_profit"}
                elif move_pct > 0.05:
                    gain = (max_gain_scaled or max_loss_scaled*1.5) * 0.5
                    return {"pnl_dollars": gain, "pnl_pct": (gain/position_value)*100, "outcome": "partial_profit"}
                elif move_pct > -0.05:
                    return {"pnl_dollars": -max_loss_scaled * 0.4,
                            "pnl_pct": (-max_loss_scaled * 0.4 / position_value) * 100, "outcome": "flat_loss"}
                else:
                    return {"pnl_dollars": -max_loss_scaled,
                            "pnl_pct": (-max_loss_scaled / position_value) * 100, "outcome": "max_loss"}
            
            elif "long_put" in strategy_key or strategy_key == "bear_put_spread":
                # Bearish: gain on down-moves
                if move_pct < -0.15:
                    return {"pnl_dollars": max_gain_scaled or max_loss_scaled*2,
                            "pnl_pct": ((max_gain_scaled or max_loss_scaled*2) / position_value) * 100,
                            "outcome": "max_profit"}
                elif move_pct < -0.05:
                    gain = (max_gain_scaled or max_loss_scaled*1.5) * 0.5
                    return {"pnl_dollars": gain, "pnl_pct": (gain/position_value)*100, "outcome": "partial_profit"}
                elif move_pct < 0.05:
                    return {"pnl_dollars": -max_loss_scaled * 0.4,
                            "pnl_pct": (-max_loss_scaled * 0.4 / position_value) * 100, "outcome": "flat_loss"}
                else:
                    return {"pnl_dollars": -max_loss_scaled,
                            "pnl_pct": (-max_loss_scaled / position_value) * 100, "outcome": "max_loss"}
            
            elif "straddle" in strategy_key or "strangle" in strategy_key:
                # Volatility: gain on either direction
                if abs(move_pct) > 0.20:
                    gain = max_loss_scaled * 2  # typical 2x on big move
                    return {"pnl_dollars": gain, "pnl_pct": (gain/position_value)*100,
                            "outcome": "large_move_profit"}
                elif abs(move_pct) > 0.10:
                    gain = max_loss_scaled * 0.5
                    return {"pnl_dollars": gain, "pnl_pct": (gain/position_value)*100,
                            "outcome": "moderate_move"}
                else:
                    return {"pnl_dollars": -max_loss_scaled,
                            "pnl_pct": (-max_loss_scaled / position_value) * 100, "outcome": "no_move_max_loss"}
            
            elif strategy_key in ("iron_condor", "butterfly"):
                # Range-bound: gain if stock stays put
                if abs(move_pct) < 0.05:
                    return {"pnl_dollars": max_gain_scaled or max_loss_scaled*0.3,
                            "pnl_pct": ((max_gain_scaled or max_loss_scaled*0.3)/position_value)*100,
                            "outcome": "max_profit"}
                elif abs(move_pct) < 0.10:
                    return {"pnl_dollars": 0, "pnl_pct": 0, "outcome": "near_breakeven"}
                else:
                    return {"pnl_dollars": -max_loss_scaled,
                            "pnl_pct": (-max_loss_scaled / position_value) * 100, "outcome": "max_loss"}
            
            elif strategy_key in ("covered_call", "cash_secured_put"):
                # Premium collected; gain if stock stays in range or favorable
                if strategy_key == "covered_call":
                    if move_pct > 0:
                        gain = (max_gain_scaled or max_loss_scaled*0.2)
                        return {"pnl_dollars": gain, "pnl_pct": (gain/position_value)*100, "outcome": "premium_kept"}
                    else:
                        # long stock loses
                        loss = abs(move_pct) * position_value * 0.8  # partially offset by premium
                        return {"pnl_dollars": -loss, "pnl_pct": -loss/position_value*100, "outcome": "stock_loss"}
                else:  # cash_secured_put
                    if move_pct > -0.05:
                        gain = max_gain_scaled or max_loss_scaled*0.2
                        return {"pnl_dollars": gain, "pnl_pct": (gain/position_value)*100, "outcome": "premium_kept"}
                    else:
                        loss = (abs(move_pct) - 0.05) * position_value * 0.8
                        return {"pnl_dollars": -loss, "pnl_pct": -loss/position_value*100, "outcome": "assigned_at_lower"}
            
            else:
                # Fallback — roughly linear between max_loss and max_gain
                if move_pct > 0:
                    ratio = min(move_pct / 0.20, 1.0)
                    gain = (max_gain_scaled or max_loss_scaled*2) * ratio
                    return {"pnl_dollars": gain, "pnl_pct": (gain/position_value)*100, "outcome": "linear_up"}
                else:
                    ratio = min(abs(move_pct) / 0.20, 1.0)
                    loss = max_loss_scaled * ratio
                    return {"pnl_dollars": -loss, "pnl_pct": (-loss/position_value)*100, "outcome": "linear_down"}
        
        return {
            "approval": estimate_pnl_at_price(stock_prices.get("approval", current)),
            "rejection": estimate_pnl_at_price(stock_prices.get("rejection", current)),
            "expected": estimate_pnl_at_price(stock_prices.get("expected", current)),
            "current": {"pnl_dollars": 0, "pnl_pct": 0, "outcome": "entry"},
            "position_value": position_value,
            "contracts": contracts,
            "max_loss_scaled": max_loss_scaled,
            "max_gain_scaled": max_gain_scaled,
        }
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"project_strategy_pnl failed: {e}")
        return {"error": str(e)}
