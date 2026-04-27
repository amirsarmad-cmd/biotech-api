"""
setup_quality — score the TRADE SETUP entering a binary catalyst.

Per user feedback after NTLA Phase 3 readout: 'NTLA news was success but
stock dropped — sell-the-news on a crowded long. Need to filter on setup
quality, not just catalyst presence.'

A "good catalyst on a bad setup" is what NTLA was that day:
  - stock ran up 50%+ into the readout
  - retail euphoria (Stocktwits, options flow visible)
  - low short interest (everyone bullish already long)
  - stretched valuation, near 52w highs
  - high IV pricing in a big move

Even a perfect print can't beat the bar in that environment because
there's no marginal buyer left.

This module computes a 0-1 SETUP score across 6 axes. Each axis ranges
from 0 (crowded/euphoric/over-extended = bad long entry) to 1 (washed-out/
bearish/under-positioned = good long entry).

The score is INDEPENDENT of catalyst probability — a high-conviction
catalyst on a bad setup is still a bad trade. Both must align.

Input data sources (already fetched elsewhere):
  - yfinance .info dict          — short_pct, insider_held, 52w hi/lo
  - price history (60 days)      — run-up, volume, distance from MA
  - options_implied dict         — IV-implied expected move
  - social sentiment             — retail/Stocktwits euphoria proxy

This module does NOT make new API calls. It synthesizes signals already
available in the screener pipeline.
"""
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Axis definitions
# ────────────────────────────────────────────────────────────
# Each axis returns:
#   score:     0.0 (bad setup) to 1.0 (good setup)
#   raw:       the underlying numeric input (for tooltip / drill-down)
#   flag:      'green' | 'amber' | 'red'  — discrete tag for UI
#   note:      one-sentence human-readable explanation


def _flag(score: float) -> str:
    if score >= 0.65:
        return "green"
    if score >= 0.40:
        return "amber"
    return "red"


def _axis_runup(history: Optional[List[Dict]]) -> Dict:
    """30-day price change. Stocks up 50%+ have priced in success."""
    if not history or len(history) < 30:
        return {"score": None, "raw": None, "flag": "unknown",
                "note": "Not enough price history."}
    closes = [h.get("close") for h in history[-30:] if h.get("close")]
    if len(closes) < 20:
        return {"score": None, "raw": None, "flag": "unknown",
                "note": "Sparse price history."}
    start, end = closes[0], closes[-1]
    if start <= 0:
        return {"score": None, "raw": None, "flag": "unknown",
                "note": "Bad price data."}
    pct = (end - start) / start * 100.0
    # Score: flat or down = 1.0, +25% = 0.5, +50% = 0.0
    if pct <= 0:
        score = 1.0
    elif pct >= 50:
        score = 0.0
    else:
        score = max(0.0, 1.0 - (pct / 50.0))
    if pct > 50:
        note = f"Up {pct:.0f}% in 30 days — most gains likely priced in, sell-the-news risk."
    elif pct > 25:
        note = f"Up {pct:.0f}% in 30 days — partial run-up reduces upside on hit."
    elif pct >= 0:
        note = f"Up {pct:.0f}% in 30 days — entry not yet over-extended."
    else:
        note = f"Down {abs(pct):.0f}% in 30 days — washed-out entry, low expectations."
    return {"score": round(score, 2), "raw": round(pct, 1), "flag": _flag(score),
            "note": note}


def _axis_52w_position(week52_pos_pct: Optional[float]) -> Dict:
    """% of 52-week range. Near 52w-high means stretched.

    week52_pos_pct ranges 0-100. 0 = at 52w-low, 100 = at 52w-high.
    """
    if week52_pos_pct is None:
        return {"score": None, "raw": None, "flag": "unknown",
                "note": "No 52-week data."}
    pos = float(week52_pos_pct)
    # Score: <40% = 1.0 (deep value), >85% = 0.0 (stretched)
    if pos <= 40:
        score = 1.0
    elif pos >= 85:
        score = 0.0
    else:
        score = max(0.0, 1.0 - (pos - 40) / 45.0)
    if pos > 85:
        note = f"At {pos:.0f}% of 52-week range — near highs, stretched."
    elif pos > 60:
        note = f"At {pos:.0f}% of 52-week range — upper half, somewhat extended."
    elif pos > 30:
        note = f"At {pos:.0f}% of 52-week range — mid-range, balanced."
    else:
        note = f"At {pos:.0f}% of 52-week range — lower half, washed out."
    return {"score": round(score, 2), "raw": round(pos, 1), "flag": _flag(score),
            "note": note}


def _axis_short_interest(short_pct_float: Optional[float]) -> Dict:
    """Short interest as % of float.

    LOW SI in biotech entering a binary catalyst means everyone bullish
    is already long → no marginal buyer. HIGH SI means real bears who
    cover on positive data.

    NOTE: The mapping is INVERTED versus most other contexts. We're not
    looking for short-squeeze fuel — we're looking for trade asymmetry.
    Low SI = crowded long = bad setup for a long heading into a binary.
    """
    if short_pct_float is None:
        return {"score": None, "raw": None, "flag": "unknown",
                "note": "No short-interest data."}
    si = float(short_pct_float)
    # Score: SI<5% = 0.0 (crowded long), SI 10-20% = 0.7, SI>25% = 1.0
    if si < 5:
        score = 0.1
    elif si < 10:
        score = 0.4
    elif si < 20:
        score = 0.7
    elif si < 30:
        score = 0.9
    else:
        score = 1.0
    if si < 5:
        note = f"SI {si:.1f}% of float — very low, longs are crowded."
    elif si < 10:
        note = f"SI {si:.1f}% of float — low, limited squeeze fuel."
    elif si < 20:
        note = f"SI {si:.1f}% of float — moderate, balanced positioning."
    else:
        note = f"SI {si:.1f}% of float — heavy short interest, squeeze possible on hit."
    return {"score": round(score, 2), "raw": round(si, 1), "flag": _flag(score),
            "note": note}


def _axis_iv_euphoria(implied_move_pct: Optional[float],
                       days_to_catalyst: Optional[int]) -> Dict:
    """ATM-straddle implied move. High IV pricing = options market expects
    a big move = lottery-ticket buying = euphoric.

    Adjusted for time to expiry — 30% IV at 5 days to expiry is normal,
    30% IV at 60 days is extreme.
    """
    if implied_move_pct is None:
        return {"score": None, "raw": None, "flag": "unknown",
                "note": "No options-implied move."}
    move = float(implied_move_pct)
    # Time-adjusted: shorter DTE → higher IV is normal
    threshold_low = 15
    threshold_high = 40
    if days_to_catalyst is not None and days_to_catalyst > 0:
        # Roughly: scale thresholds by sqrt(DTE/30)
        import math
        scale = math.sqrt(days_to_catalyst / 30.0)
        threshold_low *= scale
        threshold_high *= scale
    if move <= threshold_low:
        score = 1.0
    elif move >= threshold_high:
        score = 0.0
    else:
        score = 1.0 - (move - threshold_low) / (threshold_high - threshold_low)
    if move > threshold_high:
        note = f"Options pricing ±{move:.0f}% move — euphoric, lottery-ticket buying."
    elif move > threshold_low * 1.5:
        note = f"Options pricing ±{move:.0f}% move — elevated expectations."
    else:
        note = f"Options pricing ±{move:.0f}% move — normal pre-event positioning."
    return {"score": round(score, 2), "raw": round(move, 1), "flag": _flag(score),
            "note": note}


def _axis_sentiment(social_score: Optional[float]) -> Dict:
    """Retail / Stocktwits sentiment polarity.

    social_score: -1 (extremely bearish) to +1 (extremely bullish).
    Euphoric sentiment = bad setup for long on a binary.
    """
    if social_score is None:
        return {"score": None, "raw": None, "flag": "unknown",
                "note": "No social sentiment data."}
    s = float(social_score)
    # Score: highly bullish = 0.0, neutral or bearish = 1.0
    if s >= 0.7:
        score = 0.0
    elif s <= 0.0:
        score = 1.0
    else:
        score = 1.0 - (s / 0.7)
    if s > 0.7:
        note = f"Retail sentiment {s:+.2f} — euphoric, high pile-in risk."
    elif s > 0.3:
        note = f"Retail sentiment {s:+.2f} — bullish, expectations elevated."
    elif s > 0:
        note = f"Retail sentiment {s:+.2f} — mildly positive."
    else:
        note = f"Retail sentiment {s:+.2f} — neutral or bearish, low expectations."
    return {"score": round(score, 2), "raw": round(s, 2), "flag": _flag(score),
            "note": note}


def _axis_insider_activity(info: Dict) -> Dict:
    """Insider selling in last 90 days. Net selling into a catalyst is
    a meaningful bearish signal.

    yfinance provides netSharePurchaseActivity but this is buggy. Fall
    back to insider_held % held — a proxy where high insider holdings
    = aligned interests.
    """
    insider_held = info.get("heldPercentInsiders")
    if insider_held is None:
        return {"score": None, "raw": None, "flag": "unknown",
                "note": "No insider data."}
    held = float(insider_held) * 100  # convert from 0-1 to 0-100
    # Score: insider_held >5% = aligned, <1% = no skin in the game
    if held >= 5:
        score = 0.9
    elif held >= 2:
        score = 0.7
    elif held >= 0.5:
        score = 0.5
    else:
        score = 0.3
    if held >= 5:
        note = f"Insiders own {held:.1f}% — strong alignment."
    elif held >= 2:
        note = f"Insiders own {held:.1f}% — moderate alignment."
    elif held >= 0.5:
        note = f"Insiders own {held:.1f}% — limited skin in the game."
    else:
        note = f"Insiders own {held:.1f}% — minimal alignment."
    return {"score": round(score, 2), "raw": round(held, 2), "flag": _flag(score),
            "note": note}


# ────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────
def compute_setup_quality(
    info: Optional[Dict] = None,
    history: Optional[List[Dict]] = None,
    fundamentals: Optional[Dict] = None,
    options_implied: Optional[Dict] = None,
    social_sentiment: Optional[float] = None,
    days_to_catalyst: Optional[int] = None,
) -> Dict:
    """Compute setup-quality score for a stock entering a binary catalyst.

    Args:
      info:              yfinance .info dict (52w hi/lo, insider %, etc.)
      history:           list of {date, close, ...} bars (60-day window ideal)
      fundamentals:      Fundamentals API response (short_pct, insider_pct)
      options_implied:   {implied_move_pct: ...}
      social_sentiment:  -1.0 to +1.0 polarity score
      days_to_catalyst:  days until the binary event

    Returns:
      {
        "score":       overall 0.0-1.0 (mean of populated axes),
        "flag":        'green' | 'amber' | 'red',
        "verdict":     short text headline ('Crowded long', 'Washed out', etc.),
        "axes":        {runup, week52_position, short_interest, iv_euphoria,
                        sentiment, insider_activity} → each {score, raw, flag, note},
        "warnings":    list of strings flagging concerning axes,
        "rationale":   one-paragraph synthesis,
      }
    """
    info = info or {}
    fundamentals = fundamentals or {}

    # Pull each axis input from whatever source has it
    week52_pos = (fundamentals.get("technicals") or {}).get("week_52_position_pct")
    if week52_pos is None and info:
        # fall back to yfinance .info
        hi = info.get("fiftyTwoWeekHigh")
        lo = info.get("fiftyTwoWeekLow")
        cur = info.get("currentPrice") or info.get("regularMarketPrice")
        if hi and lo and cur and hi > lo:
            week52_pos = (cur - lo) / (hi - lo) * 100.0

    short_pct = (fundamentals.get("activity") or {}).get("short_pct_float")
    if short_pct is None and info:
        # yfinance gives 0-1, scale to 0-100
        sp = info.get("shortPercentOfFloat")
        if sp is not None:
            short_pct = float(sp) * 100

    implied_move = None
    if options_implied:
        implied_move = options_implied.get("implied_move_pct")

    # Compute each axis
    axes = {
        "runup": _axis_runup(history),
        "week52_position": _axis_52w_position(week52_pos),
        "short_interest": _axis_short_interest(short_pct),
        "iv_euphoria": _axis_iv_euphoria(implied_move, days_to_catalyst),
        "sentiment": _axis_sentiment(social_sentiment),
        "insider_activity": _axis_insider_activity(info),
    }

    # Aggregate — mean of axes that have scores
    scored = [a["score"] for a in axes.values() if a.get("score") is not None]
    if not scored:
        return {
            "score": None, "flag": "unknown",
            "verdict": "Insufficient data",
            "axes": axes, "warnings": [],
            "rationale": "Not enough signal data to score the setup.",
        }
    score = sum(scored) / len(scored)
    flag = _flag(score)

    # Surface the worst 2 axes as warnings
    worst = sorted(
        [(name, a) for name, a in axes.items() if a.get("score") is not None],
        key=lambda x: x[1]["score"],
    )
    warnings = [a["note"] for _, a in worst[:2] if a["score"] < 0.4]

    # Verdict — one-line headline based on score + worst axis
    if score < 0.30:
        verdict = "Crowded long — sell-the-news risk"
    elif score < 0.50:
        verdict = "Stretched setup — partial pile-in"
    elif score < 0.70:
        verdict = "Balanced setup"
    else:
        verdict = "Clean setup — limited expectations"

    # Rationale paragraph
    bad = [name for name, a in axes.items() if a.get("score") is not None and a["score"] < 0.4]
    good = [name for name, a in axes.items() if a.get("score") is not None and a["score"] >= 0.7]
    parts = [f"Setup score {score*100:.0f}% — {verdict.lower()}."]
    if bad:
        parts.append(f"Concerning: {', '.join(bad).replace('_', ' ')}.")
    if good:
        parts.append(f"Healthy: {', '.join(good).replace('_', ' ')}.")
    rationale = " ".join(parts)

    return {
        "score": round(score, 2),
        "flag": flag,
        "verdict": verdict,
        "axes": axes,
        "warnings": warnings,
        "rationale": rationale,
    }
