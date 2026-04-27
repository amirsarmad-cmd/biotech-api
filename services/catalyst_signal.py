"""
catalyst_signal — classify each catalyst as LONG / SHORT / NO_TRADE.

User pushback after seeing 52.5% direction accuracy across 358 catalysts:
  'Chasing 70% accuracy on every event is the wrong target. Real target
   is 70%+ on tradeable/high-confidence setups, with abstention on noisy
   ones. The system should not predict every catalyst — that gives you
   coin-flip results.'

This module implements the abstention layer. Each historical or upcoming
catalyst gets one of these labels:

  LONG                            — model expects positive abnormal return
  SHORT                           — model expects negative abnormal return
  NO_TRADE_SMALL_EDGE             — predicted edge < 5% (no actionable signal)
  NO_TRADE_LOW_CONFIDENCE         — confidence < 0.55 (data quality issue)
  NO_TRADE_BAD_DATE               — catalyst_date precision too fuzzy
  NO_TRADE_OPTIONS_TOO_EXPENSIVE  — predicted edge < 35% of options-implied
                                     (paying too much for the move)
  NO_TRADE_NON_BINARY             — earnings / submissions / partnerships
                                     where the EVENT itself isn't binary

Backtest accuracy is then computed ONLY on LONG/SHORT rows. Coverage
(fraction of all events that became tradeable) is reported alongside.

A useful screener should target:
  All-event direction:           ≈50-55% (noise-floor)
  Tradeable direction:           ≥65-70%  (the actual edge)
  Coverage:                       25-40%   (not too selective, not too noisy)
"""

import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


# Catalyst types that the screener treats as binary/material.
# Earnings, submissions, partnerships have predictable but small moves
# and aren't directional bets in the same sense.
BINARY_CATALYST_TYPES = {
    "FDA Decision",
    "PDUFA",
    "AdCom",
    "Phase 3 Readout",
    "Phase 3",
    "Phase 2 Readout",
    "Phase 2",
    "Phase 1 Readout",
    "Clinical Trial",
}

NON_BINARY_CATALYST_TYPES = {
    "Earnings",
    "BLA Submission",
    "NDA Submission",
    "Partnership",
    "Other",
}


# Default thresholds. Override via env if you want to tune:
DEFAULT_MIN_EDGE_PCT = 5.0          # |predicted_move| must be ≥ this
DEFAULT_MIN_CONFIDENCE = 0.55       # confidence_score must be ≥ this
DEFAULT_OPTIONS_RATIO_FLOOR = 0.35  # predicted_edge / options_implied must be ≥ this


def classify_trade_signal(
    *,
    predicted_move_pct: Optional[float],
    confidence_score: Optional[float] = None,
    catalyst_type: Optional[str] = None,
    date_precision: Optional[str] = None,
    options_implied_move_pct: Optional[float] = None,
    min_edge_pct: float = DEFAULT_MIN_EDGE_PCT,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    options_ratio_floor: float = DEFAULT_OPTIONS_RATIO_FLOOR,
) -> str:
    """Return one of: 'LONG', 'SHORT', 'NO_TRADE_*'.

    Order of checks matters — most specific reason wins so the abstention
    label is informative.
    """
    # Hard abstention conditions first
    if catalyst_type and catalyst_type in NON_BINARY_CATALYST_TYPES:
        return "NO_TRADE_NON_BINARY"

    if date_precision and date_precision not in ("exact", "day"):
        # 'quarter', 'h1', 'year', etc. → can't time the event
        return "NO_TRADE_BAD_DATE"

    if confidence_score is not None and confidence_score < min_confidence:
        return "NO_TRADE_LOW_CONFIDENCE"

    if predicted_move_pct is None:
        return "NO_TRADE_SMALL_EDGE"

    edge = abs(float(predicted_move_pct))
    if edge < min_edge_pct:
        return "NO_TRADE_SMALL_EDGE"

    # Options check: if the market expects ±30% and our model says +6%, we
    # don't have enough edge to justify long calls/puts. The 35% ratio is
    # a heuristic — empirical breakeven for ATM straddle thetas.
    if options_implied_move_pct is not None and options_implied_move_pct > 0:
        ratio = edge / float(options_implied_move_pct)
        if ratio < options_ratio_floor:
            return "NO_TRADE_OPTIONS_TOO_EXPENSIVE"

    # Made it through — directional signal
    return "LONG" if float(predicted_move_pct) > 0 else "SHORT"


def is_tradeable(signal: str) -> bool:
    return signal in ("LONG", "SHORT")


def signal_metadata() -> Dict[str, str]:
    """Human-readable descriptions for FE rendering."""
    return {
        "LONG":                          "Tradeable long — model expects positive abnormal return",
        "SHORT":                         "Tradeable short — model expects negative abnormal return",
        "NO_TRADE_SMALL_EDGE":           "Skip — model edge too small (|predicted| < 5%)",
        "NO_TRADE_LOW_CONFIDENCE":       "Skip — data confidence too low (< 0.55)",
        "NO_TRADE_BAD_DATE":             "Skip — catalyst date too imprecise to time entry",
        "NO_TRADE_OPTIONS_TOO_EXPENSIVE":"Skip — options imply more move than model expects (paying too much)",
        "NO_TRADE_NON_BINARY":           "Skip — earnings/submissions/partnerships aren't directional",
    }
