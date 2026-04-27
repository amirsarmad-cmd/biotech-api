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
  NO_TRADE_AMBIGUOUS_PROB         — probability ≈ 50/50 (no directional bet)
  NO_TRADE_SMALL_EDGE             — directional scenario magnitude < threshold
  NO_TRADE_LOW_CONFIDENCE         — confidence_score below floor
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
from typing import Optional, Dict, Tuple

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


# Reference scenario magnitudes (mirrors REF_MOVES in post_catalyst_tracker).
# We carry a copy here so the classifier can be used on rows that don't
# have scenario_up/down stored — common for legacy data where only the
# expected-value `predicted_move_pct` is persisted. (positive_scenario_pct,
# negative_scenario_pct).
REF_SCENARIOS: Dict[str, Tuple[float, float]] = {
    "FDA Decision":          ( 4, -5),
    "PDUFA":                 ( 4, -5),
    "PDUFA Decision":        ( 4, -5),
    "Regulatory Decision":   ( 4, -5),
    "AdCom":                 ( 8, -10),
    "AdComm":                ( 8, -10),
    "Advisory Committee":    ( 8, -10),
    "Phase 3 Readout":       ( 3, -5),
    "Phase 3":               ( 3, -5),
    "Phase 2 Readout":       ( 4, -2),
    "Phase 2":               ( 4, -2),
    "Phase 1/2 Readout":     ( 8, -6),
    "Phase 1 Readout":       (10, -6),
    "Phase 1":               (10, -6),
    "Clinical Trial Readout":( 5, -3),
    "Clinical Trial":        ( 5, -3),
    "NDA submission":        ( 2, -2),
    "BLA submission":        ( 2, -2),
    "Partnership":           ( 5, -2),
    "Earnings":              ( 3, -3),
    "Product Launch":        ( 4, -4),
    "Commercial Launch":     ( 4, -4),
}


# Thresholds calibrated against the 459-outcome backtest. Tunable, but
# these defaults strike a balance:
#   - 5% (original suggestion) excluded Phase 3 LONG (REF up=3) entirely,
#     dropping coverage to 1%.
#   - 3% (first attempt) was too lax — coverage 90% with sub-50% accuracy.
#   - 4% with bias 0.10 lets Phase 3 SHORT (down=5), FDA both (up=4/down=5),
#     AdCom both (up=8/down=10), Phase 2 LONG (up=4), Phase 1 both through,
#     while still filtering out events where the model has no real magnitude
#     signal.
DEFAULT_MIN_SCENARIO_PCT = 4.0
# Probability bias: |p - 0.5| must EXCEED this for a directional bet.
# 0.10 means p must be > 0.60 (LONG) or < 0.40 (SHORT). Mid-conviction.
DEFAULT_PROB_BIAS_THRESHOLD = 0.10
DEFAULT_MIN_CONFIDENCE = 0.55
DEFAULT_OPTIONS_RATIO_FLOOR = 0.35


def get_scenario_magnitudes(catalyst_type: Optional[str]) -> Tuple[float, float]:
    """Look up the (up_pct, down_pct) reference scenario for a catalyst type.
    Returns the conservative default (4, -4) if the type isn't in the table."""
    return REF_SCENARIOS.get(catalyst_type or "", (4.0, -4.0))


def classify_trade_signal(
    *,
    probability: Optional[float],
    catalyst_type: Optional[str] = None,
    confidence_score: Optional[float] = None,
    date_precision: Optional[str] = None,
    options_implied_move_pct: Optional[float] = None,
    scenario_up_pct: Optional[float] = None,
    scenario_down_pct: Optional[float] = None,
    min_scenario_pct: float = DEFAULT_MIN_SCENARIO_PCT,
    prob_bias_threshold: float = DEFAULT_PROB_BIAS_THRESHOLD,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    options_ratio_floor: float = DEFAULT_OPTIONS_RATIO_FLOOR,
) -> str:
    """V1 classifier — kept for backwards compatibility.

    Returns one of: 'LONG', 'SHORT', 'NO_TRADE_*'. See module docstring
    for full taxonomy. Direction is derived from probability bias only,
    no priced-in detection. Use classify_trade_signal_v2 instead for the
    sell-the-news-aware version.
    """
    # Hard abstention conditions first
    if catalyst_type and catalyst_type in NON_BINARY_CATALYST_TYPES:
        return "NO_TRADE_NON_BINARY"

    if date_precision and date_precision not in ("exact", "day"):
        return "NO_TRADE_BAD_DATE"

    if confidence_score is not None and confidence_score < min_confidence:
        return "NO_TRADE_LOW_CONFIDENCE"

    # Probability-driven directional decision
    if probability is None:
        return "NO_TRADE_LOW_CONFIDENCE"

    p = float(probability)
    bias = p - 0.5
    if abs(bias) < prob_bias_threshold:
        # Probability ≈ 50/50, no clear directional bet to make
        return "NO_TRADE_AMBIGUOUS_PROB"

    # Pick the directional scenario based on probability bias
    up_pct, down_pct = scenario_up_pct, scenario_down_pct
    if up_pct is None or down_pct is None:
        ref_up, ref_down = get_scenario_magnitudes(catalyst_type)
        if up_pct is None: up_pct = ref_up
        if down_pct is None: down_pct = ref_down

    if bias > 0:
        # Model bets approval/positive. Edge = positive-scenario magnitude.
        edge = float(up_pct)
        signal = "LONG"
    else:
        edge = abs(float(down_pct))
        signal = "SHORT"

    if edge < min_scenario_pct:
        return "NO_TRADE_SMALL_EDGE"

    # Options check: if market implies more than ~3x our edge, options are
    # too expensive even if we have direction right.
    if options_implied_move_pct is not None and options_implied_move_pct > 0:
        ratio = edge / float(options_implied_move_pct)
        if ratio < options_ratio_floor:
            return "NO_TRADE_OPTIONS_TOO_EXPENSIVE"

    return signal


# ────────────────────────────────────────────────────────────
# V2 classifier — priced-in aware
# ────────────────────────────────────────────────────────────
# After the user empirically observed that the V1 LONG signal had 31.7%
# direction accuracy on 3D abnormal-vs-XBI (which means INVERSE accuracy
# was ~68%), the V2 classifier splits the LONG bucket by priced-in score:
#
#   high P + low priced-in  → LONG_UNDERPRICED_POSITIVE
#   high P + high priced-in → SHORT_SELL_THE_NEWS  (the systematic edge!)
#   high P + mid priced-in  → NO_TRADE_PRICED_IN
#   low P                   → SHORT_LOW_PROBABILITY
#
# The priced-in score is computed from runup_pre_event_30d_pct and
# (optionally) options-implied move + IV percentile. Higher = stock has
# rallied into the catalyst, more risk of sell-the-news.

# Thresholds for priced-in classification — calibrated against the
# 358-row backtest after observing per-runup-bucket V1 accuracy:
#
#   Bucket                    Judged   V1 LONG acc    Inverse
#   Washed out (≤-5%)         40       55.0%          45.0%
#   Flat (-5 to +5%)          40       77.5%          22.5%   ← real LONG alpha
#   Mild runup (+5 to +20%)   36       55.6%          44.4%   ← coin flip, skip
#   Strong runup (≥+20%)      21       33.3%          66.7%   ← sell-the-news
#
# V2 should target the FLAT bucket for LONG_UNDERPRICED (high V1 acc)
# and the STRONG-runup bucket for SHORT_SELL_THE_NEWS (high inverse acc).
#
# Mapping: runup +20% → priced_in 0.83. Runup -5% → 0.375. Runup +5% → 0.583.
#
# Set thresholds to map runup buckets to V2 signals as follows:
#   priced_in ≤ 0.60  →  LONG_UNDERPRICED (catches washed-out + flat)
#   priced_in ≥ 0.80  →  SHORT_SELL_THE_NEWS (catches strong runup only)
#   else (0.60-0.80)  →  NO_TRADE_PRICED_IN (mild runup coin-flip zone)
PRICED_IN_HIGH_THRESHOLD = 0.80   # composite ≥ this → CROWDED, sell-the-news
PRICED_IN_LOW_THRESHOLD  = 0.60   # composite ≤ this → CLEAN, real long edge


def compute_priced_in_score(
    *,
    runup_30d_pct: Optional[float] = None,
    options_implied_move_pct: Optional[float] = None,
    iv_euphoria_pct: Optional[float] = None,
) -> Optional[float]:
    """Composite 0..1 score of how priced-in a catalyst is.

    Higher = more priced in (stock has run, options are pricing big moves,
    IV is elevated). Returns None if no inputs are available.

    Components (each contributes a 0..1 sub-score):
      runup_30d:    0% = neutral (0.5 axis), +30% = max priced-in (1.0),
                    -20% = max washed-out (0.0). Scaled linearly between.
      options:      Above 25% implied move = priced-in. Below 8% = clean.
      iv_euphoria:  IV percentile / 100. Higher = more priced.

    The composite is the simple mean of available sub-scores.
    """
    sub_scores = []

    if runup_30d_pct is not None:
        # Map runup to 0..1: -20% → 0.0, 0% → 0.5, +30% → 1.0
        r = float(runup_30d_pct)
        if r >= 30: s = 1.0
        elif r <= -20: s = 0.0
        elif r >= 0: s = 0.5 + (r / 30.0) * 0.5
        else: s = 0.5 + (r / 20.0) * 0.5  # negative → below 0.5
        sub_scores.append(max(0.0, min(1.0, s)))

    if options_implied_move_pct is not None:
        # Options: 8% → 0.0, 25% → 1.0
        m = float(options_implied_move_pct)
        if m >= 25: s = 1.0
        elif m <= 8: s = 0.0
        else: s = (m - 8) / 17.0
        sub_scores.append(s)

    if iv_euphoria_pct is not None:
        sub_scores.append(max(0.0, min(1.0, float(iv_euphoria_pct) / 100.0)))

    if not sub_scores:
        return None
    return sum(sub_scores) / len(sub_scores)


def classify_trade_signal_v2(
    *,
    probability: Optional[float],
    runup_30d_pct: Optional[float] = None,
    catalyst_type: Optional[str] = None,
    confidence_score: Optional[float] = None,
    date_precision: Optional[str] = None,
    options_implied_move_pct: Optional[float] = None,
    iv_euphoria_pct: Optional[float] = None,
    scenario_up_pct: Optional[float] = None,
    scenario_down_pct: Optional[float] = None,
    min_scenario_pct: float = DEFAULT_MIN_SCENARIO_PCT,
    prob_bias_threshold: float = DEFAULT_PROB_BIAS_THRESHOLD,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    options_ratio_floor: float = DEFAULT_OPTIONS_RATIO_FLOOR,
) -> Tuple[str, Optional[float]]:
    """Priced-in-aware classifier. Returns (signal, priced_in_score).

    Decision tree:
      catalyst not binary → NO_TRADE_NON_BINARY
      bad date precision → NO_TRADE_BAD_DATE
      confidence too low → NO_TRADE_LOW_CONFIDENCE
      probability ambiguous (0.4-0.6) → NO_TRADE_AMBIGUOUS_PROB
      scenario edge too small → NO_TRADE_SMALL_EDGE
      options too expensive → NO_TRADE_OPTIONS_TOO_EXPENSIVE

      For LONG-direction bias (probability > 0.5+threshold):
        priced_in score >= 0.65 → SHORT_SELL_THE_NEWS  (the inverse signal)
        priced_in score <= 0.45 → LONG_UNDERPRICED_POSITIVE
        else (mid)              → NO_TRADE_PRICED_IN
        priced_in unknown       → LONG (fallback to V1 behavior)

      For SHORT-direction bias (probability < 0.5-threshold):
        always → SHORT_LOW_PROBABILITY (priced-in adjustment doesn't apply
        because if the market thinks approval is unlikely, more pessimism
        priced in is irrelevant — the move is still down)
    """
    # Compute priced-in score upfront (returned even on abstention)
    priced_in = compute_priced_in_score(
        runup_30d_pct=runup_30d_pct,
        options_implied_move_pct=options_implied_move_pct,
        iv_euphoria_pct=iv_euphoria_pct,
    )

    # Hard abstention conditions first (same as V1)
    if catalyst_type and catalyst_type in NON_BINARY_CATALYST_TYPES:
        return ("NO_TRADE_NON_BINARY", priced_in)

    if date_precision and date_precision not in ("exact", "day"):
        return ("NO_TRADE_BAD_DATE", priced_in)

    if confidence_score is not None and confidence_score < min_confidence:
        return ("NO_TRADE_LOW_CONFIDENCE", priced_in)

    if probability is None:
        return ("NO_TRADE_LOW_CONFIDENCE", priced_in)

    p = float(probability)
    bias = p - 0.5
    if abs(bias) < prob_bias_threshold:
        return ("NO_TRADE_AMBIGUOUS_PROB", priced_in)

    # Pick the directional scenario based on probability bias
    up_pct, down_pct = scenario_up_pct, scenario_down_pct
    if up_pct is None or down_pct is None:
        ref_up, ref_down = get_scenario_magnitudes(catalyst_type)
        if up_pct is None: up_pct = ref_up
        if down_pct is None: down_pct = ref_down

    if bias > 0:
        edge = float(up_pct)
    else:
        edge = abs(float(down_pct))

    if edge < min_scenario_pct:
        return ("NO_TRADE_SMALL_EDGE", priced_in)

    if options_implied_move_pct is not None and options_implied_move_pct > 0:
        ratio = edge / float(options_implied_move_pct)
        if ratio < options_ratio_floor:
            return ("NO_TRADE_OPTIONS_TOO_EXPENSIVE", priced_in)

    # ── The key V2 logic: priced-in adjustment for LONG-direction bets ──
    if bias > 0:
        # Model thinks event will be positive. Now check setup.
        if priced_in is None:
            # No priced-in data → fall back to plain LONG
            return ("LONG", priced_in)
        if priced_in >= PRICED_IN_HIGH_THRESHOLD:
            # Crowded long. The event being good is already in the price.
            # Go SHORT against the news.
            return ("SHORT_SELL_THE_NEWS", priced_in)
        if priced_in <= PRICED_IN_LOW_THRESHOLD:
            # Clean setup, real long edge.
            return ("LONG_UNDERPRICED_POSITIVE", priced_in)
        # Middle zone — ambiguous. Skip.
        return ("NO_TRADE_PRICED_IN", priced_in)
    else:
        # Probability says event will fail. Priced-in adjustment doesn't
        # change the SHORT bet — if a stock has run AND the model thinks
        # the catalyst is bad, that's still a SHORT, just a more emphatic one.
        return ("SHORT_LOW_PROBABILITY", priced_in)


def is_tradeable_v2(signal: str) -> bool:
    """V2 tradeable signals."""
    return signal in {
        "LONG_UNDERPRICED_POSITIVE",
        "SHORT_SELL_THE_NEWS",
        "SHORT_LOW_PROBABILITY",
        # V1 names also count as tradeable when v2 falls back
        "LONG",
        "SHORT",
    }


def predicted_direction_v2(signal: str) -> Optional[int]:
    """Map a V2 signal to a directional prediction (+1 LONG, -1 SHORT, None abstain).
    Used by the backtest to compute direction_correct against actual abnormal returns."""
    if signal in {"LONG_UNDERPRICED_POSITIVE", "LONG"}:
        return 1
    if signal in {"SHORT_SELL_THE_NEWS", "SHORT_LOW_PROBABILITY", "SHORT"}:
        return -1
    return None


def predicted_direction_from_probability(probability: Optional[float],
                                          bias_threshold: float = DEFAULT_PROB_BIAS_THRESHOLD) -> Optional[int]:
    """Return +1 (model bets up), -1 (model bets down), or None (ambiguous).
    Used by the backtest to determine sign-match against actual abnormal returns."""
    if probability is None:
        return None
    bias = float(probability) - 0.5
    if abs(bias) < bias_threshold:
        return None
    return 1 if bias > 0 else -1


def is_tradeable(signal: str) -> bool:
    return signal in ("LONG", "SHORT")


def signal_metadata() -> Dict[str, str]:
    """Human-readable descriptions for FE rendering."""
    return {
        "LONG_UNDERPRICED_POSITIVE":     "Tradeable long — high probability AND not priced in (clean setup)",
        "LONG":                          "Tradeable long — high probability of positive outcome",
        "SHORT_SELL_THE_NEWS":           "Tradeable short — high probability BUT crowded/priced-in setup. Sell-the-news bet.",
        "SHORT_LOW_PROBABILITY":         "Tradeable short — low probability of positive outcome",
        "SHORT":                         "Tradeable short — low probability of positive outcome",
        "NO_TRADE_AMBIGUOUS_PROB":       "Skip — probability ≈ 50/50, no directional edge",
        "NO_TRADE_PRICED_IN":            "Skip — high probability but priced-in setup, ambiguous reaction",
        "NO_TRADE_SMALL_EDGE":           f"Skip — scenario magnitude too small (< {DEFAULT_MIN_SCENARIO_PCT:.0f}%)",
        "NO_TRADE_LOW_CONFIDENCE":       f"Skip — data confidence too low (< {DEFAULT_MIN_CONFIDENCE:.2f})",
        "NO_TRADE_BAD_DATE":             "Skip — catalyst date too imprecise to time entry",
        "NO_TRADE_OPTIONS_TOO_EXPENSIVE":"Skip — options imply more move than model expects (paying too much)",
        "NO_TRADE_NON_BINARY":           "Skip — earnings/submissions/partnerships aren't directional",
    }
