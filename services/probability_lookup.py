"""probability_lookup — regime-aware p (probability of positive outcome).

Replaces the misuse of `confidence_score` (LLM extraction confidence,
mean 0.881) as `p` in services/post_catalyst_tracker.py:954.

Per docs/spec-03-prediction.md § p source by regime:

  MANDATED       → historical base rate, segmented by indication when n>=50,
                   else type-only when n>=20, else abstain.
  SEMI_MANDATED  → blend(historical_base_rate, options_implied_p) where
                   options weight = min(0.5, options_coverage_quality).
                   Options-missing → base rate alone, low confidence.
  VOLUNTARY      → options-implied probability if available (days_to_expiry
                   ≤ 30); else abstain (Phase 1's 93.7% positive base
                   rate is press-release survivorship — using it would
                   replicate today's broken behavior).

The "options-implied probability" stub used in v1 is the simplified
ATM-straddle inversion documented in spec § Options-implied probability
derivation. v1 returns 0.5 when only the magnitude is known (no
direction signal). The full chain-skew extraction is deferred per the
plan's "What's NOT in this plan" section.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from services.database import BiotechDatabase
from services.disclosure_regime import DisclosureRegime

logger = logging.getLogger(__name__)


PSource = Literal[
    "historical_type_indication",      # MANDATED tier-1 (best)
    "historical_type",                  # MANDATED tier-2 fallback
    "historical_type_indication+options",  # SEMI_MANDATED blend
    "historical_type+options",          # SEMI_MANDATED blend (no indication match)
    "historical_only_low_confidence",   # SEMI_MANDATED w/o options
    "options_only",                     # VOLUNTARY w/ options
    "abstain_no_data",
    "abstain_voluntary_no_options",
    "abstain_unclassified",
]
PConfidence = Literal["high", "medium", "low"]

# Sample-size thresholds. Spec § p source by regime.
MIN_N_INDICATION_SEGMENTED = 50
MIN_N_TYPE_ONLY = 20
SEMI_MANDATED_OPTIONS_MAX_WEIGHT = 0.5
DEFAULT_OPTIONS_WEIGHT = 0.3   # used when options coverage_quality not computed


@dataclass
class ProbabilityResult:
    p: Optional[float]
    source: str
    confidence: PConfidence
    note: Optional[str] = None
    sample_n: Optional[int] = None


def _historical_base_rate(
    catalyst_type: str,
    indication: Optional[str],
) -> Tuple[Optional[float], int, str]:
    """Compute base rate of positive outcome (APPROVED + MET_ENDPOINT)
    over total clean labels (positives + negatives only). Excludes
    UNKNOWN/MIXED/WITHDRAWN/DELAYED.

    Tries (type × indication) first if indication given AND n>=MIN_N_INDICATION_SEGMENTED;
    falls back to (type) alone if n>=MIN_N_TYPE_ONLY; else returns (None, 0, ...).
    """
    db = BiotechDatabase()
    with db.get_conn() as conn:
        cur = conn.cursor()

        # Tier 1: type × indication (LOWER+TRIM normalised)
        if indication:
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE outcome_label_class IN ('APPROVED','MET_ENDPOINT')) AS pos,
                    COUNT(*) FILTER (WHERE outcome_label_class IN ('APPROVED','REJECTED','MET_ENDPOINT','MISSED_ENDPOINT')) AS clean
                FROM post_catalyst_outcomes
                WHERE catalyst_type = %s
                  AND LOWER(TRIM(indication)) = LOWER(TRIM(%s))
                  AND outcome_label_confidence >= 0.7
            """, (catalyst_type, indication))
            row = cur.fetchone()
            pos = int(row[0] or 0)
            n = int(row[1] or 0)
            if n >= MIN_N_INDICATION_SEGMENTED:
                return (pos / n, n, "historical_type_indication")

        # Tier 2: type only
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE outcome_label_class IN ('APPROVED','MET_ENDPOINT')) AS pos,
                COUNT(*) FILTER (WHERE outcome_label_class IN ('APPROVED','REJECTED','MET_ENDPOINT','MISSED_ENDPOINT')) AS clean
            FROM post_catalyst_outcomes
            WHERE catalyst_type = %s
              AND outcome_label_confidence >= 0.7
        """, (catalyst_type,))
        row = cur.fetchone()
        pos = int(row[0] or 0)
        n = int(row[1] or 0)
        if n >= MIN_N_TYPE_ONLY:
            return (pos / n, n, "historical_type")

    return (None, 0, "no_data")


def _options_implied_probability(
    ticker: Optional[str],
    catalyst_date: Optional[str],
) -> Tuple[Optional[float], Optional[float], str]:
    """v1 stub: returns (probability=0.5, coverage_quality=q, note).

    The spec's simplified rule: ATM straddle gives expected magnitude
    only (no direction). We return 0.5 as the directional probability
    so the SEMI_MANDATED blend with historical doesn't shift the base
    rate. coverage_quality reflects how usable the options data is —
    used as the blend weight.

    Once full chain-skew extraction lands (deferred), this returns a
    real directional probability.
    """
    if not ticker or not catalyst_date:
        return (None, None, "no_options_inputs")
    try:
        from services.options_implied import get_implied_move
    except ImportError:
        return (None, None, "options_module_unavailable")
    try:
        result = get_implied_move(ticker, target_date=catalyst_date)
    except Exception as e:
        logger.info(f"options-implied probe {ticker}: {e}")
        return (None, None, f"options_fetch_error:{type(e).__name__}")
    if not result:
        return (None, None, "options_data_missing")

    dte = result.get("days_to_expiry") or 0
    implied_move_pct = result.get("implied_move_pct")
    if implied_move_pct is None:
        return (None, None, "options_data_partial")

    # Coverage quality: 1.0 if dte ≤ 14 AND straddle data is present,
    # degraded linearly with longer dte. v1 shortcut.
    if dte <= 14:
        quality = 1.0
    elif dte <= 30:
        quality = 0.7
    elif dte <= 60:
        quality = 0.4
    else:
        quality = 0.2

    # v1: directional probability is neutral (0.5) until skew extraction
    # ships. We still return it so the blend formula has a value to mix in.
    return (0.5, quality, f"v1_neutral_dte{dte}")


def lookup_probability(
    regime: DisclosureRegime,
    catalyst_type: Optional[str],
    indication: Optional[str],
    ticker: Optional[str] = None,
    catalyst_date: Optional[str] = None,
) -> ProbabilityResult:
    """Resolve `p` per the regime's decision tree."""
    if not catalyst_type:
        return ProbabilityResult(
            p=None, source="abstain_unclassified", confidence="low",
            note="catalyst_type is null",
        )

    if regime == "MANDATED":
        p_hist, n_hist, source = _historical_base_rate(catalyst_type, indication)
        if p_hist is None:
            return ProbabilityResult(
                p=None, source="abstain_no_data", confidence="low",
                note=f"MANDATED but no historical base rate (n={n_hist} < {MIN_N_TYPE_ONLY})",
                sample_n=n_hist,
            )
        confidence: PConfidence = (
            "high" if source == "historical_type_indication"
            else "medium"
        )
        return ProbabilityResult(p=p_hist, source=source,
                                 confidence=confidence, sample_n=n_hist)

    if regime == "SEMI_MANDATED":
        p_hist, n_hist, hist_source = _historical_base_rate(catalyst_type, indication)
        p_opt, opt_quality, opt_note = _options_implied_probability(ticker, catalyst_date)

        if p_hist is None and p_opt is None:
            return ProbabilityResult(
                p=None, source="abstain_no_data", confidence="low",
                note=f"SEMI_MANDATED with no historical (n={n_hist}) and no options ({opt_note})",
                sample_n=n_hist,
            )
        if p_opt is not None and p_hist is not None:
            opt_weight = min(SEMI_MANDATED_OPTIONS_MAX_WEIGHT, opt_quality or DEFAULT_OPTIONS_WEIGHT)
            blended = (1 - opt_weight) * p_hist + opt_weight * p_opt
            return ProbabilityResult(
                p=blended,
                source=f"{hist_source}+options",
                confidence="medium",
                note=f"blend opt_weight={opt_weight:.2f} hist={p_hist:.3f} opt={p_opt:.3f} ({opt_note})",
                sample_n=n_hist,
            )
        if p_hist is not None:
            return ProbabilityResult(
                p=p_hist, source="historical_only_low_confidence",
                confidence="low",
                note=f"SEMI_MANDATED no options ({opt_note}); base rate alone is positively biased",
                sample_n=n_hist,
            )
        # p_opt only
        return ProbabilityResult(
            p=p_opt, source="options_only", confidence="low",
            note=f"SEMI_MANDATED no historical (n={n_hist}); options stub p=0.5",
            sample_n=n_hist,
        )

    # VOLUNTARY
    p_opt, opt_quality, opt_note = _options_implied_probability(ticker, catalyst_date)
    if p_opt is None:
        return ProbabilityResult(
            p=None, source="abstain_voluntary_no_options",
            confidence="low",
            note=f"VOLUNTARY with no options data ({opt_note}); historical base rate is survivored",
        )
    return ProbabilityResult(
        p=p_opt, source="options_only",
        confidence="low" if (opt_quality or 0) < 0.7 else "medium",
        note=f"VOLUNTARY uses options stub (p=0.5 until chain-skew lands); {opt_note}",
    )
