"""priced_in_calculator — what fraction of THIS drug's NPV is already
in the stock price?

This is a NEW concept distinct from `services/catalyst_signal.py:compute_priced_in_score`,
which is a 0-1 composite of runup + options + IV (signal of crowdedness).
The new `priced_in_fraction` answers: of the drug's risk-adjusted NPV,
what % does the current market cap reflect?

Per the user-approved plan, we compute TWO methods, store both, and
study the ratio empirically. v1 uses their average when both available;
v2 (separate session) refines based on observed disagreement vs actual
outcomes.

Method A — `cap_pipeline_split`:
    Get sibling-pipeline NPVs from catalyst_npv_cache for the same ticker.
    Attribute (market_cap - cash) proportionally to each drug's risked NPV.
    For the catalyst drug:
        priced_in = (allocated_value / drug_npv_after_risk) clamped to [0, 1]
    Fallback when sibling NPVs are missing (most cases): assume the
    catalyst drug carries 60% of the company's enterprise value
    (logged for refinement).

Method B — `options_implied`:
    v1 stub: computes the implied move % from the ATM straddle and uses
    a heuristic mapping to a priced-in fraction. Per the plan, full
    chain-skew extraction is deferred. Today's mapping:
        implied_move_pct < 5%   → priced_in ≈ 0.85 (market expects little movement → already in)
        implied_move_pct in [5,15] → linear ramp 0.85 → 0.30
        implied_move_pct >= 30  → priced_in ≈ 0.10 (market expects big movement → little priced in)
    This is a v1 heuristic. The empirical algorithm refinement is a
    research thread (per plan § What's NOT in this plan).

Returned `PricedIn` dataclass exposes both values + `value_used`
(the average when both available; otherwise whichever method returned
a value) + `methods_agreed` (gap < 0.2).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from services.database import BiotechDatabase

logger = logging.getLogger(__name__)


# When sibling-pipeline NPVs are missing, attribute this fraction of
# (market_cap - cash) to the single catalyst drug. Logged so we can
# refine empirically.
DEFAULT_SOLO_DRUG_PIPELINE_FRACTION = 0.60


@dataclass
class PricedIn:
    ratio_value: Optional[float]      # method A
    options_value: Optional[float]    # method B
    value_used: Optional[float]       # the value the prediction will use
    method_used: str                  # "ratio" / "options" / "blend" / "unavailable"
    methods_agreed: bool
    notes: str


def _fetch_market_cap_and_cash(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """Pull market_cap from screener_stocks. Cash isn't reliably populated
    in the current schema (stock_risk_factors has 10 rows; sec_financials.py
    is for UI). v1 returns cash=None; pipeline-split formula handles it.
    """
    if not ticker:
        return None, None
    try:
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT market_cap FROM screener_stocks WHERE ticker = %s",
                (ticker,),
            )
            row = cur.fetchone()
            mcap = float(row[0]) if row and row[0] is not None else None
            return mcap, None
    except Exception as e:
        logger.warning(f"_fetch_market_cap_and_cash({ticker}): {e}")
        return None, None


def _fetch_pipeline_npvs(
    ticker: str,
    exclude_catalyst_id: Optional[int],
) -> list[Tuple[int, float]]:
    """Return [(catalyst_id, drug_npv_b), ...] for cached sibling drugs
    on this ticker (excluding the focal catalyst). Used to apportion
    market_cap proportionally."""
    if not ticker:
        return []
    try:
        db = BiotechDatabase()
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT DISTINCT ON (catalyst_id) catalyst_id, drug_npv_b
                FROM catalyst_npv_cache
                WHERE ticker = %s
                  AND catalyst_id IS NOT NULL
                  AND catalyst_id IS DISTINCT FROM %s
                  AND drug_npv_b IS NOT NULL
                  AND drug_npv_b > 0
                  AND (ttl IS NULL OR ttl > NOW())
                ORDER BY catalyst_id, computed_at DESC
            """, (ticker, exclude_catalyst_id))
            return [(int(r[0]), float(r[1])) for r in cur.fetchall()]
    except Exception as e:
        logger.warning(f"_fetch_pipeline_npvs({ticker}): {e}")
        return []


def _compute_priced_in_via_cap_npv_ratio(
    ticker: str,
    catalyst_id: Optional[int],
    drug_npv_after_risk_b: Optional[float],
) -> Tuple[Optional[float], str]:
    """Method A — cap-pipeline-split.

    Returns (fraction or None, note string).
    """
    if not drug_npv_after_risk_b or drug_npv_after_risk_b <= 0:
        return None, "drug_npv_after_risk missing/zero"

    market_cap, cash = _fetch_market_cap_and_cash(ticker)
    if not market_cap or market_cap <= 0:
        return None, "market_cap unavailable from screener_stocks"

    # screener_stocks.market_cap is in thousands. Convert to $B for math.
    market_cap_b = market_cap / 1_000_000.0
    cash_b = (cash / 1_000_000.0) if cash else 0.0
    enterprise_value_b = max(0.001, market_cap_b - cash_b)

    siblings = _fetch_pipeline_npvs(ticker, catalyst_id)
    if siblings:
        # Apportion EV across [focal drug + siblings] proportionally
        total_pipeline_npv_b = drug_npv_after_risk_b + sum(n for _, n in siblings)
        if total_pipeline_npv_b <= 0:
            return None, "pipeline npv sum is zero"
        # The focal drug's "fair share" of EV = its NPV proportion × EV
        focal_share = (drug_npv_after_risk_b / total_pipeline_npv_b) * enterprise_value_b
        priced_in = focal_share / drug_npv_after_risk_b
        priced_in_clamped = max(0.0, min(1.0, priced_in))
        return (
            priced_in_clamped,
            f"pipeline_split: ev={enterprise_value_b:.2f}B, focal_npv={drug_npv_after_risk_b:.2f}B, "
            f"sibling_npv_sum={sum(n for _, n in siblings):.2f}B (n={len(siblings)})",
        )

    # No sibling NPVs cached → assume catalyst drug carries
    # DEFAULT_SOLO_DRUG_PIPELINE_FRACTION of EV
    allocated = enterprise_value_b * DEFAULT_SOLO_DRUG_PIPELINE_FRACTION
    priced_in = allocated / drug_npv_after_risk_b
    priced_in_clamped = max(0.0, min(1.0, priced_in))
    return (
        priced_in_clamped,
        f"solo_drug_fallback ({DEFAULT_SOLO_DRUG_PIPELINE_FRACTION:.0%} of "
        f"ev={enterprise_value_b:.2f}B): no sibling npvs cached",
    )


def _compute_priced_in_via_options(
    ticker: Optional[str],
    catalyst_date: Optional[str],
) -> Tuple[Optional[float], str]:
    """Method B — options-implied (v1 heuristic).

    Lower implied_move_pct → market expects less movement → more
    already priced in. Linear ramp.
    """
    if not ticker or not catalyst_date:
        return None, "no_options_inputs"
    try:
        from services.options_implied import get_implied_move
    except ImportError:
        return None, "options_module_unavailable"
    try:
        result = get_implied_move(ticker, target_date=catalyst_date)
    except Exception as e:
        return None, f"options_fetch_error:{type(e).__name__}"
    if not result:
        return None, "options_data_missing"

    implied = result.get("implied_move_pct")
    if implied is None:
        return None, "options_no_implied_move"

    implied = float(implied)
    if implied <= 5.0:
        priced_in = 0.85
    elif implied >= 30.0:
        priced_in = 0.10
    elif implied <= 15.0:
        # Ramp 5 → 15  ⇒  0.85 → 0.30
        priced_in = 0.85 - ((implied - 5.0) / 10.0) * 0.55
    else:
        # Ramp 15 → 30  ⇒  0.30 → 0.10
        priced_in = 0.30 - ((implied - 15.0) / 15.0) * 0.20

    return (
        max(0.0, min(1.0, priced_in)),
        f"v1_heuristic implied_move={implied:.1f}% → priced_in={priced_in:.2f}",
    )


def compute_priced_in(
    ticker: str,
    catalyst_id: Optional[int],
    catalyst_date: Optional[str],
    drug_npv_after_risk_b: Optional[float],
) -> PricedIn:
    """Compute both methods and pick a value to use (their average when
    both available; otherwise whichever returned a value)."""
    ratio_value, ratio_note = _compute_priced_in_via_cap_npv_ratio(
        ticker, catalyst_id, drug_npv_after_risk_b,
    )
    options_value, options_note = _compute_priced_in_via_options(
        ticker, catalyst_date,
    )

    if ratio_value is not None and options_value is not None:
        value_used = (ratio_value + options_value) / 2.0
        agreed = abs(ratio_value - options_value) < 0.2
        method_used = "blend"
        notes = (
            f"ratio={ratio_value:.2f} ({ratio_note}); "
            f"options={options_value:.2f} ({options_note}); "
            f"agreed={agreed}"
        )
    elif ratio_value is not None:
        value_used = ratio_value
        agreed = False
        method_used = "ratio"
        notes = f"ratio={ratio_value:.2f} ({ratio_note}); options unavailable: {options_note}"
    elif options_value is not None:
        value_used = options_value
        agreed = False
        method_used = "options"
        notes = f"options={options_value:.2f} ({options_note}); ratio unavailable: {ratio_note}"
    else:
        value_used = None
        agreed = False
        method_used = "unavailable"
        notes = f"neither method available; ratio: {ratio_note}; options: {options_note}"

    return PricedIn(
        ratio_value=ratio_value,
        options_value=options_value,
        value_used=value_used,
        method_used=method_used,
        methods_agreed=agreed,
        notes=notes,
    )
