"""prediction_v2 — orchestrates NPV-driven + statistical hybrid.

Single import surface for services/post_catalyst_tracker.py and any
future caller.

Flow (per the user-approved plan § Architecture):

  1. Validate catalyst_type (else NULL_REQUIRED_FIELD)
  2. Compute statistical prediction always (cheap; needed as primary
     fallback AND as the calibration check on NPV)
  3. If catalyst_type ∈ NPV_PRIMARY_TYPES, attempt NPV-driven
  4. Pick primary:
       NPV available + applicable type  → primary = NPV, check = statistical
       else                             → primary = statistical, check = NPV (may be None)
  5. If both predictions exist AND |gap| > DISAGREEMENT_PP_THRESHOLD,
     ask Opus 4.7 for a verdict; apply the verdict.
  6. Return Prediction (or abstain).

The Prediction dataclass mirrors the columns added by migration 020.
The orchestrator does NOT write to the DB — services/post_catalyst_tracker
does the INSERT/UPDATE behind the PREDICTION_V2_ENABLED flag.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from services.disclosure_regime import is_npv_primary_type
from services.move_lookup import _bucketize_market_cap
from services.prediction_disagreement import (
    DISAGREEMENT_PP_THRESHOLD,
    resolve_disagreement_with_llm,
)
from services.prediction_npv import NPVPrediction, compute_npv_driven
from services.prediction_statistical import (
    StatisticalPrediction,
    compute_statistical,
)

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Mirrors the v2 columns on post_catalyst_outcomes (migration 020)."""
    # Primary outputs
    move: Optional[float]
    low: Optional[float]
    high: Optional[float]
    primary_source: str           # "npv_v1" | "statistical_v1" | "npv_v1+disagreement_<verdict>"
    abstained: bool
    abstain_reason: Optional[str]

    # Both raw values (for telemetry + UI side-by-side)
    npv_move: Optional[float] = None
    statistical_move: Optional[float] = None

    # Common metadata
    p: Optional[float] = None
    p_source: Optional[str] = None
    p_confidence: Optional[str] = None
    regime: Optional[str] = None
    cap_bucket: Optional[str] = None

    # Statistical-side metadata
    magnitude_n: Optional[int] = None
    magnitude_fallback_level: Optional[str] = None

    # NPV-side metadata
    drug_npv_b: Optional[float] = None
    npv_p_approval_used: Optional[float] = None
    npv_computed_at: Optional[str] = None
    priced_in_fraction: Optional[float] = None
    priced_in_method: Optional[str] = None
    priced_in_ratio_value: Optional[float] = None
    priced_in_options_value: Optional[float] = None

    # Disagreement
    disagreement_pp: Optional[float] = None
    disagreement_verdict: Optional[str] = None
    disagreement_reasoning: Optional[str] = None
    disagreement_blend_weight_npv: Optional[float] = None


def _abstain(reason: str, partial: dict[str, Any] | None = None) -> Prediction:
    base = dict(
        move=None, low=None, high=None,
        primary_source="abstain",
        abstained=True, abstain_reason=reason,
    )
    if partial:
        base.update(partial)
    return Prediction(**base)


def compute_prediction(
    *,
    ticker: Optional[str],
    catalyst_id: Optional[int],
    catalyst_type: Optional[str],
    catalyst_date: Optional[str],
    drug_name: Optional[str],
    indication: Optional[str],
    market_cap: Optional[float],
    company_name: Optional[str] = None,
) -> Prediction:
    """Top-level call. All inputs come from the catalyst row + joined
    screener_stocks.market_cap.
    """
    if not catalyst_type:
        return _abstain("NULL_REQUIRED_FIELD")

    # 1. Statistical (always — it's the calibration check)
    stat_pred, stat_abstain_reason = compute_statistical(
        catalyst_type=catalyst_type,
        indication=indication,
        ticker=ticker,
        catalyst_date=catalyst_date,
        market_cap=market_cap,
    )

    # 2. NPV (only for applicable catalyst types)
    npv_pred: Optional[NPVPrediction] = None
    if is_npv_primary_type(catalyst_type) and stat_pred is not None:
        # Reuse the statistical p (regime-aware lookup) for the NPV blend
        npv_pred, _npv_abstain = compute_npv_driven(
            ticker=ticker,
            catalyst_id=catalyst_id,
            catalyst_date=catalyst_date,
            p=stat_pred.p,
        )

    # 3. Decide primary source and abstain if neither produced anything
    if stat_pred is None and npv_pred is None:
        return _abstain(stat_abstain_reason or "NO_PREDICTION_AVAILABLE")

    # NPV is preferred when both available AND catalyst_type is NPV-primary
    use_npv_primary = (
        npv_pred is not None
        and is_npv_primary_type(catalyst_type)
    )

    # 4. Disagreement check + LLM verdict (only if both predictions exist)
    disagreement_pp = None
    verdict = None
    if npv_pred is not None and stat_pred is not None:
        disagreement_pp = abs(npv_pred.move - stat_pred.move)
        if disagreement_pp >= DISAGREEMENT_PP_THRESHOLD:
            try:
                verdict = resolve_disagreement_with_llm(
                    npv_pred=npv_pred,
                    stat_pred=stat_pred,
                    catalyst_row={
                        "ticker": ticker,
                        "company_name": company_name,
                        "catalyst_type": catalyst_type,
                        "catalyst_date": catalyst_date,
                        "drug_name": drug_name,
                        "indication": indication,
                        "market_cap": market_cap,
                    },
                )
            except Exception as e:
                logger.warning(f"disagreement LLM resolver failed: {e}")
                verdict = None

    # 5. Apply verdict / pick primary
    primary_move: Optional[float] = None
    primary_low: Optional[float] = None
    primary_high: Optional[float] = None
    primary_source = "abstain"

    if verdict is not None:
        if verdict.verdict == "prefer_npv" and npv_pred is not None:
            primary_move, primary_low, primary_high = npv_pred.move, npv_pred.low, npv_pred.high
            primary_source = "npv_v1+prefer_npv"
        elif verdict.verdict == "prefer_statistical" and stat_pred is not None:
            primary_move, primary_low, primary_high = stat_pred.move, stat_pred.low, stat_pred.high
            primary_source = "statistical_v1+prefer_statistical"
        elif verdict.verdict == "blend" and npv_pred is not None and stat_pred is not None:
            w = verdict.blend_weight_npv if verdict.blend_weight_npv is not None else 0.5
            primary_move = w * npv_pred.move + (1 - w) * stat_pred.move
            primary_low = w * npv_pred.low + (1 - w) * stat_pred.low
            primary_high = w * npv_pred.high + (1 - w) * stat_pred.high
            primary_source = f"blend_w{w:.2f}"
        elif verdict.verdict == "abstain":
            return _abstain(
                "DISAGREEMENT_LLM_ABSTAIN",
                partial={
                    "npv_move": npv_pred.move if npv_pred else None,
                    "statistical_move": stat_pred.move if stat_pred else None,
                    "disagreement_pp": disagreement_pp,
                    "disagreement_verdict": "abstain",
                    "disagreement_reasoning": verdict.reasoning,
                },
            )

    if primary_move is None:
        # No verdict (or verdict failed) — fall through to default selection
        if use_npv_primary and npv_pred is not None:
            primary_move, primary_low, primary_high = npv_pred.move, npv_pred.low, npv_pred.high
            primary_source = "npv_v1"
        elif stat_pred is not None:
            primary_move, primary_low, primary_high = stat_pred.move, stat_pred.low, stat_pred.high
            primary_source = "statistical_v1"
        else:
            return _abstain("NO_PREDICTION_AVAILABLE")

    return Prediction(
        move=primary_move,
        low=primary_low,
        high=primary_high,
        primary_source=primary_source,
        abstained=False,
        abstain_reason=None,
        npv_move=npv_pred.move if npv_pred else None,
        statistical_move=stat_pred.move if stat_pred else None,
        p=stat_pred.p if stat_pred else (npv_pred.p_used if npv_pred else None),
        p_source=stat_pred.p_source if stat_pred else None,
        p_confidence=stat_pred.p_confidence if stat_pred else None,
        regime=stat_pred.regime if stat_pred else None,
        cap_bucket=(stat_pred.cap_bucket if stat_pred
                    else _bucketize_market_cap(market_cap)),
        magnitude_n=stat_pred.magnitude_n if stat_pred else None,
        magnitude_fallback_level=stat_pred.magnitude_fallback_level if stat_pred else None,
        drug_npv_b=npv_pred.drug_npv_b if npv_pred else None,
        npv_p_approval_used=npv_pred.npv_payload_p_approval if npv_pred else None,
        npv_computed_at=npv_pred.npv_computed_at if npv_pred else None,
        priced_in_fraction=npv_pred.priced_in.value_used if npv_pred else None,
        priced_in_method=npv_pred.priced_in.method_used if npv_pred else None,
        priced_in_ratio_value=npv_pred.priced_in.ratio_value if npv_pred else None,
        priced_in_options_value=npv_pred.priced_in.options_value if npv_pred else None,
        disagreement_pp=disagreement_pp,
        disagreement_verdict=verdict.verdict if verdict else None,
        disagreement_reasoning=verdict.reasoning if verdict else None,
        disagreement_blend_weight_npv=verdict.blend_weight_npv if verdict else None,
    )
