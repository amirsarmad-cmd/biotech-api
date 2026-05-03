"""prediction_disagreement — when NPV-driven and statistical disagree
substantially, ask Opus 4.7 to reason about it and produce a verdict.

Per the user-approved plan § Decisions locked from user answers:
"When NPV vs statistical gap > 50pp: publish both, log the gap, call
Opus 4.7 with both predictions + the catalyst features, get a
structured verdict (prefer_npv / prefer_statistical / blend / abstain)
plus reasoning, persist both the verdict and the reasoning. Don't
silently pick one."

Cache verdicts on (catalyst_id, npv_payload_hash, statistical_pred_hash)
to avoid re-paying LLM cost on identical re-runs.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Optional

from services.llm_gateway import LLMAllProvidersFailed, llm_call

logger = logging.getLogger(__name__)


# Threshold for invoking the LLM resolver. Tunable via env.
import os

DISAGREEMENT_PP_THRESHOLD = float(os.getenv("PREDICTION_DISAGREEMENT_THRESHOLD_PP", "50"))


@dataclass
class DisagreementVerdict:
    verdict: str               # "prefer_npv" | "prefer_statistical" | "blend" | "abstain"
    blend_weight_npv: Optional[float]   # 0..1 if verdict == "blend"
    reasoning: str
    confidence: str            # "high" | "medium" | "low"
    llm_provider: str
    llm_model: str
    cache_hit: bool = False


_PROMPT_TMPL = """You are a biotech catalyst analyst. Two prediction models gave different answers for the same upcoming binary catalyst event. Your job is to decide which one is more trustworthy for THIS specific event, or to blend them.

CATALYST:
  Ticker: {ticker}
  Company: {company}
  Catalyst type: {catalyst_type}
  Catalyst date: {catalyst_date}
  Drug: {drug_name}
  Indication: {indication}
  Market cap (USD K): {market_cap}
  Current stock price: ${current_price}
  Probability of positive outcome (p): {p:.3f} (source: {p_source})

NPV-DRIVEN PREDICTION:
  Predicted move: {npv_move:+.2f}%
  If approved (drug NPV ${drug_npv_b:.2f}B, current_price ${npv_current:.2f}): {npv_up:+.2f}%
  If rejected: {npv_down:+.2f}%
  Priced-in fraction: {priced_in_value} ({priced_in_method}; agreed_methods={priced_in_agreed})
  NPV computed at: {npv_computed_at}

STATISTICAL PREDICTION:
  Predicted move: {stat_move:+.2f}%
  Predicted range (p25/p75 conditional blend): {stat_low:+.2f}% to {stat_high:+.2f}%
  Magnitude bucket: {stat_magnitude_n} historical events (fallback level: {stat_fallback})
  Regime: {stat_regime}, cap_bucket: {stat_cap_bucket}

GAP: {gap_pp:+.1f} percentage points

Decide ONE of the following:
  prefer_npv          — the NPV math is more trustworthy for this event (e.g. clean binary catalyst, strong drug economics, statistical bucket too generic)
  prefer_statistical  — the historical statistical model is more trustworthy (e.g. NPV inputs look implausible, bucket has strong signal, NPV math doesn't fit catalyst type)
  blend               — neither dominates; weighted average. If you choose this, also output blend_weight_npv (0.0 = all statistical, 1.0 = all NPV; 0.5 = simple average)
  abstain             — both predictions look untrustworthy or contradictory enough that publishing either would mislead users

Respond in this STRICT JSON shape (no markdown, no commentary):
{{
  "verdict": "prefer_npv" | "prefer_statistical" | "blend" | "abstain",
  "blend_weight_npv": 0.0 to 1.0 (only when verdict == "blend"; otherwise omit or null),
  "reasoning": "2-4 sentences explaining the decision; cite specific numbers from the inputs above",
  "confidence": "high" | "medium" | "low"
}}"""


def _hash_payloads(npv_pred, stat_pred) -> str:
    """Cache key — re-runs on identical inputs hit the cache."""
    parts = [
        f"{npv_pred.move:.4f}" if npv_pred else "none",
        f"{npv_pred.move_up_if_approved:.4f}" if npv_pred else "none",
        f"{npv_pred.move_down_if_rejected:.4f}" if npv_pred else "none",
        f"{stat_pred.move:.4f}" if stat_pred else "none",
        f"{stat_pred.p:.4f}" if stat_pred else "none",
        f"{stat_pred.magnitude_n}" if stat_pred else "none",
    ]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


# In-process cache. Process restart wipes it; that's fine — the cache
# is a cost-saver, not a correctness requirement.
_verdict_cache: dict[str, DisagreementVerdict] = {}


def resolve_disagreement_with_llm(
    *,
    npv_pred,                # NPVPrediction
    stat_pred,               # StatisticalPrediction
    catalyst_row: dict[str, Any],
    feature: str = "prediction_disagreement",
) -> Optional[DisagreementVerdict]:
    """Call Opus 4.7 to reason about the disagreement.

    Returns None if both predictions are missing OR the gap is below
    threshold OR the LLM call fails. Caller should treat None as
    "no verdict — keep the primary".
    """
    if npv_pred is None or stat_pred is None:
        return None
    gap = abs(npv_pred.move - stat_pred.move)
    if gap < DISAGREEMENT_PP_THRESHOLD:
        return None

    cache_key = _hash_payloads(npv_pred, stat_pred)
    if cache_key in _verdict_cache:
        cached = _verdict_cache[cache_key]
        return DisagreementVerdict(
            verdict=cached.verdict,
            blend_weight_npv=cached.blend_weight_npv,
            reasoning=cached.reasoning,
            confidence=cached.confidence,
            llm_provider=cached.llm_provider,
            llm_model=cached.llm_model,
            cache_hit=True,
        )

    prompt = _PROMPT_TMPL.format(
        ticker=catalyst_row.get("ticker") or "?",
        company=catalyst_row.get("company_name") or catalyst_row.get("ticker") or "?",
        catalyst_type=catalyst_row.get("catalyst_type") or "?",
        catalyst_date=catalyst_row.get("catalyst_date") or "?",
        drug_name=catalyst_row.get("drug_name") or "(unspecified)",
        indication=catalyst_row.get("indication") or "(unspecified)",
        market_cap=catalyst_row.get("market_cap") or "unknown",
        current_price=npv_pred.current_price,
        p=stat_pred.p,
        p_source=stat_pred.p_source,
        npv_move=npv_pred.move,
        npv_up=npv_pred.move_up_if_approved,
        npv_down=npv_pred.move_down_if_rejected,
        drug_npv_b=npv_pred.drug_npv_b,
        npv_current=npv_pred.current_price,
        priced_in_value=(
            f"{npv_pred.priced_in.value_used:.2f}"
            if npv_pred.priced_in.value_used is not None else "unavailable"
        ),
        priced_in_method=npv_pred.priced_in.method_used,
        priced_in_agreed=npv_pred.priced_in.methods_agreed,
        npv_computed_at=npv_pred.npv_computed_at,
        stat_move=stat_pred.move,
        stat_low=stat_pred.low,
        stat_high=stat_pred.high,
        stat_magnitude_n=stat_pred.magnitude_n,
        stat_fallback=stat_pred.magnitude_fallback_level,
        stat_regime=stat_pred.regime,
        stat_cap_bucket=stat_pred.cap_bucket,
        gap_pp=npv_pred.move - stat_pred.move,
    )

    try:
        result = llm_call(
            capability="text_json",
            feature=feature,
            ticker=catalyst_row.get("ticker"),
            prompt=prompt,
            # Force Anthropic with Opus 4.7 per the user's request
            fallback_chain=["anthropic", "openai", "google"],
            model_overrides={"anthropic": "claude-opus-4-7"},
            max_tokens=600,
            temperature=0.1,
            timeout_s=45.0,
        )
    except LLMAllProvidersFailed as e:
        logger.warning(
            f"prediction_disagreement: all providers failed for "
            f"{catalyst_row.get('ticker')}: {len(e.attempts)} attempts"
        )
        return None
    except Exception as e:
        logger.warning(f"prediction_disagreement unexpected: {e}")
        return None

    parsed = result.parsed_json
    if not parsed or "verdict" not in parsed:
        logger.warning(
            f"prediction_disagreement parse failed for "
            f"{catalyst_row.get('ticker')}: {(result.text or '')[:200]}"
        )
        return None

    verdict_str = str(parsed.get("verdict") or "").strip()
    if verdict_str not in ("prefer_npv", "prefer_statistical", "blend", "abstain"):
        logger.warning(f"prediction_disagreement invalid verdict {verdict_str!r}")
        return None

    blend_weight = parsed.get("blend_weight_npv")
    if verdict_str == "blend":
        try:
            blend_weight = max(0.0, min(1.0, float(blend_weight)))
        except (TypeError, ValueError):
            blend_weight = 0.5  # safe default
    else:
        blend_weight = None

    confidence = str(parsed.get("confidence") or "medium").strip().lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"

    verdict = DisagreementVerdict(
        verdict=verdict_str,
        blend_weight_npv=blend_weight,
        reasoning=str(parsed.get("reasoning") or "")[:1000],
        confidence=confidence,  # type: ignore[arg-type]
        llm_provider=result.provider,
        llm_model=result.model,
    )
    _verdict_cache[cache_key] = verdict
    return verdict
