"""prediction_statistical — the S3-spec statistical model factored out.

Computes a generic predicted move using:
  - regime-aware p (services.probability_lookup)
  - per-bucket move distributions (services.move_lookup)

predicted_move = p × pos_dist.median + (1 − p) × neg_dist.median

The S3 spec called this the primary model. In the user-approved hybrid
(this implementation), it's the calibration check + fallback for
catalyst types where NPV-driven doesn't apply (Phase 1, Trial Initiation,
etc.).

Returns a `StatisticalPrediction` or None if any required input is
missing. Callers (services.prediction_v2) handle abstention.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from services.disclosure_regime import classify_disclosure_regime
from services.move_lookup import (
    MoveDistribution,
    _bucketize_market_cap,
    lookup_move_distribution,
)
from services.probability_lookup import ProbabilityResult, lookup_probability

logger = logging.getLogger(__name__)


@dataclass
class StatisticalPrediction:
    move: float
    low: float
    high: float
    p: float
    p_source: str
    p_confidence: str
    p_sample_n: Optional[int]
    pos_dist: MoveDistribution
    neg_dist: MoveDistribution
    magnitude_n: int
    magnitude_fallback_level: str
    regime: str
    cap_bucket: str
    abstain_reason: Optional[str] = None  # only set on partial-failure return


def compute_statistical(
    catalyst_type: Optional[str],
    indication: Optional[str],
    ticker: Optional[str],
    catalyst_date: Optional[str],
    market_cap: Optional[float],
) -> tuple[Optional[StatisticalPrediction], Optional[str]]:
    """Returns (prediction, abstain_reason). Exactly one of the two is None.

    Abstention reasons (mirror the spec's reason_code constants):
      - NULL_REQUIRED_FIELD       catalyst_type is null
      - NO_HISTORICAL_DATA        magnitude lookup exhausted at all tiers
      - NO_PROBABILITY_SOURCE     probability lookup returned None
    """
    if not catalyst_type:
        return None, "NULL_REQUIRED_FIELD"

    regime = classify_disclosure_regime(catalyst_type)
    pres: ProbabilityResult = lookup_probability(
        regime=regime,
        catalyst_type=catalyst_type,
        indication=indication,
        ticker=ticker,
        catalyst_date=catalyst_date,
    )
    if pres.p is None:
        return None, "NO_PROBABILITY_SOURCE"

    cap_bucket = _bucketize_market_cap(market_cap)
    pos_dist = lookup_move_distribution(
        catalyst_type, indication, cap_bucket, "positive",
    )
    neg_dist = lookup_move_distribution(
        catalyst_type, indication, cap_bucket, "negative",
    )
    if pos_dist is None or neg_dist is None:
        return None, "NO_HISTORICAL_DATA"

    p = pres.p
    move = p * pos_dist.median + (1 - p) * neg_dist.median
    low = p * pos_dist.p25 + (1 - p) * neg_dist.p25
    high = p * pos_dist.p75 + (1 - p) * neg_dist.p75

    return StatisticalPrediction(
        move=move,
        low=low,
        high=high,
        p=p,
        p_source=pres.source,
        p_confidence=pres.confidence,
        p_sample_n=pres.sample_n,
        pos_dist=pos_dist,
        neg_dist=neg_dist,
        magnitude_n=min(pos_dist.n, neg_dist.n),
        magnitude_fallback_level=pos_dist.fallback_level,
        regime=regime,
        cap_bucket=cap_bucket,
    ), None
