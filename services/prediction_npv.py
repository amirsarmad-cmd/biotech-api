"""prediction_npv — per-drug NPV-driven prediction.

The user's correct framing: predicted move on a specific stock should
be derived from THIS drug's NPV math, not from how generically-similar
drugs moved historically.

  approval_price  = current_price × (1 + remaining_upside_pct/100)   ← already in npv_payload
  rejection_price = current_price × (1 + rejection_pct/100)          ← already in npv_payload
  predicted_move_up   = (approval_price  / current_price) − 1
  predicted_move_down = (rejection_price / current_price) − 1
  predicted_move      = p × predicted_move_up + (1-p) × predicted_move_down

The npv_payload already accounts for `implied_move_pct` reduction
(npv_model.py:265 — remaining_upside subtracts what's already moved
into the price), so we use approval_price / rejection_price directly.

`priced_in_fraction` is computed but NOT used in the v1 magnitude
formula (see § Open implementation questions in the plan: how to
combine priced_in with NPV is itself a research thread). v1 stores
priced_in for telemetry; v2 (separate session) refines the formula
to use it.

Returns NPVPrediction or None if NPV cache is missing for this catalyst.
The caller (prediction_v2.compute_prediction) falls back to the
statistical model when None is returned.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from services.npv_model import get_npv_scenarios
from services.priced_in_calculator import PricedIn, compute_priced_in

logger = logging.getLogger(__name__)


@dataclass
class NPVPrediction:
    move: float
    low: float                  # placeholder = move; NPV doesn't natively give a range yet
    high: float
    move_up_if_approved: float
    move_down_if_rejected: float
    p_used: float
    npv_payload_p_approval: float
    drug_npv_b: float
    current_price: float
    priced_in: PricedIn
    npv_computed_at: str        # ISO timestamp from cache
    params_hash: str
    abstain_reason: Optional[str] = None


def compute_npv_driven(
    ticker: Optional[str],
    catalyst_id: Optional[int],
    catalyst_date: Optional[str],
    p: float,
) -> tuple[Optional[NPVPrediction], Optional[str]]:
    """Returns (prediction, abstain_reason). Exactly one is None.

    Abstain reasons:
      - NULL_REQUIRED_FIELD     ticker missing
      - NO_NPV_CACHE            no usable cached payload for this catalyst
      - NPV_PAYLOAD_INCOMPLETE  cache hit but missing approval/rejection price
      - INVALID_CURRENT_PRICE   current_price is zero/negative
    """
    if not ticker:
        return None, "NULL_REQUIRED_FIELD"

    payload = get_npv_scenarios(ticker, catalyst_id)
    if payload is None:
        # get_npv_scenarios returns None on either no row OR incomplete payload
        return None, "NO_NPV_CACHE"

    current_price = payload["current_price"]
    if not current_price or current_price <= 0:
        return None, "INVALID_CURRENT_PRICE"

    move_up = (payload["approval_price"] / current_price) - 1.0
    move_down = (payload["rejection_price"] / current_price) - 1.0
    move = p * move_up + (1 - p) * move_down

    # priced_in computed for telemetry — not folded into v1 magnitude
    priced_in = compute_priced_in(
        ticker=ticker,
        catalyst_id=catalyst_id,
        catalyst_date=catalyst_date,
        drug_npv_after_risk_b=payload["drug_npv_b"],
    )

    return NPVPrediction(
        move=move * 100.0,            # convert to percent like the rest of the system
        low=move * 100.0,             # NPV doesn't give a native range yet
        high=move * 100.0,
        move_up_if_approved=move_up * 100.0,
        move_down_if_rejected=move_down * 100.0,
        p_used=p,
        npv_payload_p_approval=payload["p_approval_used"],
        drug_npv_b=payload["drug_npv_b"],
        current_price=current_price,
        priced_in=priced_in,
        npv_computed_at=str(payload["computed_at"]),
        params_hash=payload["params_hash"],
    ), None
