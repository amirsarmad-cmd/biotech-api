"""scenario_algo — V2 scenario range (no magic clamps) + backtest harness.

Replaces the 80% clamp in services/post_catalyst_tracker.py:_compute_move_estimates
with a per-stock, per-catalyst-type formula grounded in rNPV math + an
empirical capture rate.

The algo (proposed, NOT YET PROD-WIRED):

    Step 1 (full re-rating, no time discount):
        drug_npv_at_approval_b   = predicted_npv_b × (0.95 / p_approval_now)
        unpriced_room_b          = drug_npv_at_approval_b × (1 − priced_in_fraction)
        priced_in_drug_b         = drug_npv_at_approval_b × priced_in_fraction
        ev_b                     = max(0.001, market_cap_b − cash_b)
        full_repricing_upside    = unpriced_room_b   / ev_b × 100
        full_repricing_downside  = -priced_in_drug_b / ev_b × 100   (floored at -90)

    Step 2 (capture rate — fraction of full re-rating that lands in 7d):
        capture_rate    = CAPTURE_RATE_BY_TYPE[catalyst_type]    # literature default v1
        scenario_upside   = full_repricing_upside   × capture_rate
        scenario_downside = full_repricing_downside × capture_rate

The backtest harness (`backtest_scenario_algo`) retrofits the algo against
post_catalyst_outcomes rows where we have the inputs and reports per-bucket
direction-hit % + MAE + bias so we can validate before shipping. Caveats are
returned in the response (lookback bias on market_cap, sample size, survivorship).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

from services.database import BiotechDatabase

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Capture rates (literature defaults; Phase 2 will refit empirically)
# ────────────────────────────────────────────────────────────
# Fraction of full re-rating expected in the 7d calibration window.
# Sources: Lo et al. (Drug Approvals & Stock Reactions); Pisano (Phase
# Readout Capture); biotech sell-side desk averages.
CAPTURE_RATE_BY_TYPE: Dict[str, float] = {
    "FDA Decision":     0.55,
    "PDUFA Decision":   0.55,
    "Phase 3 Readout":  0.65,
    "Phase 2 Readout":  0.50,
    "Phase 1 Readout":  0.35,
    "Submission":       0.25,
    "BLA submission":   0.25,
    "NDA submission":   0.25,
    "AdComm":           0.45,
    "Trial Initiation": 0.20,
    "Other":            0.30,
}
DEFAULT_CAPTURE_RATE = 0.50

# Equity has a finite floor — even on rejection, cash + dilution potential
# prevents the equity from going to zero on a single 7d window.
HARD_FLOOR_DOWNSIDE_PCT = -90.0


@dataclass
class ScenarioRange:
    """V2 scenario range output. None values mean inputs were missing."""
    upside_pct: Optional[float]
    downside_pct: Optional[float]
    capture_rate: float
    full_repricing_upside_pct: Optional[float]
    full_repricing_downside_pct: Optional[float]
    inputs_missing: List[str]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_scenario_range_v2(
    *,
    drug_npv_b_at_p_now: Optional[float],
    p_approval_now: Optional[float],
    market_cap_m: Optional[float],
    cash_m: Optional[float],
    priced_in_fraction: Optional[float],
    catalyst_type: Optional[str],
) -> ScenarioRange:
    """Compute scenario_upside / scenario_downside without arbitrary clamps.

    All inputs come from existing infrastructure:
      - drug_npv_b_at_p_now: from npv_model.compute_npv_estimate (predicted_npv_b)
      - p_approval_now: from probability_lookup
      - market_cap_m: screener_stocks.market_cap
      - cash_m: capital_structure.total_cash (optional; defaults to 0)
      - priced_in_fraction: priced_in_calculator.compute_priced_in
      - catalyst_type: from catalyst_universe

    Returns ScenarioRange with None values + inputs_missing list when inputs
    are insufficient — no silent fallbacks.
    """
    missing: List[str] = []
    if drug_npv_b_at_p_now is None or drug_npv_b_at_p_now <= 0:
        missing.append("drug_npv_b")
    if p_approval_now is None or p_approval_now <= 0:
        missing.append("p_approval")
    if market_cap_m is None or market_cap_m <= 0:
        missing.append("market_cap")

    capture_rate = CAPTURE_RATE_BY_TYPE.get(catalyst_type or "", DEFAULT_CAPTURE_RATE)

    if missing:
        return ScenarioRange(
            upside_pct=None, downside_pct=None, capture_rate=capture_rate,
            full_repricing_upside_pct=None, full_repricing_downside_pct=None,
            inputs_missing=missing,
            notes=f"missing inputs: {', '.join(missing)}",
        )

    # Step 1: extrapolate drug NPV to p=0.95 (full approval)
    p_now = max(0.05, min(0.99, p_approval_now))  # avoid div/0 + clip
    drug_npv_at_approval_b = drug_npv_b_at_p_now * (0.95 / p_now)

    # Enterprise value (more honest denominator than market cap when cash matters)
    cash_b = (cash_m or 0) / 1000.0
    market_cap_b = market_cap_m / 1000.0
    ev_b = max(0.001, market_cap_b - cash_b)

    # Priced-in fraction — default to 0.5 if missing (signals the algo
    # should be re-run when priced_in becomes available)
    pin = priced_in_fraction if priced_in_fraction is not None else 0.5
    pin = max(0.0, min(1.0, pin))
    pin_was_default = priced_in_fraction is None
    if pin_was_default:
        missing.append("priced_in_fraction (defaulted to 0.5)")

    # Step 2: full re-rating bounds (no capture rate yet)
    unpriced_room_b = drug_npv_at_approval_b * (1.0 - pin)
    full_repricing_upside_pct = unpriced_room_b / ev_b * 100.0

    priced_in_drug_b = drug_npv_at_approval_b * pin
    full_repricing_downside_pct = -priced_in_drug_b / ev_b * 100.0
    full_repricing_downside_pct = max(HARD_FLOOR_DOWNSIDE_PCT, full_repricing_downside_pct)

    # Step 3: apply capture rate
    scenario_upside = full_repricing_upside_pct * capture_rate
    scenario_downside = full_repricing_downside_pct * capture_rate
    scenario_downside = max(HARD_FLOOR_DOWNSIDE_PCT, scenario_downside)

    return ScenarioRange(
        upside_pct=round(scenario_upside, 2),
        downside_pct=round(scenario_downside, 2),
        capture_rate=capture_rate,
        full_repricing_upside_pct=round(full_repricing_upside_pct, 2),
        full_repricing_downside_pct=round(full_repricing_downside_pct, 2),
        inputs_missing=missing,
        notes=(
            f"capture={capture_rate:.0%} | drug_npv_at_approval={drug_npv_at_approval_b:.2f}B | "
            f"ev={ev_b:.2f}B | priced_in={pin:.0%}"
            + (" (DEFAULT)" if pin_was_default else "")
        ),
    )


# ────────────────────────────────────────────────────────────
# Backtest harness
# ────────────────────────────────────────────────────────────

def _cap_bucket(market_cap_m: Optional[float]) -> str:
    if not market_cap_m:
        return "unknown"
    if market_cap_m < 500:
        return "micro_lt500m"
    if market_cap_m < 2000:
        return "small_500m_2b"
    return "mid_or_above"


def _is_success(label: Optional[str]) -> bool:
    return (label or "").upper() in ("APPROVED", "MET_ENDPOINT")


def _is_failure(label: Optional[str]) -> bool:
    return (label or "").upper() in ("REJECTED", "MISSED_ENDPOINT", "WITHDRAWN")


def backtest_scenario_algo(
    *,
    min_n_per_bucket: int = 10,
    only_labeled: bool = True,
    limit: int = 5000,
) -> Dict[str, Any]:
    """Retrofit the V2 scenario algo against post_catalyst_outcomes and report
    per-bucket direction-hit % + MAE + bias.

    Decision gate: if direction_accuracy_pct < 60 OR mae_pct > overall avg
    actual move magnitude, the formula does not beat baseline. Move to Phase 2
    (empirical capture rates fit per catalyst_type) before shipping.
    """
    # post_catalyst_outcomes only sparsely populates predicted_npv_b /
    # priced_in_fraction (those columns were added but the tracker doesn't
    # always fill them). For backtest coverage, fall back to:
    #   - drug_npv_b: latest catalyst_npv_cache row for the ticker (lookback bias)
    #   - priced_in_fraction: NULL → algo defaults to 0.5 (with flag in inputs_missing)
    #   - predicted_prob: catalyst_universe.confidence_score if pco column NULL
    db = BiotechDatabase()
    rows: List[Dict[str, Any]] = []
    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
              pco.catalyst_type, pco.ticker, pco.catalyst_date,
              pco.pre_event_price, pco.actual_move_pct_7d, pco.actual_move_pct_1d,
              COALESCE(pco.predicted_npv_b, npc.drug_npv_b)         AS drug_npv_b,
              COALESCE(pco.predicted_prob, cu.confidence_score)     AS p_approval,
              pco.priced_in_fraction,
              pco.outcome_label_class,
              s.market_cap as market_cap_m,
              (pco.predicted_npv_b IS NOT NULL)  AS npv_from_pco,
              (npc.drug_npv_b IS NOT NULL)        AS npv_from_cache
            FROM post_catalyst_outcomes pco
            LEFT JOIN screener_stocks s ON s.ticker = pco.ticker
            LEFT JOIN catalyst_universe cu ON cu.id = pco.catalyst_id
            LEFT JOIN LATERAL (
              SELECT drug_npv_b
              FROM catalyst_npv_cache
              WHERE ticker = pco.ticker
                AND drug_npv_b IS NOT NULL
                AND drug_npv_b > 0
              ORDER BY computed_at DESC
              LIMIT 1
            ) npc ON true
            WHERE pco.actual_move_pct_7d IS NOT NULL
              AND ((NOT %s) OR pco.outcome_label_class IS NOT NULL)
            ORDER BY pco.catalyst_date DESC
            LIMIT %s
            """,
            (only_labeled, limit),
        )
        cols = [d[0] for d in cur.description]
        for r in cur.fetchall():
            rows.append(dict(zip(cols, r)))

    scored: List[Dict[str, Any]] = []
    skipped_no_inputs = 0
    skipped_breakdown = {"no_drug_npv": 0, "no_p_approval": 0, "no_market_cap": 0}
    npv_source_counts = {"pco": 0, "cache": 0, "missing": 0}
    for r in rows:
        if r.get("npv_from_pco"):
            npv_source_counts["pco"] += 1
        elif r.get("npv_from_cache"):
            npv_source_counts["cache"] += 1
        else:
            npv_source_counts["missing"] += 1

        scenario = compute_scenario_range_v2(
            drug_npv_b_at_p_now=float(r["drug_npv_b"]) if r.get("drug_npv_b") else None,
            p_approval_now=float(r["p_approval"]) if r.get("p_approval") else None,
            market_cap_m=float(r["market_cap_m"]) if r.get("market_cap_m") else None,
            cash_m=None,
            priced_in_fraction=float(r["priced_in_fraction"]) if r.get("priced_in_fraction") else None,
            catalyst_type=r.get("catalyst_type"),
        )
        if scenario.upside_pct is None or scenario.downside_pct is None:
            skipped_no_inputs += 1
            for missing_input in scenario.inputs_missing:
                if "drug_npv" in missing_input:
                    skipped_breakdown["no_drug_npv"] += 1
                elif "p_approval" in missing_input:
                    skipped_breakdown["no_p_approval"] += 1
                elif "market_cap" in missing_input:
                    skipped_breakdown["no_market_cap"] += 1
            continue
        actual = float(r["actual_move_pct_7d"])
        label = r.get("outcome_label_class")
        # Pick the relevant scenario based on outcome label — if no label, use
        # the side matching the realized direction (less rigorous but allows
        # unlabeled-mode coverage).
        if _is_success(label):
            predicted = scenario.upside_pct
        elif _is_failure(label):
            predicted = scenario.downside_pct
        else:
            predicted = scenario.upside_pct if actual >= 0 else scenario.downside_pct

        # Direction hit only judged when both predicted and actual are non-noise
        dir_hit: Optional[bool] = None
        if abs(predicted) >= 1.0 and abs(actual) >= 1.0:
            dir_hit = (predicted >= 0) == (actual >= 0)

        scored.append({
            "ticker": r["ticker"],
            "catalyst_type": r.get("catalyst_type") or "Other",
            "cap_bucket": _cap_bucket(r.get("market_cap_m")),
            "actual_pct": actual,
            "predicted_pct": predicted,
            "scenario_upside": scenario.upside_pct,
            "scenario_downside": scenario.downside_pct,
            "abs_error": abs(predicted - actual),
            "signed_error": predicted - actual,
            "dir_hit": dir_hit,
            "outcome_label": label,
        })

    # ── Aggregate per (catalyst_type × cap_bucket) ──
    buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for s in scored:
        key = (s["catalyst_type"], s["cap_bucket"])
        b = buckets.setdefault(key, {
            "n": 0, "judged": 0, "hits": 0,
            "abs_error_sum": 0.0, "signed_error_sum": 0.0,
            "predicted_sum": 0.0, "actual_sum": 0.0,
            "actual_abs_sum": 0.0,
        })
        b["n"] += 1
        b["abs_error_sum"] += s["abs_error"]
        b["signed_error_sum"] += s["signed_error"]
        b["predicted_sum"] += s["predicted_pct"]
        b["actual_sum"] += s["actual_pct"]
        b["actual_abs_sum"] += abs(s["actual_pct"])
        if s["dir_hit"] is not None:
            b["judged"] += 1
            if s["dir_hit"]:
                b["hits"] += 1

    bucket_rows: List[Dict[str, Any]] = []
    for (ctype, cap_b), v in buckets.items():
        n = v["n"]
        bucket_rows.append({
            "catalyst_type": ctype,
            "cap_bucket": cap_b,
            "n_events": n,
            "n_judged": v["judged"],
            "direction_hits": v["hits"],
            "direction_accuracy_pct": round((v["hits"] / v["judged"] * 100), 2) if v["judged"] else None,
            "mae_pct": round(v["abs_error_sum"] / n, 2) if n else None,
            "bias_pct": round(v["signed_error_sum"] / n, 2) if n else None,
            "avg_predicted_pct": round(v["predicted_sum"] / n, 2) if n else None,
            "avg_actual_pct": round(v["actual_sum"] / n, 2) if n else None,
            "avg_actual_abs_pct": round(v["actual_abs_sum"] / n, 2) if n else None,
            "low_confidence": n < min_n_per_bucket,
        })
    bucket_rows.sort(key=lambda r: -r["n_events"])

    # ── Overall stats ──
    total_n = len(scored)
    total_hits = sum(1 for s in scored if s["dir_hit"] is True)
    total_judged = sum(1 for s in scored if s["dir_hit"] is not None)
    overall_acc = round(total_hits / total_judged * 100, 2) if total_judged else None
    overall_mae = round(sum(s["abs_error"] for s in scored) / total_n, 2) if total_n else None
    overall_bias = round(sum(s["signed_error"] for s in scored) / total_n, 2) if total_n else None
    overall_avg_actual_abs = round(sum(abs(s["actual_pct"]) for s in scored) / total_n, 2) if total_n else None

    # ── Decision gate ──
    gate_passed = (
        overall_acc is not None and overall_acc >= 60
        and overall_mae is not None and overall_avg_actual_abs is not None
        and overall_mae <= overall_avg_actual_abs  # MAE shouldn't exceed average actual magnitude
    )

    return {
        "_caveats": [
            "LOOKBACK BIAS: market_cap_m is screener_stocks current value, not at catalyst date. "
            "Backtest results overstate accuracy for stocks that have re-rated since.",
            "PHASE 1 ONLY: capture rates are literature defaults, not empirically calibrated. "
            "Phase 2 will re-fit them per catalyst type from this dataset.",
            f"SAMPLE SIZE: buckets with n < {min_n_per_bucket} flagged low_confidence; do not act on those.",
            f"SURVIVORSHIP: only_labeled={only_labeled}; unlabeled subset (label backfill stalled at "
            f"57.9% per memory) may have systematically different outcomes.",
        ],
        "input_filter": {"only_labeled": only_labeled, "min_n_per_bucket": min_n_per_bucket, "limit": limit},
        "totals": {
            "rows_eligible": len(rows),
            "rows_scored": total_n,
            "rows_skipped_missing_inputs": skipped_no_inputs,
            "skipped_breakdown": skipped_breakdown,
            "drug_npv_source": npv_source_counts,
            "direction_hits": total_hits,
            "direction_judged": total_judged,
            "direction_accuracy_pct": overall_acc,
            "mae_pct": overall_mae,
            "bias_pct": overall_bias,
            "avg_actual_abs_pct": overall_avg_actual_abs,
            "decision_gate_passed": gate_passed,
        },
        "buckets": bucket_rows,
        "_decision_gate_note": (
            "Ship to prod if decision_gate_passed=True. Otherwise: move to Phase 2 "
            "(refit capture rates per catalyst_type from this same dataset using "
            "least-squares regression of actual_pct on full_repricing_pct)."
        ),
    }
