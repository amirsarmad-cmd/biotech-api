"""
post_catalyst_tracker — fills post_catalyst_outcomes table for past catalysts.

Workflow:
  1. find_due_catalysts(): SELECT from catalyst_universe WHERE catalyst_date < now()
     AND no row in post_catalyst_outcomes (LEFT JOIN, IS NULL).
  2. backfill_one(catalyst_id): for a single catalyst, fetch yfinance price
     history around catalyst date, compute pre/post prices, % moves, infer
     outcome from price-action heuristic, write row.
  3. backfill_batch(limit): runs backfill_one() for up to N due catalysts.

Outcome inference (price-action heuristic, no LLM):
  - actual_move_pct_1d:
      > +20% → 'approved' (clear positive surprise typical of FDA win or beat)
      < -20% → 'rejected' (rejection or major miss)
      |move| < 5% → 'mixed' (no real reaction; data confirmed expectations or delayed news)
      otherwise → 'unknown' (will be refined by news source if available)
  - 'delayed' is rarely inferable from price alone; require news context.
"""
import os
import json
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# DB helpers
# ============================================================

def _db():
    """Get BiotechDatabase. Imported lazily to avoid circular issues."""
    from services.database import BiotechDatabase
    return BiotechDatabase()


def find_due_catalysts(limit: int = 50, min_age_days: int = 7) -> List[Dict]:
    """SELECT catalysts whose date has passed (>= min_age_days ago) and that
    DON'T yet have a post_catalyst_outcomes row.

    Returns: [{id, ticker, catalyst_type, catalyst_date, drug_name, indication,
              probability, ...}, ...]
    Limited to `limit`. Sorted oldest first (so we backfill historical first).
    """
    cutoff = date.today() - timedelta(days=min_age_days)
    cutoff_str = cutoff.isoformat()
    try:
        with _db().get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT cu.id, cu.ticker, cu.catalyst_type, cu.catalyst_date,
                       cu.drug_name, cu.indication, cu.confidence_score,
                       cu.description, cu.phase
                FROM catalyst_universe cu
                LEFT JOIN post_catalyst_outcomes pco
                    ON pco.catalyst_id = cu.id
                    OR (pco.ticker = cu.ticker
                        AND pco.catalyst_type = cu.catalyst_type
                        AND pco.catalyst_date = cu.catalyst_date)
                WHERE cu.catalyst_date <= %s
                  AND cu.catalyst_date != ''
                  AND cu.catalyst_date IS NOT NULL
                  AND pco.id IS NULL
                  AND (cu.status IS NULL OR cu.status NOT IN ('superseded', 'invalid'))
                ORDER BY cu.catalyst_date ASC
                LIMIT %s
            """, (cutoff_str, limit))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        logger.warning(f"find_due_catalysts failed: {e}")
        return []


def get_existing_outcome(ticker: str, catalyst_type: str, catalyst_date: str) -> Optional[Dict]:
    """Check if outcome row already exists."""
    try:
        with _db().get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""SELECT * FROM post_catalyst_outcomes
                           WHERE ticker=%s AND catalyst_type=%s AND catalyst_date=%s""",
                        (ticker, catalyst_type, catalyst_date))
            row = cur.fetchone()
            if not row: return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
    except Exception as e:
        logger.warning(f"get_existing_outcome failed: {e}")
        return None


def get_outcomes_for_ticker(ticker: str, limit: int = 50) -> List[Dict]:
    """Return all post_catalyst_outcomes rows for a ticker, newest first."""
    try:
        with _db().get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""SELECT * FROM post_catalyst_outcomes
                           WHERE ticker=%s
                           ORDER BY catalyst_date DESC
                           LIMIT %s""", (ticker, limit))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        logger.warning(f"get_outcomes_for_ticker failed: {e}")
        return []


def get_aggregate_accuracy(min_outcomes: int = 10) -> Dict:
    """Compute system-wide prediction accuracy stats from filled rows.
    Useful for displaying 'our model has been right X% of the time on direction'."""
    try:
        with _db().get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE direction_correct) AS direction_hits,
                    AVG(error_abs_pct) AS avg_abs_error_pct,
                    AVG(error_signed_pct) AS avg_signed_error_pct,
                    COUNT(*) FILTER (WHERE outcome = 'approved') AS approved_count,
                    COUNT(*) FILTER (WHERE outcome = 'rejected') AS rejected_count,
                    COUNT(*) FILTER (WHERE outcome = 'mixed') AS mixed_count,
                    COUNT(*) FILTER (WHERE outcome = 'unknown') AS unknown_count
                FROM post_catalyst_outcomes
                WHERE actual_move_pct_30d IS NOT NULL
                  AND predicted_move_pct IS NOT NULL
            """)
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
            d = dict(zip(cols, row))
            total = int(d.get("total") or 0)
            d["direction_accuracy_pct"] = (
                round(100.0 * (d.get("direction_hits") or 0) / total, 1)
                if total >= min_outcomes else None
            )
            return d
    except Exception as e:
        logger.warning(f"get_aggregate_accuracy failed: {e}")
        return {"total": 0, "error": str(e)[:200]}


# ============================================================
# Price fetching (yfinance)
# ============================================================

def _fetch_price_window(ticker: str, catalyst_date_str: str) -> Optional[Dict]:
    """Fetch price history covering pre_event (~5 trading days before) through
    day30 post-catalyst. Returns dict with parsed prices or None on failure.
    """
    try:
        import yfinance as yf
        from datetime import datetime as _dt
        cdt = _dt.strptime(catalyst_date_str[:10], "%Y-%m-%d").date()
        # Pull a wide window: 30 calendar days before to 45 after
        start = cdt - timedelta(days=30)
        end = cdt + timedelta(days=45)
        # Cap end at today (can't fetch future prices)
        today = date.today()
        if end > today:
            end = today

        t = yf.Ticker(ticker)
        hist = t.history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), auto_adjust=True)
        if hist is None or hist.empty:
            return None

        # Index has tz-aware dates from yfinance — strip tz, convert to date keys
        hist.index = hist.index.tz_localize(None) if hist.index.tz else hist.index
        rows_by_date = {dt.date(): {
            "close": float(row["Close"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "volume": float(row["Volume"]),
        } for dt, row in hist.iterrows()}

        if not rows_by_date:
            return None

        sorted_dates = sorted(rows_by_date.keys())

        def _on_or_before(target: date) -> Optional[date]:
            """Find latest date <= target."""
            candidates = [d for d in sorted_dates if d <= target]
            return candidates[-1] if candidates else None

        def _on_or_after(target: date) -> Optional[date]:
            """Find earliest date >= target."""
            candidates = [d for d in sorted_dates if d >= target]
            return candidates[0] if candidates else None

        # Pre-event: last trading day BEFORE catalyst_date
        pre_event_d = _on_or_before(cdt - timedelta(days=1))
        # Day0: catalyst_date itself, or first trading day at/after if catalyst was a weekend
        day0_d = _on_or_after(cdt)
        # Day1: next trading day after day0
        day1_d = _on_or_after(cdt + timedelta(days=1)) if day0_d else None
        if day1_d == day0_d and day0_d:
            # Same day — bump to next available
            following = [d for d in sorted_dates if d > day0_d]
            day1_d = following[0] if following else None
        # Day7: ~5 trading days after day0 → ~7 calendar
        day7_d = _on_or_after(cdt + timedelta(days=7))
        # Day30: ~22 trading days after day0 → ~30 calendar
        day30_d = _on_or_after(cdt + timedelta(days=30))

        # Avg volume over 30 days BEFORE event
        pre_dates = [d for d in sorted_dates if d < cdt and (cdt - d).days <= 30]
        avg_pre_vol = (sum(rows_by_date[d]["volume"] for d in pre_dates) / len(pre_dates)) if pre_dates else None

        def _close(d): return rows_by_date[d]["close"] if d else None

        pre_event_price = _close(pre_event_d)
        day0_price = _close(day0_d)
        day1_price = _close(day1_d)
        day7_price = _close(day7_d)
        day30_price = _close(day30_d)

        # Max intraday move: |day0_high - pre_event_price| / pre_event_price (if day0 exists)
        max_intraday_pct = None
        if day0_d and pre_event_price:
            day0 = rows_by_date[day0_d]
            move_high = (day0["high"] - pre_event_price) / pre_event_price * 100.0
            move_low = (day0["low"] - pre_event_price) / pre_event_price * 100.0
            max_intraday_pct = max(abs(move_high), abs(move_low)) * (1 if abs(move_high) > abs(move_low) else -1)

        return {
            "pre_event_date": pre_event_d.isoformat() if pre_event_d else None,
            "pre_event_price": pre_event_price,
            "day0_price": day0_price,
            "day1_price": day1_price,
            "day7_price": day7_price,
            "day30_price": day30_price,
            "preevent_avg_volume_30d": avg_pre_vol,
            "postevent_volume_1d": rows_by_date[day0_d]["volume"] if day0_d else None,
            "postevent_max_intraday_move_pct": max_intraday_pct,
            "_data_present": all([pre_event_price, day0_price]),
        }
    except Exception as e:
        logger.warning(f"_fetch_price_window failed for {ticker}@{catalyst_date_str}: {e}")
        return None


# ============================================================
# Outcome inference
# ============================================================

def _infer_outcome(
    catalyst_type: str,
    move_1d: Optional[float],
    move_7d: Optional[float],
    move_30d: Optional[float],
    intraday_max_pct: Optional[float] = None,
    volume_ratio: Optional[float] = None,
) -> Tuple[str, float, str]:
    """Heuristic outcome inference from price action.
    Returns (outcome, confidence_0_to_1, basis_text).
    
    Improvements over v1:
    - Intraday max move: detects 'priced-in' approvals where close is small but
      intraday shows the news landed (high range = real reaction)
    - Volume ratio: high volume on small move = priced-in approval, not delay
    """
    if move_1d is None and move_30d is None:
        return ("unknown", 0.0, "no price data")

    primary = move_1d if move_1d is not None else move_30d
    is_binary = catalyst_type and any(k in (catalyst_type or "").lower()
                                       for k in ["fda", "pdufa", "adcomm", "advisory", "decision"])

    if primary is None:
        return ("unknown", 0.0, "no 1d/30d move")

    high_volume = volume_ratio is not None and volume_ratio > 2.0
    very_high_volume = volume_ratio is not None and volume_ratio > 4.0
    intraday_was_big = intraday_max_pct is not None and abs(intraday_max_pct) > 8
    vol_str = f"{volume_ratio:.1f}" if volume_ratio is not None else "?"

    # Binary catalysts (FDA-type): clean splits
    if is_binary:
        if primary > 25:
            return ("approved", 0.85, f"+{primary:.1f}% on day1, typical FDA approval reaction")
        if primary > 12:
            return ("approved", 0.65, f"+{primary:.1f}% on day1, likely positive outcome")
        if primary < -25:
            return ("rejected", 0.85, f"{primary:.1f}% on day1, typical FDA rejection reaction")
        if primary < -12:
            return ("rejected", 0.65, f"{primary:.1f}% on day1, likely negative outcome")

        # Small moves — disambiguate priced-in vs delayed
        if abs(primary) < 5:
            # High volume + intraday range → priced-in approval (positive direction)
            if (high_volume or intraday_was_big) and primary >= -1:
                if intraday_max_pct is not None and intraday_max_pct > 3:
                    return ("approved", 0.55, f"{primary:+.1f}% close, intraday +{intraday_max_pct:.1f}%, vol×{vol_str} — likely priced-in approval")
                if very_high_volume and primary > 0:
                    return ("approved", 0.50, f"{primary:+.1f}% close on vol×{vol_str} — high-volume small move suggests priced-in")
            # High volume + intraday range → priced-in rejection (negative direction)
            if (high_volume or intraday_was_big) and primary <= 1:
                if intraday_max_pct is not None and intraday_max_pct < -3:
                    return ("rejected", 0.50, f"{primary:+.1f}% close, intraday {intraday_max_pct:.1f}%, vol×{vol_str} — likely priced-in rejection")
            # Low volume + small move → genuinely delayed/inconclusive
            return ("delayed", 0.45, f"{primary:+.1f}% close, vol×{vol_str} — small reaction, likely delay/inconclusive")
        return ("mixed", 0.40, f"{primary:+.1f}% — moderate reaction, ambiguous")

    # Phase-readout / earnings / other: looser
    if primary > 15:
        return ("approved", 0.55, f"+{primary:.1f}% on day1, positive reaction")
    if primary < -15:
        return ("rejected", 0.55, f"{primary:.1f}% on day1, negative reaction")
    return ("mixed", 0.40, f"{primary:+.1f}% — modest reaction")


# ============================================================
# Backfill orchestration
# ============================================================

def _classify_outcome_with_llm(
    ticker: str,
    catalyst_type: Optional[str],
    catalyst_date: str,
    drug_name: Optional[str],
    indication: Optional[str],
    move_1d: Optional[float],
    move_30d: Optional[float],
    volume_ratio: Optional[float],
) -> Optional[Tuple[str, float, str]]:
    """LLM fallback classifier for ambiguous price-action cases.
    
    Used when _infer_outcome returns low-confidence (< 0.5) — typically when
    the catalyst was priced-in (small close, small intraday) and price action
    alone can't tell us if the underlying news was good/bad/delayed.
    
    The LLM has training-data knowledge of historical FDA decisions and clinical
    readouts, which can resolve cases where price action is ambiguous but the
    actual outcome is well-documented (e.g. Mounjaro 2022 — clearly approved
    but priced-in days before).
    
    Returns (outcome, confidence, basis) or None if LLM call fails.
    """
    if not catalyst_type or not catalyst_date:
        return None
    
    drug_label = drug_name or "(unspecified drug)"
    indication_label = indication or "(unspecified indication)"
    move_1d_str = f"{move_1d:+.1f}%" if move_1d is not None else "unknown"
    move_30d_str = f"{move_30d:+.1f}%" if move_30d is not None else "unknown"
    vol_str = f"{volume_ratio:.1f}x avg" if volume_ratio else "unknown"
    
    prompt = f"""You are classifying the historical outcome of a biotech catalyst for a learning-loop dataset.

TICKER: {ticker}
CATALYST TYPE: {catalyst_type}
CATALYST DATE: {catalyst_date}
DRUG: {drug_label}
INDICATION: {indication_label}

OBSERVED PRICE ACTION:
- 1-day post-catalyst move: {move_1d_str}
- 30-day post-catalyst move: {move_30d_str}
- Day-0 volume vs 30-day avg: {vol_str}

Based on your knowledge of what actually happened with this catalyst (drug approval status, trial readout outcome, etc), classify the OUTCOME of this specific event.

CLASSIFICATION RULES:
- "approved": the catalyst delivered a positive outcome (FDA approval, positive trial readout, label expansion granted, etc.). This applies even if the stock didn't move much because the result was already priced-in (common for high-probability events like Mounjaro 2022 where approval was widely expected).
- "rejected": negative outcome (FDA rejection / CRL, failed trial, missed primary endpoint, label denied).
- "delayed": event genuinely did not happen on this date (delayed PDUFA, postponed AdComm, trial readout pushed). Different from "small price reaction" — this requires evidence the event itself was delayed.
- "mixed": partial approval, met primary but failed secondary, narrow label, conditional approval, etc.
- "unknown": you genuinely don't have information about what happened.

Respond with ONLY a JSON object:
{{
  "outcome": "approved" | "rejected" | "delayed" | "mixed" | "unknown",
  "confidence": 0.0 to 1.0,
  "rationale": "1-sentence explanation citing the specific event outcome (not the price action)"
}}

Do NOT default to the price action. If you don't know what happened, return "unknown" with low confidence rather than guessing from the move %."""
    
    try:
        from services.llm_helper import call_llm_json
        result, err = call_llm_json(
            prompt,
            max_tokens=300,
            temperature=0.1,
            feature="outcome_classifier",
            ticker=ticker,
        )
        if not result or err:
            logger.info(f"[outcome-llm] {ticker}@{catalyst_date}: {err}")
            return None
        
        outcome = (result.get("outcome") or "unknown").lower().strip()
        if outcome not in ("approved", "rejected", "delayed", "mixed", "unknown"):
            return None
        
        confidence = float(result.get("confidence") or 0.0)
        confidence = max(0.0, min(1.0, confidence))
        rationale = str(result.get("rationale") or "")[:300]
        provider = result.get("_llm_provider", "?")
        basis = f"LLM[{provider}]: {rationale}"
        
        return (outcome, confidence, basis)
    except Exception as e:
        logger.warning(f"_classify_outcome_with_llm failed: {e}")
        return None


def backfill_one(catalyst: Dict) -> Dict:
    """Backfill a single post_catalyst_outcomes row.
    Returns dict {status: 'created'|'skipped'|'failed', reason, ...}.
    """
    ticker = catalyst.get("ticker")
    cat_type = catalyst.get("catalyst_type")
    cat_date = catalyst.get("catalyst_date")
    cat_id = catalyst.get("id")

    if not ticker or not cat_date:
        return {"status": "failed", "reason": "missing ticker or catalyst_date"}

    # Skip earnings — too noisy for the inference heuristic
    if cat_type and "earnings" in cat_type.lower():
        return {"status": "skipped", "reason": "earnings catalyst skipped (too noisy for outcome inference)"}

    # Fetch yfinance prices
    pw = _fetch_price_window(ticker, cat_date)
    if not pw or not pw.get("_data_present"):
        # Still write a stub row so we don't keep retrying
        try:
            _write_stub(catalyst, error="no price data from yfinance")
        except Exception as e:
            logger.warning(f"stub write failed: {e}")
        return {"status": "failed", "reason": "no price data", "ticker": ticker, "date": cat_date}

    pre = pw["pre_event_price"]
    move_1d = ((pw["day1_price"] - pre) / pre * 100.0) if (pw.get("day1_price") and pre) else None
    move_7d = ((pw["day7_price"] - pre) / pre * 100.0) if (pw.get("day7_price") and pre) else None
    move_30d = ((pw["day30_price"] - pre) / pre * 100.0) if (pw.get("day30_price") and pre) else None

    # Volume ratio: day0 volume / 30d avg pre-event volume
    avg_vol = pw.get("preevent_avg_volume_30d")
    post_vol = pw.get("postevent_volume_1d")
    volume_ratio = (post_vol / avg_vol) if (avg_vol and avg_vol > 0 and post_vol) else None

    outcome, confidence, basis = _infer_outcome(
        cat_type, move_1d, move_7d, move_30d,
        intraday_max_pct=pw.get("postevent_max_intraday_move_pct"),
        volume_ratio=volume_ratio,
    )
    
    # LLM fallback for ambiguous price-action cases.
    # Triggers when:
    # - heuristic confidence < 0.55 (mid-confidence and below)
    # - AND outcome is 'delayed' or 'mixed' (the categories that price action
    #   can't reliably distinguish from priced-in approvals/rejections)
    # - AND env var allows it (default on; opt-out via LLM_OUTCOME_CLASSIFIER=0)
    outcome_source = "price_action"
    use_llm = os.getenv("LLM_OUTCOME_CLASSIFIER", "1") != "0"
    if use_llm and confidence < 0.55 and outcome in ("delayed", "mixed", "unknown"):
        llm_result = _classify_outcome_with_llm(
            ticker=ticker,
            catalyst_type=cat_type,
            catalyst_date=cat_date,
            drug_name=catalyst.get("drug_name"),
            indication=catalyst.get("indication"),
            move_1d=move_1d,
            move_30d=move_30d,
            volume_ratio=volume_ratio,
        )
        if llm_result is not None:
            llm_outcome, llm_confidence, llm_basis = llm_result
            # Only override if LLM is more confident AND has a real answer
            if llm_outcome != "unknown" and llm_confidence >= confidence:
                outcome, confidence, basis = llm_outcome, llm_confidence, llm_basis
                outcome_source = "llm"
                logger.info(f"[post-catalyst] LLM override for {ticker}@{cat_date}: "
                            f"price-action='{outcome}' → LLM='{llm_outcome}' conf={llm_confidence:.2f}")

    # Predicted: confidence_score from catalyst_universe (probability), plus reference move
    predicted_prob = catalyst.get("confidence_score")

    # Reference move table — same as frontend uses
    REF_MOVES = {
        "FDA Decision": (25, -20), "PDUFA Decision": (25, -20),
        "AdComm": (15, -12), "Advisory Committee": (15, -12),
        "Phase 3 Readout": (35, -30), "Phase 3": (35, -30),
        "Phase 2 Readout": (25, -20), "Phase 2": (25, -20),
        "Phase 1 Readout": (15, -12), "Phase 1": (15, -12),
        "Clinical Trial": (12, -10), "Clinical Trial Readout": (20, -18),
        "NDA submission": (8, -5), "BLA submission": (8, -5),
        "Regulatory Decision": (22, -18),
        "Partnership": (10, -3), "Earnings": (6, -5),
        "Product Launch": (10, -8), "Commercial Launch": (10, -8),
    }
    up, down = REF_MOVES.get(cat_type or "", (10, -8))
    p = float(predicted_prob) if predicted_prob is not None else 0.5
    predicted_move = p * up + (1 - p) * down

    # Error metrics (vs 30d move, falling back to 1d)
    actual_for_error = move_30d if move_30d is not None else move_1d
    error_abs = abs(predicted_move - actual_for_error) if actual_for_error is not None else None
    error_signed = (predicted_move - actual_for_error) if actual_for_error is not None else None
    direction_correct = None
    if actual_for_error is not None:
        # Sign of predicted vs actual
        direction_correct = (predicted_move > 0 and actual_for_error > 0) or \
                            (predicted_move < 0 and actual_for_error < 0) or \
                            (abs(predicted_move) < 1 and abs(actual_for_error) < 5)

    # Insert
    try:
        with _db().get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO post_catalyst_outcomes
                  (catalyst_id, ticker, catalyst_type, catalyst_date,
                   drug_name, indication,
                   pre_event_date, pre_event_price,
                   day0_price, day1_price, day7_price, day30_price,
                   actual_move_pct_1d, actual_move_pct_7d, actual_move_pct_30d,
                   preevent_avg_volume_30d, postevent_volume_1d, postevent_max_intraday_move_pct,
                   predicted_prob, predicted_move_pct, prediction_source, predicted_at,
                   outcome, outcome_confidence, outcome_source, outcome_notes,
                   error_abs_pct, error_signed_pct, direction_correct,
                   computed_at, last_updated, backfill_attempts)
                VALUES
                  (%s, %s, %s, %s,
                   %s, %s,
                   %s, %s,
                   %s, %s, %s, %s,
                   %s, %s, %s,
                   %s, %s, %s,
                   %s, %s, %s, NOW(),
                   %s, %s, %s, %s,
                   %s, %s, %s,
                   NOW(), NOW(), 1)
                ON CONFLICT (ticker, catalyst_type, catalyst_date) DO UPDATE SET
                   pre_event_date = EXCLUDED.pre_event_date,
                   pre_event_price = EXCLUDED.pre_event_price,
                   day0_price = EXCLUDED.day0_price,
                   day1_price = EXCLUDED.day1_price,
                   day7_price = EXCLUDED.day7_price,
                   day30_price = EXCLUDED.day30_price,
                   actual_move_pct_1d = EXCLUDED.actual_move_pct_1d,
                   actual_move_pct_7d = EXCLUDED.actual_move_pct_7d,
                   actual_move_pct_30d = EXCLUDED.actual_move_pct_30d,
                   outcome = EXCLUDED.outcome,
                   outcome_confidence = EXCLUDED.outcome_confidence,
                   outcome_source = EXCLUDED.outcome_source,
                   outcome_notes = EXCLUDED.outcome_notes,
                   error_abs_pct = EXCLUDED.error_abs_pct,
                   error_signed_pct = EXCLUDED.error_signed_pct,
                   direction_correct = EXCLUDED.direction_correct,
                   last_updated = NOW(),
                   backfill_attempts = post_catalyst_outcomes.backfill_attempts + 1
            """, (
                cat_id, ticker, cat_type, cat_date,
                catalyst.get("drug_name"), catalyst.get("indication"),
                pw.get("pre_event_date"), pre,
                pw.get("day0_price"), pw.get("day1_price"), pw.get("day7_price"), pw.get("day30_price"),
                move_1d, move_7d, move_30d,
                pw.get("preevent_avg_volume_30d"), pw.get("postevent_volume_1d"), pw.get("postevent_max_intraday_move_pct"),
                predicted_prob, predicted_move, "reference_move",
                outcome, confidence, outcome_source, basis,
                error_abs, error_signed, direction_correct,
            ))
            conn.commit()
        
        # Best-effort: capture options-implied move and shares outstanding for
        # this catalyst window. These are written via a separate UPDATE so the
        # main INSERT path stays simple. Failures here are non-fatal.
        try:
            from services.options_implied import get_implied_move_for_catalyst
            implied = get_implied_move_for_catalyst(ticker, cat_date)
            implied_pct = implied.get("implied_move_pct") if implied else None
            implied_src = implied.get("source") if implied else None
        except Exception as e:
            logger.info(f"options_implied capture failed for {ticker}@{cat_date}: {e}")
            implied_pct, implied_src = None, None
        
        # Shares outstanding at event — historical lookup via yfinance
        shares_at_event = None
        try:
            import yfinance as yf
            tkr = yf.Ticker(ticker)
            info = tkr.info or {}
            shares_at_event = info.get("sharesOutstanding")
            if shares_at_event:
                shares_at_event = float(shares_at_event)
        except Exception:
            shares_at_event = None
        
        if implied_pct is not None or shares_at_event is not None:
            try:
                with _db().get_conn() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE post_catalyst_outcomes
                        SET options_implied_move_pct = COALESCE(%s, options_implied_move_pct),
                            options_implied_move_source = COALESCE(%s, options_implied_move_source),
                            shares_outstanding_at_event = COALESCE(%s, shares_outstanding_at_event)
                        WHERE ticker = %s AND catalyst_type = %s AND catalyst_date = %s
                    """, (implied_pct, implied_src, shares_at_event, ticker, cat_type, cat_date))
                    conn.commit()
            except Exception as e:
                logger.info(f"options_implied UPDATE failed for {ticker}@{cat_date}: {e}")
        
        return {
            "status": "created", "ticker": ticker, "date": cat_date,
            "outcome": outcome, "actual_1d_pct": round(move_1d, 2) if move_1d is not None else None,
            "predicted_pct": round(predicted_move, 2),
            "error_abs_pct": round(error_abs, 2) if error_abs is not None else None,
            "options_implied_pct": round(implied_pct, 2) if implied_pct else None,
        }
    except Exception as e:
        logger.exception(f"backfill insert failed for {ticker}@{cat_date}: {e}")
        return {"status": "failed", "reason": str(e)[:200], "ticker": ticker, "date": cat_date}


def _write_stub(catalyst: Dict, error: str) -> bool:
    """Write a placeholder row when price data fetch fails — prevents repeated retries
    on non-tradeable / delisted tickers."""
    try:
        with _db().get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO post_catalyst_outcomes
                  (catalyst_id, ticker, catalyst_type, catalyst_date, drug_name, indication,
                   outcome, outcome_confidence, outcome_source, outcome_notes,
                   last_error, computed_at, last_updated, backfill_attempts)
                VALUES (%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,
                        %s, NOW(), NOW(), 1)
                ON CONFLICT (ticker, catalyst_type, catalyst_date) DO UPDATE SET
                   last_error = EXCLUDED.last_error,
                   last_updated = NOW(),
                   backfill_attempts = post_catalyst_outcomes.backfill_attempts + 1
            """, (
                catalyst.get("id"), catalyst.get("ticker"), catalyst.get("catalyst_type"),
                catalyst.get("catalyst_date"), catalyst.get("drug_name"), catalyst.get("indication"),
                "unknown", 0.0, "price_action", "no price data available", error,
            ))
            conn.commit()
        return True
    except Exception as e:
        logger.warning(f"_write_stub failed: {e}")
        return False


def backfill_batch(limit: int = 25, min_age_days: int = 7) -> Dict:
    """Find due catalysts and backfill up to `limit` of them.
    Returns summary dict {processed, created, failed, skipped, results: [...]}.
    """
    due = find_due_catalysts(limit=limit, min_age_days=min_age_days)
    results = []
    counts = {"created": 0, "failed": 0, "skipped": 0}
    for cat in due:
        r = backfill_one(cat)
        results.append(r)
        counts[r.get("status", "failed")] = counts.get(r.get("status", "failed"), 0) + 1
    return {
        "processed": len(due),
        "due_total_found": len(due),
        "created": counts.get("created", 0),
        "failed": counts.get("failed", 0),
        "skipped": counts.get("skipped", 0),
        "results": results[:50],  # cap result list for response size
    }
