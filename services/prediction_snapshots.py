"""Forward prediction snapshots — frozen records for OOS validation.

Per ChatGPT critique on V2 in-sample-tuning risk: the only way to claim
real performance is to evaluate predictions made BEFORE outcomes are
known. This module writes immutable snapshots whenever the V2 classifier
runs, then later updates them with actual outcomes.

The historical 459-row backtest cannot be retroactively snapshotted (the
events already happened), but every catalyst classified going forward
gets a snapshot — building an OOS dataset over time.

Three operations:
  write_snapshot()       — write a new snapshot (called during classification)
  evaluate_snapshot()    — fill in actual_dir + direction_correct (called
                           when day3 outcome data arrives)
  aggregate_oos()        — compute accuracy/CI on snapshots whose
                           prediction_time predates outcome resolution
"""
from __future__ import annotations
import json
import logging
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)


# Active model version string — bump when classifier logic / thresholds
# change so OOS aggregation can segment by version.
ACTIVE_MODEL_VERSION = "v2_thresh_0.60_0.80_sector_adj"
ACTIVE_FEATURE_VERSION = "v1_runup_vs_xbi"


def write_snapshot(
    *,
    db,  # BiotechDatabase instance
    ticker: str,
    catalyst_id: Optional[int],
    catalyst_date: Optional[str],  # ISO YYYY-MM-DD
    catalyst_type: Optional[str],
    catalyst_outcome_id: Optional[int],
    signal: str,
    predicted_prob: Optional[float],
    priced_in_score: Optional[float],
    predicted_direction: Optional[int],
    full_features: Dict[str, Any],
    signal_version: str = "v2",
    feature_version: str = ACTIVE_FEATURE_VERSION,
    model_version: str = ACTIVE_MODEL_VERSION,
) -> Optional[int]:
    """Write a frozen snapshot. Returns the snapshot id, or None on failure.

    Idempotency: this writes a NEW row each time. If the same catalyst is
    re-classified later (e.g., features changed), a new snapshot is created
    so we can compare prediction drift across versions.
    """
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO prediction_snapshots (
                    ticker, catalyst_id, catalyst_date, catalyst_type,
                    catalyst_outcome_id,
                    signal_version, feature_version, model_version,
                    signal, predicted_prob, priced_in_score, predicted_direction,
                    full_features_json
                ) VALUES (
                    %s, %s, %s, %s,
                    %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s
                )
                RETURNING id
                """,
                (
                    ticker, catalyst_id, catalyst_date, catalyst_type,
                    catalyst_outcome_id,
                    signal_version, feature_version, model_version,
                    signal, predicted_prob, priced_in_score, predicted_direction,
                    json.dumps(full_features),
                ),
            )
            new_id = cur.fetchone()[0]
            conn.commit()
        return new_id
    except Exception as e:
        logger.exception(f"write_snapshot failed for {ticker}/{catalyst_id}: {e}")
        return None


def evaluate_snapshot(
    *,
    db,
    snapshot_id: int,
    actual_abnormal_3d_pct: Optional[float],
) -> bool:
    """Fill in evaluation fields once the day3 outcome data arrives.

    Direction-correct rule:
      |abnormal_3d| < 3%   → deadband, direction_correct = NULL
      else                 → actual_dir = sign(abnormal_3d)
                             direction_correct = (predicted_dir == actual_dir)

    Returns True on successful update.
    """
    try:
        if actual_abnormal_3d_pct is None:
            return False
        ab = float(actual_abnormal_3d_pct)
        if abs(ab) < 3.0:
            actual_dir = 0
            with db.get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE prediction_snapshots
                    SET evaluated_at = now(),
                        actual_abnormal_3d_pct = %s,
                        actual_dir_3d_vs_xbi = 0,
                        direction_correct = NULL
                    WHERE id = %s
                    """,
                    (ab, snapshot_id),
                )
                conn.commit()
            return True

        actual_dir = 1 if ab > 0 else -1
        with db.get_conn() as conn:
            cur = conn.cursor()
            # direction_correct only valid when predicted_direction is non-null
            cur.execute(
                """
                UPDATE prediction_snapshots
                SET evaluated_at = now(),
                    actual_abnormal_3d_pct = %s,
                    actual_dir_3d_vs_xbi = %s,
                    direction_correct = (
                        CASE WHEN predicted_direction IS NULL THEN NULL
                             ELSE (predicted_direction = %s)
                        END
                    )
                WHERE id = %s
                """,
                (ab, actual_dir, actual_dir, snapshot_id),
            )
            conn.commit()
        return True
    except Exception as e:
        logger.exception(f"evaluate_snapshot failed for snapshot_id={snapshot_id}: {e}")
        return False


def evaluate_pending_snapshots(*, db, max_rows: int = 500) -> Dict[str, int]:
    """Find unevaluated snapshots whose linked post_catalyst_outcomes
    rows now have abnormal_move_pct_3d data, and fill them in.

    Returns: {scanned, evaluated, still_pending, errors}
    """
    counts = {"scanned": 0, "evaluated": 0, "still_pending": 0, "errors": 0}
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT s.id, pco.abnormal_move_pct_3d
                FROM prediction_snapshots s
                LEFT JOIN post_catalyst_outcomes pco
                  ON pco.id = s.catalyst_outcome_id
                WHERE s.evaluated_at IS NULL
                ORDER BY s.prediction_time
                LIMIT %s
                """,
                (max_rows,),
            )
            rows = cur.fetchall()

        for snap_id, abnormal_3d in rows:
            counts["scanned"] += 1
            if abnormal_3d is None:
                counts["still_pending"] += 1
                continue
            ok = evaluate_snapshot(
                db=db, snapshot_id=snap_id,
                actual_abnormal_3d_pct=float(abnormal_3d),
            )
            if ok:
                counts["evaluated"] += 1
            else:
                counts["errors"] += 1
    except Exception as e:
        logger.exception(f"evaluate_pending_snapshots failed: {e}")
        counts["errors"] += 1
    return counts


def aggregate_oos(*, db, signal_version: str = "v2") -> Dict[str, Any]:
    """OOS aggregate: snapshots whose prediction_time pre-dates the outcome
    being known. With a fresh table, this will start at zero and grow.

    Returns the same shape as aggregate-v3's tradeable tier so the UI can
    render it identically. Includes a 'days_of_oos_data' field telling the
    user how mature the OOS dataset is.
    """
    from datetime import datetime, timezone
    try:
        with db.get_conn() as conn:
            cur = conn.cursor()
            # Aggregate
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE evaluated_at IS NOT NULL) AS evaluated,
                    COUNT(*) FILTER (WHERE direction_correct IS NOT NULL) AS judged,
                    COUNT(*) FILTER (WHERE direction_correct = TRUE) AS hits,
                    MIN(prediction_time) AS earliest_prediction,
                    MAX(prediction_time) AS latest_prediction
                FROM prediction_snapshots
                WHERE signal_version = %s
                  AND signal IN (
                      'LONG_UNDERPRICED_POSITIVE',
                      'SHORT_SELL_THE_NEWS',
                      'SHORT_LOW_PROBABILITY',
                      'LONG', 'SHORT'
                  )
                """,
                (signal_version,),
            )
            r = cur.fetchone()
            total, evaluated, judged, hits, earliest, latest = r

            # Per-bucket breakdown
            cur.execute(
                """
                SELECT signal,
                       COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE direction_correct IS NOT NULL) AS judged,
                       COUNT(*) FILTER (WHERE direction_correct = TRUE) AS hits
                FROM prediction_snapshots
                WHERE signal_version = %s
                GROUP BY signal
                ORDER BY signal
                """,
                (signal_version,),
            )
            buckets = []
            for sig, ntot, nj, nh in cur.fetchall():
                buckets.append({
                    "signal": sig,
                    "count": ntot,
                    "judged": nj or 0,
                    "hits": nh or 0,
                    "direction_accuracy_pct": (
                        round(100.0 * nh / nj, 1) if nj and nj > 0 else None
                    ),
                })

        # Days of OOS data
        days_of_data = None
        if earliest:
            now = datetime.now(timezone.utc)
            days_of_data = (now - earliest).days

        accuracy_pct = round(100.0 * hits / judged, 1) if judged and judged > 0 else None

        return {
            "signal_version": signal_version,
            "tradeable_total": total or 0,
            "evaluated": evaluated or 0,
            "judged": judged or 0,
            "hits": hits or 0,
            "direction_accuracy_pct": accuracy_pct,
            "earliest_prediction": earliest.isoformat() if earliest else None,
            "latest_prediction": latest.isoformat() if latest else None,
            "days_of_oos_data": days_of_data,
            "buckets": buckets,
            "is_oos": True,
            "_note": (
                "OOS aggregate: snapshots written prospectively. Differs from "
                "the in-sample backtest aggregate. Will be empty/sparse until "
                "snapshots accumulate over weeks/months."
            ),
        }
    except Exception as e:
        logger.exception(f"aggregate_oos failed: {e}")
        raise
