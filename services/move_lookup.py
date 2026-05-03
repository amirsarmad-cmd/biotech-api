"""move_lookup — read magnitude distributions from historical_catalyst_moves.

Replaces the cap-blind 18-entry REF_MOVES dict in
services/post_catalyst_tracker.py:42 with a per-event lookup keyed on
(catalyst_type, indication, market_cap_bucket, outcome_class).

Fallback chain (per docs/spec-03-prediction.md § Lookup function spec):

    tier1: (catalyst_type, indication, market_cap_bucket, outcome_class)  n>=10
    tier2: (catalyst_type,             market_cap_bucket, outcome_class)  n>=30
    tier3: (catalyst_type,                                outcome_class)  n>=20
    None  (caller handles abstention)

The historical_catalyst_moves rows themselves are produced by
POST /admin/post-catalyst/refresh-historical-moves (see routes/admin.py).
The `source` column on each row tags it with the tier identifier so
this module can SELECT in fallback order with no joins.

Validation against production data 2026-05-03:
  Tier 1 (n>=10): 13 cells exist (all positive — no negative cells reach n>=10)
  Tier 2 (n>=30): 14 cells
  Tier 3 (n>=20): 12 cells
  Coverage of labeled events landing on a usable cell: 99.0%
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from services.database import BiotechDatabase

OutcomeClass = Literal["positive", "negative"]
FallbackLevel = Literal[
    "tier1_type_indication_cap_outcome",
    "tier2_type_cap_outcome",
    "tier3_type_outcome",
]


@dataclass
class MoveDistribution:
    median: float          # mean_move_pct from the cell (named median for code-clarity)
    p25: float
    p75: float
    std: Optional[float]
    n: int
    fallback_level: FallbackLevel
    source: str            # the raw `source` column value from historical_catalyst_moves


def _bucketize_market_cap(market_cap: Optional[float]) -> str:
    """Three-bucket cap-size bucket (matches the population SQL).

    The data is 97.8% <500M small-caps; >$2B has essentially no labeled
    events. Three buckets give all the resolution the data supports.
    """
    if market_cap is None or market_cap == 0:
        return "unknown"
    # market_cap is in thousands per screener_stocks convention
    if market_cap < 500_000:
        return "micro_lt500m"
    if market_cap < 2_000_000:
        return "small_500m_2b"
    return "mid_or_above"


def _norm_indication(indication: Optional[str]) -> Optional[str]:
    """Same normalization the population SQL uses (LOWER + TRIM).
    Full synonym canonicalisation is deferred per spec § Open issues.
    """
    if not indication:
        return None
    return indication.lower().strip() or None


def lookup_move_distribution(
    catalyst_type: str,
    indication: Optional[str],
    market_cap_bucket: str,
    outcome_class: OutcomeClass,
) -> Optional[MoveDistribution]:
    """Look up the move distribution for a (type, indication, cap,
    outcome) combination. Falls through tiers in order. Returns None
    if every fallback exhausts.

    Caller invokes twice — once per outcome_class — to compute the
    expected-value formula.
    """
    if not catalyst_type:
        return None
    indication_norm = _norm_indication(indication)
    db = BiotechDatabase()
    with db.get_conn() as conn:
        cur = conn.cursor()

        # Tier 1: type × indication × cap × outcome
        if indication_norm:
            cur.execute("""
                SELECT mean_move_pct, p25_move_pct, p75_move_pct,
                       std_dev_pct, n_observations, source
                FROM historical_catalyst_moves
                WHERE catalyst_type = %s
                  AND indication = %s
                  AND market_cap_bucket = %s
                  AND source = %s
                LIMIT 1
            """, (
                catalyst_type, indication_norm, market_cap_bucket,
                f"tier1_type_indication_cap_outcome:{outcome_class}",
            ))
            row = cur.fetchone()
            if row:
                return MoveDistribution(
                    median=float(row[0]),
                    p25=float(row[1]),
                    p75=float(row[2]),
                    std=float(row[3]) if row[3] is not None else None,
                    n=int(row[4]),
                    fallback_level="tier1_type_indication_cap_outcome",
                    source=row[5],
                )

        # Tier 2: type × cap × outcome
        cur.execute("""
            SELECT mean_move_pct, p25_move_pct, p75_move_pct,
                   std_dev_pct, n_observations, source
            FROM historical_catalyst_moves
            WHERE catalyst_type = %s
              AND market_cap_bucket = %s
              AND indication IS NULL
              AND source = %s
            LIMIT 1
        """, (
            catalyst_type, market_cap_bucket,
            f"tier2_type_cap_outcome:{outcome_class}",
        ))
        row = cur.fetchone()
        if row:
            return MoveDistribution(
                median=float(row[0]),
                p25=float(row[1]),
                p75=float(row[2]),
                std=float(row[3]) if row[3] is not None else None,
                n=int(row[4]),
                fallback_level="tier2_type_cap_outcome",
                source=row[5],
            )

        # Tier 3: type × outcome
        cur.execute("""
            SELECT mean_move_pct, p25_move_pct, p75_move_pct,
                   std_dev_pct, n_observations, source
            FROM historical_catalyst_moves
            WHERE catalyst_type = %s
              AND market_cap_bucket IS NULL
              AND indication IS NULL
              AND source = %s
            LIMIT 1
        """, (catalyst_type, f"tier3_type_outcome:{outcome_class}"))
        row = cur.fetchone()
        if row:
            return MoveDistribution(
                median=float(row[0]),
                p25=float(row[1]),
                p75=float(row[2]),
                std=float(row[3]) if row[3] is not None else None,
                n=int(row[4]),
                fallback_level="tier3_type_outcome",
                source=row[5],
            )

    return None
