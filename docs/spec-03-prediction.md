# Spec 03 — Prediction wiring

**Session:** S3 (spec only — no code, no SQL execution)
**Date:** 2026-05-03 (UTC)
**Replaces:** the formula at
[services/post_catalyst_tracker.py:953-955](../services/post_catalyst_tracker.py#L953)
**Implements:** the recommendations from
[audit-02b-data-sources.md](audit-02b-data-sources.md)
**Implementation target:** S5 (next code-writing session). This spec
must be complete enough that S5 needs zero further design input.

## Executive summary (5 bullets — what changes vs today)

1. **`(up, down)` becomes a per-event lookup** keyed on
   `(catalyst_type, indication, market_cap_bucket, outcome_class)` from
   a populated `historical_catalyst_moves` table — replacing the 18-row
   hardcoded REF_MOVES dict that's currently cap-blind and indication-blind.
   Data shows tier-2 (`type × cap × outcome`) is the working level
   (98.9 % of labeled events have a usable cell at n ≥ 30); tier-1
   indication refinement is a bonus when it has n ≥ 10.

2. **`p` becomes a per-regime decision tree.** The catalyst-type
   regime — MANDATED / SEMI_MANDATED / VOLUNTARY — determines the
   source of `p`. **MANDATED** uses historical base rate (the data is
   un-survivored), **SEMI_MANDATED** blends base rate with options-
   implied probability, **VOLUNTARY** requires options data or
   abstains. This is the only fix that addresses the "model can't
   predict negative outcomes" anti-alpha pattern from audit-01.

3. **Abstention is a first-class state.** When inputs aren't sufficient
   the system publishes `predicted_move_pct = NULL` plus a reason
   code, instead of synthesising a number. Today every event gets
   `+3 %` regardless; that hides risk from operators who treat any
   number as a real prediction.

4. **The output gains `predicted_low` and `predicted_high` columns.**
   The new lookup returns p25/p75 of the conditional distributions, so
   the prediction can publish a calibrated range, not just a point
   estimate. This is a schema add (one migration), not a breaking change.

5. **The migration ships in 4 phases** (populate → shadow → A/B →
   switch) with a documented rollback at each phase. The current
   REF_MOVES code path stays intact through Phase 3 so we can revert
   without code change if calibration regresses.

## Magnitude source: populate `historical_catalyst_moves`

### Target horizon: `actual_move_pct_30d`

Confirmed from data: 72 % of labeled events have `actual_move_pct_30d`
populated vs. 7 % for `actual_move_pct_3d`. Standard deviations: 1d
30.6 %, 30d 39.6 %, abnormal_30d 38.5 %. **Use `actual_move_pct_30d`
(raw, not abnormal)** for the lookup table because:

- It's the same horizon REF_MOVES claims to model (per the inline
  comment at [post_catalyst_tracker.py:35-41](../services/post_catalyst_tracker.py#L35)).
- 72 % coverage on the labeled set vs 71 % for abnormal_30d — basically
  identical — but raw is what users see day-to-day.
- Operators screening positions react to raw move %, not sector-adjusted.
- The abnormal-vs-XBI version can be computed in parallel and persisted
  as a sibling table later.

### Population SQL (concrete — pasteable into psql)

```sql
-- One-shot population of historical_catalyst_moves from labeled outcomes.
-- Idempotent: TRUNCATE + INSERT pattern. Run nightly via cron once
-- the spec is implemented.

BEGIN;

TRUNCATE TABLE historical_catalyst_moves;

INSERT INTO historical_catalyst_moves (
    catalyst_type, indication, market_cap_bucket,
    mean_move_pct, p25_move_pct, p75_move_pct, std_dev_pct,
    n_observations, source, last_updated
)
WITH labeled AS (
  SELECT
    pco.catalyst_type,
    -- Indication is normalized to lowercase + trimmed to collapse
    -- the duplicate-case rows audit-02b flagged ("DMD" vs
    -- "Duchenne muscular dystrophy" still need separate canonicalisation).
    LOWER(TRIM(pco.indication)) AS indication,
    CASE
      WHEN s.market_cap IS NULL OR s.market_cap = 0 THEN 'unknown'
      WHEN s.market_cap < 500000 THEN 'micro_lt500m'
      WHEN s.market_cap < 2000000 THEN 'small_500m_2b'
      ELSE 'mid_or_above'
    END AS market_cap_bucket,
    CASE
      WHEN pco.outcome_label_class IN ('APPROVED','MET_ENDPOINT') THEN 'positive'
      WHEN pco.outcome_label_class IN ('REJECTED','MISSED_ENDPOINT') THEN 'negative'
    END AS outcome_class,
    pco.actual_move_pct_30d
  FROM post_catalyst_outcomes pco
  LEFT JOIN screener_stocks s ON s.ticker = pco.ticker
  WHERE pco.outcome_label_class IN
        ('APPROVED','REJECTED','MET_ENDPOINT','MISSED_ENDPOINT')
    AND pco.actual_move_pct_30d IS NOT NULL
    AND pco.outcome_label_confidence >= 0.7  -- exclude noisy LLM labels
)
-- Three rows per (type × cap × outcome) cell: one tier-1 with indication,
-- one tier-2 without, one tier-3 type×outcome only. The lookup function
-- queries by source dimension explicitly so all three coexist cleanly.
SELECT
  catalyst_type,
  indication,
  market_cap_bucket,
  ROUND(AVG(actual_move_pct_30d)::numeric, 2),
  ROUND(percentile_cont(0.25) WITHIN GROUP (ORDER BY actual_move_pct_30d)::numeric, 2),
  ROUND(percentile_cont(0.75) WITHIN GROUP (ORDER BY actual_move_pct_30d)::numeric, 2),
  ROUND(STDDEV(actual_move_pct_30d)::numeric, 2),
  COUNT(*),
  'tier1_type_indication_cap_outcome' || ':' || outcome_class,
  NOW()
FROM labeled
WHERE indication IS NOT NULL AND outcome_class IS NOT NULL
GROUP BY catalyst_type, indication, market_cap_bucket, outcome_class
HAVING COUNT(*) >= 10

UNION ALL

SELECT
  catalyst_type, NULL, market_cap_bucket,
  ROUND(AVG(actual_move_pct_30d)::numeric, 2),
  ROUND(percentile_cont(0.25) WITHIN GROUP (ORDER BY actual_move_pct_30d)::numeric, 2),
  ROUND(percentile_cont(0.75) WITHIN GROUP (ORDER BY actual_move_pct_30d)::numeric, 2),
  ROUND(STDDEV(actual_move_pct_30d)::numeric, 2),
  COUNT(*),
  'tier2_type_cap_outcome:' || outcome_class,
  NOW()
FROM labeled
WHERE outcome_class IS NOT NULL
GROUP BY catalyst_type, market_cap_bucket, outcome_class
HAVING COUNT(*) >= 30

UNION ALL

SELECT
  catalyst_type, NULL, NULL,
  ROUND(AVG(actual_move_pct_30d)::numeric, 2),
  ROUND(percentile_cont(0.25) WITHIN GROUP (ORDER BY actual_move_pct_30d)::numeric, 2),
  ROUND(percentile_cont(0.75) WITHIN GROUP (ORDER BY actual_move_pct_30d)::numeric, 2),
  ROUND(STDDEV(actual_move_pct_30d)::numeric, 2),
  COUNT(*),
  'tier3_type_outcome:' || outcome_class,
  NOW()
FROM labeled
WHERE outcome_class IS NOT NULL
GROUP BY catalyst_type, outcome_class
HAVING COUNT(*) >= 20;

COMMIT;
```

**Schema changes required:** the `source` column already exists. Add
an index for the lookup query if not already present:

```sql
CREATE INDEX IF NOT EXISTS idx_hcm_lookup
  ON historical_catalyst_moves (catalyst_type, market_cap_bucket, source);
```

### Sample-size minimums + observed coverage

Validation query results (run 2026-05-03 against production DB):

| tier | grouping | total cells | n ≥ 30 | 10 ≤ n < 30 | n < 10 |
|---|---|---:|---:|---:|---:|
| 1 | type × indication × cap × outcome | 2,381 | 3 | 13 | 2,365 |
| 2 | type × cap × outcome | 24 | 14 | 1 | 9 |
| 3 | type × outcome | 21 | 12 | 1 | 8 |

**Coverage of labeled events by which tier they land on:**

| | events | % |
|---|---:|---:|
| Tier 1 (indication-segmented, n ≥ 30) | 0 | 0.0 |
| Tier 2 (cap-segmented, n ≥ 30) | 4,133 | 98.9 |
| Tier 3 (type only, n ≥ 20) | 5 | 0.1 |
| Abstain | 41 | 1.0 |

**Reading:** tier 1 with `n ≥ 30` is aspirational — needs more data.
Spec uses `n ≥ 10` for tier 1 (relaxed from the prompt's `n ≥ 30`)
because that's where the indication signal actually exists in our
data. Working tier today is tier 2.

**Tier 1 cells with n ≥ 10 today** (representative sample): Phase 3
Readout × COVID-19 × <500M positive (n=20), Phase 3 Readout × IgAN ×
<500M positive (n=13), FDA Decision × cystic fibrosis × <500M
positive (n=12). All currently positive — there is essentially no
n ≥ 10 cell on the negative side at indication granularity, so on
negative outcomes the lookup will fall through to tier 2.

### Time-decay / recency

**Decision: all-time (no decay) for v1.**

Justification: the labeled set is small enough that any decay window
would slice cells below the n=10/30 thresholds. Calibration drift
should be monitored (see release gate below); revisit when labeled
set crosses ~10,000 clean rows. Document this explicitly in code as
a TODO.

### Lookup function spec

```python
# services/move_lookup.py — new file (does not exist yet)

from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class MoveDistribution:
    median: float          # mean_move_pct from the cell
    p25: float
    p75: float
    std: float
    n: int
    fallback_level: Literal[
        "tier1_type_indication_cap_outcome",
        "tier2_type_cap_outcome",
        "tier3_type_outcome",
        "default",  # global default (4, -4) only as last resort
    ]
    source: Literal["historical", "default"]

def lookup_move_distribution(
    catalyst_type: str,
    indication: Optional[str],
    market_cap_bucket: str,            # 'micro_lt500m' | 'small_500m_2b' | 'mid_or_above' | 'unknown'
    outcome_class: Literal["positive", "negative"],
) -> Optional[MoveDistribution]:
    """Look up the move distribution for a (type, indication, cap,
    outcome) combination. Falls through tiers in order.

    Returns None if every fallback exhausts. Caller should call this
    twice — once with outcome_class='positive', once with 'negative' —
    to compute the expected-value formula.

    Fallback chain:
      tier1: (catalyst_type, indication, market_cap_bucket, outcome_class) — needs n>=10
      tier2: (catalyst_type, market_cap_bucket, outcome_class) — needs n>=30
      tier3: (catalyst_type, outcome_class) — needs n>=20
      None  (caller handles abstention)
    """
```

The function reads from `historical_catalyst_moves`. The `source`
column on each row identifies its tier so the function can SELECT
in fallback order without joining ambiguity.

### Refresh strategy

**Decision: nightly cron at 04:00 UTC (after the labeler's daily
window closes).**

Implementation: a new endpoint
`POST /admin/post-catalyst/refresh-historical-moves` that runs the
population SQL above. Triggered by an APScheduler job in
`main.py` (the same scheduler that already runs the V2
reclassifier and outcome labeler).

Rejected alternatives:
- **Recompute on every prediction call** — too slow; would add a 200
  ms aggregation to every catalyst page load.
- **Trigger on outcome_label change** — adds DB triggers, complicates
  testing, marginal benefit since labels arrive in batches anyway.

## Probability source: per-regime decision tree

### Regime classification

Built from the negative-outcome publish rate per catalyst_type
(verified 2026-05-03 against production DB):

| catalyst_type | n_clean | %neg | regime | rationale |
|---|---:|---:|---|---|
| FDA Decision | 525 | 19.8 % | **MANDATED** | PDUFA decisions are publicly reported by regulation; 19.8 % rejection rate matches FDA historical 80 % approval rate. |
| AdComm | 10 | 80.0 % | **MANDATED** | Advisory committee votes are publicly recorded. Small n but structurally MANDATED. |
| Submission | 288 | 22.2 % | **MANDATED** | NDA/BLA submission outcomes (CRL vs accept) are mandatory disclosures. |
| Phase 3 Readout | 508 | 18.7 % | **SEMI_MANDATED** | Negative Phase 3 readouts often delayed/buried but eventually surface (analyst pressure, SEC requirements). |
| Other | 396 | 10.4 % | **SEMI_MANDATED** | Mixed bag of catalyst announcements; neither fully mandated nor fully voluntary. |
| Phase 2 Readout | 717 | 8.4 % | **VOLUNTARY** | Negative Phase 2 readouts often dropped from press release / quietly discontinued. |
| Phase 1 Readout | 662 | 1.5 % | **VOLUNTARY** | Heavy survivorship — only successful Phase 1 readouts get press releases; failed ones quietly continue. |
| Trial Initiation | 40 | 5.0 % | **VOLUNTARY** | "Negative" outcome of a trial initiation is non-initiation, which is rarely announced. |
| BLA submission, NDA submission, Phase 4 Readout, Phase 0 Readout, Phase 1/2 Readout, Phase 1/2/3 Readout, New Product Launch, Clinical Trial, Partnership, _no_history_known | 0–1 each | n/a | **UNCLASSIFIED** | Insufficient labeled data; treat as VOLUNTARY (require options data) until backfill grows. |

Spec must handle all 18 catalyst_types (including UNCLASSIFIED
fallback). A new constant in `services/catalyst_signal.py` (or new
`services/disclosure_regime.py`):

```python
DISCLOSURE_REGIME = {
    "FDA Decision":        "MANDATED",
    "AdComm":              "MANDATED",
    "Advisory Committee":  "MANDATED",
    "Submission":          "MANDATED",
    "PDUFA Decision":      "MANDATED",
    "Regulatory Decision": "MANDATED",
    "Phase 3 Readout":     "SEMI_MANDATED",
    "Phase 3":             "SEMI_MANDATED",
    "Other":               "SEMI_MANDATED",
    "Phase 2 Readout":     "VOLUNTARY",
    "Phase 2":             "VOLUNTARY",
    "Phase 1 Readout":     "VOLUNTARY",
    "Phase 1":             "VOLUNTARY",
    "Phase 1/2 Readout":   "VOLUNTARY",
    "Trial Initiation":    "VOLUNTARY",
    # all other types default to VOLUNTARY
}
```

### `p` source by regime

#### MANDATED (FDA Decision, AdComm, Submission)

- **Primary source:** historical base rate from labeled data,
  segmented by `catalyst_type × indication`.
- **Sample minimum:** n ≥ 50 in the bucket. Below that, fall back to
  `catalyst_type` alone.
- **Below n = 20 even at type level:** abstain.
- **Justification:** publish-rate test confirms the labeled set
  reflects the true outcome distribution for these regimes (the
  outcome MUST be published, so absence-of-positive really means a
  negative happened). FDA Decision at 19.8 % negative matches
  industry-wide PDUFA rejection rates of 15-22 % — sanity-checks the
  base rate is real, not an artifact.
- **Fallback chain:**
  1. `(catalyst_type, indication)` if n ≥ 50 → use as `p`
  2. `(catalyst_type)` if n ≥ 20 → use as `p`
  3. else → `p = None` (abstain)

#### SEMI_MANDATED (Phase 3 Readout, Other)

- **Primary source:** weighted blend of historical base rate and
  options-implied probability when both are available.
- **Weighting:** options-implied gets weight =
  `min(0.5, options_coverage_quality)` where `options_coverage_quality`
  is `1.0` if days_to_expiry ≤ 14 AND straddle bid-ask spread / mid <
  20 %, else degraded linearly. Default to 0.3 if quality not
  computed.
- **Justification:** Phase 3 readouts have selection bias toward
  positives (negatives often delayed/buried) — historical base rate
  alone is biased upward. Options markets tend to price in the actual
  binary risk of negative outcomes via straddle premium. Blend
  hedges against survivorship bias in our labels.
- **If options data missing:** historical base rate alone, but
  publish `p_confidence = "low"` so the prediction range widens.
- **Fallback chain:**
  1. `(catalyst_type, indication)` n ≥ 50 + options data → blend
  2. `(catalyst_type)` n ≥ 20 + options data → blend
  3. `(catalyst_type)` n ≥ 20 alone → use base rate, flag low confidence
  4. else → abstain

#### VOLUNTARY (Phase 1/2 Readout, Trial Initiation, UNCLASSIFIED)

- **Primary source:** options-implied probability if available.
- **Fallback:** **abstain.** Do **not** use historical base rate
  (Phase 1's 93.7 % "positive" rate from our data is essentially a
  press-release survivorship artifact — using it would predict
  +9 % move on every Phase 1 catalyst, replicating today's
  broken behavior).
- **Justification:** the audit-02b survivorship trap. Without options
  data we have no unbiased signal for these regimes. Abstaining is
  honest; pretending we know is anti-alpha.
- **Fallback chain:**
  1. options data + days_to_expiry ≤ 30 → use options-implied probability
  2. else → abstain

### Options-implied probability derivation

Spec for converting `services/options_implied.py:get_implied_move`
output into a directional probability.

The straddle gives the market's expected absolute move. To extract a
directional probability, we use the **moneyness-skew approach** — but
since the existing function only returns the ATM straddle (not the
full chain), the simple v1 formula is:

```
implied_probability_positive = 0.5 + skew_adjustment
```

where `skew_adjustment` defaults to 0 (50/50). To get a real skew
signal we'd need the full options chain (call IV vs put IV at the
same delta). **This is out of scope for v1.**

**v1 simplified spec:** options-implied probability is **only used as
a confidence-weighting blend factor for SEMI_MANDATED**, not as the
sole `p` source. The actual probability number remains the historical
base rate; options data tells us how much to trust it.

For VOLUNTARY: even the simplified version means VOLUNTARY catalysts
without indication-segmented base rate will mostly abstain, which is
the correct behavior. **A future spec (s4 or later) should add full
options-chain skew extraction.**

### Calibration check

A new admin endpoint **`GET /admin/llm/prediction-calibration`** that
returns per-bucket calibration:

```json
{
  "bucket": "FDA Decision × cystic fibrosis × micro_lt500m",
  "n": 12,
  "p_predicted_avg": 0.83,
  "p_actual_positive_rate": 0.75,
  "brier_score": 0.218,
  "reliability_bins": [
    {"p_bin": "0.0-0.1", "n": 0, "expected": 0.05, "actual": null},
    {"p_bin": "0.1-0.2", "n": 0, "expected": 0.15, "actual": null},
    ...
    {"p_bin": "0.8-0.9", "n": 9, "expected": 0.85, "actual": 0.78},
    {"p_bin": "0.9-1.0", "n": 3, "expected": 0.95, "actual": 0.67}
  ],
  "release_gate_passed": false,
  "release_gate_reason": "n=12 below threshold of 50"
}
```

**Release gate:** a bucket's `p` source is OK to use in production if
ALL of:

- **Sample size:** n ≥ 50 for the source's segmentation
- **Brier score:** < 0.25 (a flat 0.5-everywhere predictor is 0.25;
  better than that means the source carries directional information)
- **Reliability:** monotonic across populated bins (calibration curve
  doesn't invert — if prediction is 0.9 the actual rate must be ≥ the
  rate at prediction 0.7)

If a bucket fails any gate, the system **abstains** for that bucket.
The endpoint surfaces every bucket; an admin dashboard would show
which buckets are gated.

## Abstention rules

The system **abstains** (publishes `predicted_move_pct = NULL`) if
**any** of:

| condition | example |
|---|---|
| `p` source unavailable for the catalyst's regime | VOLUNTARY Phase 1 with no options data |
| Magnitude lookup returned `None` even at coarsest fallback (tier 3) | Phase 0 Readout (n < 20 in tier 3) |
| Calibration release gate failed for this bucket | New bucket with n < 50 |
| Required input field is null | `catalyst_type IS NULL` (rare; default to abstain to be safe) |
| `p` blend has zero non-null components | SEMI_MANDATED bucket with no historical data AND no options |

### User-facing display when abstaining

| surface | abstaining behavior |
|---|---|
| Catalyst row in screener / table | Catalyst still listed. `predicted_move_pct` cell shows `—`. Cell tooltip: `"Insufficient data for calibrated prediction (reason: <reason_code>)"`. |
| Stock detail page predicted-move card | Banner: `"Probability model abstains for this catalyst type / cap bucket: <reason_code>. We will publish a prediction once N additional labeled events accumulate (currently n=X, need ≥Y)."` |
| Ask AI chat | If user asks "what's your predicted move?" for an abstained catalyst, refuse with the abstention reason. **Required by tier 2 of the original anti-hallucination plan** (this rule already exists in `routes/chat.py` system prompt; spec just adds the abstention reason as a context field). |
| Backtest scoreboard | Abstained events excluded from accuracy %, but reported in coverage % (denominator). |

The full set of `reason_code` values (S5 should add to a constants file):

```
NO_HISTORICAL_DATA          tier-3 fallback also failed
NO_OPTIONS_DATA             VOLUNTARY regime, options not available
NO_INDICATION_MATCH         tier-1 cell missing AND tier-2 also missing
LOW_CONFIDENCE_GATE_FAILED  release gate failed (Brier or reliability)
INSUFFICIENT_SAMPLE         bucket n below threshold
NULL_REQUIRED_FIELD         catalyst_type or other required field null
```

## New prediction expression (pseudocode)

This replaces [post_catalyst_tracker.py:953-955](../services/post_catalyst_tracker.py#L953).

```python
# pseudocode — S5 writes the real Python

def compute_prediction(catalyst_row) -> Prediction:
    """
    catalyst_row has: catalyst_type, indication, ticker, catalyst_date,
                      and (joined) market_cap from screener_stocks.
    """

    # 1. Required-field check
    if not catalyst_row.catalyst_type:
        return Prediction.abstain(reason="NULL_REQUIRED_FIELD")

    # 2. Resolve regime
    regime = DISCLOSURE_REGIME.get(catalyst_row.catalyst_type, "VOLUNTARY")

    # 3. Resolve cap bucket
    cap_bucket = bucketize_market_cap(catalyst_row.market_cap)
    # → 'micro_lt500m' | 'small_500m_2b' | 'mid_or_above' | 'unknown'

    # 4. Get probability + source metadata
    p, p_source, p_confidence = lookup_probability(
        regime=regime,
        catalyst_type=catalyst_row.catalyst_type,
        indication=catalyst_row.indication,
        ticker=catalyst_row.ticker,
        catalyst_date=catalyst_row.catalyst_date,
    )
    if p is None:
        return Prediction.abstain(reason=p_source)
        # p_source carries the abstention reason on failure

    # 5. Get conditional move distributions
    pos_dist = lookup_move_distribution(
        catalyst_row.catalyst_type, catalyst_row.indication,
        cap_bucket, outcome_class="positive",
    )
    neg_dist = lookup_move_distribution(
        catalyst_row.catalyst_type, catalyst_row.indication,
        cap_bucket, outcome_class="negative",
    )
    if pos_dist is None or neg_dist is None:
        return Prediction.abstain(reason="NO_HISTORICAL_DATA")

    # 6. Calibration gate (cached per-bucket result, refreshed nightly)
    if not bucket_calibration_passes(
        catalyst_row.catalyst_type, catalyst_row.indication, cap_bucket
    ):
        return Prediction.abstain(reason="LOW_CONFIDENCE_GATE_FAILED")

    # 7. Compute expected move + range using p × E[move|positive]
    predicted_move = p * pos_dist.median + (1 - p) * neg_dist.median

    # The p25/p75 span uses the same blend so the range is internally
    # consistent — NOT a min/max of the conditional distributions
    # (which would be too wide).
    predicted_low  = p * pos_dist.p25 + (1 - p) * neg_dist.p25
    predicted_high = p * pos_dist.p75 + (1 - p) * neg_dist.p75

    return Prediction(
        predicted_move_pct=predicted_move,
        predicted_low_pct=predicted_low,
        predicted_high_pct=predicted_high,
        p=p,
        p_source=p_source,
        p_confidence=p_confidence,
        magnitude_n=min(pos_dist.n, neg_dist.n),
        magnitude_fallback_level=pos_dist.fallback_level,
        regime=regime,
        cap_bucket=cap_bucket,
        prediction_source="historical_v1",  # new constant; old path = "reference_move"
        abstained=False,
    )
```

### Field mapping to existing DB schema

The `post_catalyst_outcomes` table has 68 columns. The new prediction
adds:

| new field | DB column | type | migration |
|---|---|---|---|
| `predicted_move_pct` | existing | numeric | reuse |
| `predicted_low_pct` | **NEW** | numeric | add column |
| `predicted_high_pct` | **NEW** | numeric | add column |
| `predicted_p` | reuse `predicted_prob` | numeric | reuse (rename in code, leave column) |
| `predicted_p_source` | **NEW** | text | add column |
| `predicted_p_confidence` | **NEW** | text | add column ("high"/"medium"/"low") |
| `magnitude_n` | **NEW** | integer | add column |
| `magnitude_fallback_level` | **NEW** | text | add column |
| `regime` | **NEW** | text | add column |
| `cap_bucket_at_prediction` | **NEW** | text | add column (snapshot — market_cap can change later) |
| `prediction_source` | existing | text | reuse — v1 writes 'historical_v1', legacy = 'reference_move' |
| `abstained` | **NEW** | boolean | add column, default false |
| `abstain_reason` | **NEW** | text | add column, nullable |

**One alembic migration** (new revision `020_prediction_v2_columns`)
adds all 9 new columns. All nullable, default null/false. No backfill
needed — old rows keep their `prediction_source = 'reference_move'`
and the new columns stay null.

`predicted_npv_b` (currently 100 % null per audit-02b) is unrelated
and can be dropped in a separate cleanup migration.

## Frontend impact

| surface | file | what reads `predicted_move_pct` | spec |
|---|---|---|---|
| Stock detail page | [`app/stocks/[ticker]/page.tsx`](../../biotech-frontend/app/stocks/[ticker]/page.tsx) | predicted move card | Show range `predicted_low_pct` to `predicted_high_pct` instead of single number. Display `p_source` below the value. Banner if `abstained=true`. |
| Post-catalyst history panel | [`components/PostCatalystHistoryPanel.tsx:259`](../../biotech-frontend/components/PostCatalystHistoryPanel.tsx#L259) | per-row predicted vs actual | Display `predicted_move_pct ± (predicted_high - predicted_low)/2`. Strike-through if abstained. Hover tooltip shows source + n. |
| Options-implied panel | [`components/OptionsImpliedPanel.tsx:15`](../../biotech-frontend/components/OptionsImpliedPanel.tsx#L15) | comment references model | Update inline doc comment. No functional change. |
| Ask AI chat context bundle | [`routes/chat.py:289+`](../routes/chat.py#L289) | `post_catalyst_history` | Include `prediction_source`, `abstained`, `abstain_reason`, `p_source` in the slim context so the LLM can refuse intelligently. |
| Backtest scoreboard cards | [`routes/admin.py post-catalyst/aggregate-v3`](../routes/admin.py) | aggregate accuracy | Add `n_abstained` to the response shape. Frontend renders `coverage_pct` excluding abstentions from the numerator. |

**Frontend changes are out of scope for S5 (the implementation
session). Flag as a separate session — call it S5b or just batch with
S6.** S5 ships the API + DB changes; S5b updates the UI.

## Migration phases

### Phase 1 — Populate + shadow (target: 1 day)

**What ships:**
- Alembic migration `020_prediction_v2_columns` (the 9 new columns).
- New module `services/move_lookup.py` with `lookup_move_distribution()`.
- New module `services/disclosure_regime.py` with `DISCLOSURE_REGIME`
  and `lookup_probability()`.
- New endpoint `POST /admin/post-catalyst/refresh-historical-moves`.
- The new `compute_prediction()` function added alongside the legacy
  one, behind a feature flag `PREDICTION_V2_ENABLED=0` (off by
  default).
- Nightly cron job calls the refresh endpoint at 04:00 UTC.

**What it does NOT do:** the legacy `predicted_move_pct` formula
keeps writing to the column. The new compute_prediction is called in
*shadow* mode — its output writes only to the new columns
(`predicted_low_pct`, etc.) so we can compare side-by-side.

**Success criteria:**
- `historical_catalyst_moves` populated with ≥ 35 cells (matches
  validation table)
- For every existing labeled outcome, the new `compute_prediction`
  produces either a valid prediction or an abstention reason — no
  exceptions raised
- Per-bucket `Brier` score of new predictions is ≤ legacy on ≥ 60 %
  of buckets (validated by the calibration endpoint)

**Rollback:** drop the new endpoint + scheduler entry. Keep migration
in place (additive, no harm).

### Phase 2 — A/B compare on backfilled events (target: 1 week)

**What ships:** flip `PREDICTION_V2_ENABLED=1` for a 10 % sample of
backfilled events (deterministic via `hash(catalyst_id) % 10 == 0`).
Both legacy and v2 predictions written; only legacy is shown to
users.

**What changes:** A daily report
`GET /admin/post-catalyst/v1-vs-v2-prediction-compare` returns the
per-bucket Brier-score, direction-accuracy, and abs-error deltas.

**Success criteria:**
- v2 has higher direction accuracy on the negative-outcome subset
  (REJECTED, MISSED_ENDPOINT) than v1 by ≥ 10 pp on ≥ 3 catalyst types
- v2 abs-error is within 25 % of v1 on positive outcomes (we expect
  worse magnitude on positives because we're no longer over-predicting
  small positive moves on negative-outcome events; check this is a
  feature not a regression)
- Abstention rate < 25 % overall (anything higher means our coverage
  is too thin; consider relaxing thresholds before phase 3)

**Rollback:** flip flag back to 0. Frontend never saw v2 numbers.

### Phase 3 — Switch display to v2, keep v1 computed (target: 2 weeks)

**What ships:** UI reads from the v2 columns (`predicted_move_pct`
written by v2, plus the new range/source columns). Legacy
`predicted_move_pct` keeps being computed in parallel and written
to a backup column `predicted_move_pct_v1_archive` for rollback.

**Success criteria:**
- User-reported confusion rate (manual: tickets, chat questions
  about "why did the prediction change?") < 5/week after the first
  week
- Backtest direction accuracy on the live-screener subset (events
  whose predictions were made AFTER the v2 switch) ≥ legacy by
  ≥ 5 pp at the 95 % CI

**Rollback:** flip a config flag to read from `_v1_archive` column.
No code change required.

### Phase 4 — Deprecate legacy REF_MOVES path (target: when phase 3
green for 30 days)

**What ships:**
- Delete the v1 formula from `post_catalyst_tracker.py` lines 953-955.
- Delete the duplicated REF_MOVES from `catalyst_signal.py:68` and
  `routes/admin.py:1514` (3 copies → 0).
- Drop the backup column `predicted_move_pct_v1_archive`.

**Success criteria:**
- Phase 3 has been live for 30 days with no rollback events.
- No dependent code paths still read REF_MOVES (verified by grep
  before merge).

**Rollback:** restore from git history if regression discovered post-deploy.

## Open issues / deferred decisions

The following are explicitly **NOT** in this spec — flagged for S4 or
later. S5 should not include any of these in implementation.

1. **LLM-derived `p_positive_outcome` per event (the $4 backfill from
   audit-02b recommendation #5).** Defer. The historical-base-rate
   approach in this spec already addresses the root bug. The LLM PoS
   adds complexity (calibration loop, prompt engineering) for marginal
   lift. Revisit once we have ≥ 10 K labeled events.
2. **LGBM v3 retraining.** Defer per audit-02b. Most features are
   90 %+ null on labeled rows. Re-evaluate after the runup +
   priced_in backfill (which is a separate session).
3. **V1 vs V2 chat scoreboard cleanup.** Defer. The chat surfaces
   `aggregate-v2.tradeable_events.direction_accuracy_pct = 2.1 %`
   which is the SQL denominator bug from audit-01. Fixing the chat
   context to use `aggregate-v3` is unrelated to the prediction
   wiring rebuild.
4. **Drug-class segmentation in `historical_catalyst_moves`.**
   Defer. The `drug_class` field doesn't exist in our schema; would
   need a new column populated by an LLM batch ($4). Worth doing,
   but unlocks marginal lift on top of the (type × indication × cap)
   already specified.
5. **Full options-chain skew extraction.** Defer. The simplified
   `implied_probability_positive = 0.5` (i.e. options data only used
   as confidence-weighting) is an explicit limitation in §
   "Options-implied probability derivation". Extracting real direction
   from chain skew needs a new module + Polygon greeks data.
6. **Time-decay weighting on `historical_catalyst_moves`.** Specified
   as "all-time for now". Revisit when labeled set ≥ 10 K rows or if
   calibration drifts > 10 pp on rolling 30-day buckets.
7. **Indication canonicalisation.** Audit-02b flagged duplicates like
   "DMD" / "Duchenne muscular dystrophy" / "Duchenne Muscular
   Dystrophy". Spec uses `LOWER(TRIM(indication))` which catches case
   but not full synonyms. A separate canonicalisation table is a
   separate session.
8. **Sentiment / news features.** Audit-02b documented these as dead
   weight (5,300 lines of unused feature engineering). Out of scope;
   the historical base rate approach doesn't need them.

## Implementation checklist for S5

S5 should be able to ship this with **zero further design questions**.
Each item is concrete and verifiable.

### A. Schema (1 file, 1 migration)

- [ ] Create `alembic/versions/020_prediction_v2_columns.py` adding
      these nullable columns to `post_catalyst_outcomes`:
      `predicted_low_pct numeric, predicted_high_pct numeric,
       predicted_p_source text, predicted_p_confidence text,
       magnitude_n integer, magnitude_fallback_level text,
       regime text, cap_bucket_at_prediction text,
       abstained boolean DEFAULT false, abstain_reason text,
       predicted_move_pct_v1_archive numeric`.
- [ ] Add idempotent admin endpoint `POST /admin/post-catalyst/apply-migration-020`
      following the pattern of migration 019.
- [ ] Index: `CREATE INDEX IF NOT EXISTS idx_hcm_lookup
      ON historical_catalyst_moves (catalyst_type, market_cap_bucket, source)`.

### B. New modules

- [ ] `services/disclosure_regime.py` with `DISCLOSURE_REGIME` dict
      (verbatim from § "Regime classification" above) and
      `classify_disclosure_regime(catalyst_type: str) -> str` that
      defaults to `"VOLUNTARY"` for unknown types.
- [ ] `services/move_lookup.py` with `lookup_move_distribution()` per
      § "Lookup function spec". Reads from `historical_catalyst_moves`.
      Returns `None` on full fallback exhaustion.
- [ ] `services/probability_lookup.py` with `lookup_probability()`
      per § "p source by regime". Reads historical base rates from
      `post_catalyst_outcomes` (use the same labeled filter as the
      population SQL: `outcome_label_class IN ('APPROVED','REJECTED',
      'MET_ENDPOINT','MISSED_ENDPOINT') AND outcome_label_confidence
      >= 0.7`). For SEMI_MANDATED blend, reuse `services/options_implied.py`
      for the implied move; v1 stub `implied_probability_positive=0.5`
      until skew extraction lands.
- [ ] `services/prediction_v2.py` with `compute_prediction(catalyst_row) ->
      Prediction` (dataclass) per § "New prediction expression".
      Returns either a `Prediction(abstained=False, ...)` or
      `Prediction.abstain(reason="...")`.

### C. Population SQL endpoint

- [ ] `POST /admin/post-catalyst/refresh-historical-moves` running the
      population SQL from § "Population SQL" verbatim. Returns
      `{cells_inserted_tier1, cells_inserted_tier2, cells_inserted_tier3,
        duration_ms}`.
- [ ] Wire into APScheduler at 04:00 UTC daily (find the existing
      scheduler block in `main.py` or `services/v2_reclassify_scheduler.py`).

### D. Shadow-mode integration (Phase 1)

- [ ] In `services/post_catalyst_tracker.py` around line 949: after the
      legacy formula computes `predicted_move`, call
      `compute_prediction()` and if `os.getenv("PREDICTION_V2_ENABLED",
      "0") != "0"`, write the v2 outputs to the new columns. Legacy
      path continues writing the old `predicted_move_pct`.
- [ ] In the same file, ALSO write the legacy value to
      `predicted_move_pct_v1_archive` (so we have a clean rollback
      copy independent of when v2 turns on).

### E. Calibration endpoint

- [ ] `GET /admin/llm/prediction-calibration` per § "Calibration check".
      Computes Brier + reliability bins per bucket from already-resolved
      events. Surfaces `release_gate_passed` per bucket.
- [ ] `bucket_calibration_passes()` helper used by `compute_prediction()`
      reads the same bucket-level pass/fail from a small in-memory cache
      refreshed nightly (or on the same schedule as
      `refresh-historical-moves`).

### F. A/B comparison endpoint (Phase 2)

- [ ] `GET /admin/post-catalyst/v1-vs-v2-prediction-compare?days=N`
      returning per-catalyst-type and per-bucket: `n`, `v1_brier`,
      `v2_brier`, `v1_direction_acc`, `v2_direction_acc`, `v1_abs_err`,
      `v2_abs_err`, `n_abstained_v2`. Reads from rows where both
      `predicted_move_pct` and `predicted_move_pct_v2` (i.e. new column
      `predicted_move_pct` written under v2 mode) are populated.

### G. Tests

- [ ] Unit test for `classify_disclosure_regime` covering each known
      type + the unknown-default path.
- [ ] Unit test for `lookup_move_distribution` with a fixture
      `historical_catalyst_moves` row at each tier; assert correct
      fallback chain.
- [ ] Unit test for `compute_prediction` with mocked dependencies for:
      MANDATED happy path, SEMI_MANDATED with options blend, VOLUNTARY
      with options, VOLUNTARY without options (must abstain), each
      abstention reason.
- [ ] Integration test: end-to-end call against a test DB seeded with
      ~50 labeled rows; assert `compute_prediction` returns a non-null
      result for FDA Decision rows and abstains for Phase 1 with no
      options.

### H. Documentation + memory

- [ ] Update [audit-01-labels.md](audit-01-labels.md) and
      [audit-02-labels-update.md](audit-02-labels-update.md) with a
      "see spec-03 for the fix" note at the top.
- [ ] Add a memory entry `feedback_prediction_v2.md` recording the
      shipped design so future agents inherit the "abstain-first"
      philosophy.

### I. Observability

- [ ] On every `compute_prediction` call, emit a structured log line
      `{event:"prediction_v2", catalyst_id, regime, p, p_source,
       magnitude_fallback_level, abstained, abstain_reason}`. Wire into
      the existing logger; no new infra needed.
- [ ] Counter on the labeler dashboard: `% events abstained` per
      catalyst_type, refreshed daily. Goal: < 25 % for MANDATED,
      < 50 % for SEMI_MANDATED, < 80 % for VOLUNTARY.

S5 acceptance: every checkbox above passes, the v2 endpoints return
the documented shapes, and `compute_prediction` is called from the
tracker behind the feature flag. After S5 ships, the operator (the
human) flips `PREDICTION_V2_ENABLED=1` for shadow mode (Phase 1) and
monitors the calibration endpoint for ≥ 1 day before proceeding to
phase 2.
