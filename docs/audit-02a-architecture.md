# Audit 02a — Prediction pipeline architecture

**Session:** S2a (read-only architecture audit)
**Date:** 2026-05-03 (UTC)
**Scope:** End-to-end trace of how `predicted_move_pct` is produced.
Read-only. No code changes proposed.
**Prior:** [audit-01-labels.md](audit-01-labels.md), [audit-02-labels-update.md](audit-02-labels-update.md).

## Executive summary (5 bullets, prioritised)

1. **Architecture verdict: (c) — heuristic probability × reference-move
   lookup.** Per-row prediction is a one-liner:
   `predicted_move = p × up + (1 − p) × down`, where `(up, down)` come
   from a hardcoded 18-row dict (`REF_MOVES` in
   [post_catalyst_tracker.py:42](../services/post_catalyst_tracker.py#L42))
   and `p` is `catalyst_universe.confidence_score`. **There is no
   trained model in the production prediction path.**

2. **`p` is the wrong number.** `confidence_score` is set by the LLM
   normalizer ([backfill_normalizer.py:524](../services/backfill_normalizer.py#L524))
   from the LLM's `confidence` field, which the prompt explicitly
   defines as **"how confident the LLM is that the extracted text
   describes a real catalyst"** (extraction quality), not P(positive
   outcome). The formula treats it as P(positive outcome). Average
   `confidence_score` across 14,300 rows is **0.881**, so virtually
   every event gets predicted `≈ 0.88×up + 0.12×down ≈ +3 %`. **This
   is the root of the anti-alpha-on-negative-outcomes pattern from
   audit-01/02:** the model literally cannot produce a negative
   prediction unless the LLM was unsure the catalyst was real.

3. **There IS a trained LightGBM model**
   ([services/lgbm_classifier.py](../services/lgbm_classifier.py)),
   id=1, trained 2026-05-01, **`is_active=true`** in `lgbm_models`.
   Train accuracy 57.4 % on n=68, **test accuracy 45.0 % on n=20**
   (below coin-flip, walk-forward holdout). It's loaded only by the
   info endpoint
   [admin.py:6286 `/post-catalyst/v3-model-info`](../routes/admin.py#L6286).
   `predict_v3()` exists but is never called from any production code
   path. The model is dormant.

4. **REF_MOVES is cap-segment-blind, indication-blind, drug-class-blind.**
   18 entries indexed only on `catalyst_type`. Same `(4, -5)` for an
   FDA decision on AAPL-cap REGN as on $50M-cap OTLK. Audit-01's
   small-cap-clinical-wins-massively-under-predicted finding falls
   straight out of this table shape — and so does the negative-
   outcome anti-alpha (the down side is similarly miscalibrated).

5. **V1 vs V2 only changes the trade signal label, not the move
   magnitude.** Both go through the same
   `predicted_move = p × up + (1 − p) × down` formula. The V1/V2
   distinction in
   [catalyst_signal.py](../services/catalyst_signal.py) only affects
   whether a row is tagged LONG / SHORT / NO_TRADE_*, plus an added
   `priced_in_score` for sell-the-news detection. Fixing the move
   magnitude is a different code path from fixing the trade signal.

## Architecture verdict

**(c) — heuristic outcome-probability × reference move lookup.**

Justification: the actual line of code that produces every
`predicted_move_pct` value in `post_catalyst_outcomes` is:

```python
# services/post_catalyst_tracker.py:953-955
up, down = REF_MOVES.get(cat_type or "", (4, -4))
p = float(predicted_prob) if predicted_prob is not None else 0.5
predicted_move = p * up + (1 - p) * down
```

Where:
- `predicted_prob` = `catalyst.get("confidence_score")` (line 950) — a
  scalar from `catalyst_universe.confidence_score`, populated by the
  LLM normalizer.
- `REF_MOVES` = 18-entry hand-tuned dict (line 42), keyed by
  `catalyst_type` only.
- Default tuple `(4, -4)` covers any unmapped type (e.g. "Other",
  "Trial Initiation").

There is no model inference, no feature engineering, no training, no
sentiment pipeline, no NPV input, no fundamental input. Just two
lookups and one expression.

The recompute-predictions admin endpoint
([admin.py:1560-1583](../routes/admin.py#L1560)) re-applies the same
formula on existing rows when REF_MOVES is hand-tuned.

## End-to-end trace — NTLA nex-z, 2025-10-27

**Outcome row in DB:** `post_catalyst_outcomes.id = 10305`,
`catalyst_id = 7847`, `predicted_move_pct = 3.2`,
`actual_move_pct_30d = -66.7`, `outcome_label_class = DELAYED`
(Grade 4 liver tox → FDA clinical hold, evidence in
`outcome_labeled_json`).

**Step-by-step flow:**

| # | Step | File:line | What happens |
|---|---|---|---|
| 1 | EDGAR scrape | [services/edgar_scraper.py:326](../services/edgar_scraper.py#L326) | Fetches NTLA's 8-K filed 2025-10-24 (URL stored as `catalyst_universe.source_url`). |
| 2 | Stage row | edgar pipeline → `catalyst_backfill_staging` | Raw text excerpt + filing date written to staging. |
| 3 | LLM normalize | [backfill_normalizer.py:126-203](../services/backfill_normalizer.py#L126) | Gemini Flash 2.5 with `response_schema` returns `{is_clinical_catalyst:true, catalyst_type:"Other", drug_name:"nex-z", indication:"...ATTR-CM...", confidence:0.9, ...}`. The 8-K narrative described a clinical hold; the LLM's catalyst_type rubric only had Phase 1/2/3 / FDA / Submission / AdComm / **Other** — clinical hold doesn't fit, so it lands in "Other". |
| 4 | Insert universe row | [backfill_normalizer.py:524-549](../services/backfill_normalizer.py#L524) | `confidence_score = float(normalized.get("confidence") or 0.5)` → **0.9**. Row id = 7847. |
| 5 | Tracker fires | [post_catalyst_tracker.py:949-955](../services/post_catalyst_tracker.py#L949) | `predicted_prob = 0.9` (← confidence_score). `cat_type = "Other"` is **not in REF_MOVES**, so `(up, down) = (4, -4)` (the default). |
| 6 | Compute predicted move | post_catalyst_tracker.py:955 | `predicted_move = 0.9 × 4 + 0.1 × (-4) = +3.2`. |
| 7 | Compute predicted direction | post_catalyst_tracker.py:968-973 | Sign of predicted_move > 0 → predicted_direction = +1. |
| 8 | Trade signal classification | [catalyst_signal.py:328-329](../services/catalyst_signal.py#L328) | `cat_type "Other"` ∈ NON_BINARY_CATALYST_TYPES → returns `NO_TRADE_NON_BINARY`. The row is **excluded from tradeable accuracy** but included in raw all-events accuracy. |
| 9 | Insert outcome row | post_catalyst_tracker.py:980-1000 | `prediction_source = "reference_move"` (the literal string).  predicted_at, all metrics persisted. |
| 10 | Outcome scoring (later) | post_catalyst_tracker._infer_outcome → llm fallback | day1 = -45.5 %, day30 = -66.7 %, abnormal_30d = -80.1 %. `direction_correct = False`. `error_abs_pct = 83.3` (predicted +3.2, actual -80). |
| 11 | Outcome labeling (later) | [outcome_labeler.py](../services/outcome_labeler.py) | Gemini grounded search finds the press release, returns `outcome_class = DELAYED`. `outcome_labeled_at = 2026-05-01`. |

**Headline finding from the trace:** the predicted move was +3.2 %
because the LLM was 90 % confident about extraction. It would have
been +3.2 % if the same LLM had read a press release announcing a
disastrous trial failure with the same "Other" catalyst_type and
the same extraction confidence — there is no path in the code that
makes the prediction respond to the *substance* of the event.

## Feature inventory

The single feature consumed at prediction time is `catalyst_type` (to
look up `REF_MOVES`) plus `confidence_score` (the misnamed
"probability"). All other columns in `catalyst_universe` and
`post_catalyst_outcomes` are computed AFTER the prediction and used
for downstream classification (V1/V2 signals) or accuracy
attribution.

| feature | source | type | required by prediction? | nullable | observed % null |
|---|---|---|---|---|---|
| `catalyst_type` | `catalyst_universe.catalyst_type` set by LLM normalizer (8 enum values + NONE) | categorical | **yes** | yes (defaults to `(4, -4)`) | 0 % of `catalyst_universe` rows have it null |
| `confidence_score` | `catalyst_universe.confidence_score` set from LLM `confidence` field | numeric 0..1 | **yes** | yes (defaults to 0.5) | 0 % null on 14,300 rows; mean=0.881, min=0, max=1 |
| `drug_name`, `indication`, `phase`, `date_precision` | `catalyst_universe.*` | string / cat | **no** — recorded but not consumed | yes | varies, not load-bearing |
| `pre_event_price`, `runup_pre_event_30d_pct`, `priced_in_score`, `iv_euphoria`, `options_implied_move_pct`, `predicted_npv_b`, etc. | `post_catalyst_outcomes.*` set AFTER price-window backfill | numeric | **no — only consumed by V1/V2 signal classifier and the dormant LGBM** | yes | varies; ~30 % of rows lack `runup_pre_event_30d_pct`, ~40 % lack `priced_in_score` |
| LGBM features (predicted_prob, runup, priced_in, vol_30d, pre_price, npv_b, year, month, catalyst_type) | `post_catalyst_outcomes.*` | mixed | **no — not in prod path** | yes | training-set only (n=88 total used, 68/20 split) |

There is **no sentiment feature, no news feature, no analyst feature,
no insider-buying feature** consumed at prediction time. Several
fetchers exist (`fetcher.py:fetch_news_sentiment`,
`fetcher_news.py:fetch_newsapi`, `polygon_data.py:fetch_news`,
`social_sources.py`) but none are wired into the prediction.

## Reference table audit (`REF_MOVES`)

**Definition:** module-level constant in
[services/post_catalyst_tracker.py:42-68](../services/post_catalyst_tracker.py#L42),
**duplicated** in
[services/catalyst_signal.py:68-91](../services/catalyst_signal.py#L68)
(`REF_SCENARIOS`, identical values), **and again** in
[routes/admin.py:1514-1535](../routes/admin.py#L1514) (`recompute-predictions`
endpoint inlines its own copy). Three copies that must be kept in sync
manually.

**Shape:** 18 entries × 1 dimension. Indexed only on `catalyst_type`
string. Each value is a `(up, down)` tuple of percentage moves.

| dimension | covered? |
|---|---|
| catalyst_type | yes (18 entries) |
| market_cap segment | **no** |
| indication / disease area | **no** |
| drug class (small-mol / biologic / cell / gene therapy) | **no** |
| phase (within readout type) | **no** (only via the catalyst_type string itself) |
| sponsor history | **no** |
| sample size of the trial | **no** |
| short interest / float | **no** |

**Calibration source** (per code comment at line 40-41): "287 outcomes
seeded from yfinance + LLM classifier (commit 3a03cb0 / move-stats
endpoint), 2020-2025. Updated apr 26 2026." Hand-tuned against the
mean of historical 1-day moves grouped by catalyst_type and outcome.

**Last-updated history (`git blame`):**
- Last commit touching the constant: **`ac0d6ce`** —
  *"fix(catalyst-timeline): widen aggregate caps + recalibrated
  reference moves"* (the recalibration the code comment refers to).
- Before that: **`559e018`** — initial calibration against yfinance
  outcomes.

**Default for unmapped types:** `(4, -4)`. This is what the NTLA
"Other" trace landed on — and it's worth noting that "Other" plus
"Trial Initiation" account for ~3,900 rows in
`post_catalyst_outcomes`, so a non-trivial slice of the dataset is
predicted by the default tuple.

## Probability-of-success (PoS) computation

**There isn't one.** The system has no per-event PoS calculation.
What looks like PoS in the codebase is:

| candidate | what it actually is |
|---|---|
| `catalyst_universe.confidence_score` | LLM extraction confidence — the variable used by the prediction formula. NOT a PoS. |
| `catalyst_universe.probability` | secondary column, populated only sporadically by the universe seeder; not consumed by the tracker. |
| `catalyst_universe.probability_score` | seen in the schema, never written to in any code path I could find. |
| `catalyst_universe.predicted_prob` | also in the schema, also unused. |
| `catalyst_universe.probin` | also in the schema, also unused. |
| LGBM model output (`predict_v3`) | a calibrated P(direction_up), but not consumed in any prod path. |

There is no segmentation by drug class, indication, phase, sponsor
history, or any other domain feature. There is also no calibration
loop — the system has never compared "rows where confidence_score=0.9"
against "fraction that were actually positive outcomes" and adjusted.

## Sentiment / news pipeline

**Three fetchers exist; none feed the prediction.**

| fetcher | file | output |
|---|---|---|
| `fetch_news_sentiment` | [services/fetcher.py:168](../services/fetcher.py#L168) | yfinance news titles, simple sentiment score |
| `fetch_newsapi` | [services/fetcher_news.py:37](../services/fetcher_news.py#L37) | NewsAPI articles |
| `fetch_news` | [services/polygon_data.py:660](../services/polygon_data.py#L660) | Polygon news |
| `social_sources.py` | various | Reddit / X mentions |

These are surfaced on the stock detail page (`/stocks/[ticker]`) and
fed into `news_npv_impact.py` (which only adjusts the NPV scenario,
not the catalyst predicted_move). **None of them produce a signed
sentiment score that flows into `predicted_move_pct`.**

There is **no negative-event detection** in the prediction path. A
filing announcing "Phase 3 missed primary endpoint" gets the same
`predicted_move = +3.2 %` as a filing announcing "Phase 3 met primary
endpoint", because the LLM normalizer extracts the same
catalyst_type ("Phase 3 Readout") with similar confidence, and
neither the catalyst_type nor the confidence_score is sensitive to
the *outcome*. The system is structurally unable to predict in the
direction of bad news.

## V1 vs V2 — the actual difference in code

Both V1 and V2 share the **same** `predicted_move_pct` value (the
`p × up + (1 − p) × down` formula). The V1/V2 split lives in
[services/catalyst_signal.py](../services/catalyst_signal.py) and
only affects the **trade signal label**, not the move magnitude.

| aspect | V1 (`classify_trade_signal`) | V2 (`classify_trade_signal_v2`) |
|---|---|---|
| `predicted_move_pct` | identical | identical |
| Signal labels output | `LONG`, `SHORT`, `NO_TRADE_AMBIGUOUS_PROB`, `NO_TRADE_SMALL_EDGE`, `NO_TRADE_LOW_CONFIDENCE`, `NO_TRADE_BAD_DATE`, `NO_TRADE_NON_BINARY`, `NO_TRADE_OPTIONS_TOO_EXPENSIVE` | same set + `LONG_UNDERPRICED_POSITIVE`, `SHORT_SELL_THE_NEWS`, `SHORT_LOW_PROBABILITY`, `NO_TRADE_PRICED_IN` |
| Direction logic | probability bias only: `\|p − 0.5\| > 0.10` | probability bias **plus** `priced_in_score` cutoffs (≤0.60 → LONG_UNDERPRICED, ≥0.80 → SHORT_SELL_THE_NEWS, mid → NO_TRADE_PRICED_IN) |
| `priced_in_score` | not consumed | composite from `runup_30d_vs_xbi`, `options_implied_move_pct`, `iv_euphoria_pct` (line 224-277) |
| Persisted columns | `trade_signal`, `tradeable` | `signal_v2`, `priced_in_score`, `direction_correct_v2` |

**Currently in production:** both run. The nightly reclassifier
([routes/admin.py V2 reclassify endpoints](../routes/admin.py)) writes
both `trade_signal` (V1) and `signal_v2` simultaneously. Old
predictions are NOT frozen — they get re-scored against whichever
threshold the latest V2 deploy carries (which is part of why
in-sample tuning has been a concern; the
`prediction_snapshots` table exists specifically to give us frozen
records for OOS evaluation, but those snapshots only start *after*
the V2 reclassifier runs, so historical events are still in-sample).

## Change surface — files that, if modified, would change the prediction output

| layer | file | what changes here would do |
|---|---|---|
| **the formula itself** | [services/post_catalyst_tracker.py:949-955](../services/post_catalyst_tracker.py#L949) | swap `confidence_score` for a real PoS, replace the formula entirely, add features |
| **the reference table** | same file, [lines 42-68](../services/post_catalyst_tracker.py#L42) | recalibrate (up, down), add segmentation dimensions (cap × type × indication) |
| **second copy of REF_MOVES** | [services/catalyst_signal.py:68-91](../services/catalyst_signal.py#L68) | must change in lock-step with the post_catalyst_tracker copy |
| **third copy of REF_MOVES** | [routes/admin.py:1514-1535](../routes/admin.py#L1514) | recompute-predictions endpoint inlines its own copy; must change in lock-step |
| **the misnamed "probability"** | [services/backfill_normalizer.py:524](../services/backfill_normalizer.py#L524) | the line that sets `confidence_score = LLM extraction confidence`. A real fix probably means renaming this column AND adding a separate `p_positive_outcome` column populated by an actual classifier |
| **wire up the LGBM model** | [services/lgbm_classifier.py:307 `predict_v3`](../services/lgbm_classifier.py#L307) + new call site in `post_catalyst_tracker.py` | currently dormant; would need a call site at the point where `predicted_prob` is set |
| **add a sentiment / news feature** | new — none of the fetchers are connected to the prediction | would need a function that runs at prediction time and returns a signed adjustment |

A minimum-viable fix that addresses audit-01's anti-alpha finding
without re-architecting touches three of these files:
`post_catalyst_tracker.py` (split confidence_score from p_positive),
`backfill_normalizer.py` (don't conflate them), and `admin.py`
(propagate the rename through the recompute path).

## Open questions for S2b (data inventory) and S2c (miss patterns)

1. **What's the actual distribution of `confidence_score` by
   `outcome_label_class`?** If REJECTED rows have mean confidence_score
   of ~0.85 and APPROVED rows have ~0.90 (suspected), then
   `confidence_score` carries near-zero signal for outcome direction
   — confirming it's an extraction quality, not a PoS.
2. **What columns in `catalyst_universe` are actually populated?**
   `probability_score`, `probability`, `predicted_prob`, `probin` all
   exist in the schema. Are any of them populated with anything
   meaningful? If yes, where from?
3. **Why is the LGBM v3 model dormant?** Was the lack of wiring
   intentional (waiting for more labels) or an oversight? With the
   refreshed 6,139-row labeled set, retraining and wiring is
   tractable.
4. **What's the join shape of news/sentiment data with
   `post_catalyst_outcomes`?** The fetchers exist; building a join
   key (ticker × catalyst_date ± window) and running a one-shot
   sentiment join would tell us whether sentiment data is
   even available pre-event for the historical set.
5. **Per audit-02's finding** that `predicted_prob ∈ [0.90, 1.00]`
   produces all top-20 worst misses — is the **source** of those
   confidence_score=1.0 values uniformly the LLM normalizer (likely),
   or is there a path where some other process bumps it (e.g.
   `routes/admin.py:5668` — `confidence_score = GREATEST(...)`)? S2c
   should grep the write paths and check.
6. **Is there any code that ever consumes the LGBM model's output?**
   I found `predict_v3` is only called from the info endpoint, but
   audit assumes thoroughness — worth a final grep across `jobs/`
   and `scripts/` (currently both directories are nearly empty).

## What this audit deliberately did NOT do

- Propose any code changes.
- Audit `npv_model.py` beyond confirming it's not in the predicted_move
  path (NPV is a separate computation surfaced on the detail page,
  not fed into `predicted_move_pct`).
- Re-evaluate the V1 vs V2 backtest numbers (audit-01/02 already did).
- Touch the `predict_v3` wiring or REF_MOVES values.

Same hand-off contract as audit-01 / 02.
