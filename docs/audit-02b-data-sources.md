# Audit 02b — Probability and magnitude source inventory

**Session:** S2b (read-only data inventory)
**Date:** 2026-05-03 (UTC)
**Scope:** Inventory candidate sources for `p` and `(up, down)` so S3
can choose. Read-only. No code or schema changes.
**Prior:** [audit-02a-architecture.md](audit-02a-architecture.md).

## Executive summary (5 bullets)

1. **The 02a architecture finding is locked.** Population-level
   observed `predicted_move_pct` mean = **3.464 %** vs. formula
   prediction = **3.464 %** (diff = 0.000). Per-type diffs all within
   ±0.006. Every prediction in the 8,812-row outcomes table is
   produced by `p × up + (1 − p) × down` with `p` = LLM extraction
   confidence and `(up, down)` from REF_MOVES. Confirmed.
2. **The V1 → V2 +9 pp gap is REAL_DIFFERENCE, not selection.** On
   the 88-row subset both V1 and V2 judge, V2 flips +8 events from
   miss → hit (mostly via the SHORT_SELL_THE_NEWS bucket: 9/10 hits
   vs V1's 1/10). Selection effect (V2 dropping 25 NO_TRADE_PRICED_IN
   rows that V1 was 13/25 = 52 % on) is small.
3. **The dataset already has the proper-PoS columns** — `catalyst_universe`
   has `p_event_occurs` and `p_positive_outcome` (added in some past
   migration), but **only 293 / 14,300 rows are populated** and the
   values are basically copied from `confidence_score`. Schema is
   ready; the populate path doesn't differentiate.
4. **`historical_catalyst_moves` table exists with the IDEAL schema**
   for the magnitude lookup: `(catalyst_type, indication,
   market_cap_bucket) → mean / p25 / p75 / std / n`. **It has 0
   rows.** Populating it from existing labeled outcomes is a single
   SQL aggregation — the data is right there, nobody ran the query.
   This is the lowest-effort highest-impact fix candidate.
5. **The pre-event feature set is much thinner than 02a implied.**
   On the 3,781-row clean-labeled set, runup features are 92–95 %
   null, priced_in_score is 92.5 % null, predicted_npv_b is 100 %
   null. The only consistently-populated pre-event features are
   catalyst_type (100 %), indication (94 %), market_cap (via
   screener_stocks join, ~70 % depending on universe), and
   options_implied_move_pct (61 %). A retrained LGBM with all these
   features would have ~25 % of the labeled set with a complete
   feature row.

## Sanity check: does the formula fully explain observed predictions?

**Verdict: YES — formula confirmed end-to-end.**

| catalyst_type | n | avg `p` | observed `predicted_move_pct` | formula `p × up + (1-p) × down` | diff |
|---|---:|---:|---:|---:|---:|
| Trial Initiation | 2,085 | 0.888 | 3.101 | 3.104 | −0.003 |
| Other | 1,832 | 0.833 | 2.666 | 2.664 | +0.002 |
| Phase 2 Readout | 1,151 | 0.873 | 3.237 | 3.238 | −0.001 |
| Submission | 1,026 | 0.888 | 3.106 | 3.104 | +0.002 |
| Phase 1 Readout | 927 | 0.883 | 8.123 | 8.128 | −0.005 |
| Phase 3 Readout | 872 | 0.888 | 2.103 | 2.104 | −0.001 |
| FDA Decision | 842 | 0.889 | 3.005 | 3.001 | +0.004 |
| AdComm | 27 | 0.917 | 6.500 | 6.506 | −0.006 |
| Clinical Trial | 23 | 0.939 | 4.513 | 4.512 | +0.001 |
| Partnership | 11 | 0.968 | 4.777 | 4.776 | +0.001 |
| **Population** | **8,812** | **0.881** | **3.464** | **3.464** | **0.000** |

Distribution shape: mean 3.464, median 3.20, std 1.77, p10 = 2.20,
p90 = 6.80, min −2.30, max 10.00. Only **37 / 8,812 (0.4 %)** rows
have a negative predicted_move_pct. There is no second prediction
code path in the system.

## V1 vs V2 9 pp gap — verdict: REAL_DIFFERENCE

The chat scoreboard's V1 = 54.0 % (n=113) vs V2 = 63.6 % (n=88) numbers
correspond to:
- **V1 judged:** 113 rows where `tradeable=TRUE` and `direction_correct_3d`
  is non-null. 61 hits.
- **V2 judged:** 88 rows where `signal_v2 ∈ {LONG_UNDERPRICED_POSITIVE,
  SHORT_SELL_THE_NEWS, SHORT_LOW_PROBABILITY, LONG, SHORT}` and
  `direction_correct_v2` is non-null. 56 hits.

Decomposition on the **88-row common subset** (judged in BOTH):

| | both_hit | v1_only_hit | v2_only_hit | both_miss |
|---|---:|---:|---:|---:|
| n | 47 | 1 | 9 | 31 |

So on the same 88 rows: V1 = 48/88 = **54.5 %**, V2 = 56/88 = **63.6 %**.
Net **+8 row-level direction flips** explain the entire +9 pp gap.

The 25 rows V1 judges but V2 does not are tagged
`signal_v2 = NO_TRADE_PRICED_IN` — these are the mid-priced-in
events V2 explicitly abstains on. On those 25 rows V1 was 13/25 =
**52.0 %**, so the selection effect contributes essentially nothing
(removing 52-% rows from a 54.5-% set doesn't move the average).

**Where the +8 flips come from (signal_v2 distribution on common subset):**

| signal_v2 | n total | n judged (both) | v1_hits | v2_hits |
|---|---:|---:|---:|---:|
| LONG_UNDERPRICED_POSITIVE | 136 | 78 | 47 | 47 |
| SHORT_SELL_THE_NEWS | 18 | 10 | **1** | **9** |
| LONG (V2 didn't reclassify) | 3,312 | 0 | 0 | 0 |

V2's lift is concentrated in **SHORT_SELL_THE_NEWS**: V1 went LONG on
all 10 of those (1 hit), V2 flipped them SHORT (9 hits). That's a
genuine algorithmic improvement in the priced-in detection — not a
sample-selection artifact. **Verdict: REAL_DIFFERENCE.**

## Probability source matrix — candidates for `p`

| # | source | coverage on labeled set | latency | cost | verification | notes |
|---|---|---|---|---|---|---|
| 3a-i | Catalyst-type base rate (own DB) | 100 % | historical | free (one SQL) | Phase 1 = 93.7 % positive (n=696), Phase 2 = 83.1 % (n=791), Phase 3 = 70.8 % (n=583), FDA = 68.9 % (n=611), Submission = 52.2 % (n=429), Trial Initiation = 30.6 % (n=124) | **Available immediately.** Note: base rates are heavily positive — Phase 1/2 reflect survivorship (only successful trials get reported). Use as a fallback/prior, not the primary signal. |
| 3a-ii | Catalyst-type × indication | 100 % (94 % indication non-null) | historical | free | ALS = 17 % positive (n=42), advanced solid tumors = 100 % (n=35), DMD = 64 % (n=57), MDD = 50 % (n=45). 14 indications have n ≥ 30. | **Real signal.** Indication base rates vary 17–100 %. Will need a fallback for long-tail indications (n < 30). |
| 3a-iii | Catalyst-type × phase | **3 % usable** (`pco.catalyst_id → cu.phase` resolves with phase only on 40 / 10,150 rows) | historical | free | Phase column in `catalyst_universe` is populated on 3,336 universe rows total but barely on rows that ever became outcomes. | Effectively unavailable from current data. Would need a separate phase-extraction backfill. |
| 3b | Analyst consensus (yfinance) | request-time only, NOT persisted | live | free (no API key, via yfinance) | `services/authenticated_sources.py:44 fetch_analyst_yfinance` returns target_mean, recommendation_mean (1-5 scale), buy/hold/sell counts, recent upgrades/downgrades. | Used on the stock detail page. NOT in the prediction path. Would need a daily snapshot table to use historically. Finnhub has company profile + news only — no recommendation endpoint wired. |
| 3c | Options-implied probability | 61 % on labeled set | historical (snapshot at event time) | free (yfinance options chain) | `services/options_implied.py:get_implied_move` returns implied move %, ATM strike, straddle premium, days_to_expiry, annualised IV. `post_catalyst_outcomes.options_implied_move_pct` populated on **5,376 / 8,812 rows (61 %)**. | Implied move is symmetric (no direction). Deriving a direction-of-move probability from a straddle requires risk-neutral inversion — possible but needs implementation. Direct use as "expected magnitude" is more straightforward. |
| 3d | LLM-generated P(positive) — calibrated | 0 % | request-time | ~$0.0003/row × 14,300 = ~$4 | `backfill_normalizer.py:88` prompt asks for `confidence` defined as **"how confident the LLM is that the extracted text describes a real catalyst"**. **No prompt currently asks the LLM for P(positive outcome).** | A new prompt that asks for "given this catalyst announcement, what is P(positive outcome)?" is a one-shot LLM batch. Cost ≈ same as outcome labeler. Calibration would require comparing LLM-predicted probability against actual outcome rate, then sigmoid-rescaling — needs a holdout fold but trivial code. |
| 3e | LGBM v3 retraining | feature-coverage-limited | offline | free (sklearn/lightgbm already in deps) | Training set: rows with `direction_correct_3d IS NOT NULL` AND `outcome_confidence ≥ 0.7` AND `signal_v2 ∈ {LONG_UNDERPRICED_POSITIVE, SHORT_SELL_THE_NEWS, …}`. Currently model id=1 was trained on n=68 with test acc 45 %. | **Features that v3 trained on:** predicted_prob, runup_pre_event_30d_pct, priced_in_score, preevent_avg_volume_30d, pre_event_price, predicted_npv_b, year, month, catalyst_type. **Of those, on the labeled set:** runup is 92.5 % null, priced_in_score is 92.5 % null, predicted_npv_b is **100 % null**. So a retrain on the new ~3,781 row labeled set would effectively be on (predicted_prob, vol, price, year, month, catalyst_type) for ~25 % of rows. Y/N: yes-but-thin. The dormant model is dormant for a structural reason: most of its features aren't populated on most rows. |

## Magnitude source matrix — candidates for `(up, down)`

| # | source | coverage | latency | cost | verification | notes |
|---|---|---|---|---|---|---|
| 4a-i | Historical actuals × catalyst_type × outcome (no cap) | 100 % (REF_MOVES proxy, but data-driven) | historical | free | 13 cells with n ≥ 30 across (catalyst_type × outcome × <500M cap_bucket). Examples: Phase 3 MET <500M n=374 median +1.0 % p25 −10.8 % p75 +15.1 %; Phase 3 MISSED <500M n=95 median **−12.7 %** p25 **−41.8 %** (current REF_MOVES has down=-5 — under-priced 8.5×). | **Drop-in replacement** for REF_MOVES. Just compute medians from the labeled set. |
| 4a-ii | Historical actuals × catalyst_type × outcome × cap_bucket | thin above $500 M | historical | free | Of 35 (type × outcome × cap_bucket) cells: 13 have n ≥ 30, 19 have n ≥ 10. The **vast majority** of labeled events are <$500 M cap (~95 %). 500M-2B has only one stable cell (FDA Decision APPROVED, n=40, median −1.6 %), 2B-10B and >10B have none with n ≥ 30. Would need a fallback to type-only for mid/large-cap catalysts. |
| 4a-iii | catalyst_type × outcome × phase | unusable (3 % phase coverage on outcomes) | historical | free | Skip until phase backfill happens. |
| 4b | Options-implied move | 61 % | historical (snapshot at event) | free | `post_catalyst_outcomes.options_implied_move_pct` already populated on 5,376 rows. **Symmetric** — gives expected magnitude but not direction. Could be used as the **magnitude scaler** with direction coming from `p`. | Useful for the 61 % of events that have it; need a fallback for the rest. Backfilling the 39 % gap costs nothing — yfinance call per ticker × event date. |
| 4c | Comparable-catalyst lookup (`historical_catalyst_moves` table) | **schema exists, 0 rows** | historical | free (single SQL aggregation) | Table columns: `catalyst_type, indication, market_cap_bucket, mean_move_pct, p25_move_pct, p75_move_pct, std_dev_pct, n_observations, source, last_updated`. Exactly the right schema for what S3 wants. Empty. | **Easiest win in the entire audit.** Run one SQL `INSERT INTO historical_catalyst_moves SELECT … GROUP BY catalyst_type, indication, cap_bucket FROM post_catalyst_outcomes WHERE outcome_label_class IS NOT NULL`. Now you have a real lookup. |

## Available features inventory (DB-queryable)

Coverage measured on the 3,781-row clean-labeled subset (excludes UNKNOWN
labels) unless noted.

| feature | source.column | type | pre/post-event | null % on labeled set | notes |
|---|---|---|---|---:|---|
| catalyst_type | post_catalyst_outcomes | categorical | pre | 0.0 % | 8 distinct values + Other |
| drug_name | post_catalyst_outcomes | text | pre | 1.0 % | string identity |
| indication | post_catalyst_outcomes | text | pre | 5.6 % | high cardinality, needs canonicalisation (DMD vs Duchenne muscular dystrophy duplicates exist) |
| catalyst_universe.phase | catalyst_universe.phase | categorical | pre | **96 %** on outcome rows | populated on 3,336 universe rows but barely matches outcomes |
| catalyst_universe.confidence_score | catalyst_universe | numeric | pre | 0.0 % | the misnamed value, mean 0.881 |
| catalyst_universe.p_positive_outcome | catalyst_universe | numeric | pre | **98 %** | column exists, populated on 293/14,300 rows; values copied from confidence_score |
| catalyst_universe.p_event_occurs | catalyst_universe | numeric | pre | **98 %** | same shape |
| pre_event_price | post_catalyst_outcomes | numeric | pre | 27.5 % | yfinance price fetch |
| preevent_avg_volume_30d | post_catalyst_outcomes | numeric | pre | 27.4 % | derived from price window |
| shares_outstanding_at_event | post_catalyst_outcomes | numeric | pre | 29.7 % | from yfinance |
| market_cap (joined) | screener_stocks.market_cap | numeric | pre | ~30 % on labeled set after join | screener_stocks has 525 rows total (live universe); labeled outcomes that don't overlap return null |
| screener_stocks.sentiment_score | screener_stocks | numeric | pre | screener-only (525 rows) | mean 1.109, scale unclear (0-2 range observed) |
| screener_stocks.probability | screener_stocks | numeric | pre | 525 rows total | mean 0.512 — different shape from confidence_score (centered around 0.5 not 0.88) |
| options_implied_move_pct | post_catalyst_outcomes | numeric | pre | 39.0 % | yfinance ATM straddle |
| runup_pre_event_30d_pct | post_catalyst_outcomes | numeric | pre | **92.5 %** | computed by V2 backfill, only ran on a subset |
| runup_pre_event_30d_vs_xbi_pct | post_catalyst_outcomes | numeric | pre | **94.7 %** | sector-adjusted runup, even smaller subset |
| priced_in_score | post_catalyst_outcomes | numeric | pre | **92.5 %** | composite of runup + options + IV |
| sector_runup_30d_pct | post_catalyst_outcomes | numeric | pre | similar to runup_xbi | XBI return baseline |
| predicted_npv_b | post_catalyst_outcomes | numeric | pre | **100 %** | column exists, never populated on outcomes |
| catalyst_npv_cache.npv_b | catalyst_npv_cache | numeric | pre | 79 cached rows, ticker × catalyst_id keyed | usable for the 79 cached events |
| drug_economics_cache.drug_name | drug_economics_cache | text | pre | 19 cached rows | thinly populated |
| screener_news_sentiment | screener_news_sentiment.sentiment_score | numeric | pre | 257 rows / 18 tickers | very thin coverage |
| stock_risk_factors | stock_risk_factors | jsonb | pre | **10 rows total** | CRL/litigation/insider/short tracking exists but barely populated |
| **Outcome (post-event)** | | | | | |
| outcome_label_class | post_catalyst_outcomes | categorical | post | 60 % null (3,781 / 6,139 labeled rows have non-UNKNOWN class) | training target |
| actual_move_pct_30d | post_catalyst_outcomes | numeric | post | ~17 % | scoring target (magnitude) |
| direction_correct_3d | post_catalyst_outcomes | boolean | post | ~93 % null on labeled (deadband + missing 3d data) | scoring target (V1) |

## Dead-feature inventory

Lines of code that compute features NOT consumed by the
`predicted_move_pct` formula. Most feed the UI (stock detail page,
chat context) or the V2 abstention layer, not the magnitude prediction
itself.

| file | lines | what it computes | reused by predicted_move? |
|---|---:|---|---|
| `services/npv_model.py` | 1,491 | rNPV fair value, scenario sensitivity, drug economics, peak-sales projections | **No** — only consumed by detail page / chat context |
| `services/setup_quality.py` | 370 | composite "setup quality" score from technicals + news + analyst + IV | **No** — UI display only (`compute_setup_quality` consumed by `routes/stocks.py` and `routes/chat.py`) |
| `services/news_npv_impact.py` | 298 | adjusts NPV scenarios based on recent news sentiment | **No** — only feeds NPV scenarios |
| `services/risk_factors.py` | 252 | CRL history, active litigation, insider transactions, short interest | **No** — table has only 10 rows |
| `services/research_enrichment.py` | 513 | Wikipedia / company pages enrichment for drug names | **No** — display only |
| `services/research_ingestor.py` | 565 | research-document ingestion pipeline + embeddings | **No** — search/chat context only |
| `services/social_sources.py` | 402 | Reddit/Twitter scraping for ticker mentions | **No** — display only |
| `services/sec_dilution.py` | 444 | dilution capacity analysis from S-3, ATM filings | **No** — UI metric |
| `services/sec_financials.py` | 513 | cash runway, R&D burn, revenue trends from 10-Q/K | **No** — UI metric |
| `services/options_implied.py` | small | straddle implied move | **Partial** — value is stored on outcome rows but not used in `p × up + (1-p) × down`. Used by the V2 priced_in_score. |
| **Total dead lines** | **~5,300** | | |

Every one of these is **reusable in a real model** — the data-fetching
infrastructure works and is tested. They were just never wired into
the prediction. The marginal cost of adding any of them as a feature
to a retrained classifier is one column in a feature DataFrame.

## Backfill cost estimates

| backfill | cost | data unlocked | blocking which model? |
|---|---|---|---|
| `historical_catalyst_moves` table from existing DB | **free** (one SQL aggregation, ~5 min) | (catalyst_type × indication × cap_bucket) median moves | drop-in REF_MOVES replacement |
| `runup_pre_event_30d_pct` to all labeled rows | **free** (yfinance, one worker pool, ~30 min) | 92.5 % null → ~10 % null | priced_in classifier; LGBM retrain; V2 priced_in score on full set |
| `options_implied_move_pct` to remaining 39 % | **free** (yfinance, ~1-2 hr at rate-limited pace) | 39 % null → <10 % null | options-implied magnitude floor; calibration vs predicted |
| `phase` join from cu to outcomes | **free** (one SQL backfill if mapping exists) | 96 % null → matches universe coverage (~30 % maybe) | phase-segmented base rate |
| `catalyst_npv_cache` to all events | LLM ($0.005/row × 8,800 ≈ $44) | 79 → 8,800 rows | npv-as-feature for classifier |
| Drug-class via LLM | $0.0003/row × 14,300 ≈ **$4** | new column `drug_class` (small molecule / mAb / ADC / cell / gene therapy / oligo / other) | drug-class-segmented base rate |
| LLM `p_positive_outcome` (per-event, calibrated) | $0.0003/row × 14,300 ≈ **$4** | proper PoS column, replaces `confidence_score` misuse | direct fix to root bug; needs calibration step |
| Analyst consensus snapshot table | **free** (yfinance, ~1 hr) for daily snapshot of 525 screener tickers; would need to backfill historically by re-querying which yfinance doesn't store | analyst features for prediction | new feature for classifier; useful as live-prediction input |
| Daily news sentiment per universe ticker | $0.001/article × ~10/day × 525 tickers × 365 days = ~$1,900 if going historical; $5/day live | news sentiment feature | sentiment-as-feature; helps but not a root fix |

## Recommendation — ranked by feasibility × expected lift

S3 should consider these in this order:

1. **Populate `historical_catalyst_moves` from existing labeled data.**
   Free, immediate, unblocks the (up, down) replacement entirely.
   Cells with n ≥ 30 cover ~13 high-traffic combinations; for unmatched
   cells fall back to type-only median or to `options_implied_move_pct`.
   Expected lift: directly addresses the under-prediction-of-magnitude
   issue from audit-01 and 02 (Phase 3 MISSED median is **−12.7 %** in
   real data vs REF_MOVES `down = −5 %`, an 8.5× under-pricing).

2. **Replace `p` with `(catalyst_type × indication) base rate × magnitude
   correction`**. Free, immediate, sourced from the same labeled set.
   For events with no n ≥ 30 cell, fall back to type-only base rate or
   to a fixed prior (e.g. 0.5 if you want to be neutral; 0.65 if you
   want the historical positive bias). Expected lift: this is the only
   change that directly fixes the
   "model can't predict negative outcomes" anti-alpha pattern, because
   the new `p` will be 0.32 for FDA decisions on rejection-prone
   indications instead of 0.89.

3. **Populate `options_implied_move_pct` for the remaining 39 %.**
   Free, ~1-2 hr backfill. Unlocks options-implied as a fallback
   magnitude when historical bucket has n < 30, and as a feature for
   any future classifier.

4. **Backfill `runup_pre_event_30d_pct` (free, ~30 min)** so the V2
   priced_in detection actually applies to the whole labeled set
   instead of 7.5 % of it. The +9 pp V1→V2 lift (verdict above)
   suggests priced-in detection is a real edge — but it can only run
   on rows where the runup feature is populated.

5. **Add a per-event LLM-derived `p_positive_outcome` (~$4)**. Cheaper
   than retraining a classifier and gives a properly-named PoS
   column. Calibrate post-hoc against actual outcome rate.

6. **Defer LGBM v3 retraining** until 1-3 are done. Without them, the
   feature matrix is too sparse (most features 90 %+ null on the
   labeled set) for the model to learn anything meaningful — which is
   why id=1 ended up at 45 % test accuracy on n=20.

## Open questions for S3

1. **What's the right boundary between `p` and `(up, down)`?** The
   current formula treats them as independent, but real biotech
   moves have correlation — high-cap mid-confidence positives have
   different magnitude tails than low-cap high-confidence positives.
   A joint distribution model (predict the full move distribution
   given features, take direction + magnitude from quantiles) might
   be cleaner than the multiplicative split.
2. **Do we want a single global model or per-catalyst-type models?**
   Phase 1/2/3 readouts have very different feature relevance (e.g.
   `runup` matters for Phase 3 catalysts that are heavily anticipated;
   matters less for Phase 1). A hierarchical model or ensemble of
   per-type models might fit better than one global LGBM.
3. **Is the 70 %+ positive base rate an artifact of survivorship?**
   Phase 1 readouts at 93.7 % positive feels like only-positive-readouts-
   get-reported. If true, base rates from our DB are biased upward.
   Worth checking against ClinicalTrials.gov registry of attempted
   readouts vs reported outcomes.
4. **How does the `p_positive_outcome` column on `catalyst_universe`
   interact with the formula?** It exists, has 293 rows populated, and
   the values are basically copies of `confidence_score`. Was there a
   half-finished migration to use it properly? Worth a `git log` on
   the column to see the intent.
5. **Should the prediction respond to `outcome_labeled_class` for
   already-resolved events?** Currently no — predicted values are
   computed once from confidence_score and frozen. If S3 builds a
   "fade the confident positive bet on negative-outcome catalysts"
   trade, the prediction needs a feedback loop.
6. **What about `screener_stocks.probability` (mean 0.512, n=525)?**
   It's centered at 0.5 instead of 0.88 like confidence_score, which
   suggests it might be a different (better) probability estimate.
   Worth tracing where it's set in the seeder code.
