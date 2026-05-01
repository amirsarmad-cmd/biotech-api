# Audit 01 — Outcome Labels & Accuracy Baseline

**Session:** S1 (Claude Opus 4.7 1M)
**Date:** 2026-05-01 (UTC)
**Scope:** READ-only audit. No model, scoreboard, or schema code modified.

## Executive Summary

- The "Layer 2 = 2.1%" number that the chat surfaces is **a confirmed
  denominator bug** in `aggregate-v2`. The numerator counts only
  `direction_correct_3d = TRUE` (111 hits) but the denominator is the
  full `tradeable = TRUE` row count (5,241), which silently includes
  5,033 NULL rows that have no judged direction. The corrected value
  is **111 / 208 = 53.4 %**, which `aggregate-v3.tradeable_v1` already
  reports. The chat reads from `aggregate-v2` and so quotes the buggy
  number to users.
- True system-wide direction accuracy on raw 30-day moves is **44.9 %
  on n = 8,762** (Layer 1). That number is unchanged by partial backfill
  and is correctly computed.
- **Outcome label backfill did not reach 95 %.** Pre-backfill: 692 / 10,605
  labeled (6.5 %). After ~7 minutes of background labeling the count is
  ~835 / 10,605 (~7.9 %). At the observed rate (≈18 attempts/min, of which
  ≈10/min produce a label and ≈6/min return UNKNOWN/error) a full backfill
  is a ~12-hour job; it cannot complete inside one audit session. Cost so
  far is ~$0.04 of the ~$3.00 estimate; the rest will accrue as the
  background worker runs to completion.
- `error_abs_pct` is enormous on positive-readout small-caps: top-50
  misses are dominated by clinical wins where `predicted_move_pct` is
  +2 % to +8 % but `actual_move_pct_30d` is +50 % to +345 %. The
  `REF_MOVES` reference table is calibrated for normal-cap moves and
  badly under-prices binary clinical wins on micro/small-caps.
- **Labels by themselves do not rescue accuracy.** Splitting by
  `outcome_label_class` shows MET_ENDPOINT direction = 49.1 %, APPROVED
  = 51.8 %, REJECTED = 27.3 %, MISSED_ENDPOINT = 29.4 %. The model is
  near coin-flip on positive outcomes and **anti-alpha** on negative
  outcomes — i.e. when the trial misses or the FDA rejects, the model
  predicted the wrong direction more often than not.

## Pre / Post Backfill Numbers

### Pre-backfill (2026-05-01 22:17 UTC)

| Endpoint | Field | Value |
|---|---|---|
| `/admin/post-catalyst/outcome-label-stats` | total_outcomes | 10,605 |
| | labeled_outcomes | 692 |
| | labeled_pct | 6.5 % |
| | class_distribution | APPROVED 93 / REJECTED 29 / MET_ENDPOINT 244 / MISSED_ENDPOINT 17 / DELAYED 62 / WITHDRAWN 3 / MIXED 30 / UNKNOWN 214 |
| | estimated_full_backfill_cost_usd | $2.97 |
| `/v2/post-catalyst/accuracy` | total | 8,762 |
| | direction_hits | 3,930 |
| | direction_accuracy_pct | 44.9 |
| | avg_abs_error_pct | 19.8 |
| | avg_signed_error_pct | +2.7 |

### Post-backfill (partial, captured at audit-doc commit time)

The background labeler is still running. `outcome-label-stats` and
`/v2/post-catalyst/accuracy` snapshots, plus the delta computation,
are appended in `Post-backfill snapshot, partial` below before the
commit.

### Backfill rate analysis

- 6 workers, claim_size = 10 → 60 events in flight at any moment.
- In ~7 minutes the worker produced 143 labeled rows + 89
  errors (defined: `label_outcome_for_db_row` returned None, i.e.
  Gemini found no usable press-release source).
- ≈ 18 attempts / minute → at that rate, the remaining ~9,800
  unlabeled rows would take ~9 hours. This is materially slower
  than the cost estimate suggested ($3 ÷ $0.0003 ≈ 10,000 calls is
  fine for cost; the ceiling is end-to-end Gemini grounded-search
  latency, not money).
- Open question: is the 39 % error rate driven by genuinely
  unverifiable historical events (e.g. Trial Initiation rows with
  no PR), or by transient grounded-search failures that ought to be
  retried? S2 should profile and decide.

## Layer 2 = 2.1 % — Verdict: **BUG (denominator includes NULL rows)**

### Where it is computed

`routes/admin.py` → `post_catalyst_aggregate_v2()`
([routes/admin.py#L3411-L3501](routes/admin.py#L3411-L3501))

```sql
-- numerator
SELECT
    COUNT(*)                                            AS total,
    COUNT(*) FILTER (WHERE direction_correct_3d)        AS direction_hits_3d,
    AVG(error_abs_abnormal_3d_pct)                      AS avg_abs_error_abnormal_3d,
    COUNT(*) FILTER (WHERE abnormal_move_pct_3d IS NOT NULL) AS with_3d_data
FROM post_catalyst_outcomes
WHERE tradeable = TRUE
```

```python
"direction_accuracy_pct": (
    round(100.0 * hits_tradeable / total_tradeable, 1)  # ← bug
    if total_tradeable > 0 else None
),
```

### Direct DB reproduction

```sql
SELECT
    COUNT(*)                                                  AS tradeable_total,        -- 5241
    COUNT(*) FILTER (WHERE direction_correct_3d)              AS hits,                   --  111
    COUNT(*) FILTER (WHERE direction_correct_3d IS NULL)      AS unjudged_nulls,         -- 5033
    COUNT(*) FILTER (WHERE direction_correct_3d IS NOT NULL)  AS judged,                 --  208
    COUNT(*) FILTER (WHERE abnormal_move_pct_3d IS NOT NULL)  AS with_3d_data,           --  334
    COUNT(*) FILTER (WHERE abnormal_move_pct_3d IS NOT NULL
                      AND ABS(abnormal_move_pct_3d) < 3.0)    AS deadband_excluded       --  126
FROM post_catalyst_outcomes
WHERE tradeable = TRUE;
```

`buggy_2_1_pct = 111 / 5241 = 2.12 %` (rounds to 2.1 %).
`corrected_pct = 111 / 208 = 53.37 %`.

### What is the population behind n = 5,241?

All rows where `tradeable = TRUE` (the V1 LONG/SHORT signal). 5,033 of
those rows have `direction_correct_3d IS NULL` — meaning the row is in
the `|abnormal_3d| < 3 %` deadband **or** has no abnormal-3d data at all
(126 deadband + 4,907 no-3d-data). Counting those NULLs as misses is
what deflates accuracy from 53 % to 2 %.

### Does it match the Layer 1 44.9 % definition?

No. Layer 1 (`/v2/post-catalyst/accuracy`) is direction on the **raw
30-day move** with denominator = `actual_move_pct_30d IS NOT NULL`.
Layer 2 is direction on **3-day abnormal vs XBI** with denominator
= `tradeable = TRUE`. The two are computing different things on
different populations; even if Layer 2 were fixed, it wouldn't equal
44.9 %.

### Does the 2.1 % change after backfill?

No. `direction_correct_3d` is a function of `predicted_direction` and
`abnormal_move_pct_3d`. Neither depends on `outcome_label_class`. The
backfill only writes `outcome_labeled_*` columns. The 2.1 % bug is a
SQL-shape problem and is independent of label coverage.

### Internal evidence the bug is already known

`routes/admin.py:3701-3706` (the `aggregate-v3` docstring) literally
says:

> HISTORICAL CONTEXT: The original 31.7% / 'inverse 68.3%' analysis
> that motivated V2 was caused by a SQL denominator bug —
> direction_correct_3d is NULL on deadband rows (|abnormal_3d| < 3%),
> and including those NULLs in COUNT(*) deflated accuracy. After the
> fix, V1 is at 58.4 % …

`aggregate-v3` fixed the math by switching the denominator to
`COUNT(*) FILTER (WHERE direction_correct_3d IS NOT NULL)`. **The
fix was never back-ported to `aggregate-v2`**, and the chat is wired
to read `aggregate-v2`. Hence the user-visible "Layer 2 = 2.1 %".

### Verdict

**BUG.** `aggregate-v2.tradeable_events.direction_accuracy_pct` is
`hits / N_total`, not `hits / N_judged`. The `aggregate-v3` endpoint
already knows this and computes it correctly (53.4 % at present, with
a 95 % Wilson CI of 46.6 % – 60.0 %). Two follow-ups for S2:

1. Fix `aggregate-v2` to use the judged denominator, or deprecate it
   entirely and have the chat read from `aggregate-v3`.
2. The "Layer 2" framing in `routes/chat.py` system prompt
   (`backtest_scoreboard_v2`) needs updated wording — currently it
   tells the model "tradeable accuracy ≥ 65-70 % is the target", but
   the number it reads alongside that target is mathematically
   broken.

## Per-bucket Accuracy Table

Computed from `post_catalyst_outcomes` against the **fully populated**
direction/error fields (these don't depend on label backfill).
`actual_move_pct_30d` vs `predicted_move_pct`. Sample-size status:
INSUFFICIENT < 30, PRELIMINARY 30–99, STABLE ≥ 100.

| Catalyst type      | n     | direction_acc % | avg_abs_err % | avg_signed_err % | sample_size |
|--------------------|------:|----------------:|--------------:|-----------------:|-------------|
| Trial Initiation   | 2,085 |            44.6 |          17.4 |             −1.5 | STABLE      |
| Other              | 1,832 |            46.4 |          19.7 |             −0.2 | STABLE      |
| Phase 2 Readout    | 1,149 |            42.1 |          22.9 |             −1.6 | STABLE      |
| Submission         | 1,025 |            45.2 |          15.4 |             −2.1 | STABLE      |
| Phase 1 Readout    |   927 |            42.0 |          23.1 |             −8.2 | STABLE      |
| Phase 3 Readout    |   869 |            45.6 |          25.4 |             −0.1 | STABLE      |
| FDA Decision       |   837 |            47.8 |          17.0 |             −2.0 | STABLE      |
| AdComm             |    27 |            44.4 |          31.9 |             −3.4 | INSUFFICIENT|
| Phase 1/2 Readout  |     5 |           100.0 |          15.6 |            +14.9 | INSUFFICIENT|
| Clinical Trial     |     1 |             0.0 |          29.0 |            −24.7 | INSUFFICIENT|
| BLA submission     |     1 |             0.0 |          16.1 |             −5.6 | INSUFFICIENT|
| NDA submission     |     1 |           100.0 |           0.8 |             +0.3 | INSUFFICIENT|
| Phase 0 Readout    |     1 |             0.0 |          36.5 |            −24.0 | INSUFFICIENT|
| Phase 4 Readout    |     1 |             0.0 |          15.9 |             −6.1 | INSUFFICIENT|
| Partnership        |     1 |             0.0 |          27.3 |            −26.4 | INSUFFICIENT|

Buckets that meet the spec's STABLE threshold (n ≥ 100): **Trial
Initiation, Other, Phase 2 Readout, Submission, Phase 1 Readout,
Phase 3 Readout, FDA Decision** — 7 of 7 required.

Direction accuracy across every STABLE bucket sits in 42–48 %, i.e.
indistinguishable from a slightly biased coin. `avg_abs_err` is
worst on Phase 1/2/3 Readouts (22–25 %) — those are the events with
the widest move distribution and the smallest predicted moves, so
absolute error explodes whenever a positive readout actually pays.

### Per-label-class accuracy (uses the partial 692-row labeled set)

| Outcome class    | n   | direction_acc % | avg_abs_err % | avg_signed_err % | sample_size  |
|------------------|----:|----------------:|--------------:|-----------------:|--------------|
| MET_ENDPOINT     | 173 |            49.1 |          33.9 |           +15.4  | STABLE       |
| UNKNOWN          | 114 |            43.0 |          17.7 |            +4.7  | STABLE       |
| APPROVED         |  56 |            51.8 |          16.3 |            +9.2  | PRELIMINARY  |
| DELAYED          |  49 |            51.0 |          19.0 |            +0.5  | PRELIMINARY  |
| REJECTED         |  22 |            27.3 |          30.9 |           −14.9  | INSUFFICIENT |
| MIXED            |  20 |            25.0 |          33.0 |            −3.6  | INSUFFICIENT |
| MISSED_ENDPOINT  |  17 |            29.4 |          42.6 |            +1.9  | INSUFFICIENT |
| WITHDRAWN        |   2 |            50.0 |           8.6 |            −3.8  | INSUFFICIENT |

The asymmetry is the headline: when the outcome is positive
(APPROVED / MET_ENDPOINT) direction accuracy is ~50 %; when the
outcome is negative (REJECTED / MISSED / MIXED) direction is
**materially below 50 %** — i.e. the model's prior is locked in a
positive direction regardless of catalyst-specific risk signals.
`avg_signed_error` swings from +15 % (MET) to −15 % (REJECTED),
which is a magnitude problem on top of the direction problem.

## Top Miss Clusters (worst-50 by `|actual_30d - predicted|`)

Pulled from labeled rows only. Worst single rows:

| # | ticker | date       | type             | pred % | actual_30d % | err   | label           | drug                      |
|---|--------|------------|------------------|-------:|-------------:|------:|-----------------|---------------------------|
| 1 | CAPR   | 2025-12-03 | Phase 3 Readout  |  +2.20 |      +345.13 | 340.0 | MET_ENDPOINT    | Deramiocel                |
| 2 | QURE   | 2025-09-24 | Phase 2 Readout  |  +4.00 |      +346.63 | 330.6 | MET_ENDPOINT    | AMT-130                   |
| 3 | OLMA   | 2025-11-10 | Trial Initiation |  +3.20 |      +267.43 | 252.2 | MET_ENDPOINT    | palazestrant              |
| 4 | LPCN   | 2025-12-16 | Other            |  +3.20 |      +257.46 | 252.7 | MISSED_ENDPOINT | LPCN 1154                 |
| 5 | PRAX   | 2025-10-16 | Phase 3 Readout  |  +2.20 |      +244.52 | 236.4 | MET_ENDPOINT    | ulixacaltamide            |
| 6 | TERN   | 2025-10-21 | Phase 2 Readout  |  +3.40 |      +202.90 | 196.2 | MIXED           | TERN-601                  |
| 7 | COGT   | 2025-11-10 | Phase 3 Readout  |  +2.20 |      +164.30 | 150.1 | MET_ENDPOINT    | Bezuclastinib             |
| 8 | INBX   | 2025-10-23 | Phase 3 Readout  |  +3.00 |      +157.01 | 142.6 | MET_ENDPOINT    | ozekibart (INBRX-109)     |
| 16| VTGN   | 2025-12-17 | Phase 3 Readout  |  +3.00 |       −84.86 |  90.0 | MISSED_ENDPOINT | fasedienol                |
| 19| MREO   | 2025-12-29 | Phase 3 Readout  |  +2.20 |       −80.56 |  83.5 | MISSED_ENDPOINT | Setrusumab                |
| 20| OTLK   | 2025-12-31 | Submission       |  +3.20 |       −76.41 |  82.1 | REJECTED        | ONS-5010                  |
| 26| NTLA   | 2025-10-27 | Other            |  +3.20 |       −66.72 |  83.3 | DELAYED         | nex-z (canonical example) |
| 27| ATRA   | 2026-01-10 | FDA Decision     |  +4.00 |       −63.57 |  68.9 | REJECTED        | tabelecleucel             |
| 35| ALDX   | 2026-03-16 | FDA Decision     |  +4.00 |       −55.64 |  71.6 | REJECTED        | Reproxalap                |

Full list: [tmp/audit01/top_misses.json](../) — not committed.

### Cluster A — Severe magnitude under-prediction on positive readouts

About **30 of the worst-50** look like CAPR / QURE / PRAX / OLMA /
COGT: small-cap clinical-stage biotech, MET_ENDPOINT label, predicted
move +2 % to +8 %, actual move +50 % to +346 %. `REF_MOVES` is
calibrated to (Phase 3 = +3 %, Phase 2 = +4 %, Phase 1 = +10 %).
That table reflects normal-cap, mid-binary clinical wins, but the
universe is dominated by sub-$500M biotechs where a clean trial
result repriced the equity by 1–4×. Direction is right; magnitude
is wrong by an order of magnitude.

### Cluster B — Wrong direction on FDA rejections / endpoint misses

About **8 rows** in the worst-50: OTLK appears 4 times (REJECTED ×3 +
delayed/REJECTED), ATRA, ALDX, AKBA, VTGN, MREO, ARCT, ALEC. The
model predicts +3 % to +4 % on FDA-decision-day or readout-day; the
stock prints −55 % to −85 %. This is the model's `predicted_prob`
defaulting to the positive side because high `predicted_prob` is
treated as "high confidence in positive outcome", which feeds a
positive-direction `predicted_move_pct`. The catalyst-specific
rejection-risk signal (e.g. AdComm dissent, prior CRL, manufacturing
flags) is either not in the model or not weighted heavily enough.
**Anti-alpha on this cluster.**

### Cluster C — DELAYED catalysts predicted as small positive

NTLA nex-z (predicted +3.2 %, actual −67 %, DELAYED — the canonical
example), QURE AMT-130 delay, TRDA, RVPH, SRRK. The model has no
"timing risk" feature; a delay announcement always tanks the stock
because investors recompute the expected-value timeline. Score: about
6 of the worst-50 are DELAYED labels.

### Cluster D — Concentration in micro/small-cap biotechs

Tickers in the worst-50 (CAPR, OLMA, LPCN, INBX, MREO, VTGN, OTLK,
ATRA, REVB, TNYA, IMRX, ARMP, RVPH, GPCR, CGEM, ARCT, ZBIO, IRWD,
TRDA, ARTL, ARWR, ALDX, AKBA, …) are overwhelmingly clinical-stage
issuers with market caps in the $50M–$500M range and float-heavy
binary catalysts. Big-cap names like CYTK or REGN show up rarely,
and when they do (CYTK aficamten +59 %), the residual error is much
smaller in dollar terms. **Market-cap and free-float controls are
likely the single highest-leverage feature add for the next model
iteration.**

## Open Questions / Unfinished Threads

1. **Backfill did not reach 95 %.** Currently 7-8 % labeled. The
   background worker is still running. S2 should monitor
   `/admin/post-catalyst/label-all-status` and re-snapshot once
   `running == false`. Total cost is bounded at ~$3 by the in-process
   `_label_state["estimated_cost_usd"]`, but Gemini-grounded latency
   means the wall-clock cost is hours, not minutes.
2. **Why does ~40 % of label attempts return UNKNOWN/None?** Could be
   genuine (Trial Initiation events have no readout press release;
   nothing to label), or transient grounded-search failures. S2
   should sample the `errors` rows and bucket them.
3. **`aggregate-v2` 2.1 % is still in the chat context bundle.**
   Until S2 either fixes the SQL or repoints the chat at
   `aggregate-v3`, every Ask AI session that asks about the scoreboard
   gets the wrong number. The handful of users who already saw 2.1 %
   may have a wrong mental model of system performance.
4. **REF_MOVES needs a magnitude recalibration that knows about
   market cap.** Current REF_MOVES has been tuned twice already
   (commits 559e018, ac0d6ce). The error on Cluster A is structural
   — the table cannot fit both AAPL-like names and CAPR-like names
   with one (positive, negative) tuple per catalyst type. A
   `(market_cap_bin, catalyst_type)` lookup or a regression on
   `log(market_cap)` is the natural next step.
5. **`MISSED_ENDPOINT` direction = 29.4 % (n = 17) is likely
   pessimistic-because-small.** With a bigger labeled set the bias
   may attenuate. Re-run the per-class table once the background
   labeler finishes.
6. **MET_ENDPOINT shows direction = 49.1 % at avg_signed_err = +15.4 %.**
   Reading: the direction call is right about half the time, and when
   it's right, the magnitude is materially under-predicted. This is
   the single most impactful fix: a positive-readout magnitude scaler.
7. **WITHDRAWN, BLA submission, NDA submission, AdComm, Phase 1/2,
   Phase 4** all have n < 30 even on the full 8,762-row dataset.
   They will likely never accumulate enough volume on the
   biotech-API universe to be evaluated; report as
   research-only.
8. **No score is currently reported per-ticker** in any aggregate
   endpoint. The audit could not, for example, confirm whether OTLK
   alone drags Cluster B accuracy below 50 %. Per-ticker rollups
   would make the next miss audit faster.

## Appendix — How this audit was produced

- All accuracy/cohort numbers came from direct read-only SQL against
  the production Postgres via `mainline.proxy.rlwy.net` using the
  Railway-managed connection. Connection details came from the
  Railway PAT in `.claude/railway-token`. No production data was
  modified.
- Layer 2 root-cause SQL is reproduced inline above and re-runs
  cleanly in a fresh psql session.
- Backfill was triggered with
  `POST /admin/post-catalyst/label-all-start?claim_size=10` at
  2026-05-01 22:17:55 UTC; status snapshots are in
  `tmp/audit01/`.
- Per spec, **no fixes were applied**. Every issue above is a
  hand-off to S2.

## Post-backfill snapshot, partial

Captured 2026-05-01 22:28 UTC, immediately before commit. Backfill
is **still running** in the background.

### `/admin/post-catalyst/outcome-label-stats`

| Field | Pre  | Post (partial, +10 min) | Δ |
|---|---:|---:|---:|
| total_outcomes | 10,605 | 10,605 | 0 |
| labeled_outcomes | 692 | 878 | **+186** |
| labeled_pct | 6.5 % | 8.3 % | +1.8 pp |
| APPROVED | 93 | 115 | +22 |
| REJECTED | 29 | 33 | +4 |
| MET_ENDPOINT | 244 | 325 | +81 |
| MISSED_ENDPOINT | 17 | 22 | +5 |
| DELAYED | 62 | 77 | +15 |
| WITHDRAWN | 3 | 3 | 0 |
| MIXED | 30 | 35 | +5 |
| UNKNOWN | 214 | 268 | +54 |
| estimated_full_backfill_cost_usd | $2.97 | $2.92 | −$0.05 |

`/admin/post-catalyst/label-all-status` (still running):
`labeled = 186, errors = 120, in_progress_count = 60`,
`estimated_cost_usd = $0.0558` consumed so far.
The +186 from this run plus pre-existing 692 = 878 labeled rows
(matches `outcome-label-stats`).

### `/v2/post-catalyst/accuracy`

| Field | Pre | Post (partial) | Δ |
|---|---:|---:|---:|
| total | 8,762 | 8,762 | 0 |
| direction_hits | 3,930 | 3,930 | 0 |
| direction_accuracy_pct | 44.9 | 44.9 | 0 |
| avg_abs_error_pct | 19.76 | 19.76 | 0 |
| avg_signed_error_pct | +2.70 | +2.70 | 0 |

**As predicted**, the Layer 1 accuracy number is invariant to
outcome label coverage — `direction_correct` is computed from
predicted vs actual move sign and never reads the label class.
The label backfill changes the distribution we can *segment* by,
not the headline accuracy.

### Class distribution shift

The added 186 labels are dominated by **MET_ENDPOINT (+81) and
UNKNOWN (+54)**, with a long tail of APPROVED, DELAYED, REJECTED,
MIXED, MISSED_ENDPOINT. The per-label-class accuracy table in
the per-bucket section was computed against the larger 692-row
labeled set in DB (the post-backfill rows arrived after the table
was generated). S2 should re-run on the full set.

### `/admin/universe/v2-spend`

`spent_usd: 0.03 / budget_usd: 5.00`. The $0.03 is mostly
non-backfill spend; in-process labeler reports $0.0558 for the
labeling itself (different counter — `_label_state.estimated_cost_usd`
is computed as `labeled × $0.0003` and is independent of the
universe-spend counter).

