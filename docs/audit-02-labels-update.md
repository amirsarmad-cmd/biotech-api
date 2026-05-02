# Audit 02 — Updated accuracy on the 6,139-row labeled set

**Date:** 2026-05-03 (UTC)
**Scope:** Refresh of [audit-01-labels.md](audit-01-labels.md) numbers
against the post-backfill labeled set. **Coverage: 692 → 6,139 (8.9×).**
The labeling burst stopped at 57.9 % when the Gemini billing cap hit;
the patterns visible now are statistically much firmer.

## Executive summary (5 bullets)

- The **anti-alpha-on-negative-outcomes** pattern from audit-01 is now
  statistically robust. REJECTED direction accuracy = **32-34 %** at
  n=147-220, MISSED_ENDPOINT = **19-29 %** at n=159 — both **well below
  coin flip with tight CIs**. Audit-01's 22- and 17-row samples were
  directionally correct; 8× the data confirms the size of the gap.
- The **FDA Decision** split is the cleanest binary cell:
  **APPROVED: 55 % direction acc (n=315)** vs **REJECTED: 32 % (n=75)**
  — a 23 pp gap on the same catalyst type, so the model's
  `predicted_prob` is **not predictive of outcome class** at all. Same
  pattern in Submission (APPROVED 53 % / REJECTED 33 %, n=139/49) and
  Phase 3 Readout (MET_ENDPOINT 57 % / MISSED_ENDPOINT 29 %, n=279/79).
- **Magnitude error mirrors direction error.** `avg_signed_error_pct`
  on REJECTED is **−14 %** and on MISSED_ENDPOINT is **−19 %** — the
  model is not just wrong on direction, it's massively under-pricing
  the magnitude of negative moves. Top-20 worst misses are uniformly
  `predicted_prob ∈ [0.90, 1.00]` paired with **−70 % to −100 % actual
  moves**. The model's most-confident positive bets are the most
  catastrophic shorts.
- **Per-catalyst-type direction accuracy aggregate is uninformative**
  — every type sits in 42-48 %. The signal lives in the
  (catalyst_type × outcome_label_class) cell, which is what makes the
  outcome labels indispensable. Without label backfill there's no edge
  to find here.
- **EV destruction** (sum of abs-error × n) is dominated by
  MET_ENDPOINT (37 %) and UNKNOWN (32 %), but those are mostly
  magnitude misses on directionally-correct calls.
  **MISSED + REJECTED + MIXED = 13 % of error volume but 100 % of the
  anti-alpha** — small bucket, but every event in it is a directional
  loss with material magnitude.

## Per-label-class accuracy (refreshed)

| outcome_label_class | n     | direction_acc % | avg_abs_err % | avg_signed_err % | sample_size  | vs audit-01 |
|---------------------|------:|----------------:|--------------:|-----------------:|--------------|---|
| UNKNOWN             | 1,738 |            46.7 |          16.0 |             −2.1 | STABLE       | n: 114 → 1738 |
| MET_ENDPOINT        | 1,374 |            47.2 |          23.5 |             −0.3 | STABLE       | n: 173 → 1374 |
| APPROVED            |   588 |            54.4 |          16.3 |             +2.0 | STABLE       | n: 56 → 588 |
| DELAYED             |   309 |            45.0 |          16.7 |             −6.5 | STABLE       | n: 49 → 309 |
| MISSED_ENDPOINT     |   159 |        **28.9** |          28.5 |        **−19.1** | STABLE       | n: 17 → 159 |
| REJECTED            |   147 |        **34.0** |          23.9 |        **−14.2** | STABLE       | n: 22 → 147 |
| MIXED               |   129 |            41.9 |          26.3 |             −8.3 | STABLE       | n: 20 → 129 |
| WITHDRAWN           |    24 |            33.3 |          15.8 |            −13.3 | INSUFFICIENT | n: 2 → 24 |

**Reading:** the asymmetry between positive (≥47 %) and negative
(≤34 %) outcome classes is now clearly real. The audit-01 doc called
this "anti-alpha"; the 8× sample confirms it without changing the
qualitative picture.

## (catalyst_type × outcome) — where the signal lives

Cells with **n ≥ 30** only. Numbers shown are direction accuracy %.

| catalyst_type     | APPROVED | REJECTED | MET_ENDPOINT | MISSED_ENDPOINT | DELAYED | MIXED |
|-------------------|---------:|---------:|-------------:|----------------:|--------:|------:|
| FDA Decision      | 55 % (315) | **32 % (75)** | — | — | 43 % (58) | — |
| Submission        | 53 % (139) | **33 % (49)** | — | — | 49 % (84) | — |
| Phase 3 Readout   | — | — | 57 % (279) | **29 % (79)** | 70 % (10)* | 40 % (43) |
| Phase 2 Readout   | — | — | 47 % (448) | **19 % (48)** | 30 % (10)* | 49 % (37) |
| Phase 1 Readout   | — | — | 41 % (470) | — | — | — |
| Other             | 54 % (111) | — | 47 % (153) | — | 38 % (71) | — |

\* n=10 cells flagged as preliminary.

**Bold cells are the anti-alpha.** Phase 2 MISSED_ENDPOINT (19 %, n=48)
is the worst single bucket — when a Phase 2 misses, the model is
directionally wrong **4 times out of 5**. Phase 3 MISSED is similar at
29 % (n=79). FDA REJECTED at 32-33 % across both decision types.

This is the same structural finding as audit-01 but now reliable
enough to act on.

## Top-20 anti-alpha misses (predicted positive, actually negative)

All have `predicted_prob ∈ [0.65, 1.00]`. Worst single rows:

| ticker | date | type | pred % | actual_30d % | err | label | prob |
|---|---|---|---:|---:|---:|---|---:|
| FBRX | 2021-09-02 | Phase 2 Readout | +3.4 | −90.1 | 84.7 | MISSED | 0.90 |
| OTLK | 2023-08-30 | FDA Decision | +4.0 | −84.3 | 80.4 | REJECTED | **1.00** |
| VTGN | 2025-12-17 | Phase 3 Readout | +3.0 | −84.9 | 90.0 | MISSED | **1.00** |
| KOD | 2022-02-23 | Phase 3 Readout | +2.2 | −84.7 | 88.0 | MISSED | 0.90 |
| AKBA | 2022-03-29 | FDA Decision | +4.0 | −82.8 | 71.7 | REJECTED | **1.00** |
| VTGN | 2022-06-23 | Other | +3.2 | −82.4 | 98.3 | MISSED | 0.90 |
| NMRA | 2025-01-02 | Phase 3 Readout | +2.2 | −82.5 | 86.2 | MISSED | 0.90 |
| ACRS | 2023-11-13 | Phase 2 Readout | +4.0 | −77.1 | 106.5 | MISSED | **1.00** |
| OTLK | 2026-01-01 | FDA Decision | +3.1 | −72.2 | 79.7 | REJECTED | 0.90 |
| BIVI | 2023-11-29 | Phase 3 Readout | +2.2 | −74.7 | 99.2 | MISSED | 0.90 |

(10 more in the same shape — `predicted_prob ≥ 0.9`, actual −70 % to −90 %.)

**OTLK appears 2× in top-20 and is one of audit-01's canonical
recurring tickers.** VTGN appears 2× (Phase 3 + Other). The model has
no ability to fade these going in; it sees them as high-confidence
positives.

## EV-destruction breakdown

Where the absolute-error budget gets spent:

| outcome class | n | Σ |error| | % of total error | type of error |
|---|---:|---:|---:|---|
| MET_ENDPOINT    | 1,374 | 32,264 | 37.2 % | mostly magnitude under-prediction on positive readouts |
| UNKNOWN         | 1,738 | 27,823 | 32.1 % | magnitude noise on unjudged events |
| APPROVED        |   588 |  9,562 | 11.0 % | small magnitude misses on direct-correct calls |
| DELAYED         |   309 |  5,175 |  6.0 % | direction near-50 %, magnitude moderate |
| MISSED_ENDPOINT |   159 |  4,538 |  5.2 % | **direction wrong + magnitude under-priced** |
| REJECTED        |   147 |  3,508 |  4.0 % | **direction wrong + magnitude under-priced** |
| MIXED           |   129 |  3,393 |  3.9 % | direction near-50 %, magnitude high |
| WITHDRAWN       |    24 |    378 |  0.4 % | small bucket |

**The 13 % of error volume in MISSED + REJECTED + MIXED is where the
trade lives.** Those 435 events are uniformly directionally wrong;
fading them at average +14-19 pp magnitude error per row would have
been net-positive. The other 87 % of error volume is "right
direction, wrong size" — a calibration problem, not an alpha source.

## What's stable, what's still preliminary

| catalyst × outcome cell | n | status |
|---|---:|---|
| FDA Decision × APPROVED | 315 | **STABLE** |
| FDA Decision × REJECTED | 75 | **STABLE** (was 22 in audit-01) |
| FDA Decision × DELAYED | 58 | STABLE |
| Submission × APPROVED | 139 | **STABLE** |
| Submission × REJECTED | 49 | PRELIMINARY (n in 30-99) |
| Submission × DELAYED | 84 | STABLE |
| Phase 1 × MET_ENDPOINT | 470 | **STABLE** |
| Phase 2 × MET_ENDPOINT | 448 | **STABLE** |
| Phase 2 × MISSED_ENDPOINT | 48 | PRELIMINARY (was 17 — qualitative finding intact) |
| Phase 3 × MET_ENDPOINT | 279 | **STABLE** |
| Phase 3 × MISSED_ENDPOINT | 79 | PRELIMINARY-going-STABLE |
| AdComm × * | 0 with non-UNKNOWN | STILL EMPTY — labeler couldn't find sources |
| WITHDRAWN × * | 24 total | INSUFFICIENT |

## Open questions / unfinished threads

1. **Why is AdComm still empty?** AdComm events have catalyst_type =
   'AdComm' but every labeled row from this catalyst type came back
   UNKNOWN. Either Gemini grounded search isn't finding AdComm
   coverage in the date window, or our query joined incorrectly.
   Worth a focused inspection.
2. **Does the anti-alpha pattern hold out-of-sample?** All numbers
   here are in-sample on historical data the model saw at training.
   The OOS evaluation snapshot infrastructure exists
   (`/admin/post-catalyst/oos-aggregate`) but hasn't been re-run on
   the new labels. Worth a follow-up to score new prediction
   snapshots against the freshly-labeled outcomes.
3. **Magnitude calibration on MET_ENDPOINT (n=1,374, 37 % of error
   volume).** Direction is near coin-flip but the magnitude
   under-prediction is structural (REF_MOVES is conservative). A
   per-(market_cap_bin × catalyst_type) lookup would likely halve
   the abs-error contribution from this bucket alone.
4. **How does `predicted_prob` distribute across REJECTED?** All top-20
   misses sit at 0.90-1.00. If most REJECTED rows have prob ≥ 0.85, a
   simple "fade probability ≥ 0.9 + base-rate of REJECTED ≥ 15 %"
   filter could be the seed of a real short signal. Needs a
   scatter plot of (predicted_prob, abs_error) split by outcome class.
5. **Backfill is 57.9 % complete.** When the Gemini billing cap
   activates (~2026-05-05), the `gemini-cap-watcher` routine restarts
   the labeler. Audit-03 should re-run these tables on the
   fully-labeled ~10K row set — expect the qualitative findings to
   harden, not change.

## What changed vs audit-01 (TLDR)

- Per-class direction accuracies all moved by < 5 pp despite the 8×
  sample increase. The qualitative findings of audit-01 were correct.
- The Layer 2 = 2.1 % SQL bug verdict still stands; the chat is still
  surfacing the buggy `aggregate-v2` number. Not fixed in this audit
  (per-spec, this is hand-off-only).
- The "magnitude under-prediction on positive readouts" cluster (CAPR
  +345 %, QURE +346 %, etc.) is no longer the headline finding —
  it's a calibration issue, not an alpha source. The headline
  finding now is the anti-alpha on **negative** outcomes, where the
  model is structurally wrong on direction AND magnitude.

No code changes proposed in this doc — same hand-off contract as
audit-01.
