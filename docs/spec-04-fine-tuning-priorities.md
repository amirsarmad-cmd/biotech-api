# Spec-04 — Fine-tuning priorities (the slop-inventory action plan)

**Date:** 2026-05-03
**Origin:** User flagged that `services/post_catalyst_tracker.py` clamps
`scenario_upside` at 80% — a hardcoded number masquerading as analysis.
This spec inventories every place we have similar "generic slop" (hardcoded
multipliers, arbitrary thresholds, rule-based heuristics that should be
data-fit), classifies each by severity + cost-to-fix, and lays out the
ordered work plan.

The Phase 1 deliverable shipped alongside this doc is
[`/admin/post-catalyst/scenario-algo-backtest`](../routes/post_catalyst.py)
plus [`services/scenario_algo.py`](../services/scenario_algo.py): a backtest
harness for the **proposed** V2 scenario-range formula. The new formula is
NOT yet wired to prod — the gate is "does it beat baseline on labeled
history?" Run the backtest before flipping the switch.

---

## Decision framework

For every magic number in the codebase, classify into one of three buckets:

- **(a) DEFENSIBLE** — domain-derived constant or regulatory category that
  doesn't need tuning (e.g. FDA disclosure regimes, milestone names,
  industry-standard cap brackets). Leave it alone.
- **(b) CALIBRATABLE** — a single multiplier or threshold that should be fit
  from labeled data via simple regression. Cheap to refine: just need the
  data and a least-squares solve.
- **(c) REPLACE-WITH-ML** — a multi-factor heuristic where a logistic /
  GBM model would do better. Higher engineering cost; only do this when
  rule-based hits the obvious ceiling.

The order of operations is always: **(b) before (c)**. Cheap calibration first;
ML only when the calibrated rule-based version still doesn't clear the
accuracy gate. The user's stated worry — "I'm worried we're just documenting
slop" — is the reason every step has an empirical kill-line.

---

## Slop inventory (worst offenders first)

Source: codebase audit run 2026-05-03 (`Explore` subagent, full report stored
with this commit). Severity = `impact_on_decision × applicability_frequency`.

### Tier-1 (high slop, high impact)

#### 1. `services/setup_quality.py` — 6 axes × ~15 magic numbers, no calibration
**What's wrong:** Six independent piecewise-linear scoring axes (runup,
52w-position, short interest, IV-euphoria, sentiment, insider activity), each
with 2-5 hardcoded knots (40 / 85 percentiles, SI bins at 5/10/20/30%,
IV thresholds at 15-40%, sentiment ramp at 0.7). Combined via simple mean.
None of the knots are backtested.

**Why high-impact:** Drives the "Setup quality" score in the cockpit and
the "do I take this trade" gate. A wrong threshold here means rejecting good
setups and taking crowded ones.

**Fix path:**
1. (b) Calibrate each axis independently: regress `direction_correct` on each
   axis's raw input. Pick knots at the inflection points where accuracy
   crosses 50%. ~0.5 day per axis.
2. (c) If per-axis calibration plateaus, replace the whole panel with a
   single **GBM classifier** trained on `(runup, 52w_pos, SI, IV, sentiment,
   insider) → direction_correct`. ~2-3 days incl. holdout validation.

**Expected lift:** ~3-5 percentage points absolute direction accuracy
(per audit estimate).

#### 2. `services/post_catalyst_tracker.py:_compute_move_estimates` — scenario clamps + multipliers
**What's wrong:** `scenario_upside = min(80, max(up, fi * 0.4))` with
`scenario_downside = max(-80, min(down, -(fi * 0.3)))`. Three magic numbers
(0.4 / 0.3 / 80) all chosen without empirical justification.

**Why high-impact:** Drives the "if approved / if rejected" scenario range
in the cockpit, which is what the user sizes positions against. The 80%
clamp masks the real fundamental impact for micro-caps with large drug NPVs.

**Fix path (THIS SPEC):**
1. **Phase 1 (DONE)** — proposed V2 algo (`services/scenario_algo.py`)
   replaces clamps + multipliers with a per-stock formula grounded in
   `drug_npv × (1 - priced_in_fraction) × capture_rate / EV`. Backtest via
   `GET /admin/post-catalyst/scenario-algo-backtest` before flipping prod.
2. **Phase 2** — refit `CAPTURE_RATE_BY_TYPE` per catalyst_type from
   `actual_move_pct_7d / full_repricing_pct` regression on labeled data.
   Replaces the current literature-default values with empirically-fit ones.
   ~0.5 day after the label backfill resumes.
3. **Phase 3 (optional, only if Phase 2 plateaus)** — gradient-boosted
   regressor with `(catalyst_type × cap_bucket × product_class × time_to_peak
   × p_approval × priced_in)` features. ~1-2 days.

**Decision gate:** the backtest's `decision_gate_passed=True` (overall
direction-accuracy ≥ 60% AND MAE ≤ avg actual magnitude) is the ship/no-ship
test. Wire to prod when it passes.

#### 3. `services/catalyst_signal.py` — piecewise-linear priced-in mapping
**What's wrong:** Three separate linear ramps (r ≤ -20, -20 < r < 0, 0 ≤ r <
30, r ≥ 30) with hardcoded breakpoints. The `runup → priced_in` formula
uses √0.5 magic constants, `min_scenario_pct=4.0`, `min_confidence=0.55`,
`options_ratio_floor=0.35`. Some have backtest rationale documented; most
don't.

**Why high-impact:** Drives V2 trade classifier (`LONG_UNDERPRICED_POSITIVE`
vs. `NO_TRADE_PRICED_IN` etc.) — every trade decision routes through this.

**Fix path:**
1. (b) Replace the three linear ramps with a fitted **monotonic spline**
   on `(runup, options_iv, iv_percentile) → priced_in`. ~1 day.
2. Verify against the same labeled set used for setup_quality — same gate
   (≥ 60% direction-accuracy on tradeable subset).

### Tier-2 (medium slop, medium impact)

#### 4. `services/priced_in_calculator.py:DEFAULT_SOLO_DRUG_PIPELINE_FRACTION = 0.60`
**What's wrong:** When sibling-drug NPVs are missing (the common case —
most tickers have only one drug cached), assumes the catalyst drug carries
60% of enterprise value. Used in ~80% of `priced_in_fraction` calls.
**Fix:** Fit per `(sector, stage, pipeline_depth)` from where we DO have
sibling data (large-pharma cache rows). ~0.5 day. **Tier-2 not tier-1
because** it has a documented "log for refinement" pattern and a clear
upgrade path; the 60% value is at least documented as fallback.

#### 5. `services/priced_in_calculator.py` — options-implied heuristic ramp
**What's wrong:** `5%→0.85, 15%→0.30, 30%→0.10` 3-point linear. Conflates
IV level with priced-in fraction (low IV ≠ already-priced — could equally
mean low interest).
**Fix:** Joint model of `(IV, runup, days_to_catalyst, catalyst_type) →
priced_in`. ~1 day. **Category (c) — needs ML**, can't be fixed by tuning
the 3 points.

#### 6. `services/probability_lookup.py` — options stub returns fixed 0.5
**What's wrong:** "v1 stub: simplified ATM-straddle inversion ... returns
0.5 when only magnitude is known (no direction signal)." The directional
signal is **entirely discarded**.
**Fix:** Skew-based logit on `(call_IV / put_IV, spot, strike) → p_approval`.
~0.5 day. Possibly the highest ROI per hour because we're literally throwing
information away today.

#### 7. `services/npv_model.py` — cap-size revenue multiples + sentiment dampeners
**What's wrong:** Multiples (4.0/3.5/3.0 by cap size), sentiment dampener
(10%), high-short amplifier (30%), rejection overshoot (12%) — all asserted
as empirical, none cited from a backtest.
**Fix:** `(b)` Per-cap-bucket regression on traded comps. ~0.25 day each.

### Tier-3 (low slop or low impact)

- `services/move_lookup.py` — DEFENSIBLE. Cap buckets are SEC convention,
  fallback chain is transparent statistical practice.
- `services/post_catalyst_tracker.py:REF_MOVES` — DEFENSIBLE. Calibrated
  from N=287 historical events (commit `3a03cb0`), updated April 2026.
- `services/drug_programs.py` — DEFENSIBLE. Milestone POS values
  (10/25/55/85/95) are FDA/Tufts published reference rates. Room for
  per-indication stratification but not slop.
- `services/disclosure_regime.py` — DEFENSIBLE. Regulatory categories.
- `services/prediction_disagreement.py:DISAGREEMENT_PP_THRESHOLD = 50` —
  CONFIGURABLE via env, low-impact, leave alone.

---

## Sequenced action plan

### Stage 0 — Block (NOW)
- **Wait for Gemini cap reset (~2026-05-05)** so the label backfill can
  resume from 57.9% → 100%. Per `feedback_llm_keys_exist.md` we don't
  have throughput now.
- The cap-watcher routine (`trig_01QSxhcnYLgu5ZhBkzDuqtKQ`) auto-fires the
  resume sequence once Gemini returns 200. No manual action needed.

### Stage 1 — Validate the V2 scenario algo (CAN DO NOW with current data)
1. Call `GET /admin/post-catalyst/scenario-algo-backtest?only_labeled=true`.
2. Read `totals.decision_gate_passed` + per-bucket direction-accuracy.
3. If gate passes → wire `compute_scenario_range_v2()` into
   `post_catalyst_tracker.py:_compute_move_estimates` (replacing lines
   254-263). Ship as `prediction_source=npv_v2`.
4. If gate fails → Stage 2.

### Stage 2 — Refit capture rates per catalyst_type (Phase 2)
1. From the same backtest dataset, fit
   `actual_move_pct_7d = β × full_repricing_pct` per catalyst_type.
   `β` IS the empirical capture rate (replaces the literature defaults in
   `services/scenario_algo.py:CAPTURE_RATE_BY_TYPE`).
2. Re-run backtest. If gate now passes, wire to prod.

### Stage 3 — Tier-1 slop fixes (in order)
3. **`services/probability_lookup.py` options-stub fix** (Tier-2 #6 but
   highest-ROI-per-hour) — ship the skew-based directional probability
   solver. ~0.5 day.
4. **`services/setup_quality.py` per-axis calibration** (Tier-1 #1).
   Fit each of the 6 axes' knots from labeled data. ~3 days total.
5. If per-axis calibration doesn't clear the accuracy gate (~+3pp lift),
   replace setup_quality with a GBM classifier. ~2-3 days.

### Stage 4 — Tier-1 #3 (catalyst_signal.py priced-in spline)
6. Replace the three linear ramps with a monotonic spline. ~1 day.

### Stage 5 — Tier-2 cleanup
7. Fit `DEFAULT_SOLO_DRUG_PIPELINE_FRACTION` per (sector, stage). ~0.5 day.
8. Joint model for options-implied priced-in. ~1 day.
9. Per-cap-bucket revenue multiple regression for `npv_model.py`. ~1 day.

### Stage 6 — Production cutover + monitoring
10. Once all Tier-1 and Tier-2 swaps are live, add a daily
    `prediction-vs-actual` calibration report panel (extend the existing
    `PostCatalystHistoryPanel` calibration view).
11. If the per-bucket direction-accuracy drifts below 55% for 30 days,
    auto-flag the bucket for retraining.

---

## Dependencies / blockers

| Stage | Blocked on | Earliest start |
|------|------|------|
| 1 | Nothing (current data + new endpoint) | NOW |
| 2 | Stage 1 fails OR Gemini cap resets (more data) | NOW or 2026-05-05 |
| 3 step 3 | Nothing | NOW |
| 3 step 4-5 | Full label backfill complete | ~2026-05-08 (3 days post Gemini reset) |
| 4-5 | Stage 3 results | After Stage 3 |
| 6 | All prior stages | ~2026-05-15 |

---

## What this spec is NOT

- A defense of the current 80% clamp. The clamp is bad. The point of
  Phase 1 backtesting is to **prove** the V2 formula is better before
  shipping it, not to gate on it forever.
- A request to re-architect the LLM gateway, the labeler, or the rNPV math.
  Those are working as designed; this is purely about replacing
  rule-of-thumb magic numbers with data-fit values.
- An ML-for-ML's-sake plan. Every Tier-1 / Tier-2 entry has a (b)
  calibration step BEFORE the (c) ML step. We only escalate to ML if
  calibration plateaus.

---

## Appendix — full slop inventory

The complete audit table (severity ranking, fix complexity per file/line)
lives in the audit subagent's output and is summarized in
`memory/sessions/2026-05-03T*-decision-cockpit-improvements.md`. To regenerate,
re-run the `Audit biotech-api for slop` Explore agent.
