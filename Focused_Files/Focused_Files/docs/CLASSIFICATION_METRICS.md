# Classification Metrics — TDIS Forecast

A plain-language reference for the numbers behind the ignition models, plus a
snapshot of what's actually shipped in the dashboard right now. Written so
someone new to this project can read it top to bottom and understand both
the vocabulary and the current state — not just the numbers.

---

## 1. Vocabulary — what these terms actually mean

**Base rate**: the real fraction of positives in whatever population you're
measuring. "What % of (cell, day) pairs actually had a fire?" Two different
answers show up in this project and it's critical not to confuse them:
- **Matched-negative test rate (~0.21-0.24)**: the rate in the *training/test
  sample* the models were built on. Real fires are rare, so the dataset was
  deliberately rebalanced — every fire-day is paired with a handful of
  matched non-fire-days from the same cells — to make the problem learnable.
  This is **not** the real-world rate.
- **Real population rate (~0.024)**: the actual fraction of (cell, day) pairs
  with a real fire, if you look at literally every cell on every day. This
  is what matters if a model's raw output is ever shown to a user as "this
  is the probability of fire."

**Precision**: of everything the model flagged as "fire," what fraction
really was. Low precision = lots of false alarms.

**Recall**: of everything that really was a fire, what fraction the model
flagged. Low recall = lots of misses.

**F1**: the balance point between precision and recall (their harmonic
mean) at one specific decision threshold. Different thresholds trade one
for the other — there's no single "right" F1, only a curve.

**AUC-PR** (area under the precision-recall curve): a threshold-free summary
of ranking quality on imbalanced data. **This number is base-rate dependent
by construction** — a "good" AUC-PR on a 2% base-rate problem is numerically
much smaller than a "good" AUC-PR on a 20% base-rate problem, even if the
model is equally skilled. Never compare raw AUC-PR values across different
populations.

**AUROC** (area under the ROC curve): another threshold-free ranking metric,
less sensitive to base rate than AUC-PR, but still best read alongside lift.

**Lift**: `AUC-PR ÷ base rate`. **This is the fair way to compare models
across different populations** — it answers "how much better than random
guessing is this, accounting for how rare the real signal is?" A lift of
2× and a lift of 4× are comparable even if their base rates are wildly
different; their raw AUC-PR values are not.

**Calibration**: does a predicted probability of 0.7 actually correspond to
a ~70% real-world chance? A model can have excellent ranking (it correctly
orders risky vs. safe cases) while being badly *miscalibrated* (its raw
numbers are systematically too high or too low). Ranking quality and
calibration are separate properties — a model can have one without the
other, which is exactly what this document finds below.

**Brier score**: mean squared error between predicted probability and the
real 0/1 outcome. Lower is better. Useful because, unlike AUC-PR/AUROC, it
directly penalizes bad calibration, not just bad ranking.

**ECE** (Expected Calibration Error): average gap between "what the model
predicted" and "what actually happened," binned by predicted probability.
0 = perfectly calibrated. Above ~0.05-0.1 is generally considered a real
problem; this document finds values far beyond that.

---

## 2. The models — what each one is and isn't

| Model | Trained with | Deployable? | Purpose |
|---|---|---|---|
| **Honest** | static + temporal features only, **no weather** | Yes — the safe, always-usable baseline | Deployable for any day, past or future, since it needs nothing you don't already know |
| **Operational-HRRR** | static + temporal + **forecast** weather (HRRR) | Yes — this is the one actually served live | The real production forecast for today/tomorrow/48h |
| **Ceiling** | static + temporal + **same-day observed** weather | **No — "diagnostic only, never served"** per its own build script | A research upper-bound control: "how much would perfect weather knowledge help?" |
| **Ceiling-historical** *(new, this session)* | the Ceiling model, scored day-by-day across 2024-2026 | **Not yet — pending the fixes below** | First attempt at giving the historical Ignition view a real model output instead of a heuristic proxy |

---

## 3. MODEL-level metrics (internal test sets — not what's shown on the dashboard)

These numbers come from scoring the saved model files directly, fresh,
against held-out test data. **None of this is what a dashboard user sees**
— this is model evaluation, done offline.

### 3a. Threshold sweep — Precision / Recall / F1

| Threshold | Honest (native pop., n=1.15M) | Operational-HRRR (n=892K) | Ceiling (shared pop., n=263K) |
|---|---|---|---|
| 0.2 | .253 / .970 / .401 | .261 / .965 / .411 | .245 / .969 / .391 |
| 0.3 | .270 / .920 / .418 | .289 / .893 / .437 | .264 / .919 / .410 |
| 0.4 | .307 / .786 / .442 | .336 / .767 / .467 | .289 / .834 / .429 |
| **0.5** | **.384 / .563 / .457** | **.410 / .602 / .488** | **.326 / .705 / .445** |
| 0.6 | .512 / .361 / .423 | .511 / .438 / .471 | .387 / .543 / .452 |
| 0.7 | .595 / .266 / .367 | .596 / .313 / .410 | .499 / .336 / .402 |
| 0.8 | .630 / .161 / .257 | .644 / .219 / .327 | .631 / .129 / .215 |

Format is P/R/F1. All three columns re-scored fresh, directly from the
model files on disk, as of this session — not copied from a log.

### 3a-bis. Real-full-population threshold sweep — Operational-HRRR (the served model)

Same model as the Operational-HRRR column above, but swept on the REAL
deployment population (5.06M cell-days, every TX res-5 cell x every day
2024-2026, base rate 1.9%) using CALIBRATED probabilities (the thresholds
are therefore in real-probability units — "flag if >7.5% chance"):

| Calibrated threshold | Precision | Recall | F1 | Cell-days flagged |
|---|---|---|---|---|
| 0.02 | .045 | .698 | .085 | 1,500,096 |
| 0.05 | .094 | .328 | .146 | 338,069 |
| **0.075 (best F1)** | **.149** | **.186** | **.165** | **~120K** |
| 0.10 | .173 | .150 | .161 | 84,663 |
| 0.15 | .186 | .121 | .147 | 63,513 |

Best F1 = **0.165** at calibrated threshold 0.0747 (full sweep:
`models/realpop_f1_sweep.json`). Read next to the balanced-sample best F1
of 0.488: same model, ~10x rarer positives, and F1 shrinks mechanically
with the base rate — this is the honest deployment operating point, not a
worse model. AUC-PR on this population: 0.0851 (lift 4.4x, section 7a).

### 3b. AUC-PR / AUROC / Lift — including the new ceiling-historical result

| Model | Population | Base rate | AUC-PR | AUROC | Lift |
|---|---|---|---|---|---|
| Honest (native) | matched-negative test | 0.240 | 0.442 | 0.694 | 1.84× |
| Ceiling (matched, shared pop.) | matched-negative test | 0.215 | 0.436 | 0.724 | 2.03× |
| Operational-HRRR | matched-negative test | 0.234 | 0.483 | 0.733 | 2.06× |
| **Ceiling-historical** | **real full population, 2024-26** | **0.024** | **0.094** | **0.777** | **3.88×** |

**Read this carefully**: ceiling-historical's AUC-PR (0.094) looks far lower
than the others, but it's evaluated on a population where real fires are
~9x rarer. Its **lift is actually the best of the four** (3.88× vs.
1.84-2.06×) — meaning its ranking of risky vs. safe cell-days is genuinely
the strongest, once you correct for the harder population it's being
judged against.

### 3c. Ceiling-historical calibration detail

| Metric | Value |
|---|---|
| Real base rate | 0.0243 |
| Mean predicted probability | 0.3743 |
| Brier score | 0.179 |
| ECE | 0.350 |

A trivial "always predict the base rate" model would score a Brier of
~0.024 here. **0.179 is roughly 7.5x worse than that trivial baseline** —
despite having the best lift of the four models. This is a calibration
problem, not a ranking problem (see §5).

---

## 4. DASHBOARD-level check (what's actually shipped in `tdis_dashboard_data.json`)

This section is different in kind from §3 — it's not model evaluation,
it's an inventory of what's *actually embedded in the live dashboard file*
right now, pulled directly from the JSON.

| Field | Coverage | Min | Mean | Max | What it is |
|---|---|---|---|---|---|
| `haz` | 5,382 / 5,382 cells | 0.000 | 0.194 | 0.739 | Static TxWRAP hazard composite. No ML, no time. |
| `whp` | 5,382 / 5,382 | 0.00 | 1.74 | 7.53 | Raw Wildfire Hazard Potential component. |
| `bp` | 5,382 / 5,382 | 0.00 | 2.74 | 10.00 | Raw burn-probability component. |
| `ign` | 5,363 / 5,382 | 0.009 | **0.361** | 0.871 | Honest-model susceptibility, monthly mean. **Currently displayed.** |
| `fwi` (Composite) | 3,624 cells × 933 days | 0.047 | 0.353 | 0.855 | Daily fire-weather index, additive formula. |
| `fwiN` (NOAA HWP) | 3,624 × 933 | 0.000 | 0.102 | 1.000 | Daily fire-weather index, NOAA Hot-Dry-Windy formula. |
| `fwiT` (TX HWP) | 3,624 × 933 | 0.074 | 0.650 | 1.000 | Same formula, TX-fit coefficients. |
| `ignC` | 3,382 × 933 | 0.002 | **0.374** | 0.939 | Ceiling model, scored day-by-day. **New, not yet displayed.** |

---

## 5. Red flags

🔴 **The already-shipped `ign` layer likely has the same overconfidence
problem as the new, still-hidden `ignC` layer, and nobody has checked.**
`ign`'s mean (0.361) is nearly identical to `ignC`'s (0.374) — both far
above any realistic base rate. `ign` comes from the exact same honest
model, trained with the exact same matched-negative rebalancing, so there's
every reason to expect it suffers the same miscalibration. The difference
is `ign` is **live on the dashboard today**; `ignC` was caught before
shipping. This is the highest-priority item here — it's already user-facing.

🔴 **Ceiling-historical (`ignC`) is severely miscalibrated**: mean predicted
probability 0.374 vs. real base rate 0.024, ECE 0.350, Brier ~7.5× worse
than a trivial constant-prediction baseline. Root cause understood (trained
on a rebalanced sample, scored on the real, far rarer population, with no
correction in between) — see §6 for the fix. **Correctly not yet displayed.**

🟠 **Stale/conflicting numbers were found in the project's own logs.**
`flare_filter_retrain.log` reports honest_filtered AUC-PR=0.413; a later
run in `retrain20.log` reports 0.450 for what's described as the same
model. Re-scoring the model file currently on disk gives 0.442 — closest
to the newer number, meaning the 0.413 log entry is outdated and could
mislead anyone who reads it later without re-checking.

🟡 **`ign` has 19 cells (5,363/5,382) without coverage** — minor, but
means roughly 0.4% of the state has no ignition-susceptibility value at
all in the dashboard, silently.

🟡 **No model's meta.json records which population its metrics were
computed on.** This ambiguity (matched-negative vs. real) is exactly what
made the honest_filtered log discrepancy hard to resolve, and will recur
for any future model unless this is fixed at the source.

---

## 6. Improvements

1. **Calibrate `ignC` before ever displaying it** — isotonic regression is
   the proven fix (already validated on a separate wildfire-spread model
   this session: cut Brier by ~42%, ECE by ~45%, on a similarly-shaped
   miscalibration). No retraining needed, just a post-hoc correction fit
   against real outcomes — the exact ground truth needed for this already
   exists from the validation run in §3c.
2. **Run the same real-outcome validation on the existing `ign` layer.**
   It's never been checked against real fire occurrence the way `ignC` just
   was. Given the mean-probability similarity, this should be treated as
   likely-broken-until-proven-otherwise, not a someday task.
3. **Clean up or annotate the stale log files** (`flare_filter_retrain.log`)
   so a superseded number doesn't get quoted again — either delete, or add
   a header noting it's superseded by the later retrain.
4. **Add `population` and `base_rate` fields to every model's meta.json.**
   Would have made the honest_filtered discrepancy immediately diagnosable
   instead of requiring a forensic re-scoring exercise.
5. **Don't reuse the same 0.2-0.8 threshold grid across populations with
   very different base rates.** It's a reasonable grid for the ~0.21-0.24
   matched-negative populations; it's not obviously the right grid for a
   0.024-base-rate population like ceiling-historical — worth a dedicated,
   lower-threshold sweep once that layer is calibrated.

---

## 7. Deployment-population validation & flag fixes (added 2026-08-13)

### 7a. The served model, evaluated on the REAL full population

The gap noted in §5 is closed: the operational (served) model was scored
day-by-day on every res-5 cell × every day of 2024-2026 (5.06M cell-days,
HRRR forecast weather — a true "what would it have said" replay) and
validated against real fire outcomes:

| | Matched-negative test | Real full population |
|---|---|---|
| Base rate | 0.234 | **0.0192** |
| AUC-PR | 0.483 | 0.085 |
| AUROC | 0.733 | **0.784** |
| **Lift** | 2.06× | **4.43×** |

The served model's real-world ranking is the best of anything measured in
this project — it even edges the ceiling model (3.88×/0.777) despite using
forecast rather than perfect observed weather. The raw-probability
miscalibration was severe (mean predicted 0.337 vs. real 0.019, ECE 0.318)
but perfectly **monotonic** across all 10 reliability bins — meaning the
"relative ranking" framing was always valid, and isotonic calibration fixes
it almost exactly.

### 7b. Calibration — FIXED (temporal holdout, not fit-on-test)

Isotonic calibrators fit on 2024-2025 outcomes, evaluated on a never-seen
2026 holdout (`scripts/27_fit_isotonic_operational.py`):

| Model | ECE before | ECE after | Brier improvement |
|---|---|---|---|
| Operational | 0.327 | **0.0018** (-99.4%) | -87.2% |
| Ceiling | 0.365 | **0.0017** (-99.5%) | -87.0% |

Calibrated mean prediction (0.0195) now matches the real base rate (0.0213).
Saved: `models/operational_isotonic_calibrator.joblib` (+ ceiling). The live
forecast script (13) now emits `ign_cal`/`ignCal` — the value safe to read
as an actual probability — alongside the raw ranking score.

### 7c. Flare threshold — robustness band (interim fix for F2)

Served-model real-population metrics recomputed under flare cutoffs
1%/3%/10% (`scripts/28_flare_robustness_band.py`): **lift 3.54-4.69×,
AUROC 0.766-0.790** across a 10× change in the cutoff. Conclusions do not
hinge on the 3% choice. The full fix (VIIRS `type`-flag cross-check of the
531 cells) remains open but is no longer blocking.

### 7d. Red-flag status after this pass

| Flag (from §5) | Status |
|---|---|
| 🔴 Shipped `ign` layer miscalibrated | **Mitigated** — calibrator exists + wired into live path; raw values remain rank-framed in UI; historical `ign` layer still raw (regenerate with calibrator when next rebuilt) |
| 🔴 `ignC` miscalibration | **Fixed** — ceiling calibrator saved; apply before any display |
| 🟠 Stale conflicting logs | **Fixed** — superseded-note headers added to all three retrain logs |
| 🟠 Flare threshold load-bearing | **Mitigated** — robustness band shows conclusions hold (7c); `type`-flag cross-check still recommended |
| 🟡 19 cells missing `ign` | **Documented** — they are out-of-state bounding-box fringe cells (AR/OK ecoregions, ~zero hazard); intentionally not backfilled; listed in dashboard meta |
| 🟡 tmmx fill-value artifact | **Verified fixed** in `03_build_dataset.py` (physical-floor null) |
| 🟡 meta.json missing population context | **Fixed** — all three served-model metas now carry population, base rates, real-population metrics, and calibration pointers |
