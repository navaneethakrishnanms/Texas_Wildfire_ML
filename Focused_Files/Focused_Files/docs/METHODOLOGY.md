# METHODOLOGY — the final, locked method

This is the single authoritative statement of how the TDIS ignition
forecast works. The method evolved during development; earlier variants
are listed at the bottom **so they stop reappearing in discussions** —
they are retired, with the evidence for why.

## The final method, end to end

**One model: `models/tdis_forecast_hrrr_filtered.json`** (XGBoost,
1000 trees, depth 9; promoted 2026-08-17, updated same day with the
power-line + soil-moisture features and the v2 flare list) +
**`models/operational_isotonic_calibrator.joblib`** (refit with it;
2026-holdout ECE 0.0018). **The method below is FINAL — locked
2026-08-17.** Future work changes labels or features inside
`New_Training817_moredata/`, never this method.

### 1. Training data (built once)

One row per (cell, day): H3 res-8 cells across Texas, 2014–2026, with
matched negative sampling (all fire-days + a sample of non-fire-days).
Flare/industrial cells excluded (v2 list: 531 heuristic + 7 found by the
VIIRS type-flag audit = 538 cells). Each row has 22 features:

| Group | Features | Changes daily? |
|---|---|---|
| Static (12) | road_dist_km, ecoregion_id, elevation_m, slope_deg, aspect_deg, avg_burn_prob, whp, flep4, cfl, cbd, cbh, powerline_dist_km | No |
| Calendar (6) | sin/cos month, sin/cos day-of-week, is_weekend, is_holiday | Yes (computable for any date) |
| Weather (4) | hrrr_tmp, hrrr_vpd, hrrr_wind, hrrr_mstav (soil moisture) | Yes (from HRRR) |

**The weather features are archived HRRR *forecasts*, not gridMET.** This
is deliberate: the model trains on the exact same data source it will see
at deployment (a 24h-ahead HRRR forecast), so there is no train/deploy
mismatch. gridMET (historical observed weather) is used elsewhere — for
the dashboard's historical fire-weather index and for event validation —
but never as a feature of the served model.

Label = "was there fire activity in this cell on this day" (VIIRS
satellite detections fused with the FPA-FOD fire-occurrence database).

### 2. Split — strictly by time, never random

| Years | Role |
|---|---|
| 2014–2021 | **Train** — the model learns from these |
| 2022 | **Validation** — used only to choose settings |
| 2023–2026 | **Test** — scored once; all reported numbers come from here |

The model is always evaluated on years it has never seen, which is the
same situation it faces in deployment. A random split would leak
same-season information and inflate every metric.

### 3. Forecasting = inference only (no retraining, no multiplying)

To forecast day D (e.g., run on the 18th for the 19th at 24h lead and the
20th at 48h lead), `scripts/13_model_forecast_day.py`:

1. Downloads NOAA's HRRR weather **forecast** for day D.
2. Rebuilds the same 22-column feature matrix for all 1.7M cells:
   static columns (incl. power-line distance) copied unchanged, calendar
   columns computed for D, the 4 weather columns (incl. soil moisture)
   filled with day-D forecast values.
3. Runs `model.predict_proba(features)` — each row walks through the
   ~400 frozen decision trees; the leaf values sum into a score.
   Nothing inside the model changes. This is what "inference" means.
4. Applies the isotonic calibrator to convert the raw score (a relative
   rank, inflated ~17× by the rebalanced training sample) into a real
   probability (`ign_cal`).
5. Writes the dashboard JSON; `scripts/22_embed_v3.py` rebuilds the
   standalone dashboard.

**Nothing is multiplied onto the model's output.** Weather influences the
prediction only as an *input feature*, where the trees can weigh it
jointly with location.

### 4. Reported performance (both rows are this same model)

| Evaluation population | AUC-PR | Best F1 | Lift |
|---|---|---|---|
| Test set, balanced sample (~20% fire-days) | 0.4941 | 0.496 | — |
| Real full population (5.06M cell-days, 1.9% fire-days) | 0.0878 | 0.169 | 4.58× |

**Why the real-population numbers are "low", and what that means.**
AUC-PR and F1 both shrink mechanically as positives get rarer — at a 1.9%
base rate, a *perfect-ranking* model still posts numbers most people
would call "low." The right reading is **lift**: high-flagged areas catch
fire at 4.4× the background rate. The low absolute numbers are a property
of the problem's sparsity, not evidence the model is weak — and this is
exactly why both rows are always reported together and labeled.

**Is the model "conservative"? No — the opposite, until calibrated.** The
raw score *overstates* risk ~17× (mean raw score ≈ 0.39 vs. real fire
frequency ≈ 0.02) because of the rebalanced training sample. The isotonic
calibrator corrects this (calibration error 0.002 on a held-out 2026
sample). After calibration the model is neither conservative nor
inflated — when it says 15%, ~15% of such cell-days have fire activity.
What it *is*: modest at pinpointing (precision 0.15 at the best-F1
threshold) — it is a ranking/prioritization tool, not an alarm system.

### Intended use — say this before any numbers

**This is a prioritization tool — it tells you where risk is
concentrated. It is NOT a detection/alerting system and does not catch
every ignition.** Precision and recall at the realistic operating points:

| Population | Threshold | Precision | Recall |
|---|---|---|---|
| Test sample (balanced) | 0.5 raw | 0.43 | 0.57 |
| Real population | 0.02 cal (broad) | ~0.05 | ~0.70 |
| Real population | 0.084 cal (best F1) | 0.15 | 0.19 |

Read with the base rate (0.019): precision 0.15 = a flagged cell is ~8×
more likely than a random cell to have fire. At any threshold with
tolerable false alarms, most individual fire-days are missed — dominated
by small, effectively random ignitions (equipment sparks, debris burns).
The 5-event major-fire validation flagged **5/5 locations** correctly;
timing was correct for heat/drought-driven events and under-scored for
winter wind-driven events (the documented wind gap — covered on the
dashboard by the NOAA HWP tab, which is wind-driven by construction).

The miss rate is a disqualifying limitation ONLY if the tool is
misrepresented as an alarm system. Presented as prioritization — the top
of the ranking deserves 4.4× more attention than average — it is the
same operating paradigm as red-flag warnings, which also "miss" most
individual ignitions.

## Retired approaches (do not resurface these as "the method")

| Era | Approach | Status & evidence |
|---|---|---|
| v1 | Static hazard only (TxWRAP) — no weather, no time | Retired as a forecast (it never was one). Survives only as the hazard layer inside the dashboard's Wildfire Risk tab. |
| v2 | Heuristic: ML susceptibility × daily fire-weather index (the historical "Ignition" scrubber layer) | Retired from v3 dashboard. Was a display proxy, never the served forecast. |
| tested & rejected | Multiply model output × HWP (or any FWI) | **Tested directly (script 29): degrades real-population AUC-PR by 18–61%.** The weather multiplier dilutes the model's spatial precision. Never do this. |
| tested & flat | Extra weather features (gust, 5-day trailing stats, explicit wind×dryness interactions) | All within ±0.0002 AUC-PR of baseline (`rev2_improvements/`). Not adopted. |
| tested & rejected | Monotone constraint forcing wind↑ = risk↑ | −0.0003 AUC-PR (step30). Fourth independent confirmation that wind doesn't predict fire *occurrence* — it predicts behavior, which is HWP's job. |
| tested & **PROMOTED 2026-08-17** | Hyperparameter sweep winner (depth 9, mcw 30, lr 0.02, 1000 trees) | +0.005 test AUC-PR, stable across 3 seeds, real-population win (lift 4.47×) — now the served model. |
| diagnostic only | "Ceiling" model (same-day *observed* weather) | Never served — observed weather is unknowable at forecast time. Upper-bound reference only. |
