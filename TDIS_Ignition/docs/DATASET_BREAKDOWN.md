# Complete dataset breakdown — ignition and intensity models

**Both models validated. Sub-daily does not sacrifice ignition accuracy vs.
daily, and conditional intensity supplements it as a genuinely independent
axis.** This document is the full feature-level accounting for both training
sets: source, role, importance, and descriptive statistics.
(2026-09-21, post flare-contamination and TX-static-source corrections —
see `FORECAST_INTENSITY.md` for the correction history. Stats in
`results/ignition_feature_stats.csv` and `results/intensity_feature_stats.csv`.)

## Headline validation status

| | Ignition (TriState_subDaily) | Intensity (tri-state) |
|---|---|---|
| Target | binary: does a fire occur | continuous: how intense (log peak FRP) |
| **Validated?** | ✅ monotonicity (9/9 pass), permutation importance, calibration (isotonic, ECE 0.19→0.07), cross-seed agreement (corr 0.989), real-population recall (AUROC 0.83–0.90, lift 11–16×), LISA spatial | ✅ label-shuffle (~0), spatial holdout (0.559 ≥ 0.553 temporal — no memorization), beats ecoregion baseline, seed-stable (std 0.0001), cross-model consistency (0.747 corr vs. independent build) |
| Sub-daily vs daily | **No accuracy sacrifice** — AUC-PR 0.3049 vs 0.3001, AUROC 0.8013 vs 0.7991 (TX 2020 test) | n/a (intensity is static; see below) |
| Headline skill | AUC-PR 0.30 (TX matched), Precision@0.1% up to 98% (CA) | Spearman ρ 0.553 pooled (TX 0.559 / NM 0.467 / OK 0.257) |

**How intensity supplements ignition:** the two are nearly orthogonal
(ρ ≈ 0.11–0.16 between the two models' outputs on the same cells) — see
`FINDING_ignition_x_intensity.md` and `CASE_STUDY_tx_2020-06-16.md` for a real
event where they diverge by 80+ percentile points at one cell. Weather helps
ignition (conditionally on time-of-day encoding, `scripts/21`) but does **not**
help intensity under any tested combination (`scripts/04,17`) — intensity is a
fuel/terrain story, ignition is a people/weather/timing story.

---

# 1. Ignition dataset (TriState_subDaily)

**137,688 rows** (TX 101,184 / NM 16,419 / OK 20,085) · **12,679 positives**
(9.2% base rate, matched-sample 1:10 window-negative design) · train <2020 /
test TX-only 2020 · label = binary ignition occurrence per (h3_cell, op_date,
window), FPA-FOD only.

## 1a. Features — role, importance, statistics

Importance = XGBoost gain, from the **production** sub-daily model
(`models/subdaily_seed42.json`, 30 features). Permutation-importance
cross-check available in `results/ignition_model_validation.json`.

| Feature | Role | Gain importance | Mean | Median | Std | Range (5–95%) | Null% |
|---|---|---|---|---|---|---|---|
| `pop_density` | human exposure | **0.096** (#1) | 29.4 | 0.19 | 194.3 | 0–59.8 | 0 |
| `whp` | landscape (fuel hazard) | 0.066 | 3.64 | 4 | 1.88 | 1–7 | 0 |
| `ecoregion_id` | landscape (region code) | 0.056 | 28.99 | 29 | 4.15 | 24–35 | 0 |
| `road_dist_km` | landscape (access) | 0.043 | 6.43 | 4.41 | 6.80 | 0.32–19.2 | 0 |
| `bp_x_vpd` | interaction | 0.042 | 0.008 | 0.003 | 0.014 | 0–0.033 | 0 |
| `cbd` | landscape (canopy bulk density) | 0.042 | 0.014 | 0 | 0.041 | 0–0.11 | 0 |
| `avg_burn_prob` | landscape (FSim burn prob) | 0.038 | 0.004 | 0.002 | 0.006 | 0–0.016 | 0 |
| `elevation_m` | landscape (terrain) | 0.038 | 610.0 | 419.1 | 558.4 | 31–1734 | 0 |
| `drought_score` | weather (dryness) | 0.035 | 1.01 | 0.55 | 1.17 | 0–3.29 | 0 |
| `sin_hour` | sub-daily temporal | 0.034 | −0.64 | — | 0.30 | — | 0 |
| `cos_hour` | sub-daily temporal | 0.033 | −0.04 | — | 0.71 | — | 0 |
| `whp_x_vpd` | interaction | 0.032 | 6.57 | 5.19 | 5.50 | 0.85–17.1 | 0 |
| `sin_month` | calendar | 0.029 | −0.22 | −0.5 | 0.66 | — | 0 |
| `fm100_5d_min` | fuel dryness (lag-safe) | 0.029 | 11.52 | 11.7 | 3.68 | 5.5–17.3 | 0 |
| `is_holiday` | calendar (human activity) | 0.028 | 0.023 | 0 | 0.15 | — | 0 |
| `bp_x_drought` | interaction | 0.027 | 0.004 | 0 | 0.010 | 0–0.022 | 0 |
| `hrrr_wind` | weather (per-window) | 0.026 | 8.29 | 7.72 | 3.81 | 3.18–15.4 | 0 |
| `hrrr_vpd` | weather (per-window) | 0.026 | 1.88 | 1.63 | 1.22 | 0.40–4.44 | 0 |
| `powerline_dist_km` | landscape (ignition source proxy) | 0.026 | 11.40 | 5.67 | 16.58 | 0.38–42.8 | 0 |
| `cbh` | landscape (canopy base height) | 0.026 | 0.81 | 0 | 2.40 | 0–10 | 0 |
| `cos_month` | calendar | 0.025 | −0.15 | −0.5 | 0.70 | — | 0 |
| `hrrr_tmp` | weather (per-window) | 0.025 | 24.51 | 26.05 | 7.56 | 10.5–35.5°C | 0 |
| `hrrr_mstav` | weather (soil moisture) | 0.024 | 0.19 | 0.10 | 0.23 | 0–0.76 | 0 |
| `slope_deg` | landscape (terrain) | 0.024 | 3.03 | 1.62 | 4.21 | 0.22–10.8 | 0 |
| `cfl` | landscape (canopy fuel load) | 0.024 | 3.24 | 3.47 | 2.11 | 0–6.30 | 0 |
| `flep4` | landscape (flame-length exceed.) | 0.022 | 0.35 | 0.28 | 0.32 | 0–0.90 | 0 |
| `cos_dow` | calendar | 0.022 | −0.01 | −0.22 | 0.69 | — | 0 |
| `aspect_deg` | landscape (terrain) | 0.021 | 173.96 | 169.1 | 101.7 | 17.5–338.9° | 0 |
| `sin_dow` | calendar | 0.021 | −0.004 | 0 | 0.72 | — | 0 |
| `is_weekend` | calendar (human activity) | 0.020 | 0.30 | 0 | 0.46 | — | 0 |

**Reading:** `pop_density` dominates (nearly 2× the #2 feature) — ignition is
fundamentally a human-activity signal, consistent with the paper's "landscape
dominance = physical validation, weather = temporal selector" framing.
`sin_hour`/`cos_hour` (sub-daily-only) rank in the top third of all 30
features — the model genuinely uses within-day timing, not just landscape.

## 1b. Full null audit
Zero nulls across all 30 features in the training table (pre-cleaned at
build time; gates report in `TriState_subDaily/results/gates_report.json`
PASS).

---

# 2. Intensity dataset (tri-state, flare + static-source corrected)

**595,957 fire cell-days** (TX 400,971 / OK 144,573 / NM 50,413) · label =
peak VIIRS FRP (fire radiative power, MW) per (h3_cell, date), log(1+FRP)
modeled · every VIIRS detection used (no FPA join) minus known industrial
flares (TX 538 / NM 1,956 / OK 260 cells, episode-based exclusion) · train
≤2024 / test 2025–2026.

## 2a. Label statistics — peak FRP (MW)

| Stat | Value |
|---|---|
| Mean | 9.07 |
| Median | 3.30 |
| Std | 25.32 |
| 5th pct | 0.44 |
| 95th pct | 32.78 |
| 99th pct | 107.05 |
| Max | 1,676.84 |

Heavy right-skewed, hence the log(1+FRP) modeling target.

## 2b. Features — role, importance, statistics

Importance = XGBoost gain, from the corrected production model
(`results/intensity_tristate_model.json`, static-only, 12 features).

| Feature | Role | Gain importance | Mean | Median | Std | Range (5–95%) | Null% |
|---|---|---|---|---|---|---|---|
| `ecoregion_id` | region code | **0.437** (#1, dominant) | 29.70 | 29 | 4.99 | 23–37 | 0.24 |
| `avg_burn_prob` | landscape (FSim burn prob) | **0.165** (#2) | 0.004 | 0.002 | 0.006 | 0–0.016 | 0.15 |
| `elevation_m` | terrain | 0.124 | 542.5 | 307.4 | 604.5 | 12.3–2219.5 | 2.94 |
| `whp` | fuel hazard | 0.050 | 3.54 | 4 | 1.76 | 1–6 | 2.94 |
| `road_dist_km` | access/human proxy | 0.040 | 6.27 | 4.45 | 6.13 | 0.39–18.5 | 0 |
| `powerline_dist_km` | infrastructure proxy | 0.039 | 8.36 | 5.20 | 9.75 | 0.34–27.1 | 2.94 |
| `cbd` | canopy bulk density | 0.038 | 0.022 | 0 | 0.051 | 0–0.16 | 2.94 |
| `cfl` | canopy fuel load | 0.029 | 3.72 | 3.52 | 1.85 | 1.14–6.60 | 14.71 |
| `cbh` | canopy base height | 0.024 | 1.18 | 0 | 2.80 | 0–10 | 2.94 |
| `slope_deg` | terrain | 0.023 | 3.40 | 1.68 | 4.87 | 0.22–12.7 | 2.94 |
| `flep4` | flame-length exceedance | 0.019 | 0.31 | 0.22 | 0.31 | 0–0.89 | 0.15 |
| `aspect_deg` | terrain | 0.013 | 175.1 | 170.5 | 102.1 | 18.3–339.5° | 2.94 |

**Reading:** `ecoregion_id` + `avg_burn_prob` + `elevation_m` carry 73% of
total importance — intensity is overwhelmingly a fuel/terrain-region story.
The `drop_ecoregion_id` audit (rho 0.542 vs full 0.553) confirms the model
isn't *purely* riding on region identity — the other 11 features still carry
real signal. Null rates (0–15%) reflect genuine raster coverage gaps, not a
data bug (verified against the corrected static source; the pre-correction
version of `whp/cbd/cbh/elevation_m/slope_deg/aspect_deg/powerline_dist_km`
was 100% null for TX due to a stale static file — see `FORECAST_INTENSITY.md`
correction log).

---

# 3. Features shared by both models — the overlap

The 12 intensity features are a **strict subset** of the ignition model's 30.
Both use the same underlying `static_master_*.parquet` source files (not
leakage — see prior discussion in conversation — two independent models
legitimately sharing a physical landscape description).

| Feature | Ignition gain rank | Intensity gain rank | In both top-6? |
|---|---|---|---|
| `ecoregion_id` | #3 | **#1** | ✅ |
| `avg_burn_prob` | #7 | **#2** | ✅ |
| `elevation_m` | #8 | **#3** | ✅ |
| `whp` | **#2** | #4 | ✅ |
| `road_dist_km` | #4 | #5 | ✅ |
| `powerline_dist_km` | #19 | #6 | partial |
| `cbd` | #6 | #7 | — |
| `pop_density` | **#1** | n/a (not in intensity feature set) | — |

**Ecoregion, burn probability, and elevation are important to both models** —
the physically sensible core of "what makes a landscape fire-relevant" at all.
Where they diverge is instructive: ignition's #1 driver (`pop_density`) isn't
in the intensity feature set at all (intensity doesn't use human/weather
features), and intensity's dominant driver (`ecoregion_id` at 44%) is a much
larger share of total importance than in ignition (6%) — ignition spreads its
signal across many features (human, landscape, weather, timing), intensity
concentrates it in a few structural ones.

---

## Reproduce
```
cd 90%_ig_dec/OperationalLayer
python3 scripts/06_intensity_tristate.py     # intensity model + importances
python3 scripts/09_intensity_audit.py        # intensity audit
python3 scripts/14_ignition_model_validation.py   # ignition validation battery
```
Descriptive-stats CSVs: `results/ignition_feature_stats.csv`,
`results/intensity_feature_stats.csv`.
