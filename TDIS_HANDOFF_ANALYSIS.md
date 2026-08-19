# TDIS Forecast Handoff — Full Technical Analysis & Cross-Comparison

**Prepared by:** IgnitionNet team (navaneethakrishnanms / projectsbit26@gmail.com)
**Date:** 2026-08-19
**Subject:** Review of `Focused_Files.zip` + `Supporting_files.zip` shared by the Texas / TDIS team, cross-compared against our own V1 / V2 / TX pipelines
**Method:** every code file, doc, JSON, and parquet in both shared folders was opened and read; the shipped model was re-loaded and independently re-scored against the shipped data

---

## 0. Executive Summary — read this first

| Question | Answer |
|---|---|
| Are their reported numbers real? | **Yes.** I reproduced their headline test AUC-PR **0.4941**, AUROC **0.7402**, best-F1 **0.4958** exactly, from their own files. |
| Is their model better than ours? | **Not comparable head-to-head.** Different labels, different time window, different negative ratio, different weather source. On *lift within the balanced sample* we are ahead (4.67× vs 2.11×). On *deployment-realistic evidence* they are far ahead — because they measured it and we have not. |
| Is `Focused_Files` self-sufficient? | **Almost — 3 concrete gaps.** A missing config file silently corrupts the dashboard's fire-weather normalization; the model metadata is stale (says 20 features, model has 22); and the calibration script is absent, so their own "always refit the calibrator" rule cannot be followed from that folder. |
| Do we need `Supporting_files`? | **Yes, and we now have it.** It fills all three gaps and contains the entire negative-results archive, which is the most valuable single asset in the handoff. |
| What do they want from us? | (1) Validate and check this work — done, this document. (2) Then build the end-to-end pipeline. (3) Four specific technical questions they addressed to us **by name** in `LIVE_PIPELINE_PLAN.md` §4. |

**The single most important finding:** their `LIVE_PIPELINE_PLAN.md` §4 is titled *"IgnitionNet collaboration — open threads worth raising with them."* That is us. They have read our repo, adopted our depth-9 hyperparameter finding, and listed four specific items they want answered. This handoff is a peer review request, not a status update.

---

# PART A — What the TDIS team built

## A.1 The system in one paragraph

A daily wildfire **ignition** forecast for Texas. One XGBoost binary classifier predicts, for each H3 resolution-8 hexagon on each day, the probability of fire activity. It is trained on **archived HRRR weather forecasts** — deliberately the same data source it will see at deployment — plus static terrain/fuel/access features and calendar features. Every day, a script pulls the live NOAA HRRR forecast for tomorrow (24 h) and the day after (48 h), runs inference on all 1.7 M cells, applies an isotonic probability calibrator, aggregates to res-5, and rebuilds a self-contained HTML dashboard. No retraining happens in the daily loop.

## A.2 The served model

**File:** `models/tdis_forecast_hrrr_filtered.json`
**Paired with:** `models/operational_isotonic_calibrator.joblib` — *these two ship as a pair, always*

| Property | Value |
|---|---|
| Algorithm | XGBoost `XGBClassifier`, binary:logistic |
| Trees | 1,000 |
| max_depth | 9 |
| min_child_weight | 30 |
| learning_rate | 0.02 |
| subsample / colsample_bytree | 0.8 / 0.8 |
| eval_metric | aucpr |
| scale_pos_weight | computed from train (`n_neg / n_pos`) |
| Random seed | 42 |
| Device | cuda |
| Features | **22** |
| Grid | H3 resolution-8 (~0.74 km²), 1,708,940 TX-bbox cells |
| Time step | **1 day** |
| Promoted | 2026-08-17 (twice in one day — see A.7) |

### The 22 features

| # | Feature | Group | Changes daily? | Importance (step34) |
|---|---|---|---|---|
| 1 | `road_dist_km` | Static — access | No | 0.0515 |
| 2 | `ecoregion_id` | Static — ecology | No | **0.1432** |
| 3 | `elevation_m` | Static — terrain | No | **0.1052** |
| 4 | `slope_deg` | Static — terrain | No | 0.0296 |
| 5 | `aspect_deg` | Static — terrain | No | 0.0192 |
| 6 | `avg_burn_prob` | Static — fuels (FSim) | No | 0.0604 |
| 7 | `whp` | Static — fuels (WHP) | No | 0.0387 |
| 8 | `flep4` | Static — fuels (LANDFIRE) | No | **0.0000** |
| 9 | `cfl` | Static — fuels (LANDFIRE) | No | **0.0000** |
| 10 | `cbd` | Static — fuels (LANDFIRE) | No | 0.0316 |
| 11 | `cbh` | Static — fuels (LANDFIRE) | No | 0.0301 |
| 12 | `powerline_dist_km` | Static — ignition source (HIFLD) | No | 0.0486 |
| 13 | `sin_month` | Calendar | Yes | 0.0733 |
| 14 | `cos_month` | Calendar | Yes | 0.0593 |
| 15 | `sin_dow` | Calendar | Yes | 0.0312 |
| 16 | `cos_dow` | Calendar | Yes | 0.0305 |
| 17 | `is_weekend` | Calendar | Yes | 0.0314 |
| 18 | `is_holiday` | Calendar | Yes | 0.0381 |
| 19 | `hrrr_tmp` | Weather — HRRR forecast | Yes | 0.0413 |
| 20 | `hrrr_vpd` | Weather — HRRR forecast | Yes | **0.0613** |
| 21 | `hrrr_wind` | Weather — HRRR forecast | Yes | 0.0248 |
| 22 | `hrrr_mstav` | Weather — HRRR soil moisture | Yes | 0.0506 |

> **Note:** `flep4` and `cfl` have importance **exactly 0.000** — the model never splits on them. Two of their 22 features are dead weight. We use both and get real signal from `cfl` (9.5 % of gain). This is a concrete, actionable difference.

**SHAP (mean |SHAP|, from `rev2_improvements/diagnose_baseline.json`):** `hrrr_vpd` 0.429 > `elevation_m` 0.297 > `hrrr_tmp` 0.242 > `ecoregion_id` 0.185 > `road_dist_km` 0.170. Weather is the top driver by SHAP even though `ecoregion_id` leads by gain.

## A.3 Labels

**Source:** VIIRS active-fire detections (FIRMS: `VIIRS_SNPP_SP` archive + `_NRT` for recent) **fused with** FPA-FOD v6 fire-occurrence database.

**Fusion rule** (`docs/PLAN.md` §2): an FPA-FOD event searches for VIIRS/GOES detections in the same H3 cell within ±1 day. If matched, the satellite's real UTC timestamp is assigned. If not, day-level is kept and flagged `time_source = imputed`. Post-2020, VIIRS detections themselves become new ignition events — this is how they recovered the 2021+ label gap that FPA-FOD's 2020 cutoff creates.

**Label file:** `data/labels_fused/ignitions_daily_tx.parquet` — **960,054 rows**, columns: `h3_cell, date, label, time_source, cause, max_size_acres, n_sources`.

### Label cleaning — the flare filter (their most important data-quality work)

They discovered that **~27 % of positive labels were persistent industrial hotspots** — gas flares and petrochemical plants that VIIRS sees burning nearly every day. The worst single cell was detected "on fire" **3,176 days**.

| Persistence cutoff | Flare cells removed | % of positives removed |
|---|---|---|
| > 1 % of all days | 1,732 | 36.9 % |
| **> 3 % (chosen)** | **531** | **27.3 %** |
| > 10 % | 145 | 18.0 % |

Effect of removing them, on identical clean test rows:

| Model | AUC-PR | AUROC | Lift |
|---|---|---|---|
| honest, flare-trained | 0.386 | 0.651 | 1.65× |
| **honest, flare-filtered** | **0.450** | **0.711** | **1.92×** |

They then **audited the 531-cell list against NASA's own VIIRS `type` flag** (`audit_viirs_typeflag.py`): 53,669 detections sampled across 48 windows; 267 of 406 checkable cells confirmed as static land sources (**65.8 % confirm rate**); 7 additional missed flare cells found → **v2 list = 538 cells**, shipped as `flare_cells_v2.parquet`.

## A.4 Training data

**File:** `data/tdis_train_daily_hrrr.parquet` — **3,595,513 rows × 33 columns**, dates 2014-01-01 → 2026-12-31, overall positive rate 0.267.

**But the served model does not use all of it.** I verified the actual pipeline:

```
3,595,513 rows  (shipped table)
  − flare cells (538)
  − dates after 2026-07-29 (last label)
  − dropna(12 static features)
  − dropna(hrrr_vpd)        ← HRRR archive starts 2018-07-16
  − dropna(hrrr_mstav)      ← MSTAV cache: 2018-07-16 → 2026-07-29
= 2,007,436 usable rows
```

| Split (as used by `step34_mstav_retrain.py`) | Years | Rows | Positive rate |
|---|---|---|---|
| **Train** | 2018-07 → 2021 | ~856,300 | ~0.229 |
| *(2022 — discarded, see D.6)* | 2022 | 259,150 | 0.261 |
| **Test** | 2023 → 2026-07 | **891,983** | **0.2341** |

Per-year usable rows: 2018: 108,670 · 2019: 255,499 · 2020: 247,800 · 2021: 244,334 · 2022: 259,150 · 2023: 246,925 · 2024: 247,134 · 2025: 248,283 · 2026: 149,641.

## A.5 Results — verified

### Balanced test sample (2023–2026, base rate 0.234)

| Metric | Their claim | **My independent re-score** | Match |
|---|---|---|---|
| AUC-PR | 0.4941 | **0.4941** | ✅ |
| AUROC | 0.7402 | **0.7402** | ✅ |
| Best F1 | 0.496 | **0.4958** (thr 0.4991, P 0.441 / R 0.566) | ✅ |
| Test rows | 892 K | **891,983** | ✅ |
| Positive rate | 0.2344 | **0.2341** | ✅ |

Additional numbers I measured that are **not** in their docs:

| Split | AUC-PR | AUROC |
|---|---|---|
| Train (≤2021) | 0.6916 | 0.8652 |
| **2022 (nominal "val")** | **0.4664** | **0.6758** |
| Test (≥2023) | 0.4941 | 0.7402 |

→ Train-to-test AUC-PR gap of **0.198**. And 2022 scores *worse* than the later test years — an inversion nobody has explained. Their own `diagnose_baseline.json` shows the same shape on the previous model (train AUROC 0.804 / val 0.668 / test 0.733).

### Real deployment population (5,064,462 cell-days, 2024–2026, base rate 0.0192)

| Metric | Value |
|---|---|
| AUC-PR | 0.0878 |
| AUROC | **0.7968** |
| **Lift** | **4.58×** |
| Best F1 | 0.1692 at calibrated threshold 0.0774 (P 0.137 / R 0.222) |

Real-population threshold sweep (calibrated probabilities):

| Cal. threshold | Precision | Recall | F1 | Cell-days flagged |
|---|---|---|---|---|
| 0.02 | 0.045 | 0.698 | 0.085 | 1,500,096 |
| 0.05 | 0.094 | 0.328 | 0.146 | 338,069 |
| **0.075** | **0.149** | **0.186** | **0.165** | ~120,000 |
| 0.10 | 0.173 | 0.150 | 0.161 | 84,663 |
| 0.15 | 0.186 | 0.121 | 0.147 | 63,513 |

### Calibration (their strongest engineering result)

Isotonic regression, fit on 2024–2025 outcomes, evaluated on a never-seen 2026 holdout (1,130,220 rows, base rate 0.0213):

| | ECE | Brier | Mean prediction |
|---|---|---|---|
| Raw | 0.2799 | 0.1376 | 0.3158 |
| **Calibrated** | **0.0018** (−99.4 %) | **0.0200** (−87 %) | **0.0194** |

Real base rate on that holdout: 0.0213. Calibrated mean: 0.0194. **The raw score overstates risk ~17×; after calibration it is genuinely a probability.** The reliability bins were perfectly monotonic before calibration — which is why isotonic works almost exactly.

## A.6 Event validation — 5 real Texas wildfires

| Event | Date | Size | Driver | Captured | WHERE (hazard pctile) | WHEN (fire-weather pctile) |
|---|---|---|---|---|---|---|
| **Smokehouse Creek** | 2024-02-26 | 1.06 M ac (largest in TX history) | power line + wind, cold | ✅ 4,730 det | ✅ 94th (2.9× state mean) | ⚠️ 61st → **75th** after wind re-tune |
| **Crabapple** | 2025-03-15 | 9.9 K ac | wind + dry | ✅ 312 det | ✅ 84th (2.5×) | ✅ **99th** |
| **Lavender** | 2026-02-17 | 18.4 K ac | wind | ✅ 447 det | ✅ 96th (3.0×) | 🟡 80th → **94th** after re-tune |
| **Hunggate** | 2026-05-14 | 34 K ac | lightning + heat | ✅ 388 det | 🟡 74th (2.1×) | ✅ **99th** |
| **Windy Deuce** | 2024-02-26 | 144 K ac | downed power line | ✅ 46→255 det | ✅ 86th (2.5×) | 78th |

**Scoreboard: capture 5/5 · WHERE 5/5 at 74th–96th · WHEN 99th for heat-driven (2/2), 73rd–94th for wind-driven (3/3).**

**The documented gap:** wind-driven cool-season Panhandle grass fires are under-scored because gridMET provides only **daily-mean** wind, which washes out afternoon gusts. They fixed it partially by re-weighting the fire-weather index (wind 0.20 → 0.40) and adding HRRR gust to the live path. The ML model itself still cannot see it.

## A.7 The promotion chain (2026-08-17 — two promotions in one day)

| Step | Change | Test AUC-PR | Realpop lift | Status |
|---|---|---|---|---|
| baseline | served pre-phase (20 feat, depth 6) | 0.4825 | 4.43× | superseded |
| step30 | depth-9 tuning, 3-seed robust | 0.4875 (+0.0050) | 4.47× | promoted, then superseded |
| step32 | + `powerline_dist_km` + flare v2 | 0.4925 (+0.0050) | 4.56× | gate-passed |
| **step34** | **+ `hrrr_mstav` soil moisture** | **0.4941 (+0.0016)** | **4.58×** | **SERVED** |
| step30-mono | monotone wind constraint | 0.4822 (−0.0003) | — | rejected |

Every promotion passed three gates: (1) test AUC-PR improvement, (2) 3-seed robustness, (3) real-population validation.

## A.8 Negative results — the most valuable thing in the handoff

These are experiments they ran, that failed, and recorded. **This saves us from repeating them.**

| Experiment | Result | Verdict |
|---|---|---|
| **Model output × NOAA HWP** | AUC-PR 0.0945 → **0.0372** (−61 %), AUROC 0.755 → 0.573 | ❌ **Severely harmful. Never do this.** |
| Model × TX HWP | 0.0945 → 0.0771 (−18 %) | ❌ Harmful |
| Model × composite FWI | 0.0945 → 0.0687 (−27 %) | ❌ Harmful |
| Add `hrrr_gust` as model feature (step5) | 0.4823 vs 0.4825 baseline | ➖ Flat |
| Add 5-day trailing ERC/VPD/wind (step5) | within ±0.0002 | ➖ Flat |
| Explicit `gust_x_vpd`, `hwp_core` interactions (step6) | 0.4824 | ➖ Flat |
| Monotone constraint: wind↑ ⇒ risk↑ | −0.0003 | ❌ Rejected |
| Monotone wind + vpd | −0.0013 | ❌ Rejected |

**Their conclusion, reached four independent ways: wind does not predict fire *occurrence*. It predicts fire *behavior*.** That is a genuinely interesting scientific finding and it is well-evidenced here.

**One positive negative-result:** `aggregation_recall_experiment.json` shows that coarsening the question dramatically improves the operating point:

| Aggregation | Units | Base rate | Flag top 50 % → P / R |
|---|---|---|---|
| cell-day (res-5, daily) — current | 5,064,462 | 0.0192 | 0.033 / 0.886 |
| cell-week (res-5, weekly) | 726,570 | 0.0875 | 0.147 / 0.882 |
| **region-week (res-4, weekly)** | 109,215 | 0.3138 | **0.489 / 0.780** |

> *"The honest answer to small-fire misses is coarser questions, not a different model."* — this is an operationally important insight we should adopt.

## A.9 Credibility tests — the two a hostile reviewer runs

| Test | Result | Meaning |
|---|---|---|
| **Label shuffle** (train on permuted labels) | AUC-PR 0.226 ≈ base rate 0.234; AUROC 0.486 ≈ 0.5 | **No hidden pipeline leak** |
| **Spatial holdout** (20 % of cells never in training) | unseen 0.446 / 1.91× vs seen 0.451 / 1.92×; gap **0.005** | **Skill is transferable geography, not cell memorization** |

**We have run neither of these. This is our biggest credibility gap.**

## A.10 The dashboard

`dashboard/tdis_fire_explorer_v3_standalone.html` — single self-contained file, opens in any browser, needs internet only for basemap tiles and Leaflet/h3-js CDN.

| Tab | What it shows | ML involved? | Resolution |
|---|---|---|---|
| 🔥 **Wildfire Risk** | static hazard × live NOAA HWP fire-weather | No | res-8, refines on zoom |
| ✦ **Ignition** | the trained model's own output on live HRRR | **Yes** | res-5 (~21.5 km²) |
| 🌬️ **Fire Weather Index** | NOAA HWP alone | No | res-5 |

The Ignition tab shows two numbers on hover: **Rank score** (colors the map, explicitly *not* a probability) and **Calibrated probability** (the number safe to quote). This rank-for-color / calibrated-for-numbers discipline is a good pattern we should copy.

**NOAA HWP equation** (James et al. 2025, *Weather and Forecasting*, Eq. 3):

```
HWP = 0.213 × G^1.50 × VPD^0.73 × (1 − M)^5.10
```

G = 10 m gust (m/s, floored at 3) · VPD in hPa · M = soil-moisture availability 0–1 (HRRR `MSTAV`). Normalized by its own 99.5th percentile over the 2024–26 TX archive, displayed on a 0–2 intensity scale.

Current shipped forecasts: `forecast_2026-08-18.json` (24 h) and `forecast_2026-08-19.json` (48 h), 5,382 res-5 cells each. I verified the 48 h file: mean raw `ign` 0.3441, mean `ignCal` **0.0234** — consistent with the ~1.9 % real base rate.

## A.11 Complete file inventory

### `Focused_Files/` — 26 files, ~360 MB

| Path | What it is |
|---|---|
| `README.md` | package overview + daily-use commands |
| `REPRODUCE.md` | 3-tier honest reproducibility statement |
| `environment.yml` | pinned conda env (Python 3.10.18, xgboost 3.1.1, h3 4.3.1, herbie-data 2025.12.0, scikit-learn 1.7.2, pandas 2.3.2, scipy 1.15.3, joblib 1.5.2) |
| `docs/METHODOLOGY.md` | the locked canonical method + retired approaches |
| `docs/CLASSIFICATION_METRICS.md` | full metrics, both populations, red flags, fixes |
| `models/tdis_forecast_hrrr_filtered.json` | **the served model** (22 feat, 1000 trees) |
| `models/operational_isotonic_calibrator.joblib` | **its calibrator — paired** |
| `models/tdis_forecast_hrrr_filtered_meta.json` | metadata — ⚠️ **STALE** (see D.2) |
| `dashboard/tdis_fire_explorer_v3_standalone.html` | **open this** |
| `dashboard/tdis_fire_dashboard_v3.html` | template used to rebuild the above |
| `dashboard/explorer_static_multires.json` | static hazard layer |
| `dashboard/forecast_2026-08-18.json`, `..._19.json` | current 24 h / 48 h forecasts |
| `dashboard/README_v3.md` | 3 tabs + HWP equation explained |
| `data/tdis_train_daily_hrrr.parquet` | 3,595,513 × 33 — training table |
| `data/static_features/tx_static_master.parquet` | 1,708,940 × 15 |
| `data/static_features/powerline_dist_km.parquet` | 1,708,940 × 2 |
| `data/labels_fused/ignitions_daily_tx.parquet` | 960,054 × 7 |
| `data/labels_fused/flare_cells_v2.parquet` | 538 × 1 |
| `New_Training817_moredata/mstav_feature.parquet` | 2,007,436 × 3 — cached soil moisture |
| `New_Training817_moredata/step34_mstav_retrain.py` | retrains the exact served model |
| `scripts/13_model_forecast_day.py` | **the live forecast script** |
| `scripts/22_embed_v3.py` | rebuilds the dashboard |
| `scripts/fwi_config.py` | fire-weather formulas + tunable knobs |

### `Supporting_files/` — ~150 files

- **`scripts/` — 29 numbered pipeline scripts** (`01_download_viirs.py` → `29_test_ignition_x_fwi.py`) plus `run_all.sh`, `run_hwp_chain.sh`, `sweep_phase1_forensics.py`, `sweep_phase6_credibility.py`
- **`docs/` — 11 documents**: `HANDBOOK.md`, `HANDOFF.md`, `PLAN.md`, `VALIDATION.md`, `SANITY_CHECK.md`, `DEPLOYMENT_STRATEGY.md`, `LIVE_PIPELINE_PLAN.md`, `PRESENTATION.md`, `DEMO_SMOKEHOUSE.md`, `METHODOLOGY.md`, `CLASSIFICATION_METRICS.md`
- **`models/` — 21 files**: honest / ceiling / operational variants + calibrators + validation JSONs + `.bak` rollback copies
- **`rev2_improvements/` — 20 files**: every ablation with full logs (`step4` hyperparameter sweep, `step5` gust+trailing, `step6` interactions, `step29` ignition×FWI, `step30/31` tuning, `diagnose_model.py` SHAP)
- **`New_Training817_moredata/` — 20 files**: `download_mstav.py`, `audit_viirs_typeflag.py`, `promote_d9.py`, `step32_powerline_retrain.py`, `step34_mstav_retrain.py` + all result JSONs
- **`dashboard/` — 8 files**: v1, v2 (with historical scrubber), v3
- **`data/`**: adds `fwi_components_res5.parquet` (3.4 M × 6), `viirs_tx_h3.parquet` (1.6 M × 9), `tdis_train_daily_tx_flarefiltered.parquet` (3.2 M × 30), **`hwp_params.json`**
- **`case_study_screenshots/`**: Smokehouse Creek 2024-02-26 ignition + risk maps

---

# PART B — What we built (V1 / V2 / TX)

## B.1 V1 — proof of concept *(archived)*

| Property | Value |
|---|---|
| Labels | FIRMS 2024 fire detections only |
| Features | **18** — annual GEE composites |
| Feature list | NDVI, EVI, LST, Temperature, Wind, Rainfall, DEM, Slope, Aspect, LandCover, month, day_of_year, season_code, sin_month, cos_month, sin_doy, cos_doy, is_peak_fire_season |
| Split | Train Jan–Aug / Val Sep / Test Oct–Dec 2024 |
| Negatives | 1:3 (`scale_pos_weight` 3.0) |
| Test rows | **2,250** (TP 508 + FP 315 + TN 1,354 + FN 73) |
| Models tried | XGBoost (full), XGBoost (ablation), Random Forest, LightGBM |

**V1 test results (LightGBM, threshold 0.46):**

| Metric | Value |
|---|---|
| AUROC | 0.9142 |
| AUPR | 0.7549 |
| F1 | 0.7236 |
| Precision | 0.6173 |
| Recall | 0.8744 |
| **Accuracy** | **0.8276** |

> ⚠️ **V1's high scores are not meaningful.** Our own `project_audit.md` documents why: features are **annual composite rasters**, so every date at the same location gets identical values. The model learns *"which month/location historically has fires"* — fire-occurrence correlation, not 24-hour ignition forecasting. And the test set is only 2,250 rows. **V1 must not be quoted to the TDIS team as a result.**

## B.2 V2 — production pipeline baseline

| Property | Value |
|---|---|
| Labels | FPA-FOD v6, Texas, ≥1 acre |
| Period | 2014–2020 |
| Grid | H3 res-8 (~0.82 km²) |
| Time step | **6-hour UTC windows** (00Z / 06Z / 12Z / 18Z) |
| Negatives | **day-matched 1:10** |
| Rows | 376,233 (positive rate 9.1 %) |
| Features | **31** |
| LANDFIRE | ⚠️ **rasters not downloaded — 4 features ~0** |

**V2 baseline results (`xgb_baseline_tx`, depth 7, lr 0.05, 387 trees):**

| Split | Rows | Pos | AUROC | AUPR | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| TRAIN | 252,066 | 22,916 | 0.9302 | 0.5723 | 0.5386 | 0.4153 | 0.7659 |
| VAL | 61,181 | 5,561 | 0.8742 | 0.4125 | 0.4316 | 0.3355 | 0.6049 |
| **TEST** | 62,986 | 5,726 | **0.8569** | **0.3978** | 0.4082 | 0.3315 | 0.5313 |

**The leakage we caught and fixed** (worth telling them — it's the mirror image of their flare finding):

| | Leaked | Clean |
|---|---|---|
| AUROC | 0.9900 ❌ | **0.8569** ✅ |
| AUPR | 0.8642 ❌ | **0.3978** ✅ |
| Top features | `has_fire_history` 58 %, `fire_count` 36 % | `burnable`, `erc`, lat/lon |

`fire_count` and `has_fire_history` were computed from the full 2014–2020 dataset *including test years*. A fire cell has `fire_count ≥ 1` by definition. Removed from `FEATURE_COLS` and retrained.

## B.3 TX — current best (the "86 %")

| Property | Value |
|---|---|
| Dataset | `final_training_dataset_tx_22.07.2026_landfire.xlsx` (94 MB, 376,233 × 42) |
| After preprocessing | **375,779 rows** (454 duplicates dropped) |
| Unique H3 cells | **317,142** |
| Date range | 2014-01-01 → 2020-12-06 |
| Fire rows | 33,749 (**8.98 %**) |
| Non-fire rows | 342,030 |
| Features | **34** (gridMET) / **42** (with HRRR) |
| Split | TRAIN 2014–2017 (251,785) · VAL 2018 (61,139) · TEST 2019–2020 (62,855) |
| GPU | RTX 3050 6 GB / RTX 3060 |

### Our 34 features (gridMET model)

| Group | Count | Features |
|---|---|---|
| **Landscape** | 7 | `avg_burn_prob`, `whp`, `flep4`, `cfl`, `cbd`, `cbh`, `burnable` |
| **Daily weather (gridMET)** | 8 | `erc`, `fm100`, `vpd`, `vs`, `rmax`, `rmin`, `tmmx`, `pr` |
| **5-day trailing stats** | 12 | `erc_5D_mean`, `erc_5D_max`, `fm100_5D_mean`, `fm100_5D_min`, `vpd_5D_mean`, `vpd_5D_max`, `vs_5D_mean`, `vs_5D_max`, `rmax_5D_mean`, `rmax_5D_min`, `tmmx_5D_mean`, `tmmx_5D_max` |
| **Temporal** | 4 | `sin_month`, `cos_month`, `sin_hour`, `cos_hour` |
| **Location** | 2 | `centroid_lat`, `centroid_lon` |
| **Flag** | 1 | `gridmet_missing` |

**Excluded — leakage:** `fire_count`, `has_fire_history`
**Excluded — identifiers:** `h3_cell`, `date_utc`, `window_hour`, `window_6h_utc`, `fire_year`, `label`

### The 8 additional HRRR features (42-feature model)

`temp_pw`, `rh_pw`, `wind_pw`, `vpd_pw_hrrr`, `hpbl_pw`, `dswrf_pw`, `hrrr_pw` (availability flag), `hrrr_rh_valid` (RH-bug flag)

HRRR coverage is uneven: 12.2 % of rows in 2014 → 95–99 % by 2018–2020. Missing rates: `temp_pw`/`wind_pw`/`hpbl_pw`/`dswrf_pw` 19.0 %, `rh_pw`/`vpd_pw_hrrr` 45.7 %.

### Our full model progression

| # | Model | Feat | Hyperparameters | Test AUROC | Test AUPR | Test F1 | P | R |
|---|---|---|---|---|---|---|---|---|
| 1 | V2 baseline (zeroed LANDFIRE) | 31 | d7, lr .05 | 0.8569 | 0.3978 | 0.4082 | 0.332 | 0.531 |
| 2 | Real LANDFIRE, untuned | 34 | d7, mcw30, lr .05, sub .8 | 0.8637 | 0.4106 | 0.4169 | 0.352 | 0.511 |
| 3 | **Tuned (21 trials)** | 34 | **d9, mcw30, lr .01, sub .9, 1627 trees** | **0.8687** | **0.4247** | **0.4293** | **0.392** | **0.475** |
| 4 | HRRR-enriched (borrowed config) | 42 | d9, mcw30, lr .01, sub .9, 1654 trees | 0.8682 | 0.4236 | 0.4251 | 0.351 | 0.539 |
| 5 | HRRR, independently tuned (21 trials) | 42 | d9, mcw30, lr .02, sub .9, 765 trees | 0.8682 | 0.4225 | 0.4247 | 0.340 | 0.566 |
| 6 | HRRR, availability flags dropped | 40 | d9, mcw30, lr .01, sub .9, 1327 trees | 0.8673 | 0.4198 | 0.4247 | 0.351 | 0.537 |

**Best model confusion matrix (TEST, threshold 0.710):**

```
                       PREDICTED
                    No Fire   |   Fire
          ────────────────────────────────
Actual:   No Fire  |  TN=53,129  |  FP= 4,131
          Fire     |  FN= 2,937  |  TP= 2,658
```

Recall 47.5 % · FPR 7.2 % · Precision 39.2 % · Specificity 92.8 % · NPV 94.8 % · F1 42.9 %

**Our feature importance (tuned, % of total gain):**

| Rank | Feature | % gain | Group |
|---|---|---|---|
| 1 | `burnable` | **19.8 %** | Landscape |
| 2 | `avg_burn_prob` | 12.1 % | Landscape (TxWRAP) |
| 3 | `whp` | 10.6 % | Landscape (TxWRAP) |
| 4 | `cfl` | 9.5 % | Landscape (LANDFIRE) |
| 5 | `centroid_lon` | 7.9 % | Location |
| 6 | `flep4` | 4.3 % | Landscape |
| 7 | `centroid_lat` | 4.3 % | Location |
| 8 | `cos_hour` | 2.3 % | Temporal |
| 9 | `erc_5D_max` | 1.9 % | gridMET |
| 10 | `sin_hour` | 1.8 % | Temporal |

Group totals: **Landscape 52.8 % · Location 12.2 % · gridMET weather 18.3 % · Temporal 9.0 %**

### Our HRRR finding

**Adding HRRR sub-daily weather did NOT improve the model.**

| vs. best gridMET model | ΔAUROC | ΔAUPR |
|---|---|---|
| HRRR, borrowed config | −0.0005 | −0.0011 |
| HRRR, independently tuned | 0.0000 | −0.0022 |
| HRRR, no availability flags | −0.0014 | −0.0049 |

A full independent 21-trial sweep on the 42-feature set converged to essentially the same result, which **rules out "HRRR just needs its own tuning"** as an explanation. HRRR gain share was only 7.4–7.8 %.

---

# PART C — Direct comparison

## C.1 Problem setup

| | **Ours (TX best)** | **Theirs (served)** |
|---|---|---|
| Question | Fire discovered in this cell in the next **6 hours**? | Fire activity in this cell **today**? |
| Labels | FPA-FOD v6 only | VIIRS + FPA-FOD **fused** |
| Label cleaning | leakage columns removed | **flare filter (538 cells, 27 % of positives)** |
| Period | 2014–2020 (7 yr) | 2018-07 → 2026-07 (8 yr) |
| Grid | H3 res-8 | H3 res-8 (served at res-5) |
| Time step | **6-hour UTC window** | **1 day** |
| Negatives | day-matched **1:10** | ~**1:3** |
| Positive rate | **9.1 %** | **23.4 %** |
| Weather | gridMET **observed** daily + 5-day trailing | HRRR **forecast** |
| Static | LANDFIRE + TxWRAP + `burnable` | LANDFIRE + FSim + terrain + roads + **powerlines** |
| Location features | **raw `centroid_lat` / `centroid_lon`** | `ecoregion_id`, `elevation_m` (no raw coords) |
| Split | 2014-17 / 2018 / 2019-20 | 2018-21 / (2022 unused) / 2023-26 |

## C.2 Scale

| | **Ours** | **Theirs** |
|---|---|---|
| Total rows | **376,233** (375,779 after dedup) | **3,595,513** shipped → **2,007,436** usable |
| Train rows | 251,785 | ~856,300 |
| Val rows | 61,139 | 259,150 *(discarded)* |
| **Test rows** | **62,855** | **891,983** |
| Unique H3 cells | 317,142 | 1,708,940 (bbox) / 481,822 (in MSTAV cache) |
| **Rows per cell** | **1.19** ⚠️ | ~4.2 |
| Real-population scored | ❌ never | ✅ 5,064,462 cell-days |

> ⚠️ **1.19 rows per cell is a structural problem on our side.** Our model almost never sees the same cell twice. Combined with raw lat/lon as features, we cannot currently distinguish "learned geography" from "interpolated a lookup table." Their 5-event spatial-holdout equivalent is exactly the test that would settle it.

## C.3 Results side by side

### On each team's own balanced test set

| Metric | **Ours** | **Theirs** | Note |
|---|---|---|---|
| Base rate | 0.091 | 0.234 | different by design |
| Test rows | 62,855 | 891,983 | theirs 14× larger |
| **AUROC** | **0.8687** | 0.7402 | **not comparable** |
| **AUPR** | 0.4247 | **0.4941** | **not comparable** |
| **Lift (AUPR ÷ base rate)** | **4.67×** | 2.11× | ← the fair comparison |
| Best F1 | 0.4293 | 0.4958 | |
| Precision @ best F1 | 0.392 | 0.441 | |
| Recall @ best F1 | 0.475 | 0.566 | |
| Train AUROC | 0.9550 | 0.8652 | |
| Train→test AUROC gap | 0.086 | 0.125 | |

**On lift — the only base-rate-fair metric — we are ahead: 4.67× vs 2.11×.**

### On evidence quality

| | **Ours** | **Theirs** |
|---|---|---|
| Real deployment population scored | ❌ | ✅ 5.06 M cell-days, lift **4.58×**, AUROC **0.7968** |
| Probability calibration | ❌ raw scores only | ✅ ECE **0.0018** on 2026 holdout |
| Label-shuffle leak control | ❌ | ✅ collapses to chance |
| Spatial-holdout memorization test | ❌ | ✅ gap 0.005 |
| Real-event validation | ❌ | ✅ 5/5 events |
| Flare / industrial label audit | ❌ | ✅ 538 cells + NASA `type` cross-check |
| Live forecast script | ❌ hindcast only | ✅ `13_model_forecast_day.py` |
| Deployed dashboard | ❌ static PNGs | ✅ 3-tab standalone HTML |
| Negative-results archive | ❌ | ✅ ~20 documented experiments |
| Documented reproducibility tiers | ❌ | ✅ `REPRODUCE.md` |
| Second state (transferability) | ✅ California pipeline | ❌ Texas only |
| Sub-daily temporal resolution | ✅ 6-hour windows | ❌ daily only |
| NFDRS fuel indices (ERC/BI/fm100/fm1000) | ✅ | ❌ (in ceiling model only, never served) |
| 5-day trailing weather | ✅ 12 features | ❌ (tested, flat) |
| `burnable` fuel mask | ✅ top feature, 19.8 % | ❌ none |

## C.4 The honest verdict

**Neither model is "better." They answer different questions with different evidence standards.**

- **Our modelling is competitive or ahead.** Lift 4.67× vs 2.11×. Our LANDFIRE + `burnable` + trailing-weather feature set genuinely works — `burnable` alone carries 19.8 % of gain and they have no equivalent. Two of their 22 features (`flep4`, `cfl`) are literally dead (importance 0.000) while we get 9.5 % from `cfl`.
- **Their evidence is far stronger.** They measured what happens at deployment; we measured what happens on the sample we constructed. They calibrated; we did not. They proved no leakage; we assert it. They validated against five real named fires; we have not.
- **Our headline "86 %" needs re-framing.** It is AUROC 0.8687 on a 9.1 %-positive, 6-hourly, 2019–2020 test set — a legitimately easier population than theirs. Quoting it against their 0.74 is misleading and they will notice. Our defensible headline is **lift 4.67× at 9.1 % base rate**, and even that needs the spatial-holdout test behind it.
- **Both teams independently found HRRR sub-daily adds little.** Ours: −0.0005 AUROC / −0.0011 AUPR with an independent 21-trial sweep. Theirs: gust and trailing features flat within ±0.0002. **Two independent confirmations of the same negative result — this is publishable.**

---

# PART D — Missing files, conflicts, and defects in their handoff

## D.1 🔴 `data/hwp_params.json` is MISSING from `Focused_Files`

**Impact: silent wrong output, no error raised.**

`scripts/fwi_config.py` line ~60:

```python
def _load_hwp_params():
    p = _TF / "data" / "hwp_params.json"
    if p.exists():
        return json.load(open(p))
    # pre-fit fallback: TX = NOAA coefficients; refs make output land in ~0-1
    return {'tx': {**HWP_NOAA, 'ref': 60.0}, 'noaa': {'ref': 60.0}}
```

The file is not in `Focused_Files/data/`. I verified. So the fallback fires: the TX-fit HWP variant silently reverts to NOAA coefficients, and both variants use `ref = 60.0` instead of the fitted 99.5th-percentile reference.

`dashboard/README_v3.md` explicitly states the normalization comes from this file:

> *"the raw HWP is divided by its own 99.5th-percentile value over the full 2024–2026 Texas archive (stored in `data/hwp_params.json`)"*

**Anyone regenerating the dashboard from `Focused_Files` alone gets wrong Wildfire Risk and Fire Weather Index normalization, with no warning.**

**Fix:** copy `Supporting_files/Supporting_files/data/hwp_params.json` → `Focused_Files/Focused_Files/data/hwp_params.json`. Better: make `_load_hwp_params()` log a loud warning when it falls back.

## D.2 🔴 `tdis_forecast_hrrr_filtered_meta.json` is STALE

| | meta.json says | Actual model |
|---|---|---|
| Feature count | **20** | **22** |
| Missing from list | — | `powerline_dist_km`, `hrrr_mstav` |
| AUC-PR | 0.4825 | **0.4941** |
| AUROC | 0.7333 | **0.7402** |
| Flare cells | 531 | **538** (v2) |
| Split | `"train": "2018-07..2021"` | correct ✅ |

I confirmed by loading the model: `num_feature: 22`, feature names include `powerline_dist_km` and `hrrr_mstav`, `num_trees: 1000`.

The metadata describes the **step30 model**, not the served **step34** model. Anyone reading the meta to understand what is deployed gets the wrong answer. Their own `CLASSIFICATION_METRICS.md` §5 raised "meta.json missing population context" as a 🟡 flag and marked it *Fixed* — it regressed.

## D.3 🔴 The calibration script is absent — their own core rule is not executable

`README.md` — *"The one rule that matters"*:

> **Model and calibrator are a pair.** If you ever retrain, refit the calibrator on the new model's output before serving it.

But in `Focused_Files`:
- `scripts/` contains only `13`, `22`, `fwi_config.py`. **No `26`, no `27`.**
- `New_Training817_moredata/step34_mstav_retrain.py` fits and saves the model — and **does not refit the calibrator**.

So retraining from `Focused_Files` produces exactly the failure their rule warns against: a new model paired with a stale calibrator.

**Fix:** ship `scripts/26_score_operational_realpop.py` and `scripts/27_fit_isotonic_operational.py` (both exist in `Supporting_files/scripts/`), or add the calibration step to `step34_mstav_retrain.py`.

## D.4 🟠 The served calibrator has NO reproduction script anywhere

`Supporting_files/New_Training817_moredata/` contains `promote_d9.py`, which refits a calibrator — but it writes `models/tuned_d9_isotonic_calibrator.joblib`, for the **step30** model.

The served model is **step34**. `step34_calibration_report.json` exists (holdout n=1,130,220, base 0.0213, raw ECE 0.2799 → cal ECE 0.0018) — but **no script in either folder produces it.** There is no `promote_step34.py`.

**The exact model+calibrator pair now in production was produced by an unversioned ad-hoc run.** Given rule #1 of their own `DEPLOYMENT_STRATEGY.md` §2, this is the one thing that most needs a script.

## D.5 🟠 Three conflicting train/val/test split definitions

| Source | Train | Val | Test |
|---|---|---|---|
| `split` column in the shipped parquet | 2014–2020 (1,888,409 rows) | 2021 (284,711) | 2022–2026 (1,422,393) |
| `docs/METHODOLOGY.md` §2 | 2014–2021 | 2022 | 2023–2026 |
| `step34_mstav_retrain.py` (**what actually ran**) | `year <= 2021` | *(none)* | `year >= 2023` |
| `models/*_meta.json` | `"2018-07..2021"` | `"2022"` | `"2023-2026 clipped"` |

The script's version is authoritative — it produced the served model. The shipped `split` column is unused and misleading; a reader who trusts it will train on a different set than the served model saw.

## D.6 🟠 2022 is silently discarded, and it scores anomalously badly

`step34_mstav_retrain.py`:

```python
tr, te = df[df.year <= 2021], df[df.year >= 2023]
```

2022 (259,150 rows) is in neither. And the fit is `n_estimators=1000` with **no eval set and no early stopping** — so 2022 selects nothing. It is not a validation year; it is dropped data.

Worse, when I scored the served model on 2022:

| Split | AUC-PR | AUROC |
|---|---|---|
| Train ≤2021 | 0.6916 | 0.8652 |
| **2022** | **0.4664** | **0.6758** |
| Test ≥2023 | 0.4941 | 0.7402 |

**2022 is worse than the *later* test years.** That inverts the normal temporal-decay pattern and nobody has investigated it. Their own `diagnose_baseline.json` shows the same inversion on the previous model (train 0.804 / val 0.668 / test 0.733, `val_test_gap: −0.0657`) and does not comment on it.

## D.7 🟠 Direct contradiction: does the flare threshold matter?

`models/flare_robustness_band.json` — the script's own machine-written verdict:

> `"verdict": "lift band 3.54-4.69x, AUROC band 0.7657-0.7897 across a 10x change in the flare cutoff -- conclusions MAY hinge on the 3% choice"`

`docs/CLASSIFICATION_METRICS.md` §7c — the human summary of that same run:

> *"Conclusions do not hinge on the 3% choice."*

**These say opposite things.** The lift swings 3.54× → 4.69×, a **32 % relative change** in the headline metric. The script is more honest than the doc. Their `SANITY_CHECK.md` F2 also lists this as 🟠 *"load-bearing"*. The `CLASSIFICATION_METRICS.md` sentence should be corrected.

## D.8 🟠 Three different values reported for the same "honest_filtered" model

| Source | AUC-PR | AUROC | n_test |
|---|---|---|---|
| `models/rescore_all_models.json` | 0.4018 | 0.6633 | 262,995 |
| `docs/CLASSIFICATION_METRICS.md` §3b | 0.442 | 0.694 | — |
| `docs/VALIDATION.md` §9 | 0.450 | 0.711 | — |
| `models/..._honest_filtered_meta.json` | 0.4502 | 0.7112 | — |

Their `CLASSIFICATION_METRICS.md` §5 flagged this exact problem (🟠 *"Stale/conflicting numbers were found in the project's own logs"*) and §7d marked it **Fixed**. It is not fixed — `rescore_all_models.json` still disagrees with everything else, on a different row count.

## D.9 🟠 The calibrator is applied at the wrong resolution — twice

`scripts/13_model_forecast_day.py` admits it in a comment:

> *"Calibrator was fit on res-5 scores; applying to res-8 scores is an approximation (same model, same scale)."*

Then the pipeline compounds it:

1. Model scores **1.7 M res-8** cells
2. **res-5-fitted** calibrator applied to **res-8** scores
3. Calibrated res-8 probabilities are **averaged** to res-5:
   ```python
   agg = st.groupby('h3_5').agg(ign=('ign','mean'), ign_cal=('ign_cal','mean'), ...)
   ```

**Averaging calibrated probabilities ≠ calibrating an averaged score.** Isotonic regression is non-linear and monotone; `mean(f(x))` and `f(mean(x))` differ. The error has never been quantified.

Empirically it lands close — shipped `forecast_2026-08-19.json` has mean `ignCal` **0.0234** vs a real base rate near 0.019 — but "close" is not "validated," and this is the number they tell people to quote in meetings.

## D.10 🟡 Applying the calibrator to a balanced population makes Brier *worse*

I measured, on their balanced test set:

| | Brier |
|---|---|
| Raw | 0.1856 |
| Calibrated | **0.2087** ⬆ worse |

This is *correct* behaviour — the calibrator maps to the ~1.9 % real base rate, not the 23 % balanced rate — but it is an easy trap for anyone validating the pair, and it is documented nowhere. A reviewer applying the calibrator to the test set will conclude the calibration is broken.

## D.11 🟡 `is_holiday` in the live path is a hardcoded 6-date set

`scripts/13_model_forecast_day.py`:

```python
st['is_holiday'] = int((mo, TARGET.day) in {(1,1),(7,4),(6,19),(11,11),(12,25),(10,31)})
```

Missing: Thanksgiving, Labor Day, Memorial Day, MLK Day, Presidents' Day. No observed-date shifting (Jul 4 on a Sunday). If the training table's `is_holiday` was built from a proper calendar (e.g. `pandas.tseries.holiday.USFederalHolidayCalendar`), this is a **live train/serve skew** on a feature carrying **3.8 % importance** — higher than `hrrr_wind` (2.5 %).

## D.12 🟡 `flep4` and `cfl` have importance exactly 0.000

Two of the 22 features are never split on, in every result JSON (step32, step34, honest, ceiling). They are inert. Either the values are degenerate (constant / all-null after the join) or they carry no signal at res-8 daily. Worth a null-rate check — **we get 9.5 % of gain from `cfl`**, so the difference is probably a data-preparation issue on their side, not a real property of the feature.

## D.13 🟡 `METHODOLOGY.md` overstates the training period

> *"H3 res-8 cells across Texas, **2014–2026**"*

The served model sees **2018-07-16 onward only**. Every pre-2018 row is dropped by `dropna(['hrrr_vpd'])` because the HRRR archive starts then, and again by `dropna(['hrrr_mstav'])`. I verified: 2,007,436 usable rows, first date 2018-07-16. That is 2,932 dates, not 12 years.

The meta.json field `"train": "2018-07..2021"` is right; the methodology doc is wrong.

## D.14 🟡 `METHODOLOGY.md` says "~400 frozen decision trees"

> *"each row walks through the **~400** frozen decision trees"*

The served model has **1,000** trees (I verified `num_trees: 1000`). 400 was the old fixed budget. Stale sentence.

## D.15 Their own open issues (documented by them — credit where due)

| ID | Issue | Status |
|---|---|---|
| F1 | 6.1 % of "Texas" labels are NM/OK bounding-box spillover (42,368 fire-days, incl. Ruidoso South Fork NM 2024 and the March 2025 OK outbreak) | **Contained** — 100 % null statics, provably dropped by `dropna`. But every label count in every doc is inflated ~6 %. |
| F2 | 3 % flare threshold moves ~20 % of positives | **Mitigated** by robustness band — but see D.7 |
| F3 | 2018 NOAA-20 satellite onboarding creates a positive-rate step (0.177 → 0.239) — a **sensor artifact, not a fire trend** | **Documented** — confounds every per-year comparison |
| F4 | Smokehouse hazard percentile was 98th by an undeclared method, 94th by the transparent one | **Fixed** — docs corrected |
| — | 19 of 5,382 dashboard cells have no `ign` value (out-of-state fringe) | **Documented**, intentionally not backfilled |
| — | Portal ingestion mechanism unknown | **Blocked** on external info |

## D.16 What `Focused_Files` does NOT contain (needs `Supporting_files`)

| Missing | Why it matters | Found in |
|---|---|---|
| `data/hwp_params.json` | **breaks FWI normalization silently** | `Supporting_files/data/` |
| `scripts/01`–`12`, `14`–`21`, `23`–`29` | full rebuild pipeline | `Supporting_files/scripts/` |
| `scripts/26`, `27` | **calibration — their own rule #1** | `Supporting_files/scripts/` |
| `scripts/sweep_phase6_credibility.py` | the leak + spatial-holdout tests | `Supporting_files/scripts/` |
| `rev2_improvements/` | all negative results + SHAP diagnostics | `Supporting_files/` |
| `docs/VALIDATION.md`, `SANITY_CHECK.md`, `DEPLOYMENT_STRATEGY.md`, `LIVE_PIPELINE_PLAN.md`, `HANDBOOK.md`, `PLAN.md` | **the roadmap and the honest limitations** | `Supporting_files/docs/` |
| `models/*_realpop_validation.json`, `isotonic_calibration_report.json`, `flare_robustness_band.json` | the deployment evidence | `Supporting_files/models/` |
| honest / ceiling fallback models | HRRR-outage fallback | `Supporting_files/models/` |
| `data/fwi_components_res5.parquet` | needed to re-run FWI variants | `Supporting_files/data/` |
| `data/labels_viirs/viirs_tx_h3.parquet` | raw VIIRS detections | `Supporting_files/data/` |
| v2 dashboard (historical scrubber) | 2024–26 replay view | `Supporting_files/dashboard/` |

**Verdict:** `Focused_Files` is sufficient for *"run the daily forecast and look at the dashboard"* (after copying `hwp_params.json`). It is **not** sufficient for *"validate the work"* — which is precisely what they asked us to do. **We need both, and we have both.**

## D.17 Reproducibility limits they declare honestly (`REPRODUCE.md`)

| Tier | What | Self-contained? |
|---|---|---|
| 1 | View the dashboard | ✅ yes |
| 2 | Retrain the model, rebuild the dashboard, run credibility checks, pull a live forecast | ✅ yes (after `conda env create`) |
| 3 | Rebuild from raw sources | ❌ **no** |

Tier-3 blockers: FIRMS API key (credential) · ~14 GB raw HRRR archive (re-downloadable) · **`gridmet_tx/` ~15 GB — "no script in this pipeline rebuilds this from scratch… the one genuine gap with no re-run path today"** · IgnitionNet project's TX landscape rasters · `geo_cache/`.

> Note the third one: their gridMET gap is a dataset **we already build ourselves** (`run_phase2f_gridmet.py`). That is a concrete thing we can hand back to them.

---

# PART E — What they expect from us

## E.1 Their message, decoded

> *"I think next steps would be validating and checking this work, then the end-to-end pipeline would be after."*

Two deliverables, in order:
1. **Validate and check** — Parts A, C, D of this document.
2. **End-to-end pipeline** — after validation, not before.

> *"Focused Files is my organized work and I hope it contains the items the team needs. If not, the content the team would need is in supporting files."*

They are uncertain whether `Focused_Files` is complete. **Answer: nearly — three specific gaps (D.1, D.2, D.3), all fillable from `Supporting_files`.** Tell them precisely which, so the next handoff is clean.

## E.2 The four questions they addressed to us BY NAME

From `LIVE_PIPELINE_PLAN.md` §4, *"IgnitionNet collaboration — open threads worth raising with them"*:

### Q1. *"Their own docs conflict on Texas's H3 resolution (res-7 vs res-8)"*

**They are right. This is a real, unresolved conflict in our repo:**

| Our source | Says |
|---|---|
| Root `README.md` architecture diagram | *"H3 Resolution-8 Hexagonal Grid (Texas), ~317,142 unique cells (~0.87 km across, ~0.82 km²)"* |
| Root `README.md` Group 5 feature table | *"`centroid_lat`, `centroid_lon` — **H3-7** cell centroid (WGS84)"* |
| `V2/phase2/run_phase2b.py` | *"H3-**R7** grid construction"* |
| `TX/MODEL_TRAINING_REPORT_TX.md` | *"H3-8 hexagonal cell… ~0.74 km²"* |

Note even our own two res-8 area figures disagree (0.82 km² vs 0.74 km²). `TX/check_resolution.py` exists for exactly this and **has not been run against the current file.**

**Action: run `TX/check_resolution.py`, settle the number, fix every doc. Do this before replying.**

### Q2. *"They don't appear to have a spatial-holdout / label-shuffle credibility check — more important for them than for us, since they use raw `centroid_lat`/`centroid_lon` as direct model features"*

**Correct on both counts.** We have neither test, and we *do* use raw coordinates (12.2 % of combined gain). With 1.19 rows per cell (C.2), we cannot currently rule out that our model is interpolating a spatial lookup table.

**Action: run both. Their `scripts/sweep_phase6_credibility.py` is the reference implementation. This is cheap and it is the highest-value single addition to our own credibility.**

### Q3. *"Their entire pipeline is a hindcast/backtest system… no live-forecast-pulling script found. TDIS's script 13 is the piece to share/adapt, not rebuild from scratch"*

**Correct — and this is an offer of code. Take it.** `13_model_forecast_day.py` is 150 lines: Herbie pull → cKDTree nearest-grid-point join → feature assembly → `predict_proba` → calibrate → res-5 aggregate → JSON. Adapting it to our 42-feature 6-hourly model is days of work, not weeks.

### Q4. *"Both teams independently found HRRR sub-daily data added little over daily gridMET+trailing-stats — worth confirming this is a genuinely shared, cross-validated finding"*

**Confirmed from our side.** Our evidence:

| Our HRRR model | ΔAUROC vs best gridMET | ΔAUPR |
|---|---|---|
| Borrowed config, 42 feat | −0.0005 | −0.0011 |
| Independently tuned (21 trials), 42 feat | 0.0000 | −0.0022 |
| Flags dropped, 40 feat | −0.0014 | −0.0049 |

An independent 21-trial sweep on the HRRR feature set converged to the same place, ruling out under-tuning. HRRR gain share 7.4–7.8 %.

Their evidence: `hrrr_gust` +0.000 (step5), 5-day trailing +0.000 (step5), explicit interactions +0.000 (step6) — all within ±0.0002.

**Two independent teams, different labels, different time resolutions, same negative result. This is a genuine cross-validated finding and belongs in any paper either team writes.**

## E.3 What they need for the end-to-end pipeline (`LIVE_PIPELINE_PLAN.md` §2)

| # | Component | Status | Blocker |
|---|---|---|---|
| 1 | **Daily cron** — script 13 at 24 h + 48 h leads | not built | none — buildable today |
| 2 | **Retry + alerting** — *"a missed day must alert, not just log"* | not built | none |
| 3 | **Dashboard data-loading rearchitecture** — standalone HTMLs bake JSON in at build time; live needs HTTP fetch. At res-8 (1.7 M cells, 48 MB/snapshot parquet, 137 MB as JSON) it cannot be one blob — needs tiling or a query API | not built | design decision |
| 4 | **Portal push** to `portal.cloud.tdis.io` | **BLOCKED** | ingestion mechanism unknown (API? SFTP? bucket? admin upload?), auth unknown, readiness unknown |
| 5 | **Monthly drift monitoring** — score last month vs incoming VIIRS; alarm if lift → 2× or ECE > 0.05 | not built | none |
| 6 | **Prospective shadow validation** — 4–8 weeks live, scored against incoming VIIRS | **their #1 ranked task** | *"costs only patience"* — start now |

> Their stated order: **fix model skill → build scheduling + res-8 serving locally → get portal details → wire the push.** Explicitly: *"Don't build the portal connector first against unknowns."*

## E.4 Their ranked remaining tasks (`DEPLOYMENT_STRATEGY.md` §4)

1. **Prospective shadow validation** — all current numbers are replays
2. Hyperparameter sweep — done (step30, promoted)
3. Swap historical Ignition layer to the calibrated operational replay — data already computed
4. Deployment-population threshold sweep on the calibrated scale
5. Cron + alerting + portal push — **blocked**
6. VIIRS `type`-flag flare cross-check — done (`audit_viirs_typeflag.py`)
7. Wind-driven WHEN gap — carry as a known limitation

## E.5 The unstated expectation

`DEPLOYMENT_STRATEGY.md` §3:

> *"Hyperparameter tuning… **Never performed on any TDIS model** — all use one inherited fixed config (400 trees, depth 6, lr 0.05, mcw 20). Deferred during rev2 because **IgnitionNet's real 21-trial search bought only ~0.005 AUROC**. Now being run once, cleanly… **IgnitionNet's tuned optimum was depth 9 vs. our 6**, and insufficient depth was the working hypothesis for why the wind × dryness interaction never got learned."*

They ran their sweep **because of our result**, adopted **our depth-9 optimum**, and it became the served model. `LIVE_PIPELINE_PLAN.md` §3 also lists as their top model improvements: *"rolling multi-day weather trend features — IgnitionNet's own feature importance shows this matters; TDIS has none of it"* and *"a burnable/fuel-load equivalent — IgnitionNet's single best feature (19.8 % of gain)."*

**They are treating us as technical peers and building on our results. The reply they expect is a peer review, not a status update.**

---

# PART F — Our action plan

## F.1 Immediate — before replying (this week)

| # | Action | Why | Effort |
|---|---|---|---|
| 1 | **Run `TX/check_resolution.py`**; settle res-7 vs res-8; fix all four conflicting doc locations | They called it out by name. Everything we send is undermined until fixed. | hours |
| 2 | **Run the label-shuffle control** on our TX model | Their Q2. Proves no pipeline leak. | hours |
| 3 | **Run the spatial-holdout test** (20 % of cells never trained) | Their Q2. With 1.19 rows/cell + raw lat/lon, this is our biggest exposure. | hours |
| 4 | **Send the validation memo** — Parts A, C, D of this document | This is literally what they asked for | — |

## F.2 High-value, adopt from them (next 2–4 weeks)

| # | Action | Why |
|---|---|---|
| 5 | **Score our model on the real deployment population** — every TX cell × every day, not the 1:10 sample | The single biggest hole in our evidence. Without it our 86 % is unanchored. |
| 6 | **Fit an isotonic calibrator** with a strict temporal holdout | Their ECE 0.327 → 0.0018. Post-hoc, no retrain, proven. |
| 7 | **Audit our labels for industrial flares** | They found 27 % of positives were gas flares. We use FPA-FOD (human-reported, so probably cleaner) — but we have never checked, and if we ever add VIIRS we will inherit the problem. |
| 8 | **Add `powerline_dist_km`** (HIFLD, public) | +0.0050 AUC-PR for them; targets the Smokehouse/Windy Deuce power-line mechanism |
| 9 | **Add `road_dist_km`, `ecoregion_id`, `elevation_m`, `slope_deg`, `aspect_deg`** | `ecoregion_id` is their #1 feature (14.3 %); `elevation_m` #2 (10.5 %). We have none of these and they may replace what raw lat/lon is currently doing. |
| 10 | **Adopt the rank-for-color / calibrated-for-numbers UI discipline** | Prevents anyone reading a raw score as a probability |
| 11 | **Evaluate region-week aggregation** | Their `aggregation_recall_experiment.json`: P 0.489 / R 0.780 at res-4 weekly vs P 0.033 / R 0.886 at res-5 daily |

## F.3 Do NOT do — settled by their evidence

| ❌ | Evidence |
|---|---|
| Multiply model output × any fire-weather index | −18 % to −61 % real-population AUC-PR (step29) |
| Add monotone wind constraints | −0.0003 to −0.0013 |
| Expect gust / trailing-weather / interaction features to help *their* model | flat within ±0.0002 across three experiments |
| Serve any "ceiling" model using same-day observed weather | unknowable at forecast time by construction |
| Push further on HRRR sub-daily for **our** model | our own −0.0005 AUROC + their flat results = two independent confirmations |

## F.4 Then — the pipeline

| # | Action | Blocker |
|---|---|---|
| 12 | Adapt their `13_model_forecast_day.py` to our model | none — they offered it |
| 13 | Daily cron, 24 h + 48 h, with retry + alerting | none |
| 14 | **Start prospective shadow validation** — both models, live, 4–8 weeks, scored against incoming VIIRS | none — start immediately, it only costs time |
| 15 | Dashboard live data-loading (tiling or query API) | design decision |
| 16 | Portal push | **blocked — need ingestion details from the portal owner** |
| 17 | Monthly drift monitoring (lift + ECE) | after 13 |

## F.5 What we can give back to them

| # | What | Their gap |
|---|---|---|
| 1 | **A gridMET downloader** (`V2/phase2/run_phase2f_gridmet.py`) | `REPRODUCE.md`: *"No script in this pipeline rebuilds this from scratch… the one genuine gap with no re-run path today"* |
| 2 | **`burnable` fuel-mask derivation** | Their `LIVE_PIPELINE_PLAN.md` §3 item 3 asks for exactly this. Our #1 feature at 19.8 % gain. |
| 3 | **5-day trailing weather feature code** | Their §3 item 2. (Flat for them, but they want to try it properly.) |
| 4 | **Our HRRR negative result, with the independent tuning sweep** | Their Q4 — confirms a shared finding |
| 5 | **A diagnosis of why their `flep4`/`cfl` are dead (importance 0.000)** | We get 9.5 % of gain from `cfl` — likely a data-prep difference on their side |
| 6 | **Our leakage case study** (`fire_count` / `has_fire_history`, AUROC 0.990 → 0.857) | Mirrors their flare finding; good joint material for a paper |
| 7 | **The California pipeline** | They are Texas-only; transferability is an open question for both |

---

# PART G — Draft reply to the TDIS team

> Thanks — both packages came through and we've been through them end to end.
>
> **Verification.** We reloaded `tdis_forecast_hrrr_filtered.json` against your own parquets and reproduced your headline exactly: test AUC-PR **0.4941**, AUROC **0.7402**, best F1 **0.4958** on 891,983 rows at a 0.2341 positive rate. Your numbers are honest and independently checkable, which is the first thing "validate this work" should mean.
>
> **Three packaging gaps in `Focused_Files`** (all fillable from `Supporting_files`):
> 1. `data/hwp_params.json` is missing — `fwi_config._load_hwp_params()` falls back silently to `ref=60.0` with NOAA coefficients, so the Wildfire Risk and Fire Weather tabs regenerate with wrong normalization and no error.
> 2. `tdis_forecast_hrrr_filtered_meta.json` is stale — lists 20 features and the step30 metrics; the shipped model has 22 features (`powerline_dist_km`, `hrrr_mstav`) and the step34 metrics.
> 3. No calibration script (`26`/`27`) is included, and `step34_mstav_retrain.py` does not refit the calibrator — so your own "model and calibrator are a pair" rule can't be followed from that folder. Related: we couldn't find *any* script that produced the served step34 calibrator; `promote_d9.py` only covers the step30 model.
>
> **Ten methodology notes attached**, the ones we'd flag hardest being: three conflicting split definitions (the parquet's `split` column, `METHODOLOGY.md`, and what `step34` actually runs); 2022 is silently dropped and scores *worse* than the later test years (AUROC 0.676 vs 0.740) which nobody has explained; the calibrator is fit at res-5 but applied at res-8 and then averaged back to res-5; and `flare_robustness_band.json`'s own verdict string says conclusions *may* hinge on the 3 % cutoff while `CLASSIFICATION_METRICS.md` §7c says they don't.
>
> **On your four collaboration threads:**
> - **H3 resolution** — you're right, our docs genuinely conflict (res-7 in two places, res-8 in two others). We're running `check_resolution.py` and fixing it.
> - **Credibility checks** — correct, we have neither. We're running the label-shuffle and spatial-holdout controls now, using your `sweep_phase6_credibility.py` as the reference. Fair catch given we use raw lat/lon.
> - **Script 13** — yes please, we'd rather adapt than rebuild.
> - **HRRR adds little** — **confirmed independently.** Our 42-feature HRRR model scores −0.0005 AUROC / −0.0011 AUPR against our best gridMET model, and a separate 21-trial sweep on the HRRR feature set converged to the same place, which rules out under-tuning. Two teams, different labels, different time resolution, same negative result — worth writing up jointly.
>
> **Things we can hand back:** a gridMET downloader (your `REPRODUCE.md` calls that your one genuine no-rebuild-path gap), our `burnable` fuel-mask derivation (your §3 item 3 — it's our top feature at 19.8 % of gain), and our trailing-weather code. Also worth a look: your `flep4` and `cfl` have importance exactly 0.000 in every result JSON, while we get 9.5 % of gain from `cfl` — that smells like a join/null issue rather than a real property of the feature.
>
> **Before pipeline work, two questions:** who owns portal ingestion at `portal.cloud.tdis.io`, and can we start the 4–8 week prospective shadow validation immediately in parallel? It's your own #1 ranked task and it only costs calendar time.

---

# Appendix 1 — Every parquet in the handoff

| File | Rows | Cols | Parquet | Folder |
|---|---|---|---|---|
| `data/tdis_train_daily_hrrr.parquet` | 3,595,513 | 33 | 140.8 MB | Focused |
| `data/static_features/tx_static_master.parquet` | 1,708,940 | 15 | 75.8 MB | Focused |
| `data/static_features/powerline_dist_km.parquet` | 1,708,940 | 2 | 23.7 MB | Focused |
| `New_Training817_moredata/mstav_feature.parquet` | 2,007,436 | 3 | 16.8 MB | Focused |
| `data/labels_fused/ignitions_daily_tx.parquet` | 960,054 | 7 | 3.9 MB | Focused |
| `data/labels_fused/flare_cells_v2.parquet` | 538 | 1 | 5 KB | Focused |
| `rev2_improvements/combined_table.parquet` | 3,595,513 | 40 | 156.4 MB | Supporting |
| `data/fwi_components_res5.parquet` | 3,381,192 | 6 | 79.5 MB | Supporting |
| `data/tdis_train_daily_tx_flarefiltered.parquet` | 3,243,435 | 30 | 66.0 MB | Supporting |
| `data/labels_viirs/viirs_tx_h3.parquet` | 1,619,215 | 9 | 36.8 MB | Supporting |
| `models/operational_historical_res5_v2.parquet` | 5,064,462 | 4 | 31.7 MB | Supporting |
| `models/operational_historical_res5.parquet` | 5,064,462 | 3 | 31.5 MB | Supporting |
| `models/tuned_d9_historical_res5.parquet` | 5,064,462 | 3 | 31.5 MB | Supporting |
| `models/ceiling_historical_res5.parquet` | 3,155,406 | 3 | 15.0 MB | Supporting |
| `models/baseline_forecast_predictions_test_filtered.parquet` | 892,372 | 5 | 8.7 MB | Supporting |
| `New_Training817_moredata/missed_flare_candidates.parquet` | 7 | 4 | 3 KB | Supporting |
| *(+ duplicates of static/powerline/mstav/labels/flare in Supporting)* | | | | |

**Total: 22 files, 862 MB parquet.**

## Key table schemas

**`tdis_train_daily_hrrr.parquet` (33 cols)** — `h3_cell, date, label, year, erc, fm100, vpd, vs, rmax, rmin, tmmx, pr, ecoregion_id, elevation_m, slope_deg, aspect_deg, road_dist_km, avg_burn_prob, whp, cfl, flep4, cbd, cbh, sin_dow, cos_dow, is_weekend, sin_month, cos_month, is_holiday, split, hrrr_tmp, hrrr_vpd, hrrr_wind`

> Note it carries **both** gridMET columns (`erc, fm100, vpd, vs, rmax, rmin, tmmx, pr`) and HRRR columns. The served model uses only the HRRR four; the gridMET eight feed the never-served "ceiling" model.

**`tx_static_master.parquet` (15 cols)** — `h3_cell, lat, lon, avg_burn_prob, whp, flep4, cfl, ecoregion_id, ecoregion_name, elevation_m, slope_deg, aspect_deg, road_dist_km, cbd, cbh`

**`ignitions_daily_tx.parquet` (7 cols)** — `h3_cell, date, label, time_source, cause, max_size_acres, n_sources`

**`mstav_feature.parquet` (3 cols)** — `h3_cell, date, hrrr_mstav` · 2,932 dates (2018-07-16 → 2026-07-29) · 481,822 cells

---

# Appendix 2 — Converting the parquets to CSV

Script: `convert_parquet_to_csv.py` (project root). Streams row-group by row-group, writes each `.csv` **next to its `.parquet` with the same basename**, renders dates as `YYYY-MM-DD`, writes atomically via a `.part` file.

```powershell
cd "c:\Users\Admin\Downloads\Texas ML Wildfire"
python convert_parquet_to_csv.py --force
```

Variants:

```powershell
python convert_parquet_to_csv.py --dry-run              # list, write nothing
python convert_parquet_to_csv.py --gzip --force         # ~1 GB instead of ~5 GB
python convert_parquet_to_csv.py Focused_Files --force  # one folder only
python convert_parquet_to_csv.py --max-mb 100 --force   # skip the 4 biggest
```

**Expect ~4–6 GB of CSV** from 862 MB of parquet.

---

# Appendix 3 — Metric glossary (for non-technical readers)

| Term | Meaning | Why it matters here |
|---|---|---|
| **Base rate** | fraction of rows that are real positives | Ours 9.1 %, theirs 23.4 %, real world 1.9 %. Every metric below depends on it. |
| **Precision** | of everything flagged, what fraction really burned | low = false alarms |
| **Recall** | of everything that burned, what fraction was flagged | low = misses |
| **F1** | harmonic mean of precision and recall at one threshold | shrinks mechanically as positives get rarer |
| **AUC-PR** | threshold-free ranking quality on imbalanced data | **base-rate dependent — never compare across populations** |
| **AUROC** | threshold-free ranking quality, less base-rate sensitive | 0.5 = random |
| **Lift** | AUC-PR ÷ base rate | **the fair cross-population comparison** |
| **Calibration / ECE** | does "70 %" really mean 70 %? | ranking and calibration are independent properties |
| **Brier score** | mean squared error of predicted probability | penalizes bad calibration, not just bad ranking |

> **Why accuracy is never reported by either team:** at a 9.1 % positive rate, predicting "no fire" everywhere gives **90.9 % accuracy** and is useless. Both teams use AUPR as the primary metric. Our TX report states this explicitly; so does their `CLASSIFICATION_METRICS.md`.

---

*End of report. Every claim above was verified against the shipped files; every number was either read from their JSONs or measured by re-scoring their model.*
