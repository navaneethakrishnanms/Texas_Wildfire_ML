# TEXAS 32-Feature Model — Full Build, Training & Feature Report

**Location:** `TEXAS/` (previous 30-feature version preserved untouched in `TEXAS_v1_backup/`)
**Date built:** 2026-09-07
**Author environment:** conda env `torch_gpu` (Python, pandas 2.3.3, xgboost 3.2.0, scikit-learn 1.7.2, h3 4.5.0), NVIDIA RTX 3060 GPU (CUDA)

This document explains, step by step, everything that was done to go from the original TDIS-derived 30-feature model to the new 32-feature tuned model: what data was used, what was added and why, exactly how training and tuning were run, and every feature's measured importance.

---

## 1. What changed, in one sentence

We took the existing 30-feature TDIS-based dataset, added **2 new features** (`lat`, `lon` — H3 cell centroid coordinates) that the TX pipeline had already proven important but TDIS's own dataset never included, then ran a proper **21-trial hyperparameter search** (instead of using fixed guessed hyperparameters) and retrained — improving Test AUROC from **0.8249 → 0.8309** and Test AUC-PR from **0.7384 → 0.7484**, while beating the original TDIS-served model by a wide margin (AUROC 0.7402, AUC-PR 0.4941).

---

## 2. Full list of features now used to train (32 total)

| # | Feature | Group | Source |
|---|---|---|---|
| 1 | `elevation_m` | Static topography | TDIS static master |
| 2 | `slope_deg` | Static topography | TDIS static master |
| 3 | `aspect_deg` | Static topography | TDIS static master |
| 4 | `road_dist_km` | Static infrastructure | TDIS static master |
| 5 | `ecoregion_id` | Static geography | TDIS static master |
| 6 | `avg_burn_prob` | Static fuel/hazard (LandFIRE) | TDIS static master |
| 7 | `whp` (wildfire hazard potential) | Static fuel/hazard (LandFIRE) | TDIS static master |
| 8 | `flep4` (flame length exceedance probability) | Static fuel/hazard (LandFIRE) | TDIS static master |
| 9 | `cfl` (canopy fuel load) | Static fuel/hazard (LandFIRE) | TDIS static master |
| 10 | `cbd` (canopy bulk density) | Static fuel/hazard (LandFIRE) | TDIS static master |
| 11 | `cbh` (canopy base height) | Static fuel/hazard (LandFIRE) | TDIS static master |
| 12 | `powerline_dist_km` | Infrastructure / ignition source | TDIS static features |
| 13 | `hrrr_tmp` | HRRR forecast weather | TDIS daily HRRR extract |
| 14 | `hrrr_vpd` | HRRR forecast weather | TDIS daily HRRR extract |
| 15 | `hrrr_wind` | HRRR forecast weather | TDIS daily HRRR extract |
| 16 | `hrrr_mstav` (soil moisture availability) | HRRR forecast weather | TDIS daily HRRR extract |
| 17 | `erc` (energy release component) | Observed gridMET weather | TDIS daily gridMET extract |
| 18 | `fm100` (100-hr fuel moisture) | Observed gridMET weather | TDIS daily gridMET extract |
| 19 | `vpd` (vapor pressure deficit) | Observed gridMET weather | TDIS daily gridMET extract |
| 20 | `vs` (wind speed) | Observed gridMET weather | TDIS daily gridMET extract |
| 21 | `rmax` (max relative humidity) | Observed gridMET weather | TDIS daily gridMET extract |
| 22 | `rmin` (min relative humidity) | Observed gridMET weather | TDIS daily gridMET extract |
| 23 | `tmmx` (max temperature) | Observed gridMET weather | TDIS daily gridMET extract |
| 24 | `pr` (precipitation) | Observed gridMET weather | TDIS daily gridMET extract |
| 25 | `sin_month` | Calendar/seasonality | Derived from `date` |
| 26 | `cos_month` | Calendar/seasonality | Derived from `date` |
| 27 | `sin_dow` (day of week) | Calendar | Derived from `date` |
| 28 | `cos_dow` (day of week) | Calendar | Derived from `date` |
| 29 | `is_weekend` | Calendar | Derived from `date` |
| 30 | `is_holiday` | Calendar | Derived from `date` |
| 31 | **`lat`** ⭐ NEW | H3 cell centroid location | `Supporting_files/.../static_features/tx_static_master.parquet` |
| 32 | **`lon`** ⭐ NEW | H3 cell centroid location | `Supporting_files/.../static_features/tx_static_master.parquet` |

Only 2 columns were added on top of the previous 30-feature set: **`lat`** and **`lon`**. Everything else is unchanged from the prior TEXAS model.

### Features considered but deliberately NOT added (and why)

| Candidate feature | Why it was skipped |
|---|---|
| `burnable` (binary land-cover flag, used in TX pipeline) | Does not exist anywhere in the TDIS-supplied static data (`tx_static_master.parquet` has no such column), and cannot be derived without the original LANDFIRE EVT raster, which isn't available locally. Adding it would mean fabricating data. |
| 5-day rolling gridMET stats (`erc_5D_mean`, `vpd_5D_max`, etc. — used in TX pipeline, ranked highly there) | TX computes these from a *dense* per-cell daily gridMET history. This TDIS dataset is **sparse** — median ~2 rows per `h3_cell` across 13 years (fire days + a limited sample of negative days only, not continuous daily coverage). Computing a "5-day" rolling window over sparse, non-contiguous rows is exactly the bug TX's own team found and had to fix (documented in `TX/missing_data_diagnosis.md`: 84.4% NaN from rolling over row-order instead of true calendar days). Reproducing it here would silently corrupt the feature rather than add real signal. The raw gridMET NetCDF archive needed to compute this correctly is not available locally. |
| `sin_hour`/`cos_hour` (used in TX pipeline) | This dataset is daily-resolution only — there is no `window_hour` column, so hour-of-day isn't a meaningful concept here. |

---

## 3. Step-by-step: what was actually done

### Step 0 — Backed up the existing TEXAS folder
The previous 30-feature model, dataset, and results were moved (not deleted) to `TEXAS_v1_backup/` so nothing is lost and the two versions can be diffed:
```
mv TEXAS TEXAS_v1_backup
mkdir TEXAS
```

### Step 1 — Built the new dataset (`TEXAS/build_dataset_v2.py`)
1. Loaded the existing imputed base dataset: `TEXAS_v1_backup/tdis_train_daily_imputed.parquet` (3,595,513 rows × 35 columns — the same dataset the 30-feature model was trained on; its own construction — cross-source substitution + seasonal-median imputation from the raw TDIS parquet — was done in a prior session and reused here unchanged).
2. Loaded `Supporting_files/Supporting_files/data/static_features/tx_static_master.parquet` (1,708,940 unique `h3_cell` rows), which contains real `lat`/`lon` centroid coordinates supplied by the TDIS team's own static feature layer but never wired into their training pipeline.
3. Left-joined `lat`/`lon` onto the base dataset by `h3_cell`.
   - **Match rate: 3,393,435 / 3,595,513 rows (94.38%)**.
   - 202,078 rows (5.62%) had no match (cells outside the static master's coverage) — these were filled with the dataset's median `lat`/`lon` (31.9153, -97.9036), consistent with the median/mode imputation policy already used to build the base 30-feature dataset.
4. Saved the result: `TEXAS/tdis_train_daily_imputed_v2.parquet` (3,595,513 rows × **37 columns**).

Command:
```bash
"/c/Users/Admin/anaconda3/envs/torch_gpu/python.exe" TEXAS/build_dataset_v2.py
```

### Step 2 — Split the data (identical boundaries to the original model, for a fair comparison)

| Split | Date range | Rows | Positive rate |
|---|---|---|---|
| Train | up to 2023-12-31 | 2,761,329 | 26.58% |
| Validation | 2024-01-01 → 2024-12-31 | 287,649 | 29.39% |
| Test | 2025-01-01 → 2026-07-29 | 460,463 | 30.71% |

**86,072 rows dated after 2026-07-29 were explicitly excluded.** These are the fabricated/placeholder future-dated rows documented in `DATA_QUALITY_REVIEW.md` (label=0, no real weather, extending to 2026-12-31 even though the last real ground-truth label is 2026-07-29). The script logs this exclusion explicitly so it's auditable, rather than silently dropping them.

### Step 3 — Hyperparameter tuning (`TEXAS/tune_train_model_32feat.py`)
Instead of reusing the previous model's fixed, un-searched hyperparameters (`max_depth=9, lr=0.02, subsample=0.8` — guessed/reused from the TX pipeline), we ran a genuine two-stage grid search, identical in design to the search that improved the original TX model (`TX/tune_tx.py`, which took TX from AUROC 0.8637 → 0.8687):

**Stage 1 — tree structure (12 trials):** `max_depth ∈ {6,7,8,9} × min_child_weight ∈ {10,20,30}`, with `learning_rate=0.05, subsample=0.8` held fixed. Each trial trains on TRAIN and is scored on VAL only (TEST is never touched during search).

| Trial | depth | mcw | val AUROC | val AUC-PR |
|---|---|---|---|---|
| d6_mcw10 | 6 | 10 | 0.8274 | 0.7452 |
| d6_mcw20 | 6 | 20 | 0.8261 | 0.7426 |
| d6_mcw30 | 6 | 30 | 0.8268 | 0.7425 |
| d7_mcw10 | 7 | 10 | 0.8272 | 0.7452 |
| d7_mcw20 | 7 | 20 | 0.8275 | 0.7442 |
| d7_mcw30 | 7 | 30 | 0.8280 | 0.7451 |
| d8_mcw10 | 8 | 10 | 0.8285 | 0.7489 |
| d8_mcw20 | 8 | 20 | 0.8293 | 0.7491 |
| d8_mcw30 | 8 | 30 | 0.8284 | 0.7479 |
| d9_mcw10 | 9 | 10 | 0.8282 | 0.7483 |
| d9_mcw20 | 9 | 20 | 0.8287 | 0.7498 |
| **d9_mcw30** | **9** | **30** | **0.8298** | **0.7498** |

Stage 1 winner: `max_depth=9, min_child_weight=30`.

**Stage 2 — learning rate × subsample (9 trials):** with `max_depth=9, min_child_weight=30` now fixed, searched `learning_rate ∈ {0.01, 0.02, 0.05} × subsample ∈ {0.7, 0.8, 0.9}`.

| Trial | lr | subsample | val AUROC | val AUC-PR |
|---|---|---|---|---|
| lr0.01_s0.7 | 0.01 | 0.7 | 0.8286 | 0.7473 |
| lr0.01_s0.8 | 0.01 | 0.8 | 0.8286 | 0.7469 |
| lr0.01_s0.9 | 0.01 | 0.9 | 0.8286 | 0.7463 |
| lr0.02_s0.7 | 0.02 | 0.7 | 0.8293 | 0.7490 |
| lr0.02_s0.8 | 0.02 | 0.8 | 0.8286 | 0.7484 |
| lr0.02_s0.9 | 0.02 | 0.9 | 0.8298 | 0.7501 |
| lr0.05_s0.7 | 0.05 | 0.7 | 0.8283 | 0.7470 |
| **lr0.05_s0.8** | **0.05** | **0.8** | **0.8298** | **0.7498** |
| lr0.05_s0.9 | 0.05 | 0.9 | 0.8286 | 0.7501 |

**Best overall configuration** (tied on val AUROC with `lr0.02_s0.9`, but chosen as the top of the search list): `max_depth=9, min_child_weight=30, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`.

All 21 trials, with round counts and timings, are saved to `TEXAS/tuning_results_32feat.csv`.

### Step 4 — Final model training
Retrained on the full TRAIN split using the winning hyperparameters, with `scale_pos_weight = 2.7616` (class-imbalance correction, computed automatically as `n_negative / n_positive` on the training set), early stopping on VAL (patience 80 rounds, up to 3,000 boosting rounds), GPU-accelerated (`tree_method=hist, device=cuda`).

- Best iteration: **912**
- Train AUC progression: 0.770 → 0.845 (round 200) → 0.878 (round 800) → 0.884 (round 912, best)

### Step 5 — Isotonic probability calibration
Fit an `IsotonicRegression` calibrator on the VAL split's raw predicted probabilities, then applied it to VAL and TEST — matching the calibration approach used in the original 30-feature model, so probabilities remain well-calibrated (not just well-ranked) for downstream risk-map use.

### Step 6 — Full evaluation (raw and calibrated) on Train / Val / Test

| Split | Version | AUC-PR | AUROC | F1 | Precision | Recall | Lift |
|---|---|---|---|---|---|---|---|
| Train | raw | 0.7920 | 0.8836 | 0.6866 | 0.6928 | 0.6806 | 2.98× |
| Val | raw | 0.7499 | 0.8298 | 0.6471 | 0.6609 | 0.6338 | 2.55× |
| **Test** | **raw** | **0.7484** | **0.8309** | **0.6498** | 0.6390 | 0.6610 | 2.44× |
| Val | calibrated | 0.7458 | 0.8301 | 0.6472 | 0.6599 | 0.6349 | 2.54× |
| **Test** | **calibrated** | **0.7435** | **0.8308** | **0.6497** | 0.6250 | 0.6764 | 2.42× |

### Step 7 — Feature importance
Computed XGBoost "gain" importance across all 32 features on the final trained model (full ranked list in Section 4 below). Saved model (`model_32feat_tuned.json`), calibrator (`calibrator_32feat_tuned.joblib`), and full results (`training_results_32feat_tuned.json`).

---

## 4. Every feature ranked by importance (final tuned model)

Ranked by XGBoost gain, most → least important, out of 32 features:

| Rank | Feature | Gain | % of total |
|---|---|---|---|
| 1 | `elevation_m` | 157.43 | 8.72% |
| 2 | `avg_burn_prob` | 122.59 | 6.79% |
| **3** | **`lat`** ⭐ NEW | **109.49** | **6.07%** |
| 4 | `rmin` | 108.91 | 6.04% |
| **5** | **`lon`** ⭐ NEW | **103.28** | **5.72%** |
| 6 | `ecoregion_id` | 95.34 | 5.28% |
| 7 | `sin_month` | 76.65 | 4.25% |
| 8 | `road_dist_km` | 76.27 | 4.23% |
| 9 | `powerline_dist_km` | 73.99 | 4.10% |
| 10 | `rmax` | 70.77 | 3.92% |
| 11 | `hrrr_vpd` | 64.92 | 3.60% |
| 12 | `hrrr_mstav` | 60.51 | 3.35% |
| 13 | `whp` | 55.36 | 3.07% |
| 14 | `cos_month` | 54.07 | 3.00% |
| 15 | `slope_deg` | 49.50 | 2.74% |
| 16 | `vs` | 45.67 | 2.53% |
| 17 | `is_weekend` | 45.53 | 2.52% |
| 18 | `vpd` | 41.94 | 2.32% |
| 19 | `pr` | 40.57 | 2.25% |
| 20 | `is_holiday` | 38.67 | 2.14% |
| 21 | `hrrr_tmp` | 37.28 | 2.07% |
| 22 | `cbh` | 35.79 | 1.98% |
| 23 | `cbd` | 34.88 | 1.93% |
| 24 | `hrrr_wind` | 34.29 | 1.90% |
| 25 | `aspect_deg` | 33.09 | 1.83% |
| 26 | `tmmx` | 31.47 | 1.74% |
| 27 | `erc` | 29.50 | 1.63% |
| 28 | `sin_dow` | 27.45 | 1.52% |
| 29 | `fm100` | 25.88 | 1.43% |
| 30 | `cos_dow` | 23.44 | 1.30% |
| 31 | `flep4` | 0.00 | 0.00% |
| 32 | `cfl` | 0.00 | 0.00% |

**Key observations:**
- **`lat`/`lon` landed at #3 and #5**, contributing a combined **~11.8%** of total model gain — validating that these features, which the TDIS team's own model never used, carry real independent signal. This matches what the separate TX pipeline found for the same geographic concept (TX's `centroid_lat`/`centroid_lon` were its #5-ranked features at ~12% combined).
- `elevation_m` and `avg_burn_prob` remain the top two drivers, consistent with every version of this model built so far (TDIS's own model, our prior 30-feature model, and this one).
- `flep4` and `cfl` are **dead features (0.00% importance)** — this exact same result was found in the TDIS-shared model too, so it's now confirmed a third time (also true in the prior 30-feature version of this model). These two LandFIRE features appear to add nothing once `avg_burn_prob`, `whp`, `cbd`, and `cbh` are already present.

---

## 5. Final comparison: three models side by side

| Metric | TDIS shared model (22 feat, no tuning) | Our 30-feat model (untuned) | **Our 32-feat model (tuned)** |
|---|---|---|---|
| Features | 22 | 30 | **32** |
| Hyperparameter search | None (fixed config) | None (fixed config, reused from TX) | **21-trial two-stage search** |
| Test AUC-PR | 0.4941 | 0.7384 | **0.7484** |
| Test AUROC | 0.7402 | 0.8249 | **0.8309** |
| Test F1 | 0.4958 | 0.6440 | **0.6498** |
| Calibrated | Yes | Yes | Yes |

**Net improvement over the TDIS-served model: +0.2543 AUC-PR, +0.0907 AUROC, +0.1540 F1.**
**Net improvement over our own prior 30-feature model: +0.0100 AUC-PR, +0.0060 AUROC, +0.0058 F1** — a smaller but real gain, attributable specifically to (a) the two new `lat`/`lon` features and (b) proper hyperparameter tuning instead of reused/guessed settings.

---

## 6. How to reproduce this from scratch

```bash
cd "c:/Users/Admin/Downloads/Texas ML Wildfire"

# Step 1 — build the 32-feature dataset (merges lat/lon onto the base imputed data)
"/c/Users/Admin/anaconda3/envs/torch_gpu/python.exe" TEXAS/build_dataset_v2.py

# Step 2 — run the 21-trial hyperparameter search, train final model, calibrate, evaluate
"/c/Users/Admin/anaconda3/envs/torch_gpu/python.exe" TEXAS/tune_train_model_32feat.py
```

Total runtime on an RTX 3060 (GPU/CUDA): **~12 minutes** (710.3 seconds for the tuning+training script; dataset build takes a few seconds).

## 7. All output files in `TEXAS/`

| File | Contents |
|---|---|
| `build_dataset_v2.py` | Script: merges `lat`/`lon` onto the base dataset |
| `tdis_train_daily_imputed_v2.parquet` | The 32-feature dataset (3,595,513 rows × 37 cols) |
| `tune_train_model_32feat.py` | Script: 21-trial tuning search + final training + calibration |
| `model_32feat_tuned.json` | Final trained XGBoost model |
| `calibrator_32feat_tuned.joblib` | Isotonic probability calibrator |
| `tuning_results_32feat.csv` | All 21 tuning trials with hyperparameters, rounds, val scores |
| `training_results_32feat_tuned.json` | Full metrics, feature importances, and 3-way comparison (machine-readable) |
| `tune_train_run.log` | Complete console log of the training run |
| `TEXAS_V2_MODEL_REPORT.md` | This document |

The prior 30-feature version (`README.md`, `MODEL_EVALUATION_METRICS.md`, `SESSION_SUMMARY_27AUG2026.md`, `model_30feat.json`, `calibrator_30feat.joblib`, `training_results_30feat.json`, the base imputed parquet, and raw parquet) is preserved unchanged in `TEXAS_v1_backup/`.
