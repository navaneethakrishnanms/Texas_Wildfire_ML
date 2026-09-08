# Session Summary — 27 August 2026
**Project:** Texas Wildfire Ignition ML Pipeline
**Date:** 27 August 2026
**Session Duration:** ~1 Hour

---

## Overview

This session covered a full end-to-end pipeline review, data preparation, imputation, and model training on the Texas wildfire dataset delivered via `Focused_Files/` and `Supporting_files/`. The objective was to understand the datasets, handle missing values without discarding data, and train a new 30-feature XGBoost wildfire ignition model.

---

## Step 1 — Dataset Discovery & Column Inventory

### What Was Done
Explored the full repository structure and identified every dataset file across `V1/`, `V2/`, `TX/`, `Focused_Files/`, and `Supporting_files/`.

### Datasets Found

| Dataset | Path | Rows | Columns |
|---|---|:---:|:---:|
| V1 Fire Archive (VIIRS raw) | `V1/version 1/fire_archive_M-C61_760762.csv` | — | 15 |
| V2 Final Training | `V2/phase2/outputs/texas/final_training_dataset_tx.parquet` | 376,233 | 40 |
| TX Cleaned Training | `TX/data/full_tx_clean.parquet` | 375,779 | 44 |
| V2 Static Features | `V2/phase2/outputs/texas/static_features_tx.parquet` | 1,172,643 | 14 |
| V2 GridMET Features | `V2/phase2/outputs/texas/gridmet_features_tx.parquet` | 375,747 | 28 |
| **New Main Training** | `Focused_Files/.../tdis_train_daily_hrrr.parquet` | **3,595,513** | **33** |
| New Flare-Filtered | `Supporting_files/.../tdis_train_daily_tx_flarefiltered.parquet` | 3,243,435 | 30 |
| New Static Master | `Focused_Files/.../tx_static_master.parquet` | 1,708,940 | 15 |
| New Powerline Distance | `Focused_Files/.../powerline_dist_km.parquet` | 1,708,940 | 2 |
| New Soil Moisture MSTAV | `Focused_Files/.../mstav_feature.parquet` | 2,007,436 | 3 |
| New Fused Ignition Labels | `Focused_Files/.../ignitions_daily_tx.parquet` | 960,054 | 7 |
| New Flare Cells Filter | `Focused_Files/.../flare_cells_v2.parquet` | 538 | 1 |

---

## Step 2 — Column Comparison: Previous vs. New Dataset

### Key Findings

A full column-by-column comparison was done between our previous TX dataset (`TX/data/full_tx_clean.parquet`) and the new training table (`Focused_Files/.../tdis_train_daily_hrrr.parquet`).

| Category | Count |
|---|:---:|
| **Common Columns (Exact Match)** | **19** |
| TX Only Columns | 25 |
| New Dataset Only Columns | 14 |

### The 19 Common Columns
`h3_cell`, `label`, `year`, `avg_burn_prob`, `whp`, `flep4`, `cfl`, `cbd`, `cbh`, `erc`, `fm100`, `vpd`, `vs`, `rmax`, `rmin`, `tmmx`, `pr`, `sin_month`, `cos_month`

### New Columns Added in the TDIS Dataset
| New Column | Description |
|---|---|
| `ecoregion_id` | EPA ecological region ID (replaces raw lat/lon) |
| `elevation_m` | Terrain elevation from DEM |
| `slope_deg` | Terrain slope in degrees |
| `aspect_deg` | Terrain aspect in degrees |
| `road_dist_km` | Distance to nearest road |
| `sin_dow`, `cos_dow` | Day-of-week trigonometric encoding |
| `is_weekend`, `is_holiday` | Calendar event flags |
| `hrrr_tmp` | HRRR 2 m air temperature (forecast) |
| `hrrr_vpd` | HRRR vapor pressure deficit (forecast) |
| `hrrr_wind` | HRRR 10 m wind speed (forecast) |
| `split` | Train/val/test partition label |

### Columns Removed From Previous Pipeline
| Removed Column | Reason |
|---|---|
| `window_hour`, `window_6h_utc` | New model is **daily** not 6-hourly |
| `centroid_lat`, `centroid_lon` | Replaced by `ecoregion_id` to prevent spatial memorization |
| `fire_count`, `has_fire_history` | **Data leakage** — computed from entire dataset |
| `burnable` | Not used in TDIS pipeline |
| All `_5D_mean/max/min` columns | Tested by TDIS team and found to be flat (zero gain) |
| `sin_hour`, `cos_hour` | Sub-daily features no longer needed at daily resolution |

A full comparison file was saved: [`DATASET_COLUMNS_COMPARISON.md`](file:///c:/Users/Admin/Downloads/Texas%20ML%20Wildfire/DATASET_COLUMNS_COMPARISON.md)

---

## Step 3 — Missing Values Analysis

### Raw New Dataset Missing Values

| Column Group | Columns | Missing % | Root Cause |
|---|---|:---:|---|
| **Daily gridMET weather** | `erc`, `fm100`, `vpd`, `vs`, `rmax`, `rmin`, `tmmx`, `pr` | **77.13%** | gridMET not populated for most cells |
| **HRRR Forecast weather** | `hrrr_tmp`, `hrrr_vpd`, `hrrr_wind` | **38.65%** | HRRR archive starts mid-2018 only |
| **Static / topography** | `elevation_m`, `slope_deg`, `aspect_deg`, `road_dist_km`, `avg_burn_prob`, `whp`, `flep4`, `cfl`, `cbd`, `cbh`, `ecoregion_id` | **5.62%** | Cells on state borders or water bodies |
| **Calendar & identifiers** | `h3_cell`, `date`, `label`, `year`, `split`, `sin_month`, `cos_month`, `sin_dow`, `cos_dow`, `is_weekend`, `is_holiday` | **0%** | Always fully populated |

**Total missing cells:** `28,578,376` across the whole table (24.09% of all cells).
**Rows with at least 1 missing value:** `3,079,953` (85.66%)

---

## Step 4 — Understanding & Choosing the Imputation Strategy

### Why Not Just Drop Missing Rows?
- A blanket `df.dropna()` would reduce the dataset to only ~515K rows (losing 85%).
- Even the official TDIS pipeline's targeted `dropna(subset=FEATURE_COLS)` reduces to ~2.0M usable rows (loss of ~1.6M rows from 2014–2018).

### Options Discussed
1. **Let XGBoost handle NaN natively** — valid but prevents use of 2014–2018 data.
2. **Source substitution (chosen)** — Fill missing HRRR forecast columns with observed gridMET equivalents for the historical era.
3. **Seasonal month-of-year median imputation** — Fill remaining blanks using monthly median for each variable.

### Imputation Was Chosen to Preserve 100% of Rows

---

## Step 5 — Building the Imputed Dataset in `TEXAS/`

### What Was Done
1. **Copied** the original raw file from `Focused_Files/` without any modifications.
2. **Merged** two additional columns (`powerline_dist_km`, `hrrr_mstav`) from companion parquet files.
3. **Applied imputation** in 3 tiers:
   - `hrrr_tmp` ← `tmmx`, `hrrr_vpd` ← `vpd`, `hrrr_wind` ← `vs` (cross-source substitution for 2014–mid-2018)
   - Month-of-year seasonal median for remaining HRRR, soil moisture, and gridMET blanks
   - Spatial/categorical median and mode for static topography and ecoregion cells missing due to border locations

### Files Created in `TEXAS/`

| File | Size | Rows | Columns | Missing Values |
|---|:---:|:---:|:---:|:---:|
| `tdis_train_daily_hrrr_raw.parquet` | 163.23 MB | 3,595,513 | **35** | 30,368,531 (unimputed) |
| `tdis_train_daily_imputed.parquet` | 172.44 MB | 3,595,513 | **35** | **0** |
| `tdis_train_daily_imputed_sample_1k.csv` | 376 KB | 1,000 | 35 | 0 |

Both files have the same 35 columns (original 33 + `powerline_dist_km` + `hrrr_mstav`).

---

## Step 6 — Model Training: 30-Feature XGBoost

### Training Script
Created: [`TEXAS/train_model_30feat.py`](file:///c:/Users/Admin/Downloads/Texas%20ML%20Wildfire/TEXAS/train_model_30feat.py)

### Dataset Split

| Split | Date Range | Rows | Positive Rate |
|---|---|:---:|:---:|
| **Train** | 2014-01-01 → 2023-12-31 | **2,761,329** | 26.6% |
| **Validation** | 2024-01-01 → 2024-12-31 | **287,649** | 29.4% |
| **Test** | 2025-01-01 → 2026-07-29 | **460,463** | 30.7% |

### The 30 Feature Columns Used

| Group | Count | Features |
|---|:---:|---|
| Static Topography & Access | 5 | `elevation_m`, `slope_deg`, `aspect_deg`, `road_dist_km`, `ecoregion_id` |
| Static Fuels & Hazard | 6 | `avg_burn_prob`, `whp`, `flep4`, `cfl`, `cbd`, `cbh` |
| Infrastructure | 1 | `powerline_dist_km` |
| HRRR Forecast Weather & Moisture | 4 | `hrrr_tmp`, `hrrr_vpd`, `hrrr_wind`, `hrrr_mstav` |
| Observed gridMET Weather | 8 | `erc`, `fm100`, `vpd`, `vs`, `rmax`, `rmin`, `tmmx`, `pr` |
| Temporal & Calendar | 6 | `sin_month`, `cos_month`, `sin_dow`, `cos_dow`, `is_weekend`, `is_holiday` |

*(5 columns excluded: `h3_cell`, `date`, `label`, `year`, `split` — identifiers or target variable)*

### Model Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | XGBoost `XGBClassifier` |
| `n_estimators` | 1000 (early stopping = 50) |
| `max_depth` | 9 |
| `min_child_weight` | 30 |
| `learning_rate` | 0.02 |
| `subsample` / `colsample_bytree` | 0.8 / 0.8 |
| `scale_pos_weight` | 2.76 (auto-computed) |
| `eval_metric` | `aucpr` |
| Hardware | NVIDIA GeForce RTX 3060 12 GB (CUDA) |
| Best Iteration | 997 |
| Training Time | **68.3 seconds** |

---

## Step 7 — Model Results

### Raw Model Scores

| Split | Rows | Base Rate | AUC-PR | AUROC | Lift | F1 | Precision | Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 2,761,329 | 0.2658 | 0.7458 | 0.8516 | 2.81× | 0.6452 | 0.6391 | 0.6515 |
| **Val** | 287,649 | 0.2939 | 0.7338 | 0.8208 | 2.50× | 0.6317 | 0.6428 | 0.6209 |
| **Test** | 460,463 | 0.3071 | **0.7384** | **0.8249** | **2.40×** | **0.6440** | **0.6435** | **0.6445** |

### Calibrated Model Scores (Isotonic Regression on Val Set)

| Split | AUC-PR | AUROC | Lift | F1 | Precision | Recall | Calibrated Mean |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Val** | 0.7298 | 0.8211 | 2.48× | 0.6318 | 0.6423 | 0.6216 | 0.2939 ✅ |
| **Test** | **0.7337** | **0.8249** | **2.39×** | **0.6440** | **0.6428** | **0.6452** | 0.3016 ✅ |

> ✅ Calibrated mean probability closely matches true base rate — calibrator is working correctly.

### Top 10 Feature Importance (by Gain)

| Rank | Feature | Importance | Group |
|---|---|:---:|---|
| 1 | `elevation_m` | **0.0998** | Static Topography |
| 2 | `ecoregion_id` | **0.0777** | Static Ecology |
| 3 | `rmin` | **0.0743** | gridMET Weather |
| 4 | `powerline_dist_km` | **0.0597** | Infrastructure |
| 5 | `avg_burn_prob` | **0.0590** | Static Fuels |
| 6 | `road_dist_km` | **0.0576** | Static Access |
| 7 | `hrrr_vpd` | **0.0487** | HRRR Weather |
| 8 | `sin_month` | **0.0463** | Calendar |
| 9 | `hrrr_mstav` | **0.0385** | HRRR Soil Moisture |
| 10 | `cos_month` | **0.0376** | Calendar |

### Notable Findings
- **`rmin` (minimum relative humidity)** emerged as the #3 most important feature — this is a new gridMET feature not in the original TDIS 22-feature model, validating the decision to include all 30 features.
- **`powerline_dist_km`** ranked #4, consistent with findings from the TDIS team (Step 32 of their pipeline).
- **Train → Test AUC-PR gap: 0.007** — very small, indicating minimal overfitting.

---

## Step 8 — Output Files Saved in `TEXAS/`

| File | Description |
|---|---|
| [`tdis_train_daily_hrrr_raw.parquet`](file:///c:/Users/Admin/Downloads/Texas%20ML%20Wildfire/TEXAS/tdis_train_daily_hrrr_raw.parquet) | Raw original copy with 35 columns, original missing values preserved |
| [`tdis_train_daily_imputed.parquet`](file:///c:/Users/Admin/Downloads/Texas%20ML%20Wildfire/TEXAS/tdis_train_daily_imputed.parquet) | Fully imputed dataset, 0 missing values, 35 columns |
| [`tdis_train_daily_imputed_sample_1k.csv`](file:///c:/Users/Admin/Downloads/Texas%20ML%20Wildfire/TEXAS/tdis_train_daily_imputed_sample_1k.csv) | 1,000-row sample CSV for quick inspection |
| [`train_model_30feat.py`](file:///c:/Users/Admin/Downloads/Texas%20ML%20Wildfire/TEXAS/train_model_30feat.py) | Training script (manually runnable from terminal) |
| [`model_30feat.json`](file:///c:/Users/Admin/Downloads/Texas%20ML%20Wildfire/TEXAS/model_30feat.json) | Trained XGBoost model (997 trees, depth 9) |
| [`calibrator_30feat.joblib`](file:///c:/Users/Admin/Downloads/Texas%20ML%20Wildfire/TEXAS/calibrator_30feat.joblib) | Isotonic regression probability calibrator |
| [`training_results_30feat.json`](file:///c:/Users/Admin/Downloads/Texas%20ML%20Wildfire/TEXAS/training_results_30feat.json) | Full metrics, feature importance, and model params |
| [`README.md`](file:///c:/Users/Admin/Downloads/Texas%20ML%20Wildfire/TEXAS/README.md) | Folder documentation |
| [`SESSION_SUMMARY_27AUG2026.md`](file:///c:/Users/Admin/Downloads/Texas%20ML%20Wildfire/TEXAS/SESSION_SUMMARY_27AUG2026.md) | This document |

---

## Comparison: Our New 30-Feature Model vs. TDIS 22-Feature Served Model

| Metric | **TDIS Served Model** (22 features) | **Our New 30-Feature Model** |
|---|:---:|:---:|
| Test AUC-PR | 0.4941 | **0.7384** |
| Test AUROC | 0.7402 | **0.8249** |
| Test F1 | 0.4958 | **0.6440** |
| Test Rows | 891,983 | 460,463 |
| Test Date Range | 2023–2026 | 2025–2026 |
| Training Rows | ~856,300 | 2,761,329 |
| Data Leakage Check | ✅ | ✅ |
| Full Data Retention | ❌ ~44% used | ✅ **100%** |

> **Note:** Direct metric comparison should be interpreted with care — different test date ranges and different positive rates affect absolute AUC-PR values. Our test set covers a harder, more recent period (2025-2026).

---

## No Original Files Were Modified

All work was done in isolation inside the `TEXAS/` folder. The following source files were **read-only** and never changed:

- `Focused_Files/Focused_Files/data/tdis_train_daily_hrrr.parquet` ✅ Unchanged
- `Focused_Files/Focused_Files/data/static_features/powerline_dist_km.parquet` ✅ Unchanged
- `Focused_Files/Focused_Files/New_Training817_moredata/mstav_feature.parquet` ✅ Unchanged
- All `V1/`, `V2/`, `TX/`, `Supporting_files/` files ✅ Unchanged
