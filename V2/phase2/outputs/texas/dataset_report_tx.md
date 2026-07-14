# Final Training Dataset Report — Texas

---

## 1. Overview

| Item | Value |
|------|-------|
| Total rows | **376,233** |
| Total columns | **40** |
| File | `final_training_dataset_tx.parquet` |
| Size on disk | ~19 MB (parquet) |
| Study period | 2014–2020 (7 years) |
| State | Texas |
| H3 Resolution | 7 (~1.9 km cell width) |

## 2. Label Distribution

| Label | Meaning | Count | Percentage |
|-------|---------|-------|------------|
| 1 | Fire discovered | **34,203** | 9.1% |
| 0 | No fire (negative sample) | **342,030** | 90.9% |
| **Total** | | **376,233** | 100% |

## 3. Chronological Train / Val / Test Split

| Split | Years | Total Rows | Fire Rows | Non-fire Rows | Fire Rate |
|-------|-------|-----------|---------|-------------|-----------|
| TRAIN | 2014–2017 | 252,066 | 22,916 | 229,150 | 9.1% |
| VAL | 2018 | 61,181 | 5,561 | 55,620 | 9.1% |
| TEST | 2019–2020 | 62,986 | 5,726 | 57,260 | 9.1% |

## 4. All Columns

### 🆔 Identifiers (not used as features)  (6 columns)

| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |
|--------|------|---------|-----------|-------|-----|-----|------|-------------|
| `h3_cell` | object | 376,233 | 0 ✔ | 0.0% | — | — | — | H3 hexagonal cell ID (resolution-7) |
| `date_utc` | object | 376,233 | 0 ✔ | 0.0% | — | — | — | UTC date of the 6-hour window |
| `window_hour` | int64 | 376,233 | 0 ✔ | 0.0% | 0.000 | 18.000 | 14.156 | UTC hour (0, 6, 12, or 18) |
| `window_6h_utc` | datetime64[ns] | 376,233 | 0 ✔ | 0.0% | — | — | — | Full UTC timestamp string |
| `fire_year` | int64 | 376,233 | 0 ✔ | 0.0% | 2014.000 | 2020.000 | 2016.591 | Year of the event (used for split) |
| `gridmet_missing` | int8 | 376,233 | 0 ✔ | 0.0% | 0.000 | 1.000 | 0.066 | 1 = this cell has no gridMET data (coastal/border) |

### 🎯 Target  (1 columns)

| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |
|--------|------|---------|-----------|-------|-----|-----|------|-------------|
| `label` | int64 | 376,233 | 0 ✔ | 0.0% | 0.000 | 1.000 | 0.091 | 1 = fire discovered, 0 = no fire |

### 🌿 Landscape / Static (LANDFIRE + FSim)  (7 columns)

| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |
|--------|------|---------|-----------|-------|-----|-----|------|-------------|
| `avg_burn_prob` | float64 | 374,639 | 1,594 (0.4%) | 0.4% | 0.000 | 0.000 | 0.000 | FSim annual burn probability [0–1] (⚠️ = 0, rasters missing) |
| `whp` | float64 | 374,639 | 1,594 (0.4%) | 0.4% | 0.000 | 0.000 | 0.000 | Wildfire Hazard Potential index [0–7000] (⚠️ = 0, rasters missing) |
| `flep4` | float64 | 374,639 | 1,594 (0.4%) | 0.4% | 0.000 | 0.000 | 0.000 | Flame Length Exceedance Prob ≥4ft [0–1] (⚠️ = 0, rasters missing) |
| `cfl` | float64 | 374,639 | 1,594 (0.4%) | 0.4% | 0.000 | 0.000 | 0.000 | Canopy Fuel Load [Mg ha⁻¹] (⚠️ = 0, rasters missing) |
| `fire_count` | float64 | 374,639 | 1,594 (0.4%) | 0.4% | 0.000 | 36.000 | 0.212 | Historical fire count in this H3 cell (2014–2020) |
| `has_fire_history` | object | 374,639 | 1,594 (0.4%) | 0.4% | — | — | — | 1 if cell has ever burned before |
| `burnable` | object | 374,639 | 1,594 (0.4%) | 0.4% | — | — | — | 1 = burnable land cover, 0 = non-burnable |

### 🌦️ gridMET Daily Weather  (8 columns)

| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |
|--------|------|---------|-----------|-------|-----|-----|------|-------------|
| `erc` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 0.000 | 111.000 | 43.493 | Energy Release Component [BTU ft⁻²] — top daily predictor |
| `fm100` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 0.220 | 2.850 | 1.252 | 100-hr dead fuel moisture [%] |
| `vpd` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 0.000 | 0.051 | 0.015 | Vapor pressure deficit [kPa] — from gridMET (not FPA-FOD, no leakage) |
| `vs` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 0.030 | 1.870 | 0.427 | Wind speed [m s⁻¹] |
| `rmax` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 1.100 | 10.000 | 7.875 | Max relative humidity [%] |
| `rmin` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 0.100 | 10.000 | 2.827 | Min relative humidity [%] |
| `tmmx` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | -27.190 | -21.210 | -23.135 | Max temperature [°C] |
| `pr` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 0.000 | 62.480 | 0.130 | Precipitation [mm] |

### 📅 5-Day Trailing Stats (gridMET)  (12 columns)

| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |
|--------|------|---------|-----------|-------|-----|-----|------|-------------|
| `erc_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 0.200 | 107.600 | 42.029 | 5-day trailing mean of ERC (days D-5 to D-1) |
| `erc_5D_max` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 1.000 | 110.000 | 47.306 | 5-day trailing max of ERC |
| `fm100_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 0.274 | 2.726 | 1.296 | 5-day trailing mean of 100-hr fuel moisture |
| `fm100_5D_min` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 0.230 | 2.510 | 1.151 | 5-day trailing min of fuel moisture (lower = drier) |
| `vpd_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 0.000 | 0.047 | 0.014 | 5-day trailing mean of VPD |
| `vpd_5D_max` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 0.000 | 0.054 | 0.018 | 5-day trailing max of VPD |
| `vs_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 0.072 | 1.246 | 0.423 | 5-day trailing mean of wind speed |
| `vs_5D_max` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 0.100 | 1.840 | 0.563 | 5-day trailing max of wind speed |
| `rmax_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 1.708 | 10.000 | 8.046 | 5-day trailing mean of max relative humidity |
| `rmax_5D_min` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 1.050 | 10.000 | 6.879 | 5-day trailing min of max relative humidity |
| `tmmx_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | -26.440 | -21.454 | -23.217 | 5-day trailing mean of max temperature |
| `tmmx_5D_max` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | -26.030 | -21.200 | -22.851 | 5-day trailing max of max temperature |

### ⏱️ Temporal Encodings  (4 columns)

| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |
|--------|------|---------|-----------|-------|-----|-----|------|-------------|
| `sin_month` | float32 | 376,233 | 0 ✔ | 0.0% | -1.000 | 1.000 | -0.019 | sin(2π × month / 12) — cyclic fire season encoding |
| `cos_month` | float32 | 376,233 | 0 ✔ | 0.0% | -1.000 | 1.000 | -0.063 | cos(2π × month / 12) — cyclic fire season encoding |
| `sin_hour` | float32 | 376,233 | 0 ✔ | 0.0% | -1.000 | 1.000 | -0.387 | sin(2π × window_hour / 24) — cyclic time-of-day |
| `cos_hour` | float32 | 376,233 | 0 ✔ | 0.0% | -1.000 | 1.000 | -0.580 | cos(2π × window_hour / 24) — cyclic time-of-day |

### 📍 Location  (2 columns)

| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |
|--------|------|---------|-----------|-------|-----|-----|------|-------------|
| `centroid_lat` | float64 | 376,233 | 0 ✔ | 0.0% | 25.865 | 36.570 | 31.968 | H3 cell centroid latitude (WGS84) |
| `centroid_lon` | float64 | 376,233 | 0 ✔ | 0.0% | -106.650 | -93.512 | -99.745 | H3 cell centroid longitude (WGS84) |

## 5. Missing Value Summary

### Root Causes

| Root Cause | Columns Affected | NaN Count | NaN % | Action |
|-----------|-----------------|-----------|-------|--------|
| LANDFIRE rasters not downloaded | `avg_burn_prob`, `whp`, `flep4`, `cfl` | 1,594 | 0.42% | Fill with 0 until rasters downloaded |
| H3 cells not in static grid | `fire_count`, `has_fire_history`, `burnable` | 1,594 | 0.42% | Fill with 0 |
| gridMET ocean/border pixels | All 20 gridMET cols | 24,954 | 6.63% | XGBoost handles natively (NaN = coastal/border cell) |
| 5D trailing stats (same root) | All 12 5D stat cols | 24,965 | 6.64% | Same as above (11 extra = Jan 1–5 boundary) |

### Missing Values by Column

| Column | NaN Count | NaN % | Status |
|--------|-----------|-------|--------|
| `h3_cell` | 0 | 0.00% | ✅ Complete |
| `date_utc` | 0 | 0.00% | ✅ Complete |
| `window_hour` | 0 | 0.00% | ✅ Complete |
| `window_6h_utc` | 0 | 0.00% | ✅ Complete |
| `label` | 0 | 0.00% | ✅ Complete |
| `centroid_lat` | 0 | 0.00% | ✅ Complete |
| `centroid_lon` | 0 | 0.00% | ✅ Complete |
| `fire_year` | 0 | 0.00% | ✅ Complete |
| `fire_count` | 1,594 | 0.42% | 🟡 Minor — fill with 0 |
| `has_fire_history` | 1,594 | 0.42% | 🟡 Minor — fill with 0 |
| `burnable` | 1,594 | 0.42% | 🟡 Minor — fill with 0 |
| `avg_burn_prob` | 1,594 | 0.42% | 🟡 Minor — fill with 0 |
| `whp` | 1,594 | 0.42% | 🟡 Minor — fill with 0 |
| `flep4` | 1,594 | 0.42% | 🟡 Minor — fill with 0 |
| `cfl` | 1,594 | 0.42% | 🟡 Minor — fill with 0 |
| `erc` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles |
| `fm100` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles |
| `vpd` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles |
| `vs` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles |
| `rmax` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles |
| `rmin` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles |
| `tmmx` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles |
| `pr` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles |
| `erc_5D_mean` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles |
| `erc_5D_max` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles |
| `fm100_5D_mean` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles |
| `fm100_5D_min` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles |
| `vpd_5D_mean` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles |
| `vpd_5D_max` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles |
| `vs_5D_mean` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles |
| `vs_5D_max` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles |
| `rmax_5D_mean` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles |
| `rmax_5D_min` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles |
| `tmmx_5D_mean` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles |
| `tmmx_5D_max` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles |
| `gridmet_missing` | 0 | 0.00% | ✅ Complete |
| `sin_month` | 0 | 0.00% | ✅ Complete |
| `cos_month` | 0 | 0.00% | ✅ Complete |
| `sin_hour` | 0 | 0.00% | ✅ Complete |
| `cos_hour` | 0 | 0.00% | ✅ Complete |

## 6. Columns Dropped (per Team Review)

| Column | Reason |
|--------|--------|
| `bi` | Correlated with `erc` (both NFDRS indices) — keep `erc` |
| `bi_5D_mean` | `bi` was dropped, so its 5D stats also dropped |
| `bi_5D_max` | `bi` was dropped, so its 5D stats also dropped |
| `tmmn` | Correlated with `tmmx` — keep max temperature |
| `fm1000` | Correlated with `fm100` — keep 100-hr moisture |
| `sph` | Overlaps with `vpd` and `rmax`/`rmin` |
| `ecoregion_l2` | Geography captured by `centroid_lat`/`lon` |
| `ecoregion_l3` | Geography captured by `centroid_lat`/`lon` |
| `h3_resolution` | Constant value — zero information |
| `state_x` | Duplicate/constant column |
| `state_y` | Duplicate/constant column |

## 7. Weather Feature Statistics

| Stat | erc | fm100 | vpd | vs | rmax | rmin | tmmx | pr | erc_5D_mean | fm100_5D_mean | vpd_5D_mean | tmmx_5D_mean |
|------|---|---|---|---|---|---|---|---|---|---|---|---|
| count | 351279.0 | 351279.0 | 351279.0 | 351279.0 | 351279.0 | 351279.0 | 351279.0 | 351279.0 | 351268.0 | 351268.0 | 351268.0 | 351268.0 |
| mean | 43.493 | 1.252 | 0.015 | 0.427 | 7.875 | 2.827 | -23.135 | 0.13 | 42.029 | 1.296 | 0.014 | -23.217 |
| std | 16.521 | 0.37 | 0.008 | 0.152 | 1.751 | 1.511 | 0.854 | 0.606 | 16.016 | 0.362 | 0.007 | 0.864 |
| min | 0.0 | 0.22 | 0.0 | 0.03 | 1.1 | 0.1 | -27.19 | 0.0 | 0.2 | 0.274 | 0.0 | -26.44 |
| 25% | 32.0 | 0.98 | 0.009 | 0.32 | 6.72 | 1.73 | -23.72 | 0.0 | 30.4 | 1.03 | 0.008 | -23.892 |
| 50% | 42.0 | 1.25 | 0.013 | 0.4 | 8.11 | 2.64 | -22.99 | 0.0 | 40.0 | 1.306 | 0.012 | -23.088 |
| 75% | 53.0 | 1.52 | 0.02 | 0.51 | 9.39 | 3.73 | -22.43 | 0.0 | 51.6 | 1.564 | 0.019 | -22.462 |
| max | 111.0 | 2.85 | 0.051 | 1.87 | 10.0 | 10.0 | -21.21 | 62.48 | 107.6 | 2.726 | 0.047 | -21.454 |

## 8. Status and Next Steps

| Step | Status | Action |
|------|--------|--------|
| Phase 2F gridMET extraction | ✅ Complete | `gridmet_features_tx.parquet` (15 MB) |
| Phase 2G dataset assembly | ✅ Complete | `final_training_dataset_tx.parquet` (19 MB) |
| LANDFIRE rasters (4 files) | ⚠️ Missing | Download from doi.org/10.2737/RDS-2020-0016-2 and RDS-2015-0047-4 |
| Phase 3 XGBoost baseline | 🔜 Ready to run | `python run_phase3_train.py --state TX` |
| Phase 3 full model (+ LANDFIRE) | 🔜 After rasters | Re-run Phase 2E → 2G → 3 |

---
*Report generated automatically from `final_training_dataset_tx.parquet`*