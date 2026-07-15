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
| `fm100` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 2.200 | 28.500 | 12.521 | 100-hr dead fuel moisture [%] |
| `vpd` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 0.000 | 5.090 | 1.454 | Vapor pressure deficit [kPa] — from gridMET (not FPA-FOD, no leakage) |
| `vs` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 0.300 | 18.700 | 4.265 | Wind speed [m s⁻¹] |
| `rmax` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 11.000 | 100.000 | 78.746 | Max relative humidity [%] |
| `rmin` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 1.000 | 100.000 | 28.268 | Min relative humidity [%] |
| `tmmx` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | -13.550 | 46.250 | 26.999 | Max temperature [°C] |
| `pr` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 0.000 | 624.800 | 1.303 | Precipitation [mm] |

### 📅 5-Day Trailing Stats (gridMET)  (12 columns)

| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |
|--------|------|---------|-----------|-------|-----|-----|------|-------------|
| `erc_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 0.200 | 107.600 | 42.029 | 5-day trailing mean of ERC (days D-5 to D-1) |
| `erc_5D_max` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 1.000 | 110.000 | 47.306 | 5-day trailing max of ERC |
| `fm100_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 2.740 | 27.260 | 12.956 | 5-day trailing mean of 100-hr fuel moisture |
| `fm100_5D_min` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 2.300 | 25.100 | 11.511 | 5-day trailing min of fuel moisture (lower = drier) |
| `vpd_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 0.000 | 4.684 | 1.375 | 5-day trailing mean of VPD |
| `vpd_5D_max` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 0.000 | 5.440 | 1.783 | 5-day trailing max of VPD |
| `vs_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 0.720 | 12.460 | 4.227 | 5-day trailing mean of wind speed |
| `vs_5D_max` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 1.000 | 18.400 | 5.630 | 5-day trailing max of wind speed |
| `rmax_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 17.080 | 100.000 | 80.461 | 5-day trailing mean of max relative humidity |
| `rmax_5D_min` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 10.500 | 100.000 | 68.786 | 5-day trailing min of max relative humidity |
| `tmmx_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | -6.050 | 43.810 | 26.181 | 5-day trailing mean of max temperature |
| `tmmx_5D_max` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | -1.950 | 46.350 | 29.840 | 5-day trailing max of max temperature |

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
| mean | 43.493 | 12.521 | 1.454 | 4.265 | 78.746 | 28.268 | 26.999 | 1.303 | 42.029 | 12.956 | 1.375 | 26.181 |
| std | 16.521 | 3.701 | 0.786 | 1.516 | 17.511 | 15.109 | 8.542 | 6.056 | 16.016 | 3.621 | 0.734 | 8.639 |
| min | 0.0 | 2.2 | 0.0 | 0.3 | 11.0 | 1.0 | -13.55 | 0.0 | 0.2 | 2.74 | 0.0 | -6.05 |
| 25% | 32.0 | 9.8 | 0.87 | 3.2 | 67.2 | 17.3 | 21.15 | 0.0 | 30.4 | 10.3 | 0.8 | 19.43 |
| 50% | 42.0 | 12.5 | 1.32 | 4.0 | 81.1 | 26.4 | 28.45 | 0.0 | 40.0 | 13.06 | 1.216 | 27.47 |
| 75% | 53.0 | 15.2 | 1.95 | 5.1 | 93.9 | 37.3 | 34.05 | 0.0 | 51.6 | 15.64 | 1.866 | 33.73 |
| max | 111.0 | 28.5 | 5.09 | 18.7 | 100.0 | 100.0 | 46.25 | 624.8 | 107.6 | 27.26 | 4.684 | 43.81 |

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