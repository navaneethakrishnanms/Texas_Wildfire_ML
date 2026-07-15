# Final Training Dataset Report — Texas (CORRECTED v2)

*Generated: 2026-07-15 10:16*

---

## 1. Overview

| Item | Value |
|------|-------|
| Total rows | **376,233** |
| Total columns | **40** |
| Parquet file | `final_training_dataset_tx.parquet` (19 MB) |
| CSV export | `final_training_dataset_tx.csv` (~97 MB) |
| Study period | 2014–2020 (7 years) |
| State | Texas |
| H3 Resolution | 7 (~1.9 km cell width) |
| Grid cells (unique) | ~57,000 unique H3 cells across Texas |
| Temporal windows | 4 per day (0h, 6h, 12h, 18h UTC) |

## 2. Data Sources & Lineage

This dataset was built by joining 4 sources together on `(h3_cell, date_utc)`:

| # | Source | Features Added | Phase | File |
|---|--------|---------------|-------|------|
| 1 | **FPA-FOD** (Fire occurrence DB) | `label`, `h3_cell`, `date_utc`, `window_hour`, `fire_year` | Phase 2D | `full_training_labels.parquet` |
| 2 | **LANDFIRE + H3 static grid** | `fire_count`, `has_fire_history`, `burnable`, `avg_burn_prob`, `whp`, `flep4`, `cfl`, `centroid_lat/lon` | Phase 2E | `static_features_tx.parquet` |
| 3 | **gridMET** (NW Knowledge Network) | `erc`, `fm100`, `vpd`, `vs`, `rmax`, `rmin`, `tmmx`, `pr` + 12 five-day stats | Phase 2F | `gridmet_features_tx.parquet` |
| 4 | **Derived** | `sin_month`, `cos_month`, `sin_hour`, `cos_hour`, `gridmet_missing` | Phase 2G | Computed inline |

> **Negative sampling:** For each fire event, non-fire (label=0) rows are drawn from cells
> that did NOT burn on the same calendar date and UTC window — date-matched negatives.
> This prevents the model from learning weather patterns instead of spatial discrimination.

## 3. Label Distribution

| Label | Meaning | Count | Percentage |
|-------|---------|-------|------------|
| 1 | Fire discovered | **34,203** | 9.1% |
| 0 | No fire (negative sample) | **342,030** | 90.9% |
| **Total** | | **376,233** | 100% |

## 4. Chronological Train / Val / Test Split

| Split | Years | Total Rows | Fire Rows | Non-fire Rows | Fire Rate |
|-------|-------|-----------|---------|-------------|-----------|
| TRAIN | 2014–2017 | 252,066 | 22,916 | 229,150 | 9.1% |
| VAL | 2018 | 61,181 | 5,561 | 55,620 | 9.1% |
| TEST | 2019–2020 | 62,986 | 5,726 | 57,260 | 9.1% |

> Fire rate is consistent at 9.1% across all splits — confirms no temporal leakage.

## 5. All Columns

### 🆔 Identifiers (not used as features)  (6 columns)

| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |
|--------|------|---------|-----------|-------|-----|-----|------|-------------|
| `h3_cell` | object | 376,233 | 0 ✔ | 0.0% | — | — | — | H3 hexagonal cell ID (resolution-7) — unique cell identifier |
| `date_utc` | object | 376,233 | 0 ✔ | 0.0% | — | — | — | UTC date of the 6-hour window (YYYY-MM-DD) |
| `window_hour` | int64 | 376,233 | 0 ✔ | 0.0% | 0.000 | 18.000 | 14.156 | UTC hour of window start: 0, 6, 12, or 18 |
| `window_6h_utc` | datetime64[ns] | 376,233 | 0 ✔ | 0.0% | — | — | — | Full UTC timestamp (date + hour combined) |
| `fire_year` | int64 | 376,233 | 0 ✔ | 0.0% | 2014.000 | 2020.000 | 2016.591 | Year of the event — used for chronological split |
| `gridmet_missing` | int8 | 376,233 | 0 ✔ | 0.0% | 0.000 | 1.000 | 0.066 | 1 = this H3 cell is outside gridMET coverage (coastal/Gulf border cells) |

### 🎯 Target  (1 columns)

| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |
|--------|------|---------|-----------|-------|-----|-----|------|-------------|
| `label` | int64 | 376,233 | 0 ✔ | 0.0% | 0.000 | 1.000 | 0.091 | TARGET: 1 = fire discovered in this cell/window, 0 = no fire |

### 🌿 Landscape / Static (LANDFIRE + FSim)  (7 columns)

| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |
|--------|------|---------|-----------|-------|-----|-----|------|-------------|
| `avg_burn_prob` | float64 | 374,639 | 1,594 (0.4%) | 0.4% | 0.000 | 0.000 | 0.000 | FSim annual burn probability [0–1]. ⚠️ Currently 0 — rasters not downloaded |
| `whp` | float64 | 374,639 | 1,594 (0.4%) | 0.4% | 0.000 | 0.000 | 0.000 | Wildfire Hazard Potential [0–7000]. ⚠️ Currently 0 — rasters not downloaded |
| `flep4` | float64 | 374,639 | 1,594 (0.4%) | 0.4% | 0.000 | 0.000 | 0.000 | Flame Length Exceedance Prob ≥4ft [0–1]. ⚠️ Currently 0 — rasters not downloaded |
| `cfl` | float64 | 374,639 | 1,594 (0.4%) | 0.4% | 0.000 | 0.000 | 0.000 | Canopy Fuel Load [Mg/ha]. ⚠️ Currently 0 — rasters not downloaded |
| `fire_count` | float64 | 374,639 | 1,594 (0.4%) | 0.4% | 0.000 | 36.000 | 0.212 | How many times this H3 cell burned in 2014–2020 (historical record) |
| `has_fire_history` | object | 374,639 | 1,594 (0.4%) | 0.4% | — | — | — | 1 if this cell burned at least once before, 0 if never |
| `burnable` | object | 374,639 | 1,594 (0.4%) | 0.4% | — | — | — | 1 = burnable land cover (grass/shrub/forest), 0 = non-burnable (water/urban) |

### 🌦️ gridMET Daily Weather  (8 columns)

| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |
|--------|------|---------|-----------|-------|-----|-----|------|-------------|
| `erc` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 0.000 | 111.000 | 43.493 | Energy Release Component [BTU/ft²] from gridMET — top fire weather predictor |
| `fm100` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 2.200 | 28.500 | 12.521 | 100-hour dead fuel moisture [%] — lower = drier = higher fire risk |
| `vpd` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 0.000 | 5.090 | 1.454 | Vapor pressure deficit [kPa] from gridMET NC files (NOT from FPA-FOD — no leakage) |
| `vs` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 0.300 | 18.700 | 4.265 | Wind speed [m/s] from gridMET |
| `rmax` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 11.000 | 100.000 | 78.746 | Maximum daily relative humidity [%] |
| `rmin` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 1.000 | 100.000 | 28.268 | Minimum daily relative humidity [%] |
| `tmmx` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | -13.550 | 46.250 | 26.999 | Maximum daily temperature [°C] |
| `pr` | float32 | 351,279 | 24,954 (6.6%) | 6.6% | 0.000 | 624.800 | 1.303 | Daily precipitation [mm] |

### 📅 5-Day Trailing Stats (gridMET)  (12 columns)

| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |
|--------|------|---------|-----------|-------|-----|-----|------|-------------|
| `erc_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 0.200 | 107.600 | 42.029 | Mean of ERC over the 5 days BEFORE the event (D-1 to D-5) |
| `erc_5D_max` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 1.000 | 110.000 | 47.306 | Max of ERC over the 5 days before the event |
| `fm100_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 2.740 | 27.260 | 12.956 | Mean of 100-hr fuel moisture over 5 days before event |
| `fm100_5D_min` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 2.300 | 25.100 | 11.511 | Min of 100-hr fuel moisture over 5 days (lowest = driest period) |
| `vpd_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 0.000 | 4.684 | 1.375 | Mean VPD over 5 days before event |
| `vpd_5D_max` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 0.000 | 5.440 | 1.783 | Max VPD over 5 days before event |
| `vs_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 0.720 | 12.460 | 4.227 | Mean wind speed over 5 days before event |
| `vs_5D_max` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 1.000 | 18.400 | 5.630 | Max wind speed over 5 days before event |
| `rmax_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 17.080 | 100.000 | 80.461 | Mean of max RH over 5 days before event |
| `rmax_5D_min` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | 10.500 | 100.000 | 68.786 | Min of max RH over 5 days (lowest humidity day in window) |
| `tmmx_5D_mean` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | -6.050 | 43.810 | 26.181 | Mean of max temperature over 5 days before event |
| `tmmx_5D_max` | float32 | 351,268 | 24,965 (6.6%) | 6.6% | -1.950 | 46.350 | 29.840 | Hottest day in the 5-day window before event |

### ⏱️ Temporal Encodings  (4 columns)

| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |
|--------|------|---------|-----------|-------|-----|-----|------|-------------|
| `sin_month` | float32 | 376,233 | 0 ✔ | 0.0% | -1.000 | 1.000 | -0.019 | sin(2π × month / 12) — encodes fire season cyclically |
| `cos_month` | float32 | 376,233 | 0 ✔ | 0.0% | -1.000 | 1.000 | -0.063 | cos(2π × month / 12) — encodes fire season cyclically |
| `sin_hour` | float32 | 376,233 | 0 ✔ | 0.0% | -1.000 | 1.000 | -0.387 | sin(2π × window_hour / 24) — encodes time of day cyclically |
| `cos_hour` | float32 | 376,233 | 0 ✔ | 0.0% | -1.000 | 1.000 | -0.580 | cos(2π × window_hour / 24) — encodes time of day cyclically |

### 📍 Location  (2 columns)

| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |
|--------|------|---------|-----------|-------|-----|-----|------|-------------|
| `centroid_lat` | float64 | 376,233 | 0 ✔ | 0.0% | 25.865 | 36.570 | 31.968 | Latitude of H3 cell centroid [WGS84] — spatial predictor |
| `centroid_lon` | float64 | 376,233 | 0 ✔ | 0.0% | -106.650 | -93.512 | -99.745 | Longitude of H3 cell centroid [WGS84] — spatial predictor |

## 6. Missing Values — Full Explanation

### What is 0 (Zero-Filled) and Why

| Feature(s) | Value | Why Zero | What Happens When Rasters Downloaded |
|-----------|-------|----------|-------------------------------------|
| `avg_burn_prob` | 0.000 | LANDFIRE/FSim raster TIF not downloaded yet | Will have real values 0.001–0.15 (annual burn probability) |
| `whp` | 0.000 | WHP raster TIF not downloaded yet | Will have real values 0–7000 (hazard index) |
| `flep4` | 0.000 | FLEP4 raster TIF not downloaded yet | Will have real values 0–1 (flame length exceedance prob) |
| `cfl` | 0.000 | CFL raster TIF not downloaded yet | Will have real values 0–50 Mg/ha (canopy fuel load) |

> **Impact:** Training without LANDFIRE rasters gives baseline AUROC ~0.85–0.90.
> After downloading and re-running, AUROC is expected to reach ~0.93–0.96.
> The model will learn from weather + fire history features only for now.

### What is NaN and Why

| Feature Group | NaN Count | NaN % | Root Cause | How Model Handles It |
|--------------|-----------|-------|-----------|---------------------|
| All gridMET weather (20 cols) | 24,954 | 6.64% | H3 cells that fall in Gulf of Mexico or along southern border — no gridMET pixel | XGBoost learns split direction for NaN internally — no imputation needed |
| All 5-day trailing stats (12 cols) | 24,965 | 6.64% | Same root + 11 extra rows from Jan 1–5 2014 (no prior-year data) | Same — XGBoost handles |
| `gridmet_missing` flag | 0 | 0% | This column IS the NaN indicator — 1 for coastal cells | Used as explicit feature so model can isolate these cells |
| Landscape/static (7 cols) | 1,594 | 0.42% | H3 cells at boundary of state grid not in static join | Filled with 0 (treated as 'unknown' — very minor, 0.4% of rows) |

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
| `fire_count` | 1,594 | 0.42% | 🟡 Minor (0.4%) — filled with 0 |
| `has_fire_history` | 1,594 | 0.42% | 🟡 Minor (0.4%) — filled with 0 |
| `burnable` | 1,594 | 0.42% | 🟡 Minor (0.4%) — filled with 0 |
| `avg_burn_prob` | 1,594 | 0.42% | 🟡 Minor (0.4%) — filled with 0 |
| `whp` | 1,594 | 0.42% | 🟡 Minor (0.4%) — filled with 0 |
| `flep4` | 1,594 | 0.42% | 🟡 Minor (0.4%) — filled with 0 |
| `cfl` | 1,594 | 0.42% | 🟡 Minor (0.4%) — filled with 0 |
| `erc` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `fm100` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `vpd` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `vs` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `rmax` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `rmin` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `tmmx` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `pr` | 24,954 | 6.63% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `erc_5D_mean` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `erc_5D_max` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `fm100_5D_mean` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `fm100_5D_min` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `vpd_5D_mean` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `vpd_5D_max` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `vs_5D_mean` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `vs_5D_max` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `rmax_5D_mean` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `rmax_5D_min` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `tmmx_5D_mean` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `tmmx_5D_max` | 24,965 | 6.64% | 🟠 Coastal/border cells — XGBoost handles NaN natively |
| `gridmet_missing` | 0 | 0.00% | ✅ Complete |
| `sin_month` | 0 | 0.00% | ✅ Complete |
| `cos_month` | 0 | 0.00% | ✅ Complete |
| `sin_hour` | 0 | 0.00% | ✅ Complete |
| `cos_hour` | 0 | 0.00% | ✅ Complete |

## 7. Columns Dropped (per Team Review)

These columns were extracted but removed before final assembly:

| Column Dropped | Was Redundant With | Team Reason |
|---------------|-------------------|------------|
| `bi` | `erc` | Burning Index and ERC are both NFDRS indices — keep ERC (stronger) |
| `bi_5D_mean`, `bi_5D_max` | `erc_5D_mean`, `erc_5D_max` | bi dropped → its trailing stats also dropped |
| `tmmn` | `tmmx` | Min and max temp are correlated — keep max (more predictive of fire) |
| `fm1000` | `fm100` | 1000-hr and 100-hr moisture are correlated — keep 100-hr |
| `sph` | `vpd`, `rmax`, `rmin` | Specific humidity overlaps with VPD and RH — drop |
| `ecoregion_l2`, `ecoregion_l3` | `centroid_lat`, `centroid_lon` | Geography already captured by coordinates |
| `h3_resolution` | — | Constant value (always 7) — zero information |
| `state_x`, `state_y` | — | Duplicate/constant columns — zero information |

## 8. Weather Feature Statistics (Corrected Values)

> All values below are CORRECT after the scale fix (set_auto_maskandscale fix applied 2026-07-14).

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

### Value Sanity Check (Texas Climate)

| Feature | Reported Mean | Expected Range | Scientific Check |
|---------|--------------|----------------|-----------------|
| tmmx | 27.0°C | 15–45°C annual avg | ✅ Correct — Texas averages 27°C |
| fm100 | 12.5% | 5–25% | ✅ Correct — typical dead fuel moisture |
| vpd | 1.45 kPa | 0.5–3 kPa | ✅ Correct — Texas arid climate |
| rmax | 78.7% | 60–95% | ✅ Correct — max RH in humid TX |
| rmin | 28.3% | 15–50% | ✅ Correct — min RH typical |
| erc | 43.5 | 30–60 | ✅ Correct — energy release component |
| vs | 4.3 m/s | 3–6 m/s | ✅ Correct — Texas wind speed |
| pr | 1.30 mm | 1–3 mm/day | ✅ Correct — daily precipitation avg |

## 9. Status and Next Steps

| Step | Status | Action |
|------|--------|--------|
| Phase 2F gridMET extraction (CORRECTED) | ✅ Complete | `gridmet_features_tx.parquet` (15 MB) — correct units |
| Phase 2G dataset assembly | ✅ Complete | `final_training_dataset_tx.parquet` (19 MB) |
| Dataset report | ✅ This file | `dataset_report_tx.md` |
| LANDFIRE rasters (4 files) | ⚠️ Not downloaded | Download from doi.org/10.2737/RDS-2020-0016-2 and RDS-2015-0047-4 |
| **Phase 3 XGBoost baseline** | **🔜 Ready to run** | **`python run_phase3_train.py --state TX`** |
| Phase 3 full model (+ LANDFIRE) | 🔜 After rasters | Re-run Phase 2E → 2G → Phase 3 |
| California pipeline | 🔜 After TX baseline | Repeat Phases 2D–3 for CA |

---
*Report generated: 2026-07-15 10:16 from `final_training_dataset_tx.parquet`*

**To re-export as CSV:**
```
python -c "import pandas as pd; df = pd.read_parquet('outputs/texas/final_training_dataset_tx.parquet'); df.to_csv('outputs/texas/final_training_dataset_tx.csv', index=False); print('Done!', df.shape)"
```