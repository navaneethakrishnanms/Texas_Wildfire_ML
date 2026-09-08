# Dataset Column Comparison: TX Final Dataset vs. New Supporting Dataset

This document provides a side-by-side comparison of the columns between the **TX Final Cleaned Dataset** (`TX/data/full_tx_clean.parquet`) and the **New Training Dataset** (`Focused_Files/data/tdis_train_daily_hrrr.parquet` / `Supporting_files/data/tdis_train_daily_hrrr.parquet`).

---

## 1. Quick Summary

| Category | TX Final Dataset (`full_tx_clean.parquet`) | New Dataset (`tdis_train_daily_hrrr.parquet`) |
|---|:---:|:---:|
| **Total Columns** | **44** | **33** |
| **Exact Common Columns** | **19** | **19** |
| **Unique to Dataset** | **25** | **14** |
| **Time Resolution** | 6-hour UTC windows | Daily (1 day) |

---

## 2. Common Columns (Highlighted)

The following **19 columns** are shared between both datasets:

> 📌 **Exact Shared Columns:**
> 1. `==h3_cell==` (or **`h3_cell`**) — H3 Resolution-8 spatial hexagon index
> 2. `==label==` (or **`label`**) — Target ignition label (`0` or `1`)
> 3. `==year==` (or **`year`**) — Observation year
> 4. `==avg_burn_prob==` (or **`avg_burn_prob`**) — Static annual burn probability (USFS / FSim)
> 5. `==whp==` (or **`whp`**) — Wildfire Hazard Potential (USFS)
> 6. `==flep4==` (or **`flep4`**) — Flame Length Exceedance Probability ≥4 ft (LANDFIRE)
> 7. `==cfl==` (or **`cfl`**) — Canopy Fuel Load (LANDFIRE)
> 8. `==cbd==` (or **`cbd`**) — Canopy Bulk Density (LANDFIRE)
> 9. `==cbh==` (or **`cbh`**) — Canopy Base Height (LANDFIRE)
> 10. `==erc==` (or **`erc`**) — Energy Release Component (gridMET daily weather)
> 11. `==fm100==` (or **`fm100`**) — 100-hour dead fuel moisture (gridMET daily weather)
> 12. `==vpd==` (or **`vpd`**) — Vapor Pressure Deficit (gridMET daily weather)
> 13. `==vs==` (or **`vs`**) — Wind velocity / speed at 10 m (gridMET daily weather)
> 14. `==rmax==` (or **`rmax`**) — Maximum relative humidity (gridMET daily weather)
> 15. `==rmin==` (or **`rmin`**) — Minimum relative humidity (gridMET daily weather)
> 16. `==tmmx==` (or **`tmmx`**) — Maximum temperature (gridMET daily weather)
> 17. `==pr==` (or **`pr`**) — Precipitation accumulation (gridMET daily weather)
> 18. `==sin_month==` (or **`sin_month`**) — Trigonometric seasonal feature (sin of month)
> 19. `==cos_month==` (or **`cos_month`**) — Trigonometric seasonal feature (cos of month)

*(Note: `date_utc` in TX maps conceptually to `date` in New, and `_split` maps to `split`)*

---

## 3. Full Side-by-Side Column Comparison Table

In the table below, **common columns present in both datasets are highlighted with bold text and markers (`==...==`)**:

| # | TX Final Dataset (`full_tx_clean.parquet`) | New Dataset (`tdis_train_daily_hrrr.parquet`) | Status / Overlap |
|---|---|---|---|
| 1 | ==**`h3_cell`**== | ==**`h3_cell`**== | **Common (Shared Key)** |
| 2 | ==**`label`**== | ==**`label`**== | **Common (Target)** |
| 3 | ==**`year`**== | ==**`year`**== | **Common** |
| 4 | ==**`avg_burn_prob`**== | ==**`avg_burn_prob`**== | **Common (Static Fuel/Hazard)** |
| 5 | ==**`whp`**== | ==**`whp`**== | **Common (Static Fuel/Hazard)** |
| 6 | ==**`flep4`**== | ==**`flep4`**== | **Common (Static Fuel/Hazard)** |
| 7 | ==**`cfl`**== | ==**`cfl`**== | **Common (Static Fuel/Hazard)** |
| 8 | ==**`cbd`**== | ==**`cbd`**== | **Common (Static Fuel/Hazard)** |
| 9 | ==**`cbh`**== | ==**`cbh`**== | **Common (Static Fuel/Hazard)** |
| 10 | ==**`erc`**== | ==**`erc`**== | **Common (gridMET Weather)** |
| 11 | ==**`fm100`**== | ==**`fm100`**== | **Common (gridMET Weather)** |
| 12 | ==**`vpd`**== | ==**`vpd`**== | **Common (gridMET Weather)** |
| 13 | ==**`vs`**== | ==**`vs`**== | **Common (gridMET Weather)** |
| 14 | ==**`rmax`**== | ==**`rmax`**== | **Common (gridMET Weather)** |
| 15 | ==**`rmin`**== | ==**`rmin`**== | **Common (gridMET Weather)** |
| 16 | ==**`tmmx`**== | ==**`tmmx`**== | **Common (gridMET Weather)** |
| 17 | ==**`pr`**== | ==**`pr`**== | **Common (gridMET Weather)** |
| 18 | ==**`sin_month`**== | ==**`sin_month`**== | **Common (Calendar Feature)** |
| 19 | ==**`cos_month`**== | ==**`cos_month`**== | **Common (Calendar Feature)** |
| 20 | `date_utc` | `date` | Equivalent Date string |
| 21 | `_split` | `split` | Equivalent Split flag (`train`/`val`/`test`) |
| 22 | `window_hour` | — | TX Only (6h window hour) |
| 23 | `window_6h_utc` | — | TX Only (6h window label) |
| 24 | `centroid_lat` | — | TX Only (Latitude centroid) |
| 25 | `centroid_lon` | — | TX Only (Longitude centroid) |
| 26 | `fire_year` | — | TX Only |
| 27 | `fire_count` | — | TX Only (Historical leakage flag) |
| 28 | `has_fire_history` | — | TX Only (Historical leakage flag) |
| 29 | `burnable` | — | TX Only (Binary burnable mask) |
| 30 | `erc_5D_mean` | — | TX Only (5-day trailing weather) |
| 31 | `erc_5D_max` | — | TX Only (5-day trailing weather) |
| 32 | `fm100_5D_mean` | — | TX Only (5-day trailing weather) |
| 33 | `fm100_5D_min` | — | TX Only (5-day trailing weather) |
| 34 | `vpd_5D_mean` | — | TX Only (5-day trailing weather) |
| 35 | `vpd_5D_max` | — | TX Only (5-day trailing weather) |
| 36 | `vs_5D_mean` | — | TX Only (5-day trailing weather) |
| 37 | `vs_5D_max` | — | TX Only (5-day trailing weather) |
| 38 | `rmax_5D_mean` | — | TX Only (5-day trailing weather) |
| 39 | `rmax_5D_min` | — | TX Only (5-day trailing weather) |
| 40 | `tmmx_5D_mean` | — | TX Only (5-day trailing weather) |
| 41 | `tmmx_5D_max` | — | TX Only (5-day trailing weather) |
| 42 | `gridmet_missing` | — | TX Only (Missing indicator) |
| 43 | `sin_hour` | — | TX Only (Sub-daily diurnal feature) |
| 44 | `cos_hour` | — | TX Only (Sub-daily diurnal feature) |
| 45 | — | `ecoregion_id` | New Only (EPA Ecoregion categorical ID) |
| 46 | — | `elevation_m` | New Only (Elevation from DEM) |
| 47 | — | `slope_deg` | New Only (Slope in degrees) |
| 48 | — | `aspect_deg` | New Only (Aspect in degrees) |
| 49 | — | `road_dist_km` | New Only (Distance to nearest road) |
| 50 | — | `sin_dow` | New Only (Day of week sine) |
| 51 | — | `cos_dow` | New Only (Day of week cosine) |
| 52 | — | `is_weekend` | New Only (Weekend indicator) |
| 53 | — | `is_holiday` | New Only (Federal holiday indicator) |
| 54 | — | `hrrr_tmp` | New Only (HRRR 2 m temperature forecast) |
| 55 | — | `hrrr_vpd` | New Only (HRRR vapor pressure deficit) |
| 56 | — | `hrrr_wind` | New Only (HRRR 10 m wind speed) |

---

## 4. Summary of Key Structural Differences

1. **Topography & Human Geography (Added in New Dataset):**
   * New dataset adds `elevation_m`, `slope_deg`, `aspect_deg`, `road_dist_km`, and `ecoregion_id` (replacing raw `centroid_lat`/`centroid_lon` to prevent spatial memorization).
   * In addition, companion files in the new package add `powerline_dist_km` (HIFLD powerlines) and `hrrr_mstav` (soil moisture).

2. **Forecast Weather vs Observed Weather:**
   * Both datasets retain daily **gridMET** (`erc`, `fm100`, `vpd`, `vs`, `rmax`, `rmin`, `tmmx`, `pr`).
   * The new dataset integrates **HRRR forecast features** (`hrrr_tmp`, `hrrr_vpd`, `hrrr_wind`).

3. **Time Granularity:**
   * TX Final Dataset was partitioned into **6-hour windows** (`window_hour`, `sin_hour`, `cos_hour`).
   * The New Dataset operates at **1-day resolution** and introduces day-of-week / holiday features (`sin_dow`, `cos_dow`, `is_weekend`, `is_holiday`).
