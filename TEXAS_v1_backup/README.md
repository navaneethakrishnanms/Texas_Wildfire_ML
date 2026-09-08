# TEXAS Datasets

This folder contains both the **exact original raw dataset** (unimputed, with all 35 columns merged) and the **fully imputed version**.

---

## 1. Files in `TEXAS/`

| File Name | Format | Rows | Columns | Missing Values (`NaN`) | Description |
|---|:---:|:---:|:---:|:---:|---|
| **[`tdis_train_daily_hrrr_raw.parquet`](file:///c:/Users/Admin/Downloads/Texas%20ML%20Wildfire/TEXAS/tdis_train_daily_hrrr_raw.parquet)** | Parquet | 3,595,513 | **35** | 30,368,531 | **Raw Unimputed Dataset** (all 35 feature columns merged, original missing values preserved). |
| **[`tdis_train_daily_imputed.parquet`](file:///c:/Users/Admin/Downloads/Texas%20ML%20Wildfire/TEXAS/tdis_train_daily_imputed.parquet)** | Parquet | 3,595,513 | **35** | **0** | **Fully Imputed Dataset** (0 missing values, all rows complete). |
| **[`tdis_train_daily_imputed_sample_1k.csv`](file:///c:/Users/Admin/Downloads/Texas%20ML%20Wildfire/TEXAS/tdis_train_daily_imputed_sample_1k.csv)** | CSV | 1,000 | 35 | 0 | 1,000-row sample for quick inspection. |

---

## 2. Complete Column Schema (All 35 Columns)

1. `h3_cell` — Spatial index (H3 Resolution-8 hexagon)
2. `date` — Observation date
3. `label` — Binary fire label (`0` or `1`)
4. `year` — Observation year
5. `erc` — Energy Release Component (gridMET daily)
6. `fm100` — 100-hour dead fuel moisture (gridMET daily)
7. `vpd` — Vapor Pressure Deficit (gridMET daily)
8. `vs` — 10 m wind speed (gridMET daily)
9. `rmax` — Max relative humidity (gridMET daily)
10. `rmin` — Min relative humidity (gridMET daily)
11. `tmmx` — Max temperature (gridMET daily)
12. `pr` — Precipitation accumulation (gridMET daily)
13. `ecoregion_id` — EPA Level-3 Ecoregion ID
14. `elevation_m` — Elevation in meters
15. `slope_deg` — Terrain slope in degrees
16. `aspect_deg` — Terrain aspect in degrees
17. `road_dist_km` — Distance to nearest road in km
18. `avg_burn_prob` — USFS Burn probability
19. `whp` — Wildfire Hazard Potential
20. `cfl` — Canopy Fuel Load (LANDFIRE)
21. `flep4` — Flame Length Exceedance Probability ≥4 ft (LANDFIRE)
22. `cbd` — Canopy Bulk Density (LANDFIRE)
23. `cbh` — Canopy Base Height (LANDFIRE)
24. `sin_dow` — Day of week (sine)
25. `cos_dow` — Day of week (cosine)
26. `is_weekend` — Weekend indicator
27. `sin_month` — Month of year (sine)
28. `cos_month` — Month of year (cosine)
29. `is_holiday` — Federal holiday indicator
30. `split` — Dataset partition (`train`, `val`, `test`)
31. `hrrr_tmp` — HRRR forecast 2 m temperature
32. `hrrr_vpd` — HRRR forecast vapor pressure deficit
33. `hrrr_wind` — HRRR forecast 10 m wind speed
34. `powerline_dist_km` — Distance to nearest transmission line (HIFLD)
35. `hrrr_mstav` — HRRR soil moisture availability (%)
