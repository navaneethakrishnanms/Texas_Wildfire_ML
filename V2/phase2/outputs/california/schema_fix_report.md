# Schema Fix Report — California

## Summary
| Stage | Count |
|---|---|
| Original (Phase 2A) | 249 |
| Removed (leakage + missing + operational) | 33 |
| Added (mandatory + temporal + location + HRRR) | 17 |
| **Final cleaned schema** | **233** |

## Removed Columns

### Event-Based Leakage (FPA-FOD post-discovery fields)
| Column | Reason |
|---|---|
| `COUNTY` | Post-discovery / fire record only |
| `FIPS_CODE` | Post-discovery / fire record only |
| `FIRE_YEAR` | Post-discovery / fire record only |
| `LatLong_County` | Post-discovery / fire record only |
| `LatLong_State` | Post-discovery / fire record only |
| `LATITUDE` | Post-discovery / fire record only |
| `LONGITUDE` | Post-discovery / fire record only |
| `NWCG_CAUSE_AGE_CATEGORY` | Post-discovery / fire record only |
| `NWCG_CAUSE_CLASSIFICATION` | Post-discovery / fire record only |
| `NWCG_GENERAL_CAUSE` | Post-discovery / fire record only |
| `NWCG_REPORTING_AGENCY` | Post-discovery / fire record only |
| `OWNER_DESCR` | Post-discovery / fire record only |
| `SOURCE_SYSTEM` | Post-discovery / fire record only |
| `SOURCE_SYSTEM_TYPE` | Post-discovery / fire record only |

### Added Mandatory Features
| Column | Source | Notes |
|---|---|---|
| `avg_burn_prob` | USFS FSim (50,000 stochastic fire simula | MANDATORY — strongest single landscape predictor (Cohen's d  |
| `whp` | USFS Wildfire Hazard Potential | MANDATORY — Wildfire Hazard Potential index 0-7000 |
| `flep4` | LANDFIRE LF2022 | MANDATORY — Flame Length Exceedance Prob at 4ft (Cohen's d > |
| `cfl` | LANDFIRE LF2022 | MANDATORY — Canopy Fuel Load Mg/ha (2nd strongest landscape  |
| `sin_month` | Computed from window_6h_utc timestamp | sin(2π × month / 12) — cyclic month encoding |
| `cos_month` | Computed | cos(2π × month / 12) |
| `sin_hour` | Computed | sin(2π × window_hour / 24) — cyclic 6hr window encoding |
| `cos_hour` | Computed | cos(2π × window_hour / 24) |
| `centroid_lat` | H3 centroid (h3.cell_to_latlng) | H3 cell centroid latitude — NOT the fire lat/lon from FPA-FO |
| `centroid_lon` | H3 centroid (h3.cell_to_latlng) | H3 cell centroid longitude |
| `rh_pw` | NOAA HRRR (AWS S3 via Herbie) | Phase 2F — add AFTER daily baseline model is working |
| `temp_pw` | NOAA HRRR | Phase 2F |
| `wind_speed_pw` | NOAA HRRR (UGRD+VGRD → √(U²+V²)) | Phase 2F |
| `vpd_pw` | NOAA HRRR (derived TMP+RH) | Phase 2F |
| `hpbl_pw` | NOAA HRRR | Phase 2F — Planetary Boundary Layer Height |
| `dswrf_pw` | NOAA HRRR | Phase 2F — Downwelling Solar Radiation |
| `hrrr_available` | Computed (1 if HRRR available for this y | Phase 2F — binary flag: 2017+ = 1, 2014-2015 = 0, 2016 parti |