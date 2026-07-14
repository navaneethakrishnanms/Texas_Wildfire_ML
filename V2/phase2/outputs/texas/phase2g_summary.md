# Phase 2G Summary — Texas

## Dataset Statistics

| Split | Years | Total Rows | Fire (1) | Non-fire (0) | Fire Rate |
|-------|-------|-----------|---------|-------------|-----------|
| TRAIN | 2014–2017 | 252,066 | 22,916 | 229,150 | 9.1% |
| VAL   | 2018      | 61,181 | 5,561 | 55,620 | 9.1% |
| TEST  | 2019–2020 | 62,986 | 5,726 | 57,260 | 9.1% |
| **TOTAL** | 2014–2020 | **376,233** | **34,203** | **342,030** | **9.1%** |

## Feature Columns (33 total)

### Landscape (static, LANDFIRE)
fire_count, has_fire_history, burnable, avg_burn_prob, whp, flep4, cfl

### gridMET Weather (daily + 5-day trailing)
avg_burn_prob, erc, fm100, vpd, vs, rmax, rmin, tmmx, pr, erc_5D_mean, erc_5D_max, fm100_5D_mean, fm100_5D_min, vpd_5D_mean, vpd_5D_max, vs_5D_mean, vs_5D_max, rmax_5D_mean, rmax_5D_min, tmmx_5D_mean, tmmx_5D_max

### Temporal
sin_month, cos_month, sin_hour, cos_hour

### Location
centroid_lat, centroid_lon

### Other
None

## Dropped Columns (redundant per team review)
bi, tmmn, fm1000, sph, bi_5D_mean, bi_5D_max, ecoregion_l2, ecoregion_l3, h3_resolution, state_x, state_y

## Data Quality
- gridMET NaN rows: 24954 (6.6% of rows — coastal/border cells)
- Fill contamination: ✔ CLEAN (all values < 9000)
- Leakage audit: ✔ PASSED
- LANDFIRE rasters: ⚠️ LANDFIRE rasters not yet downloaded — avg_burn_prob/whp/flep4/cfl = 0

## Files
- Full:  `final_training_dataset_tx.parquet`
- Train: `train_tx.parquet`
- Val:   `val_tx.parquet`
- Test:  `test_tx.parquet`

## Next Steps
1. Download 4 LANDFIRE rasters → re-run Phase 2E → re-run Phase 2G
2. Run Phase 3: `python run_phase3_train.py --state TX`
