# Phase 2G Summary — Texas

## Dataset Statistics

| Split | Years | Total Rows | Fire (label=1) | Non-fire (label=0) |
|---|---|---|---|---|
| TRAIN | 2014–2017 | 252,066 | 22,916 | 229,150 |
| VAL   | 2018      | 61,181 | 5,561 | 55,620 |
| TEST  | 2019–2020 | 62,986 | 5,726 | 57,260 |
| **TOTAL** | 2014–2020 | **376,233** | **34,203** | **342,030** |

## Feature Columns (43 total)
centroid_lat, centroid_lon, state_x, state_y, h3_resolution, ecoregion_l3, ecoregion_l2, fire_count, has_fire_history, burnable, avg_burn_prob, whp, flep4, cfl, erc, fm100, fm1000, bi, vpd, vs, rmax, rmin, tmmx, tmmn, pr, sph, erc_5D_mean, erc_5D_max, fm100_5D_mean, fm100_5D_min, bi_5D_mean, bi_5D_max, vpd_5D_mean, vpd_5D_max, vs_5D_mean, vs_5D_max, rmax_5D_mean, rmax_5D_min, tmmx_5D_mean, tmmx_5D_max...

## Leakage Audit
- Forbidden columns found: NONE ✔

## Files
- Full: `final_training_dataset_tx.parquet`
- Train: `train_tx.parquet`
- Val: `val_tx.parquet`
- Test: `test_tx.parquet`
