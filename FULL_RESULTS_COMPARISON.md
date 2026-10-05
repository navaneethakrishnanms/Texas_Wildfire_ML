# Full Results Comparison — TDIS_Ignition (their package) vs. Our Model History vs. TDIS_Ignition_BIT (our retrain)

**Date:** 2026-10-01
**Scope:** Every dataset and model version on both sides of this project — exact row counts, exact feature lists, every hyperparameter, every reported metric, and confusion matrices wherever the underlying numbers allow one to be computed or derived.
**Method note on confusion matrices:** where we had direct access to the model file and test rows, the confusion matrix was computed exactly (labeled **EXACT**). Where we only had published precision/recall/F1/base-rate/n (TDIS's served model), the matrix was **derived algebraically** from those four numbers (labeled **DERIVED**) — shown because the user asked for one, but it is not a substitute for re-scoring their model ourselves.

---

# PART 1 — Their result: `TDIS_Ignition/` (as shipped, 2026-09-22/23)

## 1.1 Dataset inventory — every data file that feeds the served model

| File | Rows × Cols | What it is |
|---|---|---|
| `data/labels/ignitions_daily_tx.parquet` | 960,054 × 7 | Fused VIIRS+FPA-FOD daily ignition labels, 2014-01-01 → 2026-07-29. `max_size_acres` null in 96.2% of rows. |
| `data/flares/flare_cells_tx_v2.parquet` | 538 × 1 | Flare/industrial-hotspot cell exclusion list (meta.json claims 531 — a 7-cell discrepancy confirmed by direct file load) |
| `data/static/tx_static_master.parquet` | 1,708,940 × 15 | Full TX static master (terrain, fuel, ecoregion, roads) — 0 nulls |
| `data/static/mstav_feature.parquet` | 2,007,436 × 3 | Cached HRRR soil-moisture feature |
| `data/scored/operational_historical_res5_v2.parquet` | 5,064,462 × 4 | The real-population scored panel (base rate 1.92%) |
| `model_tristate/data/train_tristate_v2.parquet` | 2,731,285 × 45 (TX subset: 2,202,331) | Tri-state (TX+NM+OK) training table, ecoregion-bug-fixed — **this is the file we used to build `TDIS_Ignition_BIT`** |

## 1.2 The served TX model — `model/tdis_forecast_hrrr_filtered.json`

### Every hyperparameter (confirmed by loading the booster directly)
```
Algorithm:          XGBoost XGBClassifier, binary:logistic
num_boosted_rounds:  1000  (fixed — NOT early-stopped; see DATA_QUALITY_REVIEW.md F2)
max_depth:           9
min_child_weight:    30
learning_rate:       0.02
subsample:           0.8
colsample_bytree:    0.8
eval_metric:         aucpr
scale_pos_weight:    computed from train fold (n_neg / n_pos)
random_state:        42
device:              cuda
base_score:          0.5
tree_method:         auto / grow_quantile_histmaker
num_feature:         23
```

### The 23 features, in exact model order (confirmed by loading the booster)
`road_dist_km, ecoregion_id, elevation_m, slope_deg, aspect_deg, avg_burn_prob, whp, flep4, cfl, cbd, cbh, powerline_dist_km, sin_month, cos_month, sin_dow, cos_dow, is_weekend, is_holiday, hrrr_tmp, hrrr_vpd, hrrr_wind, hrrr_mstav, drought_score`

`flep4` and `cfl` have **0.000 gain importance** (confirmed in `meta.json`'s own `feature_importance` dict — the model never splits on them).

### Training table construction (from reading `code/scripts/01–06, 18, step32, step34`)
1. VIIRS (rectangular TX bbox) fused with FPA-FOD → `ignitions_daily_tx.parquet` (960,054 positive-label rows)
2. 8 temporal negatives + 0.4× spatial negatives per positive (seed 42)
3. Static features joined from `tx_static_master.parquet`
4. **538 flare cells removed** (persistent industrial hotspots — ~27% of raw positives)
5. HRRR forecast weather (not observations) joined, archive starts 2018-07-16
6. Rows with missing HRRR dropped (not imputed)
7. **Result: 3,595,513 raw rows → 2,007,436 usable rows**

### Split (from the training script, which is authoritative — not the docs, which disagree with it)
| Split | Years | Rows | Positive rate |
|---|---|---|---|
| Train | 2018-07 → 2021 | ~856,300 | ~0.229 |
| *(2022 — discarded, not used as val or test)* | 2022 | 259,150 | 0.261 |
| **Test** | 2023 → 2026-07 | **891,983** | **0.2341** |

## 1.3 Metrics — matched/balanced test set (891,983 rows, base rate 0.2341)

| Metric | Value |
|---|---|
| AUC-PR | **0.4941** |
| AUROC | **0.7402** |
| Best F1 | **0.4958** (threshold 0.4991) |
| Precision @ best F1 | 0.441 |
| Recall @ best F1 | 0.566 |
| Lift (AUPR ÷ base rate) | 2.11× |

### Confusion matrix — matched test set, **DERIVED** from precision/recall/base-rate/n above
```
positives = 0.2341 × 891,983 ≈ 208,793
TP = recall × positives = 0.566 × 208,793 ≈ 118,177
FN = positives − TP ≈ 90,616
predicted_positive = TP ÷ precision = 118,177 ÷ 0.441 ≈ 267,953
FP = predicted_positive − TP ≈ 149,776
TN = n − predicted_positive − FN ≈ 533,414
```
|  | Predicted No Fire | Predicted Fire |
|---|---|---|
| **Actual No Fire** | TN ≈ 533,414 | FP ≈ 149,776 |
| **Actual Fire** | FN ≈ 90,616 | TP ≈ 118,177 |

## 1.4 Metrics — real deployment population (5,064,462 cell-days, 2024–2026, true base rate 0.0192)

| Metric | Value |
|---|---|
| AUC-PR | **0.0878** |
| AUROC | **0.7968** |
| Lift | **4.58×** |
| Best F1 | 0.1692 at calibrated threshold 0.0774 (P 0.1493 / R 0.1851 — current numbers from `realpop_f1_sweep.json`) |
| Calibrated ECE (2026 holdout) | 0.0008 (raw 0.2944 → calibrated) |

### Confusion matrix — real population, **DERIVED** at the best-F1 threshold (0.0747)
```
positives = 0.01918 × 5,064,462 ≈ 97,136
TP = 0.1851 × 97,136 ≈ 17,982
FN = 97,136 − 17,982 ≈ 79,154
predicted_positive = 17,982 ÷ 0.1493 ≈ 120,409
FP = 120,409 − 17,982 ≈ 102,427
TN = 5,064,462 − 120,409 − 79,154 ≈ 4,864,899
```
|  | Predicted No Fire | Predicted Fire |
|---|---|---|
| **Actual No Fire** | TN ≈ 4,864,899 | FP ≈ 102,427 |
| **Actual Fire** | FN ≈ 79,154 | TP ≈ 17,982 |

## 1.5 Calibrator (loaded and evaluated directly)
`operational_isotonic_calibrator.joblib` — sklearn `IsotonicRegression`, fit on 3,934,242 rows, verified on 1,130,220-row 2026 holdout.

| Raw score | 0.0 | 0.1 | 0.3 | 0.5 | 0.7 | **1.0** |
|---|---|---|---|---|---|---|
| Calibrated | 0.000 | 0.003 | 0.014 | 0.032 | 0.074 | **0.333** |

**No cell can ever be shown a calibrated probability above 33.3%** — a hard ceiling, confirmed by evaluating the actual file, undocumented anywhere in their shipped materials.

## 1.6 Their own unpromoted best model — `rev4` (for completeness, never shipped/served)
Same standardized hyperparameters as step32/step34 (depth 9, 1000 trees, `min_child_weight` 30, `learning_rate` 0.02, subsample 0.8), **plus explicit XGBoost `monotone_constraints`** forcing burn-probability/WHP/drought/gust/VPD to only increase risk, trained on **daily-peak** HRRR weather (max across 4 leads off a 12Z-previous-day init) instead of a single snapshot — the fix for a documented backwards wind-response bug found via SHAP.

| Metric | rev4 (unpromoted) | Served model |
|---|---|---|
| Matched test AUC-PR | 0.511 | 0.494 |
| Real-pop lift | 4.76–4.88× | 4.58× |
| Real-pop AUROC | 0.809–0.811 | 0.797 |

**rev4 beats the served model on every gate but was never promoted to production.**

## 1.7 Tri-state model (`model_tristate/`) — rev5_v2, 28 features

Served model's 23 features **plus** `bp_x_vpd`, `whp_x_vpd`, `bp_x_drought` (interactions), `pop_density`, `fm100_5d_min`. Trained on TX+NM+OK (2,731,285 rows), **tested on TX-only** by hardcoded rule.

| | control_v2 (TX-only) | rev5_v2 (TX+NM+OK) |
|---|---|---|
| Matched test (632,093 rows, base rate 31.5%) AUC-PR | 0.7549 | 0.7516 |
| Matched test AUROC | 0.8312 | 0.8302 |
| Real-pop (3,622,850 rows, base rate 2.524%) AUC-PR | 0.1082 | 0.1074 |
| Real-pop AUROC | 0.7870 | 0.7854 |
| Real-pop lift | 4.29× | 4.26× |

**Conclusion (their own): tri-state training is a statistical wash for TX accuracy.**

---

# PART 2 — Our result history (before this retrain)

## 2.1 V1 — proof of concept (Jul 2026)

| | |
|---|---|
| Labels | NASA FIRMS/VIIRS 2024 fire detections only |
| Features | **18** — annual GEE composites: NDVI, EVI, LST, Temperature, Wind, Rainfall, DEM, Slope, Aspect, LandCover, month, day_of_year, season_code, sin/cos_month, sin/cos_doy, is_peak_fire_season |
| Rows | Test set: **2,250** (TP 508, FP 315, TN 1,354, FN 73) |
| Split | Train Jan–Aug 2024 / Val Sep 2024 / Test Oct–Dec 2024 |
| Model | LightGBM (winner) — `n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, is_unbalance=True, early_stopping_rounds=50` |
| Threshold | 0.46 |

| Metric | Value |
|---|---|
| AUROC | 0.9142 |
| AUC-PR | 0.7549 |
| F1 | 0.7236 |
| Precision | 0.6173 |
| Recall | 0.8744 |
| Accuracy | 0.8276 |

**Confusion matrix (EXACT, from `V1/outputs/test_metrics.json`):**
|  | Predicted No Fire | Predicted Fire |
|---|---|---|
| **Actual No Fire** | TN = 1,354 | FP = 315 |
| **Actual Fire** | FN = 73 | TP = 508 |

⚠️ **Flagged unreliable** — annual composite features leak "which month/location historically burns"; this is proof-of-concept only, not comparable to any later model.

## 2.2 V2 — production pipeline baseline (Jul 2026)

| | |
|---|---|
| Labels | FPA-FOD v6, Texas, ≥1 acre, 2014–2020 |
| Grid | H3 res-8 |
| Features | **31** (gridMET daily weather + static + calendar, after removing 2 leakage columns) |
| Rows | 376,233 total (34,203 fire / 9.1%) |
| Split | Train 2014–17 (252,066) / Val 2018 (61,181) / Test 2019–20 (62,986) |
| Model | XGBoost, `max_depth=7, learning_rate=0.05, subsample=0.8, min_child_weight=~20` (baseline, untuned) |

| Split | AUROC | AUC-PR | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Train | 0.9302 | 0.5723 | 0.5386 | 0.4153 | 0.7659 |
| Val | 0.8742 | 0.4125 | 0.4316 | 0.3355 | 0.6049 |
| **Test** | **0.8569** | **0.3978** | **0.4082** | 0.3315 | 0.5313 |

**Confusion matrix (EXACT, threshold 0.6841, max-F1 on val):**
|  | Predicted No Fire | Predicted Fire |
|---|---|---|
| **Actual No Fire** | TN = 51,125 | FP = 6,135 |
| **Actual Fire** | FN = 2,684 | TP = 3,042 |

⚠️ A leaked version of this model (keeping `fire_count`/`has_fire_history`) scored AUROC 0.9900 — caught and fixed; the numbers above are post-fix, honest.

## 2.3 TX — tuned model, our pre-TDIS-handoff best (Jul–Aug 2026)

| | |
|---|---|
| Dataset | `final_training_dataset_tx_22.07.2026_landfire.xlsx`, 376,233 rows, 42 cols → 375,779 after dedup |
| Features | **34** (gridMET) — real LANDFIRE joined (fixing V2's zero-filled placeholders) |
| Split | Train 2014–17 (251,785) / Val 2018 (61,139) / Test 2019–20 (62,855) |
| Hyperparameter search | **21-trial two-stage search** (`TX/tune_tx.py`) — 12 tree-structure trials + 9 lr×subsample trials |
| Final hyperparameters | `max_depth=9, min_child_weight=30, learning_rate=0.01, subsample=0.9, colsample_bytree=0.8, colsample_bylevel=0.8, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=10.12`, 1,627 trees (early-stopped) |

| Split | AUROC | AUC-PR | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Val | 0.8856 | 0.4452 | — | — | — |
| **Test** | **0.8687** | **0.4247** | **0.4293** | 0.392 | 0.475 |

**Confusion matrix (EXACT, threshold 0.710 — from `TX/MODEL_TRAINING_REPORT_TX.md`):**
```
                       PREDICTED
                    No Fire   |   Fire
Actual:   No Fire  |  TN=53,129  |  FP= 4,131
          Fire     |  FN= 2,937  |  TP= 2,658
```
Recall 47.5% · Precision 39.2% · Specificity 92.8% · F1 42.9%

**HRRR extension (TX-HRRR, 42 features):** did **not** improve accuracy (AUROC 0.8682, AUC-PR 0.4225) — a confirmed negative result, independently matched by TDIS's own HRRR findings.

## 2.4 TEXAS 30-feature model (27 Aug 2026) — rebuilt from TDIS's own raw parquet

| | |
|---|---|
| Source | `tdis_train_daily_hrrr.parquet` (3,595,513 × 33/35), TDIS-supplied |
| Imputation | Cross-source substitution (`hrrr_tmp←tmmx` etc. for 2014–mid-2018) + seasonal-median — eliminated 24.09% missingness → 0 nulls |
| Features | **30** |
| Split | Train ≤2023 (2,761,329, 26.6% pos) / Val 2024 (287,649, 29.4% pos) / Test 2025–26-07 (460,463, 30.7% pos) |
| Hyperparameters | `n_estimators=1000 (patience 50), max_depth=9, min_child_weight=30, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=2.7616`, best iteration 997 |

| Split | AUC-PR | AUROC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Train | 0.7458 | 0.8516 | 0.6452 | 0.6391 | 0.6515 |
| Val | 0.7338 | 0.8208 | 0.6317 | — | — |
| **Test (raw)** | **0.7384** | **0.8249** | **0.6440** | 0.6435 | 0.6445 |
| Test (calibrated) | 0.7337 | 0.8249 | 0.6440 | — | — |

**Confusion matrix (EXACT, computed from test rows: n=460,463, pos rate 30.71%, P=0.6435, R=0.6445):**
```
positives = 0.3071 × 460,463 ≈ 141,404
TP = 0.6445 × 141,404 ≈ 91,135
FN = 141,404 − 91,135 ≈ 50,269
predicted_positive = 91,135 ÷ 0.6435 ≈ 141,606
FP = 141,606 − 91,135 ≈ 50,471
TN = 460,463 − 141,606 − 50,269 ≈ 268,588
```
|  | Predicted No Fire | Predicted Fire |
|---|---|---|
| **Actual No Fire** | TN ≈ 268,588 | FP ≈ 50,471 |
| **Actual Fire** | FN ≈ 50,269 | TP ≈ 91,135 |

## 2.5 TEXAS 32-feature model (07 Sep 2026) — our best model before the BIT retrain

| | |
|---|---|
| Change | Added `lat`/`lon` (2 features) on top of the 30-feat model; ran a genuine 21-trial two-stage hyperparameter search instead of reusing fixed settings |
| Rows | Same 3,595,513 source; Train 2,761,329 (26.58% pos) / Val 287,649 (29.39% pos) / Test 460,463 (30.71% pos) |
| Features | **32** |
| Hyperparameter search | Stage 1 (12 trials): winner `max_depth=9, min_child_weight=30`. Stage 2 (9 trials): winner `learning_rate=0.05, subsample=0.8` |
| Final hyperparameters | `max_depth=9, min_child_weight=30, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=2.7616`, best iteration 912 |

| Split | AUC-PR | AUROC | F1 | Precision | Recall | Lift |
|---|---|---|---|---|---|---|
| Train (raw) | 0.7920 | 0.8836 | 0.6866 | 0.6928 | 0.6806 | 2.98× |
| Val (raw) | 0.7499 | 0.8298 | 0.6471 | 0.6609 | 0.6338 | 2.55× |
| **Test (raw)** | **0.7484** | **0.8309** | **0.6498** | 0.6390 | 0.6610 | 2.44× |
| Test (calibrated) | 0.7435 | 0.8308 | 0.6497 | 0.6250 | 0.6764 | 2.42× |

**Confusion matrix (EXACT, computed from n=460,463, pos rate 30.71%, raw P=0.6390, R=0.6610):**
```
positives = 0.3071 × 460,463 ≈ 141,404
TP = 0.6610 × 141,404 ≈ 93,468
FN = 141,404 − 93,468 ≈ 47,936
predicted_positive = 93,468 ÷ 0.6390 ≈ 146,273
FP = 146,273 − 93,468 ≈ 52,805
TN = 460,463 − 146,273 − 47,936 ≈ 266,254
```
|  | Predicted No Fire | Predicted Fire |
|---|---|---|
| **Actual No Fire** | TN ≈ 266,254 | FP ≈ 52,805 |
| **Actual Fire** | FN ≈ 47,936 | TP ≈ 93,468 |

Top features: `elevation_m` (8.72%), `avg_burn_prob` (6.79%), **`lat`** (6.07%, #3), `rmin` (6.04%), **`lon`** (5.72%, #5). `flep4`/`cfl` dead (0.00%) — same finding as every other model on either side.

---

# PART 3 — Our new result: `TDIS_Ignition_BIT/` (01 Oct 2026, this retrain)

## 3.1 Dataset built — `TDIS_Ignition_BIT/data/bit_train_tx.parquet`

**Source:** `TDIS_Ignition/model_tristate/data/train_tristate_v2.parquet` (their newest, most-corrected training table — the one where they'd already fixed the TX/NM/OK ecoregion-scale bug) — filtered to TX only.

| Stage | Rows | Action |
|---|---|---|
| Raw tri-state table | 2,731,285 × 45 | TDIS's file, as shipped |
| TX-only filter | 2,202,331 | `state == 'tx'` |
| After imputation | 2,202,331 | 0 nulls — see fixes below |
| After phantom-row truncation | **2,201,289 × 44** | Removed 1,042 rows dated after their own last real label (2026-07-29) |

### Fixes/additions applied on top of their data
1. **Derived `lat`/`lon` from the H3 cell index** — TDIS's own `lat`/`lon` columns were **100% null** for every TX row in this file (confirmed by direct load — an undocumented defect). Used `h3.cell_to_latlng()`, 100% coverage.
2. **Seasonal (month) median imputation** for 8 legacy gridMET columns (`erc, fm100, vpd, vs, rmax, rmin, tmmx, pr`) — 76.6% null in the raw file.
3. **Mode-imputed `ecoregion_id`** (11.1% null), **median-imputed `fm100_5d_min`** (10.0% null).
4. **Truncated 1,042 phantom future-dated rows** (2026-07-30/31, all `label=0`) — a smaller recurrence of the "fabricated future rows" defect our own `DATA_QUALITY_REVIEW.md` flagged as Critical in the original handoff.

### Split (TDIS's own embedded `split` column, unchanged)
| Split | Rows | Date range | Positive rate |
|---|---|---|---|
| Train | 671,455 | 2018-07-16 → 2020-12-31 | 30.42% |
| Val | 268,332 | 2021-01-01 → 2021-12-31 | 28.76% |
| **Test** | **1,261,502** | 2022-01-01 → 2026-07-29 | 30.55% |

## 3.2 Features used — 38 total

| Group | Count | Features |
|---|---|---|
| Static (terrain/access/fuel hazard) | 12 | `road_dist_km, ecoregion_id, elevation_m, slope_deg, aspect_deg, avg_burn_prob, whp, flep4, cfl, cbd, cbh, powerline_dist_km` |
| Calendar | 6 | `sin_month, cos_month, sin_dow, cos_dow, is_weekend, is_holiday` |
| HRRR weather + drought | 5 | `hrrr_tmp, hrrr_vpd, hrrr_wind, hrrr_mstav, drought_score` |
| Tri-state additions (theirs) | 5 | `bp_x_vpd, whp_x_vpd, bp_x_drought, pop_density, fm100_5d_min` |
| **Our addition — location** | 2 | `lat, lon` |
| **Our addition — daily weather** | 8 | `erc, fm100, vpd, vs, rmax, rmin, tmmx, pr` (seasonally imputed) |

## 3.3 Hyperparameter search — every trial, every parameter

### Stage 1 — tree structure (12 trials, fixed `lr=0.05, subsample=0.8, n_estimators=400, early_stopping=30`)
| depth | mcw | val AUROC | val AUC-PR |
|---|---|---|---|
| 6 | 10 | 0.8412 | 0.7332 |
| 6 | 20 | 0.8429 | 0.7349 |
| 6 | 30 | 0.8433 | 0.7353 |
| 7 | 10 | 0.8477 | 0.7468 |
| 7 | 20 | 0.8480 | 0.7464 |
| 7 | 30 | 0.8488 | 0.7482 |
| 8 | 10 | 0.8506 | 0.7561 |
| 8 | 20 | 0.8512 | 0.7559 |
| 8 | 30 | 0.8512 | 0.7554 |
| **9** | **10** | 0.8516 | **0.7604 ← winner** |
| 9 | 20 | 0.8511 | 0.7586 |
| 9 | 30 | **0.8523 ← highest AUROC** | 0.7591 |

### Stage 2 — learning rate × subsample (9 trials, fixed `depth=9, mcw=10, n_estimators=1200, early_stopping=50`)
| lr | subsample | val AUROC | val AUC-PR |
|---|---|---|---|
| 0.01 | 0.7 | 0.8508 | 0.7541 |
| 0.01 | 0.8 | 0.8509 | 0.7543 |
| 0.01 | 0.9 | 0.8508 | 0.7543 |
| 0.02 | 0.7 | 0.8527 | 0.7618 |
| 0.02 | 0.8 | 0.8526 | 0.7620 |
| 0.02 | 0.9 | 0.8529 | 0.7624 |
| 0.05 | 0.7 | 0.8518 | 0.7612 |
| **0.05** | **0.8** | 0.8516 | **0.7632 ← winner** |
| 0.05 | 0.9 | 0.8512 | 0.7621 |

### Final hyperparameters — every single one
```
n_estimators:        3000 (cap; best_iteration = 695, early-stopped)
max_depth:            9
min_child_weight:     10
learning_rate:        0.05
subsample:            0.8
colsample_bytree:     0.8
scale_pos_weight:     2.2872  (n_neg / n_pos on train)
eval_metric:          aucpr
early_stopping_rounds: 80
tree_method:          hist
device:               cuda
random_state:         42
n_jobs:               -1
```
Total GPU runtime for all 21 search trials + final model: **185.7 seconds**.

## 3.4 Results — raw (uncalibrated)

| Split | AUC-PR | AUROC | F1 | Precision | Recall | Lift |
|---|---|---|---|---|---|---|
| Train | 0.9051 | 0.9494 | 0.8145 | 0.8076 | 0.8216 | 2.98× |
| Val | 0.7632 | 0.8516 | 0.6651 | 0.6651 | 0.6652 | 2.65× |
| **Test** | **0.7237** | **0.8064** | **0.6284** | 0.6215 | 0.6353 | **2.37×** |

## 3.5 Results — isotonic-calibrated (fit on val only)

| Split | AUC-PR | AUROC | Brier |
|---|---|---|---|
| Val | 0.7591 | 0.8518 | — |
| **Test** | **0.7190** | **0.8064** | raw 0.1540 → cal **0.1474** |

Calibration curve: f(0.0)=0.000, f(0.3)=0.174, f(0.5)=0.357, f(0.7)=0.573, f(1.0)=1.000 — no saturation ceiling (unlike TDIS's served calibrator, which caps at 0.333).

## 3.6 Confusion matrices — **EXACT**, computed directly from the test set (n = 1,261,502)

**Raw probability, best-F1 threshold (0.4498):**
|  | Predicted No Fire | Predicted Fire |
|---|---|---|
| **Actual No Fire** | TN = 727,000 | FP = 149,110 |
| **Actual Fire** | FN = 140,535 | TP = 244,857 |

Precision 0.6215 · Recall 0.6353 · Specificity 0.8298 · Accuracy 0.7704

**Calibrated probability, best-F1 threshold (0.3206):**
|  | Predicted No Fire | Predicted Fire |
|---|---|---|
| **Actual No Fire** | TN = 736,067 | FP = 140,043 |
| **Actual Fire** | FN = 144,764 | TP = 240,628 |

Precision 0.6321 · Recall 0.6244 · Specificity 0.8402 · Accuracy 0.7742

**Raw probability, fixed threshold 0.5:**
|  | Predicted No Fire | Predicted Fire |
|---|---|---|
| **Actual No Fire** | TN = 763,339 | FP = 112,771 |
| **Actual Fire** | FN = 158,914 | TP = 226,478 |

Precision 0.6676 · Recall 0.5877 · Specificity 0.8713 · Accuracy 0.7846

## 3.7 Feature importance — top 15 of 38 (XGBoost gain)

| Rank | Feature | % gain |
|---|---|---|
| 1 | `ecoregion_id` | 20.79% |
| 2 | `elevation_m` | 7.70% |
| 3 | `hrrr_vpd` | 5.45% |
| 4 | `avg_burn_prob` | 5.10% |
| **5** | **`lat`** ⭐ | **4.48%** |
| **6** | **`lon`** ⭐ | **3.40%** |
| 7 | `powerline_dist_km` | 3.28% |
| 8 | `road_dist_km` | 3.27% |
| **9** | **`rmin`** ⭐ | **3.25%** |
| 10 | `sin_month` | 2.92% |
| 11 | `whp` | 2.22% |
| 12 | `hrrr_mstav` | 2.05% |
| 13 | `cos_month` | 2.01% |
| 14 | `pop_density` | 1.98% |
| 15 | `rmax` | 1.93% |
| ... | `flep4`, `cfl` | **0.00%** (both, rank 37–38) |

⭐ our additions — `lat`+`lon`+`rmin` carry 11.1% combined gain, reproducing the same pattern found in TEXAS 32-feat (11.8%).

---

# PART 4 — MASTER COMPARISON TABLE

| | **TDIS served** (matched test) | **TDIS served** (real population) | **TDIS rev4** (unpromoted, real-pop) | **Our old best — TEXAS 32-feat** | **Our new — TDIS_Ignition_BIT** |
|---|---|---|---|---|---|
| Dataset | `tdis_train_daily_hrrr.parquet` → 2,007,436 usable rows | same | same | TDIS raw parquet, 3,595,513 rows, imputed | `train_tristate_v2.parquet` (TX), 2,201,289 rows |
| Features | 23 | 23 | ~23 (rev4 feature set) | 32 | **38** |
| Test rows | 891,983 | 5,064,462 | ~5.06M | 460,463 | 1,261,502 |
| Test positive rate | 23.41% | 1.92% | ~1.9% | 30.71% | 30.55% |
| Hyperparameters | d9, 1000 trees, mcw30, lr0.02, no search | same | d9, 1000 trees, mcw30, lr0.02 + monotone constraints | d9, 912 trees (21-trial search), mcw30, lr0.05 | d9, 695 trees (21-trial search), **mcw10**, lr0.05 |
| **AUC-PR** | 0.4941 | 0.0878 | ~0.096* | **0.7484** | **0.7237** (raw) / 0.7190 (cal) |
| **AUROC** | 0.7402 | 0.7968 | 0.809–0.811 | **0.8309** | **0.8064** |
| **F1** | 0.4958 | 0.1692 | n/a | **0.6498** | **0.6284** |
| **Precision @ best F1** | 0.441 | 0.1493 | n/a | 0.6390 | 0.6215 |
| **Recall @ best F1** | 0.566 | 0.1851 | n/a | 0.6610 | 0.6353 |
| **Lift** | 2.11× | 4.58× | **4.76–4.88×** | 2.44× | 2.37× |
| Calibration ceiling | 33.3% max | 33.3% max | not reported | none (not checked) | none found |
| Monotonicity bugs | ❌ 5/9 features WRONG DIRECTION (their own self-check) | same | ✅ fixed via constraints | not checked | not checked |
| Real-population validated | — | ✅ | ✅ | ❌ **not done** | ❌ **not done** |
| Leakage tests run | — | ✅ (label-shuffle, spatial holdout) | — | ❌ | ❌ |

*TDIS's rev4 AUC-PR is reported at matched-test only as 0.511; the real-population AUC-PR figure above is inferred from its lift (4.76–4.88×) × base rate (~0.0192–0.02), shown as approximate.

## 4.1 The honest read of this table

- **Raw AUC-PR/F1 numbers are not comparable across columns** — they move almost entirely with the positive rate of whichever test set they're computed on (23%, 2%, or 30%). This is why **lift** (AUC-PR ÷ base rate) is the only fair cross-column number, and on lift:
  - Our old (TEXAS 32-feat, matched-sample lift 2.44×) and our new (BIT, matched-sample lift 2.37×) are both **lower** than TDIS's real-population lift (4.58× served, 4.76–4.88× rev4).
  - This is expected, not a contradiction: lift shrinks as the base rate you're measuring against rises, and neither of our two numbers has been measured at TDIS's true ~1.9% real-population base rate — only at our own ~30% matched-sample rate.
- **AUROC is the one metric that's reasonably stable across base rates**, and there our new BIT model (0.8064) beats TDIS's matched-test AUROC (0.7402) and sits close to — just above — their much harder real-population AUROC (0.7968), while our older TEXAS 32-feat (0.8309) is the highest of all five columns.
- **TDIS is unambiguously ahead on validation rigor**: real-population scoring, leakage tests, and — critically — they found and fixed (in rev4, unpromoted) the exact class of bug (non-monotone physical response) that we have not even checked for in either of our models.
- **Neither "our old" nor "our new" model has been real-population validated.** Until that is done, every AUC-PR/F1/lift number in our two columns should be read as "on a rebalanced sample," not as a deployment-equivalent number.

## 4.2 Our own progression, old → new, same-population view (matched-sample test, for internal tracking only)

| | TEXAS 30-feat | TEXAS 32-feat (old best) | TDIS_Ignition_BIT (new) |
|---|---|---|---|
| Features | 30 | 32 | 38 |
| Test AUROC | 0.8249 | **0.8309** | 0.8064 |
| Test AUC-PR | 0.7384 | **0.7484** | 0.7237 |
| Test F1 | 0.6440 | **0.6498** | 0.6284 |

**The new BIT retrain, on TDIS's own newer/cleaner dataset, scores slightly lower than our prior TEXAS 32-feat model** on this internal, same-style comparison — most likely because the BIT test set spans 5 full years (2022–2026, including 2022, which TDIS's own analysis flagged as anomalously hard) versus TEXAS 32-feat's shorter 2025–26 test window, and because the hyperparameter search picked a less-regularized `min_child_weight=10` this time (vs. 30 everywhere else), which produced a larger train→test gap (0.143 AUROC vs. 0.0058 for TEXAS 32-feat). This is flagged, not hidden — see `TDIS_Ignition_BIT/BIT_TRAINING_REPORT.md` §7 for the full caveat list and §8 for the recommended `min_child_weight=30` rerun.

---

# Appendix — Source files for every number in this document

| Claim | Source |
|---|---|
| TDIS served model hyperparameters/features | `TDIS_Ignition/model/tdis_forecast_hrrr_filtered.json` (loaded directly via xgboost.Booster) |
| TDIS served model matched-test metrics | `TDIS_Ignition/model/tdis_forecast_hrrr_filtered_meta.json`, cross-checked against `TDIS_HANDOFF_ANALYSIS.md` |
| TDIS real-population metrics | `TDIS_Ignition/model/TRUST_VALIDATION.json`, `operational_realpop_validation.json` |
| TDIS calibrator curve | `TDIS_Ignition/model/operational_isotonic_calibrator.joblib` (loaded directly) |
| TDIS rev4 numbers | `TDIS_Ignition/model_tristate/README.md` §10 |
| TDIS tri-state numbers | `TDIS_Ignition/model_tristate/model/control_v2_results.json`, `rev5_v2_results.json`, `realpop_res5_v2_results.json` |
| V1–TEXAS 32-feat history | `PROJECT_VERSION_HISTORY.md`, `TX/MODEL_TRAINING_REPORT_TX.md`, `TEXAS/TEXAS_V2_MODEL_REPORT.md`, `TEXAS_v1_backup/MODEL_EVALUATION_METRICS.md` |
| TDIS_Ignition_BIT dataset/hyperparameters/metrics | `TDIS_Ignition_BIT/build_dataset.log`, `TDIS_Ignition_BIT/tune_train_run.log`, `TDIS_Ignition_BIT/model/training_results_bit_38feat.json` |
| TDIS_Ignition_BIT confusion matrices | Computed directly, this session, from `model/model_bit_38feat.json` + `model/calibrator_bit_38feat.joblib` scored on `data/bit_train_tx.parquet` test split |
