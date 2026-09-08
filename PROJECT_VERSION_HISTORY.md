# IgnitionNet / Texas-California Wildfire ML Project — Chronological Version History

**Compiled:** 2026-08-31
**Sources:** Git history (`git log`), `V1/`, `V2/`, `TX/`, `TEXAS/`, `Supporting_files/`, `Focused_Files/`, and root-level project docs (`IGNITIONNET_PROJECT_SCOPE_MASTER.md`, `TDIS_HANDOFF_ANALYSIS.md`, `DATA_QUALITY_REVIEW.md`, `DATASET_COLUMNS_COMPARISON.md`, `TEAM_DATA_GUIDE.MD`, various README/report files).
**Method note:** Only facts found directly in code, configs, logs, docs, or git history are included. Anything not found is marked **Not available**. Where source documents contradict each other, both values are reported with the discrepancy flagged.

---

## 1. Summary Table

| Version | Model | Major Changes | Accuracy / AUROC | Other Metrics |
|---|---|---|---|---|
| **V1** (Jul 2026) | LightGBM (winner) vs XGBoost/RandomForest, threshold 0.46 | First working prototype. NASA FIRMS/VIIRS 2024 fire points + annual GEE composite features (NDVI, LST, wind, DEM, etc.), 18 features, chronological train/val/test split within 2024 | Test AUROC **0.9142**, Accuracy 0.8276 | Test AUPR 0.7549, F1 0.7236, Precision 0.6173, Recall 0.8744. **Flagged as unreliable** — annual composite features cause label leakage; documented as "must not be quoted to TDIS" |
| **V2 baseline** (15 Jul 2026) | XGBoost (leakage present) | Switched to FPA-FOD (2014–2020) + H3 hex grid (Res. 8) + gridMET daily weather; TX dataset 376,233 rows × 40 cols | Test AUROC **0.9900** (inflated) | AUPR/F1 not primary — later invalidated |
| **V2 clean** (15 Jul 2026, commit `544d89b`) | XGBoost, leakage columns removed | Removed `fire_count`/`has_fire_history` (computed using future test-year data) | Test AUROC **0.8569** | AUPR 0.3978, F1 0.4082, Precision 0.3315, Recall 0.5313 |
| **TX v1 (HRRR-folder baseline)** | XGBoost, LANDFIRE zero-filled | Restructured into `TX/` folder; H3 resolution bug fixed (R7→R8, commits `0a265c7`/`a50995b`); LANDFIRE columns present but unpopulated | Test AUROC **0.8569** | AUPR 0.3978, F1 0.408 |
| **TX v2 (real LANDFIRE)** | XGBoost | Added real LandFIRE LF2022 rasters (avg_burn_prob, whp, flep4, cfl, cbd, cbh) | Test AUROC **0.8637** | AUPR 0.4106, F1 0.417 |
| **TX v3 (tuned)** (commit `544d89b`→`6a31ca2` era, Jul 2026) | XGBoost, 21-trial tuned | Hyperparameter search (depth/min_child_weight/lr/subsample grid); fixed 5-day rolling-stat bug (84.4% NaN) and gridMET nodata-scaling bug | Test AUROC **0.8687** | AUPR 0.4247, F1 0.429, Precision 39.2%, Recall 47.5% |
| **TX-HRRR (sub-daily)** (Aug 2026, commits `66f73f9`/`de902e0`) | XGBoost + HRRR sub-daily features | Added 8 HRRR features; found and fixed a critical RH-unit bug (`fix_hrrr_rh_bug.py`, 100,981 rows corrupted 2014–2016); HRRR did **not** improve performance | Test AUROC **0.8682** (best HRRR variant) | AUPR 0.4225 — essentially flat vs. gridMET-only, documented as a negative result |
| **TDIS served model** (promoted 2026-08-17, external team) | XGBoost, 1,000 trees, 22 features | Reproduced independently in `TDIS_HANDOFF_ANALYSIS.md`; flare-contamination filter applied (removed 531 persistent industrial-hotspot cells) | Test AUROC **0.7402** (raw); flare-filtered AUROC 0.711 on flare-cleaned subset | AUC-PR 0.4941, F1 0.4958; real-population deployment score AUC-PR 0.0878 / AUROC 0.7968 / Lift 4.58× |
| **TEXAS 30-feat model** (27 Aug 2026, latest/uncommitted) | XGBoost, 30 features, isotonic calibration | Rebuilt from TDIS raw HRRR parquet (3.59M rows); new imputation strategy (cross-source substitution + seasonal median) eliminated all missing values (24.09%→0%) | Test AUROC **0.8249** | Test AUC-PR 0.7384, F1 0.6440, Precision 0.6435, Recall 0.6445, Lift 2.40×; calibrated ECE 0.0018 (vs 0.2799 raw) |

*Note on "Accuracy" column:* Only V1 reports a literal accuracy percentage (0.8276 test). All later stages (V2 onward) are evaluated primarily via AUROC/AUPR/F1 because of severe class imbalance (~9–30% positive rate depending on version), which is documented in the project's own reports as the reason accuracy is not the headline metric.

---

## 2. Git Commit Timeline (chronological, oldest → newest)

| Date | Commit | Message |
|---|---|---|
| 2026-06-30 | `eb063e4` | Phase 1 + EDA complete: preprocessing pipeline, 7 EDA scripts per state (TX + CA) |
| 2026-06-30 | `b84213d` | Full project: V1 + V2 code, EDA outputs, maps, reports — data files excluded |
| 2026-07-02 | `4f02248` | Add root-level README so GitHub displays it on repo landing page |
| 2026-07-02 | `c401d85` | Merge pull request #1 from navaneethakrishnanms/clean-main |
| 2026-07-03 | `8d627fc` | Phase 1 complete + Phase 2A Feature Finalization |
| 2026-07-03 | `e3a0147` | Phase 2B: H3 Grid Construction complete |
| 2026-07-10 | `2a105fd` | Phase 2C-2G: Fire labeling, negative sampling, static/gridMET extraction, assembly pipeline + gitignore update |
| 2026-07-10 | `e0eb09f` | docs: rewrite README with full IgnitionNet documentation |
| 2026-07-13 | `65bd4d8` | Phase 2G complete: dataset assembly TX + diagnostic scripts + missing data report |
| 2026-07-13 | `a6d1c55` | Dataset creation code changes |
| 2026-07-13 | `89478ff` | Update README + Phase 2F production final (cross-year lag fix, cache-based 5D stats) |
| 2026-07-14 | `6a31ca2` | Phase 2F scale fix + Phase 3 trainer + dataset report + gitignore update |
| 2026-07-15 | `9833c01` | dataset_report_md |
| 2026-07-15 | `a50995b` | Fix stale R7 references — TX dataset confirmed as H3 R8 from actual cell data |
| 2026-07-15 | `544d89b` | Phase 3 complete: XGBoost baseline trained (AUROC 0.857, AUPR 0.398) |
| 2026-07-17 | `628c9ff` | first commit |
| 2026-07-17 | `f686253` | Clean up README and add comment placeholder |
| 2026-07-20 | `3fb2f60` | MD file updates |
| 2026-07-20 | `809c276` | Resolve merge conflict in README.md |
| 2026-07-20 | `0a265c7` | Fix H3 resolution: correct R7 to R8 throughout (Res. 8, ~0.87km width, 317,142 unique cells) |
| 2026-07-22 | `7df0d28` | Map code changes |
| 2026-07-27 | `0cc6e74` | Add Texas HRRR pipeline + training/tuning scripts + results |
| 2026-07-28 | `c15b38a` | Restructure: move all Texas pipeline files into TX/ folder |
| 2026-07-28 | `c54e38c` | Fix .gitignore for TX/ nesting and untrack dataset files |
| 2026-08-04 | `66f73f9` | Add HRRR pipeline, training/tuning scripts, analysis reports, and model metadata |
| 2026-08-12 | `de902e0` | Add TX risk map scripts and output figures/maps |
| 2026-08-19 | `a54e486` | Add Supporting_files, Focused_Files, TDIS handoff analysis, and parquet converter |
| *uncommitted* | — | `TEXAS/` folder, `DATA_QUALITY_REVIEW.md`, `DATASET_COLUMNS_COMPARISON.md` — newest work, internally dated 27–31 Aug 2026, not yet committed to git as of this report |

---

## 3. Detailed Version-by-Version History

### V1 — Proof of Concept

**Model:** LightGBM (selected winner) vs. XGBoost (2 configs, "A" and "B") vs. Random Forest ("C") — 4 candidate models compared.

**Dataset:** NASA FIRMS/VIIRS 2024 fire detections (`V1/version 1/fire_archive_M-C61_760762.csv`). Features are **annual** Google Earth Engine composites: NDVI, EVI, LST, Temperature, Wind, Rainfall, DEM, Slope, Aspect, LandCover, plus calendar features (month, day_of_year, season_code, sin/cos_month, sin/cos_doy, is_peak_fire_season) — **18 features total**.

**Split:** Chronological within 2024 — Train Jan–Aug 2024, Validation Sep 2024, Test Oct–Dec 2024 (test set = 2,250 rows).

**Hyperparameters** (`V1/configs/config.yaml`):
- XGBoost: `n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=3.0, eval_metric=aucpr, early_stopping_rounds=50, tree_method=hist, device=cuda`
- Random Forest: `n_estimators=500, max_depth=20, min_samples_leaf=5, max_features=sqrt, class_weight=balanced`
- LightGBM: `n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, is_unbalance=true, early_stopping_rounds=50`

**Libraries** (`V1/requirements.txt`): numpy≥1.24, pandas≥2.0, geopandas≥0.13, shapely≥2.0, rasterio≥1.3, xgboost≥1.7, scikit-learn≥1.2, shap≥0.41, lightgbm≥3.3, gradio≥4.0.

**Results (best model: LightGBM, decision threshold = 0.46, `V1/models/optimal_threshold.json`):**
| Split | AUROC | AUPR | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| Validation | 0.8994 | 0.6905 | 0.6915 | 0.5821 | 0.8516 | 0.8187 |
| Test | 0.9142 | 0.7549 | 0.7236 | 0.6173 | 0.8744 | 0.8276 |

Test confusion matrix: TP 508 / FP 315 / TN 1,354 / FN 73 (n=2,250).

Comparison of the 4 candidates: XGBoost A (195/600 trees, AUROC 0.8868/AUPR 0.6691), XGBoost B (456/600 trees, AUROC 0.8871/AUPR 0.6761), Random Forest C (AUROC 0.8831/AUPR 0.6611), LightGBM (winner, 206/600 trees).

Top SHAP features: Temperature (0.10583), Wind (0.08204), Rainfall (0.06085), DEM (0.04556), LST (0.01613), EVI (0.01360), NDVI (0.01303).

**Known issue — why V1 was superseded:** `V1/project_audit.md` documents that using *annual* composite features means the model effectively learns "which month/location has historically had fires" rather than performing genuine short-horizon (24h) ignition forecasting — i.e., a form of temporal leakage. This is explicitly flagged in `TDIS_HANDOFF_ANALYSIS.md`: **"V1 must not be quoted to the TDIS team as a result."** The small test set (2,250 rows) also limits confidence in the reported metrics.

---

### V2 — Production-Style Pipeline (FPA-FOD + H3 Grid)

**Why it changed:** Move from annual composite/GEE-derived features (V1) to a proper daily-resolution, spatially-gridded pipeline using an established fire-occurrence dataset and true daily weather, to fix V1's temporal-leakage problem.

**Model:** XGBoost.

**Dataset:** FPA-FOD (USDA Forest Service Fire Program Analysis – Fire Occurrence Database), 2014–2020. Raw: TX 51,033 fire records / 309 columns, CA 50,881 / 309 columns. Spatial index: H3 hexagonal grid, **Resolution 8** (~0.73 km edge, ~0.74 km² per cell) — this replaced an earlier, incorrect assumption of Resolution 7 (fixed in commits `a50995b` and `0a265c7`, both 2026-07-15/20).

Final TX assembled dataset: **376,233 rows × 40 columns**, 317,142 unique H3 cells, label balance 34,203 fire (9.1%) / 342,030 non-fire (90.9%).

**Split (chronological by year):**
| Split | Years | Rows | Fire rows | Fire rate |
|---|---|---|---|---|
| Train | 2014–2017 | 252,066 | 22,916 | 9.1% |
| Val | 2018 | 61,181 | 5,561 | 9.1% |
| Test | 2019–2020 | 62,986 | 5,726 | 9.1% |

**Environment** (`V2/environment.yml`): conda env `torch_gpu`, Python 3.10, pandas, pyarrow, numpy, seaborn, matplotlib, scipy, folium, openpyxl.

**Critical bug found and fixed (leakage):** `fire_count` and `has_fire_history` features were computed from the *full* 2014–2020 dataset — including test-period years — before the train/val/test split was applied. This let the model see future fire information, inflating **Test AUROC to 0.9900**. Root cause and fix documented directly in the pipeline reports; the two leakage columns were removed from `FEATURE_COLS` and the model retrained.

**Results — clean XGBoost baseline (leakage removed):**
| Split | AUROC | AUPR | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Train | 0.9302 | 0.5723 | 0.5386 | 0.4153 | 0.7659 |
| Val | 0.8742 | 0.4125 | 0.4316 | 0.3355 | 0.6049 |
| Test | **0.8569** | 0.3978 | 0.4082 | 0.3315 | 0.5313 |

Test confusion matrix (threshold 0.6841, max-F1 on val): TN 51,125 / FP 6,135 / FN 2,684 / TP 3,042.

Top features (clean model): `burnable` 26.0% (654.7 gain), `centroid_lon` 13.4%, `centroid_lat` 8.5%, `erc_5D_max` 6.8%, `vs_5D_max` 6.4%.

**Other data-quality note:** All 4 LANDFIRE features (`avg_burn_prob`, `whp`, `flep4`, `cfl`) were zero-filled placeholders at this stage — the rasters had not yet been downloaded/joined (fixed in the TX pipeline stage, below).

**Columns dropped by team review as redundant/leaky:** `bi` (redundant with `erc`), `tmmn` (redundant with `tmmx`), `fm1000` (redundant with `fm100`), `sph` (redundant with `vpd`/`rmax`/`rmin`), `ecoregion_l2`/`l3`, `h3_resolution`, `state_x`/`state_y`.

---

### TX — HRRR/LandFIRE Pipeline (Restructured `TX/` folder)

**Why it changed:** V2's code was restructured into a dedicated `TX/` folder (commit `c15b38a`, 2026-07-28); real LandFIRE rasters were joined in (replacing the zero-filled placeholders); hyperparameters were tuned; and sub-daily HRRR weather was tested as an additional feature source.

**Dataset:** `TX/final_training_dataset_tx_22.07.2026_landfire.xlsx` — 376,233 rows, 42 columns, 317,142 unique H3-8 cells, date range 2014-01-01–2020-12-06, positive rate 9.09%. Data-quality notes: 454 duplicate rows, 24,954 rows (6.6%) missing weather. New LandFIRE LF2022 features: `cbd` (canopy bulk density, mean 0.012, 85.2% zero), `cbh` (canopy base height, mean 0.632, 85.2% zero), `avg_burn_prob` (scale 0–11, **not** 0–1), `cfl` (class-mapped 0–110 ft), `flep4` (class-mapped 0–0.9 probability).

After preprocessing (dupes dropped): 375,779 rows. Split: Train 2014–2017 (251,785 rows, 22,635 fire), Val 2018 (61,139, 5,519), Test 2019–2020 (62,855, 5,595). **34 of 42 columns used as features** (5 identifiers + label + the 2 leakage columns excluded).

**Model iteration 1 (`TX v1`, LANDFIRE still zeros):** XGBoost, `max_depth=7, min_child_weight=30, learning_rate=0.05, subsample=0.8`, 387 trees.
- Test: AUROC 0.8569, AUPR 0.3978, F1 0.408 (matches V2 clean baseline — LANDFIRE columns not yet populated).

**Model iteration 2 (`TX v2`, real LandFIRE joined):** Same hyperparameters, 330 trees.
- Test: AUROC **0.8637**, AUPR 0.4106, F1 0.417.

**Two additional bugs found and fixed during this phase** (`TX/missing_data_diagnosis.md`):
1. 5-day rolling weather statistics were 84.4% NaN because rolling stats were computed on the sparse per-cell row set (1–2 rows/cell) instead of pulling the true 5 preceding days from the source NetCDF weather files.
2. gridMET's fill/nodata value (32767) was leaking through as real data because the nodata check ran *after* the scale_factor/add_offset unit conversion instead of before it.

Both were fixed in the "Phase 2F production final" work (commit `89478ff`, "cross-year lag fix, cache-based 5D stats").

**Model iteration 3 (`TX v3`, tuned — `TX/tune_tx.py`):** 21-trial hyperparameter search (12 tree-structure trials + 9 learning-rate × subsample grid trials). Best config: `max_depth=9, min_child_weight=30, learning_rate=0.01, subsample=0.9, colsample_bytree=0.8, colsample_bylevel=0.8, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=10.12`, 1,627 trees (early-stopped from a cap of 1,706).
- Val: AUROC 0.8856, AUPR 0.4452. **Test: AUROC 0.8687, AUPR 0.4247, F1 0.429**, Precision 39.2%, Recall 47.5%, Specificity 92.8% (at threshold 0.710). Confusion matrix: TN 53,129 / FP 4,131 / FN 2,937 / TP 2,658.
- Train AUROC 0.9550 vs. Test 0.8687 — a train/test generalization gap documented and accepted as expected for this feature set.
- Top features: `burnable` 19.8% (560.7 gain), `avg_burn_prob` 12.1%, `whp` 10.6%, `cfl` 9.5%, `centroid_lon` 7.9%.

**TX-HRRR sub-daily weather experiment** (commits `66f73f9`, `de902e0`, Aug 2026): Added 8 HRRR-derived sub-daily features (`temp_pw`, `rh_pw`, `wind_pw`, `vpd_pw_hrrr`, `hpbl_pw`, `dswrf_pw`, `hrrr_pw` flag, `hrrr_rh_valid` flag). HRRR data coverage rose from 12.2% (2014) to 95–99% (2018–2020).

**Critical HRRR bug found and fixed** (`TX/fix_hrrr_rh_bug.py`): for 2014–2016, HRRR files lacking the `:RH:2 m above ground:anl` field silently fell back to specific-humidity (`:SPFH:`) values (~0.001–0.02 kg/kg), which were then mislabeled as 0–100% relative humidity, corrupting the derived `vpd_pw_hrrr` feature. This affected **100,981 rows (33% of matched HRRR rows)**, all in 2014–2016. Fix: null out `rh_pw`/`vpd_pw_hrrr` wherever `rh_pw < 1.0`, and add a `hrrr_rh_valid` flag column.

**Result of the HRRR experiment:** 6 model iterations were tested; sub-daily HRRR weather **did not improve accuracy**. The best independently-tuned HRRR model (21 trials, `max_depth=9, min_child_weight=30, learning_rate=0.02, subsample=0.9`) scored **AUROC 0.8682, AUPR 0.4225** — essentially flat and slightly below the gridMET-only tuned model (0.8687/0.4247). HRRR features contributed only 7.8% combined feature-importance gain. This was recorded as a negative result, not adopted for the production model.

---

### TEXAS/ — Latest 30-Feature Model (uncommitted, 27–31 Aug 2026)

**Why it changed:** Rebuilt from the raw TDIS-team parquet dataset with a new missing-data strategy, aiming to retain 100% of rows (previous pipelines dropped or NaN-filled rows with missing weather) and add isotonic probability calibration.

**Source dataset:** `tdis_train_daily_hrrr.parquet` — 3,595,513 rows × 33 columns (+2 merged columns = 35), sourced via `Focused_Files/`/`Supporting_files/` TDIS deliverable data.

**Data-quality issues found:** gridMET fields 77.13% missing, HRRR forecast fields 38.65% missing, static/topography fields 5.62% missing; 28,578,376 missing cells total (24.09% of all cells); 85.66% of rows had at least one missing value.

**Imputation strategy** (`TEXAS/SESSION_SUMMARY_27AUG2026.md`):
1. Cross-source substitution — `hrrr_tmp`←`tmmx`, `hrrr_vpd`←`vpd`, `hrrr_wind`←`vs`, for 2014–mid-2018 (before HRRR forecast coverage was reliable).
2. Month-of-year seasonal median for remaining weather gaps.
3. Spatial/categorical median or mode for static topography fields.

Outputs: `TEXAS/tdis_train_daily_hrrr_raw.parquet` (3,595,513 × 35, 30,368,531 NaN) → `TEXAS/tdis_train_daily_imputed.parquet` (3,595,513 × 35, **0 NaN**).

**Model:** XGBoost `XGBClassifier`, **30 features** (excludes `h3_cell`, `date`, `label`, `year`, `split`). Code: `TEXAS/train_model_30feat.py`.

**Split (chronological, 12+ year range):**
| Split | Years | Rows | Positive rate |
|---|---|---|---|
| Train | 2014-01-01 – 2023-12-31 | 2,761,329 | 26.6% |
| Val | 2024 | 287,649 | 29.4% |
| Test | 2025-01-01 – 2026-07-29 | 460,463 | 30.7% |

**Hyperparameters** (`TEXAS/training_results_30feat.json`): `n_estimators=1000` (early stopping patience 50), `max_depth=9`, `min_child_weight=30`, `learning_rate=0.02`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=2.7616`, `eval_metric=aucpr`, best iteration 997. Hardware: RTX 3060 12GB, CUDA. Training time: 68.33 seconds.

**Results — raw scores:**
| Split | AUC-PR | AUROC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Train | 0.7458 | 0.8516 | 0.6452 | 0.6391 | 0.6515 |
| Val | 0.7338 | 0.8208 | 0.6317 | — | — |
| Test | **0.7384** | **0.8249** | **0.6440** | 0.6435 | 0.6445 |

Test Lift: 2.40×.

**Results — isotonic-calibrated (fit on validation set):**
| Split | AUC-PR | AUROC | F1 |
|---|---|---|---|
| Val | 0.7298 | 0.8211 | — |
| Test | 0.7337 | 0.8249 | 0.6440 |

Calibration was near-neutral for ranking metrics; its purpose is well-calibrated probabilities rather than improved discrimination. `TEXAS/MODEL_EVALUATION_METRICS.md` frames the 0.007 train-test AUC-PR gap as evidence the model is not meaningfully overfitting.

**Top feature importances:** `elevation_m` 0.0998, `ecoregion_id` 0.0777, `rmin` 0.0743, `powerline_dist_km` 0.0597, `avg_burn_prob` 0.0590, `road_dist_km` 0.0576, `hrrr_vpd` 0.0487, `sin_month` 0.0463. Notably, `flep4` and `cfl` (LandFIRE features that mattered in the TX pipeline) have **exactly 0.0 importance** in this model — a dead-feature finding.

**Comparison vs. the externally-served "TDIS" model** (22 features, different test period/positive rate — explicitly flagged in the source doc as *not directly comparable*):
| Model | AUC-PR | AUROC | F1 |
|---|---|---|---|
| TDIS served model | 0.4941 | 0.7402 | 0.4958 |
| TEXAS 30-feat model | 0.7384 | 0.8249 | 0.6440 |

---

### External Reference — TDIS Served Model (independently reproduced, `TDIS_HANDOFF_ANALYSIS.md`, 2026-08-19)

This is not a version this project trained originally, but a separately-maintained model the project team reproduced/audited for a handoff review.

- Model: XGBoost, 1,000 trees, `max_depth=9, min_child_weight=30, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8`, 22 features, H3 Resolution 8, 1,708,940 TX-bounding-box cells. Promoted to production 2026-08-17.
- Environment (`Supporting_files/`): conda env `UAI2526`, Python 3.10.18, xgboost 3.1.1, h3 4.3.1, herbie-data 2025.12.0, scikit-learn 1.7.2, pandas 2.3.2.
- Reproduced test metrics (891,983 rows): AUC-PR 0.4941, AUROC 0.7402, best-F1 0.4958.
- **Flare contamination found:** ~27% of positive labels were persistent industrial gas-flare hotspots, not wildfires. Removing 531 persistently-hot cells (>3% persistence) raised AUC-PR from 0.386→0.450 and AUROC from 0.651→0.711 on the affected subset.
- Real deployment-population score (5,064,462 cell-days, 2024–2026, true base rate 0.0192): AUC-PR 0.0878, AUROC 0.7968, Lift 4.58×.
- Calibration: Expected Calibration Error (ECE) improved from 0.2799 (raw) to 0.0018 (calibrated, 2026 holdout).
- Event validation: 5/5 real Texas wildfires captured (Smokehouse Creek 2024-02-26, Crabapple, Lavender, Hunggate, Windy Deuce fires).
- 17 defects (D.1–D.17) were documented in the TDIS handoff package review, including: missing `hwp_params.json`, a stale `meta.json` (claims 20 features when the model actually uses 22), missing calibration script, 4 conflicting split definitions across files, and anomalous 2022 performance vs. later years.

---

## 4. Data-Quality Issues Identified Across the Project (chronological)

| Issue | Found in | Impact | Fix |
|---|---|---|---|
| Annual-composite temporal leakage | V1 | Inflated apparent skill; model not truly forecasting | Redesigned pipeline in V2 using daily FPA-FOD + gridMET data |
| H3 resolution mislabeled as R7 | V2/TX (fixed commits `a50995b`, `0a265c7`, Jul 2026) | Grid-cell size and count assumptions wrong throughout early docs | Corrected to Resolution 8 (317,142 cells) everywhere; some stale R7 references remain in root `README.md`'s Group-5 feature description (unresolved inconsistency) |
| `fire_count`/`has_fire_history` computed over full date range including test years | V2 (found & fixed 15 Jul 2026, commit `544d89b`) | Test AUROC inflated from 0.857 to 0.990 | Columns removed from feature set |
| 5-day rolling weather stats 84.4% NaN | TX pipeline (`TX/missing_data_diagnosis.md`) | Weather features mostly missing | Recomputed rolling stats from true preceding NetCDF days with caching (commit `89478ff`) |
| gridMET nodata value (32767) leaking as real data | TX pipeline (`TX/missing_data_diagnosis.md`) | Corrupted weather values | Reordered nodata check before scale/offset conversion (commit `6a31ca2`, "Phase 2F scale fix") |
| HRRR relative humidity mislabeled from specific humidity, 2014–2016 | TX-HRRR pipeline (`TX/fix_hrrr_rh_bug.py`) | 100,981 rows (33% of matched HRRR rows) had corrupted `rh_pw`/`vpd_pw_hrrr` | Null out invalid RH values, add `hrrr_rh_valid` flag |
| LANDFIRE features (`avg_burn_prob`, `whp`, `flep4`, `cfl`) zero-filled placeholders | V2 | Missing signal from 4 features | Real LandFIRE LF2022 rasters joined in TX v2 iteration |
| Gas-flare industrial hotspots mislabeled as wildfire positives | TDIS served model (`TDIS_HANDOFF_ANALYSIS.md`) | ~27% of positive labels contaminated | Persistence-based flare filter (removed 531 cells) |
| Fabricated future-dated rows (label=0, 0% weather) inside test split through 2026-12-31 | `DATA_QUALITY_REVIEW.md`, TDIS package | Test split included dates beyond the last real labeled date (2026-07-29) | Documented as unresolved critical defect (D-level) at time of report |
| 53.9% of "Texas" grid cells fall outside Texas state boundary (bounding box vs. state-clipped) | `DATA_QUALITY_REVIEW.md` | Spatial mislabeling of "Texas" coverage | Documented as unresolved defect |
| Calibrator saturates at 18.71% max probability; calibrate-then-average ordering error | `DATA_QUALITY_REVIEW.md` | Understated high-risk probabilities in aggregated views | Documented as unresolved defect |
| `vpd` sourced from FPA-FOD instead of gridMET | `TEAM_DATA_GUIDE.MD` | ~0.015 AUROC inflation if not corrected | Documented; re-sourced from gridMET in later pipelines |

---

## 5. Overall Improvement Trajectory (Initial → Final)

- **Feature richness:** 18 annual composite features (V1) → 40 daily/spatial features (V2) → 42 features with real LandFIRE rasters (TX) → 30 selected, fully-imputed features with cross-source substitution (TEXAS latest).
- **Dataset scale:** ~few thousand VIIRS points (V1, 2024 only) → 376,233 rows / 317,142 H3 cells (V2/TX, 2014–2020) → 3,595,513 rows (TEXAS latest, TDIS parquet, extended date range through 2026).
- **Label integrity:** V1's temporal leakage → V2's discovered feature-leakage (fixed) → TDIS model's discovered flare-contaminated labels (fixed) → TEXAS latest model built on a cleaned/imputed dataset (though `DATA_QUALITY_REVIEW.md` flags newer unresolved defects in the TDIS-supplied test split).
- **Weather granularity:** Annual composites (V1) → daily gridMET (V2/TX) → sub-daily HRRR tested but not adopted for accuracy gains (TX-HRRR) → HRRR reintroduced as an imputation source (not primary signal) in TEXAS latest.
- **Model calibration:** No calibration in V1/V2/TX → isotonic calibration introduced in the TDIS served model and the TEXAS latest model, substantially reducing Expected Calibration Error (TDIS: 0.2799 → 0.0018).
- **Metric honesty:** Early reports (V1, initial V2) present single strong numbers; later reports (TX, TDIS handoff, TEXAS latest) explicitly discuss and correct for leakage, class imbalance, and label contamination, and document negative results (e.g., HRRR not improving accuracy) rather than omitting them.
- **Realistic difficulty:** AUROC trends downward from V1's 0.9142 to V2/TX's ~0.86–0.87 to TDIS's 0.74 on its true holdout — this reflects the removal of leakage and the move to harder, more realistic prediction settings, not a regression in true model quality. The TEXAS latest model (AUROC 0.8249, AUC-PR 0.7384) represents the most feature-rich, most leakage-scrutinized, and most calibrated version to date, trained on the largest dataset.

---

## 6. Items Marked "Not Available"

- Exact accuracy (as opposed to AUROC/AUPR/F1) for V2, TX, TDIS, and TEXAS models — these reports do not compute a plain accuracy metric, likely due to class imbalance; only V1 reports it.
- mAP (mean Average Precision) — not used anywhere in this project; it is an object-detection metric and does not apply to this tabular binary-classification task.
- CI/CD or automated test logs — none found in the repository.
- Production monitoring/drift metrics post-deployment for the TDIS served model, beyond the point-in-time evaluation in `TDIS_HANDOFF_ANALYSIS.md`.
- Resolution status of the "fabricated future-dated rows" and "53.9% out-of-state grid cells" defects — flagged in `DATA_QUALITY_REVIEW.md` as open at time of writing; no follow-up commit found addressing them.
