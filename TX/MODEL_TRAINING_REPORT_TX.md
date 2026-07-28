# Texas Wildfire Ignition Model — Complete Training & Tuning Report

**State:** Texas  
**Report Date:** 2026-07-27  
**Dataset:** `final_training_dataset_tx_22.07.2026_landfire.xlsx`  
**Model type:** XGBoost Binary Classifier  
**Task:** Predict wildfire ignition discovery risk per H3-8 hexagonal cell per 6-hour UTC window

---

## 1. Project Background

IgnitionNet predicts **where and when** a wildfire will be *discovered* in Texas, at a spatial resolution of ~0.74 km² (H3 resolution-8 hexagons) and a temporal resolution of 6-hour UTC windows.

| Property | Value |
|---|---|
| Label source | FPA-FOD v6 (USDA Forest Service fire occurrence database) |
| Positive label | Fire discovered in that cell during that 6-hour window |
| Negative sampling | DAY_MATCHED — 10 non-fire cells per fire event, same calendar date + window |
| State | Texas |
| Date range | 2014–2020 (7 years) |
| Total rows | 376,233 (before preprocessing) |
| Positive rate | 9.09% (by design, preserved across all years) |
| Primary metric | **AUPR** (Average Precision-Recall) — not accuracy |
| Secondary metric | **AUROC** |
| Accuracy | Never reported — predicting "no fire" everywhere = 90.9% accuracy (useless) |

---

## 2. Dataset — 42 Columns, 34 Features

The dataset has **42 columns** but only **34 are used as features**. The rest are excluded:

| Category | Columns | Count | Used? |
|---|---|---|---|
| **Identifiers** | `h3_cell`, `date_utc`, `window_hour`, `window_6h_utc`, `fire_year` | 5 | No |
| **Target** | `label` | 1 | No (target, not input) |
| **Flag** | `gridmet_missing` | 1 | Yes (as availability signal) |
| **Leakage** | `fire_count`, `has_fire_history` | 2 | No — LEAKAGE |
| **Landscape** | `avg_burn_prob`, `whp`, `flep4`, `cfl`, `cbd`, `cbh`, `burnable` | 7 | Yes |
| **Daily weather** | `erc`, `fm100`, `vpd`, `vs`, `rmax`, `rmin`, `tmmx`, `pr` | 8 | Yes |
| **5-day stats** | `erc_5D_*`, `fm100_5D_*`, `vpd_5D_*`, `vs_5D_*`, `rmax_5D_*`, `tmmx_5D_*` | 12 | Yes |
| **Temporal** | `sin_month`, `cos_month`, `sin_hour`, `cos_hour` | 4 | Yes |
| **Location** | `centroid_lat`, `centroid_lon` | 2 | Yes |

**42 total − 8 excluded = 34 features used for training.**

### Why `fire_count` and `has_fire_history` are EXCLUDED

These columns were computed from the **full 2014–2020 FPA-FOD dataset** including test years.
A fire cell always has `fire_count >= 1` by definition. The model trivially learns:

```
fire_count > 0  ->  predict fire   (was 58% of gain, AUROC = 0.990 — cheating)
```

This was discovered and fixed before the final training run.

---

## 3. Preprocessing Steps Applied

All steps sourced from `README_tx_dataset.md`:

| Step | Action | README Reference | Rows After |
|---|---|---|---|
| **Load** | Read Excel (94 MB, 376,233 rows x 42 cols) | — | 376,233 |
| **Date fix** | Convert `date_utc`, `window_6h_utc` to datetime | pyarrow requirement | 376,233 |
| **Step 1** | Drop 454 duplicate rows on `(h3_cell, date_utc, window_hour)` | s8 line 194 | 375,779 |
| **Step 2** | Keep `gridmet_missing=1` rows — XGBoost handles NaN natively | s9 line 200 | 375,779 |
| **Step 3** | Zero-fill 1,581 NaN `burnable` / `fire_count` (boundary cells) | s10 line 203 | 375,779 |
| **Step 4** | `avg_burn_prob` kept on 0–11 scale — NO normalization | s4 line 183 | 375,779 |

### Why gridmet_missing rows were KEPT (not dropped)

The first training attempt dropped all 24,954 rows with missing weather. Result:
- Reduced training rows: 252K -> 235K (-17K rows)
- AUROC DROPPED from 0.864 -> 0.849

Fix: keep all rows, add `gridmet_missing` as a binary feature, let XGBoost's native NaN splitter handle the missing weather values.

### Why `avg_burn_prob` was NOT normalized

README line 183: *"Normalize if combining with national WRC products or the CA training data."*  
Texas-only training = normalization not required. XGBoost is tree-based — scale is irrelevant.

---

## 4. Train / Validation / Test Split

Strictly chronological — never random (per project scope document).

| Split | Years | Total Rows | Fire Rows | Fire Rate |
|---|---|---|---|---|
| **TRAIN** | 2014–2017 | 251,785 | 22,635 | 9.0% |
| **VAL** | 2018 | 61,139 | 5,519 | 9.0% |
| **TEST** | 2019–2020 | 62,855 | 5,595 | 8.9% |
| **TOTAL** | 2014–2020 | 375,779 | 33,749 | 9.0% |

The test set was **never used during training or tuning**. Evaluated exactly once per model version.

---

## 5. Model Versions — What Changed Each Time

### Version 1 — V2 Pipeline Baseline (zeros LANDFIRE)

LANDFIRE rasters were never downloaded. All 4 landscape features were effectively zero.

**Parameters:**
```
max_depth=7, min_child_weight=30, learning_rate=0.05, subsample=0.8
```

| Metric | Value |
|---|---|
| TEST AUROC | 0.8569 |
| TEST AUPR | 0.3978 |
| TEST F1 | 0.408 |
| Trees | 387 |
| Top feature | burnable (26%), centroid_lon (13%), centroid_lat (8%) |

---

### Version 2 — Real LANDFIRE Dataset (train_tx.py)

Used `final_training_dataset_tx_22.07.2026_landfire.xlsx` with real TxWRAP + LANDFIRE LF2022 values.

**New feature values (vs V2 placeholder zeros):**

| Feature | V2 | New dataset | Source |
|---|---|---|---|
| `avg_burn_prob` | ~0 | mean=4.15, 67% non-zero | Texas WRC archive (0–11 scale) |
| `whp` | ~0 | mean=2.67, 67% non-zero | Wildfire Hazard Potential |
| `flep4` | ~0 | mean=0.296, 43% non-zero | Flame Length Exceedance |
| `cfl` | ~0 | mean=3.80, 57% non-zero | Canopy Flame Length |
| `cbd` | absent | mean=0.013, 15% non-zero | NEW — LANDFIRE LF2022 Canopy Bulk Density |
| `cbh` | absent | mean=0.631, 15% non-zero | NEW — LANDFIRE LF2022 Canopy Base Height |

**Parameters:** same as V1 (max_depth=7, lr=0.05, subsample=0.8)

| Metric | Value | vs V1 |
|---|---|---|
| TEST AUROC | 0.8637 | +0.0068 |
| TEST AUPR | 0.4106 | +0.0128 |
| TEST F1 | 0.417 | +0.009 |
| Trees | 330 | |

---

### Version 3 — Hyperparameter Tuned (tune_tx.py)

21 trials across 2 stages. Loaded from pre-saved parquets — no Excel re-read.

#### Stage 1: Tree Structure (12 trials) — Fixed lr=0.05, sub=0.8

| Trial | depth | mcw | Val AUROC | Val AUPR | Rounds |
|---|---|---|---|---|---|
| d6_mcw30 | 6 | 30 | 0.8792 | 0.4276 | 492 |
| d6_mcw20 | 6 | 20 | 0.8793 | 0.4282 | 605 |
| d6_mcw10 | 6 | 10 | 0.8796 | 0.4309 | 608 |
| d7_mcw20 | 7 | 20 | 0.8801 | 0.4314 | 409 |
| d7_mcw10 | 7 | 10 | 0.8804 | 0.4346 | 577 |
| d7_mcw30 | 7 | 30 | 0.8811 | 0.4313 | 330 |
| d8_mcw10 | 8 | 10 | 0.8806 | 0.4308 | 325 |
| d8_mcw20 | 8 | 20 | 0.8822 | 0.4368 | 330 |
| d8_mcw30 | 8 | 30 | 0.8820 | 0.4386 | 502 |
| d9_mcw10 | 9 | 10 | 0.8815 | 0.4340 | 213 |
| d9_mcw20 | 9 | 20 | 0.8826 | 0.4413 | 394 |
| **d9_mcw30** | **9** | **30** | **0.8832** | **0.4415** | 321 |

**Stage 1 winner: depth=9, min_child_weight=30**

#### Stage 2: Learning Rate x Subsample (9 trials) — Fixed depth=9, mcw=30

| Trial | LR | Sub | Val AUROC | Val AUPR | Rounds | Time |
|---|---|---|---|---|---|---|
| lr0.05_s0.7 | 0.05 | 0.7 | 0.8803 | 0.4330 | 320 | 4s |
| lr0.02_s0.7 | 0.02 | 0.7 | 0.8832 | 0.4382 | 709 | 8s |
| lr0.01_s0.7 | 0.01 | 0.7 | 0.8835 | 0.4378 | 1421 | 16s |
| lr0.05_s0.8 | 0.05 | 0.8 | 0.8832 | 0.4415 | 321 | 4s |
| lr0.02_s0.8 | 0.02 | 0.8 | 0.8839 | 0.4404 | 738 | 9s |
| lr0.01_s0.8 | 0.01 | 0.8 | 0.8848 | 0.4434 | 1637 | 18s |
| lr0.05_s0.9 | 0.05 | 0.9 | 0.8844 | 0.4445 | 383 | 5s |
| lr0.02_s0.9 | 0.02 | 0.9 | 0.8849 | 0.4444 | 884 | 11s |
| **lr0.01_s0.9** | **0.01** | **0.9** | **0.8856** | **0.4452** | 1627 | **18s** |

**Stage 2 winner: lr=0.01, subsample=0.9**

**Best overall config:**
```python
max_depth         = 9      # deeper trees capture LANDFIRE x weather interactions
min_child_weight  = 30     # prevents tiny-leaf overfitting on rare fire cells
learning_rate     = 0.01   # slow learner = better generalization
subsample         = 0.9    # 90% of data per tree
colsample_bytree  = 0.8
colsample_bylevel = 0.8
gamma             = 0.1
reg_alpha         = 0.1
reg_lambda        = 1.0
scale_pos_weight  = 10.12  # non-fire:fire ratio
```

**Final training curve (up to 5000 rounds, best at 1627):**
```
[   0]  val-auc: 0.79723
[ 100]  val-auc: 0.86782
[ 500]  val-auc: 0.87717
[1000]  val-auc: 0.88372
[1400]  val-auc: 0.88502
[1600]  val-auc: 0.88563
[1627]  val-auc: 0.88560  <- Best (early stopped at 1706)
```

---

## 6. Final Results — All Three Versions

| Metric | V1 (zeros LANDFIRE) | V2 (real LANDFIRE) | V3 (tuned) | Total gain |
|---|---|---|---|---|
| **TEST AUROC** | 0.8569 | 0.8637 | **0.8687** | **+0.0118** |
| **TEST AUPR** | 0.3978 | 0.4106 | **0.4247** | **+0.0269** |
| **TEST F1** | 0.408 | 0.417 | **0.429** | **+0.021** |
| VAL AUROC | 0.8742 | 0.8810 | **0.8856** | +0.0114 |
| VAL AUPR | 0.4125 | 0.4313 | **0.4452** | +0.0327 |
| Trees | 387 | 330 | **1,627** | — |

---

## 7. Tuned Model — Confusion Matrix (TEST Set)

Threshold = 0.710 (max F1 on validation set)

```
                          PREDICTED
                     No Fire   |   Fire
         ──────────────────────────────────
Actual:  No Fire  |  TN=53,129  |  FP=4,131
         Fire     |  FN= 2,937  |  TP=2,658
```

| Metric | Value | Interpretation |
|---|---|---|
| Recall (TPR) | **47.5%** | 47.5% of real fires detected |
| False Positive Rate | **7.2%** | 7.2% of non-fire cells falsely flagged |
| Precision | **39.2%** | Of all flagged cells, 39.2% are real fires |
| Specificity | **92.8%** | 92.8% of non-fire cells correctly cleared |
| Negative Predictive Value | **94.8%** | Of cells cleared, 94.8% truly had no fire |
| F1 Score | **42.9%** | Harmonic mean of precision and recall |

The model is **4.7x better than random** on AUPR (random baseline = 0.091 = fire rate).

---

## 8. Feature Importance — Tuned Model

| Rank | Feature | Gain | % of Total | Group |
|---|---|---|---|---|
| 1 | `burnable` | 560.7 | 19.8% | Landscape |
| 2 | `avg_burn_prob` | 341.9 | 12.1% | Landscape — TxWRAP |
| 3 | `whp` | 300.0 | 10.6% | Landscape — TxWRAP |
| 4 | `cfl` | 268.2 | 9.5% | Landscape — TxWRAP |
| 5 | `centroid_lon` | 223.4 | 7.9% | Location |
| 6 | `flep4` | 121.6 | 4.3% | Landscape — TxWRAP |
| 7 | `centroid_lat` | 121.6 | 4.3% | Location |
| 8 | `cos_hour` | 65.2 | 2.3% | Temporal |
| 9 | `erc_5D_max` | 52.9 | 1.9% | gridMET weather |
| 10 | `sin_hour` | 52.3 | 1.8% | Temporal |
| 11 | `rmin` | 48.1 | 1.7% | gridMET weather |
| 12 | `sin_month` | 42.2 | 1.5% | Temporal |
| 13 | `vs_5D_max` | 41.4 | 1.5% | gridMET weather |
| 14 | `erc` | 39.9 | 1.4% | gridMET weather |
| 15 | `cos_month` | 39.3 | 1.4% | Temporal |
| 16 | `cbh` | 38.9 | 1.4% | Landscape — LANDFIRE LF2022 (NEW) |
| 17 | `cbd` | 37.9 | 1.3% | Landscape — LANDFIRE LF2022 (NEW) |
| 18 | `pr` | 36.1 | 1.3% | gridMET weather |
| 19 | `vs` | 33.5 | 1.2% | gridMET weather |
| 20 | `erc_5D_mean` | 32.7 | 1.2% | gridMET weather |

### Feature Group Contribution Shift

| Group | V1 gain % | V3 gain % | Change |
|---|---|---|---|
| **Landscape (LANDFIRE + TxWRAP)** | ~26% | **52.8%** | +26.8% up |
| Location (lat/lon) | ~21.9% | 12.2% | -9.7% down |
| gridMET weather | ~28.1% | 18.3% | -9.8% down |
| Temporal | ~11.0% | 9.0% | -2.0% down |

The shift is physically correct. LANDFIRE now dominates because it captures where fire-prone fuel exists. In V1, location and weather were over-compensating for absent LANDFIRE data.

---

## 9. Train vs Test Gap Analysis

```
TRAIN AUROC = 0.9550
VAL   AUROC = 0.8856
TEST  AUROC = 0.8687

Val-to-Test gap = 0.017  <- real temporal generalization gap (acceptable)
Train-to-Val gap = 0.070 <- model memorizes more than it needs (expected)
```

The val-test gap (0.017) is small, confirming that 2018 (VAL) correctly predicted 2019-2020 (TEST) performance.

---

## 10. Output Files

| File | Path | Description |
|---|---|---|
| **Production model** | `outputs/texas_landfire/models/xgb_tx_tuned.ubj` | Tuned model |
| Metadata | `outputs/texas_landfire/models/xgb_tx_tuned_meta.json` | Config + all metrics |
| Baseline model | `outputs/texas_landfire/models/xgb_tx_landfire.ubj` | Pre-tuning model |
| Trial results | `outputs/texas_landfire/tuning_results.csv` | 21 trials ranked |
| Training log | `outputs/texas_landfire/train_tx.log` | Full training log |
| Tuning log | `outputs/texas_landfire/tune_tx.log` | Full tuning log |
| Clean parquets | `data/train_tx_clean.parquet` | Preprocessed TRAIN |
| | `data/val_tx_clean.parquet` | Preprocessed VAL |
| | `data/test_tx_clean.parquet` | Preprocessed TEST |
| Training script | `train_tx.py` | Main pipeline |
| Tuning script | `tune_tx.py` | Hyperparameter search |

---

## 11. Performance Progress

```
AUROC
0.990  Leaked model (fire_count included) — DISCARDED
0.8856 Val score, tuned model
0.8687 TEST score, tuned model      <- CURRENT BEST
0.8637 TEST score, real LANDFIRE
0.8569 TEST score, V2 baseline (zeros LANDFIRE)
0.500  Random guessing

AUPR (primary metric)
0.4247 TEST score, tuned model      <- CURRENT BEST
0.4106 TEST score, real LANDFIRE
0.3978 TEST score, V2 baseline
0.091  Random baseline (= fire rate)
```

---

## 12. Next Steps Roadmap

| Priority | Action | Expected TEST AUROC |
|---|---|---|
| Done | Real LANDFIRE (6 features, 2 new) | 0.864 |
| Done | Hyperparameter tuning (21 trials) | **0.869** |
| Highest ROI | Add HRRR sub-daily features (6-hourly wind, RH, PBL, solar) | ~0.93–0.96 |
| Parallel | California pipeline (phases 2B–3) | — |
| After HRRR | SHAP analysis (publication-quality feature importance) | — |
| Research | Zero-shot CA->TX transferability analysis | — |
| Research | Feature ablation study (AUPR by group) | — |
| Research | Negative sampling ratio sensitivity (1:5, 1:10, 1:20, 1:50) | — |
| Operational | Precision@TopK analysis (K=10, 50, 100, 500) | — |
| Operational | Risk map generation (6-hourly hex map of Texas) | — |

The current feature set is near its ceiling. Reaching 0.93+ AUROC requires HRRR sub-daily atmospheric features — 6-hourly snapshots of actual fire-weather conditions (not just daily gridMET aggregates).

---

*Generated: 2026-07-27 | GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU | Trials: 21 | Best model: xgb_tx_tuned.ubj*
