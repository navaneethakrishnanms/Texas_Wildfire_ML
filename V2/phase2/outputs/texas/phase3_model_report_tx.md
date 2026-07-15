# Texas Wildfire Ignition Model — Phase 3 Baseline Report

*Generated automatically by `run_phase3_visualize.py`*

---

## 1. What is AUROC?

> **AUROC** = Area Under the ROC (Receiver Operating Characteristic) Curve.
> It measures how well the model **ranks** fire cells above non-fire cells.
> - **1.0** = perfect ranking (every fire cell scored higher than every non-fire cell)
> - **0.5** = random guessing (coin flip)
> - **0.85** = our current model correctly ranks 85% of fire/non-fire pairs

The ROC curve plots **True Positive Rate (Recall)** vs **False Positive Rate** at every
possible threshold. AUROC summarises that entire curve as a single number.

---

## 2. What is AUPR?

> **AUPR** = Area Under the Precision-Recall Curve.
> This is the **primary metric** for rare-event prediction like wildfire ignition.

| Metric | What it measures | Why it matters for fires |
|--------|-----------------|--------------------------|
| **AUROC** | Overall ranking quality | Good general indicator |
| **AUPR** | Precision vs Recall tradeoff | Better for 9% fire rate — not fooled by easy negatives |
| Accuracy | % correct predictions | **Useless** — predicting 'no fire' everywhere gives 90.9% |

**Our AUPR = 0.3978** means: across all thresholds, the model achieves average precision
of 39.8% when recalling fire events. Random baseline AUPR = 0.091 (fire rate).
Our model is **4.4× better than random**.

---

## 3. Model Performance Summary

### 3a. Metrics per Split

| Split | Years | Rows | Fire Rows | AUROC | AUPR | F1 | Precision | Recall |
|-------|-------|------|-----------|-------|------|----|-----------|--------|
| TRAIN | — | 252,066 | 22,916 | 0.9302 | 0.5723 | 0.5386 | 0.4153 | 0.7659 |
| VAL | — | 61,181 | 5,561 | 0.8742 | 0.4125 | 0.4316 | 0.3355 | 0.6049 |
| TEST | — | 62,986 | 5,726 | 0.8569 | 0.3978 | 0.4082 | 0.3315 | 0.5313 |

### 3b. Confusion Matrix — TEST Set
*(Threshold = 0.6841, chosen to maximise F1 on validation set)*

```
                      PREDICTED
                   No Fire  |  Fire
         ──────────────────────────────
Actual:  No Fire  |  TN=51,125  |  FP= 6,135  |  (81.2%)  (9.7%)
         Fire     |  FN= 2,684  |  TP= 3,042  |  (4.3%)  (4.8%)
```

| Metric | Value | Interpretation |
|--------|-------|----------------|
| True Positive Rate (Recall/Sensitivity) | **0.531** | 53.1% of real fires are detected |
| False Positive Rate | **0.107** | 10.7% of non-fire cells are falsely flagged |
| Precision | **0.331** | Of all flagged cells, 33.1% are real fires |
| Specificity | **0.893** | 89.3% of non-fire cells correctly identified |
| Negative Predictive Value | **0.950** | Of cells cleared, 95.0% truly had no fire |
| F1 Score | **0.408** | Harmonic mean of precision and recall |

---

## 4. Missing Value Analysis

| Feature Group | Missing Count | % Missing | Treatment | Reason |
|---------------|--------------|-----------|-----------|--------|
| `avg_burn_prob`, `whp`, `flep4`, `cfl` | ~1,594 | 0.42% | Zero-filled | Boundary H3 cells not in LANDFIRE raster extent |
| `avg_burn_prob`, `whp`, `flep4`, `cfl` | 99.6% are **zero** | — | Kept as 0 | **Rasters NOT downloaded** — currently placeholder |
| `erc`, `fm100`, `vpd`, `vs`, `rmax`, `rmin`, `tmmx`, `pr` | 24,954 | 6.63% | Left as NaN | Coastal/border H3 cells outside gridMET grid — XGBoost handles natively |
| `erc_5D_*` etc. | 24,965 | 6.64% | Left as NaN | Same border cells — 5-day rolling window also missing |
| `burnable`, `has_fire_history`, `fire_count` | ~1,594 | 0.42% | Zero-filled | Same border cells |

> **Note:** `burnable` was a placeholder `True` for all cells from Phase 2B. It has
> 0.42% NaN (border cells filled to 0). In the current model it acts as a 'valid cell'
> flag and is the top feature by gain — this will be replaced with real LANDFIRE
> vegetation data after raster download.

---

## 5. ⚠️ Leakage Issue: Why AUROC Was 0.990 — and How It Was Fixed

### 5a. What Happened

The **first training run** included `fire_count` and `has_fire_history` as features.
These columns were built in Phase 2B from the **full fire dataset (2014–2020)**,
which includes the **test years (2019–2020)**.

### 5b. Why This Is Leakage

```
  fire_count    = how many FPA-FOD fires occurred in that H3 cell (2014–2020)
  has_fire_history = True if fire_count > 0

  Label=1 row (fire cell)     → fire_count ≥ 1  (by definition)
  Label=0 row (non-fire cell) → fire_count = 0  (almost always)

  ∴ Model learns: fire_count > 0 → predict fire
  No need to learn weather, landscape, or time-of-day patterns.
```

This is exactly what the official scope document warns against (line 391):
> *'ignition_density, burn_count — Historical fire count — trivially separates
> fire/non-fire by construction'*

### 5c. Evidence of Leakage

| Feature | Gain (leaked model) | Share |
|---------|--------------------:|-------|
| `has_fire_history` | 12,078 | 58% |
| `fire_count` | 7,390 | 36% |
| `burnable` | 2,582 | 12% |
| `erc_5D_max` *(real fire signal)* | 267 | 1% |
| All other features | < 250 total | <1% |

AUROC = **0.9900** (artificially inflated — model is 'cheating')

### 5d. Fix Applied

Removed `fire_count` and `has_fire_history` from `FEATURE_COLS` in
`run_phase3_train.py`. Retrained from scratch. No data changes needed.

### 5e. Results After Fix

| Metric | Leaked (wrong) | Clean (correct) |
|--------|---------------|-----------------|
| AUROC | 0.9900 ❌ | **0.8569** ✅ |
| AUPR | 0.8642 ❌ | **0.3978** ✅ |
| Feature importance | fire_count dominant | erc, vs, lat/lon — physically correct |
| Trees used | 143 | 387 (more complexity needed for real signal) |

---

## 6. Top Feature Importance (Clean Model)

| Rank | Feature | Gain | % of Total | Group |
|------|---------|------|-----------|-------|
| 1 | `burnable` | 654.7 | 26.0% | Landscape |
| 2 | `centroid_lon` | 336.6 | 13.4% | Location |
| 3 | `centroid_lat` | 214.2 | 8.5% | Location |
| 4 | `erc_5D_max` | 170.8 | 6.8% | gridMET weather |
| 5 | `vs_5D_max` | 161.5 | 6.4% | gridMET weather |
| 6 | `cos_hour` | 103.3 | 4.1% | Temporal |
| 7 | `rmin` | 64.7 | 2.6% | gridMET weather |
| 8 | `sin_hour` | 60.5 | 2.4% | Temporal |
| 9 | `erc` | 59.8 | 2.4% | gridMET weather |
| 10 | `sin_month` | 57.8 | 2.3% | Temporal |
| 11 | `cos_month` | 52.7 | 2.1% | Temporal |
| 12 | `vs` | 46.3 | 1.8% | gridMET weather |
| 13 | `vs_5D_mean` | 45.5 | 1.8% | gridMET weather |
| 14 | `erc_5D_mean` | 44.4 | 1.8% | gridMET weather |
| 15 | `fm100` | 41.4 | 1.6% | gridMET weather |

---

## 7. What Happens Next (Expected Improvement)

| Step | Action | Expected AUROC |
|------|--------|----------------|
| Current | gridMET + temporal + location (no LANDFIRE) | **0.8569** |
| Step 1 | Download LANDFIRE rasters → re-run Phase 2E + 2G + 3 | **~0.90–0.93** |
| Step 2 | Add HRRR per-window features (6-hourly atmospheric) | **~0.93–0.96** |

### LANDFIRE rasters to download:
- `avg_burn_prob` (Burn Probability): `firelab.org/fsim`
- `whp` (Wildfire Hazard Potential): `doi.org/10.2737/RDS-2015-0047-4`
- `flep4` (Flame Length Exceedance): `landfire.gov`
- `cfl` (Canopy Fuel Load): `landfire.gov`

---

## 8. Figures Generated

- `confusion_matrix_tx.png`
- `feat_importance_tx.png`
- `phase3_evaluation_tx.png`
- `pr_curve_tx.png`
- `roc_curve_tx.png`
- `score_dist_tx.png`
