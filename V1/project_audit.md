# 🔥 Texas Wildfire ML — Full Project Audit

---

## 1. Why Did Training Finish So Fast? (~6 minutes total)

Three reasons working together:

### A. Early Stopping Kicked In Quickly
| Model | Max Trees Set | Stopped At | Reason |
|-------|--------------|------------|--------|
| XGBoost A (full) | 600 | **195** | No improvement for 50 rounds |
| XGBoost B (ablation) | 600 | **456** | No improvement for 50 rounds |
| Random Forest | 500 | **500** (full) | No early stopping in RF |
| LightGBM D | 600 | **206** | No improvement for 50 rounds |

> Early stopping means the model stops growing trees the moment the validation score stops improving — so you never wastefully train all 600 trees. XGBoost A only needed **195 trees**, not 600.

### B. GPU Acceleration (XGBoost only)
- Your RTX 3050 was detected via `nvidia-smi` and XGBoost set `device="cuda"` automatically
- GPU histogram building is 3–5× faster than CPU for your dataset size

### C. The Dataset Is Actually Not That Large
From `data_prep.py`:
- Train: ~Jan–Aug rows from FIRMS fire detections + 3× negatives
- Val: Sep only
- Test: Oct–Dec

The actual row count is thousands to tens of thousands (real FIRMS events), **not** 900K. The 900K simulation in the README was for the geospatial simulator — your actual pipeline used real FIRMS CSV data + sampled negatives. At 10K–50K rows × 18 features with early stopping + GPU, 6 minutes is completely normal and correct.

---

## 2. Is the Training Approach Correct?

### ✅ What Is Done Right
- **Chronological split** (no data leakage): Train=Jan–Aug, Val=Sep, Test=Oct–Dec
- **TimeSeriesSplit CV** (3-fold diagnostic) — temporal ordering preserved
- **scale_pos_weight=3.01** — correctly handles 1:3 class imbalance
- **Threshold tuning on val set** — finds optimal threshold (0.46) before touching test
- **SHAP explainability** — confirms model learned real fire physics (Temperature, Wind, Rainfall)
- **Ablation study** (Model B vs A) — tests contribution of `is_peak_fire_season`

### ⚠️ Critical Issue: The 24-Hour Prediction Problem

> [!CAUTION]
> **The labeling is WRONG for true 24-hour forecasting.**

**Current approach (V1 — what's built):**
```
Row: lat=30.1, lon=-97.5, acq_date=2024-03-15
Features: annual composite NDVI, EVI, LST + date encodings
Label: Fire=1 (because FIRMS detected fire at this location on Mar 15)
```

**Problem:** The model is trained with features from the **same day** as the fire. In production you want to predict tomorrow's fire using today's features. With annual composite rasters, every date at the same location gets **identical raster values** — the only signal is temporal (month, day_of_year). This means:

- Model A can't tell if conditions right now are dangerous
- It essentially learns: "what month/location combination historically has fires"
- This is **fire occurrence correlation**, not **24-hour ignition forecasting**

**What 24-hour forecasting actually requires:**

```
Row: lat=30.1, lon=-97.5, date=2024-03-14  ← YESTERDAY'S features
Features: NDVI on Mar 14, Temp on Mar 14, Wind on Mar 14, RH on Mar 14, etc.
Label: Fire=1 if FIRMS detected fire at this location on 2024-03-15  ← TOMORROW
```

The dataset builder comment in `build_dataset.py` (lines 10-39) already acknowledges this — it calls V1 "annual composites" and describes V2 as "date-specific rasters":

```
VERSION 2 (future upgrade):
For each FIRMS event (lat, lon, acq_date):
  → Query GEE for NDVI/EVI/LST/Temperature/Wind/Rainfall
     within a ±N day window around acq_date
```

**So the architecture is correct for V1 (proof of concept), but NOT a true 24-hour predictor yet.**

---

## 3. Why Did LightGBM Beat XGBoost?

This is a great question and here's the honest analysis:

| Factor | XGBoost A | LightGBM D |
|--------|-----------|------------|
| AUC-ROC | 0.8868 | **0.8994** |
| AUC-PR | 0.6691 | **0.6905** |
| Recall @0.5 | 0.7527 | **0.8269** |
| Stopped at | 195 trees | 206 trees |

### Reasons LightGBM Won Here:

**1. LandCover treated as true categorical**
LightGBM registered `LandCover` as a `category` dtype and uses its native categorical split algorithm (optimal split over all category values at once). XGBoost treats it as a regular integer — less efficient.

**2. Leaf-wise vs level-wise tree growth**
LightGBM grows trees **leaf-wise** (deepest most informative leaf first). XGBoost grows **level-wise** (all leaves at same depth). For tabular data with heterogeneous features like this (mix of vegetation index, weather, terrain), leaf-wise often wins.

**3. `is_unbalance=True` vs `scale_pos_weight`**
Both handle class imbalance but differently. `is_unbalance` in LightGBM reweights during gradient computation — can be more numerically stable for certain datasets.

### Is This "Wrong Training"? 

**No — but with caveats:**
- LightGBM winning by ~1% AUC is within normal variance for simulated/small datasets
- On real large-scale wildfire data with proper daily rasters, XGBoost or LightGBM would likely be comparable
- The margin is too small to conclude LightGBM is fundamentally better here
- XGBoost stopping at only 195 iterations (vs LightGBM's 206) suggests XGBoost may have stopped too early — try lowering `min_child_weight` or `gamma` for more flexibility

---

## 4. Full Report Card — Project Health

### Final Test Metrics (Model E = LightGBM, threshold=0.46)

| Metric | Validation | Test | Assessment |
|--------|-----------|------|------------|
| AUC-ROC | 0.8994 | **0.9142** | 🟢 Excellent |
| AUC-PR | 0.6905 | **0.7549** | 🟡 Good (flagged as overfit) |
| F1 | 0.6915 | **0.7236** | 🟢 Good |
| Recall | 0.8516 | **0.8744** | 🟢 87% fires caught |
| Precision | 0.5821 | **0.6173** | 🟡 1 in 3 alarms is false |

### The "OVERFIT" Flag in the Report — Is It Real?

> [!WARNING]
> The pipeline flagged `AUC-PR: +0.0644` as overfitting. But this is actually **inverse overfitting** — test performance is *better* than validation. This means:

1. **The test period (Oct–Dec) happens to align with peak fire season** (Oct–Nov are peak months in Texas). More fires = better signal = higher PR-AUC on test.
2. This is **temporal distribution shift**, not model overfitting.
3. A truly overfit model would score *lower* on test, not higher.

### Confusion Matrix (Test Set)
```
                Predicted No Fire   Predicted Fire
Actual No Fire       1354 (TN)         315 (FP)
Actual Fire            73 (FN)         508 (TP)
```
- **508/581 fires caught** (87.4% recall) ✅
- **315 false alarms** — roughly 1 false alarm per 1.7 real fires 🟡
- **73 missed fires** — these are the dangerous misses ⚠️

### SHAP Feature Importance — Is the Model Learning Real Physics?

```
1. Temperature    0.10583  ✅ Hot = fire risk
2. Wind           0.08204  ✅ Wind drives fire spread
3. Rainfall       0.06085  ✅ Wet = low risk
4. DEM (Elevation)0.04556  ✅ Terrain affects fire behavior  
5. LST (Surface T)0.01613  ✅ Surface heat = ignition risk
6. EVI            0.01360  ✅ Vegetation fuel load
7. NDVI           0.01303  ✅ Vegetation moisture
```

**Verdict: The model IS learning real fire science, not memorizing noise.** ✅

---

## 5. Overall Project Status

### What Works ✅
- Full pipeline from raw FIRMS data → trained model → SHAP explanation
- Chronological train/val/test split (no leakage)
- AUC-ROC of 0.91 — genuinely good for fire prediction
- 87% recall — catching the vast majority of fires
- SHAP confirms physically meaningful features

### What Is Missing / Needs Fixing ⚠️

| Issue | Severity | Fix |
|-------|----------|-----|
| Annual composites (not daily features) | 🔴 HIGH | V2: Use GEE date-specific raster queries |
| No true 24-hr label shift | 🔴 HIGH | Label = fire on day T+1, features = day T |
| Simulated coordinates not validated against Texas polygon | 🟡 MEDIUM | Add shapely polygon boundary check |
| `AUC-PR gap` misidentified as overfit | 🟡 MEDIUM | Fix the gap threshold logic |
| LightGBM no GPU support | 🟢 LOW | Install LightGBM GPU build if needed |

### Next Phases to Build

```
Current: V1 POC — annual composites, same-day labels
     ↓
Phase Next: V2 — Daily raster features (GEE API per date)
                  + True 24-hr label offset (predict T+1 from T)
     ↓
Phase After: REST API (FastAPI) serving daily predictions
     ↓
Phase After: Dashboard (Leaflet/Deck.gl) — interactive risk map
     ↓
Phase Final: Operational deployment with live NOAA/NASA feeds
```
