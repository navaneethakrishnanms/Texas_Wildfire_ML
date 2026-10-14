# TDIS_Ignition_BIT_V3 — HRRR-Only Retrain + True Real-Population Validation

**Date:** 2026-10-14
**Environment:** conda env `torch_gpu` (GPU/CUDA, xgboost 3.2.0, Herbie 2025.12.0 + ecCodes)
**Location:** `TDIS_Ignition_BIT_V3/`
**Trigger:** Internal feedback (Haripriya, 2026-10-14) — two specific corrections, both addressed here:
1. *"Use HRRR data only... Gridmet is the same day data and we are focused on a forecasting model. HRRR is forecast data so please just have one weather source."*
2. *"The metrics presented are on a sample dataset not the whole population. I would like to see metrics on the population as well."*

---

## 1. One-sentence summary

Dropping all gridMET-sourced features (9 columns) cost almost nothing on the matched test set (AUROC 0.8064→0.8046, AUC-PR 0.7237→0.7229 — a ~0.002 drop), confirming gridMET was adding negligible value once HRRR+static+location features are present; and the first true real-population validation of any model in this project line (32.5M real cell-days, pulled from live archived HRRR, scored against real recorded ignitions) gives **AUROC 0.69, Lift 3.87–4.91×** — below TDIS's own real-population numbers (AUROC 0.80, Lift 4.58×), the honest gap this report exists to surface.

---

## 2. Part A — HRRR-only retrain (29 features)

### 2.1 What was removed and why

| Dropped | Why |
|---|---|
| `erc, fm100, vpd, vs, rmax, rmin, tmmx, pr` | Legacy gridMET daily weather — **same-day observed**, not a forecast. Not knowable at real forecast-issuance time. |
| `fm100_5d_min` | Also gridMET-sourced (100-hr dead fuel moisture, 5-day trailing min) — same issue, dropped for consistency. |

**Kept:** `bp_x_vpd`, `whp_x_vpd` — verified by direct correlation check (r = 0.9999999) to already be built from `hrrr_vpd`, not gridMET `vpd`, so they're safe under an HRRR-only rule. `lat`/`lon` were also kept — that's a separate finding from our own BIT v1/v2 ablation, unrelated to the weather-source question.

### 2.2 Final feature set — 29 features
`road_dist_km, ecoregion_id, elevation_m, slope_deg, aspect_deg, avg_burn_prob, whp, flep4, cfl, cbd, cbh, powerline_dist_km` (static, 12) + `sin_month, cos_month, sin_dow, cos_dow, is_weekend, is_holiday` (calendar, 6) + `hrrr_tmp, hrrr_vpd, hrrr_wind, hrrr_mstav` (HRRR, 4) + `drought_score` (1) + `bp_x_vpd, whp_x_vpd, bp_x_drought` (interactions, 3) + `pop_density` (1) + `lat, lon` (2, our addition).

### 2.3 Hyperparameter search (3 stages, 34 trials — same regularization-aware method as BIT V2)

**Stage 1 winner:** `max_depth=9, min_child_weight=50` (val AUC-PR 0.7583, chosen for lower train/val gap over the single-best 0.7591 at mcw=20)
**Stage 2 winner:** `learning_rate=0.05, subsample=0.8` (val AUC-PR 0.7621)
**Stage 3 winner:** `gamma=0.1, colsample_bytree=0.9` (val AUC-PR 0.7627)

**Final hyperparameters:**
```
max_depth: 9, min_child_weight: 50, learning_rate: 0.05, subsample: 0.8,
colsample_bytree: 0.9, gamma: 0.1, scale_pos_weight: 2.2872,
eval_metric: aucpr, early_stopping_rounds: 80, n_estimators cap: 3000
best_iteration: 837, device: cuda, random_state: 42
```
Total GPU runtime (34 trials + final model): **263.4 seconds**.

### 2.4 Results — matched test set (n = 1,261,502, 30.6% positive rate, same split as BIT v1/v2)

| Split | AUC-PR | AUROC | F1 | Precision | Recall | Lift |
|---|---|---|---|---|---|---|
| Train | 0.8890 | 0.9376 | — | — | — | — |
| Val | 0.7627 | 0.8508 | — | — | — | — |
| **Test (raw)** | **0.7229** | **0.8046** | **0.6272** | 0.6256 | 0.6289 | **2.37×** |
| Test (calibrated) | 0.7186 | 0.8045 | — | — | — | — |

**Train→Test AUROC gap: 0.1330** (vs. BIT v1's 0.1430, BIT V2's 0.1280 — in between, not a clear win or loss on this axis).

### 2.5 Confusion matrices — EXACT, test set

**Raw, best-F1 threshold (0.4657):**
|  | Predicted No Fire | Predicted Fire |
|---|---|---|
| **Actual No Fire** | TN = 731,073 | FP = 145,037 |
| **Actual Fire** | FN = 143,027 | TP = 242,365 |

Precision 0.6256 · Recall 0.6289 · Accuracy 0.7716

### 2.6 Top 10 features by gain

| Rank | Feature | % gain |
|---|---|---|
| 1 | `ecoregion_id` | 21.73% |
| 2 | `elevation_m` | 8.67% |
| 3 | `hrrr_vpd` | 7.35% |
| 4 | `avg_burn_prob` | 6.49% |
| 5 | `lat` | 5.16% |
| 6 | `lon` | 4.04% |
| 7 | `road_dist_km` | 3.92% |
| 8 | `sin_month` | 3.78% |
| 9 | `powerline_dist_km` | 3.78% |
| 10 | `cos_month` | 2.50% |

`flep4`/`cfl` dead again (0.00%) — now confirmed in **every single model** built on either side of this project.

### 2.7 Comparison: BIT v1 (38 feat) vs. BIT V3 (29 feat, HRRR-only)

| Metric | BIT v1 (38, incl. gridMET) | **BIT V3 (29, HRRR-only)** | Change |
|---|---|---|---|
| Test AUROC | 0.8064 | 0.8046 | **−0.0018** |
| Test AUC-PR | 0.7237 | 0.7229 | **−0.0008** |
| Test F1 | 0.6284 | 0.6272 | −0.0012 |
| Train→Test gap | 0.1430 | 0.1330 | improved |

**This is the headline finding of Part A: removing 9 gridMET-sourced features cost almost nothing.** The near-zero accuracy drop confirms gridMET was carrying very little information the HRRR+static+location features didn't already have — meaning the "cleaner, more realistic" model Haripriya asked for isn't a meaningful accuracy sacrifice, just a bigger forecast-integrity win than expected. This is the model that should now be treated as our reference going forward.

### 2.8 Visuals — `results/figures/` (9 plots)
ROC curve, PR curve, confusion matrices (raw + calibrated), precision/recall/F1 vs. threshold, feature importance, calibration reliability diagram, prediction distribution, calibrator mapping curve — same suite as BIT V2, generated via `scripts/generate_visuals.py`.

---

## 3. Part B — True real-population validation

### 3.1 Why this required pulling real data (not an approximation)

Every metric in every report before this one — ours and TDIS's matched-test numbers — was computed on a **rebalanced sample** (8 temporal + 0.4× spatial negatives per positive, ~30% positive rate). TDIS's own real-population numbers (AUROC 0.7968, lift 4.58×) were the one benchmark we hadn't replicated. Doing this properly requires scoring the model against **every actual TX grid cell on real calendar days**, using **real weather** for those days — not a sampled subset.

**Herbie (the HRRR-pull library) was already installed in `torch_gpu`** — it had failed earlier in this session only because of a Windows DLL PATH issue (the env's `Library/bin`, containing `eccodes.dll`, wasn't on PATH when calling `python.exe` directly without activating the environment). Fixed by prepending that directory to PATH. No reinstall was needed.

### 3.2 Method

1. **19 representative real dates**, one per season per year, 2022–2026 (2026 limited to Jan/Apr/Jul since the real label data ends 2026-07-29).
2. For each date D, pulled the **archived HRRR forecast** issued at D−1 12Z, lead 24h (valid at D 12Z) — the same forecast-realistic design used throughout training — via Herbie from the public NOAA AWS S3 archive (`s3://noaa-hrrr-bdp-pds`), fields `TMP:2m`, `DPT:2m`, `UGRD:10m`, `VGRD:10m`, `MSTAV:0m`.
3. Built a nearest-neighbor map (cKDTree) from the HRRR grid (1,059×1,799 points) to **all 1,708,940 TX H3 res-8 cells** — computed once, reused for every date.
4. Assembled the full 29-feature row for **every cell** (minus 511 flare cells, same exclusion rule as training): static features from `tx_static_master.parquet`, this day's real HRRR weather, nearest-week real USDM drought score, calendar features computed from the date, interaction terms.
5. Scored with the trained BIT V3 model + its calibrator.
6. **Real labels**, not sampled: label = 1 if that (cell, date) has an actual recorded ignition in TDIS's `ignitions_daily_tx.parquet`, else 0 — the true population, not a matched subset.

### 3.3 Results — res-8, single-day granularity (32,460,151 real cell-days, true base rate 0.008%)

| | RAW | Calibrated |
|---|---|---|
| AUROC | 0.6968 | 0.6957 |
| AUC-PR | 0.0004 | 0.0004 |
| Lift | **4.91×** | 4.77× |
| Brier | 0.00749 | 0.00388 |

**Confusion matrix (raw, best-F1 threshold 0.7967):**
|  | Predicted No Fire | Predicted Fire |
|---|---|---|
| **Actual No Fire** | TN = 32,443,616 | FP = 13,851 |
| **Actual Fire** | FN = 2,649 | TP = 35 |

**Threshold sweep (calibrated probability) — the operationally meaningful view at this extreme imbalance:**

| Threshold | Precision | Recall | F1 | Cells flagged |
|---|---|---|---|---|
| 0.01 | 0.0001 | 0.899 | 0.0002 | 24.4M |
| 0.05 | 0.0002 | 0.450 | 0.0004 | 5.9M |
| 0.10 | 0.0004 | 0.256 | 0.0007 | 1.9M |
| 0.20 | 0.0008 | 0.139 | 0.0016 | 475,149 |
| 0.50 | 0.0018 | 0.050 | 0.0036 | 72,456 |

**Why AUC-PR looks tiny here and that's expected, not a bug:** at res-8 single-day resolution, the true base rate is **0.008%** (1 fire in ~12,500 cells) — far more extreme than TDIS's own reported real-population base rate of 1.92%, because that 1.92% figure is computed at the coarser **res-5** aggregation (~5,382 cells), not res-8 (1,708,940 cells). AUC-PR is mechanically bounded by the base rate, so comparing our 0.0004 directly to TDIS's 0.0878 is not fair — they're different resolutions. Fixed in §3.4.

### 3.4 Results — res-5 aggregated (TDIS's own convention, for a true apples-to-apples comparison)

Aggregated the res-8 panel up to res-5 the same way TDIS does (mean predicted probability per res-5 cell/date, label = 1 if any underlying res-8 cell had a real fire): **102,258 res-5 cell-days, base rate 1.61%** — now directly comparable to TDIS's 1.92%.

| Metric | **This model (res-5 agg.)** | **TDIS served model (res-5, real-pop)** |
|---|---|---|
| Base rate | 1.61% | 1.92% |
| AUROC | **0.6903** | **0.7968** |
| AUC-PR | **0.0624** | **0.0878** |
| Lift | **3.87×** | **4.58×** |
| Best F1 | 0.1319 | 0.1692 |
| Precision @ best F1 | 0.1355 | 0.1493 |
| Recall @ best F1 | 0.1285 | 0.1851 |

**Confusion matrix (res-5, raw, best-F1 threshold 0.278):**
|  | Predicted No Fire | Predicted Fire |
|---|---|---|
| **Actual No Fire** | TN = 99,256 | FP = 1,352 |
| **Actual Fire** | FN = 1,438 | TP = 212 |

### 3.5 The honest read

**This is the first real-population number we've ever produced, and it's below TDIS's.** On every metric at the comparable res-5 resolution, our model trails their served model: AUROC −0.11, AUC-PR −0.025, Lift −0.7×. This is a genuinely useful, humbling result — it answers Haripriya's second point directly, and the answer is "our model doesn't yet hold up as well on the real population as it does on the matched sample," which is exactly the kind of gap matched-sample metrics hide.

**Caveats on this specific measurement, stated plainly:**
- **Only 19 days, not a full year or multi-year continuous panel.** TDIS's real-population number comes from 941 continuous days (2024–2026); ours comes from 19 sparsely-sampled calendar days across 5 years. Day-to-day variance in a rare-event problem like this means 19 days is a meaningfully smaller, noisier sample.
- **Single fixed calendar day per quarter** (the 15th of Jan/Apr/Jul/Oct) — not a random sample of days, so any systematic pattern tied to day-of-month is a (small, likely negligible) confound.
- The gap could reflect either a real model-quality difference, or simply that our smaller sample landed on a harder subset of days. **More dates would tighten this estimate** — this is a natural next step, not a final verdict.

---

## 4. What this means, put together

1. **The HRRR-only retrain (Part A) is good news:** fixing Haripriya's weather-source concern cost almost nothing in accuracy. There's no longer a tradeoff to debate — use the clean, forecast-honest 29-feature model.
2. **The real-population validation (Part B) is a genuine, overdue check**, and it shows our model is **not yet as strong as TDIS's served model** once measured the way that actually matters for deployment. This had been flagged as an open gap in every prior report in this project; now it has a real number attached, and that number says there's more work to do, not that the project is ready to ship.
3. **Recommended next step:** expand the real-population validation to more dates (ideally a full continuous year, matching TDIS's methodology exactly) before drawing a final conclusion — 19 sparse dates is a first measurement, not the last word.

---

## 5. Files produced

```
TDIS_Ignition_BIT_V3/
├── data/processed/bit_v3_train_tx.parquet, .csv   (29-feature dataset, 2,201,289 rows)
├── models/model_bit_v3_29feat.json, calibrator_bit_v3_29feat.joblib, feature_list.json
├── results/training_results_bit_v3_29feat.json, tuning_stage{1,2,3}.csv, figures/ (9 PNGs)
├── realpop/
│   ├── data/scored_days/{19 dates}.parquet        (per-date full-grid scored panels)
│   ├── results/realpop_panel_all_days.parquet     (32.5M-row concatenated panel)
│   ├── results/realpop_results.json               (res-8 metrics)
│   ├── results/realpop_res5_aggregated.json        (res-5 metrics, comparable to TDIS)
│   └── results/figures/ (6 PNGs: ROC, PR, 2x confusion matrix, reliability, prediction dist.)
└── BIT_V3_TRAINING_REPORT.md                       (this file)
```

### Reproduce
```bash
cd "TDIS_Ignition_BIT_V3"
"/c/Users/Admin/anaconda3/envs/torch_gpu/python.exe" scripts/build_dataset.py
"/c/Users/Admin/anaconda3/envs/torch_gpu/python.exe" scripts/tune_train_model.py
"/c/Users/Admin/anaconda3/envs/torch_gpu/python.exe" scripts/generate_visuals.py

# Real-population validation (needs eccodes DLL on PATH for Herbie/GRIB decoding)
PATH="/c/Users/Admin/anaconda3/envs/torch_gpu/Library/bin:$PATH" PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "/c/Users/Admin/anaconda3/envs/torch_gpu/python.exe" realpop/scripts/01_pull_score_days.py
"/c/Users/Admin/anaconda3/envs/torch_gpu/python.exe" realpop/scripts/02_evaluate_realpop.py
```
