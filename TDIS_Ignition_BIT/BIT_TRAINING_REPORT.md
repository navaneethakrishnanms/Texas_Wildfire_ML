# TDIS_Ignition_BIT — Retraining TDIS's Newest Dataset With Our Own Methodology

**Date:** 2026-10-01
**Environment:** conda env `torch_gpu` (Python, xgboost 3.2.0, scikit-learn 1.7.2, h3 4.5.0), NVIDIA CUDA GPU
**Location:** `TDIS_Ignition_BIT/`
**Goal:** Take the newest data TDIS/Texas team handed us (`TDIS_Ignition/`) and retrain it end-to-end using our own methodology (real hyperparameter search, our proven feature additions, isotonic calibration) to see how far our pipeline can push their data.

---

## 1. One-sentence summary

We rebuilt a clean, fully-imputed, 38-feature training table from TDIS's own most-corrected data file (`train_tristate_v2.parquet` — the tri-state table where they'd already fixed the ecoregion-scale bug), ran a genuine 21-trial two-stage hyperparameter search on GPU, and got **Test AUROC 0.8064 / AUC-PR 0.7237 / F1 0.6284** on their own embedded 2022–2026 test split — an improvement over their *matched-test* served-model numbers (AUROC 0.7402 / AUC-PR 0.4941) on every metric, though **not a like-for-like comparison** (different positive rate, different years, and we have not yet run real-population validation — see §7, the honest caveat section).

---

## 2. Where the data came from, exactly

**Source file:** `TDIS_Ignition/model_tristate/data/train_tristate_v2.parquet` — chosen deliberately because it is the most-corrected table TDIS shipped: it's the one where their own team found and fixed the TX/NM/OK ecoregion-code incommensurability bug (local 0–19 TX scale vs. national EPA scale), and it already contains their full feature set including the interaction terms and tri-state additions (`pop_density`, `fm100_5d_min`, `bp_x_vpd`, `whp_x_vpd`, `bp_x_drought`).

We filtered to **TX-only rows** (`state == 'tx'`): **2,202,331 rows**, keeping TDIS's own embedded temporal split (`split` column) rather than inventing our own:

| Split | Rows | Date range | Positive rate |
|---|---|---|---|
| Train | 671,455 | 2018-07-16 → 2020-12-31 | 30.42% |
| Val | 268,332 | 2021-01-01 → 2021-12-31 | 28.76% |
| Test | 1,261,502 | 2022-01-01 → 2026-07-29 | 30.55% |

### Fixes/additions we applied on top of their data (`build_dataset.py`)

1. **Derived real `lat`/`lon` from the H3 cell index.** TDIS's own `lat`/`lon` columns in this file are **100% null for every TX row** (confirmed by direct inspection — a defect in their tri-state table, not something documented anywhere). We computed them directly and deterministically via `h3.cell_to_latlng()` — 100% coverage, zero nulls. These proved valuable in our own TEXAS 32-feature model (ranked #3/#5 by importance there) and rank **#5/#6** here too (§5).
2. **Seasonal (month-of-year) median imputation for the 8 legacy gridMET columns** (`erc, fm100, vpd, vs, rmax, rmin, tmmx, pr`) — these were **76.6% null** in the raw file (a known, documented TDIS limitation: the gridMET columns are carried along from an old schema and were never fully populated for the full cell universe). Rather than drop them — `rmin` was our TEXAS model's **#4** feature by importance — we imputed them the same way our TEXAS pipeline did (seasonal median), keeping them in the feature set.
3. **Mode-imputed `ecoregion_id`** (11.1% null) and **median-imputed `fm100_5d_min`** (10.0% null) — both minor residual gaps in TDIS's own corrected file.
4. **Truncated 1,042 phantom future-dated rows** (2026-07-30/31, all `label=0`, past TDIS's own last real observed label of 2026-07-29) — a smaller recurrence of the exact "fabricated future-dated rows" defect our own `DATA_QUALITY_REVIEW.md` flagged as **Critical** in the original handoff (there it was 86,072 rows; here it's 1,042 — three orders of magnitude smaller, but the same root cause, and we applied our own previously-recommended fix: truncate at the last observed label rather than ship it silently).

**Result:** `TDIS_Ignition_BIT/data/bit_train_tx.parquet` — 2,201,289 rows × 44 columns, **zero nulls**.

---

## 3. Full feature list used (38 features)

| Group | Features | Source |
|---|---|---|
| Static — terrain/access/fuel hazard (12) | `road_dist_km`, `ecoregion_id`, `elevation_m`, `slope_deg`, `aspect_deg`, `avg_burn_prob`, `whp`, `flep4`, `cfl`, `cbd`, `cbh`, `powerline_dist_km` | TDIS's own served (23-feat) set |
| Calendar (6) | `sin_month`, `cos_month`, `sin_dow`, `cos_dow`, `is_weekend`, `is_holiday` | TDIS's own served set |
| HRRR forecast weather + drought (5) | `hrrr_tmp`, `hrrr_vpd`, `hrrr_wind`, `hrrr_mstav`, `drought_score` | TDIS's own served set (this is the full 23-feature served set, 12+6+5) |
| Tri-state additions (5) | `bp_x_vpd`, `whp_x_vpd`, `bp_x_drought` (interactions), `pop_density`, `fm100_5d_min` | TDIS's rev5/28-feature tri-state set |
| **Our addition — location (2)** | `lat`, `lon` | Derived from `h3_cell` (TDIS's own copies were 100% null) |
| **Our addition — daily weather (8)** | `erc`, `fm100`, `vpd`, `vs`, `rmax`, `rmin`, `tmmx`, `pr` | Legacy gridMET columns, seasonally imputed |

**38 total** — TDIS's full 28-feature tri-state recipe, plus our 10-feature addition (2 location + 8 gridMET) that our own prior models found to carry real signal.

---

## 4. Methodology — the two-stage hyperparameter search

Same recipe that took our earlier TX model from 0.8637 → 0.8687 AUROC and our TEXAS model from 0.8249 → 0.8309 AUROC: never touch test during search, score every trial on the validation fold only.

### Stage 1 — tree structure (12 trials: `max_depth` × `min_child_weight`)
Fixed: `learning_rate=0.05, subsample=0.8, n_estimators=400` (early stop 30 rounds).

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
| **9** | **10** | 0.8516 | **0.7604** |
| 9 | 20 | 0.8511 | 0.7586 |
| 9 | 30 | **0.8523** | 0.7591 |

**Winner (by val AUC-PR, our selection metric, consistent with TDIS's own convention): `max_depth=9, min_child_weight=10`.**

### Stage 2 — learning rate × subsample (9 trials)
Fixed: `max_depth=9, min_child_weight=10, n_estimators=1200` (early stop 50 rounds).

| lr | subsample | val AUROC | val AUC-PR |
|---|---|---|---|
| 0.01 | 0.7 | 0.8508 | 0.7541 |
| 0.01 | 0.8 | 0.8509 | 0.7543 |
| 0.01 | 0.9 | 0.8508 | 0.7543 |
| 0.02 | 0.7 | 0.8527 | 0.7618 |
| 0.02 | 0.8 | 0.8526 | 0.7620 |
| 0.02 | 0.9 | 0.8529 | 0.7624 |
| 0.05 | 0.7 | 0.8518 | 0.7612 |
| **0.05** | **0.8** | 0.8516 | **0.7632** |
| 0.05 | 0.9 | 0.8512 | 0.7621 |

**Winner: `learning_rate=0.05, subsample=0.8`.**

### Final configuration
```
max_depth=9, min_child_weight=10, learning_rate=0.05, subsample=0.8,
colsample_bytree=0.8, scale_pos_weight=2.2872 (n_neg/n_pos on train),
eval_metric=aucpr, early_stopping_rounds=80, n_estimators cap=3000,
tree_method=hist, device=cuda, random_state=42
```
**Best iteration: 695** (early-stopped on val). Total runtime for all 21 search trials + final model: **185.7 seconds** on GPU.

---

## 5. Results

### 5.1 Raw (uncalibrated) metrics

| Split | AUC-PR | AUROC | F1 | Precision | Recall | Lift (AUPR/base-rate) |
|---|---|---|---|---|---|---|
| Train | 0.9051 | 0.9494 | 0.8145 | 0.8076 | 0.8216 | 2.98× |
| Val | 0.7632 | 0.8516 | 0.6651 | 0.6651 | 0.6652 | 2.65× |
| **Test** | **0.7237** | **0.8064** | **0.6284** | 0.6215 | 0.6353 | **2.37×** |

**Train → test gap:** AUROC 0.9494 → 0.8064 (0.143 gap), AUC-PR 0.9051 → 0.7237 (0.168 gap). This is a **larger** generalization gap than our earlier TX (0.086) or TEXAS (0.0058) models — see the honest caveats in §7 for why (test spans 5 full years including 2022, which TDIS's own original analysis flagged as scoring anomalously worse than later years; `min_child_weight=10` is less regularized than the `30` TDIS and our own prior models standardized on).

### 5.2 Isotonic-calibrated metrics (calibrator fit on val only, never on test)

| Split | AUC-PR | AUROC | Brier |
|---|---|---|---|
| Val | 0.7591 | 0.8518 | — |
| **Test** | **0.7190** | **0.8064** | raw 0.1540 → **calibrated 0.1474** |

Calibration barely moved ranking metrics (as expected — it's a monotone remap) but did improve Brier score, i.e., the probabilities are better-sized, not just better-ranked.

**Calibration curve** (`calibrator_bit_38feat.joblib`):

| Raw score | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Calibrated | 0.000 | 0.082 | 0.129 | 0.174 | 0.261 | 0.357 | 0.449 | 0.573 | 0.699 | 0.813 | 1.000 |

No saturation ceiling (unlike TDIS's served-model calibrator, which caps at 33%) — because this calibrator is fit on a ~29% base-rate validation set, not a true ~2% real-population rate. This is directly comparable to their matched-sample calibration, **not** their real-population one.

### 5.3 Threshold sweep (test set, raw probability)

| Threshold | Precision | Recall | F1 | Cells flagged |
|---|---|---|---|---|
| 0.30 | 0.487 | 0.786 | 0.602 | 622,068 |
| 0.40 | 0.574 | 0.684 | 0.625 | 459,206 |
| **0.45** | **0.622** | **0.635** | **0.628** (best F1) | 393,783 |
| 0.50 | 0.668 | 0.588 | 0.625 | 339,249 |
| 0.60 | 0.754 | 0.499 | 0.601 | 255,000 |
| 0.70 | 0.829 | 0.418 | 0.556 | 194,150 |

### 5.4 Full feature importance (all 38, XGBoost gain)

| Rank | Feature | % gain | | Rank | Feature | % gain |
|---|---|---|---|---|---|---|
| 1 | `ecoregion_id` | 20.79% | | 20 | `slope_deg` | 1.56% |
| 2 | `elevation_m` | 7.70% | | 21 | `cbh` | 1.56% |
| 3 | `hrrr_vpd` | 5.45% | | 22 | `hrrr_wind` | 1.51% |
| 4 | `avg_burn_prob` | 5.10% | | 23 | `is_weekend` | 1.48% |
| **5** | **`lat`** ⭐ | **4.48%** | | 24 | `tmmx` | 1.47% |
| **6** | **`lon`** ⭐ | **3.40%** | | 25 | `vpd` | 1.45% |
| 7 | `powerline_dist_km` | 3.28% | | 26 | `bp_x_drought` | 1.43% |
| 8 | `road_dist_km` | 3.27% | | 27 | `drought_score` | 1.43% |
| **9** | **`rmin`** ⭐ | **3.25%** | | 28 | `aspect_deg` | 1.41% |
| 10 | `sin_month` | 2.92% | | 29 | `fm100_5d_min` | 1.34% |
| 11 | `whp` | 2.22% | | 30 | `fm100` | 1.30% |
| 12 | `hrrr_mstav` | 2.05% | | 31 | `sin_dow` | 1.26% |
| 13 | `cos_month` | 2.01% | | 32 | `cos_dow` | 1.25% |
| 14 | `pop_density` | 1.98% | | 33 | `vs` | 1.23% |
| 15 | `rmax` | 1.93% | | 34 | `pr` | 1.22% |
| 16 | `hrrr_tmp` | 1.83% | | 35 | `whp_x_vpd` | 1.15% |
| 17 | `bp_x_vpd` | 1.80% | | 36 | `erc` | 1.14% |
| 18 | `is_holiday` | 1.69% | | **37** | **`flep4`** | **0.00%** |
| 19 | `cbd` | 1.63% | | **38** | **`cfl`** | **0.00%** |

⭐ = our additions. **Combined, `lat`+`lon`+`rmin` carry 11.1% of total gain** — consistent with what we found in TEXAS 32-feat (`lat`+`lon` there: 11.8%), independently reproduced on a different (TDIS-sourced) dataset this time.

**`flep4` and `cfl` are dead (0.000 importance) — the fourth independent confirmation of this finding** (previously: TDIS's own served model, our TX model, our TEXAS 30/32-feat models, and now this one).

---

## 6. Direct comparison table

| Metric | TDIS served model (matched test, 23.4% pos rate, 2023–2026) | TDIS served model (real population, 1.92% base rate) | **This retrain (TDIS's own 2022–2026 test, 30.6% pos rate)** |
|---|---|---|---|
| AUC-PR | 0.4941 | 0.0878 | **0.7237** (raw) / 0.7190 (cal) |
| AUROC | 0.7402 | 0.7968 | **0.8064** |
| F1 | 0.4958 | 0.1692 | **0.6284** |
| Lift | 2.11× | 4.58× | **2.37×** |

**Read carefully — this is not a clean win, and we want to say so plainly:** our AUROC (0.8064) beats their matched-test number (0.7402) and is close to — barely above — their *real-population* number (0.7968), which is the much harder, more honest evaluation. Our AUC-PR looks far higher (0.72 vs 0.49) but that's substantially explained by our much higher base rate (30.6% vs 23.4%) — AUC-PR scales with base rate, so it's not a fair side-by-side. The only base-rate-fair number is **lift**, and there our 2.37× is **below** both of their numbers (2.11× matched, 4.58× real-population) once you account for the fact that their real-population lift is the number that actually matters operationally, and we have not computed our own equivalent.

---

## 7. Honest caveats — what this result does and does not prove

1. **We have not run real-population validation.** Every number above is on a rebalanced/matched-sample test set (~30% positive). TDIS's strongest, most credible number (lift 4.58× at the true ~1.9% base rate) has no equivalent from us yet. This remains the single biggest gap, carried over from our prior two reports.
2. **The train→test gap (0.143 AUROC) is larger than our previous models' gaps.** The hyperparameter search picked `min_child_weight=10` (less regularized) because it scored marginally higher on validation AUC-PR than `min_child_weight=30` — the convention TDIS and our own earlier models standardized on. This is a real trade-off the search made automatically; it is not clearly the "right" choice for stability. A rerun forcing `min_child_weight=30` would be a reasonable next check.
3. **No label-shuffle or spatial-holdout leakage test was run on this model** — TDIS's own package passes both; we still have not run either on anything we've built.
4. **No monotonicity constraints.** TDIS's own `TRUST_VALIDATION.json` shows their *served* model fails physical-direction sanity checks on 5 of 9 features (wind, temperature, burn probability, WHP, soil moisture all point the wrong way); we have not checked whether this model has the same problem, and we did not add the `monotone_constraints` their own unpromoted `rev4` model uses to fix it.
5. **This is TX-only**, using TDIS's TX rows from their tri-state table — not a tri-state model, and not using their very newest rev4 peak-weather features (daily max/min across HRRR leads) which is the actual fix behind their best unpromoted result (AUC-PR 0.511, real-pop lift 4.76–4.88×).

---

## 8. What's next (recommended, not yet done)

1. Score this model against a true real-population panel (every TX cell × every day at the ~1.9–2.5% actual base rate) the way TDIS's `26_score_operational_realpop.py` / `20_realpop_res5_v2.py` do — this is the only way to make an apples-to-apples lift comparison.
2. Run the label-shuffle and spatial-holdout credibility tests.
3. Try `min_child_weight=30` (TDIS's and our own prior standard) to check whether the larger train-test gap shrinks without costing meaningful validation AUC-PR.
4. Consider adopting TDIS's rev4 "daily-peak weather + monotone constraints" fix, since it is the one documented, concrete improvement path neither of our pipelines has implemented yet.

---

## 9. Files produced

| File | Contents |
|---|---|
| `build_dataset.py` | Builds the 38-feature, zero-null training table from TDIS's `train_tristate_v2.parquet` |
| `data/bit_train_tx.parquet` | The built dataset (2,201,289 × 44) |
| `build_dataset.log` | Console log of the build step |
| `tune_train_model.py` | 21-trial two-stage hyperparameter search + final training + isotonic calibration |
| `tune_train_run.log` | Full console log of the training run |
| `model/model_bit_38feat.json` | Final trained XGBoost model |
| `model/calibrator_bit_38feat.joblib` | Isotonic probability calibrator (fit on val) |
| `model/tuning_stage1.csv`, `model/tuning_stage2.csv` | All 21 hyperparameter trials |
| `model/training_results_bit_38feat.json` | Full machine-readable results: metrics, hyperparameters, feature importances |
| `BIT_TRAINING_REPORT.md` | This document |

### Reproduce
```bash
cd "TDIS_Ignition_BIT"
"/c/Users/Admin/anaconda3/envs/torch_gpu/python.exe" build_dataset.py
"/c/Users/Admin/anaconda3/envs/torch_gpu/python.exe" tune_train_model.py
```
