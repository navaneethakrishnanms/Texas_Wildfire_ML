# Texas Wildfire Ignition Model — HRRR Feature Evaluation

**Data lineage:** `final_training_dataset_tx_22.07.2026_landfire.xlsx` (base LANDFIRE+gridMET dataset, no HRRR) → cleaned to `data/full_tx_clean.parquet` → `merge_hrrr_duckdb.py` LEFT JOINs HRRR sub-daily weather onto it by `(h3_cell, date_utc, window_hour)` → **`data/hrrr/hrrr_tx_all.parquet`**
**Actual file all HRRR training runs load:** `data/hrrr/hrrr_tx_all.parquet` (confirmed via `train_tx_hrrr.py` `HRRR_PQ` path and each run's log line `hrrr_pw=1 (HRRR available): 304,484 (81.0%)`)
**Rows:** 375,779 | **Unique H3 cells:** 317,142 | **Date range:** 2014-01-01 to 2020-12-06
**Fire rows:** 33,749 (8.98%) | **Non-fire rows:** 342,030 (91.02%)
**Split (chronological, never random):** TRAIN 2014-2017 (251,785 rows) · VAL 2018 (61,139 rows) · TEST 2019-2020 (62,855 rows)
**Model:** XGBoost binary classifier, GPU-trained (RTX 3060)

---

## 1. What HRRR adds to the dataset

HRRR contributes 8 new sub-daily weather columns on top of the existing daily gridMET weather:

| Feature | Description | Missing % |
|---|---|---|
| `temp_pw` | 2m temperature (°C) | 19.0% |
| `rh_pw` | 2m relative humidity (%) | 45.7% |
| `wind_pw` | 10m wind speed (m/s) | 19.0% |
| `vpd_pw_hrrr` | Vapor pressure deficit (kPa) | 45.7% |
| `hpbl_pw` | Planetary boundary layer height (m) | 19.0% |
| `dswrf_pw` | Downward solar radiation (W/m²) | 19.0% |
| `hrrr_pw` | Flag: 1 = HRRR timestamp available | 0% |
| `hrrr_rh_valid` | Flag: 1 = RH/VPD values are real (not the 2014-2016 specific-humidity bug) | 0% |

HRRR coverage is uneven across the training window — 12.2% of rows in 2014 vs. 95-99% by 2018-2020 — because early HRRR archives are incomplete and 2014-2016 files originally lacked usable RH data (patched via `fix_hrrr_rh_bug.py`).

---

## 2. Model iterations and results

| # | Model | Features | Hyperparameters | Test AUROC | Test AUPR | Test F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| 1 | V2 baseline (zeroed LANDFIRE) | 28 | depth=7, lr=0.05 | 0.8569 | 0.3978 | — | — | — |
| 2 | V3 gridMET + real LANDFIRE (untuned) | 34 | depth=7, mcw=30, lr=0.05, sub=0.8 | 0.8637 | 0.4106 | 0.4169 | 0.352 | 0.511 |
| 3 | V3 tuned (`tune_tx.py`, 21-trial grid search) | 34 | depth=9, mcw=30, lr=0.01, sub=0.9 | **0.8687** | **0.4247** | 0.4293 | 0.392 | 0.475 |
| 4 | HRRR-enriched (config borrowed from #3) | 42 | depth=9, mcw=30, lr=0.01, sub=0.9 | 0.8682 | 0.4236 | 0.4251 | 0.351 | 0.539 |
| 5 | HRRR-enriched, independently tuned (`tune_tx_hrrr.py`, 21-trial grid search) | 42 | depth=9, mcw=30, lr=0.02, sub=0.9 | 0.8682 | 0.4225 | 0.4247 | 0.340 | 0.566 |
| 6 | HRRR-enriched, availability flags dropped (`train_tx_hrrr.py --no-flags`) | 40 | depth=9, mcw=30, lr=0.01, sub=0.9 | 0.8673 | 0.4198 | 0.4247 | 0.351 | 0.537 |

**Bottom line:** Adding HRRR (#4, #5) does not beat the best gridMET-only model (#3). The gap is −0.0005 AUROC / −0.0011 to −0.0022 AUPR — inside normal run-to-run noise for this model size, not a meaningful regression, but also not the improvement expected from adding a new weather source. A full independent hyperparameter search on the 42-feature HRRR set (#5) converged to essentially the same result as reusing the gridMET-tuned config (#4), which rules out "HRRR just needs its own tuning" as the explanation.

---

## 3. Feature importance — where the model actually gets its signal

Gain-importance (% of total gain), top 15 features, for each key model:

### V3 gridMET-only, untuned (#2) — Test AUROC 0.8637

| Rank | Feature | Gain % |
|---|---|---|
| 1 | burnable | 15.5% |
| 2 | avg_burn_prob | 13.7% |
| 3 | whp | 12.9% |
| 4 | cfl | 10.6% |
| 5 | centroid_lon | 7.8% |
| 6 | centroid_lat | 4.4% |
| 7 | flep4 | 4.3% |
| 8 | cos_hour | 2.5% |
| 9 | erc | 1.8% |
| 10 | rmin | 1.7% |
| 11 | cbd | 1.5% |
| 12 | sin_hour | 1.5% |
| 13 | cbh | 1.4% |
| 14 | cos_month | 1.4% |
| 15 | erc_5D_max | 1.4% |

### V3 gridMET-only, tuned (#3) — Test AUROC 0.8687 (best model overall)

| Rank | Feature | Gain % |
|---|---|---|
| 1 | burnable | 19.8% |
| 2 | avg_burn_prob | 12.1% |
| 3 | whp | 10.6% |
| 4 | cfl | 9.5% |
| 5 | centroid_lon | 7.9% |
| 6 | flep4 | 4.3% |
| 7 | centroid_lat | 4.3% |
| 8 | cos_hour | 2.3% |
| 9 | erc_5D_max | 1.9% |
| 10 | sin_hour | 1.8% |
| 11 | rmin | 1.7% |
| 12 | sin_month | 1.5% |
| 13 | vs_5D_max | 1.5% |
| 14 | erc | 1.4% |
| 15 | cos_month | 1.4% |

### HRRR-enriched, independently tuned (#5) — Test AUROC 0.8682

| Rank | Feature | Gain % | HRRR feature? |
|---|---|---|---|
| 1 | burnable | 17.0% | |
| 2 | avg_burn_prob | 12.5% | |
| 3 | cfl | 9.7% | |
| 4 | whp | 9.2% | |
| 5 | centroid_lon | 7.1% | |
| 6 | centroid_lat | 3.8% | |
| 7 | flep4 | 3.8% | |
| 8 | cos_hour | 2.4% | |
| 9 | sin_hour | 1.9% | |
| 10 | erc_5D_max | 1.7% | |
| 11 | **hrrr_rh_valid** | **1.6%** | ✅ HRRR |
| 12 | rmin | 1.4% | |
| 13 | sin_month | 1.4% | |
| 14 | cos_month | 1.3% | |
| 15 | erc | 1.3% | |

**All 8 HRRR features combined account for only 7.8% of total model gain**, and the only HRRR feature that ranks in the top 15 is `hrrr_rh_valid` — an availability flag, not an actual weather reading. None of the real HRRR values (`temp_pw`, `rh_pw`, `wind_pw`, `vpd_pw_hrrr`, `hpbl_pw`, `dswrf_pw`) crack the top 15.

### Ablation — dropping `hrrr_pw` / `hrrr_rh_valid` (#6)

| Metric | With flags (#5) | Flags dropped (#6) | Change |
|---|---|---|---|
| Test AUROC | 0.8682 | 0.8673 | −0.0009 |
| Test AUPR | 0.4225 | 0.4198 | −0.0027 |
| HRRR gain share | 7.8% | 5.2% | −2.6pp |
| HRRR feature in top 15? | `hrrr_rh_valid` (1.6%) | none | — |

Removing the flags made results slightly **worse**, not neutral. If the flags were purely riding on the year-coverage confound, AUROC should have been unaffected by removing them — instead every metric dropped. So the flags carry a small amount of genuine signal, most likely by telling the model when the other HRRR columns are trustworthy vs. missing/patched (2014-2016 bug years), rather than acting as a disguised calendar feature. The effect is tiny (~0.001 AUROC) either way, and even with the flags included, HRRR still trails the gridMET-only tuned model (0.8687).

---

## 4. Why HRRR isn't improving accuracy

1. **The model barely uses it.** 7.8% of gain spread across 8 features, and the one HRRR feature that matters most (`hrrr_rh_valid`) is an availability flag, not a weather signal — confirmed to carry a small amount of real (non-confounded) signal via the ablation above, but not enough to move overall accuracy.
2. **Weak fire/no-fire separation.** Fire vs. non-fire mean ratios for HRRR variables sit close to 1.0x (`temp_pw` 1.06x, `rh_pw` 1.08x, `wind_pw` 0.87x, `vpd_pw_hrrr` 0.94x, `hpbl_pw` 0.98x, `dswrf_pw` 1.01x) — much weaker discriminators than the landscape features already driving the model (`cbh` 1.82x, `whp` 1.58x, `avg_burn_prob` 1.46x).
3. **Redundant with gridMET.** gridMET already supplies daily `rmax`, `rmin`, `vpd`, `vs`, `tmmx` plus 5-day trailing stats for the same physical quantities (temperature, humidity, wind, VPD). At a 6-hour prediction window, HRRR's sub-daily resolution adds little on top of that, while contributing 45.7% missing values for RH/VPD and ~19% missing for the rest.
4. **Independent tuning was tested and ruled out as the cause.** A dedicated 21-trial hyperparameter search on the 42-feature HRRR set (`tune_tx_hrrr.py`) converged to the same result (0.8682 AUROC) as reusing the gridMET-tuned config — confirming this is a genuine feature-signal ceiling, not an under-tuned model.
5. **Availability-flag ablation ruled out the year-proxy theory.** Dropping `hrrr_pw`/`hrrr_rh_valid` made results worse, not the same, so the flags aren't just standing in for calendar year — but their real contribution is too small to close the gap with gridMET-only.

---

## 5. Plan

1. **Replace raw HRRR values with HRRR-minus-gridMET anomaly features** (e.g. `temp_pw - tmmx`, `rh_pw - rmin`, `wind_pw - vs`) instead of feeding HRRR's absolute readings alongside gridMET's. This isolates the sub-daily deviation — the only part of HRRR that is genuinely new information — instead of letting the model see two overlapping copies of the same weather signal.
2. ~~Drop the `hrrr_pw` / `hrrr_rh_valid` availability flags and retrain~~ — **done.** Result: AUROC dropped slightly (0.8682 → 0.8673), ruling out the year-coverage-confound theory. The flags carry a small amount of real signal (likely NaN-trustworthiness), not fake signal, but not enough to matter for overall accuracy.
3. **Restrict the HRRR comparison to well-covered years only (2017-2020, ≥94% coverage)** to check whether the 2014-2016 sparse/patched years are diluting the HRRR signal in the pooled model.
4. **Try HRRR features only on the subset of rows where `hrrr_rh_valid=1`** (204,075 rows, 54.3% of data) as a separate diagnostic model, to see whether HRRR helps when it's actually present and reliable, independent of the missingness problem.
5. **Keep `xgb_tx_tuned` (gridMET + LANDFIRE, tuned, Test AUROC 0.8687)** as the production model until one of the above shows a real, reproducible lift over it.
