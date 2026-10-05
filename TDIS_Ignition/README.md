# TDIS Wildfire Ignition — 24–48 h Model Package

**Deliverable for the TDIS data-engineering / platform team.**
Everything needed to retrain, score, and serve the **forecast-realistic daily
wildfire-ignition model** (the "24–48 hour" model), minus the 130 GB raw weather
archive, which is re-pullable from public AWS (see §6).

Packaged 2026-09-22 from `.../miguel_shared/alphaearth_nds/TDIS_Forecast`.
Total size ≈ 787 MB (core TX package; ~1.9 GB with the sub-packages below).

**Companion sub-packages (added 2026-09-23, each with its own README):**
- **`model_tristate/`** — the rev5 tri-state model: trained on diverse TX+NM+OK
  data, operational target = all-of-Texas population. Research-validated,
  needs pipeline work — read its README §1/§5/§6 before building on it.
- **`fireweather/`** — the HWP/FWI fire-weather product (NOAA HWP Eq. 3 exact +
  TX-fit + composite variants, params, validation, worked example). Separate
  product from ignition — **never multiply the two** (see its README §5).
  Team-facing spec: `fireweather/PRD_FireWeather_HWP.md` (TRD-style PRD with
  problem statement, data flow, handoff assets, and acceptance criteria).
- **`validation_visuals/`** — six peace-of-mind figures (served PR curve,
  reliability, load-response for served + rev5, fire-weather physics and
  severity signal). Regenerate with `serve/make_validation_visuals.py`;
  the numbers behind them live in `model/TRUST_VALIDATION.json`.

---

## 1. What "the 24–48 hour model" is

A single **XGBoost binary classifier** that predicts, for each **H3 res-8 cell
(~860 m) × day**, the probability that a wildfire ignites. It is
**forecast-realistic**: every day's weather features come from the **12Z HRRR
run of the day *before*** the target day, leads **F24 / F27 / F30 / F33 / F36**.
Because the run is issued ~18Z (≈2 pm local) the day before, the target day sits
**24–48 h ahead of issue time** — hence "24–48 h model." It is deployable, not a
hindcast.

- **Grid:** H3 resolution-8 (~860 m), keyed `h3_cell`.
- **Label:** flare-filtered fused VIIRS + FPA-FOD ignition (cell-day binary).
- **Calibration:** raw XGBoost output is **not** a probability — apply the
  isotonic calibrator (§2). Post-calibration ECE 0.0019 (holdout).
- **Lineage:** this is the promoted **d9** model (method locked 2026-08-17),
  with `hrrr_mstav` (soil-moisture availability), `drought_score` (USDM), and
  `powerline_dist_km` added on top of the rev4 base. Flare cells excluded from
  training.

### Real-population performance (TX, 2024–2026, base rate 1.92 %)
| Metric | Value (re-derived 2026-09-23 from the packaged scored panel — authoritative) |
|---|---|
| AUC-PR | **0.0878** |
| AUROC | **0.797** |
| Lift @ base rate | **4.58×** |
| ECE calibrated (2026 holdout) | **0.0008** |

*(An earlier claims file said 0.085/0.784/4.43× — generated pre-promotion and
understated the shipped model. `serve/validate_trust.py` re-derives the numbers
above from `data/scored/operational_historical_res5_v2.parquet` — run it
yourself for independent proof; full battery in `model/TRUST_VALIDATION.json`.)*

Numbers on a *matched* (fires 9× over-represented) test set are higher (AUC-PR
0.49 / AUROC 0.74) — see `model/tdis_forecast_hrrr_filtered_meta.json`. Always
quote the **real-population** numbers operationally; the matched numbers are for
model-selection only.

---

## 2. `model/` — trained artifacts

| File | What it is |
|---|---|
| **`tdis_forecast_hrrr_filtered.json`** | **THE served model.** XGBoost JSON, 23 features, flare-filtered, d9-promoted. Load with `xgboost.Booster().load_model(...)`. |
| **`operational_isotonic_calibrator.joblib`** | **THE paired calibrator.** Apply to the model's raw margin to get a real probability. `joblib.load(...)`. |
| `tdis_forecast_hrrr_filtered_meta.json` | Feature order, split, importances. ✅ *corrected 2026-09-23 to the 23-feature ground truth* (was 20/pre-d9). Matches the model file's `feature_names`. |
| `tdis_forecast_hrrr_tuned_d9.json` | Training-side twin of the served model (same weights, from the training run). |
| `tuned_d9_isotonic_calibrator.joblib` | Calibrator snapshot from the d9 promotion step (lineage; the *operational* one above is the live one). |
| `promote_d9_report.json` | Promotion audit: rows, holdout ECE/Brier raw vs calibrated. |
| `operational_realpop_validation.json` | Full real-population validation (the §1 table + reliability bins). |
| `isotonic_calibration_report.json` | Calibration fit report. |

**Feature order matters** — score with the exact order in the model file's
`feature_names` (list in §3).

---

## 3. Feature set (23 features — ground truth from the model file)

**Static (landscape/terrain — one-time extraction, `data/static/`):**
`road_dist_km`, `ecoregion_id`, `elevation_m`, `slope_deg`, `aspect_deg`,
`avg_burn_prob`, `whp`, `flep4`, `cfl`, `cbd`, `cbh`, `powerline_dist_km`

**Temporal encodings (computed from the target date, no data pull):**
`sin_month`, `cos_month`, `sin_dow`, `cos_dow`, `is_weekend`, `is_holiday`

**Dynamic (live pull — see PRD §3.4 in `docs/`):**
`hrrr_tmp`, `hrrr_vpd`, `hrrr_wind`, `hrrr_mstav` (from 12Z HRRR), `drought_score` (USDM weekly)

`flep4` and `cfl` have **zero importance** in the trained model — kept for schema
compatibility only. Top drivers: `ecoregion_id`, `elevation_m`, `hrrr_vpd`,
`sin_month`, `road_dist_km`.

---

## 4. `data/` — inputs

| Path | Contents |
|---|---|
| `data/static/` | All static feature stores keyed by H3 cell: `tx_static_master.parquet` (the joined TX master), plus per-component parquets (`elevation_slope_aspect_full`, `ecoregion_full`, `cbd_cbh_full`, `road_dist_full`, `powerline_dist_km`, `landscape_by_cell`, `tx_county_fips`), the `drought_weekly`/`drought_latest` USDM caches, and the d9-added `mstav_feature`, `pop_density_feature`, `fire_nearby_3d_feature`. |
| `data/labels/` | `ignitions_daily_tx.parquet` (fused VIIRS+FPA-FOD training target, cell-day) and `viirs_tx_h3.parquet` (raw VIIRS 375 m active-fire, H3-keyed). ⚠ **Neither file has flare cells removed yet** — ~27% of `ignitions_daily_tx.parquet`'s positive rows sit in flare cells (verified `serve/verify_package.py` §3). Flare filtering is a downstream join step (see `code/scripts/18_flare_filter_retrain.py`), not baked into these files — always join against `data/flares/` before treating either as ground truth. The **served model itself was trained flare-filtered**; only these two raw label files are unfiltered. |
| `data/flares/` | Flare-exclusion cell lists: `flare_cells_tx_v2.parquet` (531/538), `flare_cells_nm.parquet` (1,956), `flare_cells_ok.parquet` (260). **Remove these cells from `data/labels/` before using it as ground truth** — they are not pre-applied. |
| `data/nm_ok/` | New Mexico + Oklahoma parallel data (FPA-FOD H3, VIIRS-derived, canopy, drought) for the tri-state extension. |
| `data/scored/` | `tuned_d9_historical_res5.parquet` + `operational_historical_res5_v2.parquet` (historical scored maps, res-5 aggregated, for the dashboard/back-testing); `peak_weather_features.parquet` (rev4 12Z-HRRR daily peaks joined to cell-days — the assembled dynamic training table); `rev4_realpop_scores.parquet`. |

**Not included (too large / raw):** raw FPA-FOD CSVs
(`30%_Rev2/data/ignition/20*_FPA_FOD_cons.csv`) and the 130 GB HRRR archive
(§6). The processed derivatives above are sufficient to retrain and serve.

---

## 5. `code/` — pipeline

Run under the lab's conda env (`UAI2526`). Requires `xgboost`, `h3`, `pandas`,
`pyarrow`, `scikit-learn`, `joblib`, `herbie-data`, `xarray`, `cfgrib`.

| Path | Contents |
|---|---|
| `code/scripts/` | Main TDIS_Forecast pipeline, numbered end-to-end: `01_download_viirs` → `02_build_labels` → `03_build_dataset` → `04_build_full_tx_static` → `05_download_hrrr_forecast` → `06_attach_hrrr_forecast` → `11_train_hrrr_forecast` → `18/19_flare_filter_retrain` → `26_score_operational_realpop` → `27_fit_isotonic_operational`. **Serving:** `13_model_forecast_day.py` / `10_live_forecast_day.py` score one target day; `fetch_live_drought.py` pulls the weekly USDM value. |
| `code/training/` | d9 promotion + feature-add scripts (`promote_d9.py`, `step32_powerline_retrain.py`, `step34_mstav_retrain.py`, `build_contagion_population.py`, `download_mstav.py`) + `README.md`. |
| `code/rev4/` | `rev4_pipeline.py` (the daily-max base training pipeline the d9 model builds on) + ablations (`fuel_pop_ablation`, `fuel_lag_ablation`, `run_lofo`, `rev4_tau_and_ecoregion`). |
| `code/nm_ok/` | NM/OK static-master, VIIRS, FPA, drought, flare-list, and train-table builders (`01`–`10`) for reproducing the tri-state inputs. |

The `code/` scripts are the *research/training* lineage (source-repo paths, kept
for reproducibility). For **production scoring use `serve/score_day.py`** (§5b) —
it is self-contained and package-relative; do not run the `code/scripts/13`/`10`
serving scripts directly (they point at the original source repo).

## 5b. `serve/` — production scoring entrypoint ✅ validated

`serve/score_day.py` scores one target day for all 1.7M TX cells using only files
in this package. Paths resolve relative to the package, so it runs anywhere.

```bash
conda activate UAI2526            # or: pip install -r requirements.txt

# LIVE: pull the 12Z-day-before HRRR (lead F24 = 24-48h) and score the full grid
python serve/score_day.py 2026-09-20 24

# OFFLINE: score from packaged historical HRRR peaks (no network) — smoke test/backfill
python serve/score_day.py 2024-02-27 --offline
```

**Pipeline (built in):** load model + calibrator → static master (+ powerline) →
attach USDM `drought_score` (cached `data/static/drought_latest.parquet`) → add
temporal encodings → HRRR weather (live via Herbie, or packaged peaks offline) →
assemble the 23 features in `feature_names` order → `predict_proba` → isotonic
calibrate → flag flare cells.

**Outputs** (to `serve/output/`, sample included):
- `forecast_<date>_lead<LEAD>h_res8.parquet` — per cell: `h3_cell,lat,lon,ign,ign_cal,is_flare`
- `forecast_<date>.json` — res-5 aggregate `{ign, ignCal}`, **flares excluded**

**Read `ign_cal`, not `ign`.** `ign` is a rank (rebalanced-sample score, ~17× overstated);
`ign_cal` is the calibrated fire-day probability (holdout ECE 0.0018).

**Both modes validated 2026-09-23:** live pulled HRRR + scored 1,708,940 cells
(cal mean 0.0143); offline scored the packaged date (cal mean 0.0163) — both
consistent with the 1.9% base rate.

## 5c. Production-readiness checklist

| Item | Status |
|---|---|
| Served model + paired calibrator load & score | ✅ |
| Feature list ground-truth (23) — meta corrected | ✅ |
| Self-contained, package-relative serving entrypoint | ✅ `serve/score_day.py` |
| End-to-end live HRRR pull → full-grid score | ✅ tested |
| Offline smoke test (no network) | ✅ tested |
| Flare exclusion in operational output | ✅ (`is_flare` flag; aggregate excludes) |
| Calibration applied (raw→probability) | ✅ |
| `requirements.txt` / env pinned | ✅ |
| Docs (TRD/PRD/overview/dataset) | ✅ `docs/` |
| Drought cache freshness | ⚠ `drought_latest.parquet` is a snapshot; USDM updates weekly (Thu) — schedule a refresh (`code/scripts/fetch_live_drought.py`, fix its path to `data/static/`) |
| Live weather = daily HRRR pull | ⚠ operational dependency on AWS `noaa-hrrr-bdp-pds` (public, no auth) |
| Monitoring / drift | ⬜ not included — add downstream (compare daily cal-mean vs ~0.015 baseline) |

---

## 6. Excluded: the 130 GB raw HRRR archive (re-pull, don't copy)

The raw GRIB2 forecast archive
(`alphaearth_nds/TDIS_Forecast/data/weather_hrrr_forecast/`, **130 GB**) is **not**
in this package. It is **fully re-pullable** from public AWS and is only needed
to *rebuild history* — live serving pulls fresh 12Z HRRR each day.

- **Source:** `s3://noaa-hrrr-bdp-pds/hrrr.{YYYYMMDD}/conus/hrrr.t12z.wrfsfcf{FF}.grib2`
  (public, no auth), via **Herbie** `Herbie(init, model="hrrr", product="sfc", fxx=lead)`.
- **Leads:** F24, F27, F30, F33, F36 off the day-before 12Z run.
- **GRIB subset:** `:(TMP|DPT):2 m above ground|:WIND:10 m|:GUST:surface|:MSTAV:`;
  derive VPD from TMP+DPT (Tetens); peaks = max over leads (mstav = min).
- **Re-pull script:** `code/scripts/05_download_hrrr_forecast.py`
  (CA reference: `90%_ig_dec/CA_rev5/scripts/01_download_hrrr_12z_ca.py`).
- Full spec: `docs/PRD_daily_ignition_section3_datareqs.md` §3.4-B.

The assembled dynamic training table (`data/scored/peak_weather_features.parquet`)
already contains the derived HRRR peaks, so **you do not need the raw archive to
retrain** — only to extend history to new dates.

---

## 7. `docs/`

| File | Purpose |
|---|---|
| `TRD_wildfire_ignition_v1.md` | Full standalone Technical Requirements Doc (ignition, 24–48 h horizon), TDIS TRDv5.1 format. |
| `TRD_wildfire_ignition_dataflow.md` | Data-flow section (2.3.11) variant. |
| `PRD_daily_ignition_section3_datareqs.md` | Input-data requirements (DR-* IDs) + **§3.4 live-pull specs** for the DE team (HRRR / USDM / gridMET / VIIRS endpoints, cadence, auth). |
| `PROJECT_OVERVIEW.md` | Plain-English guide to the whole project (two-axis framing, glossary, findings). |
| `DATASET_BREAKDOWN.md` | Full feature list, importances, descriptive stats. |

---

## 8. Notes / caveats

- **TX-first.** The served model and most data are Texas; NM/OK inputs are
  staged in `data/nm_ok/` for the tri-state extension but the packaged model is
  TX. Re-extract static features CONUS-wide before scoring other states.
- **TX label times are 87 % imputed** (daily-only FPA) — fine for this daily
  model; do not use for sub-daily claims.
- **Flare exclusion is mandatory** — ~28 % of raw TX VIIRS detections are
  industrial gas flares; training on them corrupts the model.
- **Calibrate before thresholding** — the raw score is a ranking, not a
  probability.
- **Do not interpret single-feature responses causally** (finding from
  `serve/validate_trust.py`, full curves in `model/TRUST_VALIDATION.json`):
  the served model is *unconstrained*, and its partial responses on
  wind/temp/hazard run against physical direction (e.g. higher gust → slightly
  lower score) — the training data confounds windy/hot with other conditions
  (cold-front rain days, low-ignition summer humidity). This is the documented
  motivation for the later rev4 candidate (monotone constraints; source repo
  `Training_rev4_dailymax/RESULTS.md` Finding 9), which was never promoted.
  The model's *measured ranking performance* (re-derived above) is unaffected —
  it ranks risk well — but its per-feature behavior is correlational, not
  physical. VPD and drought do respond in the correct direction.
