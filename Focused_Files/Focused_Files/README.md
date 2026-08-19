# TDIS Forecast — Minimal Package (data2)

One model, one dashboard, one dataset. Everything in this folder is
verified to work standalone — no other folder in `deliverablePackage/`
is needed for daily use. (358MB, vs. 1.1GB for the full package.)

If you want the full experiment history, every retired approach with its
evidence, the older dashboard versions, or the raw-data rebuild pipeline
— that's all one level up in `deliverablePackage/`. This folder is
deliberately just "run it and understand it," nothing else.

## What's here

```
models/
  tdis_forecast_hrrr_filtered.json          the served model (XGBoost, 22 features)
  operational_isotonic_calibrator.joblib     its calibrator — ALWAYS use together
  tdis_forecast_hrrr_filtered_meta.json      metadata

dashboard/
  tdis_fire_explorer_v3_standalone.html      OPEN THIS — drag into a browser
  tdis_fire_dashboard_v3.html                template (to rebuild the above)
  explorer_static_multires.json              static hazard layer (Wildfire Risk tab)
  forecast_2026-08-18.json, forecast_2026-08-19.json   current 24h/48h forecast
  README_v3.md                               explains the 3 tabs + the NOAA HWP equation

data/
  tdis_train_daily_hrrr.parquet              training table (what the model was trained on)
  static_features/tx_static_master.parquet   static terrain/fuels/access features
  static_features/powerline_dist_km.parquet  power-line-distance feature (added 2026-08-17)
  labels_fused/ignitions_daily_tx.parquet    fire labels (fused VIIRS + FPA-FOD)
  labels_fused/flare_cells_v2.parquet        flare/industrial cells excluded (538, audited)

New_Training817_moredata/
  mstav_feature.parquet                      cached soil-moisture feature (avoids re-download)
  step34_mstav_retrain.py                    retrains the EXACT served model from this data

scripts/
  13_model_forecast_day.py                   run this daily — pulls live HRRR, scores, calibrates
  22_embed_v3.py                              rebuilds the dashboard from fresh forecasts
  fwi_config.py                               fire-weather formula (imported by both scripts)

docs/
  METHODOLOGY.md              the locked, canonical method — read this first
  CLASSIFICATION_METRICS.md   full metrics, both populations, all context

environment.yml    conda environment
REPRODUCE.md       what's reproducible from here vs. what needs the full package
```

## Daily use (verified working from this folder alone)

```bash
cd scripts
python3 13_model_forecast_day.py <tomorrow's date> 24
python3 13_model_forecast_day.py <day after> 48
python3 22_embed_v3.py
```
Produces a fresh `dashboard/tdis_fire_explorer_v3_standalone.html`. Needs
internet only (public NOAA HRRR data, no credentials).

## Retraining (verified working from this folder alone)

```bash
cd New_Training817_moredata
python3 step34_mstav_retrain.py
```
Reproduces the exact served model from the included training table +
cached features.

## The one rule that matters

**Model and calibrator are a pair.** If you ever retrain, refit the
calibrator on the new model's output before serving it — a new model
under the old calibrator will report wrong probabilities. See
`docs/METHODOLOGY.md` for why (isotonic regression, temporal holdout).
