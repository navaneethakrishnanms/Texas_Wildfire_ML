# Reproducibility Guide

An honest, tiered answer to "can a team reproduce this from the package
alone?" Short version: **yes for viewing and retraining, no for a from-zero
rebuild of the raw data** — and this document says exactly why, with no
hand-waving.

## Tier 1 — View the results (100% self-contained, works today)

No setup. Open `dashboard/tdis_fire_dashboard_standalone.html`,
`dashboard/tdis_fire_explorer_standalone.html` (v2, zoom-adaptive, includes
the historical scrubber), or `dashboard/tdis_fire_explorer_v3_standalone.html`
(v3, live-forecast-only: Wildfire Risk / Ignition / Fire Weather Index tabs,
24h+48h horizons, no historical scrubber) directly in a browser. Needs
internet only for base-map tiles + Leaflet/h3-js CDN libraries.

## Tier 2 — Retrain / rebuild dashboard from the included data

**This tier is now genuinely reproducible from this package alone**, on a
machine with the same NAS mount. Two things were fixed to make this true:

1. **Environment**: `environment.yml` (this directory) is a pinned export
   of the exact working conda environment (`UAI2526`, Python 3.10.18).
   Recreate with:
   ```bash
   conda env create -n tdis_repro -f environment.yml
   ```
   **Note**: this environment is shared across several unrelated projects
   on the source machine, so it contains packages TDIS Forecast doesn't
   use (e.g. `torch`, `monai`, a local editable `ts-satfire` package that
   won't resolve on a new machine — safe to ignore/remove if `conda env
   create` complains about it; nothing in `scripts/` imports it).
   TDIS-relevant packages: `xgboost`, `pandas`, `numpy`, `h3`, `herbie-data`
   (HRRR/GFS pulls), `rasterio`/`shapely`/`pyproj` (script 04 only),
   `scikit-learn`.

2. **Paths**: every script in `scripts/` originally hardcoded an absolute
   path back to the *source* project (`/net/.../alphaearth_nds/TDIS_Forecast`).
   Running them from inside this package would have silently read/written
   the source project's files, not this package's copies. **Fixed**: every
   script's `TF` (project-root) variable now resolves via
   `Path(__file__).resolve().parent.parent`, i.e. relative to wherever the
   script actually lives. Run the copies in `scripts/` here and they
   operate on `data/`, `models/`, `dashboard/` in *this* package.
   Verified (2026-08-12): `TF` resolves correctly to this package's root
   for all 25 scripts; the package's `data/` and `models/` folders were
   confirmed present under that resolved path.

**What you can actually run in this tier** (needs only files already in
`data/`/`models/`/`dashboard/` here):
- `scripts/15_reweight_fwi.py` — rebuild all 3 fire-weather-formula variants
  from the cached `data/fwi_components_res5.parquet` (~8 seconds, no
  external data).
- `scripts/16_fit_hwp_tx.py` — refit the TX Hot-Dry-Windy coefficients.
- `scripts/17_embed_standalone.py`, `20_build_explorer_data.py`,
  `21_embed_explorer.py` — rebuild the standalone dashboards from the
  included dashboard JSON + static layers.
- `scripts/18_flare_filter_retrain.py`, `19_retrain_hrrr_filtered.py`,
  `22_retrain_ceiling_filtered.py` — retrain any of the three served
  models from `data/tdis_train_daily_tx_flarefiltered.parquet` (GPU
  recommended; `xgboost` GPU support needed for 19).
- `scripts/sweep_phase6_credibility.py` — rerun the two credibility checks
  (label-shuffle leak test, spatial-holdout memorization test).
- `scripts/13_model_forecast_day.py` — run the operational model on a
  **live** HRRR/GFS forecast pull (needs internet + `herbie`, not stored
  data — this one always reaches out live, by design).
- `scripts/22_embed_v3.py` — rebuild the v3 (live-forecast-only) standalone
  dashboard from `dashboard/explorer_static_multires.json` + the two newest
  `dashboard/forecast_*.json` files (run `13` twice first — lead 24 and 48 —
  if you want a fresh forecast date).

**Validation / understand-the-model scripts** (this is the actual audit
trail — start here if you want to check the numbers, not just view them;
all run from cached data already in this package, no external deps):
- `scripts/25_rescore_all_models.py` — re-scores honest/ceiling/operational
  fresh from the training tables; ground-truth source for every AUC-PR/AUROC
  number in `docs/CLASSIFICATION_METRICS.md` §3.
- `scripts/26_score_operational_realpop.py` — scores the served operational
  model against the true deployment population (every cell × every day,
  not the rebalanced training sample) — this is the number that actually
  matters for "is this good in the real world."
- `scripts/27_fit_isotonic_operational.py` — fits + validates the
  probability calibrator with a strict temporal holdout (fit 2024-25,
  evaluate 2026-only) — proves the calibration fix isn't overfit to the
  data it was tuned on.
- `scripts/28_flare_robustness_band.py` — sensitivity check: do the
  headline numbers survive a 10x change in the flare-filtering threshold?
  (They do — see output.)
- `New_Training817_moredata/step34_mstav_retrain.py` — retrain the SERVED
  22-feature model (power-line + soil-moisture) from this package alone:
  `powerline_dist_km.parquet` and the cached `mstav_feature.parquet` are
  both included, so no external archive is needed. (The raw 2,932-day
  MSTAV grid archive stays on the source NAS; `download_mstav.py` rebuilds
  it from the public AWS HRRR archive if ever needed — internet only, no
  credentials, ~45 min.)
- `scripts/29_test_ignition_x_fwi.py` — tests whether multiplying the
  ignition model by a fire-weather index improves real-population skill.
  (It doesn't — all three variants degrade AUC-PR; the model alone wins.
  Results: `rev2_improvements/step29_ignition_x_fwi.json`.)
- `rev2_improvements/diagnose_model.py --variant baseline` — SHAP
  feature-importance + direction, plus train/val/test gap diagnostics
  (memorization vs. generalization risk). Read this to understand *why*
  the model predicts what it predicts, not just how accurate it is.
  `rev2_improvements/` also holds every feature-ablation experiment tried
  (gust, trailing weather stats, explicit interaction terms) with full
  logs and results JSONs — the negative results are recorded here too,
  not just the ones that worked.

**What you can NOT run in this tier**, because these specific scripts read
raw source files that are genuine external dependencies, not build
artifacts (see Tier 3): `01`-`06`, `09`, `14`, and the raw-rebuild path of
`04`. Attempting them will fail with a clear file-not-found on one of the
paths listed below — not a silent wrong-data bug, because of the path fix
above.

## Tier 3 — Rebuild everything from raw sources (NOT self-contained)

This is the honest limit. Full from-zero reproduction needs access to
external resources that were deliberately not copied into this package
(too large, or credentials, or belonging to other projects entirely). Exact
list, every one verified to exist at the source path as of this writing:

| Dependency | Used by | Size | Why not packaged |
|---|---|---|---|
| FIRMS API key (`firms_map_key.txt`) | `01_download_viirs.py` | tiny | Credential — request your own at https://firms.modaps.eosdis.nasa.gov/api/ |
| Raw historical HRRR forecast archive | `05_download_hrrr_forecast.py`, `06`, `11` | ~14 GB | Build cache; script 05 *can* redownload it given time + internet, no manual step needed beyond running it |
| `alphaearth_nds/gridmet_tx/` (raw gridMET NetCDF + per-cell parquet) | `03_build_dataset.py`, `09_build_perday_dynamic.py`, `14_event_validation.py` | ~15 GB | **No script in this pipeline rebuilds this from scratch** — it was populated by an earlier, undocumented process. This is the one genuine gap with no re-run path today. If you need to rebuild it, you'd need to write a gridMET (climatologylab.org) downloader — none exists in `scripts/`. |
| `alphaearth_nds/90%_ig_dec/Transferibility_OutOfState/data/...` (IgnitionNet project's TX landscape/national-raster data) | `02_build_labels.py` (FPA-FOD), `04_build_full_tx_static.py` (landscape, national rasters) | not measured here | Belongs to a separate project (IgnitionNet); shared, not duplicated |
| `alphaearth_nds/geo_cache/` (cached roads/ecoregion downloads) | `04_build_full_tx_static.py` | not measured here | Build-time cache for a different script, shared across projects |

**If your team has the same NAS mount**, all five of these already exist
and Tier 3 scripts will just work — they were left as absolute paths
precisely because they're legitimate, currently-valid shared resources, not
because of an oversight. **If your team does NOT have this NAS mount**,
Tier 3 is not achievable without first obtaining each of the above
independently; say so up front rather than promising a rebuild that can't
happen.

## Summary table

| Want to... | Tier | Works from this package alone? |
|---|---|---|
| See the dashboard / results | 1 | ✅ Yes |
| Retrain a served model | 2 | ✅ Yes (after `conda env create`) |
| Rebuild dashboard/FWI variants | 2 | ✅ Yes |
| Run credibility/leakage checks | 2 | ✅ Yes |
| Pull a fresh live forecast | 2 | ✅ Yes (needs internet) |
| Rebuild training labels from VIIRS+FPA-FOD | 3 | ⚠️ Needs FIRMS key + same-NAS access |
| Rebuild HRRR forecast-weather features | 3 | ⚠️ Needs same-NAS access (or ~14GB redownload) |
| Rebuild daily fire-weather from raw gridMET | 3 | ❌ No rebuild script exists — same-NAS access to `gridmet_tx/` required, no other path |
| Rebuild full-TX static features from scratch | 3 | ⚠️ Needs same-NAS access to `90%_ig_dec/` + `geo_cache/` |
