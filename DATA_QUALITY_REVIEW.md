# TDIS Wildfire Forecast — Data & Process Quality Review

**Scope:** `Focused_Files/` and `Supporting_files/` deliverable packages
**Review date:** 20 August 2026
**Method:** direct inspection of the shipped parquet/CSV/JSON artefacts, model files, and scripts. Every quantitative claim below was recomputed from the files; nothing is carried over from the project's own documentation unless explicitly attributed.

---

## Executive Summary

The packages contain a coherent, well-instrumented machine-learning pipeline: an XGBoost ignition model over H3 cells, a temporal train/validation/test split, an isotonic probability calibrator, a live NOAA HRRR scoring path, and a browser-based dashboard. The engineering discipline is above average for a research handoff — negative experimental results are retained, a calibrator is shipped paired with its model, and `REPRODUCE.md` makes a genuine attempt at an honest reproducibility statement.

However, the review found **eleven material defects**, four of which are severe enough to block operational use in their current state:

| # | Finding | Severity |
|---|---|---|
| 1 | 86,072 fabricated future-dated rows in the shipped training table, all labelled `0`, all inside the `test` split | **Critical** |
| 2 | The served model's `meta.json` describes a different model (20 features vs. 22, 531 flare cells vs. 538, wrong split) | **Critical** |
| 3 | Four mutually exclusive definitions of the train/val/test split across docs, data, and code | **Critical** |
| 4 | `Focused_Files` is missing `data/hwp_params.json`; the fire-weather code silently falls back to placeholder constants, changing the shipped dashboard's index by ~50% | **Critical** |
| 5 | Headline performance numbers differ across four artefacts for the same model | High |
| 6 | The retrain script's input paths do not resolve inside `Focused_Files`; the package is not self-contained as claimed | High |
| 7 | 64.3% of training rows carry no weather values at all; 2014–2017 have none | High |
| 8 | ~54% of the "Texas" grid lies outside Texas, with all US hazard layers zero | High |
| 9 | The project's own robustness JSON contradicts the documentation that cites it | High |
| 10 | Label provenance changes regime at the train/test boundary (FPA-FOD stops after 2020) | Medium |
| 11 | The shipped dashboard presents 18–19 August forecasts as "current", with no staleness check | Medium |

**On the date question raised in the brief:** the data extending to 31 December 2026 is **not** a forecast and **not** planned data. It is 73,873 machine-generated negative-sample rows for dates that have not yet occurred, with empty weather columns and hard-coded `label = 0`, sitting inside the evaluation split. This is addressed in full in the *Date/Timeline Issue* section.

---

## Files Reviewed

**`Focused_Files/Focused_Files/`** — the package the README designates as the operational deliverable:

| Group | Artefacts inspected |
|---|---|
| Data | `data/tdis_train_daily_hrrr.parquet` (3,595,513 × 33), `data/labels_fused/ignitions_daily_tx.parquet` (960,054 × 7), `data/labels_fused/flare_cells_v2.parquet` (538), `data/static_features/tx_static_master.parquet` (1,708,940 × 15), `data/static_features/powerline_dist_km.parquet` (1,708,940 × 2), `New_Training817_moredata/mstav_feature.parquet` (2,007,436 × 3) |
| Models | `models/tdis_forecast_hrrr_filtered.json`, `..._meta.json`, `models/operational_isotonic_calibrator.joblib` |
| Dashboard | `dashboard/forecast_2026-08-18.json`, `forecast_2026-08-19.json`, `explorer_static_multires.json`, v3 template + standalone HTML |
| Code | `scripts/13_model_forecast_day.py`, `scripts/22_embed_v3.py`, `scripts/fwi_config.py`, `New_Training817_moredata/step34_mstav_retrain.py` |
| Docs | `README.md`, `REPRODUCE.md`, `docs/METHODOLOGY.md`, `docs/CLASSIFICATION_METRICS.md`, `dashboard/README_v3.md` |

**`Supporting_files/Supporting_files/`** — the fuller archive: 29 numbered pipeline scripts, `data/tdis_train_daily_tx_flarefiltered.parquet` (3,243,435 × 30), `data/labels_viirs/viirs_tx_h3.parquet`, `data/fwi_components_res5.parquet`, `data/hwp_params.json`, 12 model/validation JSONs, the `rev2_improvements/` and `New_Training817_moredata/` experiment trails, and 12 documents.

Binary-identical duplication was confirmed across the two packages for 17 of 19 common paths; `README.md` and `tdis_fire_explorer_v3_standalone.html` differ.

---

## Process Performed

Reconstructed from the scripts and the artefacts they emit.

**1. Label construction (`01`–`02`).** VIIRS active-fire detections are pulled from NASA FIRMS, gridded to H3 res-8, and fused with the FPA-FOD fire-occurrence database into `ignitions_daily_tx.parquet` — one row per (cell, day) that had fire activity, 960,054 rows spanning 2014-01-01 to **2026-07-29**. Every row carries `label = 1`; negatives are not stored here.

**2. Static feature build (`04`).** A 1,708,940-cell H3 res-8 grid over a Texas bounding box, joined to terrain (elevation, slope, aspect), USFS/LANDFIRE fuel and hazard rasters (`avg_burn_prob`, `whp`, `flep4`, `cfl`, `cbd`, `cbh`), EPA ecoregion, and road distance. `powerline_dist_km` (HIFLD transmission lines) was appended 2026-08-17.

**3. Training table assembly (`03`).** Every fire-day is retained and a matched sample of non-fire-days is drawn from the same cells, producing a deliberately rebalanced table at ~21–29% positive rate against a real-world rate near 1.9%. Calendar features are derived trigonometrically (`sin/cos` of month and day-of-week, `is_weekend`, `is_holiday`).

**4. Weather attachment (`05`–`06`).** Archived NOAA HRRR **24-hour forecast** fields — not observations — are nearest-neighbour-joined per cell per day: `hrrr_tmp`, `hrrr_vpd`, `hrrr_wind`, and later `hrrr_mstav` (soil-moisture availability). The stated rationale is sound: training on the same product served at inference avoids train/deploy skew.

**5. Filtering and training (`18`–`19`, `step32`, `step34`).** 538 flare/industrial cells are excluded, rows with missing static or weather values are dropped, and XGBoost is fitted (1,000 trees, depth 9, `min_child_weight` 30, lr 0.02, `scale_pos_weight` derived from the train fold).

**6. Calibration (`26`–`27`).** The model is replayed over the real full population (5,064,462 cell-days), then an isotonic regression is fitted on 2024–25 outcomes and validated on a 2026 holdout, mapping the inflated raw score onto a real probability (`ign_cal`).

**7. Daily inference (`13`).** Herbie pulls the live HRRR (or GFS beyond 48 h), rebuilds the 22-column matrix for all 1.7M res-8 cells, predicts, calibrates, computes three fire-weather indices, and aggregates res-8 → res-5 into `forecast_<date>.json`.

**8. Dashboard build (`22_embed_v3`).** The two lexicographically newest `forecast_*.json` files plus the static hazard layer are string-substituted into an HTML template to produce a standalone explorer.

---

## Key Findings / Flaws

### F1 — The served model's metadata describes a different model *(Critical, confirmed)*

`models/tdis_forecast_hrrr_filtered_meta.json` is the machine-readable contract for the shipped model. Reading the model binary directly contradicts it on every structural field:

| Field | `meta.json` says | Model file actually contains |
|---|---|---|
| Feature count | 20 | **22** (`num_feature: 22`) |
| `powerline_dist_km` | absent | **present** |
| `hrrr_mstav` | absent | **present** |
| Flare cells filtered | 531 | 538 (`flare_cells_v2.parquet`) |
| Split | train `2018-07..2021`, val `2022` | contradicted by data and code (see F2) |
| Test positive rate | 0.2344 | 0.2871 in the shipped `test` split |

Any consumer who builds a feature matrix from `meta.json` will hand XGBoost 20 columns and either crash or — worse — silently misalign. The `note` field ("Operational HRRR model retrained without flare cells + phantom dates") is itself evidence that the team knew about the phantom-date problem and shipped the uncorrected table anyway.

### F2 — Four incompatible split definitions *(Critical, confirmed)*

| Source | Train | Validation | Test |
|---|---|---|---|
| `docs/METHODOLOGY.md` | 2014–2021 | 2022 | 2023–2026 |
| `meta.json` | 2018-07–2021 | 2022 | 2023–2026 clipped |
| `split` column in the shipped parquet | 2014–2020 | **2021** | **2022**–2026 |
| `step34_mstav_retrain.py` (built the served model) | `year <= 2021` | **none** | `year >= 2023` |
| `docs/HANDOFF.md` | ≤2020 | 2021 | 2022+ |

The code is authoritative, and it reveals two problems beyond documentation drift. First, **there is no validation set**: `tr = df[df.year <= 2021]`, `te = df[df.year >= 2023]` — the year 2022 is silently dropped, and the `split` column shipped in the parquet is never read. Second, **model selection was performed on the test set**. The script's own docstring states *"Gate 1: test AUC-PR vs step32's 0.4925"*, and steps 30/32/34 each promote on the same 2023–2026 fold. The reported test AUC-PR of 0.4941 is therefore a selection-biased estimate, not the held-out estimate `METHODOLOGY.md` claims ("scored once; all reported numbers come from here").

### F3 — Missing config file causes a silent, unflagged change in dashboard output *(Critical, confirmed)*

`scripts/fwi_config.py` loads HWP normalisation constants:

```python
def _load_hwp_params():
    p = _TF / "data" / "hwp_params.json"
    if p.exists():
        return json.load(open(p))
    return {'tx': {**HWP_NOAA, 'ref': 60.0}, 'noaa': {'ref': 60.0}}   # silent fallback
```

`data/hwp_params.json` **exists in `Supporting_files` and is absent from `Focused_Files`** — the package designated as operational. The fallback is silent: no warning, no log line, no exception. Two consequences are directly visible in the shipped forecast JSONs for the *same target dates*:

| Field, target 2026-08-18 | `Focused_Files` | `Supporting_files` |
|---|---|---|
| `fwiN` mean (NOAA HWP) | 0.4835 | 0.7252 |
| `fwiT` mean (TX-fit HWP) | **0.4835** — identical to `fwiN` | 0.8753 |

The TX-fit variant collapses onto NOAA because the fallback assigns NOAA coefficients to `tx`, and both are divided by a placeholder reference of 60.0 instead of the fitted 26.25 / 20.10. `dashboard/README_v3.md` tells the reader the index is normalised by "its own 99.5th-percentile value over the full 2024–2026 Texas archive (stored in `data/hwp_params.json`)" — in the shipped package it is not. The dashboard's Fire Weather tab is therefore on an undocumented scale, and severity bands read from it are wrong.

### F4 — The retrain path does not resolve inside `Focused_Files` *(High, confirmed)*

`README.md` states retraining is "verified working from this folder alone". Three of the four inputs in `step34_mstav_retrain.py` do not exist at the paths given:

| Code | Resolves to | Present? |
|---|---|---|
| `pd.read_parquet(TF / "tdis_train_daily_hrrr.parquet")` | package root | **No** — the file is at `data/` |
| `pd.read_parquet(HERE / "powerline_dist_km.parquet")` | `New_Training817_moredata/` | **No** — it is at `data/static_features/` |
| `pd.read_parquet(HERE / "flare_cells_v2.parquet")` | `New_Training817_moredata/` | **No** — it is at `data/labels_fused/` |
| `MST = TF / "data/weather_hrrr_forecast/hrrr_24h_mstav"` | — | **No** (masked by the cached `mstav_feature.parquet`) |

`New_Training817_moredata/` in `Focused_Files` contains exactly three files. The script also hard-codes `device='cuda'`, which will fail on any CPU-only machine. `REPRODUCE.md` (shipped identically in both packages) additionally lists ~15 scripts and two data files under "works from this package alone" that are present only in `Supporting_files`.

### F5 — Headline metrics disagree across four artefacts *(High, confirmed)*

For what is described as one model:

| Source | Balanced AUC-PR | Real-population AUC-PR | Lift | Real-pop best F1 |
|---|---|---|---|---|
| `README_v3.md`, `METHODOLOGY.md` | 0.4941 | 0.0878 | 4.58× | 0.169 |
| `step34_results.json` / `step34_gates.json` | 0.4941 | 0.0878 | 4.58× | 0.1692 |
| `docs/CLASSIFICATION_METRICS.md` §3b, §7a | 0.483 | 0.085 | 4.43× | 0.165 |
| `models/operational_realpop_validation.json` | — | 0.08506 | 4.435× | — |
| `meta.json` | 0.4825 | — | — | — |

The lower set traces to the pre-promotion model (`*_pre_d9`, `*_pre_plms` backups confirm the lineage). `CLASSIFICATION_METRICS.md` — the document `METHODOLOGY.md` points to for "full metrics" — therefore documents a superseded model while presenting itself as current. `METHODOLOGY.md` also states the model has "~400 frozen decision trees" while `num_trees` in the model file is **1000**.

### F6 — The robustness JSON contradicts the document that cites it *(High, confirmed)*

`models/flare_robustness_band.json` ends with the verdict string:

> `"lift band 3.54-4.69x, AUROC band 0.7657-0.7897 across a 10x change in the flare cutoff -- conclusions MAY hinge on the 3% choice"`

`docs/CLASSIFICATION_METRICS.md` §7c cites this exact artefact and reverses its conclusion: *"Conclusions do not hinge on the 3% choice."* The underlying numbers favour the JSON: AUC-PR moves 0.0584 → 0.1015 across the band, a **74% swing** in the headline metric driven purely by a preprocessing threshold.

Relatedly, `viirs_typeflag_audit.json` shows the flare exclusion list was only partially validated: of 406 listed cells with ≥5 detections, **267 were confirmed static (65.8%)**. Roughly a third of the 531-cell exclusion is unverified, meaning genuine fire days may have been deleted from both training and evaluation.

### F7 — The served system cannot express a probability above 18.7% *(Medium, confirmed)*

Evaluating `operational_isotonic_calibrator.joblib` across its input range shows it saturates: an input of 1.0 maps to **0.1871**. No cell, on any day, under any weather, can be assigned a calibrated probability above ~19%. This is visible in `realpop_f1_sweep.json`, where thresholds 0.2 and 0.3 return `precision = recall = f1 = 0, flagged = 0`. `CLASSIFICATION_METRICS.md` §3a-bis reproduces the sweep but omits those two rows, so the ceiling is not disclosed anywhere in the documentation.

### F8 — Roughly half the "Texas" grid is not Texas *(High, confirmed)*

`tx_static_master.parquet` is a bounding-box grid, not a state-clipped one. Of 1,708,940 cells:

- **921,124 (53.9%)** have `avg_burn_prob`, `whp`, `cfl`, and `flep4` all exactly `0` — the signature of falling outside US raster coverage.
- **9,200** cells exceed 2,667 m elevation, higher than Guadalupe Peak, the highest point in Texas (max in file: 3,917 m).
- Ecoregion labels include **Ouachita Mountains (21,909)**, **Arkansas Valley (22,177)**, **Southern Rockies (15,073)**, and **Arizona/New Mexico Mountains (16,647)** — none of which touch Texas. The largest block, *Chihuahuan Deserts* (399,606 cells), extends well into Mexico.
- A sanity check confirms the scale: Texas is ~695,662 km², and an H3 res-8 cell averages 0.737 km², so tiling Texas needs ~944,000 cells. The grid has **1.81× that number**.

The model is scored on all 1.7M cells daily and the results are averaged up to res-5 for display, so out-of-domain cells with zeroed hazard inputs dilute genuine Texas cells in the published aggregates.

### F9 — `powerline_dist_km` is unusable over large regions *(High, confirmed)*

This is one of the 22 model features. Its distribution is not physically plausible for a covered service territory:

| Statistic | Value |
|---|---|
| Median | 12.5 km |
| 90th percentile | 234.1 km |
| Maximum | 561.2 km |
| Cells > 100 km | 372,188 (21.8%) |
| Cells > 200 km | 213,367 (12.5%) |

Broken out by latitude band, the median is **201.0 km** in the southernmost band (25.8–27.6 °N) — a densely electrified part of the Rio Grande Valley if it were Texas. Cross-referencing F8, the 213,367 cells beyond 200 km sit entirely in Chihuahuan Deserts / Western Gulf Coastal Plain / Southern Texas Plains, have `avg_burn_prob = whp = 0`, and a minimum `road_dist_km` of 78.7 km. The distance layer is measuring "distance to the nearest US transmission line" from points outside US coverage, which is not the quantity the feature name implies.

### F10 — Label provenance shifts exactly at the train/test boundary *(Medium, confirmed)*

`ignitions_daily_tx.parquet` records `cause` and `time_source`. Share of labels that are satellite-only, by year:

| 2014 | 2016 | 2018 | 2020 | **2021** | 2022–2026 |
|---|---|---|---|---|---|
| 88.3% | 89.6% | 93.5% | 95.1% | **100%** | **100%** |

FPA-FOD contributions stop after 2020 — precisely where the training fold ends. The model learns a label definition that is ~91% VIIRS + 9% ground-reported, and is then evaluated against one that is 100% VIIRS. Only **1,169 of 960,054 rows (0.12%)** have `n_sources = 2`, so `METHODOLOGY.md`'s framing of the labels as "VIIRS satellite detections fused with the FPA-FOD fire-occurrence database" materially overstates the fusion. `max_size_acres` is null for **923,895 rows (96.2%)**.

### F11 — The dashboard has no freshness check *(Medium, confirmed)*

`22_embed_v3.py` selects forecasts by filename sort order only:

```python
fps = sorted(D.glob("forecast_*.json"), key=lambda p: p.name)[-2:]
```

There is no comparison against the current date. The script asserts that its HTML placeholders exist but never asserts that the forecast is for a future day. The shipped standalone dashboard therefore embeds targets **2026-08-18 (24 h)** and **2026-08-19 (48 h)** — both in the past as of 20 August 2026 — while `README.md` describes the same files as "current 24h/48h forecast" and `README_v3.md` says the tabs show "only what's live and forward-looking".

---

## Missing Values & Data-Quality Issues

Recomputed on `Focused_Files/data/tdis_train_daily_hrrr.parquet` (3,595,513 rows):

| Column group | Null rows | Null % | Assessment |
|---|---|---|---|
| `erc, fm100, vpd, vs, rmax, rmin, tmmx, pr` (gridMET) | 2,773,317 | **77.13%** | Not model features; but shipping eight three-quarters-empty columns in the primary training table invites misuse |
| `hrrr_tmp, hrrr_vpd, hrrr_wind` | 1,389,787 | **38.65%** | **The model's only weather inputs** — see below |
| 11 static columns (`elevation_m`, `whp`, …) | 202,078 | **5.62%** | 18,520 distinct cells present in the training table are absent from `tx_static_master.parquet` |
| `h3_cell`, `date`, `label`, calendar, `split` | 0 | 0% | Clean |

**The weather gap is not uniform — it is concentrated in the training years.** HRRR null rate by year:

| 2014 | 2015 | 2016 | 2017 | 2018 | 2019–2025 | 2026 |
|---|---|---|---|---|---|---|
| 100% | 100% | 100% | 100% | 58.4% | 5.5–5.8% | 36.7% |

By split: **train 64.26% null**, val 5.75%, test 11.24%. Four of the seven training years contain **no weather data whatsoever**. Because `step34_mstav_retrain.py` calls `dropna(subset=['hrrr_vpd'])`, those rows are dropped rather than learned from — so the "2014–2026" training claim in `METHODOLOGY.md` reduces in practice to roughly 2019–2021 for the weather-bearing model. Separately, rows with weather present have a materially different label rate (0.3027) than rows without (0.2104), so the drop is not missing-at-random and biases the retained sample.

**Other quality observations**

- **Duplicates: none.** `(h3_cell, date)` is unique in the training table (0 duplicates), in `ignitions_daily_tx` (0), and `h3_cell` is unique in both static files (0). This is a genuine pass.
- **Cross-file reconciliation: passes on labels.** Positive counts per year in the training table match row counts per year in `ignitions_daily_tx.parquet` exactly for all 13 years (2014: 45,174 / 45,174 … 2026: 55,981 / 55,981).
- **Cross-file reconciliation: fails on cells.** 18,520 training cells are missing from the 1,708,940-cell static master, and the model is served on the static master's cell set, so those cells exist in training but can never be scored.
- **Cross-package divergence.** `Supporting_files/data/tdis_train_daily_tx_flarefiltered.parquet` (3,243,435 rows) ends cleanly at 2026-07-29 with **zero** future-dated rows. The older, superseded package ships the *corrected* table; the current package ships the uncorrected one.
- **`is_holiday` is a six-date fixed set** — `{1/1, 7/4, 6/19, 11/11, 12/25, 10/31}` — consistently applied in both training (`03_build_dataset.py:132`) and serving (`13_model_forecast_day.py:100`), so there is no train/serve skew, but it omits Memorial Day, Labor Day, and Thanksgiving, three of the highest outdoor-activity ignition periods in the year. The code comments it as "(v1)".
- **Calibrate-then-average.** `13_model_forecast_day.py` applies the isotonic map at res-8 and then averages `ign_cal` to res-5. Isotonic regression is non-linear, so `mean(f(x)) ≠ f(mean(x))`; the published res-5 probabilities are not the calibrated probabilities of the res-5 aggregate. The script's own comment concedes a related issue: *"Calibrator was fit on res-5 scores; applying to res-8 scores is an approximation."*

---

## Date/Timeline Issue

**This is the most serious defect in the packages, and the answer to the brief's question is unambiguous.**

`Focused_Files/data/tdis_train_daily_hrrr.parquet` spans **2014-01-01 to 2026-12-31**. Relative to today, 20 August 2026, that is 133 days into the future. Relative to the last real observation it is worse: the label inventory `ignitions_daily_tx.parquet` ends at **2026-07-29**, and `meta.json` records `"last_label": "2026-07-29"`. Everything after that date is unobservable.

**These rows are not forecasts and not planned data.** The evidence is decisive:

| Property of the 86,072 rows dated after 2026-07-29 | Value |
|---|---|
| Distinct dates | 155 (2026-07-30 → 2026-12-31) |
| Positive labels (`label = 1`) | **0** |
| Rows with any HRRR weather (`hrrr_tmp` non-null) | 1,042 of 86,072 (1.2%) |
| Rows dated after **today** (2026-08-21 →) | **73,873** across 133 dates |
| — of those, positives | **0** |
| — of those, with weather | **0** |
| `split` assignment | **`test` — 100% of them** |

A genuine forecast record would carry predicted weather and either no label or a pending marker. These rows carry neither weather nor prediction: they are matched-negative sampling rows generated across a date range that ran past the end of the observation window, then stamped `label = 0` by default. **A hard-coded zero on a date where no fire could yet have been observed is a fabricated observation, not a forecast.**

**Impact.** The rows sit inside the evaluation fold, where they constitute **6.05% of the test split (86,072 of 1,422,393)** and **8.49% of all test negatives**. Every test-set metric computed from the `split` column is therefore inflated: these rows are trivially easy to classify (features null or absent), and they add ~86k guaranteed true negatives to the denominator. Any reported AUC-PR, F1, or precision derived this way is optimistic by an unquantified margin. Because the calibrator is fitted against real outcomes, a phantom-contaminated fit would also depress the mapped probabilities.

**Mitigating fact, and why it does not close the issue.** The build code for the served model does guard against this — `step34_mstav_retrain.py` sets `LAST_LABEL = pd.Timestamp('2026-07-29')` and applies `df = df[df.date <= LAST_LABEL]` — and the `meta.json` note explicitly says the model was retrained "without flare cells + **phantom dates**". So the team identified this defect and fixed it *in one script*. What was never fixed is the **shipped artefact**: the corrupted table is still the primary data file in the operational package, its `split` column still marks the phantom rows as `test`, and nothing in `README.md`, `METHODOLOGY.md`, or `REPRODUCE.md` warns a downstream user. Anyone who does the natural thing — read the parquet, filter on `split == 'test'`, evaluate — reproduces the contaminated result. The `Supporting_files` copy of the table has already been cleaned, which makes the operational package strictly the worse of the two on this axis.

**Second, independent date problem.** The delivered dashboard is stale on arrival. The two embedded forecasts target **2026-08-18** and **2026-08-19**, initialised **2026-08-17**; both dates are in the past today. Because the build script sorts by filename and never compares to the current date, a dashboard rebuilt without a fresh `13_model_forecast_day.py` run will silently re-publish the same past-dated forecast indefinitely, labelled "live".

**Assumption stated explicitly:** I treat 20 August 2026 as the true current date, per the brief. If the packages were in fact assembled at a later date, the staleness finding changes in magnitude but not in kind, because the missing freshness assertion is a code-level defect independent of when it is run.

---

## Process/Control Gaps

1. **No schema or freshness contract at any boundary.** Nothing validates that a written parquet has the expected columns, row count, date range, or null profile. F1, F2, and the phantom dates would all have been caught by a ten-line post-write assertion.
2. **Metadata is hand-maintained rather than emitted.** `meta.json` is written separately from the model rather than derived from `booster.feature_names` and the actual fit configuration, which is why it drifted (F1).
3. **No single source of truth for the split.** The split is variously a document sentence, a `meta.json` string, a materialised column, and an inline code filter — with no test that they agree (F2).
4. **Model selection and final evaluation share a fold.** Steps 30/32/34 each promote on the 2023–2026 test set; no untouched fold survives.
5. **Silent fallbacks in configuration loading.** `fwi_config.py` degrades without a warning (F3). A missing file that changes published numbers must raise, not default.
6. **Documentation is not generated from artefacts.** Every metric in the four headline documents is transcribed by hand, so they diverge from the JSONs they cite — including a case where the document reverses its source's verdict (F6).
7. **No packaging test.** No CI step imports the package fresh and runs the documented commands, so `README.md`'s "verified working from this folder alone" was never mechanically checked (F4).
8. **No domain (geographic) validation.** No check confirms the grid is inside Texas or that feature ranges are physically plausible (F8, F9).
9. **No drift monitoring on labels.** The FPA-FOD cut-off after 2020 is not detected or annotated anywhere (F10).
10. **No actual-vs-forecast reconciliation.** Forecast JSONs are published; nothing later joins them to observed outcomes to score them. Backtesting exists (`26_score_operational_realpop.py`) but is a one-off research script, not a recurring control.

---

## Corrective & Preventive Actions

### Immediate — before any further use (target: 1 week)

| # | Action | Fixes |
|---|---|---|
| C1 | Truncate the shipped training table at `2026-07-29` and re-emit it; or, if future rows are wanted as a scoring skeleton, move them to a separate file with `label = NULL` and `split = 'unscored'`. Re-derive the `split` column so it never contains unobservable dates. | Date issue |
| C2 | Regenerate `meta.json` **from the model object** — `booster.feature_names`, `num_feature`, `num_trees` — plus the exact split filter, flare-list hash, training row count, evaluation population, and base rate. Never hand-edit it again. | F1 |
| C3 | Copy `data/hwp_params.json` into `Focused_Files/data/`, and change `_load_hwp_params()` to `raise FileNotFoundError` instead of returning placeholder constants. Regenerate the two forecast JSONs and the standalone dashboard. | F3 |
| C4 | Fix the four input paths and replace `device='cuda'` with `device=os.environ.get('XGB_DEVICE','cpu')` in `step34_mstav_retrain.py`; run it end-to-end from a clean copy of `Focused_Files` and confirm it reproduces the served model. | F4 |
| C5 | Regenerate today's forecast and add to `22_embed_v3.py`: `assert max(f['target'] for f in fcs) >= str(date.today())`, and render the target date and init time in the dashboard header. | F11 |

### Short term (target: 1 month)

| # | Action | Fixes |
|---|---|---|
| C6 | Publish one metrics file per model, generated by the scoring script, and have every document include from it rather than transcribing. Immediately reconcile `CLASSIFICATION_METRICS.md` to the promoted model or mark it superseded in its header. Correct "~400 trees" → 1000. | F5 |
| C7 | Carve out a genuine validation fold (2021 or 2022) used for all gate decisions, and re-score the promoted model once on an untouched test fold. Report both numbers and label which is which. | F2 |
| C8 | Clip the cell grid to the Texas state boundary — or add an `in_texas` flag and exclude non-Texas cells from training, scoring, and all res-5 aggregation. Re-derive `powerline_dist_km` over a transmission dataset covering the retained extent, or drop the feature until it does (its reported importance is 0.049, so the cost of dropping is small). | F8, F9 |
| C9 | Amend `CLASSIFICATION_METRICS.md` §7c to state the source JSON's actual verdict, publish the full 0.0584–0.1015 AUC-PR band alongside every headline number, and complete the type-flag verification for the ~34% of flare cells still unconfirmed. | F6 |
| C10 | Document the 0.1871 calibrated ceiling in `README_v3.md` and the dashboard hover text, and publish the full threshold sweep including the zero rows. Refit the calibrator at the resolution it is applied at, or aggregate raw scores to res-5 and calibrate once, after aggregation. | F7 |
| C11 | Add a `label_regime` column to the label inventory, and either restrict training to the satellite-only regime for consistency with evaluation, or extend FPA-FOD coverage through 2026 if obtainable. Re-word the "fused VIIRS + FPA-FOD" claim to reflect 0.12% two-source corroboration. | F10 |

### Preventive (structural)

- **P1 — Fail-fast configuration.** No `if exists() … else default` for any file whose absence changes a published number.
- **P2 — Emit, never transcribe.** Metadata, metrics, and documentation tables generated from artefacts by script.
- **P3 — Gate on the pipeline, not the person.** The validation suite below runs in CI on every data or model change; a red gate blocks publication.
- **P4 — Package smoke test.** A CI job that copies the deliverable to a clean directory, creates the environment, and executes every command in `README.md` and `REPRODUCE.md`.
- **P5 — Single split registry.** One `split_config.json` read by every script; the materialised `split` column asserted equal to it at build time.
- **P6 — Immutable observation horizon.** A `LAST_OBSERVED_LABEL` constant loaded from the label inventory's true max date and enforced by every build script, so the horizon cannot be exceeded by construction.

---

## Recommended Validation Process

A single `validate.py`, run automatically after every data build, model train, and dashboard publish. Each check emits PASS / WARN / FAIL; **any FAIL blocks publication.** Results append to `validation_history.jsonl` so drift is visible over time.

**Gate 1 — Schema and completeness**
- Column names, order, and dtypes match a checked-in schema file exactly.
- Row count within ±5% of the previous build (FAIL outside; WARN at ±2%).
- Per-column null rate compared against a declared budget. Any feature consumed by the model exceeding its budget is a FAIL — under today's data, `hrrr_tmp` at 38.65% overall and 64.26% within the training fold would fail immediately.
- Nulls in a key column (`h3_cell`, `date`, `label`) are always a FAIL.

**Gate 2 — Uniqueness and duplicates**
- `(h3_cell, date)` unique in every daily table; `h3_cell` unique in every static table.
- Full-row duplicate count = 0.
- Report duplicate keys with differing feature values separately — these indicate a bad join, not a bad append.

**Gate 3 — Dates and horizon**
- `max(date) <= LAST_OBSERVED_LABEL`, read from the label inventory. **FAIL** otherwise. This single rule catches the phantom-date defect.
- No date in the future relative to run time in any table intended as historical.
- Calendar completeness: every date between min and max present; flag gaps and verify leap-year day counts.
- Forecast artefacts inverted: `target > today` is required, and `target - init == lead_h` must hold.
- Dashboard publish gate: newest embedded `target >= today`, else **FAIL**.

**Gate 4 — Value and domain consistency**
- Physical bounds per column: `slope_deg ∈ [0,90]`, `aspect_deg ∈ [0,360)`, `label ∈ {0,1}`, probabilities `∈ [0,1]`, distances `>= 0`.
- Plausibility bounds: `elevation_m <= 2700` (Texas high point 2,667 m), `powerline_dist_km` 99th percentile `<= 50`, `road_dist_km` 99th percentile `<= 50`. Today's data fails all three.
- Geographic domain: every cell centroid inside the Texas state polygon, or explicitly flagged `in_texas = False` and excluded downstream.
- Distribution drift: per-column mean and standard deviation versus the previous build; WARN beyond 2σ, FAIL beyond 4σ.

**Gate 5 — Cross-file reconciliation**
- Positive-label count per year in the training table **equals** the row count per year in the label inventory. *(Currently passes — keep as a regression test.)*
- Every `h3_cell` in the training table exists in the static master. *(Currently fails on 18,520 cells.)*
- Static feature files share an identical cell set. *(Currently passes.)*
- Model `feature_names` **equals** `meta.json` features **equals** the feature list in the training script. *(Currently fails — F1.)*
- Split definition in code, in `meta.json`, in the materialised column, and in the docs all resolve to the same year ranges. *(Currently fails — F2.)*
- Every file referenced by any config loader or README command exists at the stated path. *(Currently fails — F3, F4.)*
- Checksums of files duplicated across packages match, or the difference is declared.

**Gate 6 — Actual-vs-forecast validation** *(the recurring control that does not exist today)*
- Every published `forecast_<date>.json` is archived unmodified at publish time.
- A weekly job joins forecasts aged ≥ 7 days to observed VIIRS outcomes for the same cell-days and computes AUC-PR, AUROC, lift, Brier, and ECE **on the real population**, appending to a scorecard.
- Alert if rolling 30-day lift drops below 3.5× (the low end of the project's own robustness band) or 30-day ECE exceeds 0.05.
- Reliability check: mean calibrated probability over the window versus observed frequency; a gap beyond 2× triggers a calibrator refit.
- Explicit forecast-vs-actual separation in every published table: a `record_type` column with values `actual` / `forecast`, never inferred from whether a label happens to be zero.

**Gate 7 — Model and metric integrity**
- No promotion decision reads the test fold; gate metrics come from validation only, asserted by fold-hash comparison.
- Model and calibrator versions pinned together; loading a mismatched pair raises.
- Every metric quoted in a document traces to a generated JSON, verified by a doc-lint step that fails on an unsourced number.

---

## Conclusion

The underlying method is defensible. Training on archived HRRR *forecasts* rather than observations is the right call for a forecasting system, the temporal split is the right shape, the isotonic calibration is real and well-validated at 2026-holdout ECE 0.0018, and the team's own documents are unusually candid about the model being a prioritisation tool rather than an alarm — a framing the metrics support.

What has failed is not the modelling but the **control layer around it**. Every one of the eleven findings is a bookkeeping failure rather than a scientific one: metadata that drifted from the model it describes, a config file left out of the package with a silent fallback to mask the omission, four written definitions of one split, documentation that reverses the verdict of the JSON it cites, a grid that was never clipped to the state it is named after, and a training table whose date range ran 155 days past the last observable fact and was never truncated before shipping.

The date issue is the clearest illustration. It was found by the team, correctly diagnosed, given a name ("phantom dates"), fixed in the training code, and recorded in the model's own metadata note — and the corrupted file was shipped anyway, in the package designated as operational, while the superseded package carries the clean copy. That is the signature of a pipeline with good instincts and no gates.

The corrective actions C1–C5 are days of work, not weeks, and none require re-collecting data. The larger value is in the preventive layer: a validation script with the seven gates above, run in CI, would have caught nine of the eleven findings automatically at the moment they were introduced — including the phantom dates, on a single line asserting that no historical table may contain a date beyond the last observed label.

**Confirmed vs. assumed.** All findings above are confirmed by direct measurement of the shipped files, with the following exceptions, stated as inference: (a) the *cause* of the F3 divergence is inferred from the code path and the exact `fwiN == fwiT` equality observed in the Focused forecasts, not from a build log; (b) the identification of out-of-Texas cells in F8 rests on ecoregion names, elevation exceeding the Texas high point, and cell-count arithmetic rather than a state-boundary intersection, which was not performed; (c) the *magnitude* of test-metric inflation from the phantom rows is not quantified here, only its mechanism and its 6.05% share of the test fold; (d) the assessment that `powerline_dist_km` reflects incomplete coverage rather than a genuine measurement is an inference from its joint distribution with the zeroed US hazard layers.
