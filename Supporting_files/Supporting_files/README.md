# TDIS Forecast — Deliverable Package

A clean, self-contained copy of the current, correct TDIS Forecast
artifacts: dashboard, models, data, pipeline scripts, and reference docs.
Assembled 2026-08-11/12 from the live project at
`/net/cven-mosta-nas.engr.tamu.edu/volume2/NASdata2/miguel_shared/alphaearth_nds/TDIS_Forecast/`
— that path remains the source of truth for raw intermediate data (the 14 GB
HRRR archive), logs, and day-to-day iteration; this package is a curated
snapshot of the finished, correct outputs plus everything needed to
understand or reproduce them.

## 0. THE MODEL — read this first

If you take one thing from this package: **the ignition forecast model is
`models/tdis_forecast_hrrr_filtered.json`** (XGBoost, with live HRRR
weather-forecast features), paired with
`models/operational_isotonic_calibrator.joblib`. This is the model behind
the dashboard's Ignition tab, the one validated on real deployment data,
and the one to build on. All other model files in `models/` are baselines,
diagnostics, or superseded variants — see `docs/DEPLOYMENT_STRATEGY.md` §1.

**Promoted 2026-08-17 (second promotion of the day)**: the served pair is
the depth-9 config with TWO new features and cleaned labels —
`powerline_dist_km` (distance to nearest HIFLD transmission line, static)
and `hrrr_mstav` (HRRR soil-moisture availability, forecast weather) —
trained against the type-flag-audited flare list (v2, 538 cells excluded).
22 features total. Each addition passed all three evidence gates (test set,
3-seed robustness, real full population) before stacking. Cumulative
tinker-phase gain: real-population lift 4.43→4.58×, AUROC 0.784→0.797.
Pre-promotion backups: `models/*_pre_d9*.bak` (original) and
`models/*_pre_plms*.bak` (depth-9-only); full experiment provenance in
`New_Training817_moredata/`.

**Classification results (AUC-PR and F1):**

| Evaluation population | AUC-PR | Best F1 | Notes |
|---|---|---|---|
| **Test set** (balanced sample, ~20% fire-days) | **0.4941** | **0.496** (P=.441, R=.566 @ raw threshold 0.5) | The number to compare against other teams' models — they report on balanced samples too |
| **Real full population** (every TX cell × every day 2024–26, 5.06M cell-days, 1.9% fire-days) | **0.0878** | **0.169** (P=.137, R=.222 @ calibrated threshold 0.077) | The honest deployment number — same model, real base rate. Lift = **4.58×** over random. AUROC 0.797. |

Both rows are the **same model** — the numbers differ only because fires
are ~10× rarer in reality than in the balanced test sample, and AUC-PR/F1
both shrink mechanically as the positive class gets rarer. Reporting only
the first row would overstate real-world performance; reporting only the
second would understate it relative to how every other team reports.
This package reports both, always labeled. Full derivation, threshold
sweeps, and every other metric: `docs/CLASSIFICATION_METRICS.md`
(real-population F1 sweep: `models/realpop_f1_sweep.json`).

**Deployment — no retraining needed per forecast.** The model is already
trained; producing tomorrow's forecast is *inference only*: pull the new
HRRR weather forecast, run `scripts/13_model_forecast_day.py <date> <24|48>`
(seconds of model runtime), rebuild the dashboard with
`scripts/22_embed_v3.py`. Retraining is an annual maintenance event (add
the new label-year, refit calibrator), not part of the daily cycle.

## 1. Project context — what TDIS Forecast is

**Goal:** a Texas statewide wildfire ignition/risk forecasting system —
given a location and a day (historical or forecast), estimate how likely a
fire is there, and separately, how hazardous the fire-weather conditions
are. Built on 13 years (2014–2026) of real satellite fire detections
(NASA VIIRS) fused with the FPA-FOD federal fire-occurrence database,
joined against static terrain/fuels/access features (TxWRAP, LANDFIRE, DEM,
roads, ecoregion — 1.71M cells covering all of Texas) and both historical
(gridMET) and forecast (NOAA HRRR/GFS) weather.

**Two distinct products, easy to conflate — kept separate throughout:**
- **Wildfire Risk** = static hazard(cell) × fire-weather-index(day). No ML.
  A physically-motivated hazard/weather product, not a learned model.
- **Ignition** = a trained XGBoost-GPU model's per-cell fire-occurrence
  probability. Two different presentations of this exist in the dashboard
  (see §4) — one is the model's live/forecast output, the other is a
  heuristic proxy built from the model's aggregated past predictions. This
  package's docs are explicit about which is which; don't assume "Ignition"
  always means "the model just ran."

**Honest, verified skill level (test = 2023–2026, real future years, never
seen in training):** ignition model AUC-PR ≈ 0.45 / AUROC ≈ 0.71 / lift ≈
1.9× over random (no-weather "honest" baseline); ≈ 0.48 / 0.73 / 2.1× with
HRRR forecast weather added (the "operational" model). Location (elevation,
ecoregion, burn probability, roads) dominates; weather is a real but modest
daily modulator. At the standard 0.5 threshold it still misses ~40% of real
fire-days — this is a credible baseline, not a high-accuracy system, and
the docs never claim otherwise.

**Verification performed (see §5 below for exactly which doc covers what):**
temporal (not random) train/val/test split; a label-shuffle negative
control (confirms no data leakage — shuffled labels collapse to base-rate
skill); a spatial holdout test (confirms skill is transferable geography,
not cell memorization — gap of 0.005 AUC-PR); a controlled ablation
isolating HRRR forecast weather's true contribution; a full threshold/
precision-recall sweep; a 3-way fire-weather-formula comparison (Composite
vs. NOAA HWP vs. a TX-fit HWP); and a 5-event real-wildfire validation
battery against independently documented Texas fires (Smokehouse Creek,
Crabapple, Lavender, Hunggate, Windy Deuce) checking whether the system
would have flagged each one's location and timing. Result: location
flagged correctly 5/5; timing flagged correctly for heat/drought-driven
fires (2/2) but originally under-scored for wind-driven cool-season fires
(a real, diagnosed, partially-fixed limitation — the historical weather
source only has daily-mean wind, missing the gusts that drive those fires;
live forecasts use HRRR gust data and don't inherit this as badly).

**Known, documented limitations** (not fixed, but characterized): a 2018
step-change in fire detection rate from a second VIIRS satellite coming
online (label instrument-era effect, not a real fire-activity trend); the
3%-of-days flare/hotspot filter threshold is somewhat load-bearing (moves
~20% of positive labels if changed 3×); ~6% of raw labels are technically
outside Texas (bounding-box spillover into NM/OK) but are provably dropped
before training, never seen by any model; a few minor data QC items (a
temperature fill-value artifact, some partial weather nulls). None of these
invalidate the results; all are disclosed in `docs/SANITY_CHECK.md`.

## 2. Folder structure

```
deliverablePackage/
├── README.md                                     this file
├── REPRODUCE.md                                   tiered reproducibility guide — read before handing off to a team
├── environment.yml                                pinned conda environment (Python 3.10.18)
├── docs/                                          reference docs, copied verbatim from source
│   ├── README.md                                  full project narrative, results, file-path index
│   ├── HANDBOOK.md                                 deep technical reference (§-numbered: data sources, formulas)
│   ├── PRESENTATION.md                              stakeholder-facing summary
│   ├── VALIDATION.md                                the 5-event real-fire validation battery + credibility retrains
│   ├── SANITY_CHECK.md                              full adversarial sanity sweep, 8 findings ranked by severity
│   ├── DEMO_SMOKEHOUSE.md                           walkthrough demo (Smokehouse Creek fire case study)
│   ├── PLAN.md                                      dataset & model design decisions
│   └── HANDOFF.md                                   original unattended-build-run notes
├── scripts/                                        full pipeline, run in numeric order (§3 below)
├── dashboard/                                       open directly in a browser, no server needed
│   ├── tdis_fire_dashboard_standalone.html           MAIN — drag-and-drop, all data embedded (~62 MB)
│   ├── tdis_fire_explorer_standalone.html            v2 — zoom-adaptive grid, all data embedded (~102 MB)
│   ├── tdis_fire_dashboard.html + tdis_dashboard_data.json        served version of v1
│   ├── tdis_fire_dashboard_v2.html + explorer_static_multires.json   served version of v2
│   └── forecast_2026-08-06.json, forecast_2026-08-07.json          example live-forecast outputs
├── models/                                          current/served model artifacts only (flare-filtered — see §6)
│   ├── tdis_forecast_hrrr_filtered.json (+_meta)      OPERATIONAL model (static + temporal + HRRR forecast weather)
│   ├── tdis_forecast_baseline_honest_filtered.json (+_meta)   no-forecast-weather baseline ("honest")
│   ├── tdis_forecast_baseline_ceiling_filtered.json (+_meta)  same-day observed-weather upper bound ("ceiling")
│   └── baseline_forecast_predictions_test_filtered.parquet     test-set predictions (892,372 rows)
└── data/                                            current/served data artifacts only
    ├── tdis_train_daily_tx_flarefiltered.parquet      training table, flare-cleaned (3,243,435 rows × 30 cols)
    ├── labels_fused/ignitions_daily_tx.parquet          fused FPA-FOD + VIIRS ignitions (960,054 rows)
    ├── labels_viirs/viirs_tx_h3.parquet                  raw VIIRS detections, H3-gridded (1,619,215 rows)
    ├── static_features/tx_static_master.parquet          full-TX static features, 0 nulls (1,708,940 cells × 15 cols)
    ├── fwi_components_res5.parquet                       daily fire-weather-index components (3,381,192 rows)
    └── hwp_params.json                                   fitted TX Hot-Dry-Windy coefficients
```

## 3. Scripts — full pipeline, in run order

Run with `PY=/home/mte1224/mambaforge/envs/UAI2526/bin/python`, from the
source project directory (paths inside the scripts are absolute to the
source project, not this package — see §7 for why).

| # | Script | What it does |
|---|---|---|
| 01 | `01_download_viirs.py` | Download VIIRS active-fire detections for Texas (archive + NRT), H3-assign |
| 02 | `02_build_labels.py` | Fuse FPA-FOD + VIIRS → daily fire-occurrence labels |
| 03 | `03_build_dataset.py` | Assemble the training table: positives + matched negatives + weather + temporal features |
| 04 | `04_build_full_tx_static.py` | Build the full-Texas static feature master (terrain/fuels/access, 1.71M cells) |
| 05 | `05_download_hrrr_forecast.py` | Download historical HRRR forecast-weather archive (build input, ~14 GB, not in this package) |
| 06 | `06_attach_hrrr_forecast.py` | Attach HRRR forecast weather to training rows |
| 07 | `07_train_baseline_forecast.py` | Train the honest (no-weather) and ceiling (perfect-weather) baseline models |
| 08 | `08_build_dashboard_data.py` | Build the hazard + ignition-susceptibility layers for the dashboard |
| 09 | `09_build_perday_dynamic.py` | Build the daily fire-weather-index arrays for the historical scrubber |
| 10 | `10_live_forecast_day.py` | Pull a live HRRR forecast (fire-weather only) for one target day |
| 11 | `11_train_hrrr_forecast.py` | Train the operational model with HRRR forecast weather (GPU) |
| 12 | `12_hrrr_ablation.py` | Controlled ablation isolating HRRR forecast weather's true contribution |
| 13 | `13_model_forecast_day.py` | Run the trained operational model on live forecast weather → the real per-cell forecast |
| 14 | `14_event_validation.py` | Real-wildfire validation battery (VIIRS capture / WHERE / WHEN) |
| 15 | `15_reweight_fwi.py` | Instant rebuild of all fire-weather-formula variants from cached components (~8s) |
| 16 | `16_fit_hwp_tx.py` | Fit TX-calibrated Hot-Dry-Windy coefficients to VIIRS fire-intensity data |
| 17 | `17_embed_standalone.py` | Build the drag-and-drop standalone v1 dashboard |
| 18 | `18_flare_filter_retrain.py` | Identify + remove persistent industrial flare cells from labels, retrain the honest model — the fix behind every "_filtered" artifact in this package |
| 19 | `19_retrain_hrrr_filtered.py` | Retrain the OPERATIONAL (HRRR) model on flare-filtered labels; the one actually served |
| 20 | `20_build_explorer_data.py` | Build multi-resolution static layers (H3 res 4-8) for the zoom-adaptive v2 dashboard |
| 21 | `21_embed_explorer.py` | Build the zoom-adaptive standalone v2 dashboard ("explorer") |
| 22 | `22_retrain_ceiling_filtered.py` | Retrain the ceiling (perfect-weather) model on flare-filtered labels, for a fair 3-way comparison |
| — | `fwi_config.py` | The knobs: fire-weather formula weights (Composite) and formula picker (Composite/NOAA HWP/TX HWP) |
| — | `sweep_phase1_forensics.py` | Read-only database forensics — the adversarial sanity sweep, phase 1 |
| — | `sweep_phase6_credibility.py` | The two credibility retrains: label-shuffle leakage check + spatial-holdout memorization check |
| — | `run_all.sh` | Unattended end-to-end pipeline driver (VIIRS download → dataset build) |
| — | `run_hwp_chain.sh` | Chains the HWP fit → FWI rebuild → dashboard embed → live-forecast refresh, tmux-safe |

**Not yet reflected in the source project's own `docs/README.md` §10 script
table** (as of this package's assembly): scripts 19–22, the two sweep
scripts, and both shell runners. All are real, current, and referenced
elsewhere in the docs (`VALIDATION.md`, `SANITY_CHECK.md`) — the table above
is the complete, corrected list.

## 4. Reading the dashboard correctly — modes and what's really underneath

The dashboard (`dashboard/tdis_fire_dashboard_standalone.html` or the v2
`tdis_fire_explorer_standalone.html`) has 4 layer modes and a day
scrubber/live-forecast toggle. **The single most important thing to know:
"Ignition" means two different things depending on whether you're scrubbing
history or looking at the live forecast.**

- **🔥 Wildfire Risk** = static hazard(cell) × fire-weather-index(day). No
  ML anywhere. The FWI term is computed from your formula-picker selection
  (Composite / NOAA HWP / TX HWP) using historical gridMET weather (ERC,
  VPD, wind) for the scrubbed day, or HRRR/GFS forecast weather (VPD + wind
  only — HRRR carries no ERC) when the Live Forecast toggle is on.
- **✦ Ignition, historical scrubber** = **not** a live model run. It's
  `ML susceptibility(cell) × fire-weather-index(day)` — the ML term is the
  honest model's real predictions on the 2023-2026 test set, but collapsed
  to a **monthly mean per cell** (script 08), then multiplied by the same
  daily FWI term used for Wildfire Risk. You can verify this yourself in
  the tooltip: susceptibility × fire-weather ≈ the displayed Ignition value.
  This is a proxy, explicitly labeled as such — it reuses a real model
  output but doesn't re-run the model per day.
- **✦ Ignition, live forecast toggle on** = the actual trained operational
  model (`tdis_forecast_hrrr_filtered.json`), scored directly on real HRRR/
  GFS forecast weather for that specific day. This is the one genuine
  "the model just ran on today's forecast" case.
- **▦ Static Hazard (full TX)** = the WRAP/LANDFIRE hazard composite alone,
  no ML, no weather, all 5,382 res-5 cells (vs. 3.6k for the dynamic layers,
  which are capped to the gridMET/modeled universe).

## 5. Which doc answers which question

- **"What IS the method, finally and definitively?"** → `docs/METHODOLOGY.md`
  — the locked, canonical statement: training data & split, how daily
  inference works step-by-step, both metric rows, and a table of every
  retired approach (static-only, susceptibility×FWI heuristic, output×HWP
  multiplication) with the evidence for why each was dropped. If a
  discussion contradicts this file, this file wins.
- **"Is this any good / has it been checked?"** → `docs/VALIDATION.md`
  (quantitative skill + 5-event real-fire battery + credibility retrains)
  and `docs/SANITY_CHECK.md` (adversarial sweep for hidden bugs/leakage).
- **"What exactly is in the dashboard and how do I read it?"** → §6 of
  `docs/README.md`, or §4 above for the short version.
- **"What's the full data/methodology story?"** → `docs/HANDBOOK.md`
  (§-numbered, most detailed).
- **"Explain this to a non-technical stakeholder"** → `docs/PRESENTATION.md`.
- **"Walk me through one real fire end-to-end"** → `docs/DEMO_SMOKEHOUSE.md`.

## 6. ⚠️ Correction to be aware of

The live project's own `docs/README.md` §10 "File paths" table (written
before the 2026-08-05 flare-label fix) still names the **old, superseded**
files (`tdis_forecast_hrrr.json`, `tdis_train_daily_tx.parquet`, etc.). §5
of that same document explains why those are wrong: ~27% of the original
positive labels were persistent industrial gas-flare hotspots, not real
wildfires; removing them changed the honest model's AUC-PR from 0.63
(inflated) to a real 0.45. The `_filtered` variants are what's actually
served (§5: *"it's the one served (`models/tdis_forecast_hrrr_filtered.json`,
used by script 13)"*), but the file-paths table was never updated to match.

**This package copies the `_filtered` / flare-cleaned files throughout,
not the ones the live README's table literally names**, and lists §3's
script table completions above for the same reason (consistency over
literal fidelity to a stale doc section).

## 7. What's intentionally NOT included

- **`data/weather_hrrr_forecast/hrrr_24h/`** — the raw historical HRRR
  forecast archive (~14 GB). A build input consumed by scripts 05/06/11,
  not needed to view or use these deliverables. At the source path if
  retraining is ever needed.
- **`firms_map_key.txt`** — a FIRMS API credential. Deliberately excluded;
  never belongs in a shared package.
- **Non-`_filtered` model/data files** — superseded by the flare-label fix
  (§6). Kept in the source project for provenance only.
- **Logs, `.done`/`_DONE` marker files, `retired/` dashboard subfolder,
  `__pycache__/`** — pipeline bookkeeping, not deliverable content.

**Update**: the scripts in `scripts/` originally referenced absolute paths
back into the *source* project, meaning running them from inside this
package would have silently touched the source project's files. This is
now fixed — every script resolves its own project root relative to its own
location, so the copies here operate self-containedly on this package's
`data/`/`models/`/`dashboard/`. See `REPRODUCE.md` for exactly what that
does and doesn't unlock (some scripts still need external raw archives not
included here — that document says precisely which).

**Added 2026-08-13/14**: scripts 23-28 (calibration, real-population
validation, flare robustness, hyperparameter sweep), the calibration
report and calibrators (`models/*_isotonic_calibrator.joblib`), and the
full `rev2_improvements/` experiment log (gust/trailing-stat/interaction
ablations, SHAP diagnostics, hyperparameter sweep — includes a cached
`combined_table.parquet` so these can be re-run without the external
gridMET/HRRR archives). See `docs/CLASSIFICATION_METRICS.md` §7 and
`docs/DEPLOYMENT_STRATEGY.md` for the full writeup.

## 8. How to use this package

- **To see the results**: open `dashboard/tdis_fire_dashboard_standalone.html`
  or `dashboard/tdis_fire_explorer_standalone.html` directly in a browser
  (needs internet access for base-map tiles + Leaflet/h3-js libraries; no
  server or install required).
- **For the honest numbers**: `docs/README.md` §5 — quote the flare-filtered
  table, not the earlier superseded one in the same section.
- **For real-fire validation**: `docs/VALIDATION.md`.
- **For a stakeholder-facing summary**: `docs/PRESENTATION.md`.
- **To retrain or extend**: go back to the source project path above and
  run the scripts in numeric order (§3) — this package has the finished
  artifacts and the code, not the raw build-time data.
