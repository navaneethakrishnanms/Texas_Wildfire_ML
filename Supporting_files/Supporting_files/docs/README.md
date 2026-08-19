# TDIS Forecast — Texas Wildfire Risk & Ignition Forecasting

> **Start with the package-level `README.md` §0 ("THE MODEL — read this
> first")** for the current headline results. Short version: the ignition
> forecast model is `models/tdis_forecast_hrrr_filtered.json` + its
> isotonic calibrator. Test-set (balanced sample): AUC-PR 0.4825, best F1
> 0.488. Real full population (2024–26, 1.9% fire-days): AUC-PR 0.0851,
> best F1 0.165, lift 4.4× over random. Forecasting is inference-only —
> no retraining in the daily loop. This file below is the full project
> README, kept for depth; where numbers differ, §0 and
> `CLASSIFICATION_METRICS.md` are newer and authoritative.

**Location:** `/net/cven-mosta-nas.engr.tamu.edu/volume2/NASdata2/miguel_shared/alphaearth_nds/TDIS_Forecast/`
**Status:** preliminary end-to-end system working (data → model → live-forecast dashboard). Two enrichment jobs still finishing (see §9).
**Contact:** mte1224@tamu.edu

> 📖 **New to this project or the data?** Start with **[`HANDBOOK.md`](HANDBOOK.md)** — a
> beginner-friendly "zero → hero" guide explaining every feature (with literature sources),
> how the data combines, how the ignition and wildfire-risk models learn and forecast, what
> features matter, and how it all flows to the dashboard.
> 🔎 **How do we know it works?** See **[`VALIDATION.md`](VALIDATION.md)** — quantitative
> skill plus event tests against 4 real documented Texas wildfires (2024–2026), including the
> record Smokehouse Creek Fire, with an honest account of where the model succeeds and falls short.
> 🗺️ **Where is this going?** See **[§12 Roadmap](#12-roadmap--how-we-make-this-better)** for the
> prioritized improvement plan and **[§13](#13-synergy-with-ignitionnet-90_ig_dec)** for how it
> combines with the IgnitionNet (California sub-daily) project.

---

## 1. TL;DR — what this is

A Texas wildfire system that does **two distinct things**, both mapped on an H3 hex grid and both in a single drag-and-drop dashboard:

1. **Wildfire risk (hazard)** — "if a fire occurs here, how likely/severe" — static WRAP-based hazard, modulated day-by-day by fire-weather → a **forecast** of danger.
2. **Ignition forecast** — "will a fire *be active* here" (strictly fire-*occurrence* risk — the label marks satellite/agency-observed fire days, not pure ignitions; see `HANDBOOK.md §19`) — an ML model, forward-tested on years it never saw, with a **real live 24-hour forecast**.

The headline capability that did **not** exist before: it **forecasts forward in time**, validated on genuinely future years using **real satellite fire labels that extend past 2020**, and can pull a **live HRRR forecast for tomorrow**.

---

## 2. How this differs from the previous model (v5) — honest comparison

The prior model (`../tx_ignition_xgb_v5_tuned.json`) and this one answer *different questions on different data*, so their headline metrics are **not directly comparable**. Here is the honest side-by-side:

| | **Previous: v5_tuned** | **This: TDIS forecast baseline** |
|---|---|---|
| Purpose | static **susceptibility** ranking | **forward forecast** (24/48/72h) |
| Labels | FPA-FOD only, 2014–2020 (times 87% imputed) | **FPA-FOD ∪ VIIRS satellite**, 2014–2026 (real timestamps, extends past 2020) |
| Evaluation | held-out 2019–2020 | **temporal split, tested on 2023–2026 (unseen future)** |
| Forward labels past 2020 | none (couldn't validate the future) | **490k VIIRS-derived** post-2020 labels |
| Spatial coverage | 317k curated cells (~⅓ of TX) | **full 1.7M-cell Texas grid** (static hazard); 3.6k res-5 in dynamic layers |
| Live "tomorrow" forecast | not possible | **yes — real HRRR pull** |
| Test AUC-PR | 0.3744 | honest **0.450**, ceiling **0.443**\*, operational **0.483** *(clean, flare-filtered — superseded the original 0.63/0.66, see §5 note)* |
| Test AUROC | 0.8477 | honest **0.7899**, ceiling **0.8243** |
| Lift over random | ~4.1× | ~3.3× |

### ⚠️ Read the metrics carefully — bigger ≠ better here
The clean honest AUC-PR (0.45) is higher than v5 (0.37), but **the two use different positive base rates** (this dataset ~23% positives vs v5's synthetic ~9%). AUC-PR scales with base rate, so raw numbers can't compare head-to-head. The **base-rate-robust lift over random** is the fairer comparison: ~1.9× here vs v5's ~4.1× — v5 is NOT beaten on raw signal. By AUROC, v5 is also higher (0.85 vs 0.71).

**So is the model itself more skillful? No — not clearly, on the raw signal.** Texas ignition is dominated by *where* (roads, terrain, fuels), and weather adds only modest daily skill (the honest→ceiling gap is just +0.03 AUC-PR). That ceiling was already known from the v5 QC and it still holds.

### What IS genuinely better (the real advance)
Not the metric — the **capability and honesty of the system**:
- It is a **true forecast** (temporal split, forward-tested), not a hindcast/susceptibility map mislabeled as a forecast.
- It has **real forward labels** (VIIRS) so 2021–2026 performance can actually be *measured*, which v5 never could.
- It has **full-Texas coverage** (v5 silently missed ⅔ of the state's cells).
- It can produce a **live forecast for a day that hasn't happened yet**.
- The architecture is the industry-standard **static hazard × dynamic fire-weather** (like USFS WFPI), so it extends cleanly.

Bottom line: **a more capable, more honest, deployable *system* — built on comparable underlying signal.** Don't sell it as "2× more accurate."

---

## 3. Data pipeline

### 3a. Fire labels — fused, real-timestamp (the key upgrade)
- **FPA-FOD** (`../90%_ig_dec/.../fpa_fod_tx_h3.parquet`): complete ignition inventory 2014–2020, but 87% of ignition *times* are imputed.
- **VIIRS active fire** (NASA FIRMS, S-NPP + NOAA-20, 2014–2026): real satellite detection timestamps at 375 m; **extends labels past 2020** (where FPA-FOD ends).
- **Fusion** (`scripts/02_build_labels.py` → `data/labels_fused/ignitions_daily_tx.parquet`): daily fire-occurrence inventory, **960,054 raw positive cell-days** (≈6.1% of which are NM/OK bounding-box spillover — provably excluded from all training; **655,749 clean in-Texas** after flare + polygon filters, `SANITY_CHECK.md F1`), ~490k post-2020 (VIIRS) — this is what makes forward validation possible. ⚠️ Label-era note: positive rate steps up in **2018 when NOAA-20 joined S-NPP** — a sensor artifact, not a fire trend (`SANITY_CHECK.md F3`).

### 3b. Static features — full Texas (fixed a real bug)
The old `../tx_geo_features.parquet` covered only 317k cells; 55% of VIIRS fires fell *outside* it → silent coverage gap. Rebuilt for the full **1.7M-cell** grid:
- `scripts/04_build_full_tx_static.py` → `data/static_features/tx_static_master.parquet` (1,708,940 cells, 0 missing).
- Ecoregion (EPA L3), elevation/slope/aspect (Copernicus GLO-30 DEM), road distance (Census TIGER 2022), canopy CBD/CBH (LANDFIRE), WHP + burn probability (TxWRAP).
- Known limitation: `flep4`/`cfl` (flame length) are NaN in the full-grid source, so they're excluded from the full-coverage hazard composite (real values exist only for the 317k training cells).

### 3c. Weather — observed (training) and forecast (deployment)
- **gridMET** (observed daily, 2014–2026): ERC, VPD, wind, etc. Used for training features and the 2024 daily fire-weather layer.
- **HRRR forecast** (`scripts/05_download_hrrr_forecast.py`): what the forecast *actually said* 24h ahead, so the model can eventually train on forecast-realistic inputs (avoids the observed→forecast train/serve mismatch). **Important product finding: HRRR's F24 forecast did not exist before 2018-07-12** (HRRRv3 upgrade); ~35% of label dates predate it and are flagged for gridMET backfill. HRRR maxes at F48 → **GFS** used for F72.

---

## 4. The model — forecasting discipline

`scripts/07_train_baseline_forecast.py` → `models/tdis_forecast_baseline_{honest,ceiling}.json`

Two disciplines make it a real forecast (not a leaky hindcast):
1. **Temporal split** — train **2014–2021**, validate **2022**, test **2023–2026** (pure future holdout, real VIIRS labels). Never a random split.
2. **No same-day-weather leak** — the deployable **honest** model uses only what's knowable at forecast time: static (roads/terrain/fuels/ecoregion) + temporal (season/day-of-week/holiday). The **ceiling** model adds same-day weather to show the optimistic bound *if* forecasts were perfect.

The real operational model swaps in **HRRR forecast weather** and lands between honest and ceiling — but since that gap is only ~0.03 AUC-PR, expect a marginal (not transformative) gain.

---

## 5. Results (test = 2023–2026, unseen future years)

**⚠️ Table below is SUPERSEDED / pre-flare-fix — kept for provenance only. Current numbers: §5 clean-label table further down.**

| Model | AUC-PR | AUROC | Lift | Reading |
|---|---|---|---|---|
| honest (deployable, OLD flare-trained) | 0.6296 | 0.7899 | 3.31× | floor, no leakage |
| ceiling (perfect wx, OLD flare-trained) | 0.6598 | 0.8243 | 3.11× | optimistic bound |

Per-year performance is **stable across 2023/2024/2025/2026** (AUC-PR 0.58–0.65) — the best sign it generalizes forward rather than overfitting one year. Top features: elevation, ecoregion, burn probability, road distance (note: WHP and burn probability correlate at r=0.94 — treat them as ONE fuels factor when reading importances, `SANITY_CHECK.md F5`) — i.e., *where*, consistent with prior findings that weather is a modest modulator.

### ⚠️ 2026-08-05 label-cleaning update — the honest numbers dropped (and that's correct)
The label deep-dive found **~27% of positives are persistent industrial hotspots** (gas
flares — 531 cells "on fire" >3% of all days). Removing them (plus 86k phantom post-label
negatives) and re-scoring on the SAME clean test rows (`scripts/18_flare_filter_retrain.py`):

| Model, clean test rows (identical rows old vs new) | AUC-PR | AUROC | Lift |
|---|---|---|---|
| Honest OLD (flare-trained) | 0.386 | 0.651 | 1.65× |
| **Honest NEW (flare-filtered, 400 trees)** | **0.450** | **0.711** | **1.92×** |
| Operational-HRRR OLD (flare-trained) | 0.484 | 0.732 | 2.07× |
| **Operational-HRRR NEW (flare-filtered)** | 0.483 | 0.733 | 2.06× |

**Read:** the previously reported 0.63 honest AUC-PR was heavily inflated by trivially-
predictable always-on flare cells. On *real wildfires only*: honest skill **AUC-PR ≈ 0.45 /
AUROC ≈ 0.71 / lift ≈ 1.9×**; operational-HRRR **≈ 0.48 / 0.73 / 2.1×**. Retraining the
honest model without flares is a clear win (+0.064 AUC-PR on identical rows); the HRRR
model's metrics are unchanged but the filtered version no longer *learns* flare cells, so
it's the one served (`models/tdis_forecast_hrrr_filtered.json`, used by script 13).
Training note: early stopping was removed for the filtered retrains — the 2022 validation
year shows a spurious AUCPR spike at ~3 trees under clean labels (fixed 400-tree budget;
val still rising at 399). The dashboard ignition layer was regenerated from the filtered
honest model (flare hexagons visibly drop, e.g. Smokehouse-box max 0.716 → 0.495).
Event validation (VALIDATION.md — real documented fires) and the wildfire-risk product
(hazard × FWI) are unaffected. Quote THESE numbers; the tables below are pre-fix provenance.

### Classification metrics (precision/recall/F1) + threshold sweep
See `VALIDATION.md §10` / `HANDBOOK.md §22` for the full threshold sweep (0.2–0.8) across
honest/operational-HRRR/ceiling. Headline: at threshold 0.5, operational-HRRR reaches
precision 0.410 / recall 0.602 / F1 0.488 — beating honest (0.383/0.584/0.463) at every
threshold on identical rows. F1 peaks near 0.5 but still misses ~40% of real fire-days;
a recall-first policy would use threshold 0.3–0.4 instead.

### HRRR forecast-weather model (operational) — controlled ablation
We then trained the **operational model** with HRRR *forecast* weather added
(`models/tdis_forecast_hrrr.json`, GPU-trained). To measure HRRR's true contribution we ran
a **clean ablation** — identical rows, identical temporal split, the *only* difference being
whether the 3 HRRR forecast features are present:

| Model (identical rows/test) | AUC-PR | AUROC | Lift |
|---|---|---|---|
| static + temporal (no weather) | 0.6865 | 0.7775 | 2.30× |
| **+ HRRR forecast weather (OLD, flare-trained)** | **0.6976** | **0.7975** | 2.33× |
| **True HRRR contribution** | **+0.0111 (+1.6%)** | **+0.0200** | — |

**Interpretation (honest):** HRRR forecast weather adds a **small but real** gain (+0.011
AUC-PR, +0.020 AUROC), landing below the +0.03 perfect-observed-weather ceiling exactly as
expected (forecast weather is noisier than observed). `hrrr_vpd` is a top-5 feature and the
3 HRRR features carry ~15% of model importance — so the model genuinely uses forecast
dryness/wind. **But** *where* (elevation, ecoregion, burn-prob, roads = top 4) still
dominates; weather is a modest daily modulator. The value of the HRRR model is that it's a
**genuine, deployable forecast** (responds to forecast conditions directly), not a large
accuracy jump. ⚠️ Note: an earlier uncontrolled comparison showed +0.068 — that was a
base-rate artifact (test pos-rate 0.30 vs 0.19); the **ablation's +0.011 is the trustworthy number.**

---

## 6. The dashboard

**`dashboard/tdis_fire_dashboard_standalone.html`** — single self-contained file (~62 MB, all data + 3 FWI variants embedded). **Double-click / drag-and-drop** — no server needed (needs internet for base-map tiles + Leaflet/h3-js libraries).

Every view is a **relative ranking** of hexes on a 0–1 colour scale (bands: Extreme ≥.88 … Low <.25), **not a calibrated probability**. The legend names the exact machinery of whatever tab is active. Modes:

- **🔥 Wildfire Risk** — `TxWRAP hazard × daily FWI`, **no ML**. Per-day scrubber **2024–2026** + play. (Legend: "Wildfire risk (2024–26)".)
- **✦ Ignition** — computed **two different ways**, and the legend says which:
  - *scrubber* → **"Ignition · heuristic"**: honest susceptibility × daily FWI — a **proxy, not the model's replay** (§HANDBOOK 20A);
  - *live* → **"Ignition · live model"**: the operational HRRR model run directly (no FWI).
- **▦ Static Hazard (full TX)** — WRAP baseline, **no ML**, all 5,382 res-5 cells, metric picker (Composite/WHP/Burn Prob/Canopy).
- **📅 Live Forecast (24h / 48h)** — real forward HRRR forecasts (`scripts/13_model_forecast_day.py`). In **Ignition** = the actual HRRR-trained model's per-cell predictions on forecast weather; in **Wildfire Risk** = hazard × forecast-FWI. (72 h GFS tab retired: weakest skill + GFS soil-moisture gaps.)
- **Fire-weather formula picker** (Composite / NOAA HWP / TX HWP) — re-ranks every **FWI-based** view (Wildfire Risk both, scrubber-Ignition); **hidden** for live-Ignition (weather is inside the model) and Static Hazard.

Tooltips decompose every hex into combined value + static hazard + fire-weather index, labeled with the active formula, so the hazard-vs-ignition distinction stays legible.

**NEW — Dashboard v2 (zoom-adaptive):** `dashboard/tdis_fire_explorer_standalone.html` (~102 MB) —
same content plus an h3geo-style **zoom-adaptive grid** (res 4 statewide → **res 8 = full model
resolution** when zoomed in; statics at fine res × res-5 weather) and a **"% of record" color
scale** that fixes the summer saturation of dryness-driven HWP formulas by coloring each hex by
today's percentile within its own 2024–26 record. Details: `HANDBOOK.md §21`.

*(Served alternative: `dashboard/tdis_fire_dashboard.html` + `tdis_dashboard_data.json` — use `python -m http.server` in that folder.)*

---

## 7. Are things better overall? — the straight answer

**As a system: clearly yes.** It forecasts forward, is validated on real unseen years, covers all of Texas, and can produce a live forecast for tomorrow — none of which v5 could do.

**As a raw predictor: about the same.** The underlying skill (lift over random, AUROC) is comparable to v5; the higher AUC-PR number is a base-rate artifact, not a real accuracy jump. Weather still adds only a small daily signal. Don't over-claim.

**What this unlocks:** a defensible operational forecast product and — critically — the ability to *measure* forward performance (via VIIRS) that we were previously flying blind on.

---

## 8. Known limitations / honest caveats
- **Metric non-comparability** with v5 (different base rates & label sources) — always quote **lift**, not raw AUC-PR, across models.
- **Ignition dynamics are weak day-to-day** — the honest model has no daily weather; genuine daily motion comes from the `× fire-weather` two-stage step, not the ML model itself.
- **Live forecast uses VPD+wind only** (HRRR carries no ERC), a slightly different fire-weather recipe than the gridMET-based 2024 layers. Labeled as such in the dashboard.
- **Dynamic layers cover 3.6k res-5 cells** (gridMET/modeled universe); static hazard covers the full 5,382.
- **Detection/reporting bias** in the fire labels (near-road fires over-represented) — inherited from FPA-FOD/VIIRS, affects all models on this data.
- **"Ignition" = fire occurrence.** The fused label marks days a fire was *active* in a cell (96% satellite-detected), not strict ignition events; multi-day fires contribute multiple positive days. Fusion validity + literature: `HANDBOOK.md §19`.
- **This is a preliminary baseline** — the full-coverage dataset (`rebuild` job) and HRRR-forecast-weather model are still being built (§9).

---

## 9. In-progress background jobs (as of this writing)
- **`rebuild`** (tmux) — final training dataset on the full 1.7M-cell static layer (removes the coverage gap from the current 30.7%-complete dataset). → will overwrite `tdis_train_daily_tx.parquet`.
- **`hrrrfc`** (tmux) — HRRR forecast weather, ~2,000/4,018 dates done (~2h ETA) → `data/weather_hrrr_forecast/`. Feeds the operational (forecast-weather) model.
When both finish: rebuild dataset → retrain with forecast weather → regenerate dashboard data. Dashboard *shape* won't change.

---

## 10. File paths

### Dashboard (share these)
- `dashboard/tdis_fire_dashboard_standalone.html` — **drag-and-drop dashboard** (data embedded)
- `dashboard/tdis_fire_dashboard.html` + `dashboard/tdis_dashboard_data.json` — served version
- `dashboard/forecast_2026-07-31.json` — live HRRR forecast for tomorrow

### Models
- `models/tdis_forecast_hrrr.json` (+ `_meta.json`) — **OPERATIONAL forecast model** (static + temporal + HRRR forecast weather; GPU-trained; used for live dashboard forecasts)
- `models/tdis_forecast_baseline_honest.json` (+ `_meta.json`) — no-weather baseline
- `models/tdis_forecast_baseline_ceiling.json` (+ `_meta.json`) — perfect-weather bound
- `models/baseline_forecast_predictions_test.parquet` — future-test predictions

### Data
- `tdis_train_daily_tx.parquet` — daily training dataset (preliminary; rebuild pending)
- `data/labels_fused/ignitions_daily_tx.parquet` — fused FPA-FOD + VIIRS ignitions
- `data/labels_viirs/viirs_tx_h3.parquet` — VIIRS detections, H3-gridded
- `data/static_features/tx_static_master.parquet` — full-TX static features (1.7M cells)
- `data/weather_hrrr_forecast/hrrr_24h/*.parquet` — HRRR F24 forecast weather (building)

### Scripts (pipeline order)
- `scripts/01_download_viirs.py` — VIIRS download + H3 assign
- `scripts/02_build_labels.py` — fuse FPA-FOD + VIIRS → daily ignitions
- `scripts/03_build_dataset.py` — assemble training dataset (matched negatives, features)
- `scripts/04_build_full_tx_static.py` — full-TX static feature master
- `scripts/05_download_hrrr_forecast.py` — historical HRRR/GFS forecast weather
- `scripts/06_attach_hrrr_forecast.py` — attach HRRR forecast weather to training rows
- `scripts/07_train_baseline_forecast.py` — train forecast baselines (honest + ceiling)
- `scripts/08_build_dashboard_data.py` — hazard + ignition → dashboard JSON
- `scripts/09_build_perday_dynamic.py` — daily fire-weather (2024–2026) for per-day scrubber
- `scripts/10_live_forecast_day.py` — pull a live HRRR forecast (FWI only) for a target day
- `scripts/11_train_hrrr_forecast.py` — train operational model with HRRR forecast weather (GPU)
- `scripts/12_hrrr_ablation.py` — controlled ablation isolating HRRR's true contribution
- `scripts/13_model_forecast_day.py` — run the HRRR model on live forecast weather (HRRR≤48h / GFS 72h; incl. gust + MSTAV soil moisture) → dashboard forecast
- `scripts/14_event_validation.py` — 4-event validation battery (VIIRS capture / WHERE / WHEN)
- `scripts/15_reweight_fwi.py` — instant rebuild of ALL 3 fire-weather variant arrays from cached components
- `scripts/16_fit_hwp_tx.py` — fit TX-calibrated HWP coefficients to VIIRS FRP (large-fire episodes)
- `scripts/17_embed_standalone.py` — rebuild the drag-and-drop standalone dashboard
- `scripts/fwi_config.py` — **the knobs**: `FWI_WEIGHTS` (composite) + `FWI_MODE` (composite / NOAA HWP / TX HWP)
- `data/hwp_params.json` — fitted TX HWP coefficients + normalization refs

### Reference docs
- `PLAN.md` — dataset & model plan / locked decisions
- `HANDOFF.md` — unattended-run handoff notes
- `../DATA_AND_FORECAST_ROADMAP.md` — data sources, wildfire-vs-ignition, roadmap

### How the training database was built (full narrative + per-step code links: `HANDBOOK.md §18`)
1. **Labels** ([`01`](scripts/01_download_viirs.py)–[`02`](scripts/02_build_labels.py)): VIIRS 1.62M detections ∪ FPA-FOD → 960k positive cell-days (96% satellite-only; median known size 2 ac; ⚠️ ~18–27% flare contamination → filtered variant via [`18`](scripts/18_flare_filter_retrain.py))
2. **Statics** ([`04`](scripts/04_build_full_tx_static.py)): TxWRAP + LANDFIRE + DEM + roads + ecoregion → 1.71M cells, 0 nulls
3. **Table** ([`03`](scripts/03_build_dataset.py)): positives + matched negatives (same cells, non-fire days) + gridMET weather + temporal encodings → 3.6M rows, synthetic 0.267 label rate
4. **Forecast weather** ([`05`](scripts/05_download_hrrr_forecast.py)–[`06`](scripts/06_attach_hrrr_forecast.py)): historical HRRR F24 archive (~14 GB, 2018-07+)
5. **Train** ([`07`](scripts/07_train_baseline_forecast.py)/[`11`](scripts/11_train_hrrr_forecast.py)/[`12`](scripts/12_hrrr_ablation.py)): temporal split, XGBoost-GPU → [`models/`](models/)
6. **Serve** ([`08`](scripts/08_build_dashboard_data.py)–[`09`](scripts/09_build_perday_dynamic.py), [`13`](scripts/13_model_forecast_day.py), [`15`](scripts/15_reweight_fwi.py)–[`17`](scripts/17_embed_standalone.py)): dashboard + live forecasts + FWI knobs

### Data at a glance (full descriptive statistics + source links: `HANDBOOK.md §17`)
| Dataset | Size | Coverage | Notes |
|---|---|---|---|
| Fused fire labels | 960,054 positive cell-days | 2014 → 2026-07-29 | FPA-FOD ∪ VIIRS; ~490k post-2020 |
| VIIRS detections | 1,619,215 | 375 m, real timestamps | FRP 0–1,677 MW; flare contamination QC item |
| Training table | 3,595,513 cell-days | 619,218 cells × 13 yr | label rate 0.267 (synthetic, matched negatives) |
| Dynamic components | 3,381,192 cell-days | 3,624 res-5 cells × 933 days | ERC mean 44, VPD 1.33 kPa, wind 4.2 m/s |
| Static master | 1,708,940 cells | full TX, 0 nulls | flep4/cfl dead (100% zero) |

Sources: [FPA-FOD](https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6) ·
[NASA FIRMS/VIIRS](https://firms.modaps.eosdis.nasa.gov/) ·
[gridMET](https://www.climatologylab.org/gridmet.html) ·
[HRRR](https://registry.opendata.aws/noaa-hrrr-pds/) ·
[TxWRAP](https://www.texaswildfirerisk.com/) · [LANDFIRE](https://landfire.gov/) ·
[GLO-30 DEM](https://registry.opendata.aws/copernicus-dem/) ·
[TIGER roads](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) ·
[EPA ecoregions](https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states) ·
[NOAA HWP paper](https://doi.org/10.1175/WAF-D-24-0068.1)

---

## 11. Reproduce / run
```bash
PY=/home/mte1224/mambaforge/envs/UAI2526/bin/python
cd /net/cven-mosta-nas.engr.tamu.edu/volume2/NASdata2/miguel_shared/alphaearth_nds/TDIS_Forecast

# view the dashboard
#   → just open dashboard/tdis_fire_dashboard_standalone.html in a browser

# refresh the live forecasts (24/48h HRRR, 72h GFS) and re-embed the standalone
$PY scripts/13_model_forecast_day.py 2026-08-06 24
$PY scripts/17_embed_standalone.py

# tune the fire-weather knobs (weights / formulation), then rebuild arrays in ~30 s
#   edit scripts/fwi_config.py  (FWI_WEIGHTS, FWI_MODE)
$PY scripts/15_reweight_fwi.py && $PY scripts/17_embed_standalone.py

# retrain the baseline
$PY scripts/07_train_baseline_forecast.py
```

**Always use `/home/mte1224/mambaforge/envs/UAI2526/bin/python`.** Do not recreate a `.venv` in this shared dir.

---

## 12. Roadmap — how we make this better

Honest status: the system is a **credible relative-risk ranking + demonstration dashboard**, not yet a
calibrated life-safety operational forecast. The work below is ordered by **value ÷ effort** and grouped
into three tiers. Tier 0 (the wind knob) is **done**; everything else is open.

### ✅ Tier 0 — done
- **Wind re-weighting of the fire-weather index** (`fwi_config.py`, wind 0.20 → 0.40) + HRRR gust in live
  forecasts. Lifted the two under-scored wind-driven validation fires (Smokehouse 61→75th, Lavender 80→94th)
  with no regression on heat-driven fires. See `VALIDATION.md §6`.
- **HWP fire-weather variants (2026-08-05)** — implemented NOAA GSL's Hourly Wildfire Potential
  (James et al. 2025, WAF, Eq. 3 **exact**, incl. HRRR gust + MSTAV soil moisture on the live path)
  and a **Texas-calibrated** version (fit to VIIRS FRP over large-fire episodes). Both are dashboard-
  toggleable alongside the composite (covers roadmap item 2.4). Head-to-head on the 4-event battery:
  composite stays default; NOAA HWP wins Smokehouse (80th); TX fit reveals daily fire activity is
  **dryness-dominated** (wind signal only exists at hourly/gust resolution). `HANDBOOK.md §16`,
  `VALIDATION.md §8`.

### 🔨 Tier 1 — production-hardening (cheap, high-value; do these first)
These are what stand between "demo" and "sealed for an internal operational dashboard."

| # | Item | Why it matters | Effort | Deliverable |
|---|---|---|---|---|
| 1.1 | **Automated daily refresh (cron)** | It's manual today. A production forecast must self-update each morning with the new HRRR run. | Low | cron job → `13_model_forecast_day.py` for tomorrow + re-embed dashboard |
| 1.2 | **Calibration pass** | Bands ("Extreme"…"Low") are relative tiers, not probabilities. A reliability curve (predicted vs. observed fire rate) makes them mean something and enables honest % language. | Med | `16_calibrate.py` → isotonic/Platt calibration + reliability plot in `VALIDATION.md` |
| 1.3 | **Expand event validation to ~15–20 fires** | n=4 is a sanity check, not a statistical claim. Add more years/regions/causes. | Med | extend `EVENTS` in `14_event_validation.py`; validation table |
| 1.4 | **Fix flep4/cfl train/serve mismatch** | Flame-length features are real for training cells but zero-filled on the full 1.7M grid — a genuine inconsistency. | Med | rasterize LANDFIRE flame-length to full grid in `04_build_full_tx_static.py` |
| 1.5 | **Drift / freshness monitoring** | An operational feed silently breaks (HRRR outage, gridMET lag). Need an alert. | Low | freshness check + log in the cron; banner in dashboard when data is stale |

### 🚀 Tier 2 — model skill (moderate effort, targets the weak "WHEN" signal)
The known gap is that **daily ignition timing skill is modest** — most day-to-day motion comes from the
FWI heuristic, not the learned model. These add genuine dynamic signal.

| # | Item | Why | Source |
|---|---|---|---|
| 2.1 | **Sub-daily (6-hour window) forecasting** | The single biggest skill upgrade. IgnitionNet proved per-window HRRR adds real WHEN discrimination (see §13). Directly attacks our weakest dimension. | port IgnitionNet architecture |
| 2.2 | **Lightning (GOES-GLM)** | Natural ignitions have no road/human signal; Hunggate was caught on weather alone. A lightning-density feature would catch them on cause. | NOAA GOES-16/18 GLM |
| 2.3 | **NDVI / live fuel curing** | Grass-fire danger tracks green-up/cure state, which static fuels miss. Explains why identical weather burns differently in spring vs. summer. | MODIS/VIIRS NDVI, Sentinel-2 |
| 2.4 | **Hot-Dry-Windy (HDW) index option** | A physically-grounded alternative to our composite FWI; another tunable knob for wind-driven events. | published HDW formulation |
| 2.5 | **Full-grid dynamic weather** | Dynamic layers cover 3.6k res-5 cells; static covers 5,382. Extend gridMET/HRRR interpolation to the full grid. | existing pipeline, wider footprint |

### 🔬 Tier 3 — research & scale (larger, higher-risk/higher-reward)
| # | Item | Why |
|---|---|---|
| 3.1 | **Two-state benchmark (CA + TX)** | Publishable generalization result; IgnitionNet already has a CA→TX transfer pipeline (§13). |
| 3.2 | **Spread / severity module** | Move from "will it start / how bad if it burns" to "how far will it run" (fire-spread modeling). |
| 3.3 | **Uncertainty quantification** | Ensemble/quantile outputs so each hex carries a confidence, not just a point risk. |
| 3.4 | **FT-Transformer / sequence model** | IgnitionNet has a transformer variant; a temporal sequence model could capture multi-day fuel build-up. |
| 3.5 | **Public-facing hardening** | Only after 1.x + calibration: access control, audit trail, SLA, formal accuracy statement. |

**Recommended immediate next two:** **1.1 (daily-refresh cron)** and **1.2 (calibration)** — together they
move the system furthest toward "sealed" for the least effort, and 1.2 is a prerequisite for any probability
claim on the dashboard.

---

## 13. Synergy with IgnitionNet (`../90%_ig_dec`)

The sister project **IgnitionNet** (`../90%_ig_dec/`) is a **California, sub-daily (6-hour window)**
wildfire-ignition study (XGBoost + FT-Transformer, H3-8, 2014–2020, with per-window HRRR, LISA spatial-
cluster analysis, and cause-stratified lightning-vs-human bivariate LISA). TDIS and IgnitionNet are highly
complementary — same H3-8 grid, same core landscape hazard layers (`avg_burn_prob`, `whp`, `flep4`, `cfl`),
same DAY_MATCHED negative-sampling and AUPR base-rate discipline — differing mainly in **geography (CA vs
TX)**, **time resolution (6-hour vs daily)**, and **purpose (retrospective analysis vs live forward forecast)**.

**Crucially, IgnitionNet already contains a California→Texas transfer pipeline**
(`../90%_ig_dec/Transferibility_OutOfState/`: `state_configs.py`, `build_train_state.py`,
`fetch_hrrr_state.py`, etc.), so the two projects already share Texas as a target.

### What each gives the other

| IgnitionNet → TDIS (borrow) | TDIS → IgnitionNet (contribute) |
|---|---|
| **Sub-daily per-window HRRR architecture** — proven to add WHEN discrimination (18Z afternoon peak AUROC +0.026, intra-day variance 1.5× daily). This is TDIS's biggest open skill gap (Tier 2.1). | **Real forward VIIRS labels past 2020** — IgnitionNet stops at 2020 (FPA-FOD only); TDIS's 490k post-2020 VIIRS labels let CA models be *validated on the genuine future* too. |
| **FT-Transformer** and the sub-daily model-comparison harness. | **Live operational forecast loop** (HRRR/GFS pull → model → dashboard) — IgnitionNet is retrospective; TDIS's `13_model_forecast_day.py` makes it deployable. |
| **LISA / bivariate-LISA spatial validation** — a rigorous, publishable way to show risk clusters in genuinely fire-prone terrain and to separate lightning vs human ignition geography. | **Full-state coverage + drag-and-drop dashboard** — a stakeholder-facing product IgnitionNet lacks. |
| **Cause-stratified modeling** (lightning vs human) — informs TDIS Tier 2.2 (GOES-GLM lightning). | **Wind-forward FWI knob + gust handling** (`fwi_config.py`) — reusable for CA's wind-driven (Diablo/Santa Ana) events. |

### Concrete joint opportunities
1. **Shared feature/label pipeline.** Both already build H3-8 FPA-FOD labels, LANDFIRE/FSim hazard, and
   gridMET/HRRR weather. Converge on one parameterized-by-state builder (IgnitionNet's
   `Transferibility_OutOfState/*_state.py` is already this) so TX and CA stay methodologically identical and
   directly comparable.
2. **Two-state generalization paper (Tier 3.1).** CA (sub-daily, retrospective) + TX (daily, forward-validated)
   = a strong "does ignition prediction transfer across fire regimes?" result. The CA→TX transfer pipeline
   already exists; TDIS supplies the forward VIIRS labels and the live-forecast angle.
3. **Upgrade TX to sub-daily (Tier 2.1) using IgnitionNet's harness** — the highest-value model improvement,
   with a proven blueprint next door.
4. **Cross-validate FWI weightings.** Test whether the wind-forward TX knob also improves CA's Diablo/Santa
   Ana wind events — a cheap cross-regime robustness check for both.

> **Data-sharing caveat:** IgnitionNet's `train_data_perwindow_v1.parquet` embeds proprietary AlphaEarth
> features and is **never shared**. Any joint pipeline must use only the public feature set (landscape +
> gridMET/HRRR + geometry), which both projects already isolate.
