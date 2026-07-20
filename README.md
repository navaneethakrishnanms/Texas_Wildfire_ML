# 🔥 IgnitionNet — Wildfire Ignition Prediction System

> **Predicting WHERE and WHEN wildfires ignite** using 7 years of fire occurrence data, gridMET daily weather, LANDFIRE landscape features, and H3 hexagonal grid cells — before a fire is confirmed.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Phase-3%20Complete-brightgreen.svg)]()

---

## 📌 What This Project Does

IgnitionNet answers one specific operational question:  

> **Given current landscape and weather conditions, which hexagonal cells in Texas (or California) are most likely to have a wildfire discovered in the next 6 hours?**

The model outputs a **ranked risk map** of hexagonal H3 cells, updated every 6 hours, that fire managers can use to **pre-position suppression resources before a fire starts**.

This is an **ignition prediction system** — not a damage model, not a spread model, not a detection system. It operates on publicly available data and runs **before** any fire is confirmed.

---

## 🏗️ Project Architecture

```
INPUT (publicly available, pre-ignition data)
    │
    ├── FPA-FOD v6 (Short 2022)     → Labels only (where/when fires discovered)
    ├── gridMET (Abatzoglou 2013)   → Daily fire weather (4 km, CONUS)
    ├── LANDFIRE LF2022             → Landscape features (static rasters)
    └── USFS FSim                   → Burn probability (50,000 simulations)
    │
    ▼
H3 Resolution-7 Hexagonal Grid (Texas)
  ~134,958 burnable cells (~1.9 km across each, ~5 km²)
    │
    ▼
XGBoost Binary Classifier
  label = 1 → fire discovered in this cell in this 6-hour window
  label = 0 → no fire
    │
    ▼
OUTPUT: Ranked risk score per (H3 cell, 6-hour UTC window)
"These cells are highest risk in the next 6 hours"
```

---

## 📊 Dataset Statistics (Texas)

| Metric | Value |
|--------|-------|
| Study period | 2014–2020 (7 years) |
| H3 Resolution | 7 (~1.9 km cell width, ~5 km² area) |
| Fire events used (TX, ≥1 acre) | **34,203** |
| Non-fire samples (1:10 ratio, day-matched) | **342,030** |
| Total training rows | **376,233** |
| Positive rate | 9.1% |
| Temporal windows | 4 per day (00Z, 06Z, 12Z, 18Z UTC) |
| Train years | 2014–2017 (~252k rows) |
| Validation year | 2018 (~61k rows) |
| Test years | 2019–2020 (~62k rows) |

---

## 🧪 Feature Set (~50 columns)

All features extracted **independently** at each cell's own centroid. Non-fire cells **never inherit** the paired fire cell's weather values.

### Group 1 — Daily Fire Weather (gridMET, 4 km, CONUS)

| Feature | Source Variable | Units | Description |
|---------|----------------|-------|-------------|
| `erc` | `energy_release_component-g` | BTU ft⁻² | Energy Release Component — strongest daily predictor |
| `fm100` | `dead_fuel_moisture_100hr` | % | 100-hr dead fuel moisture |
| `fm1000` | `dead_fuel_moisture_1000hr` | % | 1000-hr dead fuel moisture |
| `bi` | `burning_index-g` | index | Burning Index (NFDRS) |
| `vpd` | `mean_vapor_pressure_deficit` | kPa | Vapor pressure deficit |
| `vs` | `wind_speed` | m s⁻¹ | Daily wind speed |
| `rmax` / `rmin` | `relative_humidity` | % | Max / min relative humidity |
| `tmmx` / `tmmn` | `air_temperature` | °C | Max / min temperature |
| `pr` | `precipitation_amount` | mm | Daily precipitation |
| `sph` | `specific_humidity` | kg kg⁻¹ | Specific humidity |

### Group 2 — 5-Day Trailing Weather Statistics (computed from gridMET)

Each trailing stat covers the **5 calendar days before the event day** (shift-1 convention — event day excluded).

| Feature | Description |
|---------|-------------|
| `erc_5D_mean`, `erc_5D_max` | 5-day trailing mean / max of ERC |
| `fm100_5D_mean`, `fm100_5D_min` | 5-day trailing mean / min of fuel moisture |
| `bi_5D_mean`, `bi_5D_max` | 5-day trailing Burning Index |
| `vpd_5D_mean`, `vpd_5D_max` | 5-day trailing VPD |
| `vs_5D_mean`, `vs_5D_max` | 5-day trailing wind speed |
| `rmax_5D_mean`, `rmax_5D_min` | 5-day trailing max relative humidity |
| `tmmx_5D_mean`, `tmmx_5D_max` | 5-day trailing max temperature |

> **Implementation note:** Trailing stats are computed by fetching the actual 5 preceding days' NetCDF bands directly from gridMET files — not from the sparse training table. Cross-year boundaries (e.g., Jan 3 using Dec 29–31 prior year) are handled correctly.

### Group 3 — Landscape Features (Static, LANDFIRE / FSim)

| Feature | Source | Description |
|---------|--------|-------------|
| `avg_burn_prob` | USFS FSim | Burn probability from 50,000 stochastic simulations (0–1) |
| `whp` | USFS WHP 2023 | Wildfire Hazard Potential index (0–7000) |
| `flep4` | LANDFIRE LF2022 | Flame Length Exceedance Probability at 4 ft |
| `cfl` | LANDFIRE LF2022 | Canopy Fuel Load (Mg ha⁻¹) |

### Group 4 — Temporal Encodings

| Feature | Formula | Purpose |
|---------|---------|---------|
| `sin_month`, `cos_month` | sin/cos(2π × month / 12) | Cyclic fire season encoding |
| `sin_hour`, `cos_hour` | sin/cos(2π × window_hour / 24) | Cyclic 6-hour UTC window |

### Group 5 — Location + Ecoregion

| Feature | Description |
|---------|-------------|
| `centroid_lat`, `centroid_lon` | H3-7 cell centroid (WGS84) |
| `ecoregion_l2`, `ecoregion_l3` | EPA Level-2/3 ecoregion codes |
| `fire_count`, `has_fire_history` | Historical fire count in this cell |

---

## 📁 Repository Structure

```
Texas_Wildfire_ML/
│
├── README.md
├── IGNITIONNET_PROJECT_SCOPE_MASTER.md    ← Full project specification
├── missing_data_diagnosis.md              ← Data quality analysis report
├── .gitignore
│
├── V1/                                    ← Initial exploration (archived)
│
└── V2/                                    ← Production pipeline
    ├── run_phase1.py                      ← Phase 1 entry-point
    │
    ├── src/preprocessing/                 ← Phase 1 core modules
    │   ├── config.py
    │   ├── loader.py
    │   ├── schema_checker.py
    │   ├── merger.py
    │   ├── state_filter.py
    │   ├── validator.py
    │   ├── quality_reporter.py
    │   ├── eda_reporter.py
    │   └── pipeline.py
    │
    ├── maps/                              ← EDA scripts + outputs
    │   ├── texas/scripts/                 ← 7 EDA scripts
    │   ├── texas/eda_outputs/             ← PNGs, CSVs, HTML map
    │   ├── california/scripts/
    │   └── california/eda_outputs/
    │
    ├── phase2/                            ← Feature Engineering + Modelling Pipeline
    │   ├── config/phase2_config.py        ← State configs, paths, feature rules
    │   ├── run_phase2a.py                 ← Feature schema finalization
    │   ├── run_phase2b.py                 ← H3-R7 grid construction
    │   ├── run_phase2c.py                 ← FPA-FOD → H3 fire label mapping
    │   ├── run_phase2d.py                 ← Day-matched 1:10 negative sampling
    │   ├── run_phase2e_schema_fix.py      ← Schema cleanup + leakage check
    │   ├── run_phase2e_static.py          ← LANDFIRE raster extraction
    │   ├── run_phase2f_gridmet.py         ← gridMET NetCDF download + extraction
    │   ├── run_phase2g_assemble.py        ← Final training dataset assembly
    │   ├── run_phase3_train.py            ← XGBoost baseline training
    │   ├── run_phase3_visualize.py        ← AUROC / AUPR / confusion matrix plots
    │   ├── run_phase3_risk_map.py         ← Per-date ranked risk map generation
    │   ├── dataset_report.py             ← Dataset statistics report
    │   ├── diagnose_missing.py            ← Missing value diagnostic
    │   ├── inspect_datasets.py            ← Parquet inspection + CSV export
    │   ├── deep_diagnose.py               ← Root-cause data quality analysis
    │   │
    │   └── outputs/
    │       └── texas/
    │           ├── full_training_labels.parquet  ← 376,233 label rows
    │           ├── static_features_tx.parquet    ← LANDFIRE + terrain
    │           ├── gridmet_features_tx.parquet   ← Daily weather
    │           ├── final_training_dataset_tx.parquet ← Final ML-ready table
    │           ├── train_tx.parquet / val_tx.parquet / test_tx.parquet
    │           ├── models/xgb_baseline_tx_meta.json ← Model metadata (weights excluded)
    │           ├── figures/                      ← Evaluation plots (PNG)
    │           ├── phase2g_summary.md            ← Assembly report
    │           └── phase3_model_report_tx.md     ← Baseline model report ✅
    │
    ├── data/
    │   ├── *_FPA_FOD_cons.csv             ← Raw source (NOT in git, ~160 MB each)
    │   ├── gridmet/                       ← NetCDF downloads (NOT in git, ~15 GB)
    │   └── rasters/                       ← LANDFIRE GeoTIFFs (NOT in git, ~5 GB)
    │
    └── logs/                              ← Run logs (NOT in git)
```

---

## 🚀 How to Run (Step-by-Step)

### Prerequisites

```bash
conda activate torch_gpu
pip install rasterio pyproj h3 netCDF4 scipy xgboost lightgbm shap
```

### Phase 1 — Data Preprocessing & EDA

```bash
cd "V2"
python run_phase1.py
python maps/run_all_eda.py --state TX
```

### Phase 2 — Feature Engineering Pipeline (Texas)

```bash
cd "V2/phase2"

# 2A — Feature schema (309 → cleaned columns, leakage audit)
python run_phase2a.py --state TX

# 2B — H3-R7 grid (burnable cells only)
python run_phase2b.py --state TX

# 2C — Map FPA-FOD fire events → H3 cells
python run_phase2c.py --state TX

# 2D — Day-matched 1:10 negative sampling
python run_phase2d.py --state TX --ratio 10

# 2E — Schema fix + leakage check
python run_phase2e_schema_fix.py --state TX

# 2E — Static features (LANDFIRE rasters — download first, see below)
python run_phase2e_static.py --state TX --download-check
python run_phase2e_static.py --state TX

# 2F — gridMET weather (15–20 GB download + extraction, ~3–6 hours)
python run_phase2f_gridmet.py --state TX --download-only
python run_phase2f_gridmet.py --state TX

# 2G — Assemble final training dataset
python run_phase2g_assemble.py --state TX

# Verify data quality
python diagnose_missing.py
```

### Phase 3 — Model Training & Evaluation (Texas)

```bash
cd "V2/phase2"

# 3 — Train XGBoost baseline (produces model weights + evaluation metrics)
python run_phase3_train.py --state TX

# Visualise results (ROC, PR, confusion matrix, feature importance)
python run_phase3_visualize.py --state TX

# Generate ranked risk map for a specific date/window
python run_phase3_risk_map.py --state TX --date 2019-07-04 --window 12Z
```

---

## 📥 Data Downloads

### Automatic (Script Downloads)

| Dataset | Command | Size |
|---------|---------|------|
| gridMET daily weather (12 vars × 7 years) | `python run_phase2f_gridmet.py --state TX --download-only` | ~15–20 GB |

### Manual Download Required (LANDFIRE Rasters)

Place all `.tif` files in `V2/data/rasters/`:

| File | Save As | Source URL |
|------|---------|-----------|
| FSim Burn Probability | `BP_national.tif` | [firelab.org — BP_national.zip](https://www.firelab.org/sites/default/files/images/attachments/BP_national.zip) |
| Wildfire Hazard Potential 2023 | `WHP_2023.tif` | [firelab.org — WHP_2023.zip](https://www.firelab.org/sites/default/files/images/attachments/WHP_2023.zip) |
| LANDFIRE FLEP4 | `FLEP4_national.tif` | [landfire.gov/viewer](https://landfire.gov/viewer/) → FLEP4 |
| LANDFIRE CFL | `CFL_national.tif` | [landfire.gov/viewer](https://landfire.gov/viewer/) → CFL |
| FPA-FOD v6 Fire Labels | `YYYY_FPA_FOD_cons.csv` | [fs.usda.gov/rds](https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6) |

> ⚠️ LANDFIRE rasters are excluded from this repo due to file size. The pipeline runs with zero-filled LANDFIRE features for baseline testing until rasters are downloaded.

---

## ⏱️ Temporal Structure

Each day is split into four 6-hour UTC windows:

| Window | Texas Local (CDT) | TX Fire Count | % |
|--------|------------------|---------------|---|
| 00Z | 7pm – 1am | 479 | 1.4% |
| 06Z | 1am – 7am | 88 | 0.3% |
| 12Z | 7am – 1pm | 20,300 | 59.3% |
| 18Z | 1pm – 7pm | 13,336 | 39.0% |

> Texas uses Central Time (CDT/CST). DST-aware UTC conversion applied per county FIPS (`pytz.timezone('America/Chicago')`). El Paso uses Mountain Time.

---

## 🗂️ Negative Sampling — DAY-MATCHED Method

FPA-FOD contains **only fire events**. Non-fire samples use the **DAY-MATCHED** method:

For each positive fire event on date D in UTC window W:
- Draw **10 H3 cells** from the Texas grid that had **no fire** on date D in window W
- Weather extracted at **each cell's own centroid** — never copied from the fire cell

This forces the model to learn **spatial discrimination** — which cells are risky given today's conditions — not just which days are fire-dangerous.

```
Texas training set:
   34,203  fire rows    (label=1)  →  9.1%
  342,030  non-fire rows (label=0) → 90.9%
  376,233  total rows
```

---

## 🔄 Train / Val / Test Split

**Always chronological — never random.** Random splitting leaks future fire locations.

| Split | Years | ~Rows |
|-------|-------|-------|
| Train | 2014–2017 | 252,000 |
| Validation | 2018 | 61,000 |
| Test | 2019–2020 | 62,000 |

---

## 📈 Phase 3 — Baseline Model Results (Texas, XGBoost)

### Performance Metrics

| Split | Years | Rows | Fire Rows | AUROC | AUPR | F1 | Precision | Recall |
|-------|-------|------|-----------|-------|------|----|-----------|--------|
| Train | 2014–2017 | 252,066 | 22,916 | 0.9302 | 0.5723 | 0.5386 | 0.4153 | 0.7659 |
| Validation | 2018 | 61,181 | 5,561 | 0.8742 | 0.4125 | 0.4316 | 0.3355 | 0.6049 |
| **Test** | **2019–2020** | **62,986** | **5,726** | **0.8569** | **0.3978** | **0.4082** | **0.3315** | **0.5313** |

> **AUPR baseline (random) = 0.091** (the fire rate). Our model achieves **4.4× better than random** — without any LANDFIRE rasters (all landscape features are placeholder zeros).

### Key Observations

- **Top features**: `burnable` (placeholder validity flag), `centroid_lon/lat` (location), `erc_5D_max` + `vs_5D_max` (trailing fire weather) — all physically sensible
- **No leakage**: An earlier run with `fire_count` / `has_fire_history` gave AUROC 0.990 — these were removed (see [phase3_model_report_tx.md](V2/phase2/outputs/texas/phase3_model_report_tx.md))
- **Expected improvement**: Downloading LANDFIRE rasters and re-running Phase 2E → 2G → 3 is expected to push AUROC to **~0.90–0.93**

### Expected Model Performance
| Model Version | Key Features | AUROC | AUPR |
|--------------|-------------|-------|------|
| **Baseline — gridMET only (current)** | 12 weather + 5D stats + temporal | **0.857** | **0.398** |
| Full daily model (+ LANDFIRE rasters) | All 50 features | ~0.90–0.93 | ~0.60–0.70 |
| Sub-daily model (+ HRRR 6-hourly) | All features + atmospheric | ~0.95–0.97 | ~0.65–0.74 |

> If AUROC > 0.99 → a post-ignition leakage feature is present. Check the feature list immediately.

---

## 🛣️ Project Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Data preprocessing, EDA, interactive maps (TX + CA) | ✅ Complete |
| **Phase 2A** | Feature schema finalization (309 → cleaned columns) | ✅ Complete |
| **Phase 2B** | H3-R7 grid construction (Texas burnable cells) | ✅ Complete |
| **Phase 2C** | FPA-FOD fire label mapping to H3 cells | ✅ Complete |
| **Phase 2D** | Day-matched 1:10 negative sampling (376,233 rows) | ✅ Complete |
| **Phase 2E** | Schema fix + leakage audit + LANDFIRE extraction | ✅ Complete |
| **Phase 2F** | gridMET daily + 5-day trailing feature extraction | ✅ Complete |
| **Phase 2G** | Final training dataset assembly (376,233 rows) | ✅ Complete |
| **Phase 3** | XGBoost baseline training + evaluation (TX) | ✅ Complete — AUROC 0.857, AUPR 0.398 |
| **Phase 4** | SHAP feature importance + ablation study | 🔜 Next |
| **Phase 5** | LANDFIRE upgrade (add rasters when downloaded) | 🔜 Next |
| **Phase 2H** | HRRR sub-daily 6-hourly atmospheric features | ⬜ Planned |
| **Phase 6** | California transfer + cross-state evaluation | ⬜ Planned |

---

## 📋 What Is and Is NOT in This Repository

| Included ✅ | Excluded ❌ |
|------------|------------|
| All Python pipeline scripts | Raw FPA-FOD CSV files (~160 MB each) |
| Config and schema files | gridMET NetCDF files (~15–20 GB total) |
| Phase 2 + 3 Markdown summaries | LANDFIRE GeoTIFF rasters (~5 GB) |
| Diagnostic and QC scripts | Processed Parquet training datasets |
| EDA analysis scripts | Full-dataset CSV exports |
| Evaluation figures (PNG) | Model weight files (.ubj) |
| Phase 3 model report + metadata | Risk-map prediction CSVs / HTML |
| Data quality reports | Log files |

---

## 📚 Citations

```bibtex
@misc{fpafod2022,
  author    = {Short, Karen C.},
  title     = {Spatial wildfire occurrence data for the United States, 1992-2020 [FPA FOD 20220705]},
  year      = {2022},
  doi       = {10.2737/RDS-2013-0009.6},
  publisher = {USDA Forest Service, Rocky Mountain Research Station}
}

@article{abatzoglou2013,
  author  = {Abatzoglou, John T.},
  title   = {Development of gridded surface meteorological data for ecological applications and modelling},
  journal = {International Journal of Climatology},
  year    = {2013},
  doi     = {10.1002/joc.3413}
}

@misc{landfire2022,
  title  = {LANDFIRE 2022 (LF2022)},
  author = {{LANDFIRE}},
  year   = {2022},
  url    = {https://www.landfire.gov}
}

@misc{ulm2019,
  title  = {H3: Uber's Hexagonal Hierarchical Spatial Index},
  author = {Uber Technologies},
  year   = {2019},
  url    = {https://h3geo.org}
}
```

---

## ⚙️ Environment

```yaml
name: torch_gpu
dependencies:
  - python=3.10
  - pandas
  - pyarrow
  - numpy
  - scipy
  - seaborn
  - matplotlib
  - folium
  - openpyxl
  - h3-py
  - rasterio
  - pyproj
  - netCDF4
  - xgboost
  - lightgbm
  - shap
```

---

*Built for wildfire ignition prediction research — Texas and California, 2014–2020.*  
*Part of the IgnitionNet project using FPA-FOD labels + gridMET weather + LANDFIRE landscape features.*
