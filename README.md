# 🔥 IgnitionNet — Wildfire Ignition Prediction System

> **Predicting WHERE and WHEN wildfires ignite** using 7 years of fire occurrence data, gridMET daily weather, LANDFIRE landscape features, and H3 hexagonal grid cells — before a fire is confirmed.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Phase-2%20In%20Progress-yellow.svg)]()

---

## 📌 What This Project Does

IgnitionNet answers one specific operational question:

> **Given current landscape and weather conditions, which hexagonal cells in Texas (or California) are most likely to have a wildfire discovered in the next 6 hours?**

The model outputs a **ranked risk map** of ~1.17 million H3 hexagonal cells, updated every 6 hours, that fire managers can use to **pre-position suppression resources before a fire starts**.

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
H3 Resolution-8 Hexagonal Grid
  Texas:      1,172,643 burnable cells (~860 m across each)
  California: ~1,470,000 burnable cells
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
| H3 Resolution | 8 (~860 m cell width, ~0.74 km² area) |
| Total H3 burnable cells (TX) | **1,172,643** |
| Fire events used (TX, ≥1 acre) | **34,203** |
| Non-fire samples (1:10 ratio) | **342,030** |
| Total training rows | **376,233** |
| Positive rate | 9.1% |
| Temporal windows | 4 per day (00Z, 06Z, 12Z, 18Z UTC) |
| Train years | 2014–2017 |
| Validation year | 2018 |
| Test years | 2019–2020 |

---

## 🧪 Feature Set

All features are extracted **independently** for each cell at its own centroid location. Non-fire cells do **not** inherit the paired fire cell's weather values.

### Group 1 — Landscape Features (Static, LANDFIRE / FSim)

| Feature | Source | Description | Cohen's d |
|---------|--------|-------------|-----------|
| `avg_burn_prob` | USFS FSim | Long-run burn probability from 50,000 stochastic simulations | **> 1.5** |
| `whp` | USFS WHP 2023 | Wildfire Hazard Potential index (0–7000) | > 0.7 |
| `flep4` | LANDFIRE LF2022 | Flame Length Exceedance Probability at 4 ft | **> 1.0** |
| `cfl` | LANDFIRE LF2022 | Canopy Fuel Load (Mg ha⁻¹) | > 0.8 |

### Group 2 — Daily Fire Weather (gridMET, 4 km daily CONUS)

| Feature | gridMET Variable | Units | Description |
|---------|-----------------|-------|-------------|
| `erc` | `energy_release_component-g` | BTU ft⁻² | Energy Release Component — best single daily predictor |
| `fm100` | `dead_fuel_moisture_100hr` | % | 100-hr dead fuel moisture |
| `fm100_5D_mean` | derived | % | 5-day trailing mean of fm100 |
| `vpd` | `mean_vapor_pressure_deficit` | kPa | Vapor pressure deficit |
| `vs` | `wind_speed` | m s⁻¹ | Daily wind speed |
| `bi` | `burning_index` | index | Burning Index (NFDRS) |
| `rmax` / `rmin` | `max/min_relative_humidity` | % | Relative humidity bounds |
| `tmmx` / `tmmn` | `air_temperature` | °C | Max/min temperature |
| `pr` | `precipitation_amount` | mm | Daily precipitation |
| `sph` | `specific_humidity` | kg kg⁻¹ | Specific humidity |

### Group 3 — Temporal Encodings (Computed from timestamp)

| Feature | Formula | Purpose |
|---------|---------|---------|
| `sin_month`, `cos_month` | sin/cos(2π × month/12) | Cyclic fire season encoding |
| `sin_hour`, `cos_hour` | sin/cos(2π × hour/24) | Cyclic 6-hour window encoding |

### Group 4 — Location (From H3 cell centroid)

| Feature | Description |
|---------|-------------|
| `centroid_lat` | H3-8 cell centroid latitude |
| `centroid_lon` | H3-8 cell centroid longitude |

### Phase 2H — HRRR Sub-daily Features *(Added after daily baseline)*

| Feature | Source | Description |
|---------|--------|-------------|
| `rh_pw` | NOAA HRRR | 2-m relative humidity at fire's 6-hour window |
| `temp_pw` | NOAA HRRR | 2-m temperature |
| `wind_speed_pw` | NOAA HRRR | 10-m wind speed (√U²+V²) |
| `vpd_pw` | NOAA HRRR | VPD derived from TMP + RH |
| `hpbl_pw` | NOAA HRRR | Planetary boundary layer height |
| `dswrf_pw` | NOAA HRRR | Downwelling solar radiation |

---

## 📁 Repository Structure

```
Texas_Wildfire_ML/
│
├── README.md                              ← This file
├── IGNITIONNET_PROJECT_SCOPE_MASTER.md    ← Full project specification
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
    │   ├── run_all_eda.py
    │   ├── texas/scripts/                 ← 7 EDA scripts
    │   ├── texas/eda_outputs/             ← PNGs, CSVs, HTML map
    │   ├── california/scripts/
    │   └── california/eda_outputs/
    │
    ├── phase2/                            ← Feature Engineering Pipeline
    │   ├── config/phase2_config.py        ← State configs, paths
    │   ├── run_phase2a.py                 ← Feature schema finalization
    │   ├── run_phase2b.py                 ← H3 grid construction
    │   ├── run_phase2c.py                 ← FPA-FOD → H3 fire label mapping
    │   ├── run_phase2d.py                 ← DAY-MATCHED 1:10 negative sampling
    │   ├── run_phase2e_schema_fix.py      ← Schema cleanup + leakage check
    │   ├── run_phase2e_static.py          ← LANDFIRE raster extraction
    │   ├── run_phase2f_gridmet.py         ← gridMET NetCDF download + extract
    │   ├── run_phase2g_assemble.py        ← Final dataset assembly
    │   └── outputs/
    │       ├── texas/                     ← Phase 2 outputs (Texas)
    │       │   ├── h3_grid_tx.parquet     ← 1,172,643 cell grid
    │       │   ├── full_training_labels.parquet   ← 376,233 rows
    │       │   ├── cleaned_feature_schema.csv
    │       │   ├── schema_fix_report.md
    │       │   ├── phase2c_summary.csv
    │       │   └── phase2d_summary.csv
    │       └── california/
    │
    ├── data/
    │   ├── *_FPA_FOD_cons.csv             ← Raw source (NOT in git, ~160 MB each)
    │   ├── gridmet/                       ← NetCDF downloads (NOT in git, ~15 GB)
    │   └── rasters/                       ← LANDFIRE GeoTIFFs (NOT in git, ~5 GB)
    │
    └── logs/                              ← Run logs (NOT in git)
```

---

## 🚀 Quick Start

### Prerequisites

```bash
conda activate torch_gpu

# Phase 2 dependencies
pip install rasterio pyproj h3 netCDF4 scipy xgboost lightgbm shap
```

### Phase 1 — Data Preprocessing & EDA

```bash
cd "V2"
python run_phase1.py

# EDA + Interactive Maps
python maps/run_all_eda.py --state TX
python maps/run_all_eda.py --state CA
```

### Phase 2 — Feature Engineering Pipeline

```bash
cd "V2/phase2"

# Step 1: Build feature schema (runs automatically)
python run_phase2a.py --state TX

# Step 2: Build H3 grid (1,172,643 burnable cells for Texas)
python run_phase2b.py --state TX

# Step 3: Map FPA-FOD fires to H3 cells
python run_phase2c.py --state TX

# Step 4: Generate 1:10 DAY-MATCHED negative samples (~34 min)
python run_phase2d.py --state TX --ratio 10

# Step 5: Fix schema + leakage check
python run_phase2e_schema_fix.py --state TX

# Step 6: Download LANDFIRE rasters (manual — see Data Downloads section)
# Then extract static features
python run_phase2e_static.py --state TX --download-check
python run_phase2e_static.py --state TX

# Step 7: Download + extract gridMET (15–20 GB, 1–3 hours)
python run_phase2f_gridmet.py --state TX --download-only
python run_phase2f_gridmet.py --state TX

# Step 8: Assemble final training dataset
python run_phase2g_assemble.py --state TX
```

### Phase 3 — Model Training *(Coming Soon)*

```bash
cd "V2/phase2"
python run_phase3_train.py --state TX
```

---

## 📥 Data Downloads

### Automatic (Script Downloads)

| Dataset | Script | Size | URL |
|---------|--------|------|-----|
| gridMET daily weather | `run_phase2f_gridmet.py --download-only` | ~15–20 GB | [climatologylab.org](https://www.climatologylab.org/gridmet.html) |
| NOAA HRRR (Phase 2H) | `herbie` Python library | via AWS S3 | Public, no auth needed |

### Manual Download Required

| File | Save As | Source |
|------|---------|--------|
| FSim Burn Probability | `V2/data/rasters/BP_national.tif` | [firelab.org](https://www.firelab.org) |
| Wildfire Hazard Potential 2023 | `V2/data/rasters/WHP_2023.tif` | [firelab.org](https://www.firelab.org) |
| LANDFIRE FLEP4 | `V2/data/rasters/FLEP4_national.tif` | [landfire.gov/viewer](https://landfire.gov/viewer/) |
| LANDFIRE CFL | `V2/data/rasters/CFL_national.tif` | [landfire.gov/viewer](https://landfire.gov/viewer/) |
| FPA-FOD v6 (Labels) | `V2/data/YYYY_FPA_FOD_cons.csv` | [fs.usda.gov/rds](https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6) |

> ⚠️ LANDFIRE rasters and raw FPA-FOD CSVs are excluded from this repository due to file size. The pipeline runs with zero-filled LANDFIRE features for baseline testing.

---

## 📈 Expected Model Performance

| Model Version | Features | AUROC | AUPR | Precision@Top59 |
|--------------|---------|-------|------|-----------------|
| Baseline (gridMET only) | 12 gridMET + temporal + location | ~0.87–0.90 | ~0.55–0.63 | ~75% |
| Full daily model (+ LANDFIRE) | All 16 features | ~0.93–0.96 | ~0.60–0.70 | ~89% |
| Sub-daily model (+ HRRR) | All 22 features | ~0.95–0.97 | ~0.65–0.74 | ~92% |

> If AUROC exceeds 0.99 → a post-ignition leakage feature is present. Stop and check the feature list.

---

## 🗂️ Negative Sampling — DAY-MATCHED Method

FPA-FOD contains **only fire events**. Non-fire samples are generated using the **DAY-MATCHED** method:

For each positive fire event on date D in UTC window W:
- Draw **10 H3 cells** from the Texas grid that had **no fire** on date D in window W
- These 10 cells become negative samples (label = 0)
- Each cell gets weather extracted **at its own location** — not copied from the fire cell

This design forces the model to learn **spatial discrimination** — which cells are high-risk given today's conditions — not just which days are fire-dangerous.

```
Texas training set:
  34,203  fire rows    (label=1)  →  9.1%
  342,030 non-fire rows (label=0) → 90.9%
  376,233 total rows
```

---

## ⏱️ Temporal Structure

Each day is split into four 6-hour UTC windows:

| Window | Texas Local Time (CDT) | TX Fire Count | % |
|--------|----------------------|---------------|---|
| 00Z | 7pm – 1am | 479 | 1.4% |
| 06Z | 1am – 7am | 88 | 0.3% |
| 12Z | 7am – 1pm | 20,300 | 59.3% |
| 18Z | 1pm – 7pm | 13,336 | 39.0% |

> Texas uses Central Time (CDT/CST). DST-aware UTC conversion is applied using `pytz.timezone('America/Chicago')` per county FIPS. El Paso area uses Mountain Time.

---

## 🔬 Train / Validation / Test Split

**Always chronological — never random.** Random splitting leaks future fire locations into training data.

| Split | Years | Rows | Role |
|-------|-------|------|------|
| Train | 2014–2017 | ~246,150 | Model learning |
| Validation | 2018 | ~61,181 | Early stopping |
| Test | 2019–2020 | ~68,902 | Final evaluation only |

---

## 🛣️ Project Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Data preprocessing, EDA, interactive maps (TX + CA) | ✅ Complete |
| **Phase 2A** | Feature schema finalization (309 → 235 columns) | ✅ Complete |
| **Phase 2B** | H3 Resolution-8 grid construction (1.17M cells, TX) | ✅ Complete |
| **Phase 2C** | FPA-FOD fire label mapping to H3 cells | ✅ Complete |
| **Phase 2D** | DAY-MATCHED 1:10 negative sampling (376,233 rows) | ✅ Complete |
| **Phase 2E** | LANDFIRE static feature extraction | ⏳ Pending rasters |
| **Phase 2F** | gridMET daily weather extraction | 🔄 Downloading |
| **Phase 2G** | Final training dataset assembly | 🔜 Next |
| **Phase 3** | XGBoost baseline model training + evaluation | 🔜 Next |
| **Phase 4** | SHAP feature ablation + LOO analysis | ⬜ Planned |
| **Phase 5** | LANDFIRE model upgrade (AUROC ~0.87 → ~0.96) | ⬜ Planned |
| **Phase 2H** | HRRR sub-daily extension (6-hourly atmospheric) | ⬜ Planned |
| **Phase 6** | California transfer + cross-state evaluation | ⬜ Planned |

---

## 📚 Citations

```bibtex
@misc{fpafod2022,
  author  = {Short, Karen C.},
  title   = {Spatial wildfire occurrence data for the United States, 1992-2020 [FPA FOD 20220705]},
  year    = {2022},
  doi     = {10.2737/RDS-2013-0009.6},
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
  title   = {LANDFIRE 2022 (LF2022)},
  author  = {{LANDFIRE}},
  year    = {2022},
  url     = {https://www.landfire.gov}
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

## 📋 What Is and Is NOT in This Repository

| Included ✅ | Excluded ❌ |
|------------|------------|
| All Python pipeline scripts | Raw FPA-FOD CSV files (~160 MB each) |
| Config and schema files | gridMET NetCDF files (~15–20 GB total) |
| Phase 2 output CSVs and reports | LANDFIRE GeoTIFF rasters (~5 GB) |
| Phase 2 Markdown summaries | Processed Parquet training datasets |
| EDA scripts and analysis | Model weight files |
| Interactive map scripts | Log files |

---

*Built for wildfire ignition prediction research — Texas and California, 2014–2020.*  
*Part of the IgnitionNet project using FPA-FOD labels + gridMET weather + LANDFIRE landscape features.*
