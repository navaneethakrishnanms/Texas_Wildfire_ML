# 🔥 Texas Wildfire Ignition Prediction System

> **Phase 1 complete:** Historical data preprocessing, EDA, and interactive hotspot maps for Texas and California (2014–2020).

---

## Project Overview

A production-grade machine learning pipeline that processes 7 years of US wildfire records (FPA-FOD dataset) into clean, analysis-ready datasets for Texas and California, then generates full exploratory data analysis reports and interactive Folium hotspot maps.

| State | Fire Records | Years | Features |
|-------|-------------|-------|---------|
| Texas | 51,033 | 2014–2020 | 309 |
| California | 50,881 | 2014–2020 | 309 |

---

## Repository Structure

```
Texas_Wildfire_ML/
├── V1/                                ← Version 1 (initial exploration)
└── V2/                                ← Version 2 (production pipeline)
    ├── run_phase1.py                  ← Phase 1 entry-point
    ├── README.md                      ← Detailed V2 documentation
    │
    ├── src/preprocessing/             ← Core pipeline modules
    │   ├── config.py                  ← All constants and paths
    │   ├── logger.py                  ← ASCII-safe logging
    │   ├── loader.py                  ← Load + standardise columns
    │   ├── schema_checker.py          ← Schema verification
    │   ├── merger.py                  ← Merge years + quality summary
    │   ├── state_filter.py            ← State split + save
    │   ├── validator.py               ← Data validation (no rows removed)
    │   ├── quality_reporter.py        ← Quality report (MD + CSV)
    │   ├── eda_reporter.py            ← EDA report (MD + correlation CSV)
    │   └── pipeline.py                ← Orchestrator
    │
    ├── maps/
    │   ├── run_all_eda.py             ← Master EDA runner (TX + CA)
    │   ├── texas/scripts/             ← 7 EDA scripts for Texas
    │   ├── texas/eda_outputs/         ← Generated PNGs, CSVs, TXT, HTML map
    │   ├── california/scripts/        ← 7 EDA scripts for California
    │   └── california/eda_outputs/    ← Generated PNGs, CSVs, TXT, HTML map
    │
    ├── reports/
    │   ├── texas/                     ← Phase 1 pipeline reports
    │   └── california/
    │
    ├── data/
    │   ├── *.csv                      ← Raw FPA-FOD yearly files (NOT in git)
    │   └── processed/                 ← Processed Parquet/CSV files (NOT in git)
    │
    └── logs/                          ← Pipeline run logs (NOT in git)
```

---

## Quick Start

### Prerequisites

```bash
conda activate torch_gpu
# Required: pandas, pyarrow, numpy, seaborn, matplotlib, scipy, folium
```

### Phase 1 — Data Preprocessing

```bash
cd "V2"

# Run the full Phase 1 pipeline (creates processed/ + reports/)
conda run -n torch_gpu --no-capture-output python run_phase1.py
```

Place raw data files in `V2/data/` with naming: `YYYY_FPA_FOD_cons.csv`

### EDA + Interactive Maps

```bash
# Run all 7 EDA scripts for both Texas and California (~6 min)
conda run -n torch_gpu --no-capture-output python maps/run_all_eda.py

# Run for one state only
conda run -n torch_gpu --no-capture-output python maps/run_all_eda.py --state TX
conda run -n torch_gpu --no-capture-output python maps/run_all_eda.py --state CA

# Run one specific script
conda run -n torch_gpu --no-capture-output python maps/run_all_eda.py --script 07
```

### View the Interactive Maps

```bash
# Texas
cmd /c start "" "V2\maps\texas\eda_outputs\wildfire_hotspot_map.html"

# California
cmd /c start "" "V2\maps\california\eda_outputs\wildfire_hotspot_map.html"
```

---

## EDA Scripts

| Script | Output | Description |
|--------|--------|-------------|
| `01_data_overview.py` | 3 PNGs + 3 CSVs/TXT | Column info, missing values heatmap, fire size class distribution |
| `02_distributions.py` | 5 PNGs | Weather/terrain histograms split by fire size, box plots, violin plots |
| `03_correlation_analysis.py` | 3 PNGs + 1 CSV | Pearson correlation heatmaps, feature vs FIRE_SIZE bar chart |
| `04_geospatial_temporal.py` | 7 PNGs | Lat/lon scatter maps, hexbin density, fires per year/month/heatmap |
| `05_advanced_eda.py` | 6 PNGs + 1 CSV | Pair plots, outlier IQR, skewness/kurtosis, Cohen's d |
| `06_summary_report.py` | 1 PNG + 1 TXT | 28×20 executive dashboard + full analyst text report |
| `07_interactive_hotspot_map.py` | 1 HTML | Folium map with 4 layer types, cluster markers, ERC hotspots |

---

## Interactive Map Features

The Folium hotspot map (`wildfire_hotspot_map.html`) includes:

- **Dark CartoDB base** with 4 tile switchers (Dark / Street / Light / Satellite)
- **Fire Density HeatMap** — blue→teal→orange→red heat cloud
- **MarkerCluster layers** per fire cause (Lightning / Human / Equipment / Debris)
- **High-Risk Hotspots** — top 200 fires by ERC, sized and colored by risk level
- **Lightning-only layer** — filtered subset
- **HTML panels** — title banner, dataset statistics, cause legend
- **Plugins** — Fullscreen, MiniMap, LayerControl

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1–2 | `loader.py` | Read all yearly CSVs, standardise column names |
| 3 | `schema_checker.py` | Verify schema consistency across 7 years |
| 4 | `merger.py` | Merge into master DataFrame, print quality summary |
| 5 | `state_filter.py` | Filter by state, save Parquet + CSV |
| – | `validator.py` | Validate lat/lon/date/size — report but never remove |
| – | `quality_reporter.py` | Per-state quality report (MD + missing CSV) |
| – | `eda_reporter.py` | Full EDA report (MD + correlation CSV) |

---

## Data Source

**FPA-FOD (Fire Program Analysis Fire-Occurrence Database)**  
USDA Forest Service — 2014 to 2020 yearly records for the entire United States.

> Raw data files are excluded from this repository (each file is 140–190 MB).  
> Download from: https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009

---

## Environment

```yaml
name: torch_gpu
dependencies:
  - python=3.10
  - pandas
  - pyarrow
  - numpy
  - seaborn
  - matplotlib
  - scipy
  - folium
  - openpyxl
```

---

## Roadmap

- [x] Phase 1 — Data Preprocessing & EDA
- [ ] Phase 2 — Feature Engineering & Model Training
- [ ] Phase 3 — Model Evaluation & Deployment
