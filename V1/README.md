# 🔥 Texas ML Wildfire — Short-Term Ignition Risk Prediction

> **V1 Proof-of-Concept** · Real NASA FIRMS data · XGBoost + LightGBM + Random Forest · SHAP Explainability · Gradio Dashboard

---

## Table of Contents

1. [Overview](#overview)
2. [Results at a Glance](#results-at-a-glance)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Data Sources](#data-sources)
6. [Pipeline Phases](#pipeline-phases)
7. [Models & Training](#models--training)
8. [Feature Engineering](#feature-engineering)
9. [Evaluation Metrics](#evaluation-metrics)
10. [SHAP Explainability](#shap-explainability)
11. [Gradio Dashboard](#gradio-dashboard)
12. [Spatial Configuration](#spatial-configuration)
13. [Known Limitations (V1)](#known-limitations-v1)
14. [Roadmap](#roadmap)
15. [License](#license)

---

## Overview

This project predicts the **probability of wildfire ignition** in **Central/East Texas** using real fire detection records from NASA FIRMS (2024). It trains and compares multiple gradient-boosted tree classifiers, applies SHAP-based explainability to confirm the model has learned real fire physics, and exposes predictions through an interactive **Gradio dashboard**.

**What this is:**
- A fully working end-to-end ML pipeline on real fire data
- A multi-model comparison framework (XGBoost, LightGBM, Random Forest)
- An explainability-first approach using TreeSHAP

**What this is NOT (yet):**
- A true operational 24-hour forecaster (see [Known Limitations](#known-limitations-v1))
- A real-time prediction system (no live API feeds)

---

## Results at a Glance

Best model: **LightGBM (Model E)** at decision threshold `0.46`

| Metric | Validation | **Test** | Assessment |
|--------|-----------|----------|------------|
| AUC-ROC | 0.8994 | **0.9142** | 🟢 Excellent |
| AUC-PR | 0.6905 | **0.7549** | 🟡 Good |
| F1-Score | 0.6915 | **0.7236** | 🟢 Good |
| Recall | 0.8516 | **0.8744** | 🟢 87% of fires caught |
| Precision | 0.5821 | **0.6173** | 🟡 ~1 in 3 alarms is false |

**Confusion Matrix (Test: Oct–Dec 2024)**

```
                  Predicted No Fire   Predicted Fire
Actual No Fire        1354 (TN)          315 (FP)
Actual Fire             73 (FN)          508 (TP)
```

- ✅ **508 / 581 fires caught** (87.4% recall)
- ⚠️ **73 missed fires** — the dangerous edge cases
- 🟡 **315 false alarms** — roughly 1 false alarm per 1.7 real fires

> **Note on the "overfit" flag:** The pipeline flags the test AUC-PR gap as overfitting, but test performance is actually *higher* than validation. This is **temporal distribution shift** — Oct–Nov is peak fire season in Texas, so the test set naturally has stronger signal. A truly overfit model would score lower on the test set, not higher.

---

## Project Structure

```
Texas ML Wildfire/
├── configs/
│   └── config.yaml                  # All pipeline configuration parameters
├── data/
│   ├── raw/                         # Real FIRMS CSV + downloaded rasters
│   ├── interim/                     # Reprojected intermediate files
│   ├── processed/                   # Final feature table (Parquet)
│   └── external/                    # Static reference data
├── models/
│   ├── xgb_model.json               # XGBoost native weights
│   ├── xgb_model.pkl                # Pickle wrapper for inference
│   ├── scaler.pkl                   # Fitted StandardScaler
│   ├── imputer.pkl                  # Fitted median imputer
│   ├── features.json                # Ordered feature list used at training
│   └── optimal_threshold.json       # Validation-tuned decision threshold
├── outputs/
│   ├── evaluation_report.json       # Full metrics for all model variants
│   ├── roc_curve_test.png
│   ├── pr_curve_test.png
│   ├── confusion_matrix_test.png
│   ├── shap_summary_plot.png
│   ├── shap_local_high_risk.png
│   ├── shap_local_low_risk.png
│   └── shap_local_explanations.json
├── logs/
│   └── wildfire_pipeline.log
├── src/
│   ├── ingestion/                   # Raw data acquisition & simulation
│   ├── harmonization/               # Geospatial reprojection & alignment
│   ├── feature_engineering/         # FFWI, HDW, terrain lags, land cover
│   ├── dataset_builder/             # V1 (annual) & V2 (daily) dataset logic
│   ├── preprocessing/               # Chronological split, scaling, imbalance
│   ├── training/                    # XGBoost / LightGBM / RF training + CV
│   ├── evaluation/                  # Metrics, ROC/PR curves, threshold sweep
│   ├── inference/                   # Prediction helpers for the Gradio app
│   ├── visualization/               # Map and time-series plot utilities
│   ├── pipelines/                   # Phase orchestrators
│   └── utils/                       # Logging, config helpers
├── notebooks/                       # EDA and exploration notebooks
├── fire_archive_M-C61_760762.csv    # Raw NASA FIRMS fire archive (2024)
├── main.py                          # CLI entrypoint
├── app_gradio.py                    # Interactive Gradio dashboard
├── NDVI.PY                          # NDVI raster utility
├── firms_2024.py                    # FIRMS data download helper
├── reorganize_data.py               # Data directory restructure script
├── requirements.txt
└── environment.yml
```

---

## Quick Start

### 1. Create and Activate the Conda Environment

```bash
conda env create -f environment.yml
conda activate wildfire-risk
```

### 2. Install Dependencies (pip alternative)

```bash
pip install -r requirements.txt
```

> **GPU acceleration:** If you have an NVIDIA GPU, XGBoost will automatically detect CUDA and set `device="cuda"`. This gives ~3–5× speedup over CPU. No additional config is needed.

### 3. Run the Full Pipeline

```bash
python main.py --config configs/config.yaml --phase all
```

This runs all pipeline phases in order (see [Pipeline Phases](#pipeline-phases)).

### 4. Run Individual Phases

```bash
# Data ingestion only (simulates or downloads raw data)
python main.py --phase ingest

# Training only (requires phases 1–4 to be complete)
python main.py --phase train

# Evaluation only (requires training to be complete)
python main.py --phase evaluate

# SHAP explainability only
python main.py --phase explain
```

### 5. Use Real Data (Skip Simulation)

The pipeline ships with `fire_archive_M-C61_760762.csv` (NASA FIRMS 2024). To use it instead of the built-in simulator:

```bash
python main.py --phase ingest --no-simulate
```

Place any additional rasters in `data/raw/` following the naming conventions in `configs/config.yaml`, then run subsequent phases normally.

### 6. Launch the Gradio Dashboard

```bash
python app_gradio.py
```

Opens an interactive UI for uploading data, running predictions, and exploring SHAP explanations.

---

## Data Sources

| Dataset | Variable | Format | Source |
|---------|----------|--------|--------|
| **NASA FIRMS / VIIRS** | Active fire points, FRP, confidence | CSV | [firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov) |
| NOAA NDFD | Temperature, RH, Wind Speed, Precip Prob | GeoTIFF | graphical.weather.gov |
| MODIS MOD13A2 | NDVI, EVI | GeoTIFF | NASA LP DAAC / AppEEARS |
| MODIS MOD11A1 | Land Surface Temperature (LST) | GeoTIFF | NASA LP DAAC |
| SRTM | Digital Elevation Model (DEM) | GeoTIFF | USGS EarthExplorer |
| NLCD | Land Cover Classification | GeoTIFF | USGS / MRLC |
| USDM / PDSI | Drought Severity | CSV | droughtmonitor.unl.edu |
| OSM | Roads, Powerlines | GeoJSON | Overpass API |
| NIFC / Texas A&M | Historical Fire Perimeters | GeoJSON | data-nifc.opendata.arcgis.com |

> **Simulation mode (default):** All raster datasets are simulated by `src/ingestion/simulator.py` with physically consistent spatial and temporal patterns. The pipeline runs fully offline with **zero API keys** required. Switch to `--no-simulate` to use the real FIRMS CSV included in the repo.

---

## Pipeline Phases

| Phase | CLI Flag | Description |
|-------|----------|-------------|
| **Ingest** | `ingest` | Simulate rasters or load real FIRMS CSV; outputs to `data/raw/` |
| **Harmonize** | `harmonize` | Reproject, align, cloud-fill, and stack all rasters to a common grid |
| **Features** | `features` | Compute FFWI, HDW, terrain features, 3-day lag windows, land cover |
| **Prepare** | `prepare` | Chronological train/val/test split + StandardScaler + median imputation |
| **Train** | `train` | XGBoost + LightGBM + Random Forest; TimeSeriesSplit CV + early stopping |
| **Evaluate** | `evaluate` | ROC-AUC, PR-AUC, F1, confusion matrix, threshold optimisation |
| **Explain** | `explain` | TreeSHAP global beeswarm + local waterfall plots |

---

## Models & Training

Four model variants were trained and compared:

| Model | Type | AUC-ROC | AUC-PR | Stopped At |
|-------|------|---------|--------|------------|
| **A** | XGBoost (full features) | 0.8868 | 0.6691 | 195 / 600 trees |
| **B** | XGBoost (ablation: no `is_peak_fire_season`) | — | — | 456 / 600 trees |
| **C** | Random Forest | — | — | 500 / 500 (full) |
| **E** ✅ | **LightGBM** | **0.8994** | **0.6905** | 206 / 600 trees |

**Why LightGBM won:**
- Native categorical splits for `LandCover` (vs integer encoding in XGBoost)
- Leaf-wise tree growth better suits this heterogeneous feature mix
- `is_unbalance=True` offers more stable gradient reweighting for the 1:3 imbalance

**Training split (chronological — no data leakage):**

| Split | Period | Purpose |
|-------|--------|---------|
| Train | Jan – Aug 2024 | Model fitting |
| Validation | Sep 2024 | Threshold tuning, early stopping |
| Test | Oct – Dec 2024 | Final held-out evaluation |

**Class imbalance handling:**
1. `scale_pos_weight` (XGBoost) / `is_unbalance=True` (LightGBM) — loss reweighting
2. **Threshold sweep on validation PR curve** — finds threshold maximising F1 (found: `0.46`)
3. **Temporal exclusion** — cells already actively burning on day `t` excluded to prevent label leakage from fire propagation

---

## Feature Engineering

### Derived Fire Weather Indices

**Fosberg Fire Weather Index (FFWI)**
Uses equilibrium fuel moisture content (EMC) derived from temperature and relative humidity, combined with wind speed to produce a unitless 0–100 index representing fire spread potential.

**Hot-Dry-Windy Index (HDW)**
```
HDW = Wind Speed (m/s) × Vapor Pressure Deficit (kPa)
```
Simple but highly effective indicator of extreme fire weather conditions.

### Temporal Lag Features

| Feature | Window |
|---------|--------|
| `temp_max_3d` | 3-day rolling maximum temperature |
| `rh_min_3d` | 3-day rolling minimum relative humidity |
| `wind_mean_3d` | 3-day rolling mean wind speed |

### Terrain Features (from SRTM DEM)

| Feature | Description |
|---------|-------------|
| `slope` | Slope in degrees (DEM gradient magnitude) |
| `aspect_sin`, `aspect_cos` | Circular encoding of slope aspect direction |
| `tri` | Riley's Terrain Ruggedness Index (8-neighbour variance) |

### SHAP-Confirmed Top Features

The model learned **real fire physics** — not noise:

```
1. Temperature       0.10583  ✅ Hot conditions → fire risk
2. Wind Speed        0.08204  ✅ Wind drives fire spread
3. Rainfall          0.06085  ✅ Wet conditions → low risk
4. DEM (Elevation)   0.04556  ✅ Terrain affects fire behaviour
5. LST               0.01613  ✅ Surface heat → ignition risk
6. EVI               0.01360  ✅ Vegetation fuel load
7. NDVI              0.01303  ✅ Vegetation moisture proxy
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **ROC-AUC** | Rank discrimination across all thresholds |
| **PR-AUC** | Precision-Recall area; critical for rare-event (imbalanced) data |
| **Precision** | Of predicted fires, how many are actually fires |
| **Recall** | Of actual fires, how many are detected |
| **F1-Score** | Harmonic mean; balances precision and recall |
| **Confusion Matrix** | TP / FP / TN / FN breakdown |

All outputs are saved to `outputs/evaluation_report.json`.

---

## SHAP Explainability

Global and local TreeSHAP analyses are automatically saved to `outputs/`:

| File | Content |
|------|---------|
| `shap_summary_plot.png` | Global beeswarm plot of feature importances across all test samples |
| `shap_local_high_risk.png` | Waterfall diagram for the highest-risk prediction |
| `shap_local_low_risk.png` | Waterfall diagram for the lowest-risk prediction |
| `shap_local_explanations.json` | Top-5 contributing features for both extreme cases |

---

## Gradio Dashboard

`app_gradio.py` provides an interactive web UI:

- Upload fire detection data or use the built-in sample
- Run predictions and see risk probability outputs
- Explore SHAP explanations visually
- Adjust decision threshold interactively

```bash
python app_gradio.py
# Opens at http://127.0.0.1:7860
```

---

## Spatial Configuration

The default grid targets **Central/East Texas** in UTM Zone 14N (EPSG:26914):

```yaml
# configs/config.yaml
min_x: 600000.0    # Easting (metres)
max_x: 650000.0
min_y: 3300000.0   # Northing (metres)
max_y: 3350000.0
resolution: 500.0  # Grid cell size in metres
```

Simulation mode generates a **100 × 100 grid** = 10,000 spatial cells × ~90 days ≈ 900,000 rows.
The real FIRMS pipeline uses actual detection coordinates and sampled negatives, resulting in **10,000–50,000 rows** — which explains why training completes in ~6 minutes (especially with GPU acceleration).

---

## Known Limitations (V1)

> These are documented, understood limitations — not surprises.

### 🔴 Critical: V1 is NOT a True 24-Hour Forecaster

**Current labeling approach (V1):**
```
Row: lat=30.1, lon=-97.5, acq_date=2024-03-15
Features: annual composite NDVI, EVI, LST + date encodings
Label: Fire=1 (FIRMS detected fire here on Mar 15)
```

**The problem:** Features and labels are from the **same day**. Because annual composites don't change day-to-day, the model primarily learns *which locations and months historically have fires*, not *whether current conditions are dangerous*.

**What true 24-hour forecasting requires (V2):**
```
Row: lat=30.1, lon=-97.5, date=2024-03-14   ← YESTERDAY's features
Features: NDVI on Mar 14, Temp/Wind/RH on Mar 14
Label: Fire=1 if FIRMS detects fire here on 2024-03-15   ← TOMORROW
```

This requires date-specific raster queries (e.g., via Google Earth Engine), planned for V2.

### Other Known Issues

| Issue | Severity | Planned Fix |
|-------|----------|-------------|
| Annual composites instead of daily rasters | 🔴 High | V2: GEE date-specific queries |
| No true T+1 label offset | 🔴 High | Label = fire on day T+1, features = day T |
| Simulated coordinates not validated against Texas polygon | 🟡 Medium | Add Shapely boundary check |
| AUC-PR gap misidentified as overfit in evaluation report | 🟡 Medium | Fix gap threshold logic in `evaluation/` |
| LightGBM no GPU support in default build | 🟢 Low | Install LightGBM GPU build if needed |

---

## Roadmap

```
Current: V1 POC
  → Real FIRMS data, annual composites, same-day labels
  → AUC-ROC 0.91, Recall 87%

V2: Daily Feature Rasters
  → GEE API queries per date (NDVI/LST/Weather for date T)
  → True 24-hr label shift: predict fire on T+1 from T features
  → Proper 24-hour ignition forecasting

V3: REST API
  → FastAPI endpoint serving daily predictions as JSON
  → Structured GeoJSON output with risk polygons

V4: Interactive Dashboard
  → Leaflet / Deck.gl interactive risk map
  → Time-series animation of fire risk

V5: Operational Deployment
  → Containerised pipeline (Docker)
  → Live NOAA/NASA data feeds
  → Scheduled daily inference
```

---

## License

MIT License — For research and educational purposes.

---

*Built with: XGBoost · LightGBM · scikit-learn · SHAP · Gradio · GeoPandas · Rasterio · NASA FIRMS data*
