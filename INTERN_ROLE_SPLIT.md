# Intern Role Split — Texas Wildfire Ignition Prediction (IgnitionNet / TDIS BIT)

Two roles cover the whole project end to end. Neither role is a sub-part of the other: each one owns complete phases from start to finish.

| | **Role 1** | **Role 2** |
|---|---|---|
| **Title** | **ML Engineer & MLOps / Deployment Lead** | **Data Engineer & Full-Stack (Frontend) Developer** |
| **One-line scope** | Everything from a clean training table to a trained, validated, served and deployed model | Everything from raw public data to the clean training table, plus the live data feed and the web portal users see |
| **Owns** | Model training, tuning, calibration, validation, model serving API, containers, cloud deployment, CI/CD | Data ingestion, labels, feature engineering, imputation, data quality, daily feed, React map portal |
| **Hand-off point** | Receives `bit_train_tx.parquet` (training table) and the daily feature feed | Receives the model file + API contract (`/api/v1/...`) and renders the output |

---

## 1. Whole project at a glance

```
RAW DATA                 DATA ENGINEERING            MODEL / MLOPS                 APP
VIIRS, FPA-FOD labels    clean, join, impute,        train, tune, calibrate,       map portal
HRRR, LANDFIRE, USDM  →  feature engineer,        →  validate, serve, containerize → React + MapLibre
roads, powerlines,       H3 grid, daily feed         deploy to Azure, CI/CD        (risk cells, fires)
population               ──── Role 2 ────            ──────── Role 1 ────────      ── Role 2 ──
```

---

## 2. Phase-by-phase ownership

| # | Project phase | Folders / files | **Role 1** | **Role 2** |
|---|---|---|---|---|
| 1 | Data sourcing (FPA-FOD, VIIRS, gridMET, HRRR, LANDFIRE, USDM, roads, powerlines, population) | `V1/src/ingestion`, `V2/src/preprocessing`, `TX/fetch_hrrr_tx.py`, `Supporting_files/scripts/01-06` | — | **Owner** |
| 2 | H3 res-8 grid, fire labels, negative sampling | `V2/phase2/run_phase2a-2d`, `Supporting_files/scripts/02-04` | Reviews label definition and split design | **Owner** |
| 3 | Static + dynamic feature extraction (LANDFIRE, terrain, HRRR, drought) | `V2/phase2/run_phase2e-2g`, `TX/merge_hrrr_duckdb.py` | — | **Owner** |
| 4 | Data-quality fixes (HRRR RH-unit bug, 84 % NaN rolling stats, gridMET nodata scaling, phantom rows, flare cells) | `TX/fix_hrrr_rh_bug.py`, `DATA_QUALITY_REVIEW.md`, `TX/missing_data_diagnosis.md` | Flags leakage and anomalies found during training | **Owner** |
| 5 | Imputation and final training table (3.59 M → 2.2 M rows, 0 % missing) | `TEXAS/build_dataset_v2.py`, `TDIS_Ignition_BIT*/build_dataset.py` | Defines which columns to keep or drop | **Owner** |
| 6 | Baseline models (V1 LightGBM, V2 XGBoost, leakage removal) | `V1/src/training`, `V2/phase2/run_phase3_train.py`, `TX/train_tx.py` | **Owner** | Supplies features |
| 7 | Hyperparameter search (21-trial and 34-trial, 3-stage regularization-aware) | `TX/tune_tx*.py`, `TDIS_Ignition_BIT*/scripts/tune_train_model.py` | **Owner** | — |
| 8 | Model generations: TDIS 23 → BIT v1 38 → V2 36 → V3 27 feature | `TDIS_Ignition_BIT`, `_V2`, `_V3` | **Owner** | Rebuilds dataset for each feature set |
| 9 | Isotonic calibration, threshold selection, Brier/ECE | `models/calibrator_*.joblib` | **Owner** | — |
| 10 | Evaluation and visuals (ROC, PR, confusion matrices, feature importance, reliability) | `results/figures/`, `generate_visuals.py` | **Owner** | — |
| 11 | Real-population validation (32.5 M cell-days, archived HRRR, res-5 aggregation vs TDIS) | `TDIS_Ignition_BIT_V3/realpop/` | **Owner** | Provides static grid, labels, drought join |
| 12 | Reports and comparison tables | `FINAL_MODEL_COMPARISON_REPORT.md`, `BIT_*_REPORT.md`, `EXECUTIVE_SUMMARY_*.md` | **Owner** (metrics, model sections) | **Owner** (data, dataset sections) |
| 13 | Daily feed: historical snapshot, live feed builder, feed daemon | `TDIS_REPLICA/ingestion/` | Supplies scoring function and model loading | **Owner** |
| 14 | Model serving API (FastAPI, API-key auth, bbox/zoom filtered risk cells) | `TDIS_REPLICA/api/main.py`, `texas-backend/api/` | **Owner** | Agrees endpoint contract |
| 15 | Containerization (Dockerfile, `.dockerignore`, env-var config, no secrets in image) | `TDIS_REPLICA/Dockerfile`, `texas-backend/Dockerfile` | **Owner** | — |
| 16 | Cloud deployment and CI/CD (Azure Container Apps, Azure Static Web Apps, GitHub Actions) | `.github/workflows/`, `tdis-portal-handoff-main/instruction_manual/` | **Owner** | Provides frontend build |
| 17 | Frontend portal (React 18 + Vite + MapLibre GL: hex risk layer, active-fire markers, tabs, legend, status strip) | `TDIS_REPLICA/frontend/src/` | Reviews risk-score display and thresholds | **Owner** |
| 18 | Architecture study of the TDIS portal (React/Mapbox, FastAPI, PostGIS, pg_tileserv, Databricks) | `tdis-portal-handoff-main/` | Reads deployment and secrets docs | Reads frontend and schema docs |
| 19 | Dashboards (static HTML risk explorer) | `Focused_Files/.../dashboard/` | Supplies forecast JSON | **Owner** |
| 20 | Git hygiene, `.gitignore`, repo structure, versioning | repo root, `PROJECT_VERSION_HISTORY.md` | Shared | Shared |

---

## 3. Role 1 — ML Engineer & MLOps / Deployment Lead

### Responsibilities
- Train, tune and compare every model generation, and keep the comparison honest (matched-sample vs real-population metrics).
- Design the 3-stage, regularization-aware XGBoost search (34 trials) and manage GPU training.
- Calibrate probabilities (isotonic) and choose operating thresholds.
- Validate on the **real population**: full Texas grid, real archived HRRR forecasts, real ignitions.
- Build and own the FastAPI serving layer, Docker images and Azure deployment pipeline.
- Own reproducibility: scripts, feature lists, model/calibrator artifacts, versioned reports.

### Key results to cite (all from the repo)
| Item | Result |
|---|---|
| Model generations compared | TDIS 23 → BIT v1 38 → V2 36 → V3 Final 27 features |
| Final model (matched test, n = 1,261,502) | AUROC 0.7975 · AUC-PR 0.7141 · F1 0.6179 · lift 2.34× |
| Hyperparameter search | 34 trials, 3 stages, ~274 s on GPU |
| Gridmet removal (HRRR-only, forecast-honest) | cost ≈ 0.002 AUROC |
| Calibration | Brier 0.1582 → 0.1500 (matched); 0.00672 → 0.00376 (real population) |
| Real-population validation | 32.5 M cell-days, 19 dates, res-5 AUROC 0.644, lift 3.15× vs TDIS 0.797 / 4.58× |
| Honest finding | Matched-sample accuracy overstated real-population skill; removing lat/lon hurt far more on the real population |

### Skills and tools
Python · XGBoost (CUDA) · scikit-learn · pandas/pyarrow · Herbie (HRRR) · matplotlib · FastAPI · Docker · Azure Container Apps · Azure Static Web Apps · GitHub Actions · Git

### Certificate wording (suggested)
> Served as **ML Engineer & MLOps/Deployment Lead** on a Texas wildfire ignition-prediction system. Trained and tuned XGBoost models across four feature-set generations, calibrated and validated them on a 32.5-million-row real-population panel, and built the FastAPI serving layer, Docker containers and Azure/GitHub Actions deployment pipeline.

### Resume bullets
- Trained and compared four XGBoost generations (23 → 38 → 36 → 27 features) with a regularization-aware 34-trial search; final model AUROC 0.7975 on 1.26 M test rows.
- Built a real-population validation pipeline over 32.5 M cell-days using archived HRRR forecasts, exposing the gap between matched-sample and deployment metrics.
- Containerized the FastAPI model service and set up Azure Container Apps / Static Web Apps deployment with GitHub Actions, keeping secrets out of images.

---

## 4. Role 2 — Data Engineer & Full-Stack (Frontend) Developer

### Responsibilities
- Source and ingest all public datasets: FPA-FOD, VIIRS active fire, gridMET, HRRR, LANDFIRE, USDM drought, roads, powerlines, population.
- Build the H3 res-8 grid (1.7 M Texas cells), fire labels and day-matched negative sampling.
- Engineer static and dynamic features, then clean and impute to a 0 %-missing training table.
- Find and fix data-quality defects before they reach the model.
- Build the historical snapshot, simulated live feed and daemon that feed the portal.
- Build the React + MapLibre frontend that displays risk cells and active fires.

### Key results to cite (all from the repo)
| Item | Result |
|---|---|
| Grid | H3 res-8, ~1.7 M Texas cells (~0.82 km² each) |
| Final training table | 2,201,289 rows, 27 features, 0 % missing, 30.4 % positive rate |
| Imputation | 24.09 % missing → 0 % (cross-source substitution + seasonal median) |
| Bugs found and fixed | HRRR RH-unit error (100,981 rows), 84.4 % NaN in rolling stats, gridMET nodata scaling, H3 res 7 → 8 mix-up, flare-cell contamination (531 cells) |
| Leakage removed | `fire_count` / `has_fire_history` computed with future data |
| Feed | Historical snapshot of 609,705 cells; 31-day live feed with daemon |
| Portal | React 18 + Vite + MapLibre GL, bbox/zoom-filtered risk-cell endpoint, active-fire markers |

### Skills and tools
Python · pandas / DuckDB / pyarrow · H3 · GeoPandas / rasterio · HRRR / GRIB2 · SQL / PostgreSQL schema · React 18 · Vite · MapLibre GL JS · JavaScript · HTML/CSS · Git

### Certificate wording (suggested)
> Served as **Data Engineer & Full-Stack Frontend Developer** on a Texas wildfire ignition-prediction system. Built the end-to-end data pipeline (VIIRS, HRRR, LANDFIRE, drought and infrastructure data onto a 1.7 M-cell H3 grid), found and fixed critical data-quality defects, built the daily data feed, and developed the React/MapLibre risk-map portal.

### Resume bullets
- Built the data pipeline from raw public sources to a 2.2 M-row, 27-feature training table with zero missing values, on a 1.7 M-cell H3 grid.
- Diagnosed and fixed data-quality issues including an HRRR unit bug, 84 % NaN rolling statistics, label leakage and persistent flare-cell contamination.
- Built the React 18 + MapLibre GL portal and the ingestion/feed daemon that drive a live-style Texas wildfire risk map.

---

## 5. Where the two roles meet

| Interface | Role 2 provides | Role 1 consumes |
|---|---|---|
| Training table | `bit_train_tx.parquet` with `feature_list.json` order | Trains, tunes, validates |
| Real-population panel | Static grid, labels, flare list, drought join | Scores and evaluates |
| Daily feed | Features per cell per day | Model scoring function |

| Interface | Role 1 provides | Role 2 consumes |
|---|---|---|
| Model + calibrator | `model_*.json`, `calibrator_*.joblib` | Scoring in the ingestion scripts |
| API contract | `/api/v1/wildfire/risk-cells`, `/active-fires`, `/live-status`, `/health` | Frontend map layers |
| Deployment | Azure Container App URL, API key handling | `.env.production` backend URL |

---

## 6. Notes before using this for a certificate

- This split is a **proposal built from the repo structure**. The git history has a single author (`navaneethakrishnanms`, 40 commits), so it does not record who did which part. Confirm each person's actual contribution before signing anything, and keep the wording to work each of you really did.
- Role 1's real-population result is below the TDIS served model (res-5 AUROC 0.644 vs 0.797). It is a legitimate finding, but it should be described as validation work, not as an accuracy win.
- The portal is a **replica demo** with a simulated August 2026 feed, labelled `simulated_live` in the data. Describe it that way. The Azure deployment is workflow-driven and manual: auto-deploy on push is disabled in `.github/workflows/`.
- If the two of you do not want to share phase 12 (reports), assign it fully to one role. It is shared above only because it covers both data and model sections.
