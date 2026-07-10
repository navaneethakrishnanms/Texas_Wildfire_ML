# IgnitionNet — Project Scope, Data Card & Team Guide

---

## ⚠️ READ THIS FIRST — Scope Clarification

Based on the team meeting discussions and your submitted documents, there is a clear confusion about the scope of this project. This README exists to resolve it.

**This project has ONE scope: predicting WHERE and WHEN wildfire ignitions are discovered, using sub-daily 6-hour windows.**

It is NOT a damage assessment project. It is NOT a multi-hazard project covering floods and hurricanes. It does NOT use satellite imagery. It does NOT use DamageTriage-Bench. It will not use CalFire building damage

Those topics belong to a separate project that your team has been asking about. Post damage assessment is independent work and is not related to ignition proition.

**Ignition goal:** Build a working daily ignition prediction model for California and Texas using FPA-FOD labels + gridMET weather + LANDFIRE landscape features. 

---

## What This Project Is

We are building a machine learning model that answers one specific question:

> **Given a 6-hour window on a given day, which H3 hexagonal cells in California (or Texas) are most likely to have a wildfire discovered during that window?**

The model takes as input publicly available landscape and weather data. It outputs a ranked list of high-risk cells that could be used by fire managers to pre-position resources before a fire starts.

This is an ignition PREDICTION system — not a detection system, not a damage system, not a spread model. It operates BEFORE a fire is confirmed, not after.

---



---

## Project Architecture — One Diagram

```
INPUT (pre-fire data, publicly available)
    │
    ├── FPA-FOD v6           → LABELS ONLY (where/when fires discovered)
    ├── LANDFIRE / FSim      → Landscape features (static)
    ├── gridMET (Abatzoglou) → Daily fire weather features
    └── HRRR (NOAA)          → Per-6-hour atmospheric features
    │
    ▼
H3 Resolution-8 Grid (~860m hexagonal cells, ~1.47M cells in California)
    │
    ▼
XGBoost Binary Classifier
    │
    ▼
OUTPUT: Ranked risk score per (H3 cell, 6-hour window)
"These 59 cells are highest risk in the next 6 hours"
```

---

## Labels — The Most Important Section

### What FPA-FOD provides

FPA-FOD v6 (Fire Program Analysis Fire Occurrence Database, USDA Forest Service) is your **label source only**. It is not a feature source.

It provides a list of where and when fires were discovered by federal agencies. Every record in FPA-FOD is a fire that actually occurred.

**Download:** https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6
**Citation:** Short, K.C. (2022). DOI: 10.2737/RDS-2013-0009.6

### Positive labels (Fire = 1)

- California fires ≥ 1 acre, 2014–2020: **15,639 events**
- Texas fires ≥ 1 acre, 2014–2020: **36,182 events**
- Each fire is mapped to one H3-8 cell using its lat/lon
- Each fire is assigned to one 6-hour UTC window using its discovery time
- The label represents **discovery time** — when the fire was reported to authorities — not the exact moment of ignition. This distinction matters for interpretation but does not affect the model pipeline.

### Cause breakdown (California)

| Cause | Count | % | Notes |
|---|---|---|---|
| Unknown | 8,502 | 54.4% | Most records — cause not determined |
| Human misc | 3,564 | 22.8% | Arson, debris burning, recreation |
| Equipment | 2,378 | 15.2% | Power lines, vehicles |
| Lightning | 1,195 | 7.6% | Natural ignition |

### Negative labels (Fire = 0)

FPA-FOD contains ONLY fire events. You must generate the non-fire samples yourself.

**Method: DAY_MATCHED (this is the method we use)**

For each positive fire event on date D in window W:
- Draw 10 H3 cells from California (or Texas) that did NOT have a fire on date D in window W
- These 10 cells become negative samples with label = 0
- They inherit the same date and window as the fire event
- Their weather features are extracted at their own location (NOT copied from the fire cell)

**Why this method:** It forces the model to learn spatial discrimination — which cells are high risk given today's conditions — not temporal discrimination (which days have fires). This is the operationally correct question for dispatch pre-positioning.

**Result:** 15,639 positives + 156,390 negatives = **172,073 total rows** at 10:1 ratio.

**Sensitivity analysis:** Miguel ran the model at ratios from 1:1 to 1:100 and confirmed AUROC is stable. The 1:10 ratio is the default but you can adjust it.

### The target column

```python
df['label']  # 1 = fire discovered, 0 = no fire
```

There is no other target column. FPA-FOD columns like FIRE_SIZE, FIRE_NAME, FIRE_YEAR, CAUSE are metadata and index fields — not features, not targets.

---

## Temporal Structure

The study period is 2014–2020. Each day is divided into four 6-hour UTC windows:

| Window | California local (PDT fire season) | % of CA fires |
|---|---|---|
| 00Z | 4 pm – 10 pm | 21.6% |
| 06Z | 10 pm – 4 am | 8.6% |
| 12Z | 4 am – 10 am | 18.4% |
| **18Z** | **10 am – 4 pm** | **51.4%** |

The afternoon 18Z window has 51% of all fire discoveries. This aligns with peak daytime heat, low humidity, and human activity patterns.

**Important timezone note for Texas:**
Texas uses Central Time (UTC-5 CDT / UTC-6 CST) for most counties. Five El Paso-area counties use Mountain Time. Apply DST-aware timezone conversion using `pytz.timezone('America/Chicago')` per county FIPS. Do not use a fixed UTC offset — this caused a 23% label error in earlier California processing when DST was ignored.

---

## Train / Validation / Test Split

**Always use chronological splitting. Never random splitting.**

Random splitting leaks future fire locations into training data, producing artificially inflated performance that will not hold in deployment.

| Split | Years | n positive (CA) | Role |
|---|---|---|---|
| Train | 2014–2017 | ~8,162 | Model learning |
| Validation | 2018 | ~2,146 | Early stopping |
| Test | 2019–2020 | ~5,300 | Final evaluation only |

---

## Feature Table — Complete Reference

All features listed here apply to BOTH fire cells and non-fire cells. Non-fire cells get their own independently extracted values — they do not inherit the paired fire cell's weather.

### Group 1 — Landscape Features (Static — same for all dates)

Source: LANDFIRE LF2022 + USFS FSim
Download: https://landfire.gov + https://www.firelab.org

| Feature | Type | Units | Description | Why it matters |
|---|---|---|---|---|
| `avg_burn_prob` | float | 0–1 | Long-run probability a cell burns in a large wildfire. From 50,000 stochastic fire simulations (FSim). | Strongest single landscape predictor. Encodes topography, fuels, and ignition density holistically. |
| `whp` | float | index 0–7000 | Wildfire Hazard Potential — relative index of fire intensity and suppression difficulty. | Tells WHERE fires become severe, not just where they start. |
| `flep4` | float | probability 0–1 | Flame Length Exceedance Probability at 4 ft. Probability modeled fire exceeds 4-ft flame length — threshold where direct suppression becomes dangerous. | Cohen's d effect size > 1.0 in your EDA. Strong discriminator. |
| `cfl` | float | Mg ha⁻¹ | Canopy Fuel Load — dry biomass of forest canopy available for combustion. | High CFL = crown fire potential. Second strongest landscape feature. |

**How to extract for non-fire cells:**
```python
import rasterio
import h3
import numpy as np

def extract_at_centroid(h3_cell, raster_path):
    lat, lon = h3.cell_to_latlng(h3_cell)
    with rasterio.open(raster_path) as src:
        # reproject lat/lon to raster CRS
        from pyproj import Transformer
        t = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
        x, y = t.transform(lon, lat)
        row, col = rasterio.transform.rowcol(src.transform, x, y)
        row = int(np.clip(row, 0, src.height - 1))
        col = int(np.clip(col, 0, src.width - 1))
        return float(src.read(1)[row, col])
```

---

### Group 2 — Daily Fire Weather (Dynamic — one value per cell per day)

Source: gridMET (Abatzoglou 2013, 4 km daily CONUS)
Download: https://www.climatologylab.org/gridmet.html
NetCDF URL pattern: `https://www.northwestknowledge.net/metdata/data/{variable}_{year}.nc`
Citation: Abatzoglou (2013), Int. J. Climatol. DOI: 10.1002/joc.3413

**These are the features Haripriya asked about (pr, tmmx, erc, vpd) — they all come from gridMET, not FPA-FOD.**

| Feature | gridMET variable | Units | Description | URL |
|---|---|---|---|---|
| `erc` | `energy_release_component-g` | BTU ft⁻² | Energy Release Component. Primary NFDRS fire danger index. Integrates multi-day fuel drying. Best single daily fire danger predictor. | `erc_{year}.nc` |
| `fm100_5D_mean` | `dead_fuel_moisture_100hr` → 5-day lag | % moisture | 5-day trailing mean of 100-hr fuel moisture. Captures drought trajectory rather than single-day conditions. | `fm100_{year}.nc` |
| `pr` | `precipitation_amount` | mm | Daily precipitation. Optional additional feature. | `pr_{year}.nc` |
| `tmmx` | `air_temperature` | K | Daily maximum temperature. Convert to °C: `tmmx - 273.15` | `tmmx_{year}.nc` |
| `vpd` | `mean_vapor_pressure_deficit` | kPa | Vapor pressure deficit from gridMET. Use THIS vpd, not the one in FPA-FOD. | `vpd_{year}.nc` |

**How to download and join gridMET for non-fire cells:**
```python
import netCDF4 as nc
import numpy as np
import urllib.request
import os

def download_gridmet(variable, year, cache_dir='gridmet_cache'):
    os.makedirs(cache_dir, exist_ok=True)
    url = f'https://www.northwestknowledge.net/metdata/data/{variable}_{year}.nc'
    path = f'{cache_dir}/{variable}_{year}.nc'
    if not os.path.exists(path):
        print(f'Downloading {url}...')
        urllib.request.urlretrieve(url, path)
    return path

def get_gridmet_value(lat, lon, date, variable, year, cache_dir='gridmet_cache'):
    """Get gridMET value for a cell centroid on a specific date."""
    path = download_gridmet(variable, year, cache_dir)
    ds = nc.Dataset(path)

    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    times = nc.num2date(ds.variables['day'][:], ds.variables['day'].units)

    # Find nearest grid point
    lat_idx = int(np.argmin(np.abs(lats - lat)))
    lon_idx = int(np.argmin(np.abs(lons - lon)))

    # Find date index
    date_idx = None
    for i, t in enumerate(times):
        if t.date() == date:
            date_idx = i
            break

    if date_idx is None:
        return np.nan

    # Get variable name inside NetCDF (may differ from filename)
    varname = [v for v in ds.variables if v not in
               ['lat', 'lon', 'day', 'crs']][0]
    value = float(ds.variables[varname][date_idx, lat_idx, lon_idx])
    ds.close()
    return value

# Usage for a non-fire cell:
# lat, lon = h3.cell_to_latlng(neg_cell)
# erc_val = get_gridmet_value(lat, lon, fire_date, 'energy_release_component-g', year)
```

**Critical rule:** gridMET values are extracted at each cell's own centroid for its assigned date. Non-fire cells do NOT copy the fire cell's ERC or FM100 value. Each row gets its own independently extracted weather.

---

### Group 3 — Per-Window HRRR Atmospheric Features (Dynamic — 6-hourly)

Source: NOAA High-Resolution Rapid Refresh (HRRR), analysis cycles (f00)
AWS Archive: s3://noaa-hrrr-bdp-pds (public, no authentication)
Retrieval: Herbie Python library (`pip install herbie-data`)
Citation: Benjamin et al. (2016), Mon. Wea. Rev. DOI: 10.1175/MWR-D-15-0242.1

**These features are what make this a sub-daily model. Each row gets HRRR conditions from its own 6-hour window — not a fixed morning snapshot.**

| Feature | HRRR variable | Units | Description |
|---|---|---|---|
| `rh_pw` | `RH:2 m above ground:anl` | % | 2-m relative humidity at the fire's own UTC window. Low RH = rapid fire spread. |
| `temp_pw` | `TMP:2 m above ground:anl` | °C | 2-m temperature. Afternoon peak accelerates fuel drying. |
| `wind_speed_pw` | `UGRD` + `VGRD` → √(U²+V²) | m s⁻¹ | 10-m wind speed. Primary driver of fire spread and spotting distance. |
| `vpd_pw` | Derived from TMP + RH | kPa | Vapour pressure deficit from HRRR. Combined drying power. |
| `hpbl_pw` | `HPBL:surface:anl` | m | Planetary boundary layer height. Tall PBL = strong mixing and fire weather intensity. |
| `dswrf_pw` | `DSWRF:surface:anl` | W m⁻² | Downwelling solar radiation. Near-zero at night, >900 at 18Z summer afternoons. |

**HRRR coverage by year:**

| Year | HRRR available? | Notes |
|---|---|---|
| 2014 | ❌ No | Archive not captured — skip these rows |
| 2015 | ❌ No | Archive not captured — skip these rows |
| 2016 | ⚠️ Partial (17%) | Early operational period |
| 2017 | ✅ 97% | Use |
| 2018 | ✅ 98% | Use |
| 2019 | ✅ 100% | Test set — full coverage |
| 2020 | ✅ 100% | Test set — full coverage |

**Note:** If you are starting without HRRR, build the daily model first (Groups 1+2+4+5 only). Add HRRR later as an extension step. Your daily baseline is already a valid publication-quality result.

**Basic HRRR fetch code using Herbie:**
```python
from herbie import Herbie
from scipy.spatial import cKDTree
import numpy as np

def fetch_hrrr_window(date_str, window_hour, lat, lon):
    """
    Fetch HRRR analysis fields for one 6-hour window at one location.
    date_str: '2018-11-08'
    window_hour: 0, 6, 12, or 18 (UTC)
    """
    H = Herbie(
        f'{date_str} {window_hour:02d}:00',
        model='hrrr',
        product='sfc',
        fxx=0  # analysis, not forecast
    )
    # Fetch RH
    rh_ds = H.xarray(':RH:2 m above ground:anl')
    # Use nearest-neighbour to get value at (lat, lon)
    # ... (see full pipeline code in hrrr.zip)
    return rh_val, temp_val, wind_val, hpbl_val, dswrf_val
```

Full HRRR extraction code was shared in the team channel as `hrrr.zip`. Use that as your baseline.

---

### Group 4 — Temporal Encodings (Derived from timestamp)

These features are computed directly from the window timestamp — no external data source needed.

| Feature | Formula | Purpose |
|---|---|---|
| `sin_month` | `sin(2π × month / 12)` | Cyclic month encoding. Captures fire season peak (Jul–Oct) without assuming January follows December discontinuously. |
| `cos_month` | `cos(2π × month / 12)` | Paired with sin_month. Both needed for full cyclic representation. |
| `sin_hour` | `sin(2π × hour / 24)` where hour ∈ {0,6,12,18} | Cyclic hour encoding. Model learns 18Z is riskier than 06Z. |
| `cos_hour` | `cos(2π × hour / 24)` | Paired with sin_hour. |

```python
import numpy as np
df['month'] = df['window_6h_utc'].dt.month
df['hour']  = df['window_6h_utc'].dt.hour
df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
df['sin_hour']  = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_hour']  = np.cos(2 * np.pi * df['hour'] / 24)
```

---

### Group 5 — Location (From H3 centroid)

| Feature | Units | Description |
|---|---|---|
| `lat` | degrees N | H3-8 cell centroid latitude |
| `lon` | degrees E | H3-8 cell centroid longitude |

```python
import h3
df['lat'] = df['h3_cell'].apply(lambda c: h3.cell_to_latlng(c)[0])
df['lon'] = df['h3_cell'].apply(lambda c: h3.cell_to_latlng(c)[1])
```

---

### Complete Feature Summary Table

| Feature | Group | Temporality | Source | Both fire and non-fire? |
|---|---|---|---|---|
| `avg_burn_prob` | Landscape | Static | USFS FSim | ✅ Yes — same value for both |
| `whp` | Landscape | Static | USFS WHP | ✅ Yes |
| `flep4` | Landscape | Static | LANDFIRE | ✅ Yes |
| `cfl` | Landscape | Static | LANDFIRE | ✅ Yes |
| `erc` | Daily weather | Daily | gridMET | ✅ Yes — cell-specific |
| `fm100_5D_mean` | Daily weather | Daily | gridMET | ✅ Yes — cell-specific |
| `rh_pw` | HRRR | 6-hourly | NOAA HRRR | ✅ Yes — cell-specific |
| `temp_pw` | HRRR | 6-hourly | NOAA HRRR | ✅ Yes — cell-specific |
| `wind_speed_pw` | HRRR | 6-hourly | NOAA HRRR | ✅ Yes — cell-specific |
| `vpd_pw` | HRRR | 6-hourly | NOAA HRRR (derived) | ✅ Yes — cell-specific |
| `hpbl_pw` | HRRR | 6-hourly | NOAA HRRR | ✅ Yes — cell-specific |
| `dswrf_pw` | HRRR | 6-hourly | NOAA HRRR | ✅ Yes — cell-specific |
| `sin_month` | Temporal | Derived | Computed | ✅ Yes |
| `cos_month` | Temporal | Derived | Computed | ✅ Yes |
| `sin_hour` | Temporal | Derived | Computed | ✅ Yes |
| `cos_hour` | Temporal | Derived | Computed | ✅ Yes |
| `lat` | Location | Static | H3 centroid | ✅ Yes |
| `lon` | Location | Static | H3 centroid | ✅ Yes |

**Total: 18 features for daily model (no HRRR), 12 for minimal baseline, 18 for sub-daily model**

---

### Features to NEVER Include (Leakage)

These columns exist in the FPA-FOD dataset or elsewhere but must NOT be used as model features:

| Column | Why excluded |
|---|---|
| `vpd` from FPA-FOD | Recorded post-discovery — leakage. Use `vpd_pw` from HRRR or `vpd` from gridMET instead. |
| `FIRE_SIZE` | Only known after containment — future leakage |
| `CAUSE`, `STAT_CAUSE_CODE` | Assigned post-discovery — not available before ignition |
| `CONT_DATE`, `CONT_TIME` | Post-hoc |
| `ignition_density`, `burn_count` | Historical fire count — trivially separates fire/non-fire by construction |
| Any FPA-FOD weather field (ERC, BI, FM100 from report) | Recorded at fire location after discovery |

---

## Negative Sampling — Detailed Guide

This is the question that generated the most confusion in team meetings. Here is the complete answer.

### Why negatives are needed

FPA-FOD contains ONLY fire events — approximately 15,000 records for California over seven years. A machine learning model requires both positive examples (fires) and negative examples (no fires). Since almost all H3 cells on almost all days have no fire, you need to sample a manageable subset of those non-fire cases.

### The two methods

**Method 1: DAY_MATCHED (recommended for this project)**

```python
import numpy as np
import pandas as pd
import h3

def build_day_matched_negatives(positives_df, all_state_cells, ratio=10, seed=42):
    """
    For each positive fire event, sample `ratio` non-fire H3 cells
    from the same calendar date and UTC window.
    """
    rng = np.random.default_rng(seed)
    neg_rows = []

    # Group positives by (date, window)
    positives_df['date_d'] = pd.to_datetime(
        positives_df['window_6h_utc']).dt.date

    for (date_d, window), group in positives_df.groupby(
            ['date_d', 'window_6h_utc']):

        fire_cells = set(group['h3_cell'].values)
        eligible = [c for c in all_state_cells if c not in fire_cells]

        if len(eligible) < ratio:
            continue

        n_sample = len(group) * ratio
        sampled_cells = rng.choice(
            eligible, size=min(n_sample, len(eligible)), replace=False)

        for cell in sampled_cells:
            lat, lon = h3.cell_to_latlng(cell)
            neg_rows.append({
                'h3_cell': cell,
                'window_6h_utc': window,
                'label': 0,
                'lat': lat,
                'lon': lon,
                'fire_year': pd.Timestamp(date_d).year,
            })

    return pd.DataFrame(neg_rows)
```

**Method 2: HyperSampling (from Pourmohamad 2026 — alternative)**

Non-fire cells are drawn randomly across ALL dates and times, not matched to fire event dates. This is a different design that answers a different question — it learns both when days are fire-dangerous AND which cells are high-risk. For comparison: on a common test set, DAY_MATCHED achieved AUPR 0.67 vs HyperSampling 0.43. DAY_MATCHED is recommended for the operational dispatch use case.

### After sampling — feature extraction for negatives

Each negative cell must have its own features extracted at its assigned date and window. Do NOT copy the paired positive cell's features.

```
For each negative row (cell, date, window):
  1. lat, lon = h3.cell_to_latlng(cell)
  2. avg_burn_prob = extract from LANDFIRE raster at (lat, lon)
  3. erc = gridMET value at (lat, lon) for this date
  4. fm100 = gridMET value at (lat, lon) for this date → compute 5-day mean
  5. rh_pw = HRRR value at (lat, lon) for this window (if HRRR available)
  6. Temporal features: computed from date and window
```

---

## Missing Value Handling

### Landscape features (avg_burn_prob, whp, flep4, cfl)

Zero-fill when a raster returns NaN or nodata. For FSim burn probability, a value of 0 is meaningful — it indicates the cell was never simulated to burn. Zero-fill is semantically correct, not an approximation.

### Daily weather (erc, fm100)

gridMET has near-complete CONUS coverage. If a join fails due to boundary cells, use the nearest valid grid point. Do not impute from other dates.

### HRRR per-window features

HRRR is missing for 2014–2015 entirely (archive gaps) and ~84% missing for 2016. This is structural missingness, not random noise.

- **At training time:** filter out rows where HRRR is unavailable (flag: `pw_available = False`). Do not impute.
- **At inference time:** zero-fill if HRRR unavailable. XGBoost's native sparsity handling learns the optimal behavior for missing values during training.
- **Do not interpolate NWP fields spatially or temporally** — HRRR missingness is structural, and interpolated NWP values are physically meaningless.

### Feature removal criteria

Before removing any feature due to high missingness, check whether its LOO (leave-one-out) AUPR drop is > 0.002. If removing the feature causes less than 0.002 AUPR change, it adds missingness overhead for no benefit — drop it. If the AUPR drop is > 0.002, retain and handle missingness as above.

Correlation analysis (as shown in your team's EDA figures) is useful for understanding relationships but should NOT be the sole basis for feature removal. Correlated features (FLEP4 and CFL show r ≈ 0.96) can both be retained — XGBoost tree splits handle correlated inputs without instability.

---

## Model Configuration

```python
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

FEATURE_COLS = [
    # Landscape
    'avg_burn_prob', 'whp', 'flep4', 'cfl',
    # Daily weather
    'erc', 'fm100_5D_mean',
    # HRRR per-window (add these only after daily baseline works)
    'rh_pw', 'temp_pw', 'wind_speed_pw', 'vpd_pw', 'hpbl_pw', 'dswrf_pw',
    # Temporal
    'sin_month', 'cos_month', 'sin_hour', 'cos_hour',
    # Location
    'lat', 'lon',
]

# Compute class weight from training set
pos_weight = int(y_train.value_counts()[0] / y_train.value_counts()[1])

model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,   # handles 1:10 imbalance
    eval_metric='aucpr',
    early_stopping_rounds=25,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
```

---

## Evaluation Metrics

**Do not report accuracy.** A model that predicts "no fire" everywhere achieves 91% accuracy. This number is meaningless for rare-event prediction.

**Primary metrics:**

| Metric | Why it is correct |
|---|---|
| AUPR (Average Precision, area under PR curve) | Threshold-free, sensitive to rare positive class, not affected by easy negatives |
| AUROC (Area under ROC curve) | Overall ranking quality — comparable across sampling designs |
| Precision@K | Operational relevance — "of the top K flagged cells, how many had fires?" |

**Expected results for daily baseline:**
- Test AUROC: ~0.93–0.96
- Test AUPR: ~0.60–0.70
- Precision at top 59 cells: ~89%

If AUROC exceeds 0.99 → you have a post-hoc leakage feature included. Stop and check your feature list.

---

## Team Progress Feedback

*This section reviews what was submitted in the Presentation Document (Wildfire Risk Prediction, July 2, 2026) and the Technical Discussion document.*

### What the team did well

The pipeline diagram (5-stage flowchart) is clear and correctly structured. The EDA dashboard figures are professional — the Cohen's d effect size chart correctly identifies FLEP4 and CFL as top discriminators, which is consistent with validated results. The correlation heatmaps by feature group are well organized and show correct relationships (ERC–FM100 inverse correlation, FLEP4–CFL positive correlation). The UTC window analysis figure (Section 6) correctly shows 18Z as the dominant fire window. The geographic fire density map and cause distribution figures are accurate.

The team correctly identified that FPA-FOD contains only positive labels and asked the right question about how to generate negatives. That shows good scientific thinking.

### What needs to be corrected

**Scope confusion — most critical issue.**
Your document extensively discusses Stage 2 damage assessment, DamageTriage-Bench, flood data, and hurricane data. None of this is part of your current assignment. Focus on Stage 1 ignition prediction only. Do not spend time on Stage 2 until the ignition model is fully working and validated.

**The 72-hour forecast horizon in your pipeline diagram is incorrect.**
Your diagram shows "Predict Wildfire Risk for Next 24/48/72 Hours." The current model does NOT produce 72-hour forecasts. It produces a ranked risk map for a specific 6-hour window using conditions available at that time. The operational claim is: given morning atmospheric data, which cells are highest risk in the next 6 hours? Remove 72-hour language unless you specifically build and validate a 72-hour lagged forecast.

**FPA-FOD weather fields must be dropped.**
Your feature selection step retained 309 columns without applying the pre-ignition availability filter. Apply the test: could this value be known before the fire was discovered? This will reduce your feature set to approximately 20 operationally valid predictors. Features like FIRE_SIZE, VPD from FPA-FOD, CAUSE, and CONT_DATE must be removed.

**The gridMET question has been answered.**
Haripriya asked about the source for pr, tmmx, erc, vpd for non-fire samples. The answer is gridMET for all of these. The exact NetCDF URLs and extraction code are provided in this README above. Use the cKDTree nearest-neighbour approach to join gridMET values to H3 cell centroids. This applies identically to both fire and non-fire cells.

**Negative sampling is now clarified.**
The team was confused about whether the provided dataset contains non-fire records. It does not — FPA-FOD is positive labels only. You generate negatives using DAY_MATCHED sampling as described in this README. The step-by-step code is provided above.

**Cause class should not be used as a model feature at this stage.**
Your EDA includes cause breakdown analysis, which is good for understanding the data. But cause is a post-hoc field — it is only known after the fire is investigated, sometimes weeks or months later. Do not include it as a model input feature. It can be used for stratified analysis after training (e.g., comparing model performance for lightning fires vs. human fires), but not as a predictor.

---

## Prioritized Next Steps for the Team

Complete these in strict order. Do not move to step N+1 until step N is working and validated.

**Step 1 — Confirm scope (this week)**
Read this README. Confirm in your next meeting that the team understands the project is ignition prediction only. Stage 2 damage assessment is deferred.

**Step 2 — Build H3 grid for your state (this week)**
Install h3 (`pip install h3`). Generate all H3-8 cells within your state bounding box. Confirm cell count: California ~1.47M, Texas ~2.2M.

**Step 3 — Process FPA-FOD labels (this week)**
Load FPA-FOD CSVs. Filter to your target state. Apply DST-aware UTC conversion. Snap to H3 cell. Assign to 6-hour window. Produce a clean parquet: (h3_cell, window_6h_utc, label=1, lat, lon, fire_year).

**Step 4 — Generate DAY_MATCHED negatives (next week)**
Use the code in this README to generate 10 negative cells per positive. Produce the combined dataset with label column. Confirm ~9% positive rate.

**Step 5 — Extract LANDFIRE features (next week)**
Download the four national LANDFIRE rasters (avg_burn_prob, whp, flep4, cfl). Sample at H3-8 cell centroids using rasterio. Join to the full dataset by h3_cell.

**Step 6 — Extract gridMET features (next week)**
Download annual ERC and FM100 NetCDF files from climatologylab.org for 2014–2020. Join to each row by (cell centroid, date). Compute 5-day trailing FM100 mean. This answers Haripriya's question about pr, tmmx, erc, vpd sources.

**Step 7 — Train daily baseline model (end of next week)**
Train XGBoost on landscape + gridMET features + temporal encodings + location. Confirm AUROC 0.93–0.96 and AUPR 0.60–0.70 on test set. If AUROC > 0.99, stop and remove leakage features.

**Step 8 — Run feature ablation (following week)**
Remove each feature group one at a time and record AUROC change. This tells you which features are contributing and validates the pipeline.

**Step 9 — HRRR extension (after Step 7 is validated)**
Add per-window HRRR features using the code in hrrr.zip. Retrain. Compare against daily baseline. This is optional for the first model version.

**Step 10 — Texas transfer (after California model works)**
Apply California-trained model to Texas test data. Retrain on Texas training data. Compare zero-shot vs. fine-tuned vs. trained-from-scratch performance.

---


---

## Quick Reference — Common Questions

| Question | Answer |
|---|---|
| Where do labels come from? | FPA-FOD v6 — fire occurrence database |
| Where does ERC come from? | gridMET — NOT from FPA-FOD |
| Where does VPD come from? | gridMET or HRRR — NOT from FPA-FOD |
| How do I get non-fire samples? | DAY_MATCHED negative sampling — code in this README |
| Do non-fire cells get their own weather? | Yes — extract gridMET at each cell's own centroid |
| Should I include cause as a feature? | No — post-hoc, known only after investigation |
| Should I use random train/test split? | No — chronological only |
| What if AUROC is above 0.99? | You have a leakage feature — check the feature list |
| Is damage assessment part of this? | No — separate Stage 2, not current scope |
| What prediction horizon do we use? | 6-hour windows — not 24/48/72 hour forecasts |



## Complete Q&A — Every Question from Team Documents and Chat

This section answers every specific question raised across both presentation documents (Wildfire Risk Prediction 02.07.2026 and Technical Discussion on Two-Stage Framework) and the Teams channel chat log. Questions are grouped by topic and answered directly.

---

### SECTION A — Dataset and Labels

**Q1. Does the provided dataset contain any non-fire (no wildfire occurrence) records, or does it consist only of fire incident data?**

The dataset provided to you (FPA-FOD) contains ONLY fire events — positive samples only. Every record is a wildfire that was discovered and reported. There are no non-fire records in FPA-FOD. You must generate the non-fire samples yourself using the DAY_MATCHED negative sampling method described in this README. This is standard practice in ignition prediction — fire databases only record where fires occurred, not the much larger set of places where fires did not occur.

---

**Q2. I can see each record corresponds to a wildfire event with attributes like FIRE_SIZE, FIRE_NAME, DISCOVERY_DATE. I do not see a target variable indicating whether a fire occurred. Which column should be used as the target variable? If there is no target column, how should non-fire observations be generated?**

The target variable does not exist in FPA-FOD — you create it yourself during dataset construction:
- Every record in FPA-FOD gets `label = 1` (fire occurred)
- Every H3 cell you sample as a non-fire absence gets `label = 0` (no fire)

Columns like FIRE_SIZE, FIRE_NAME, DISCOVERY_DATE are metadata and index keys. They are not features. The only information you take from FPA-FOD is where and when fires were discovered (lat/lon and discovery date/time), which you use to assign H3 cells and UTC windows. Everything else — features and labels — is constructed by your pipeline.

```python
# After processing FPA-FOD:
fire_df['label'] = 1      # all FPA-FOD records are positive

# After DAY_MATCHED sampling:
absence_df['label'] = 0   # all sampled non-fire cells are negative

# Combined dataset:
dataset = pd.concat([fire_df, absence_df])
# dataset['label'] is your target variable
```

---

**Q3. Does the dataset contain both fire and non-fire observations? Are non-fire samples included elsewhere?**

No. FPA-FOD is fires only. Non-fire samples do not exist anywhere as a pre-built dataset — you generate them. See the Negative Sampling section of this README for the complete DAY_MATCHED code. The training parquet Miguel uses (train_data_perwindow_v1.parquet) already contains both positive and negative rows at 1:10 ratio, but that file was built for California. For Texas, you need to run the same pipeline on the Texas-filtered FPA-FOD data.

---

**Q4. The IgnitionNet dataset was described as having 172,073 rows with 9.1% fire events. But we only received the raw FPA-FOD data which has only fire records. How do we get to 172,073 rows?**

The 172,073-row training table was built through this pipeline:
1. Filter FPA-FOD to California, fires ≥ 1 acre, 2014–2020 → 15,639 positive events
2. For each positive, generate 10 DAY_MATCHED negatives → 156,390 negative rows
3. Join landscape, gridMET, and HRRR features to every row
4. Result: 15,639 + 156,390 = 172,073 total rows

You need to run this same construction process for Texas. The FPA-FOD CSV files already contain Texas records — filter by `STATE == 'TX'`.

---

**Q5. What is the target label — does it represent wildfire ignition, discovery, or spread?**

The label represents **wildfire discovery** — the moment the fire was reported to a federal agency, not the precise ignition moment. This distinction matters for interpretation:
- The model predicts which cells are likely to have a fire **discovered** in a given 6-hour window
- Discovery typically lags ignition by hours in accessible terrain, potentially days for remote lightning fires
- The model does NOT predict fire spread — what happens after discovery is a different problem

For operational pre-positioning, discovery time is what matters because it is the moment dispatch decisions are triggered.

---

**Q6. How is the spatial and temporal data mapped to H3 cells?**

**Spatial mapping:** Each fire's latitude and longitude is converted to an H3-8 cell ID using `h3.latlng_to_cell(lat, lon, 8)`. The resulting 15-character hex string is the spatial key for all joins.

**Temporal mapping:** The fire's local discovery datetime is converted to UTC using DST-aware timezone conversion, then floored to the nearest 6-hour boundary (00Z, 06Z, 12Z, or 18Z). The resulting timestamp is the temporal key.

```python
import h3

# Spatial: lat/lon → H3 cell
h3_cell = h3.latlng_to_cell(fire_lat, fire_lon, 8)

# Temporal: local time → UTC window
# (See DST conversion note in this README for timezone handling)
window_hour = (utc_datetime.hour // 6) * 6  # floor to 00/06/12/18
```

---

### SECTION B — Features and Data Sources

**Q7. While constructing the non-fire dataset, we identified data sources for some features (MODIS, NASA). The source for pr, tmmx, erc, vpd is unclear. What is the appropriate data source for these features for non-fire samples?**

All four of these come from **gridMET** (Abatzoglou 2013), not from FPA-FOD, not from MODIS, not from NASA:

| Feature | gridMET variable name | Download URL |
|---|---|---|
| `erc` | `energy_release_component-g` | `https://www.northwestknowledge.net/metdata/data/erc_{year}.nc` |
| `fm100` | `dead_fuel_moisture_100hr` | `https://www.northwestknowledge.net/metdata/data/fm100_{year}.nc` |
| `pr` | `precipitation_amount` | `https://www.northwestknowledge.net/metdata/data/pr_{year}.nc` |
| `tmmx` | `air_temperature` | `https://www.northwestknowledge.net/metdata/data/tmmx_{year}.nc` |
| `vpd` | `mean_vapor_pressure_deficit` | `https://www.northwestknowledge.net/metdata/data/vpd_{year}.nc` |

The process is identical for both fire and non-fire cells: compute the H3 cell centroid lat/lon using `h3.cell_to_latlng(cell)`, find the nearest gridMET 4km grid point, and extract the daily value for the assigned date. Non-fire cells get their own independently extracted values — they do NOT copy the fire cell's gridMET values.

---

**Q8. Could you clarify the spatial extraction process for non-fire H3 cells? What spatial join or mapping technique is used to associate each sampled H3 cell with predictor features from source datasets?**

H3 cell centroid coordinates are the spatial key for all raster joins:

```python
import h3
import numpy as np
from scipy.spatial import cKDTree

# Get centroid for any H3 cell
lat, lon = h3.cell_to_latlng(h3_cell)

# For raster data (LANDFIRE GeoTIFFs): point-sample at centroid
# using rasterio — see LANDFIRE extraction code in this README

# For gridMET NetCDF: nearest-neighbor snap to 4km grid
# Build cKDTree from gridMET grid points once, reuse for all cells
gridmet_coords = np.column_stack([gridmet_lats.ravel(), gridmet_lons.ravel()])
tree = cKDTree(gridmet_coords)
dist, idx = tree.query([lat, lon], k=1)
gridmet_value = data_array.ravel()[idx]
```

At H3 resolution-8 (~0.74 km²), centroid-based nearest-neighbor is appropriate for 4km gridMET and 3km HRRR. No interpolation is needed — nearest-neighbor is the correct approach.

---

**Q9. How are temporal features aligned for non-fire samples? How are observation date and time determined when extracting dynamic features like weather and vegetation?**

Non-fire cells inherit the calendar date and 6-hour UTC window from the positive fire event they were matched to. This is the core of DAY_MATCHED design:

- If a fire occurred on 2018-11-08 at 18Z, its 10 matched negatives are also assigned date=2018-11-08 and window=18Z
- Each negative cell then has gridMET extracted at its own centroid coordinates for the date 2018-11-08
- Each negative cell then has HRRR extracted at its own centroid for the 18Z analysis cycle on 2018-11-08

The temporal stamp is inherited from the paired positive to define which weather slice to extract. The spatial coordinates determine the actual extracted values. These are different things — make sure your pipeline extracts weather at the negative cell's own location, not at the fire cell's location.

---

**Q10. Strong correlations were found among environmental variables (FLEP4 and CFL, ERC and Fuel Moisture). Do you recommend retaining all correlated variables or applying feature selection techniques like RFE or SHAP before training?**

Retain all correlated landscape features. XGBoost handles multicollinearity through its tree splitting structure without instability — unlike linear models. The high correlation between FLEP4 and CFL (r ≈ 0.96) reflects real physical overlap (both measure fuel hazard) but each still adds independent discriminative information at cell level. Removing one based on correlation alone risks losing information.

Do NOT use RFE on correlated tree features — it tends to remove genuinely informative variables because importance gets split between correlated pairs, making each look individually weaker than it is.

Use SHAP for feature importance analysis after training. SHAP correctly allocates marginal contributions even among correlated features and is now the standard for ML interpretability in environmental science publications. Raw XGBoost gain (the default) is biased toward high-cardinality features.

---

**Q11. Several features contain substantial missing values. What criteria or thresholds determine whether a feature is retained, imputed, or removed? Do you perform feature importance or correlation analysis before excluding high-missingness features?**

Three-tier decision process:

**Tier 1 — Structural missingness (HRRR 2014–2015):** Filter rows entirely. This is not random dropout — the HRRR archive simply did not exist for those years. Imputation would inject fabricated atmospheric data. Do not impute.

**Tier 2 — Sporadic missingness at inference:** Zero-fill. XGBoost's native sparsity-aware split finder learns the optimal branch direction for missing values during training. Zero-fill at both training and inference time is the correct approach.

**Tier 3 — Any other missing feature:** Before removing it, run a leave-one-out (LOO) AUPR check — train the model with and without the feature and measure AUPR change. If ΔAUPR < 0.002, the feature adds missingness overhead for negligible benefit and can be dropped. If ΔAUPR ≥ 0.002, retain and handle missingness as Tier 2.

Do not use a percentage threshold (e.g., "drop if > 30% missing") as the primary decision criterion. Always check predictive contribution first. A feature missing 60% of the time can still be the most important feature if it is highly predictive when present.

Never use spatial or temporal interpolation for NWP (HRRR) fields — missingness is structural, not noise, and interpolated NWP values are physically meaningless.

---

### SECTION C — Negative Sampling

**Q12. How is the negative sampling ratio determined (e.g., 1:1 or 1:10)?**

The starting ratio is 1:10 (one fire cell to ten non-fire cells), giving a ~9% positive rate. This is a practical default that balances training efficiency against class imbalance. The ratio is not theoretically fixed — it is a hyperparameter.

After building the initial model, run a sensitivity analysis: retrain at ratios 1:5, 1:10, 1:20, and 1:50. If AUPR remains stable (within ±0.005) across ratios, the model is robust to this choice. Miguel's California experiments confirmed this — the model was stable across a wide range of ratios. You should do the same sanity check for Texas.

---

**Q13. How do you ensure negative sampling does not introduce spatial or temporal bias?**

**Temporal bias prevention:** Negatives are drawn from the same calendar date and UTC window as each positive. Both the fire cell and its paired negatives experience identical ambient weather on that date. The model therefore cannot exploit weather differences between fire and non-fire rows — it must learn spatial terrain and fuel differences.

**Spatial bias prevention:** Negatives are drawn from the full state H3 grid, not from a neighborhood around the fire. This prevents creating an artificially easy local contrast. The full-state sampling means the model sees fire cells versus any non-fire cell statewide, which is the operationally correct decision boundary.

---

**Q14. What prediction time window is used — 6, 12, or 24 hours?**

6-hour UTC windows. This is the core of the sub-daily design. Each day is divided into four windows:
- 00Z: 4pm–10pm PDT
- 06Z: 10pm–4am PDT
- 12Z: 4am–10am PDT
- 18Z: 10am–4pm PDT

The model produces a separate ranked risk map for each window. This is NOT a 24, 48, or 72-hour forecast. Your pipeline diagram showed "24/48/72 hour prediction" — please remove that language. The model predicts risk within a specific 6-hour window using conditions available at the start of that window.

---

**Q15. Which NOAA weather products are queried and how are weather variables spatially aggregated to each H3 cell?**

**Product:** NOAA HRRR (High-Resolution Rapid Refresh), analysis cycles (f00). These are analysis fields — the model's best estimate of current conditions at each analysis time — not forecasts.

**Access:** AWS S3 public archive (`s3://noaa-hrrr-bdp-pds`) via the Herbie Python library. No authentication needed, no cost.

**Spatial aggregation:** Nearest-neighbor snap from H3-8 cell centroid to the nearest HRRR 3km grid point using cKDTree. No averaging or interpolation — the centroid's nearest HRRR pixel value is used directly. Miguel shared the extraction code as `hrrr.zip` in the Teams channel — use that as your starting point.

---

### SECTION D — Scope and Multi-Hazard Questions
Note: Use the damage Triage for the post-damage assessment scope. The ignition data has nothing to do with scope/stage 2. Scope2/stage 2 is an agnostic hazard damage assessment tool. The answers below are from the perspective of stage 1. 

**Q16. Since our POC focuses on multi-hazard post-disaster damage assessment (wildfire, flood, hurricane, and building damage), are there companion datasets or extensions that support these other disaster types? Is IgnitionNet intended solely for wildfire occurrence prediction?**

IgnitionNet is solely for wildfire ignition discovery prediction. It does not support flood, hurricane, or building damage prediction — those require completely different datasets, labels, spatial frameworks, and model architectures.

For your current assignment, ignore flood and hurricane entirely. Your scope is wildfire ignition prediction for Stage 1. If Stage 2 (damage assessment) becomes your assignment later, the CAL FIRE Damage Inspection dataset is the appropriate starting point for wildfire building damage, and Miguel has offered to help source those datasets separately.

---

**Q17. For the disaster assessment use case we require datasets covering floods, hurricanes, and building damage. Can we proceed with damage assessment using the ignition dataset alone?**

No. The ignition dataset cannot be used for damage assessment. It contains only the location and time of fire discovery — it has no information about burn severity, building damage, vegetation loss, or structural impacts. These are completely different outcome variables requiring different data sources.

Do not attempt to repurpose the ignition dataset for damage assessment. That would produce scientifically invalid results. Wait for Stage 2 assignment before pursuing damage datasets.

---

**Q18. There is a spatial and temporal mismatch between IgnitionNet (CA 2014–2020) and DamageTriage-Bench (2025 LA fires). How do you recommend integrating them?**

You should not integrate them for your current scope. This is a Stage 2 question and is outside your current assignment. When Stage 2 is assigned, the recommended approach is to use CAL FIRE Damage Inspection data (which covers California-wide fires from 2014 onward) rather than DamageTriage-Bench (which is specific to the 2025 LA fires). The CAL FIRE dataset aligns geographically and temporally with the ignition data and avoids the 2025-specific regional limitation of DamageTriage-Bench.

---

**Q19. Would you recommend maintaining Stage 1 and Stage 2 as two independent models, or should Stage 1 wildfire occurrence probability be incorporated as an input feature for Stage 2?**

Two independent models, with the Stage 1 probability as one input feature to Stage 2. Do not train them jointly end-to-end. The reasons: they operate at different spatial units (H3 cell vs. building), different temporal contexts (pre-fire vs. post-fire), and different data sources. The Stage 1 probability score is useful context for Stage 2 — higher predicted ignition probability may correlate with worse damage conditions — but it should be one feature among many, not a coupling mechanism.

This is a Stage 2 question. 

---

**Q20. From a research contribution and practical deployment perspective, would a purely tabular workflow provide sufficient predictive capability, or would a multimodal architecture with wildfire imagery offer significant improvements?**

Tabular is sufficient and is the correct choice for Stage 1. The ignition prediction task does not use imagery — it uses gridded weather and landscape rasters at H3 resolution. There are no fire images in Stage 1. Tabular XGBoost on the 19 features described in this README achieves AUPR ~0.63 and Precision@K ~98% at tight thresholds. There is no image-based approach that improves on this for the ignition prediction task.

For Stage 2 damage assessment, tabular is still recommended as the starting point because LANDFIRE + HRRR + CAL FIRE DINS data is structured and well-aligned. Image-based approaches can be explored as an extension if tabular Stage 2 results are strong enough to justify the additional complexity.

---

**Q21. What additional engineered features would improve Stage 2 damage prediction?**

This is a Stage 2 question — outside current scope. When Stage 2 begins, three feature categories are worth prioritizing: (1) antecedent drought conditions (30-day/90-day precipitation deficit, PDSI, soil moisture anomaly at ignition time); (2) topographic exposure (slope aspect relative to prevailing wind, topographic position index for windward exposure); (3) structure vulnerability proxies (distance from building to unmanaged vegetation, construction year as proxy for building code compliance). These are not in LANDFIRE but are available from county assessor databases and OpenStreetMap for California.

---

**Q22. Would integrating LANDFIRE layers alone be sufficient to represent environmental damage, or should additional remote sensing products like Sentinel-2, Landsat, MTBS, or BAER datasets be incorporated?**

LANDFIRE alone describes pre-fire fuel conditions, not post-fire damage. For Stage 2, MTBS (Monitoring Trends in Burn Severity) dNBR rasters are the appropriate burn severity input — they provide 30m Landsat-derived burn severity for fires above 1,000 acres and are publicly available for the full study period. BAER products cover smaller fires. Treat dNBR and burn severity class as INPUT FEATURES to Stage 2, not as target labels. The target label is the CAL FIRE building damage classification (destroyed / major / minor / no damage).



---

**Q23. Should environmental damage indicators (dNBR, burn severity) be used as input features or as target labels?**

Input features. The target label is observed building damage from CAL FIRE Damage Inspection records. dNBR and burn severity class are intermediate outputs of the fire process that help explain which buildings were damaged — they are predictors, not outcomes.

---

### SECTION E — Model Architecture and Evaluation

**Q24. How is class imbalance handled in the model?**

Two mechanisms:

First, the `scale_pos_weight` hyperparameter in XGBoost compensates for the 1:10 positive-to-negative ratio. Set it to the ratio of negative to positive count in the training set:
```python
pos_weight = int(y_train.value_counts()[0] / y_train.value_counts()[1])  # ≈ 10
model = XGBClassifier(scale_pos_weight=pos_weight, ...)
```

Second, AUPR is used as the primary metric rather than accuracy. AUPR is sensitive to the positive minority class and correctly reflects model performance on the rare fire events. Accuracy is not reported — at 1:10 ratio, a model predicting "no fire" always achieves 91% accuracy, which is meaningless.

Do not use SMOTE or synthetic oversampling. DAY_MATCHED already controls class balance by design and SMOTE would create artificial fire locations that distort the spatial signal.

---

**Q25. Is cause_class used during training or only for analysis?**

Only for analysis — NOT as a training feature. Cause is assigned post-discovery, sometimes weeks or months after the fire when investigation is complete. It is not available before ignition and therefore cannot be used in an operational prediction system.

After training, cause can be used to stratify the evaluation — for example, computing AUROC separately for lightning fires (AUPR ≈ 0.04, near chance) vs. human fires (AUPR ≈ 0.43) reveals important mechanistic insights. But cause never enters the feature matrix.

---

**Q26. What cloud platform should we use for data storage and computation?**

AWS is appropriate and is what the team is already using. Specific recommendations:
- Store large parquet files (training data, HRRR extracts) in S3
- Cache gridMET NetCDF files locally after first download (~80–130MB each per year) to avoid repeated large downloads
- HRRR is already on AWS S3 (`s3://noaa-hrrr-bdp-pds`) and can be accessed directly without downloading using Herbie
- For model training, EC2 (r5.4xlarge or similar memory-optimized instance) or SageMaker is appropriate
- Google Drive is suitable only for sharing documents, not for large dataset storage

---

**Q27. The EDA showed July has the highest wildfire incidents. But your pipeline uses UTC windows — does the seasonal peak month change when converted to UTC?**

No. The seasonal peak (June–October) is robust to UTC conversion because the offset is small (7-8 hours) relative to the seasonal timescale (months). A July fire in California PDT is still a July fire in UTC. The UTC conversion affects which 6-hour window a fire falls into (important for sub-daily analysis) but does not shift fires across months or seasons. Your EDA result showing July as peak month is correct and consistent with the IgnitionNet training data.

---

*All questions from the Presentation Document (Wildfire Risk Prediction, 02.07.2026), the Technical Discussion on Two-Stage Framework, and the Teams channel chat log (June 8 – July 5, 2026) have been addressed above. If new questions arise, share them in the Teams channel and Miguel will update this document.*

---

## SECTION F — Formal Questions from Submitted Documents (July 2026)

This section answers every question exactly as written in the two formal documents submitted by the team. Cross-references to earlier answers are provided where a question was already addressed above.

---

### Outstanding Technical Questions — Stage 1

**Q-A1. Negative Sampling: How were non-fire H3 cells selected to avoid spatial or temporal bias, and what criteria were used in the sensitivity analysis?**

**Selection methodology:** For each positive fire event on calendar date D in UTC window W, exactly 10 absence H3 cells are drawn uniformly at random from all state H3-8 cells that had zero documented fire discoveries in FPA-FOD on that same date D and window W. No neighborhood constraint — negatives come from anywhere in the state that was fire-free on that date and window.

**Temporal bias prevention:** Negatives share the exact calendar date and UTC window as their paired positive. Both the fire cell and non-fire cells experience identical ambient weather on that day. ERC for fire cells and non-fire cells on the same day is approximately equal by construction (mean ERC fire ≈ 64, non-fire ≈ 65). The model cannot exploit weather differences between fire and non-fire rows — it must learn spatial terrain and fuel discrimination. This is the operationally correct task.

**Spatial bias prevention:** Drawing from the full state grid rather than a local neighborhood around the fire prevents an artificially easy local spatial contrast. The model sees fire cells versus cells from anywhere statewide — matching the real dispatch decision boundary.

**Sensitivity analysis criteria:** Retrain at ratios 1:1, 1:5, 1:10, 1:20, 1:50. The criterion is AUPR stability — if AUPR changes by less than ±0.005 across ratios, the ratio is non-influential. AUPR was chosen over F1 (threshold-dependent) or accuracy (dominated by majority class). Results confirmed AUPR was stable across all tested ratios. The 1:10 default was retained for computational efficiency and consistent ~9% positive rate across all four UTC windows.

---

**Q-A2. Missing Weather Data: Should missing HRRR records be removed, interpolated, or imputed? How does preprocessing avoid adversely affecting feature engineering and predictive performance?**

The 24.4% missing figure refers specifically to HRRR per-window features. This missingness is entirely structural — the NOAA HRRR archive was not captured for 2014–2015 and was sparse in 2016. It is not random sensor dropout. Coverage bias analysis confirmed that rows with missing HRRR are statistically identical to HRRR-present rows across every feature and the positive rate (9.07% vs 9.10%). Filtering missing rows does not distort the training distribution.

**Three-tier strategy:**

Tier 1 — Training time: filter out rows where HRRR is unavailable (hpbl_pw is NaN). Do not impute — these years had no archive and fabricated HRRR values would introduce spurious patterns.

Tier 2 — Inference time: zero-fill any missing HRRR field. XGBoost's native sparsity-aware split-finding learns the optimal branch direction for missing values during training, making zero-fill at inference statistically consistent.

Tier 3 — Never: do not use spatial or temporal interpolation for NWP fields. HRRR missingness is structural and archive-based. Interpolating from neighboring cells or dates produces physically meaningless smooth artefacts.

For gridMET daily features (erc, fm100): near-complete CONUS coverage. Any gaps are boundary-cell artefacts from the spatial join — handle with nearest-valid-neighbor fallback during extraction, not date-adjacent imputation.

another path:
Option 1 — Train sub-daily model on all years, zero-fill missing HRRR
Train Model D on all 2014–2017 data. For rows where HRRR is unavailable (2014–2015), fill all six HRRR features with zero and add a binary flag column hrrr_available = 0.
python# Fill missing HRRR with zero, add availability flag
hrrr_cols = ['rh_pw','temp_pw','wind_speed_pw','vpd_pw','hpbl_pw','dswrf_pw']
dataset['hrrr_available'] = dataset['hpbl_pw'].notna().astype(int)
dataset[hrrr_cols] = dataset[hrrr_cols].fillna(0)

# Add hrrr_available as a feature
FEATURE_COLS = [..., 'hrrr_available']  # add to your feature list
XGBoost learns from the hrrr_available flag that zero-filled HRRR rows carry no atmospheric signal. For 2014–2015 rows, the model effectively operates like the daily model — landscape and gridMET carry all the weight. For 2017+ rows, the per-window HRRR adds its contribution on top.
Result: One model, full training set, honest about when HRRR was and was not available. The model is self-aware about its own data quality.
This is the recommended approach. It uses all your ignition data, is transparent, and the hrrr_available flag gives the model explicit information rather than hiding the gap.

Option 2 — Two-stage feature set with a learned fallback
Train on all years but use two separate feature configurations within a single model:
When HRRR is present: landscape + gridMET + HRRR (full 18 features)
When HRRR is absent: landscape + gridMET only (12 features, zero-filled HRRR)
This is mechanically identical to Option 1 but framed differently in the paper. You describe it as "the model gracefully degrades to a daily-equivalent prediction when sub-daily atmospheric data is unavailable, which is the realistic operational scenario when NWP archives are incomplete or forecasts are delayed."
This framing is actually more operationally honest than filtering. In real deployment, there will be days when HRRR is delayed or unavailable. A model that can fall back gracefully to daily-equivalent prediction is more robust than one that requires HRRR for every prediction.

Option 3 — Use gridMET sub-daily approximation for 2014–2015
gridMET is daily, but you can partially reconstruct within-day variation for 2014–2015 using the hour encoding and gridMET's daily values together. You would not have true per-window atmospheric fields, but the model would at least have the diurnal temporal signal via sin_hour and cos_hour.
This is already what happens under Option 1 — the temporal encodings carry the within-day signal for 2014–2015 rows when HRRR is zero-filled. So Option 3 is not really a separate option — it is just a more explicit description of what Option 1 does.

What to update in the paradigm
If you go with Option 1 (recommended), the description changes from:

"Sub-daily model trained on HRRR-filtered rows (2016–2017)"

To:

"Sub-daily model trained on full 2014–2017 dataset with per-window HRRR where available (2016–2017) and graceful degradation to daily-equivalent features for years with no HRRR archive (2014–2015). An explicit binary flag hrrr_available is included as a feature so the model learns to weight HRRR appropriately given archive coverage."

This is a more honest and more operationally meaningful model. It does not pretend HRRR was always available. It uses all your data. And it produces a single model that works in both HRRR-present and HRRR-absent conditions — which is exactly what an operational system needs.

The one thing to check after implementing Option 1
Run the HRRR coverage bias analysis to confirm that 2014–2015 rows (zero-filled HRRR) are not systematically different from 2016–2017 rows in landscape or gridMET features. If they are statistically similar — which Miguel's analysis confirmed they are (positive rate 9.07% vs 9.10%, landscape features within 5%) — then zero-filling is safe and the full training set is usable without distortion.
---

**Q-A3. Feature Correlation: Retain all correlated variables or apply RFE/SHAP/permutation importance before training?**

Retain all correlated features. Do not apply RFE before training.

**Why correlation does not require removal for tree models:** XGBoost uses greedy split-finding at each node. Correlated features do not destabilize tree models the way they destabilize linear models. FLEP4–CFL correlation (r ≈ 0.96) reflects physical overlap in what they measure, but at the H3 cell level each captures slightly different spatial variation that contributes independent discriminative signal.

**Why RFE is the wrong tool:** When features are correlated, their importance is split between them, making each look individually weaker than it is. RFE therefore tends to remove genuinely informative variables from correlated groups — it would likely drop either FLEP4 or CFL despite both contributing to AUPR.

**The correct tool is SHAP — but apply it AFTER training, not before.** SHAP correctly computes each feature's marginal contribution given all others, even under correlation. Use SHAP to understand which features are doing most work and to identify candidates for removal if a simpler model is needed. Do not use raw XGBoost gain for importance — it is biased toward high-cardinality continuous features.

**ERC–FM100 inverse correlation (r ≈ −0.92):** Physically expected — drier fuels have higher ERC and lower moisture simultaneously. Retain both — ERC integrates a 10-day rolling moisture trajectory while FM100 captures the 5-day lagged trend. They answer slightly different sub-questions about fuel drying.

Bottom line: Train with all 19 features. Run SHAP after training. Only remove a feature if SHAP shows near-zero marginal contribution AND leave-one-out AUPR confirms ΔAUPR < 0.002.

---

### Outstanding Technical Questions — Stage 2 (B through E)

**Q-B4 through Q-E11 (dataset integration, target variable, spatial matching key, missing environmental damage indicators, damage labels as features vs. targets, stage coupling, tabular vs. multimodal, and additional engineered features):**

All of these are Stage 2 questions. Stage 2 is NOT your current assignment. Complete Stage 1 first.

When Stage 2 becomes your active scope, the answers are in Section E of this README (Q18–Q23). Summary of key points:
- Use CAL FIRE Damage Inspection dataset, not DamageTriage-Bench, for California-wide training
- Predict individual building damage severity (destroyed/major/minor/no damage), do not aggregate to H3 cell level
- Primary spatial join key: fire incident identifier → fire perimeter → lat/lon proximity (in that priority order)
- dNBR from MTBS is the correct burn severity input feature, not a target label — the target label is the CAL FIRE damage classification
- Maintain Stage 1 and Stage 2 as independent models; include Stage 1 probability as one input feature to Stage 2
- Tabular workflow is sufficient and is the correct starting point for Stage 2

Do not pursue Stage 2 dataset collection or model architecture until Stage 1 baseline is validated and running.

---

###  Questions from the Presentation Document

**Q-H1. The source for some remaining features (pr, tmmx, erc, vpd) is unclear. What is the appropriate data source for these features for non-fire samples?**

All four come from **gridMET** — not MODIS, not NASA, not NAVA. gridMET is a single unified source maintained by the Climatology Lab at UC Merced (climatologylab.org/gridmet.html), cited as Abatzoglou 2013.

Complete source map for all features in the model:

| Feature | Source | How to access |
|---|---|---|
| avg_burn_prob, whp | USFS FSim / WHP | firelab.org |
| flep4, cfl | LANDFIRE LF2022 | landfire.gov |
| erc | gridMET | `erc_{year}.nc` from northwestknowledge.net |
| fm100_5D_mean | gridMET | `fm100_{year}.nc` — compute 5-day rolling mean |
| pr | gridMET | `pr_{year}.nc` |
| tmmx | gridMET | `tmmx_{year}.nc` — convert K to °C |
| vpd (daily) | gridMET | `vpd_{year}.nc` |
| rh_pw, temp_pw, wind_speed_pw, vpd_pw, hpbl_pw, dswrf_pw | NOAA HRRR | Herbie library + AWS S3 |
| sin/cos month/hour | Computed | From window timestamp |
| lat, lon | H3 centroid | h3.cell_to_latlng(cell) |

The extraction method for non-fire cells is identical to fire cells for every feature. Nothing is copied from the paired positive — each negative cell has its own independently extracted values at its own centroid coordinates.

If your EDA dashboard references MODIS or NASA as sources, those variables came from the FPA-FOD Attributes dataset (the 309-column version). That dataset should not be used as a feature source for this project. Strip those variables out and use only the 19-feature set in this README.

---

**Q-H2. What spatial join or mapping technique is used to associate each sampled non-fire H3 cell with predictor features from source datasets?**

H3 cell centroid coordinates are the universal spatial key. The method is nearest-neighbor matching for all raster and gridded sources. No polygon intersection or area-weighted averaging is used — point-to-nearest-pixel is appropriate at H3-8 resolution (~0.74 km²) for both 4km gridMET and 3km HRRR.

**For LANDFIRE GeoTIFFs (avg_burn_prob, whp, flep4, cfl):**
```python
import h3, rasterio, numpy as np
from pyproj import Transformer

def extract_landfire(h3_cell, raster_path):
    lat, lon = h3.cell_to_latlng(h3_cell)
    with rasterio.open(raster_path) as src:
        t = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
        x, y = t.transform(lon, lat)
        row, col = rasterio.transform.rowcol(src.transform, x, y)
        row = int(np.clip(row, 0, src.height - 1))
        col = int(np.clip(col, 0, src.width - 1))
        val = src.read(1)[row, col]
        return float(val) if val != src.nodata else np.nan
```

**For gridMET NetCDF (erc, fm100, pr, tmmx, vpd):**
Build a cKDTree from gridMET grid point coordinates once, reuse for all H3 cells. Query by (lat, lon) to find the nearest 4km grid point. Extract the daily value for the assigned date. This gives each cell its own independently extracted weather value.

**For HRRR (per-window fields):** Same cKDTree approach using HRRR's 3km native grid. Query nearest grid point. Extract the 6-hourly analysis value for the assigned UTC window.

**Critical rule:** Non-fire cells DO NOT inherit any spatial values from the paired positive fire cell. Each negative cell is a completely independent observation. If you copy feature values from the fire cell to the non-fire cell, the model will learn a trivial pattern and results will be invalid.

---

**Q-H3. How are temporal features aligned for non-fire samples? How are observation date and time determined when extracting dynamic features like weather and vegetation?**

Non-fire cells inherit the calendar date and UTC window from the positive fire event they were matched to. This is the core of the DAY_MATCHED design.

**Concrete example:** A fire discovered on 2018-11-08 at 10:30 AM PDT converts to 18Z UTC on 2018-11-08. Its 10 matched negative cells are assigned date = 2018-11-08, window = 18Z. For each negative cell, gridMET ERC is extracted at that cell's own centroid for 2018-11-08. For each negative cell, HRRR fields are extracted at that cell's own centroid for the 18Z analysis cycle on 2018-11-08.

**What is inherited from the positive:** Only the date and UTC window slot. This defines which weather time slice to extract.

**What is NOT inherited:** The spatial coordinates, and therefore the actual weather and landscape values. Each negative cell's own centroid determines what those values are.

**Temporal encodings (sin_month, cos_month, sin_hour, cos_hour):** Computed directly from the assigned window timestamp. These are identical for the fire cell and all its matched negatives on the same date and window — which is correct, since the diurnal and seasonal context is the same for all observations at the same (date, window).

**DST edge case:** When converting local discovery times to UTC, use DST-aware timezone conversion (pytz with localize + tz_convert), not a fixed UTC offset. A fixed UTC+8 offset incorrectly assigned ~23% of California fire records to the wrong UTC window during fire season when PDT (UTC-7) is in effect. The code for correct DST handling is in the Labels section of this README.

**Vegetation indices (NDVI, if used):** Use the nearest 8-day or 16-day MODIS composite prior to the assigned date. Do not interpolate across composite periods.

---

