# Texas Ignition Dataset — README
**File:** `final_training_dataset_tx_22.07.2026_landfire.xlsx`
**Prepared:** 2026-07-22

---

## Overview

| Property | Value |
|---|---|
| Rows | 376,233 |
| Columns | 42 |
| Unique H3-8 cells | 317,142 |
| Date range | 2014-01-01 – 2020-12-06 |
| Spatial extent | Lat 25.9°–36.6°N, Lon 106.7°–93.5°W (Texas) |
| Positive label (fire) | 34,203 rows (9.09%) |
| Non-fire rows | 342,030 (90.91%) |
| Duplicate rows (pre-existing) | 454 |
| Rows with missing weather | 24,954 (6.6%, `gridmet_missing=1`) |

Each row is a **6-hour window observation** for one H3 level-8 hexagonal cell (~0.74 km²). The sampling strategy preserves a fixed 9.09% positive rate across all years (see label distribution below).

---

## Column Groups

| Group | Columns | Description |
|---|---|---|
| **Index** | `h3_cell`, `date_utc`, `window_hour`, `window_6h_utc` | Unique identifier per observation |
| **Label** | `label` | 1 = ignition event in this 6h window, 0 = no ignition |
| **Spatial** | `centroid_lat`, `centroid_lon` | H3 cell centroid (WGS84) |
| **Fire history** | `fire_year`, `fire_count`, `has_fire_history` | Historical fire occurrence per cell |
| **Burnable** | `burnable` | 1 = cell has burnable fuel, 0 = non-burnable (urban/water/agriculture) |
| **Landscape** | `avg_burn_prob`, `whp`, `flep4`, `cfl`, `cbd`, `cbh` | Static wildfire risk features (see below) |
| **Daily weather** | `erc`, `fm100`, `vpd`, `vs`, `rmax`, `rmin`, `tmmx`, `pr` | gridMET daily weather at cell centroid |
| **5-day weather** | `erc_5D_mean/max`, `fm100_5D_mean/min`, `vpd_5D_mean/max`, `vs_5D_mean/max`, `rmax_5D_mean/min`, `tmmx_5D_mean/max` | 5-day rolling statistics |
| **Flags** | `gridmet_missing` | 1 = gridMET data unavailable for this cell/date |
| **Temporal** | `sin_month`, `cos_month`, `sin_hour`, `cos_hour` | Cyclical encoding of month and 6h window |

---

## Label Distribution

### By year

| Year | Fire rows | Total rows | Rate |
|---|---|---|---|
| 2014 | 5,382 | 59,202 | 9.09% |
| 2015 | 5,441 | 59,851 | 9.09% |
| 2016 | 5,730 | 63,030 | 9.09% |
| 2017 | 6,362 | 69,982 | 9.09% |
| 2018 | 5,562 | 61,182 | 9.09% |
| 2019 | 3,970 | 43,670 | 9.09% |
| 2020 | 1,756 | 19,316 | 9.09% |

The fixed 9.09% rate per year reflects a controlled sampling design, not the raw ignition frequency. 2020 has fewer total rows because the archive ends December 2020-12-06.

### By 6-hour window

| Window (UTC) | Fire rows | Total rows |
|---|---|---|
| 00:00 | 479 | 5,269 |
| 06:00 | 88 | 968 |
| 12:00 | 20,300 | 223,300 |
| 18:00 | 13,336 | 146,696 |

Most observations fall in the 12:00–18:00 UTC windows (06:00–12:00 CST), consistent with peak afternoon fire activity in Texas.

---

## Descriptive Statistics

### Landscape features

| Feature | Mean | Std | Min | P25 | Median | P75 | Max | Zeros |
|---|---|---|---|---|---|---|---|---|
| `avg_burn_prob` | 4.155 | 3.371 | 0 | 0 | 5.0 | 7.0 | 11.0 | 32.5% |
| `whp` | 2.671 | 2.389 | 0 | 0 | 3.0 | 5.0 | 9.0 | 32.5% |
| `cfl` (ft) | 3.799 | 5.330 | 0 | 0 | 1.5 | 7.0 | 110.0 | 43.1% |
| `flep4` (prob) | 0.296 | 0.398 | 0 | 0 | 0 | 0.9 | 0.9 | 57.1% |
| `cbd` (kg/m³) | 0.012 | 0.039 | 0 | 0 | 0 | 0 | 0.45 | 85.2% |
| `cbh` (m) | 0.632 | 2.093 | 0 | 0 | 0 | 0 | 10.0 | 85.2% |

High zero rates for `cbd` and `cbh` reflect Texas's predominantly grassland/shrubland landscape (no tree canopy). Zeros in `avg_burn_prob`/`whp` correspond to non-burnable or unmodeled cells.

### Landscape features: fire vs. no-fire

| Feature | Fire mean | No-fire mean | Ratio |
|---|---|---|---|
| `avg_burn_prob` | 5.808 | 3.990 | 1.46× |
| `whp` | 4.013 | 2.537 | 1.58× |
| `cfl` (ft) | 4.117 | 3.768 | 1.09× |
| `flep4` (prob) | 0.314 | 0.294 | 1.07× |
| `cbd` (kg/m³) | 0.018 | 0.012 | 1.48× |
| `cbh` (m) | 1.072 | 0.588 | 1.82× |

All landscape features are elevated at fire locations, confirming signal alignment. `whp` and `cbh` show the strongest relative separations.

### Weather features (gridMET)

| Feature | Description | Units | Mean | Std | Min | P25 | Median | P75 | Max | NaN |
|---|---|---|---|---|---|---|---|---|---|---|
| `erc` | Energy Release Component | — | 43.49 | 16.52 | 0 | 32 | 42 | 53 | 111 | 24,954 |
| `fm100` | 100-hr fuel moisture | % | 12.52 | 3.70 | 2.2 | 9.8 | 12.5 | 15.2 | 28.5 | 24,954 |
| `vpd` | Vapor pressure deficit | kPa | 1.45 | 0.79 | 0 | 0.87 | 1.32 | 1.95 | 5.09 | 24,954 |
| `vs` | Wind speed | m/s | 4.27 | 1.52 | 0.3 | 3.2 | 4.0 | 5.1 | 18.7 | 24,954 |
| `rmax` | Max relative humidity | % | 78.75 | 17.51 | 11 | 67.2 | 81.1 | 93.9 | 100 | 24,954 |
| `rmin` | Min relative humidity | % | 28.27 | 15.11 | 1 | 17.3 | 26.4 | 37.3 | 100 | 24,954 |
| `tmmx` | Max temperature | °C | 27.00 | 8.54 | −13.6 | 21.2 | 28.5 | 34.1 | 46.3 | 24,954 |
| `pr` | Precipitation | mm | 1.30 | 6.06 | 0 | 0 | 0 | 0 | 624.8 | 24,954 |

### Weather features: fire vs. no-fire

| Feature | Fire mean | No-fire mean | Note |
|---|---|---|---|
| `erc` | 39.68 | 43.90 | Lower ERC at fire locations — counter-intuitive; may reflect sampling design or temporal mismatch |
| `fm100` | 13.17 | 12.45 | Slightly higher moisture at fire locations |
| `vpd` | 1.46 | 1.45 | Near-identical |
| `vs` | 4.38 | 4.25 | Slightly higher wind at fire locations |
| `rmax` | 81.97 | 78.40 | Higher RH max at fire |
| `rmin` | 30.03 | 28.08 | Higher RH min at fire |
| `tmmx` | 28.07 | 26.88 | Slightly warmer at fire locations |
| `pr` | 1.17 | 1.32 | Lower precipitation at fire locations (expected) |

Note: weather features capture the day-of-fire conditions and may be subject to temporal offset effects depending on how ignition time aligns with gridMET daily aggregates.

---

## Landscape Feature Sources

| Feature | Description | Source | Units | Notes |
|---|---|---|---|---|
| `avg_burn_prob` | Annual burn probability | TX WRC landscape archive | 0–11 scale | Scale is NOT 0–1 probability |
| `whp` | Wildfire Hazard Potential | TX WRC landscape archive | 0–9 class | |
| `cfl` | Weighted-average flame length | TxWRAP `cFL.tif` (WildEST simulation) | feet | Classes 1–9 mapped to ft midpoints via official VAT labels |
| `flep4` | Probability of flame length > 4 ft | TxWRAP `xmanualctrl_4.tif` (WildEST simulation) | 0–1 probability | Classes 1–6 mapped to probability midpoints |
| `cbd` | Canopy Bulk Density | LANDFIRE LF2022 ImageServer | kg/m³ | 300m resolution, TX clip |
| `cbh` | Canopy Base Height | LANDFIRE LF2022 ImageServer | meters | 300m resolution, TX clip |

**TxWRAP** = Texas Wildfire Risk Assessment Portal (Pyrologix/WildEST), produced for Texas A&M Forest Service. Same WildEST simulation framework used for the California training data (`CFL_CA.tif`, `FLEP4_CA.tif`).

### CFL class mapping (TxWRAP VAT labels)

| Class | VAT label | Assigned value (ft) |
|---|---|---|
| 0 | No data | 0.0 |
| 1 | 0 ft | 0.0 |
| 2 | < 1 ft | 0.5 |
| 3 | 1–2 ft | 1.5 |
| 4 | 2–4 ft | 3.0 |
| 5 | 4–10 ft | 7.0 |
| 6 | 10–21 ft | 15.5 |
| 7 | 21–46 ft | 33.5 |
| 8 | 46–100 ft | 73.0 |
| 9 | > 100 ft | 110.0 |

### FLEP4 class mapping (TxWRAP VAT labels)

| Class | VAT label | Assigned value (prob) |
|---|---|---|
| 0 | No data | 0.0 |
| 1 | 0% | 0.0 |
| 2 | > 0–20% | 0.10 |
| 3 | 20–40% | 0.30 |
| 4 | 40–60% | 0.50 |
| 5 | 60–80% | 0.70 |
| 6 | 80–100% | 0.90 |

---

## Caveats

### 1. Landscape features are discrete (CFL and FLEP4)
The TxWRAP archive published `cfl` and `flep4` as classified rasters. All pixels in the same class receive the same representative value (class midpoint). The California training data used continuous 30m simulation outputs. This discretization is a data limitation of the TX archive, not a processing error.

### 2. CFL scale extends beyond California range
CA CFL max ≈ 48 ft; TX CFL max = 110 ft (class 9: >100 ft, dense East Texas piney woods). This is a real physical difference, not a scale mismatch.

### 3. CBD and CBH coverage is low (14.8% non-zero)
Texas is predominantly grassland and shrubland. Zero CBD/CBH correctly reflects the absence of tree canopy — it is not missing data.

### 4. avg_burn_prob scale (0–11, not 0–1)
The TX archive stores annual burn probability on a 0–11 scale. Normalize if combining with national WRC products or the CA training data.

### 5. Positive rate is fixed at 9.09% by design
The positive rate is exactly 9.09% in every year — this is a result of the sampling strategy, not the raw ignition frequency. Do not interpret as an empirical fire probability.

### 6. Weather signal is weak relative to CA
Fire vs. no-fire means for ERC and VPD are nearly identical (and ERC is slightly *lower* at fire locations). This may reflect temporal offset between gridMET daily aggregates and ignition window times, or that landscape/fuel features carry more predictive signal in TX than weather.

### 7. Date range ends December 2020
The dataset covers 2014–2020, not 2022. This reflects the temporal extent of the TX WRC archive.

### 8. 454 duplicate rows (pre-existing)
Exact duplicates on (h3_cell, date_utc, window_hour) were present in the original input. Drop before training:
```python
df.drop_duplicates(subset=['h3_cell', 'date_utc', 'window_hour'], inplace=True)
```

### 9. ~24,954 rows have NaN weather features
Rows with `gridmet_missing = 1` have NaN for all weather features. Handle with imputation or exclusion before training.

### 10. 1,594 rows have NaN fire_count / burnable
A small set of cells could not be joined to the fire history or burnable layer. These are edge/boundary cells.

---

## Quick-Start

```python
import pandas as pd

df = pd.read_excel('final_training_dataset_tx_22.07.2026_landfire.xlsx')

# Recommended pre-processing before training
df.drop_duplicates(subset=['h3_cell', 'date_utc', 'window_hour'], inplace=True)
df = df[df['gridmet_missing'] != 1]   # drop rows with missing weather

# Note: avg_burn_prob is on 0-11 scale - normalize if needed
# df['avg_burn_prob'] = df['avg_burn_prob'] / 11.0

X = df.drop(columns=['label', 'h3_cell', 'date_utc', 'window_6h_utc'])
y = df['label']
```

---

## File Locations

| File | Path |
|---|---|
| **Output dataset** | `miguel_shared/alphaearth_nds/final_training_dataset_tx_22.07.2026_landfire.xlsx` |
| Fill script (v4) | `miguel_shared/alphaearth_nds/fill_landfire_tx.py` |
| CBD raster (cached) | `miguel_shared/alphaearth_nds/90%_ig_dec/Transferibility_OutOfState/data/national_rasters/CBD_TX300m.tif` |
| CBH raster (cached) | `miguel_shared/alphaearth_nds/90%_ig_dec/Transferibility_OutOfState/data/national_rasters/CBH_TX300m.tif` |
| TxWRAP source rasters | `archived/shared/TX_H3L10/Texas_wildfire/Data/` |

---

## Contact
Questions about this dataset: mte1224@tamu.edu
