# Feature Data Guide — Wildfire Ignition Model

This document covers data sources, download procedures, and preprocessing notes
for building a wildfire ignition model following the IgnitionNet methodology.

---

## Feature overview

| Group | Features | Source | Temporal? |
|---|---|---|---|
| Landscape (static) | avg_burn_prob, whp, flep4, cfl | LANDFIRE / USFS FSim | No — join once |
| Daily weather | erc, fm100_5D_mean | gridMET | Yes — per date |
| Sub-daily weather | rh_pw, temp_pw, wind_speed_pw, vpd_pw, hpbl_pw, dswrf_pw | HRRR via Herbie | Yes — per 6h window |
| Temporal | sin_month, cos_month, sin_hour, cos_hour | Derived | — |
| Location | lat, lon | H3 centroid | — |

---

## Step 1 — LANDFIRE landscape features (one-time download, then automated)

Four static rasters are needed. **CFL and FLEP4** come from LANDFIRE.
**WHP and Burn Probability** come from the USFS national products.

> **Note:** firelab.org is currently inaccessible. Use the alternative sources below.

### Download the four GeoTIFFs (browser, one-time per file)

| Feature | Source | Steps |
|---|---|---|
| `cfl` — Canopy Fuel Load | landfire.gov | Go to **landfire.gov/bulk/downloader.php** → Select LF2022 → Canopy → Canopy Fuel Load → CONUS → Download ZIP → extract `LC22_CFL_220.tif` |
| `flep4` — Flame Length Exceedance ≥4 ft | landfire.gov | Same bulk downloader → Fire Behavior → Flame Length Exceedance Probability → select 4-ft threshold → extract `LC22_FLEP4_220.tif` |
| `whp` — Wildfire Hazard Potential | USFS RDS Archive | Go to **fs.usda.gov/rds/archive** → search `RDS-2015-0047` → download `whp2023_cnt.zip` → extract `.tif` |
| `avg_burn_prob` — Burn Probability | USFS RDS Archive | Same archive → search `FSim national burn probability` → download and extract `.tif` |

Rename and place all four files in a single folder:

```
data/national_rasters/
    BP_CONUS.tif
    WHP_CONUS.tif
    CFL_CONUS.tif
    FLEP4_CONUS.tif
```

### Sample to H3 grid (fully automated)

Run this script once per state. It generates all H3-8 cells within the state
bounding box and samples each raster at the cell centroids via rasterio.

```python
# fetch_landscape.py
# Usage: python fetch_landscape.py CA
#        python fetch_landscape.py TX

import sys
import numpy as np
import pandas as pd
import h3
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform
from rasterio.transform import rowcol
from pathlib import Path

RASTER_DIR = Path("data/national_rasters")
OUT_DIR    = Path("data")

RASTERS = {
    "avg_burn_prob": RASTER_DIR / "BP_CONUS.tif",
    "whp":           RASTER_DIR / "WHP_CONUS.tif",
    "flep4":         RASTER_DIR / "FLEP4_CONUS.tif",
    "cfl":           RASTER_DIR / "CFL_CONUS.tif",
}

STATE_BBOX = {
    "CA": dict(lat_min=32.5, lat_max=42.0, lon_min=-124.5, lon_max=-114.0),
    "TX": dict(lat_min=25.8, lat_max=36.6, lon_min=-106.7, lon_max=-93.5),
    "OR": dict(lat_min=41.9, lat_max=46.5, lon_min=-124.6, lon_max=-116.3),
}

def generate_h3_cells(bbox, res=8, step=0.008):
    lats = np.arange(bbox["lat_min"], bbox["lat_max"], step)
    lons = np.arange(bbox["lon_min"], bbox["lon_max"], step)
    lat_g, lon_g = np.meshgrid(lats, lons)
    cells = [h3.latlng_to_cell(la, lo, res)
             for la, lo in zip(lat_g.ravel(), lon_g.ravel())]
    df = (pd.DataFrame({"h3_cell": cells})
            .drop_duplicates("h3_cell")
            .reset_index(drop=True))
    centroids = [h3.cell_to_latlng(c) for c in df["h3_cell"]]
    df["lat"] = [c[0] for c in centroids]
    df["lon"] = [c[1] for c in centroids]
    return df

def sample_raster(tif_path, lats, lons, fill=0.0):
    with rasterio.open(tif_path) as src:
        xs, ys = warp_transform(
            CRS.from_epsg(4326), src.crs, lons.tolist(), lats.tolist()
        )
        rows, cols = rowcol(src.transform, xs, ys)
        rows, cols = np.asarray(rows), np.asarray(cols)
        valid = (rows >= 0) & (rows < src.height) & \
                (cols >= 0) & (cols < src.width)
        values = np.full(len(lats), fill, dtype=np.float32)
        n_pixels = src.width * src.height
        if n_pixels <= 500_000_000:
            data = src.read(1)
            values[valid] = data[rows[valid], cols[valid]].astype(np.float32)
        else:
            # Large raster — batch point sampling to avoid OOM
            valid_idx = np.where(valid)[0]
            coords = [(float(xs[i]), float(ys[i])) for i in valid_idx]
            sampled = []
            for start in range(0, len(coords), 50_000):
                sampled += [v[0] for v in src.sample(coords[start:start+50_000])]
            values[valid] = np.array(sampled, dtype=np.float32)
        if src.nodata is not None:
            values[values == src.nodata] = fill
        values[~valid] = fill
    return values

def main(state):
    bbox = STATE_BBOX[state]
    print(f"Generating H3-8 cells for {state}...")
    df = generate_h3_cells(bbox)
    print(f"  {len(df):,} unique cells")

    for feat, path in RASTERS.items():
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {feat}")
            df[feat] = float("nan")
            continue
        print(f"  Sampling {feat}...")
        df[feat] = sample_raster(path, df["lat"].values, df["lon"].values)
        print(f"    mean={df[feat].mean():.4f}  max={df[feat].max():.4f}  "
              f"nan={df[feat].isna().mean()*100:.1f}%")

    out = OUT_DIR / f"landscape_h3_{state.lower()}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved → {out}  ({len(df):,} rows × {len(df.columns)} cols)")

if __name__ == "__main__":
    state = sys.argv[1].upper() if len(sys.argv) > 1 else "CA"
    main(state)
```

Output: `data/landscape_h3_ca.parquet`, `data/landscape_h3_tx.parquet`, etc.

---

## Step 2 — gridMET daily weather (fully automated)

gridMET is downloaded directly from the Northwest Knowledge Network HTTPS server.
No account or API key required.

```python
# fetch_gridmet.py
# Usage: python fetch_gridmet.py CA 2014 2020
#        python fetch_gridmet.py TX 2014 2020

import sys
import tempfile
import numpy as np
import pandas as pd
import requests
import netCDF4 as nc
from pathlib import Path

OUT_DIR = Path("data")

GRIDMET_URL = "https://www.northwestknowledge.net/metdata/data/{var}_{year}.nc"

GRIDMET_VARS = {
    "erc":   "energy_release_component-g",
    "fm100": "dead_fuel_moisture_100hr",
    "pr":    "precipitation_amount",
    "tmmx":  "air_temperature",
    "vpd":   "mean_vapor_pressure_deficit",
}

# gridMET CONUS grid
GM_LAT_NORTH = 49.4
GM_LON_WEST  = -124.76666663
GM_DLAT      = 1 / 24
GM_DLON      = 1 / 24

STATE_BBOX = {
    "CA": dict(lat_min=32.5, lat_max=42.0, lon_min=-124.5, lon_max=-114.0),
    "TX": dict(lat_min=25.8, lat_max=36.6, lon_min=-106.7, lon_max=-93.5),
    "OR": dict(lat_min=41.9, lat_max=46.5, lon_min=-124.6, lon_max=-116.3),
}

def latlon_to_idx(lat, lon):
    row = round((GM_LAT_NORTH - lat) / GM_DLAT)
    col = round((lon - GM_LON_WEST)  / GM_DLON)
    return row, col

def download_nc(var_key, year, bbox):
    var_name = GRIDMET_VARS[var_key]
    url = GRIDMET_URL.format(var=var_name, year=year)
    print(f"  Downloading {var_key} {year}...", end=" ", flush=True)

    r0, c0 = latlon_to_idx(bbox["lat_max"], bbox["lon_min"])
    r1, c1 = latlon_to_idx(bbox["lat_min"], bbox["lon_max"])
    r0, r1 = max(0, r0-1), r1+2
    c0, c1 = max(0, c0-1), c1+2

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        tmp_path = f.name
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)

    rows = []
    with nc.Dataset(tmp_path) as ds:
        lats = ds.variables["lat"][r0:r1]
        lons = ds.variables["lon"][c0:c1]
        # gridMET day dimension
        time_var = ds.variables["day"]
        times = nc.num2date(time_var[:], time_var.units)
        dates = [str(t)[:10] for t in times]
        data = ds.variables[var_name][:, r0:r1, c0:c1]  # (days, lat, lon)

    Path(tmp_path).unlink(missing_ok=True)

    for di, date_str in enumerate(dates):
        for ri, lat in enumerate(lats):
            for ci, lon in enumerate(lons):
                val = float(data[di, ri, ci])
                rows.append({"date": date_str, "lat": float(lat),
                             "lon": float(lon), var_key: val})
    df = pd.DataFrame(rows)
    print(f"{len(df):,} rows")
    return df

def main(state, year_min, year_max):
    bbox = STATE_BBOX[state]
    frames = []
    for year in range(year_min, year_max + 1):
        yr_frames = []
        for var_key in ["erc", "fm100"]:          # add others if needed
            df = download_nc(var_key, year, bbox)
            yr_frames.append(df.set_index(["date", "lat", "lon"]))
        merged = yr_frames[0].join(yr_frames[1:], how="outer").reset_index()
        frames.append(merged)

    out_df = pd.concat(frames, ignore_index=True)

    # 5-day lagged FM100 mean
    out_df = out_df.sort_values(["lat", "lon", "date"])
    out_df["fm100_5D_mean"] = (
        out_df.groupby(["lat", "lon"])["fm100"]
              .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    out = OUT_DIR / f"gridmet_{state.lower()}_daily.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(f"Saved → {out}  ({len(out_df):,} rows)")

if __name__ == "__main__":
    state    = sys.argv[1].upper() if len(sys.argv) > 1 else "CA"
    yr_min   = int(sys.argv[2]) if len(sys.argv) > 2 else 2014
    yr_max   = int(sys.argv[3]) if len(sys.argv) > 3 else 2020
    main(state, yr_min, yr_max)
```

Output: `data/gridmet_ca_daily.parquet` with columns `date, lat, lon, erc, fm100, fm100_5D_mean`.

---

## Step 3 — HRRR sub-daily weather (fully automated via Herbie)

The correct approach downloads the full HRRR grid for each analysis cycle, builds
a one-time nearest-neighbour mapping from HRRR native points to your H3 grid, and
samples each cell individually. Do **not** average over the state bbox — each H3
cell must receive its own HRRR value.

`vpd_pw` and `wind_speed_pw` are derived from HRRR outputs, not fetched directly.

```python
# fetch_hrrr.py
# Usage: python fetch_hrrr.py CA 2017 2020
#        python fetch_hrrr.py TX 2017 2020

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
from herbie import Herbie

OUT_DIR = Path("data")

STATE_BBOX = {
    "CA": dict(lat_min=32.5, lat_max=42.0, lon_min=-124.5, lon_max=-114.0),
    "TX": dict(lat_min=25.8, lat_max=36.6, lon_min=-106.7, lon_max=-93.5),
    "OR": dict(lat_min=41.9, lat_max=46.5, lon_min=-124.6, lon_max=-116.3),
}

WINDOWS = [0, 6, 12, 18]

# HRRR GRIB search strings (f00 = analysis cycle, no forecast lead)
SEARCH = {
    "temp_k":  ":TMP:2 m above ground:anl",
    "rh":      ":RH:2 m above ground:anl",
    "ugrd":    ":UGRD:10 m above ground:anl",
    "vgrd":    ":VGRD:10 m above ground:anl",
    "hpbl":    ":HPBL:surface:anl",
    "dswrf":   ":DSWRF:surface:anl",
}


def build_grid_mapping(bbox):
    """
    One-time setup: build a ~4 km output grid matching gridMET convention,
    then find the nearest HRRR native grid point for each output cell.
    Returns (out_lats, out_lons, state_indices, nn_idx) where:
      - out_lats/out_lons : the output grid coordinates
      - state_indices     : which HRRR flat indices fall inside the state bbox
      - nn_idx            : for each output cell, index into state_indices array
    """
    print("Building HRRR → H3 grid mapping (one-time)...")
    step = 1.0 / 24.0
    lats = np.arange(bbox["lat_min"], bbox["lat_max"] + step, step)
    lons = np.arange(bbox["lon_min"], bbox["lon_max"] + step, step)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    out_lats = lat_g.ravel().astype(np.float32)
    out_lons = lon_g.ravel().astype(np.float32)

    # Fetch the HRRR grid template once
    H = Herbie("2019-06-01 18:00", model="hrrr", product="sfc", fxx=0, verbose=False)
    ds = H.xarray(SEARCH["temp_k"])
    h_lats = ds.latitude.values.ravel().astype(np.float32)
    h_lons = ds.longitude.values.ravel().astype(np.float32)
    h_lons = np.where(h_lons > 180, h_lons - 360, h_lons)   # convert 0-360 → -180-180

    # Mask to state bbox
    mask = ((h_lats >= bbox["lat_min"]) & (h_lats <= bbox["lat_max"]) &
            (h_lons >= bbox["lon_min"]) & (h_lons <= bbox["lon_max"]))
    state_indices = np.where(mask)[0].astype(np.int32)

    # Nearest-neighbor: each output cell → closest HRRR state point
    tree = cKDTree(np.column_stack([h_lons[state_indices], h_lats[state_indices]]))
    _, nn_idx = tree.query(np.column_stack([out_lons, out_lats]), k=1)

    print(f"  Output cells: {len(out_lats):,} | HRRR points in state: {len(state_indices):,}")
    return out_lats, out_lons, state_indices, nn_idx.astype(np.int32)


def fetch_window(date_str, hour, state_indices, nn_idx):
    dt_str = f"{date_str} {hour:02d}:00"
    try:
        H = Herbie(dt_str, model="hrrr", product="sfc", fxx=0, verbose=False)

        def extract(search):
            ds = H.xarray(search)
            var = [v for v in ds.data_vars
                   if v not in ("latitude","longitude","time","valid_time",
                                "step","heightAboveGround","surface")][0]
            vals = ds[var].values.ravel().astype(np.float32)
            return vals[state_indices][nn_idx]

        temp_k = extract(SEARCH["temp_k"])
        rh     = extract(SEARCH["rh"])
        ugrd   = extract(SEARCH["ugrd"])
        vgrd   = extract(SEARCH["vgrd"])
        hpbl   = extract(SEARCH["hpbl"])
        dswrf  = extract(SEARCH["dswrf"])

        T_c  = temp_k - 273.15
        es   = (0.6108 * np.exp(17.27 * T_c / (T_c + 237.3))).astype(np.float32)

        return {
            "window_hour":    hour,
            "date":           date_str,
            "temp_pw":        T_c,
            "rh_pw":          rh,
            "wind_speed_pw":  np.sqrt(ugrd**2 + vgrd**2).astype(np.float32),
            "vpd_pw":         (es * (1.0 - rh / 100.0)).astype(np.float32),
            "hpbl_pw":        hpbl,
            "dswrf_pw":       dswrf,
        }
    except Exception:
        return None


def main(state, year_min, year_max):
    bbox = STATE_BBOX[state]
    out_lats, out_lons, state_indices, nn_idx = build_grid_mapping(bbox)

    records = []
    d = datetime(year_min, 1, 1)
    while d <= datetime(year_max, 12, 31):
        date_str = d.strftime("%Y-%m-%d")
        for hour in WINDOWS:
            res = fetch_window(date_str, hour, state_indices, nn_idx)
            if res is None:
                d += timedelta(days=1) if hour == WINDOWS[-1] else timedelta(0)
                continue
            n = len(out_lats)
            df_w = pd.DataFrame({
                "date":           date_str,
                "gridmet_lat":    out_lats,
                "gridmet_lon":    out_lons,
                "window_hour":    hour,
                "temp_pw":        res["temp_pw"],
                "rh_pw":          res["rh_pw"],
                "wind_speed_pw":  res["wind_speed_pw"],
                "vpd_pw":         res["vpd_pw"],
                "hpbl_pw":        res["hpbl_pw"],
                "dswrf_pw":       res["dswrf_pw"],
            })
            records.append(df_w)
        d += timedelta(days=1)

    out_df = pd.concat(records, ignore_index=True)
    out = OUT_DIR / f"hrrr_{state.lower()}_subdaily.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(f"Saved → {out}  ({len(out_df):,} rows)")


if __name__ == "__main__":
    state  = sys.argv[1].upper() if len(sys.argv) > 1 else "CA"
    yr_min = int(sys.argv[2]) if len(sys.argv) > 2 else 2017
    yr_max = int(sys.argv[3]) if len(sys.argv) > 3 else 2020
    main(state, yr_min, yr_max)
```

Output: `data/hrrr_{state}_subdaily.parquet` with columns `date, gridmet_lat, gridmet_lon, window_hour, temp_pw, rh_pw, wind_speed_pw, vpd_pw, hpbl_pw, dswrf_pw`. Join to your training table on `(date, nearest gridmet_lat/lon, window_hour)`.

> **Note:** HRRR archive starts 2016. Years 2014–2015 have no HRRR coverage — exclude those rows from sub-daily training or use daily-only features for those years. `dswrf_pw` (downwelling shortwave radiation) had near-zero contribution in our ablation — verify early before spending time extracting it.

---

## Notes on your current feature set

### Are your current features sufficient for a baseline?

Yes — your current features are a reasonable daily-weather baseline. Once you add
the four LANDFIRE landscape features (`avg_burn_prob`, `whp`, `flep4`, `cfl`),
you have a strong two-tier model. Landscape features are the single most important
group: in our experiments, landscape alone outperforms daily weather alone, and
adding HRRR sub-daily features on top provides only marginal additional lift.
The HRRR features (`rh_pw`, `temp_pw`, etc.) are worth adding for sub-daily
window-level timing, but they are not required for a competitive baseline.

### Feature-by-feature notes on your list

| Feature | Status | Note |
|---|---|---|
| `vpd` | **Check source — likely leakage** | If from FPA-FOD: recorded at discovery time → post-hoc, drop it. Re-extract from gridMET (`vpd_{year}.nc`). Inflates AUROC by ~0.015 if kept. |
| `fm100` | Good | 100-hr dead fuel moisture — strong predictor. Compute 5-day lagged mean (`fm100_5D_mean`) for best signal. |
| `fm1000` | Redundant | 1000-hr moisture is correlated with fm100. Not used in our model — drop or check LOO contribution before keeping. |
| `erc` | Good | Add this if not already present — Energy Release Component is one of the top daily weather predictors. |
| `bi` | Redundant with erc | Burning Index and ERC are correlated NFDRS indices. Keep one; drop the other. |
| `vs` | Weak | Daily gridMET wind speed. Not predictive at H3 scale — daily wind does not discriminate which cell ignites within a wind-affected corridor. HRRR `wind_speed_pw` is more useful if you add HRRR. |
| `rmax`, `rmin` | Partially redundant | Max/min relative humidity correlates strongly with vpd and tmmx. If you have vpd (from gridMET), rmax/rmin add little. Check LOO before keeping both. |
| `tmmx` | Good | Maximum temperature — useful. |
| `tmmn` | Redundant | Min temperature is correlated with tmmx. One is sufficient. |
| `pr` | Marginal | Daily precipitation. Weak individual predictor; fm100 already integrates antecedent moisture. |
| `sph` | Redundant | Specific humidity overlaps with vpd and rmax/rmin. Drop. |
| `fire_count`, `has_fire_history` | Good | Fire history is one of the strongest spatial predictors — keep both. |
| `burnable` | Good | Binary non-burnable mask. Cells with burnable=0 should score near zero; keeping this column helps the model. |
| `ecoregion_l2`, `ecoregion_l3` | Low value for tabular models | Categorical — requires one-hot encoding. Since you already have `lat`/`lon`, geography is captured directly. Ecoregion adds little on top of landscape features and coordinates. Drop unless you have a specific reason to keep it. |
| `state_x` / `state_y` (duplicate) | Drop one | Redundant columns. lat/lon already encode location. |
| `h3_resolution` | Drop | Constant — no information. |
| `sin_month`, `cos_month`, `sin_hour`, `cos_hour` | Good | Cyclic temporal encoding — keep all four. |
| `centroid_lat`, `centroid_lon` | Good | Location proxy — keep. |

### vpd — confirm the source

The FPA-FOD database includes a `vpd` field recorded at fire discovery time.
**Do not use it as a training feature** — it is post-hoc (measured after ignition)
and inflates AUROC by ~0.015. Drop it and re-extract from gridMET instead
(`vpd_{year}.nc` from the Northwest Knowledge Network, same URL pattern as ERC).

### vpd_pw — this is derived, not fetched directly from HRRR

`vpd_pw` is not a direct HRRR output variable. It is computed from 2-m temperature
and 2-m relative humidity using the Tetens formula:

```python
T_c  = temp_kelvin - 273.15
es   = 0.6108 * np.exp(17.27 * T_c / (T_c + 237.3))   # saturation vapour pressure [kPa]
vpd_pw = es * (1.0 - rh / 100.0)                        # actual VPD [kPa]
```

Similarly, `wind_speed_pw` is derived from the U and V wind components:
```python
wind_speed_pw = np.sqrt(ugrd**2 + vgrd**2)
```

Both derivations are handled automatically in the HRRR fetch script below.

### Feature selection — avoid individual AUC ranking

Selecting ~200 features by individual AUC score is unreliable for two reasons:
1. Post-hoc leakage fields can score high AUC but are invalid at inference
2. Correlated weather variables (vpd, rmax, tmmx all encode heat/dryness) each
   appear important individually without adding new information

Recommended approach: group features by source type (landscape, daily weather,
sub-daily weather, temporal, location) and run a **leave-one-group-out ablation
on AUPR**. Drop groups that do not contribute, then use SHAP within surviving
groups to trim redundant individual features.

### Sampling design matters

Use **date-matched negative sampling**: for each fire event, draw negatives from
cells that did not burn on the same calendar date and UTC window. This forces the
model to learn spatial discrimination under identical weather conditions — the
correct operational question ("given today is dangerous, which cells are highest
risk?"). Unconditional random negatives answer a different question and produce a
model that is severely miscalibrated for same-day dispatch use.

---

## Run order summary

```bash
python fetch_landscape.py CA        # landscape → data/landscape_h3_ca.parquet
python fetch_landscape.py TX        # landscape → data/landscape_h3_tx.parquet
python fetch_gridmet.py CA 2014 2020  # gridMET  → data/gridmet_ca_daily.parquet
python fetch_gridmet.py TX 2014 2020  # gridMET  → data/gridmet_tx_daily.parquet
python fetch_hrrr.py CA 2017 2020     # HRRR     → data/hrrr_ca_subdaily.parquet
python fetch_hrrr.py TX 2017 2020     # HRRR     → data/hrrr_tx_subdaily.parquet
```

Join all three outputs to your FPA-FOD label table on `(h3_cell, date, window_hour)`
to produce the training parquet.

---

*Document prepared for the wildfire ignition prediction project. Questions: contact the project lead.*
