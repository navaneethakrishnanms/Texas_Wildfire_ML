"""
TDIS_Ignition_BIT_V3/realpop -- TRUE real-population validation.

Direct response to internal feedback (Haripriya, 2026-10-14):
  "The metrics presented are on a sample dataset not the whole population.
   I would like to see metrics on the population as well."

This pulls REAL archived HRRR forecasts (via Herbie, from the public NOAA
AWS archive -- not simulated, not climatological) for a representative set
of real calendar dates, scores the model against EVERY TX cell for each of
those dates (not a sampled/matched subset), and uses REAL observed labels
from TDIS's own label file. This is only possible for the HRRR-only (V3,
29-feature) model -- BIT v1/v2 depend on gridMET columns that have no
equivalent full-grid archive available locally, which is itself a practical
argument for the HRRR-only design.

Method:
  1. Pick N real dates spread across the test period (2022-2026), one per
     season per year, so the sample isn't concentrated in one fire season.
  2. For each date D, pull the archived HRRR forecast issued at D-1 12Z,
     lead=24h (valid at D 12Z) -- the same forecast-realistic design used
     in training (archived FORECAST, not an observation).
  3. Nearest-neighbor join the HRRR grid onto all 1,708,940 TX H3 cells
     (built once, reused for every date -- the HRRR grid itself doesn't move).
  4. Assemble the full 29-feature row for every cell, score with the
     trained BIT V3 model + its calibrator.
  5. Label = 1 if that (h3_cell, date) has a real recorded ignition in
     TDIS's `ignitions_daily_tx.parquet`, else 0. Flare cells excluded
     from scoring entirely (consistent with how the model was trained).
  6. Save one parquet per date to realpop/data/scored_days/.
     Script 02 concatenates all of them and computes the real-population
     metrics (true base rate ~1-3%, not the ~30% matched-sample rate).
"""
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
from herbie import Herbie

ROOT = Path(__file__).resolve().parent.parent.parent  # TDIS_Ignition_BIT_V3/
TDIS = ROOT.parent / "TDIS_Ignition"
REALPOP = ROOT / "realpop"
OUT_DAYS = REALPOP / "data" / "scored_days"
OUT_DAYS.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT / "models" / "model_bit_v3_29feat.json"
CALIB_PATH = ROOT / "models" / "calibrator_bit_v3_29feat.joblib"
FEATURE_LIST_PATH = ROOT / "models" / "feature_list.json"

# Representative dates: one per season per year across the test period (2022-2026)
TARGET_DATES = [
    "2022-01-15", "2022-04-15", "2022-07-15", "2022-10-15",
    "2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15",
    "2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15",
    "2025-01-15", "2025-04-15", "2025-07-15", "2025-10-15",
    "2026-01-15", "2026-04-15", "2026-07-15",   # label data ends 2026-07-29
]


def load_static_grid():
    print("Loading full TX static grid (1,708,940 cells) ...")
    static = pd.read_parquet(TDIS / "data" / "static" / "tx_static_master.parquet")
    powerline = pd.read_parquet(TDIS / "data" / "static" / "powerline_dist_km.parquet")
    popdens = pd.read_parquet(TDIS / "data" / "static" / "pop_density_feature.parquet")
    fips = pd.read_parquet(TDIS / "data" / "static" / "tx_county_fips.parquet")

    grid = static.merge(powerline, on="h3_cell", how="left")
    grid = grid.merge(popdens, on="h3_cell", how="left")
    grid = grid.merge(fips, on="h3_cell", how="left")
    print(f"  grid shape: {grid.shape}")
    return grid


def load_flares():
    f = pd.read_parquet(TDIS / "data" / "flares" / "flare_cells_tx_v2.parquet")
    return set(f["h3_cell"])


def load_drought():
    d = pd.read_parquet(TDIS / "data" / "static" / "drought_weekly.parquet")
    d["ValidStart"] = pd.to_datetime(d["ValidStart"])
    d["ValidEnd"] = pd.to_datetime(d["ValidEnd"])
    return d


def drought_for_date(drought_df, fips_series, date):
    date = pd.Timestamp(date)
    week = drought_df[(drought_df.ValidStart <= date) & (drought_df.ValidEnd >= date)]
    if week.empty:
        # nearest available week
        drought_df = drought_df.assign(dist=(drought_df.ValidStart - date).abs())
        nearest_week = drought_df.ValidStart[drought_df.dist.idxmin()]
        week = drought_df[drought_df.ValidStart == nearest_week]
    lookup = week.set_index("fips")["drought_score"]
    return fips_series.map(lookup).fillna(lookup.median() if len(lookup) else 1.0)


def load_labels():
    lab = pd.read_parquet(TDIS / "data" / "labels" / "ignitions_daily_tx.parquet")
    lab["date"] = pd.to_datetime(lab["date"])
    return lab[["h3_cell", "date"]]


def build_hrrr_nn_index(grid_lat, grid_lon):
    """Fetch one HRRR field just to get the grid's lat/lon, build a cKDTree,
    and return the nearest-neighbor flat index for every TX cell -- computed
    once, reused for every date (the HRRR grid geometry never changes)."""
    print("Building HRRR-grid nearest-neighbor index (one-time) ...")
    H = Herbie("2024-07-14 12:00", model="hrrr", product="sfc", fxx=24)
    ds = H.xarray(":TMP:2 m above ground")
    hlat = ds["latitude"].values.ravel()
    hlon = ds["longitude"].values.ravel()
    hlon_signed = np.where(hlon > 180, hlon - 360, hlon)  # HRRR lon is 0-360
    tree = cKDTree(np.column_stack([hlat, hlon_signed]))
    _, idx = tree.query(np.column_stack([grid_lat, grid_lon]), k=1)
    print(f"  HRRR grid shape: {ds['t2m'].shape}, flattened: {len(hlat):,} points")
    return idx, ds["t2m"].shape


def fetch_day_weather(date_str, nn_idx, grid_shape):
    """Pull the archived HRRR forecast issued the day before 12Z, lead=24h
    (valid at target date 12Z) -- same forecast-realistic design as training."""
    target = pd.Timestamp(date_str)
    init = target - pd.Timedelta(days=1)
    init_str = f"{init.date()} 12:00"

    H = Herbie(init_str, model="hrrr", product="sfc", fxx=24)
    tmp = H.xarray(":TMP:2 m above ground")["t2m"].values.ravel()
    dpt = H.xarray(":DPT:2 m above ground")["d2m"].values.ravel()
    u = H.xarray(":UGRD:10 m above ground")["u10"].values.ravel()
    v = H.xarray(":VGRD:10 m above ground")["v10"].values.ravel()
    mstav = H.xarray(":MSTAV:0 m underground")
    mstav_var = [v for v in mstav.data_vars][0]
    mstav = mstav[mstav_var].values.ravel()

    tmp_c = tmp[nn_idx] - 273.15
    dpt_c = dpt[nn_idx] - 273.15
    u10 = u[nn_idx]
    v10 = v[nn_idx]
    mstav_pct = mstav[nn_idx]

    # VPD via Tetens (kPa), same formula used throughout this project
    def es_kpa(t_c):
        return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))
    vpd = es_kpa(tmp_c) - es_kpa(dpt_c)
    vpd = np.clip(vpd, 0, None)

    wind = np.sqrt(u10 ** 2 + v10 ** 2)
    mstav_frac = mstav_pct / 100.0  # MSTAV is a percentage in the GRIB field

    return pd.DataFrame({
        "hrrr_tmp": tmp_c, "hrrr_vpd": vpd, "hrrr_wind": wind, "hrrr_mstav": mstav_frac,
    })


def main():
    t0 = time.time()
    feat_info = json.load(open(FEATURE_LIST_PATH))
    FEATURES = feat_info["features"]
    print(f"Model features ({len(FEATURES)}): {FEATURES}")

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    calibrator = joblib.load(CALIB_PATH)

    grid = load_static_grid()
    flares = load_flares()
    drought_df = load_drought()
    labels = load_labels()

    grid = grid[~grid.h3_cell.isin(flares)].reset_index(drop=True)
    print(f"Grid after flare exclusion: {len(grid):,} cells")

    nn_idx, hrrr_shape = build_hrrr_nn_index(grid["lat"].values, grid["lon"].values)

    for date_str in TARGET_DATES:
        out_path = OUT_DAYS / f"{date_str}.parquet"
        if out_path.exists():
            print(f"[skip] {date_str} already scored")
            continue

        print(f"\n=== Scoring {date_str} ===")
        t_day = time.time()
        try:
            weather = fetch_day_weather(date_str, nn_idx, hrrr_shape)
        except Exception as e:
            print(f"  FAILED to fetch weather for {date_str}: {e}")
            continue

        day = grid.copy().reset_index(drop=True)
        day = pd.concat([day, weather], axis=1)

        date_ts = pd.Timestamp(date_str)
        day["drought_score"] = drought_for_date(drought_df, day["fips"], date_ts).values

        month, dow = date_ts.month, date_ts.dayofweek
        day["sin_month"] = np.sin(2 * np.pi * month / 12)
        day["cos_month"] = np.cos(2 * np.pi * month / 12)
        day["sin_dow"] = np.sin(2 * np.pi * dow / 7)
        day["cos_dow"] = np.cos(2 * np.pi * dow / 7)
        day["is_weekend"] = int(dow >= 5)
        HOLIDAYS = {(1, 1), (7, 4), (6, 19), (11, 11), (12, 25), (10, 31)}
        day["is_holiday"] = int((month, date_ts.day) in HOLIDAYS)

        day["bp_x_vpd"] = day["avg_burn_prob"] * day["hrrr_vpd"]
        day["whp_x_vpd"] = day["whp"] * day["hrrr_vpd"]
        day["bp_x_drought"] = day["avg_burn_prob"] * day["drought_score"]

        day["date"] = date_ts

        missing = [f for f in FEATURES if f not in day.columns]
        assert not missing, f"Missing features for scoring: {missing}"
        nulls = day[FEATURES].isnull().sum()
        if nulls.sum() > 0:
            print(f"  WARNING: nulls present, filling with column median:\n{nulls[nulls>0]}")
            day[FEATURES] = day[FEATURES].fillna(day[FEATURES].median())

        p_raw = model.predict_proba(day[FEATURES])[:, 1]
        p_cal = calibrator.predict(p_raw)

        day_labels = set(labels[labels.date == date_ts]["h3_cell"])
        y = day["h3_cell"].isin(day_labels).astype(int)

        out = pd.DataFrame({
            "h3_cell": day["h3_cell"], "date": date_ts,
            "p_raw": p_raw.astype("float32"), "p_cal": p_cal.astype("float32"),
            "label": y.astype("int8"),
        })
        out.to_parquet(out_path, index=False)
        print(f"  cells={len(out):,}  positives={y.sum()}  base_rate={y.mean():.5f}  "
              f"time={time.time()-t_day:.1f}s  -> {out_path.name}")

    print(f"\nAll dates processed. Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
