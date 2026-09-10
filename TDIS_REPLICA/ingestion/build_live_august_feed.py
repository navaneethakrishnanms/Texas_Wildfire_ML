"""
build_live_august_feed.py
---------------------------
Builds the SIMULATED "live" daily feed for 2026-08-01 .. 2026-08-31.

WHY THIS IS SIMULATED, NOT REAL:
  There is no live weather/ignition data-injection source connected to this
  project (no Databricks job, no real-time gridMET/HRRR feed). The dataset's
  own rows after 2026-07-29 are fabricated placeholder rows (label=0, no real
  weather -- see DATA_QUALITY_REVIEW.md) and are never used here.

  Instead, for each h3_cell we build a plausible August feature vector by
  averaging that cell's own historical August observations (all years present
  in the dataset, 2014-2025) for every weather/HRRR column. Static features
  (elevation, slope, LandFIRE, roads, lat/lon, etc.) don't change and are
  reused as-is. Calendar features (sin/cos_dow, is_weekend, is_holiday) are
  computed correctly for the REAL August 2026 calendar dates -- those are not
  approximated, they're exact.

  Cells with no historical August observations fall back to their overall
  latest known feature snapshot (from build_historical_snapshot.py) so
  coverage matches the historical map with no gaps.

  This produces a full 31-day, per-cell forecast table, scored with the same
  trained model. It is clearly labeled "source": "simulated_live" everywhere
  it appears (data files, API responses, and the frontend UI) -- this is a
  demo of what a live feed WOULD look like once a real ingestion pipeline
  exists, not a claim of real forecast skill for August 2026.

Output:
  TDIS_REPLICA/data/live_august_staging.parquet
      One row per (h3_cell, date) for all of August 2026, with risk_score
      already computed. NOT yet "published" -- live_feed_daemon.py reveals
      one day at a time from this staging file.

Usage:
    python TDIS_REPLICA/ingestion/build_live_august_feed.py
"""

import time

import numpy as np
import pandas as pd

from common import (
    DATASET_PATH, DATA_OUT, HISTORICAL_CUTOVER, LIVE_START, LIVE_END,
    FEATURES, load_model_and_calibrator, score, risk_class,
)

US_FEDERAL_HOLIDAYS_AUG_2026: set[str] = set()  # no US federal holidays fall in August

WEATHER_COLS = [
    "hrrr_tmp", "hrrr_vpd", "hrrr_wind", "hrrr_mstav",
    "erc", "fm100", "vpd", "vs", "rmax", "rmin", "tmmx", "pr",
]
STATIC_COLS = [
    "elevation_m", "slope_deg", "aspect_deg", "road_dist_km", "ecoregion_id",
    "avg_burn_prob", "whp", "flep4", "cfl", "cbd", "cbh",
    "powerline_dist_km", "lat", "lon",
]


def build_calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    dow = dates.dayofweek  # Monday=0
    month = dates.month
    cal = pd.DataFrame({
        "date": dates,
        "sin_month": np.sin(2 * np.pi * month / 12),
        "cos_month": np.cos(2 * np.pi * month / 12),
        "sin_dow": np.sin(2 * np.pi * dow / 7),
        "cos_dow": np.cos(2 * np.pi * dow / 7),
        "is_weekend": (dow >= 5).astype(int),
        "is_holiday": 0,
    })
    return cal


def run():
    t0 = time.time()
    print("=" * 60)
    print("  Building SIMULATED live August 2026 feed")
    print("=" * 60)

    print(f"\n[1/5] Loading full dataset: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= HISTORICAL_CUTOVER].copy()  # never touch fabricated future rows

    print(f"\n[2/5] Computing per-cell seasonal-August weather averages "
          f"(historical Augusts 2014-2025) ...")
    august_hist = df[df["date"].dt.month == 8]
    print(f"      Historical August rows available: {len(august_hist):,}")
    seasonal_weather = august_hist.groupby("h3_cell")[WEATHER_COLS].mean()
    print(f"      Cells with historical August weather: {len(seasonal_weather):,}")

    print(f"\n[3/5] Building fallback static snapshot (latest known row per cell) ...")
    latest = df.sort_values("date").groupby("h3_cell", as_index=False).last()
    latest_static = latest.set_index("h3_cell")[STATIC_COLS + WEATHER_COLS]
    all_cells = latest_static.index

    print(f"      Total cells to generate August rows for: {len(all_cells):,}")
    with_seasonal = seasonal_weather.index.intersection(all_cells)
    without_seasonal = all_cells.difference(seasonal_weather.index)
    print(f"      -> using seasonal-August average weather: {len(with_seasonal):,} cells")
    print(f"      -> falling back to latest-known weather:  {len(without_seasonal):,} cells")

    # Base per-cell feature frame: static columns from latest snapshot,
    # weather columns from seasonal August average where available, else
    # latest known weather.
    base = latest_static[STATIC_COLS].copy()
    weather_base = latest_static[WEATHER_COLS].copy()
    weather_base.loc[with_seasonal] = seasonal_weather.loc[with_seasonal]
    base = base.join(weather_base)
    base = base.reset_index()  # h3_cell back as a column

    print(f"\n[4/5] Expanding to 31 daily rows (2026-08-01 .. 2026-08-31) "
          f"and scoring ...")
    dates = pd.date_range(LIVE_START, LIVE_END, freq="D")
    calendar = build_calendar_features(dates)

    model, calibrator = load_model_and_calibrator()

    all_days = []
    for _, cal_row in calendar.iterrows():
        day_df = base.copy()
        for col in ["sin_month", "cos_month", "sin_dow", "cos_dow", "is_weekend", "is_holiday"]:
            day_df[col] = cal_row[col]
        day_df["date"] = cal_row["date"]
        day_df["risk_score"] = score(model, calibrator, day_df)
        day_df["risk_class"] = risk_class(day_df["risk_score"].values)
        day_df["source"] = "simulated_live"
        all_days.append(day_df[["h3_cell", "lat", "lon", "date", "risk_score", "risk_class", "source"]])
        print(f"      {cal_row['date'].date()}  mean risk={day_df['risk_score'].mean():.4f}  "
              f"pct_high_plus={(day_df['risk_class'].isin(['high', 'extreme'])).mean()*100:.1f}%")

    full = pd.concat(all_days, ignore_index=True)

    print(f"\n[5/5] Saving staging file ({len(full):,} rows, {full['date'].nunique()} days) ...")
    out_path = DATA_OUT / "live_august_staging.parquet"
    full.to_parquet(out_path, index=False)
    print(f"      -> {out_path}")

    elapsed = round(time.time() - t0, 1)
    print(f"\nDone in {elapsed}s. Nothing is 'published' yet -- run live_feed_daemon.py "
          f"to start revealing days one at a time.")


if __name__ == "__main__":
    run()
