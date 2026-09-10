"""
build_historical_snapshot.py
-----------------------------
Builds the "current risk" hexagon layer for everything up to and including
2026-07-31 (the last date our dataset actually has real, non-fabricated
ground truth for -- see DATA_QUALITY_REVIEW.md).

Design decision (documented, not hidden):
  The dataset is SPARSE -- median ~2 rows per h3_cell across 13 years (fire
  days + a limited sample of negative days, not continuous daily coverage).
  So "the most recent row per cell" is often from months or years ago, not
  actually "yesterday". For a portal risk map, we treat each cell's MOST
  RECENT available feature row (whatever its date) as that cell's best
  current static/climatological risk estimate -- this is a standard approach
  for background wildfire-hazard maps built on sparse point-in-time data,
  and it's what "Hazard Severity Outlook" style products usually mean (a
  standing hazard surface, not a live weather nowcast).

  Rows with label == 1 (an actual historical fire) in the most recent 90 days
  of the historical window become "Active Events" flame markers, exactly
  like the red flame icons in the reference screenshot.

Outputs (into TDIS_REPLICA/data/):
  historical_risk_cells.json   -- GeoJSON FeatureCollection, one polygon per
                                   h3_cell, with risk_score + risk_class
  historical_active_fires.json -- GeoJSON FeatureCollection of point markers
  hex_geometry_cache.parquet   -- h3_cell -> boundary ring cache (reused by
                                   the live-feed script so geometry is only
                                   computed once)

Usage:
    python TDIS_REPLICA/ingestion/build_historical_snapshot.py
"""

import json
import time

import numpy as np
import pandas as pd

from common import (
    DATASET_PATH, DATA_OUT, HISTORICAL_CUTOVER, FEATURES,
    load_model_and_calibrator, score, risk_class, hex_boundary_geojson,
)


def run():
    t0 = time.time()
    print("=" * 60)
    print("  Building historical risk snapshot (<= 2026-07-31)")
    print("=" * 60)

    print(f"\n[1/5] Loading dataset: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= HISTORICAL_CUTOVER].copy()
    print(f"      Rows in historical window: {len(df):,}")

    print(f"\n[2/5] Taking each cell's most recent available row ...")
    latest = df.sort_values("date").groupby("h3_cell", as_index=False).last()
    print(f"      Unique cells: {len(latest):,}")
    print(f"      Snapshot date range used: {latest['date'].min().date()} .. {latest['date'].max().date()}")

    print(f"\n[3/5] Scoring with model_32feat_tuned.json + calibrator ...")
    model, calibrator = load_model_and_calibrator()
    latest["risk_score"] = score(model, calibrator, latest)
    latest["risk_class"] = risk_class(latest["risk_score"].values)
    print(f"      Risk class distribution:\n{latest['risk_class'].value_counts()}")

    print(f"\n[4/5] Building hexagon geometry + a queryable parquet "
          f"(not a single GeoJSON blob -- 609k polygons is too large to ship "
          f"to a browser in one response; the API will filter by map "
          f"viewport + zoom instead, same effect as pg_tileserv's tiling) ...")
    rings = []
    for h3_cell in latest["h3_cell"]:
        rings.append(json.dumps(hex_boundary_geojson(h3_cell)))
    latest = latest.reset_index(drop=True)
    latest["ring"] = rings

    out_cols = ["h3_cell", "lat", "lon", "date", "risk_score", "risk_class", "ring"]
    out_df = latest[out_cols].rename(columns={"date": "as_of_date"})
    out_path = DATA_OUT / "historical_risk_cells.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"      Saved {len(out_df):,} hex rows -> {out_path}")

    out_df[["h3_cell", "ring"]].to_parquet(DATA_OUT / "hex_geometry_cache.parquet", index=False)
    print(f"      Geometry cache saved -> hex_geometry_cache.parquet")

    print(f"\n[5/5] Building a PER-DATE active-fire table (label=1 rows, "
          f"exact date match -- not a flattened static dump) ...")
    fires = df[df["label"] == 1][["h3_cell", "date", "lat", "lon"]].copy()
    fires["date"] = fires["date"].dt.strftime("%Y-%m-%d")
    fires_path = DATA_OUT / "active_fires_by_date.parquet"
    fires.to_parquet(fires_path, index=False)
    per_day_counts = fires.groupby("date").size()
    print(f"      {len(fires):,} total historical fire-day rows -> {fires_path}")
    print(f"      Per-day count -- min={per_day_counts.min()} "
          f"median={int(per_day_counts.median())} max={per_day_counts.max()}")
    print(f"      Example, {HISTORICAL_CUTOVER.date()}: "
          f"{(fires['date'] == str(HISTORICAL_CUTOVER.date())).sum()} active fires")

    elapsed = round(time.time() - t0, 1)
    print(f"\nDone in {elapsed}s.")


if __name__ == "__main__":
    run()
