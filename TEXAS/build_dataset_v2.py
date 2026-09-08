"""
build_dataset_v2.py
--------------------
Builds the v2 TEXAS training dataset by adding one new feature to the existing
30-feature imputed TDIS dataset: H3-cell centroid coordinates (lat, lon).

WHY THIS FEATURE:
  A comparison of the TX pipeline (Texas ML Wildfire/TX/) against the TDIS-shared
  model showed that TX's `centroid_lat`/`centroid_lon` were its #5 most important
  features (~12% combined gain, TX/outputs/texas_landfire/models/xgb_tx_tuned_meta.json)
  -- yet the TDIS-derived dataset used in TEXAS/train_model_30feat.py never
  included any raw coordinate feature at all (only derived static covariates:
  elevation, slope, aspect, road_dist, ecoregion_id).

WHY NOT OTHER TX FEATURES:
  - `burnable` (binary land-cover flag): not present anywhere in the TDIS
    Supporting_files/Focused_Files static data, and cannot be derived without
    the original LANDFIRE EVT raster. Not added -- would be fabricated.
  - 5-day rolling gridMET stats (`erc_5D_mean`, `vpd_5D_max`, etc.): TX computes
    these from *dense* per-cell daily gridMET history. This TDIS dataset is
    sparse (median ~2 rows per h3_cell across 13 years -- fire days + sampled
    negative days only, not continuous daily coverage). Computing a "5-day"
    rolling window over sparse, non-contiguous rows is exactly the bug TX's own
    team found and fixed (TX/missing_data_diagnosis.md: 84.4% NaN from rolling
    over row-order instead of true calendar days). Reproducing it here would
    silently corrupt the feature. Not added -- the raw gridMET NetCDF archive
    needed to do this correctly is not available locally (see REPRODUCE.md
    Tier 3 requirements).
  - sin_hour/cos_hour: this dataset is daily-resolution (no window_hour column
    -- that concept doesn't exist here). Not applicable.

Source data:
  TEXAS_v1_backup/tdis_train_daily_imputed.parquet   (3,595,513 rows x 35 cols)
  Supporting_files/Supporting_files/data/static_features/tx_static_master.parquet
      (1,708,940 unique h3_cell rows with lat, lon, + other static covariates
       already present in the 30-feat set)

Output:
  TEXAS/tdis_train_daily_imputed_v2.parquet   (3,595,513 rows x 37 cols)

Usage:
    python TEXAS/build_dataset_v2.py
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

BASE_PARQUET = ROOT / "TEXAS_v1_backup" / "tdis_train_daily_imputed.parquet"
STATIC_MASTER = (
    ROOT / "Supporting_files" / "Supporting_files" / "data" / "static_features"
    / "tx_static_master.parquet"
)
OUT_PARQUET = HERE / "tdis_train_daily_imputed_v2.parquet"


def run():
    print("=" * 60)
    print("  Building TEXAS v2 dataset (30 feat + centroid lat/lon)")
    print("=" * 60)

    print(f"\n[1/4] Loading base dataset: {BASE_PARQUET}")
    df = pd.read_parquet(BASE_PARQUET)
    print(f"      Rows: {len(df):,}  Columns: {len(df.columns)}")

    print(f"\n[2/4] Loading static master (source of lat/lon): {STATIC_MASTER}")
    static = pd.read_parquet(STATIC_MASTER, columns=["h3_cell", "lat", "lon"])
    static = static.drop_duplicates(subset="h3_cell")
    print(f"      Unique h3_cell rows: {len(static):,}")

    print(f"\n[3/4] Merging lat/lon onto base dataset by h3_cell ...")
    before_cols = set(df.columns)
    df = df.merge(static, on="h3_cell", how="left")
    new_cols = [c for c in df.columns if c not in before_cols]
    print(f"      New columns added: {new_cols}")

    matched = df["lat"].notna().sum()
    unmatched = df["lat"].isna().sum()
    print(f"      Matched:   {matched:,} / {len(df):,} rows ({100*matched/len(df):.2f}%)")
    print(f"      Unmatched: {unmatched:,} rows ({100*unmatched/len(df):.2f}%)")

    if unmatched > 0:
        # Unmatched cells (outside the static master's TX bbox extract, if any)
        # get median-filled so no rows are dropped -- consistent with the
        # existing imputation policy used to build the base 30-feat dataset.
        med_lat, med_lon = df["lat"].median(), df["lon"].median()
        df["lat"] = df["lat"].fillna(med_lat)
        df["lon"] = df["lon"].fillna(med_lon)
        print(f"      Filled {unmatched:,} unmatched rows with median lat/lon "
              f"({med_lat:.4f}, {med_lon:.4f})")

    print(f"\n[4/4] Saving: {OUT_PARQUET}")
    df.to_parquet(OUT_PARQUET, index=False, compression="snappy")
    print(f"      Rows: {len(df):,}  Columns: {len(df.columns)}")
    print(f"      Columns: {list(df.columns)}")

    print("\n" + "=" * 60)
    print("  DATASET V2 BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run()
