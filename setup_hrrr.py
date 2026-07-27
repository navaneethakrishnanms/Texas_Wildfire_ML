"""
setup_hrrr.py
--------------
Pre-flight script before running fetch_hrrr_tx.py

Does:
  1. Checks / installs herbie-data, cfgrib, scipy
  2. Merges train/val/test parquets -> data/full_tx_clean.parquet
  3. Verifies HRRR AWS access with a 3-sample test download
  4. Prints final go/no-go

Run this FIRST before fetch_hrrr_tx.py
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
HRRR_DIR = DATA_DIR / "hrrr"
HRRR_DIR.mkdir(parents=True, exist_ok=True)
FULL_PQ  = DATA_DIR / "full_tx_clean.parquet"

def run(cmd):
    print(f"  >> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

print("=" * 60)
print("HRRR PRE-FLIGHT SETUP")
print("=" * 60)

# ── Step 1: Install / check dependencies ──────────────────────
print("\n[1/4] Checking dependencies...")

missing = []
for pkg, import_name in [
    ("herbie-data", "herbie"),
    ("cfgrib",      "cfgrib"),
    ("scipy",       "scipy"),
]:
    try:
        __import__(import_name)
        print(f"  {pkg:<15} OK")
    except ImportError:
        print(f"  {pkg:<15} MISSING — installing...")
        missing.append(pkg)

if missing:
    for pkg in missing:
        run(f"pip install {pkg}")
    print("  All packages installed.")
else:
    print("  All dependencies present.")

# Re-check herbie version
import herbie as _h
print(f"  herbie version: {_h.__version__}")

# ── Step 2: Merge split parquets -> full_tx_clean.parquet ─────
print("\n[2/4] Merging train/val/test parquets...")

splits = {
    "train": DATA_DIR / "train_tx_clean.parquet",
    "val":   DATA_DIR / "val_tx_clean.parquet",
    "test":  DATA_DIR / "test_tx_clean.parquet",
}
found = {k: v for k, v in splits.items() if v.exists()}
if not found:
    print("  ERROR: No split parquets found in data/")
    print("         Run train_tx.py first.")
    sys.exit(1)

if FULL_PQ.exists():
    df_full = pd.read_parquet(FULL_PQ)
    print(f"  full_tx_clean.parquet already exists: {len(df_full):,} rows — skipping merge")
else:
    dfs = []
    for name, path in found.items():
        df = pd.read_parquet(path)
        df["_split"] = name
        dfs.append(df)
        print(f"  {name:<6}: {len(df):,} rows  ({path.name})")
    df_full = pd.concat(dfs, ignore_index=True)
    df_full.to_parquet(FULL_PQ, index=False, compression="snappy")
    print(f"  Merged: {len(df_full):,} rows -> {FULL_PQ.name}  "
          f"({FULL_PQ.stat().st_size/1e6:.0f} MB)")

# Print info
print(f"\n  Columns:        {list(df_full.columns)}")
print(f"  Unique H3:      {df_full['h3_cell'].nunique():,}")
df_full["date_utc"] = pd.to_datetime(df_full["date_utc"])
print(f"  Date range:     {df_full['date_utc'].min().date()} – {df_full['date_utc'].max().date()}")
dw = df_full[["date_utc", "window_hour"]].drop_duplicates()
print(f"  (date, window): {len(dw):,} unique pairs")
print(f"  Labels:         fire={int((df_full['label']==1).sum()):,}  "
      f"non-fire={int((df_full['label']==0).sum()):,}")

# ── Step 3: Test HRRR access (3 sample downloads) ─────────────
print("\n[3/4] Testing HRRR access (3 sample downloads)...")
print("  Downloading 3 HRRR analysis files from AWS S3...")
print("  If this hangs >60s, check your internet connection.")

from herbie import Herbie
from scipy.spatial import cKDTree

# Pick 3 test dates well within HRRR archive
test_cases = [
    ("2018-07-04 12:00", "summer peak"),
    ("2019-03-15 18:00", "spring afternoon"),
    ("2020-11-01 06:00", "fall morning"),
]

# Use a small sample of centroids for speed
sample_centroids = (
    df_full[["h3_cell", "centroid_lat", "centroid_lon"]]
    .drop_duplicates("h3_cell")
    .head(200)
    .reset_index(drop=True)
)
cell_lats = sample_centroids["centroid_lat"].values
cell_lons = sample_centroids["centroid_lon"].values

all_ok = True
for dt_str, label in test_cases:
    try:
        t0 = __import__("time").time()
        H  = Herbie(dt_str, model="hrrr", product="sfc", fxx=0,
                    verbose=False, priority=["aws"])
        ds = H.xarray(":TMP:2 m above ground", remove_grib=True)
        var_name = list(ds.data_vars)[0]
        lats = ds["latitude"].values.ravel()
        lons = ds["longitude"].values.ravel()
        vals = ds[var_name].values.ravel()
        tree = cKDTree(np.column_stack([lats, lons]))
        _, idx = tree.query(np.column_stack([cell_lats, cell_lons]), k=1)
        temps_c = vals[idx] - 273.15
        elapsed = __import__("time").time() - t0
        print(f"  OK  {dt_str} ({label})  "
              f"mean_temp={temps_c.mean():.1f}C  {elapsed:.0f}s")
    except Exception as e:
        print(f"  FAIL {dt_str}: {e}")
        all_ok = False

# ── Step 4: Summary ───────────────────────────────────────────
print("\n[4/4] PRE-FLIGHT SUMMARY")
print("=" * 60)
if all_ok:
    n_pairs = len(dw)
    n_avail = len(dw[pd.to_datetime(dw["date_utc"]) >= pd.Timestamp("2014-11-01")])
    est_4w  = n_avail * 20 / 4 / 3600
    est_8w  = n_avail * 20 / 8 / 3600
    print(f"  Status:           GO — all checks passed")
    print(f"  Total pairs:      {n_pairs:,}  (HRRR available: {n_avail:,})")
    print(f"  Estimated time:   {est_4w:.0f}h with 4 workers | "
          f"{est_8w:.0f}h with 8 workers")
    print(f"  Disk estimate:    ~2-5 GB (extracted parquets, not raw GRIB)")
    print("")
    print("  Run next:")
    print("    python fetch_hrrr_tx.py --workers 4")
    print("    (or leave overnight: --workers 8 for faster)")
else:
    print("  Status:           NO-GO — HRRR access failed")
    print("  Check internet and try again.")
print("=" * 60)
