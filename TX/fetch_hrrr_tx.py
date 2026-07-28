"""
fetch_hrrr_tx.py
-----------------
HRRR Sub-Daily Feature Extraction for Texas Wildfire Model

Downloads 6 atmospheric variables from NOAA HRRR archive (AWS S3, free)
for each unique (date_utc, window_hour) in the Texas training dataset.

HRRR = High-Resolution Rapid Refresh
  - NOAA operational hourly weather model for CONUS
  - 3 km spatial resolution
  - Available Nov 2014 onward on AWS S3 (noaa-hrrr-bdp-pds)
  - Free, no API key needed

Why this matters:
  gridMET gives ONE value per day (daily avg/min/max)
  HRRR gives the actual atmospheric state at the exact 6-hour window
  → Model can distinguish "2am calm" from "2pm 35°C gusty fire weather"

Features extracted per (h3_cell, date_utc, window_hour):
  temp_pw     : 2m temperature [°C]
  rh_pw       : 2m relative humidity [%]
  wind_pw     : 10m wind speed [m/s]   (sqrt(u² + v²))
  vpd_pw      : vapor pressure deficit [kPa]  (derived)
  hpbl_pw     : planetary boundary layer height [m]
  dswrf_pw    : downward solar radiation [W/m²]
  hrrr_pw     : 1 = HRRR available for this row, 0 = not available

Coverage:
  2014 (Jan–Oct) : HRRR not yet operational → hrrr_pw=0, features=NaN
  2014 (Nov–Dec) : partial coverage
  2015–2020      : full coverage

Output:
  data/hrrr/hrrr_tx_YYYY.parquet   (one file per year)
  data/hrrr/hrrr_tx_all.parquet    (merged, joined on h3_cell+date+window)
  data/hrrr/checkpoint.csv         (resume support — tracks completed dates)

Install requirements (run first):
  pip install herbie-data cfgrib scipy

Usage:
  python fetch_hrrr_tx.py               # 4 parallel workers (recommended)
  python fetch_hrrr_tx.py --workers 8   # faster if internet allows
  python fetch_hrrr_tx.py --workers 1   # single-threaded (safest)
  python fetch_hrrr_tx.py --year 2018   # extract one year only (test first!)
  python fetch_hrrr_tx.py --dry-run     # shows what would be downloaded

Estimated time:
  ~10,220 HRRR files total (2014-2020, 4 windows/day × 365 days × 7 years)
  ~15-25 seconds per file (partial download + extract)
  4 workers → ~7-14 hours total (run overnight)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
DATA_DIR  = ROOT / "data"
HRRR_DIR  = DATA_DIR / "hrrr"
HRRR_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = HRRR_DIR / "checkpoint.csv"   # completed (date, window_hour) pairs
CLEAN_PQ   = DATA_DIR  / "full_tx_clean.parquet"   # from train_tx.py

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(HRRR_DIR / "fetch_hrrr.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── HRRR variables to extract ─────────────────────────────────────────────────
# (Herbie search strings, feature name, unit)
HRRR_VARS = [
    (":TMP:2 m above ground",  "tmp_raw",   "K"),
    (":RH:2 m above ground",   "rh_pw",     "%"),
    (":UGRD:10 m above ground","u_raw",      "m/s"),
    (":VGRD:10 m above ground","v_raw",      "m/s"),
    (":HPBL:surface",          "hpbl_pw",   "m"),
    (":DSWRF:surface",         "dswrf_pw",  "W/m2"),
]

# HRRR window_hour → UTC hour mapping
# Your dataset: window_hour 0=00-06UTC, 6=06-12UTC, 12=12-18UTC, 18=18-24UTC
WINDOW_HOURS = [0, 6, 12, 18]


# ── VPD computation ───────────────────────────────────────────────────────────
def compute_vpd(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Compute vapor pressure deficit [kPa] from temperature [°C] and RH [%]."""
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))   # saturation VP [kPa]
    ea = es * rh / 100.0                                        # actual VP [kPa]
    return np.maximum(es - ea, 0.0)


# ── Load training data centroids ──────────────────────────────────────────────
def load_centroids() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load (h3_cell, centroid_lat, centroid_lon) and
    unique (date_utc, window_hour) pairs from the clean training parquet.
    """
    log.info(f"Loading training data: {CLEAN_PQ.name}")
    df = pd.read_parquet(CLEAN_PQ, columns=[
        "h3_cell", "date_utc", "window_hour", "centroid_lat", "centroid_lon"
    ])
    df["date_utc"] = pd.to_datetime(df["date_utc"]).dt.normalize()

    # Unique cell centroids
    centroids = (
        df[["h3_cell", "centroid_lat", "centroid_lon"]]
        .drop_duplicates("h3_cell")
        .reset_index(drop=True)
    )
    log.info(f"  Unique H3 cells:         {len(centroids):,}")

    # Unique (date, window_hour) pairs — one HRRR download per pair
    date_windows = (
        df[["date_utc", "window_hour"]]
        .drop_duplicates()
        .sort_values(["date_utc", "window_hour"])
        .reset_index(drop=True)
    )
    log.info(f"  Unique (date, window):   {len(date_windows):,}")
    log.info(f"  Date range:              "
             f"{df['date_utc'].min().date()} – {df['date_utc'].max().date()}")

    return centroids, date_windows


# ── Load checkpoint ───────────────────────────────────────────────────────────
def load_checkpoint() -> set[tuple]:
    """Return set of (date_str, window_hour) already completed."""
    if not CHECKPOINT.exists():
        return set()
    cp = pd.read_csv(CHECKPOINT)
    done = set(zip(cp["date_str"], cp["window_hour"].astype(int)))
    log.info(f"  Checkpoint loaded: {len(done):,} completed pairs")
    return done


def save_checkpoint(date_str: str, window_hour: int):
    """Append one completed pair to checkpoint CSV."""
    with open(CHECKPOINT, "a") as f:
        if CHECKPOINT.stat().st_size == 0:
            f.write("date_str,window_hour\n")
        f.write(f"{date_str},{window_hour}\n")


# ── HRRR extraction for one timestamp ─────────────────────────────────────────
def extract_one(
    date_utc: pd.Timestamp,
    window_hour: int,
    centroids: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Download HRRR analysis for one (date, window_hour) and extract
    values at all H3 cell centroids.
    Returns DataFrame with columns: h3_cell, temp_pw, rh_pw, wind_pw,
                                    vpd_pw, hpbl_pw, dswrf_pw, hrrr_pw
    Returns None if HRRR not available (pre Nov 2014 or missing file).
    """
    from herbie import Herbie
    from scipy.spatial import cKDTree

    date_str = date_utc.strftime("%Y-%m-%d")
    dt_str   = f"{date_str} {window_hour:02d}:00"

    # HRRR not operational before Nov 2014
    if date_utc < pd.Timestamp("2014-11-01"):
        return None

    try:
        H = Herbie(
            dt_str,
            model="hrrr",
            product="sfc",
            fxx=0,             # analysis cycle (not forecast)
            verbose=False,
            priority=["aws"],  # free AWS S3 archive
        )

        # Try each variable individually — skip if not in this HRRR version
        # RH known issue: not available in 2015-2016 HRRR surface files as
        # ':RH:2 m above ground' — try fallback search strings
        RH_FALLBACKS = [
            ":RH:2 m above ground",    # 2017+ standard
            ":RH:2 m",                  # some older versions
            ":SPFH:2 m above ground",   # specific humidity (rare fallback)
        ]

        raw = {}
        lats_ref = lons_ref = None

        for search_str, key, _ in HRRR_VARS:
            search_attempts = (
                RH_FALLBACKS if key == "rh_pw"
                else [search_str]
            )
            success = False
            for attempt in search_attempts:
                try:
                    # remove_grib=False: keep the GRIB2 file in cache so
                    # subsequent variable calls reuse it (no re-download)
                    ds = H.xarray(attempt, remove_grib=False)
                    var_name = list(ds.data_vars)[0]
                    data     = ds[var_name].values.ravel()
                    lats     = ds["latitude"].values.ravel()
                    lons     = ds["longitude"].values.ravel()
                    raw[key] = (lats, lons, data)
                    if lats_ref is None:
                        lats_ref, lons_ref = lats, lons
                    success = True
                    break
                except Exception:
                    continue
            if not success:
                raw[key] = None   # NaN in final output

        # Cleanup cached GRIB2 files after extracting all variables
        try:
            save_dir = getattr(H, "save_dir", None)
            if save_dir is not None and Path(str(save_dir)).exists():
                for f in Path(str(save_dir)).glob("*.grib2"):
                    f.unlink(missing_ok=True)
        except Exception:
            pass

        # Need at least temperature to proceed
        if raw.get("tmp_raw") is None or lats_ref is None:
            log.warning(f"  HRRR FAIL: {dt_str} — temperature not available")
            return None

        # Build KD-tree on HRRR grid
        tree = cKDTree(np.column_stack([lats_ref, lons_ref]))

        # Query nearest HRRR grid point for each H3 centroid
        cell_lats = centroids["centroid_lat"].values
        cell_lons = centroids["centroid_lon"].values
        _, idx = tree.query(np.column_stack([cell_lats, cell_lons]), k=1)

        # Build result DataFrame
        result = centroids[["h3_cell"]].copy()
        result["date_utc"]    = date_utc
        result["window_hour"] = window_hour

        # Temperature: K → °C
        _, _, tmp_data = raw["tmp_raw"]
        result["temp_pw"] = tmp_data[idx] - 273.15

        # Relative humidity (NaN for 2015-2016 where RH not in surface file)
        if raw["rh_pw"] is not None:
            _, _, rh_data = raw["rh_pw"]
            result["rh_pw"] = rh_data[idx]
        else:
            result["rh_pw"] = np.nan

        # Wind speed: sqrt(u² + v²)
        if raw["u_raw"] is not None and raw["v_raw"] is not None:
            _, _, u_data = raw["u_raw"]
            _, _, v_data = raw["v_raw"]
            result["wind_pw"] = np.sqrt(u_data[idx]**2 + v_data[idx]**2)
        else:
            result["wind_pw"] = np.nan

        # VPD (derived from temp + rh — NaN if rh missing)
        if raw["rh_pw"] is not None:
            result["vpd_pw"] = compute_vpd(
                result["temp_pw"].values,
                result["rh_pw"].values
            )
        else:
            result["vpd_pw"] = np.nan

        # PBL height
        if raw["hpbl_pw"] is not None:
            _, _, hpbl_data = raw["hpbl_pw"]
            result["hpbl_pw"] = hpbl_data[idx]
        else:
            result["hpbl_pw"] = np.nan

        # Solar radiation
        if raw["dswrf_pw"] is not None:
            _, _, dswrf_data = raw["dswrf_pw"]
            result["dswrf_pw"] = dswrf_data[idx]
        else:
            result["dswrf_pw"] = np.nan

        # Availability flag
        result["hrrr_pw"] = 1

        return result

    except Exception as e:
        log.warning(f"  HRRR FAIL: {dt_str} — {type(e).__name__}: {e}")
        return None


# ── Process one year ──────────────────────────────────────────────────────────
def process_year(
    year: int,
    date_windows: pd.DataFrame,
    centroids: pd.DataFrame,
    done: set,
    n_workers: int,
    dry_run: bool,
) -> pd.DataFrame:
    """Extract all HRRR features for one year, return as DataFrame."""

    year_dw = date_windows[
        pd.to_datetime(date_windows["date_utc"]).dt.year == year
    ].reset_index(drop=True)

    out_path = HRRR_DIR / f"hrrr_tx_{year}.parquet"

    # Check if this year is FULLY complete via checkpoint
    year_pairs = set(
        zip(
            year_dw["date_utc"].dt.strftime("%Y-%m-%d"),
            year_dw["window_hour"].astype(int),
        )
    )
    done_this_year = done & year_pairs
    is_complete = len(done_this_year) >= len(year_pairs)

    if out_path.exists() and is_complete:
        log.info(f"  Year {year}: COMPLETE ({len(done_this_year)}/{len(year_pairs)} pairs) "
                 f"— loading {out_path.name}")
        return pd.read_parquet(out_path)

    if out_path.exists() and not is_complete:
        missing_count = len(year_pairs) - len(done_this_year)
        log.info(f"  Year {year}: PARTIAL — {len(done_this_year)}/{len(year_pairs)} pairs done, "
                 f"{missing_count} remaining — deleting partial parquet and resuming...")
        out_path.unlink()   # remove partial file so we rebuild it fully after

    log.info(f"  Year {year}: {len(year_dw):,} (date, window) pairs to process")



    if dry_run:
        log.info(f"  [DRY RUN] Would process {len(year_dw):,} pairs for {year}")
        return pd.DataFrame()

    # Pending pairs (not in checkpoint)
    pending = [
        (row.date_utc, int(row.window_hour))
        for _, row in year_dw.iterrows()
        if (row.date_utc.strftime("%Y-%m-%d"), int(row.window_hour)) not in done
    ]
    log.info(f"  Pending: {len(pending):,}  Already done: {len(year_dw)-len(pending):,}")

    results = []
    success = 0
    failed  = 0
    t_start = time.time()

    if n_workers == 1:
        # Single-threaded
        for i, (dt, wh) in enumerate(pending):
            result = extract_one(dt, wh, centroids)
            date_str = dt.strftime("%Y-%m-%d")
            if result is not None:
                results.append(result)
                save_checkpoint(date_str, wh)
                success += 1
            else:
                failed += 1
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                remaining = (len(pending) - i - 1) / rate
                log.info(
                    f"    [{year}] {i+1}/{len(pending)}  "
                    f"ok={success}  miss={failed}  "
                    f"rate={rate:.1f}/s  "
                    f"ETA={remaining/3600:.1f}h"
                )
    else:
        # Multi-process (ProcessPoolExecutor)
        # cfgrib/eccodes is NOT thread-safe on Windows — using separate
        # processes gives each worker its own isolated eccodes instance
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(extract_one, dt, wh, centroids): (dt, wh)
                for dt, wh in pending
            }
            for i, future in enumerate(as_completed(futures)):
                dt, wh = futures[future]
                date_str = dt.strftime("%Y-%m-%d")
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        save_checkpoint(date_str, wh)
                        success += 1
                    else:
                        failed += 1
                except Exception as e:
                    log.warning(f"  Error {date_str} {wh}h: {e}")
                    failed += 1

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed
                    remaining = (len(pending) - i - 1) / rate
                    log.info(
                        f"    [{year}] {i+1}/{len(pending)}  "
                        f"ok={success}  miss={failed}  "
                        f"rate={rate:.1f}/s  "
                        f"ETA={remaining/3600:.1f}h"
                    )

    # Assemble year result
    if results:
        year_df = pd.concat(results, ignore_index=True)
        year_df.to_parquet(out_path, index=False, compression="snappy")
        log.info(
            f"  Year {year}: {len(year_df):,} rows saved "
            f"-> {out_path.name}  "
            f"({out_path.stat().st_size/1e6:.0f} MB)"
        )
        return year_df
    else:
        log.warning(f"  Year {year}: no HRRR data extracted")
        return pd.DataFrame()


# ── Merge HRRR with training data ─────────────────────────────────────────────
def merge_and_save(centroids: pd.DataFrame, date_windows: pd.DataFrame):
    """
    Merge all per-year HRRR parquets into one file,
    then left-join with training data to produce hrrr_tx_all.parquet.
    Rows with no HRRR coverage get hrrr_pw=0 and NaN features.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    log.info("\nMerging per-year HRRR parquets...")
    year_files = sorted(HRRR_DIR.glob("hrrr_tx_????.parquet"))
    if not year_files:
        log.error("No year parquets found. Run extraction first.")
        return

    log.info(f"  Found {len(year_files)} year file(s): {[f.name for f in year_files]}")

    # Read year files with PyArrow (memory-efficient) then concat
    tables = []
    for f in year_files:
        log.info(f"    Reading {f.name} ({f.stat().st_size/1e6:.0f} MB)...")
        tables.append(pq.read_table(f))
    hrrr_arrow = pa.concat_tables(tables)
    del tables  # free memory before converting
    hrrr_df = hrrr_arrow.to_pandas()
    del hrrr_arrow

    hrrr_df["date_utc"] = pd.to_datetime(hrrr_df["date_utc"]).dt.normalize()
    log.info(f"  Combined HRRR rows: {len(hrrr_df):,}")

    # Load full training parquet to get h3_cell + date + window key
    log.info("  Loading training parquet for join key...")
    train_df = pd.read_parquet(CLEAN_PQ)
    train_df["date_utc"] = pd.to_datetime(train_df["date_utc"]).dt.normalize()

    # Left join: every training row gets its HRRR values (NaN if unavailable)
    merged = train_df.merge(
        hrrr_df.drop(columns=["centroid_lat", "centroid_lon"], errors="ignore"),
        on=["h3_cell", "date_utc", "window_hour"],
        how="left",
    )
    merged["hrrr_pw"] = merged["hrrr_pw"].fillna(0).astype(np.int8)

    out = DATA_DIR / "hrrr" / "hrrr_tx_all.parquet"
    merged.to_parquet(out, index=False, compression="snappy")

    n_with = int((merged["hrrr_pw"] == 1).sum())
    n_without = int((merged["hrrr_pw"] == 0).sum())
    log.info(f"  Rows with HRRR:    {n_with:,}  ({100*n_with/len(merged):.1f}%)")
    log.info(f"  Rows without HRRR: {n_without:,}  (hrrr_pw=0, features=NaN)")
    log.info(f"  Merged parquet:    {out}  ({out.stat().st_size/1e6:.0f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="HRRR Sub-Daily Feature Extraction — Texas Wildfire Model"
    )
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download workers (default: 4)")
    parser.add_argument("--year",    type=int, default=None,
                        help="Extract one year only (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without downloading")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip extraction, only merge existing year files")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("HRRR Sub-Daily Feature Extraction — Texas Wildfire Model")
    log.info("=" * 70)
    log.info(f"  Workers:   {args.workers}")
    log.info(f"  Source:    NOAA AWS S3 (noaa-hrrr-bdp-pds) — free")
    log.info(f"  Variables: temp, rh, wind, vpd, hpbl, dswrf")
    log.info(f"  Output:    {HRRR_DIR}/")

    # Check dependencies
    try:
        import herbie
        from scipy.spatial import cKDTree
        log.info(f"  herbie:    {herbie.__version__}  OK")
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        log.error("Install with: pip install herbie-data cfgrib scipy")
        sys.exit(1)

    if not CLEAN_PQ.exists():
        log.error(f"Training parquet not found: {CLEAN_PQ}")
        log.error("Run train_tx.py first to generate clean parquets.")
        sys.exit(1)

    if args.merge_only:
        centroids, date_windows = load_centroids()
        merge_and_save(centroids, date_windows)
        return

    # Load data
    centroids, date_windows = load_centroids()
    done = load_checkpoint()

    years = [args.year] if args.year else list(range(2014, 2021))

    log.info(f"\nYears to process: {years}")
    if args.dry_run:
        log.info("[DRY RUN MODE — no downloads]")

    # Estimate time
    total_pairs = len(date_windows)
    remaining   = total_pairs - len(done)
    est_seconds = remaining * 20 / args.workers   # ~20s per pair per worker
    log.info(f"\nTotal (date, window) pairs:  {total_pairs:,}")
    log.info(f"Already completed:           {len(done):,}")
    log.info(f"Remaining:                   {remaining:,}")
    log.info(f"Estimated time ({args.workers} workers): "
             f"{est_seconds/3600:.1f} hours")
    log.info("")

    t_global = time.time()
    all_results = []

    for year in years:
        log.info(f"\n{'─'*60}")
        log.info(f"YEAR {year}")
        log.info(f"{'─'*60}")
        year_df = process_year(
            year=year,
            date_windows=date_windows,
            centroids=centroids,
            done=done,
            n_workers=args.workers,
            dry_run=args.dry_run,
        )
        if len(year_df) > 0:
            all_results.append(year_df)

    if args.dry_run:
        log.info("\n[DRY RUN COMPLETE — no data downloaded]")
        log.info("Run without --dry-run to start actual download.")
        return

    total_elapsed = time.time() - t_global
    log.info(f"\nExtraction complete in {total_elapsed/3600:.2f} hours")

    # Only auto-merge when ALL years were processed (not single-year --year runs)
    # For single-year runs: run --merge-only manually after all years are done
    if args.year is None:
        log.info("\nRunning merge...")
        merge_and_save(centroids, date_windows)
        log.info("\n" + "=" * 70)
        log.info("DONE — HRRR features ready")
        log.info("=" * 70)
        log.info("Next: run train_tx_hrrr.py to retrain with HRRR features")
        log.info(f"      Expected TEST AUROC: ~0.93–0.96 (current: 0.8687)")
        log.info("=" * 70)
    else:
        log.info("\n" + "=" * 70)
        log.info(f"Year {args.year} extraction complete.")
        log.info(f"  File: {HRRR_DIR / f'hrrr_tx_{args.year}.parquet'}")
        log.info("  Run the next year, then --merge-only after all years done.")
        log.info("=" * 70)


if __name__ == "__main__":
    main()
