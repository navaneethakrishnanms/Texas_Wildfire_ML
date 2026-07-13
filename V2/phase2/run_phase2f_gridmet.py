"""
run_phase2f_gridmet.py
-----------------------
Phase 2F — gridMET Daily Feature Extraction

Extracts daily weather features from gridMET NetCDF files for every
(h3_cell, date_utc) pair in the full_training_labels.parquet.

KEY RULE — CRITICAL:
  Each row (whether fire or non-fire) gets weather extracted at ITS OWN
  centroid lat/lon. Non-fire cells do NOT inherit the paired fire cell's
  weather values. This is the single most important rule for avoiding bias.

gridMET VARIABLES DOWNLOADED:
  erc      → Energy Release Component (NFDRS)     ← best single predictor
  fm100    → 100-hr dead fuel moisture
  fm1000   → 1000-hr dead fuel moisture
  bi       → Burning Index
  vpd      → Vapor Pressure Deficit (kPa)
  vs       → Wind Speed (m/s)
  rmax     → Max Relative Humidity (%)
  rmin     → Min Relative Humidity (%)
  tmmx     → Max Temperature (K) → convert to °C
  tmmn     → Min Temperature (K) → convert to °C
  pr       → Precipitation (mm)
  sph      → Specific Humidity (kg/kg)

5-DAY TRAILING STATS (computed after extraction):
  erc_5D_mean, erc_5D_max
  fm100_5D_mean, fm100_5D_min
  bi_5D_mean, bi_5D_max
  vpd_5D_mean, vpd_5D_max
  vs_5D_mean, vs_5D_max
  rmax_5D_mean, rmax_5D_min
  tmmx_5D_mean, tmmx_5D_max

DOWNLOAD STEPS:
  Run: python run_phase2f_gridmet.py --download-only
  This downloads all required NetCDF files to: V2/data/gridmet/

  Files are ~200–500 MB per variable per year = ~15–20 GB total for 12 vars × 7 years.
  All publicly available at: https://www.climatologylab.org/wget-gridmet.html

Usage:
    conda activate torch_gpu
    pip install netCDF4 h3 scipy
    python run_phase2f_gridmet.py --state TX --download-only
    python run_phase2f_gridmet.py --state TX
    python run_phase2f_gridmet.py --state CA
    python run_phase2f_gridmet.py --state ALL

Output:
    phase2/outputs/<state>/gridmet_features_<state>.parquet
    (one row per (h3_cell, date_utc) pair in training table)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PHASE2_ROOT = Path(__file__).resolve().parent
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from config.phase2_config import STATE_CONFIG, LOGS_DIR, V2_ROOT

logger = logging.getLogger(__name__)

GRIDMET_DIR = V2_ROOT / "data" / "gridmet"

# Variable name → gridMET file abbreviation
GRIDMET_VARS = {
    "erc":   "erc",
    "fm100": "fm100",
    "fm1000":"fm1000",
    "bi":    "bi",
    "vpd":   "vpd",
    "vs":    "vs",
    "rmax":  "rmax",
    "rmin":  "rmin",
    "tmmx":  "tmmx",
    "tmmn":  "tmmn",
    "pr":    "pr",
    "sph":   "sph",
}

# gridMET internal NetCDF variable name (inside the .nc file)
# These differ from the filename abbreviations used above
NC_VAR_NAME = {
    "erc":    "energy_release_component-g",
    "fm100":  "dead_fuel_moisture_100hr",
    "fm1000": "dead_fuel_moisture_1000hr",
    "bi":     "burning_index-g",
    "vpd":    "mean_vapor_pressure_deficit",
    "vs":     "wind_speed",
    "rmax":   "relative_humidity",   # rmax file contains max RH
    "rmin":   "relative_humidity",   # rmin file contains min RH
    "tmmx":   "air_temperature",
    "tmmn":   "air_temperature",
    "pr":     "precipitation_amount",
    "sph":    "specific_humidity",
}

GRIDMET_BASE = "https://www.northwestknowledge.net/metdata/data"
YEARS = list(range(2014, 2021))  # 2014–2020


# ── NetCDF loader and cKDTree builder ──────────────────────────────────────────
class GridMETExtractor:
    """
    Efficient extractor: builds one cKDTree per year and caches it.
    For each query (lat, lon, date, variable): finds nearest 4km grid point.
    """

    def __init__(self, year: int):
        try:
            import netCDF4 as nc4
            from scipy.spatial import cKDTree
        except ImportError:
            raise ImportError("pip install netCDF4 scipy")

        self.year = year
        self.nc4  = nc4
        self._datasets = {}
        self._tree     = None
        self._lats     = None
        self._lons     = None
        self._time_idx_cache = {}  # date_str → time index
        self._CKDTree = cKDTree

    def _load_grid(self, var: str):
        """Load one NetCDF and build/return the cKDTree (built once per year)."""
        var_file = GRIDMET_DIR / f"{var}_{self.year}.nc"
        if not var_file.exists():
            raise FileNotFoundError(f"gridMET file not found: {var_file}")

        ds = self.nc4.Dataset(var_file, "r")

        # Build lat/lon grid and cKDTree on first load
        if self._tree is None:
            lats = ds.variables["lat"][:]
            lons = ds.variables["lon"][:]
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
            self._tree = self._CKDTree(coords)
            self._lats = lats
            self._lons = lons
            self._shape = (len(lats), len(lons))
            logger.debug(f"    cKDTree built: {len(coords):,} grid points")

        return ds

    def _get_time_idx(self, ds, date_str: str) -> int | None:
        """Find the index in the time dimension for a given date string."""
        key = f"{self.year}_{date_str}"
        if key in self._time_idx_cache:
            return self._time_idx_cache[key]

        try:
            import netCDF4 as nc4
            times = nc4.num2date(
                ds.variables["day"][:],
                ds.variables["day"].units
            )
            target = datetime.strptime(date_str, "%Y-%m-%d").date()
            for i, t in enumerate(times):
                if hasattr(t, "date"):
                    td = t.date()
                else:
                    td = datetime(t.year, t.month, t.day).date()
                if td == target:
                    self._time_idx_cache[key] = i
                    return i
            return None
        except Exception as e:
            logger.warning(f"Time index lookup failed for {date_str}: {e}")
            return None

    def extract_batch(self, var: str, date_str: str, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Extract gridMET values for a list of (lat, lon) points on a given date.

        Returns
        -------
        np.ndarray of shape (n,) with extracted values (NaN on failure)
        """
        result = np.full(len(lats), np.nan, dtype=np.float32)

        try:
            ds       = self._load_grid(var)
            time_idx = self._get_time_idx(ds, date_str)
            if time_idx is None:
                return result

            # Find nearest grid point for all query points
            pts = np.column_stack([lats, lons])
            _, idxs = self._tree.query(pts, k=1)

            # Resolve internal NC variable name (may differ from file abbreviation)
            nc_name = NC_VAR_NAME.get(var, var)

            # Fallback: if mapped name not found, try the short name
            if nc_name not in ds.variables:
                # Try to find any variable that isn't a dimension
                dims = set(ds.dimensions.keys())
                data_vars = [v for v in ds.variables if v not in dims]
                nc_name = data_vars[0] if data_vars else var
                logger.debug(f"NC name fallback for '{var}' → '{nc_name}'")

            # Read band for this date (2D array: lat × lon)
            data = ds.variables[nc_name][time_idx, :, :].ravel()

            # Apply scale_factor and add_offset if present (NetCDF conventions)
            var_obj = ds.variables[nc_name]
            if hasattr(var_obj, "scale_factor"):
                data = data * var_obj.scale_factor
            if hasattr(var_obj, "add_offset"):
                data = data + var_obj.add_offset

            # Fill missing
            fill = getattr(var_obj, "_FillValue", None)
            if fill is not None:
                data = np.where(data == fill, np.nan, data)

            result = data[idxs].astype(np.float32)

        except Exception as e:
            logger.warning(f"gridMET extraction failed [{var}, {date_str}]: {e}")

        return result

    def close_all(self):
        for ds in self._datasets.values():
            try:
                ds.close()
            except Exception:
                pass


# ── Download ──────────────────────────────────────────────────────────────────
def download_gridmet_files():
    """Download all required gridMET NetCDF files."""
    import urllib.request
    GRIDMET_DIR.mkdir(parents=True, exist_ok=True)

    total_files = len(GRIDMET_VARS) * len(YEARS)
    done = 0
    skipped = 0

    for var in GRIDMET_VARS:
        for year in YEARS:
            fname = f"{var}_{year}.nc"
            dest  = GRIDMET_DIR / fname
            if dest.exists():
                logger.info(f"  [SKIP] {fname} already exists ({dest.stat().st_size/1e6:.0f} MB)")
                skipped += 1
                continue

            url = f"{GRIDMET_BASE}/{fname}"
            logger.info(f"  Downloading {fname} from {url} ...")
            try:
                urllib.request.urlretrieve(url, dest)
                mb = dest.stat().st_size / 1e6
                logger.info(f"    ✔ Saved {fname} ({mb:.0f} MB)")
                done += 1
            except Exception as e:
                logger.error(f"    ✘ Failed {fname}: {e}")

    logger.info(f"\n  Download complete: {done} new, {skipped} skipped, {total_files} total")


# ── Main extraction ───────────────────────────────────────────────────────────
def extract_gridmet_features(state_key: str, cfg: dict) -> bool:
    output_dir = cfg["output_dir"]

    logger.info(f"{'=' * 60}")
    logger.info(f"PHASE 2F — {cfg['name'].upper()} — gridMET Daily Feature Extraction")
    logger.info(f"{'=' * 60}")

    # ── Load training labels (fire + non-fire) ────────────────────────────────
    labels_path = output_dir / "full_training_labels.parquet"
    if not labels_path.exists():
        logger.error(f"full_training_labels.parquet not found — run Phase 2D first")
        return False

    labels_df = pd.read_parquet(labels_path)
    logger.info(f"  Loaded {len(labels_df):,} training rows")

    # Get unique (lat, lon, date) combinations to minimize API calls
    unique_rows = labels_df[["centroid_lat", "centroid_lon", "date_utc", "h3_cell"]].copy()
    unique_rows = unique_rows.drop_duplicates()
    logger.info(f"  Unique (cell, date) pairs: {len(unique_rows):,}")

    # ── Process year by year ──────────────────────────────────────────────────
    all_results = []

    for year in YEARS:
        year_mask = unique_rows["date_utc"].str.startswith(str(year))
        year_rows = unique_rows[year_mask].copy()

        if len(year_rows) == 0:
            continue

        logger.info(f"\n  Year {year}: {len(year_rows):,} unique (cell, date) pairs")

        # Check if all required files exist for this year
        missing = []
        for var in GRIDMET_VARS:
            fp = GRIDMET_DIR / f"{var}_{year}.nc"
            if not fp.exists():
                missing.append(f"{var}_{year}.nc")

        if missing:
            logger.warning(f"  Missing files: {missing[:3]}{'...' if len(missing) > 3 else ''}")
            logger.warning(f"  Run with --download-only first")
            continue

        extractor = GridMETExtractor(year)
        year_lats = year_rows["centroid_lat"].values
        year_lons = year_rows["centroid_lon"].values

        # Group by date for batch extraction
        date_groups = year_rows.groupby("date_utc")
        logger.info(f"  Processing {len(date_groups):,} unique dates...")

        date_results = []
        n_dates_done = 0

        for date_str, date_group in date_groups:
            dlats = date_group["centroid_lat"].values
            dlons = date_group["centroid_lon"].values
            dcells = date_group["h3_cell"].values

            row_dict = {
                "h3_cell":   dcells,
                "date_utc":  [date_str] * len(dcells),
            }

            for var_name, var_key in GRIDMET_VARS.items():
                vals = extractor.extract_batch(var_key, date_str, dlats, dlons)

                # Convert temperature K → °C
                if var_name in ("tmmx", "tmmn") and not np.all(np.isnan(vals)):
                    # gridMET temp is already in K for raw, in °C with scale factor
                    # Check range: if > 200 assume Kelvin, convert
                    if np.nanmean(vals) > 100:
                        vals = vals - 273.15

                row_dict[var_name] = vals

            date_results.append(pd.DataFrame(row_dict))
            n_dates_done += 1

            if n_dates_done % 100 == 0:
                logger.info(f"    Dates processed: {n_dates_done:,}/{len(date_groups):,}")

        if date_results:
            year_df = pd.concat(date_results, ignore_index=True)
            all_results.append(year_df)
            logger.info(f"  Year {year} done: {len(year_df):,} rows")

        extractor.close_all()

    if not all_results:
        logger.error("No gridMET data extracted — check files exist in " + str(GRIDMET_DIR))
        return False

    # ── Combine all years ──────────────────────────────────────────────────────
    gridmet_df = pd.concat(all_results, ignore_index=True)
    logger.info(f"\n  Total extracted: {len(gridmet_df):,} rows × {len(gridmet_df.columns)} columns")

    # ── Compute 5-day trailing statistics ─────────────────────────────────────
    logger.info("\n  Computing 5-day trailing statistics...")
    gridmet_df = gridmet_df.sort_values(["h3_cell", "date_utc"]).reset_index(drop=True)

    lag_vars = {
        "erc":   ["5D_mean", "5D_max"],
        "fm100": ["5D_mean", "5D_min"],
        "bi":    ["5D_mean", "5D_max"],
        "vpd":   ["5D_mean", "5D_max"],
        "vs":    ["5D_mean", "5D_max"],
        "rmax":  ["5D_mean", "5D_min"],
        "tmmx":  ["5D_mean", "5D_max"],
    }

    for var, stats in lag_vars.items():
        if var not in gridmet_df.columns:
            continue
        grp = gridmet_df.groupby("h3_cell")[var]
        for stat in stats:
            col_name = f"{var}_{stat}"
            if "mean" in stat:
                gridmet_df[col_name] = grp.transform(
                    lambda x: x.shift(1).rolling(5, min_periods=1).mean())
            elif "max" in stat:
                gridmet_df[col_name] = grp.transform(
                    lambda x: x.shift(1).rolling(5, min_periods=1).max())
            elif "min" in stat:
                gridmet_df[col_name] = grp.transform(
                    lambda x: x.shift(1).rolling(5, min_periods=1).min())

    logger.info(f"  Columns after lag features: {len(gridmet_df.columns)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = output_dir / f"gridmet_features_{state_key.lower()}.parquet"
    gridmet_df.to_parquet(out_path, index=False, compression="snappy")
    mb = out_path.stat().st_size / 1e6
    logger.info(f"\n  Saved: {out_path}  ({mb:.0f} MB)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 2F — gridMET Daily Feature Extraction")
    parser.add_argument("--state", choices=["TX", "CA", "ALL"], required=True)
    parser.add_argument("--download-only", action="store_true",
                        help="Only download gridMET files, do not extract")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    GRIDMET_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "phase2f_gridmet.log", encoding="utf-8"),
        ],
    )

    if args.download_only:
        logger.info("Downloading gridMET NetCDF files...")
        download_gridmet_files()
        return

    states = ["TX", "CA"] if args.state == "ALL" else [args.state]
    for s in states:
        ok = extract_gridmet_features(s, STATE_CONFIG[s])
        print(f"  {STATE_CONFIG[s]['name']:<15} {'✔ SUCCESS' if ok else '✘ FAILED'}")


if __name__ == "__main__":
    main()
