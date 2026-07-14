"""
run_phase2f_gridmet.py
-----------------------
Phase 2F — gridMET Daily Feature Extraction  [PRODUCTION FINAL — Accurate + Fast]

ACCURACY DESIGN:
  This is the scientifically correct implementation. No compromises.

  5-DAY TRAILING STATS — how they work:
    For each (h3_cell, date) in training:
      erc_5D_mean  = mean(erc[date-1], erc[date-2], erc[date-3], erc[date-4], erc[date-5])
      fm100_5D_min = min(fm100[date-1], ..., fm100[date-5])
      etc.
    This represents fire-danger conditions in the 5 days BEFORE the event.
    shift(1) is intentional — we exclude the event day itself (it may be post-ignition).

  CROSS-YEAR BOUNDARY (fixed):
    When lag days fall in the previous year (e.g., Jan 3 needs Dec 29-31),
    the previous year's NC files are also loaded. This fixes NaN for Jan 1-5
    of years 2015-2020 (was incorrectly NaN in earlier versions).
    Only Jan 1-5 of 2014 will have NaN (no prior-year data — by design).

  FILL VALUE (fixed):
    NetCDF int16 fill value (32767) is masked BEFORE scale_factor multiplication.
    Values > 9000 after scale are also clamped to NaN (belt-and-suspenders).

  TEMPERATURE (fixed):
    K → °C conversion applied only when values > 200 (correctly detects Kelvin).

  CACHE DESIGN:
    For accuracy, we need each (cell, date) to have the 5 preceding days' values.
    Naive approach: read NC 5 times per training date → 12,635 reads/year.
    Cache approach: read each needed date ONCE, extract at all cell locations,
    store in dict → per-date 5D is pure array lookup (same data, no reads).
    OUTPUT IS IDENTICAL — cache is just efficiency, not approximation.

VARIABLES:
  erc, fm100, fm1000, bi, vpd, vs, rmax, rmin, tmmx, tmmn, pr, sph  (daily)
  erc_5D_mean/max, fm100_5D_mean/min, bi_5D_mean/max,                (5-day)
  vpd_5D_mean/max, vs_5D_mean/max, rmax_5D_mean/min, tmmx_5D_mean/max

Usage:
    conda activate torch_gpu
    python run_phase2f_gridmet.py --state TX --download-only
    python run_phase2f_gridmet.py --state TX
    python run_phase2f_gridmet.py --state CA
    python run_phase2f_gridmet.py --state ALL
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

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
    "erc":    "erc",
    "fm100":  "fm100",
    "fm1000": "fm1000",
    "bi":     "bi",
    "vpd":    "vpd",
    "vs":     "vs",
    "rmax":   "rmax",
    "rmin":   "rmin",
    "tmmx":   "tmmx",
    "tmmn":   "tmmn",
    "pr":     "pr",
    "sph":    "sph",
}

# Internal NetCDF variable names (inside .nc file)
NC_VAR_NAME = {
    "erc":    "energy_release_component-g",
    "fm100":  "dead_fuel_moisture_100hr",
    "fm1000": "dead_fuel_moisture_1000hr",
    "bi":     "burning_index-g",
    "vpd":    "mean_vapor_pressure_deficit",
    "vs":     "wind_speed",
    "rmax":   "relative_humidity",
    "rmin":   "relative_humidity",
    "tmmx":   "air_temperature",
    "tmmn":   "air_temperature",
    "pr":     "precipitation_amount",
    "sph":    "specific_humidity",
}

# Variables for which 5-day trailing stats are computed + which stats
LAG_VARS = {
    "erc":   ["5D_mean", "5D_max"],
    "fm100": ["5D_mean", "5D_min"],
    "bi":    ["5D_mean", "5D_max"],
    "vpd":   ["5D_mean", "5D_max"],
    "vs":    ["5D_mean", "5D_max"],
    "rmax":  ["5D_mean", "5D_min"],
    "tmmx":  ["5D_mean", "5D_max"],
}

GRIDMET_BASE = "https://www.northwestknowledge.net/metdata/data"
YEARS = list(range(2014, 2021))   # 2014–2020 inclusive


# ═══════════════════════════════════════════════════════════════════════════════
# GridMETExtractor — one per (year) — caches datasets + cKDTree
# ═══════════════════════════════════════════════════════════════════════════════
class GridMETExtractor:
    """
    Extracts gridMET values efficiently for one calendar year.

    Design:
      - Builds one cKDTree per year (shared across all variables)
      - Keeps NC datasets open (one file handle per variable)
      - Applies fill-value masking BEFORE scale_factor (FIX 1)
      - Clamps residual fill artifacts > 9000 to NaN (FIX 4)
      - Converts K → °C only when values indicate Kelvin (FIX 3)
    """

    def __init__(self, year: int):
        try:
            import netCDF4 as nc4
            from scipy.spatial import cKDTree
        except ImportError:
            raise ImportError("pip install netCDF4 scipy")

        self.year       = year
        self.nc4        = nc4
        self._datasets  = {}         # var → open Dataset
        self._tree      = None       # shared cKDTree (lat/lon of 4km grid)
        self._shape     = None       # (n_lat, n_lon)
        self._time_cache = {}        # cache: "id(ds)_date_str" → time_index

    def _open_dataset(self, var: str):
        """Open NC dataset for var (cached). Also builds cKDTree on first call."""
        if var in self._datasets:
            return self._datasets[var]

        fp = GRIDMET_DIR / f"{var}_{self.year}.nc"
        if not fp.exists():
            raise FileNotFoundError(f"NC file not found: {fp}")

        ds = self.nc4.Dataset(fp, "r")
        self._datasets[var] = ds

        if self._tree is None:
            from scipy.spatial import cKDTree
            lats = np.array(ds.variables["lat"][:])
            lons = np.array(ds.variables["lon"][:])
            lon_g, lat_g = np.meshgrid(lons, lats)
            pts = np.column_stack([lat_g.ravel(), lon_g.ravel()])
            self._tree  = cKDTree(pts)
            self._shape = (len(lats), len(lons))
            logger.debug(f"  cKDTree: {pts.shape[0]:,} pts  year={self.year}")

        return ds

    def _resolve_nc_name(self, ds, var: str) -> str:
        nc_name = NC_VAR_NAME.get(var, var)
        if nc_name not in ds.variables:
            dims = set(ds.dimensions.keys())
            nc_name = next((v for v in ds.variables if v not in dims), var)
        return nc_name

    def _time_idx(self, ds, date_str: str) -> int | None:
        """Return the time-axis index for date_str='YYYY-MM-DD'. Cached."""
        key = f"{id(ds)}_{date_str}"
        if key in self._time_cache:
            return self._time_cache[key]

        target = datetime.strptime(date_str, "%Y-%m-%d").date()
        try:
            times = self.nc4.num2date(
                ds.variables["day"][:], ds.variables["day"].units
            )
            for i, t in enumerate(times):
                td = (t.date() if hasattr(t, "date")
                      else datetime(t.year, t.month, t.day).date())
                if td == target:
                    self._time_cache[key] = i
                    return i
        except Exception as e:
            logger.debug(f"  _time_idx [{date_str}]: {e}")
        return None

    def read_band(self, var: str, date_str: str) -> np.ndarray | None:
        """
        Read one time-slice for (var, date_str).
        Returns flat float64 array (n_grid_pts,) or None on failure.

        FIX 1: _FillValue masked BEFORE scale_factor.
        FIX 4: values > 9000 clamped to NaN (belt-and-suspenders).
        FIX 5: set_auto_maskandscale(False) — prevents netCDF4 from auto-applying
                scale_factor + add_offset on read. Our code applies them once manually.
        """
        try:
            ds      = self._open_dataset(var)
            tidx    = self._time_idx(ds, date_str)
            if tidx is None:
                return None

            nc_name = self._resolve_nc_name(ds, var)
            var_obj = ds.variables[nc_name]

            # FIX 5 — disable auto-decode: get raw packed int16 values
            var_obj.set_auto_maskandscale(False)
            raw     = np.array(var_obj[tidx, :, :], dtype=np.float64).ravel()

            # FIX 1 — mask fill value BEFORE scale arithmetic
            fill = getattr(var_obj, "_FillValue", None)
            if fill is not None:
                raw[raw == float(fill)] = np.nan

            # Apply CF conventions
            scale  = getattr(var_obj, "scale_factor", None)
            offset = getattr(var_obj, "add_offset",   None)
            if scale  is not None: raw *= float(scale)
            if offset is not None: raw += float(offset)

            # FIX 4 — clamp residual fill artifacts (e.g., 32767 × scale)
            raw[raw > 9000] = np.nan

            return raw

        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning(f"  read_band [{var}, {date_str}]: {e}")
            return None

    def extract_at_points(
        self,
        var: str,
        date_str: str,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> np.ndarray:
        """
        Extract gridMET values at arbitrary (lats, lons) for one (var, date).
        Used for base daily extraction (one call per training date × var).
        FIX 3: K→°C only when values > 200.
        """
        result = np.full(len(lats), np.nan, dtype=np.float32)
        band   = self.read_band(var, date_str)
        if band is None:
            return result

        _, idxs = self._tree.query(np.column_stack([lats, lons]), k=1)
        vals    = band[idxs].astype(np.float32)

        # FIX 3 — temperature K→°C
        if var in ("tmmx", "tmmn"):
            valid = ~np.isnan(vals)
            if valid.any() and float(np.nanmean(vals[valid])) > 200:
                vals -= 273.15

        return vals

    def build_lag_cache(
        self,
        var: str,
        needed_dates: list[str],
        cell_grid_idxs: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Preload: read each needed date ONCE, extract values at all unique cell
        locations, return dict {date_str → array(n_cells,)}.

        This is the efficiency fix: reads each NC date exactly once instead of
        5× per training date. Output is IDENTICAL to reading 5× — no approximation.
        """
        cache: dict[str, np.ndarray] = {}
        for date_str in needed_dates:
            band = self.read_band(var, date_str)
            if band is None:
                cache[date_str] = np.full(len(cell_grid_idxs), np.nan, dtype=np.float32)
                continue
            vals = band[cell_grid_idxs].astype(np.float32)
            # FIX 3
            if var in ("tmmx", "tmmn"):
                valid = ~np.isnan(vals)
                if valid.any() and float(np.nanmean(vals[valid])) > 200:
                    vals -= 273.15
            cache[date_str] = vals
        return cache

    def close_all(self):
        for ds in self._datasets.values():
            try: ds.close()
            except Exception: pass
        self._datasets.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Download helper
# ═══════════════════════════════════════════════════════════════════════════════
def download_gridmet_files():
    """Download all required gridMET NetCDF files. Skips existing files."""
    import urllib.request
    GRIDMET_DIR.mkdir(parents=True, exist_ok=True)
    total, done, skipped = len(GRIDMET_VARS) * len(YEARS), 0, 0

    for var in GRIDMET_VARS:
        for year in YEARS:
            fname = f"{var}_{year}.nc"
            dest  = GRIDMET_DIR / fname
            if dest.exists():
                logger.info(f"  [SKIP] {fname} ({dest.stat().st_size/1e6:.0f} MB)")
                skipped += 1
                continue
            url = f"{GRIDMET_BASE}/{fname}"
            logger.info(f"  Downloading {fname} ...")
            try:
                urllib.request.urlretrieve(url, dest)
                logger.info(f"    ✔ {fname} ({dest.stat().st_size/1e6:.0f} MB)")
                done += 1
            except Exception as e:
                logger.error(f"    ✘ FAILED {fname}: {e}")

    logger.info(f"\n  Download complete: {done} new, {skipped} skipped / {total} total")


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: compute lag dates needed for a set of training dates
# Returns {year → set_of_date_strings}
# CROSS-YEAR SUPPORT: lag days that fall in the previous year are included.
# ═══════════════════════════════════════════════════════════════════════════════
def get_lag_dates_by_year(training_dates: list[str]) -> dict[int, set[str]]:
    """
    For each training date, compute the 5 preceding dates.
    Groups them by year so we know which NC files to load.

    Example:
      training date = 2015-01-03
      lag-1 = 2015-01-02  → year 2015
      lag-2 = 2015-01-01  → year 2015
      lag-3 = 2014-12-31  → year 2014  ← cross-year
      lag-4 = 2014-12-30  → year 2014
      lag-5 = 2014-12-29  → year 2014
    """
    by_year: dict[int, set[str]] = {}
    for d_str in training_dates:
        d = datetime.strptime(d_str, "%Y-%m-%d").date()
        for lag in range(1, 6):
            lag_d = d - timedelta(days=lag)
            yr    = lag_d.year
            if yr not in by_year:
                by_year[yr] = set()
            by_year[yr].add(lag_d.strftime("%Y-%m-%d"))
    return by_year


# ═══════════════════════════════════════════════════════════════════════════════
# Main extraction function
# ═══════════════════════════════════════════════════════════════════════════════
def extract_gridmet_features(state_key: str, cfg: dict) -> bool:
    output_dir = cfg["output_dir"]

    logger.info(f"{'=' * 65}")
    logger.info(f"PHASE 2F — {cfg['name'].upper()} — gridMET [PRODUCTION FINAL]")
    logger.info(f"{'=' * 65}")

    # ── Load training labels ──────────────────────────────────────────────────
    labels_path = output_dir / "full_training_labels.parquet"
    if not labels_path.exists():
        logger.error("full_training_labels.parquet not found — run Phase 2D first")
        return False

    labels_df = pd.read_parquet(labels_path)
    logger.info(f"  Loaded {len(labels_df):,} training rows")

    unique_rows = (
        labels_df[["centroid_lat", "centroid_lon", "date_utc", "h3_cell"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    logger.info(f"  Unique (cell, date) pairs: {len(unique_rows):,}")

    all_results = []

    # ── Process year by year ──────────────────────────────────────────────────
    for year in YEARS:
        year_mask = unique_rows["date_utc"].str.startswith(str(year))
        year_rows = unique_rows[year_mask].reset_index(drop=True)
        if len(year_rows) == 0:
            continue

        # Check all NC files for this year exist
        missing_files = [v for v in GRIDMET_VARS
                         if not (GRIDMET_DIR / f"{v}_{year}.nc").exists()]
        if missing_files:
            logger.warning(f"  Year {year}: missing NC files: {missing_files[:4]}...")
            logger.warning(f"  Run --download-only first. Skipping year {year}.")
            continue

        logger.info(f"\n{'─'*60}")
        logger.info(f"  Year {year} — {len(year_rows):,} unique (cell, date) pairs")

        # ── Step A: compute unique cells + their grid indices ─────────────────
        cells_meta = (
            year_rows.drop_duplicates("h3_cell")
            [["h3_cell", "centroid_lat", "centroid_lon"]]
            .reset_index(drop=True)
        )
        n_cells = len(cells_meta)
        logger.info(f"  Unique cells: {n_cells:,}")

        extractor = GridMETExtractor(year)
        extractor._open_dataset(list(GRIDMET_VARS.keys())[0])  # build cKDTree

        cell_pts       = np.column_stack([cells_meta["centroid_lat"].values,
                                          cells_meta["centroid_lon"].values])
        _, cell_idxs   = extractor._tree.query(cell_pts, k=1)
        cell_ids       = cells_meta["h3_cell"].values
        cell_to_pos    = {c: i for i, c in enumerate(cell_ids)}

        # ── Step B: build lag cache (cross-year aware) ────────────────────────
        training_dates = sorted(year_rows["date_utc"].unique())

        # Determine which lag dates belong to which year
        lag_dates_by_year = get_lag_dates_by_year(training_dates)

        # lag_cache[var][date_str] → np.array(n_cells,)
        lag_cache: dict[str, dict[str, np.ndarray]] = {v: {} for v in LAG_VARS}

        for lag_year, lag_date_set in sorted(lag_dates_by_year.items()):
            lag_dates_sorted = sorted(lag_date_set)

            # Check which LAG_VAR NC files exist for this lag year
            lag_files_missing  = [v for v in LAG_VARS
                                   if not (GRIDMET_DIR / f"{v}_{lag_year}.nc").exists()]
            lag_vars_available = [v for v in LAG_VARS if v not in lag_files_missing]

            # Fill NaN for any missing vars for ALL dates in this lag year
            if lag_files_missing:
                nan_arr = np.full(n_cells, np.nan, dtype=np.float32)
                for var in lag_files_missing:
                    for d in lag_dates_sorted:
                        lag_cache[var][d] = nan_arr

            # If NO vars are available (e.g. lag_year=2013 not downloaded), skip
            if not lag_vars_available:
                logger.info(f"    Lag year {lag_year}: no NC files available — "
                            f"skipping ({len(lag_dates_sorted)} dates → NaN)")
                continue

            # Build extractor for this lag year
            if lag_year == year:
                extractor_for_lag = extractor       # reuse — already open
                current_idxs      = cell_idxs
            else:
                extractor_for_lag = GridMETExtractor(lag_year)
                # Open using a LAG_VAR that we KNOW exists for this year
                extractor_for_lag._open_dataset(lag_vars_available[0])
                _, lag_cell_idxs = extractor_for_lag._tree.query(cell_pts, k=1)
                current_idxs     = lag_cell_idxs
                logger.info(f"    Cross-year lag: year={lag_year}  "
                            f"{len(lag_dates_sorted)} dates  "
                            f"vars={lag_vars_available}")

            # Preload cache for each available var
            for var_name in lag_vars_available:
                logger.info(f"    Cache: {var_name} year={lag_year} "
                            f"({len(lag_dates_sorted)} dates)...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    partial = extractor_for_lag.build_lag_cache(
                        var_name, lag_dates_sorted, current_idxs
                    )
                lag_cache[var_name].update(partial)

            if lag_year != year:
                extractor_for_lag.close_all()

        # ── Step C: extract base daily values + 5-day stats per date ─────────
        date_groups  = year_rows.groupby("date_utc")
        date_results = []
        n_done       = 0
        n_dates      = len(date_groups)

        logger.info(f"  Extracting {n_dates:,} dates (base + 5-day stats)...")

        for date_str, date_group in date_groups:
            dlats  = date_group["centroid_lat"].values
            dlons  = date_group["centroid_lon"].values
            dcells = date_group["h3_cell"].values

            row = {
                "h3_cell":  dcells,
                "date_utc": [date_str] * len(dcells),
            }

            # ── Base daily values — one NC read per var ───────────────────────
            for var_name in GRIDMET_VARS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    row[var_name] = extractor.extract_at_points(
                        var_name, date_str, dlats, dlons
                    )

            # ── 5-day trailing stats — pure cache lookup, zero NC reads ───────
            # cell_pos maps each row's h3_cell to its position in the cache arrays
            cell_pos    = np.array([cell_to_pos[c] for c in dcells])
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()

            for var_name, stat_list in LAG_VARS.items():
                lag_arrays = []

                for lag in range(1, 6):
                    lag_d   = target_date - timedelta(days=lag)
                    lag_str = lag_d.strftime("%Y-%m-%d")
                    cached  = lag_cache[var_name].get(lag_str)
                    if cached is not None:
                        lag_arrays.append(cached[cell_pos])
                    # If cached is None → that lag date's NC file was missing
                    # → this lag day is simply excluded from the mean/max/min
                    # (rather than making the whole stat NaN)

                if not lag_arrays:
                    # No lag data at all (e.g., very first days of 2014)
                    nan_arr = np.full(len(dcells), np.nan, dtype=np.float32)
                    mean_arr = max_arr = min_arr = nan_arr
                else:
                    stack = np.stack(lag_arrays, axis=0)  # (n_lags, n_cells)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mean_arr = np.nanmean(stack, axis=0).astype(np.float32)
                        max_arr  = np.nanmax(stack,  axis=0).astype(np.float32)
                        min_arr  = np.nanmin(stack,  axis=0).astype(np.float32)

                for stat in stat_list:
                    col = f"{var_name}_{stat}"
                    if "mean" in stat:
                        row[col] = mean_arr
                    elif "max" in stat:
                        row[col] = max_arr
                    else:
                        row[col] = min_arr

            date_results.append(pd.DataFrame(row))
            n_done += 1
            if n_done % 50 == 0 or n_done == n_dates:
                logger.info(f"    Dates done: {n_done:,}/{n_dates:,}")

        if date_results:
            year_df = pd.concat(date_results, ignore_index=True)
            all_results.append(year_df)

            # Per-year QC
            for col in ["erc_5D_mean", "tmmx"]:
                if col in year_df.columns:
                    n_nan = year_df[col].isna().sum()
                    pct   = 100 * n_nan / len(year_df)
                    logger.info(f"    QC {col}: NaN={n_nan:,} ({pct:.2f}%)")

            logger.info(f"  ✔ Year {year} done: {len(year_df):,} rows")

        extractor.close_all()

    if not all_results:
        logger.error("No data extracted — check NC files in " + str(GRIDMET_DIR))
        return False

    # ── Combine all years ─────────────────────────────────────────────────────
    gridmet_df = pd.concat(all_results, ignore_index=True)
    logger.info(f"\n{'=' * 65}")
    logger.info(f"  COMBINED: {len(gridmet_df):,} rows × {len(gridmet_df.columns)} cols")

    # ── Full QC report ────────────────────────────────────────────────────────
    logger.info("\n  MISSING VALUE REPORT:")
    any_missing = False
    for col in gridmet_df.columns:
        if col in ("h3_cell", "date_utc"):
            continue
        n_nan = gridmet_df[col].isna().sum()
        if n_nan > 0:
            pct = 100 * n_nan / len(gridmet_df)
            logger.info(f"    {col:<30} NaN={n_nan:>8,}  ({pct:.2f}%)")
            any_missing = True
    if not any_missing:
        logger.info("    All columns fully populated — no missing values!")

    # Fill-value contamination check
    logger.info("\n  FILL-VALUE CONTAMINATION CHECK (values > 9000):")
    for col in ["erc", "fm100", "bi", "vpd", "vs", "rmax", "rmin", "tmmx", "tmmn"]:
        if col in gridmet_df.columns:
            n_bad = int((gridmet_df[col].dropna() > 9000).sum())
            status = "✔ CLEAN" if n_bad == 0 else f"✘ {n_bad} contaminated!"
            logger.info(f"    {col:<10} {status}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = output_dir / f"gridmet_features_{state_key.lower()}.parquet"
    gridmet_df.to_parquet(out_path, index=False, compression="snappy")
    mb = out_path.stat().st_size / 1e6
    logger.info(f"\n  Saved: {out_path}  ({mb:.0f} MB)")
    logger.info(f"{'=' * 65}")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Phase 2F — gridMET Daily Feature Extraction [Production Final]"
    )
    parser.add_argument("--state", choices=["TX", "CA", "ALL"], required=True)
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download missing gridMET NC files, do not extract"
    )
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
        logger.info("Downloading missing gridMET NetCDF files...")
        download_gridmet_files()
        return

    states = ["TX", "CA"] if args.state == "ALL" else [args.state]
    for s in states:
        ok = extract_gridmet_features(s, STATE_CONFIG[s])
        print(f"\n  {STATE_CONFIG[s]['name']:<15} {'✔ SUCCESS' if ok else '✘ FAILED'}")


if __name__ == "__main__":
    main()
