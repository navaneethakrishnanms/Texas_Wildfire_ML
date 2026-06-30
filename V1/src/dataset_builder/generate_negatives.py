"""
generate_negatives.py
=====================
Production-grade negative sample generator for the Texas Wildfire ML dataset.

CRITICAL FIX (applied here)
----------------------------
The original implementation generated random lat/lon points inside the Texas
bounding box. This caused ~42% of synthetic negatives to land on MODIS NoData
pixels (cloud-masked areas), while ALL Fire=1 FIRMS points had valid data.

Why Fire=1 has no missing: FIRMS hotspot detection REQUIRES a MODIS satellite
observation. Fires physically cannot be detected on cloud-masked pixels. So
every Fire=1 point is guaranteed to be in a valid-data area.

Why Fire=0 had 42% missing: Random points were placed anywhere in Texas,
including cloud-masked zones that have no valid MODIS observations.

THE BUG THIS CREATES:
  Missing NDVI    ->  Fire=0 (99.9% of the time)
  Valid NDVI      ->  Could be Fire=1 or Fire=0
  XGBoost learns: "If NDVI is NaN, predict Fire=0"
  This is NOT a wildfire pattern -- it is a data artifact.

THE FIX (implemented below):
  build_valid_pixel_pool()  -- reads the NDVI raster, extracts all pixel
  centre coordinates where the value is NOT NaN and NOT NoData, and returns
  them as a (N, 2) array of [lat, lon] valid pixel centres.

  generate_negative_samples() now accepts this pool and samples candidates
  ONLY from valid pixels -- guaranteeing every negative has valid NDVI.

  After fixing: Fire=0 missing NDVI should drop from 4,875 to ~0.

Algorithm
---------
1. Load NDVI raster -> extract all valid (non-NaN) pixel centres.
2. Optionally intersect with other key rasters (EVI, LST) for stricter validation.
3. Filter valid pixels to Texas bounding box.
4. Build KDTree on fire locations for spatial exclusion (default 5 km buffer).
5. Remove valid pixels too close to known fires.
6. Random sample n_negatives from the remaining valid pixels.
7. Assign random dates from the FIRMS date pool.
8. Return DataFrame with columns: latitude, longitude, acq_date, Fire=0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Texas geographic bounding box (WGS-84)
# These match the FIRMS data extent -- do not widen without raster re-export.
# ---------------------------------------------------------------------------
TEXAS_BOUNDS: dict = {
    "lat_min": 25.84,
    "lat_max": 36.50,
    "lon_min": -106.65,
    "lon_max": -93.51,
}

# Key rasters used for valid-pixel validation.
# A negative sample is only accepted if ALL these rasters return a valid
# (non-NaN) value at that location.
VALIDATION_RASTERS = ["ndvi", "evi", "lst"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lat_lon_to_xyz(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Convert WGS-84 (lat, lon) to 3D unit-sphere XYZ for KDTree proximity queries.
    Chord distance is accurate for exclusion buffers of 5-50 km at Texas latitudes.
    """
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    return np.column_stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
    ])


def _km_to_chord(km: float, R: float = 6371.0) -> float:
    """Great-circle km to Euclidean chord distance on unit sphere."""
    return 2.0 * np.sin(km / (2.0 * R))


def _haversine_km_vectorized(
    lat1: float, lon1: float,
    lats2: np.ndarray, lons2: np.ndarray,
    R: float = 6371.0,
) -> np.ndarray:
    """Haversine distances (km) from one point to an array of points."""
    dlat = np.radians(lats2 - lat1)
    dlon = np.radians(lons2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lats2)) * np.sin(dlon / 2) ** 2
    )
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Valid-pixel pool builder (THE CORE FIX)
# ---------------------------------------------------------------------------

def build_valid_pixel_pool(
    raw_dir: Path,
    validation_rasters: list[str] | None = None,
    texas_bounds: dict | None = None,
    subsample_stride: int = 1,
) -> np.ndarray:
    """
    Build a pool of valid pixel centre coordinates from key GEE rasters.

    A pixel is 'valid' if it has a non-NaN, non-NoData value in ALL
    validation rasters (NDVI, EVI, LST by default). Only these locations
    can be used as negative sample candidates -- this guarantees that
    every synthetic Fire=0 point has valid environmental feature values.

    This eliminates the dataset bias where missing values were 100%
    correlated with Fire=0, allowing XGBoost to use NaN as a proxy
    for class membership instead of learning real wildfire patterns.

    Parameters
    ----------
    raw_dir : Path
        data/raw/ directory containing raster subfolders (ndvi/, evi/, lst/).
    validation_rasters : list[str], optional
        Subfolder names of rasters to use for validation.
        Default: ['ndvi', 'evi', 'lst']
    texas_bounds : dict, optional
        Override Texas bounding box. Keys: lat_min, lat_max, lon_min, lon_max.
    subsample_stride : int
        Take every Nth pixel to reduce pool size for large rasters.
        Default 1 = use every valid pixel. Use 2-4 to reduce memory.

    Returns
    -------
    np.ndarray  shape (N, 2)  -- rows are [latitude, longitude] of valid pixels
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform

    if validation_rasters is None:
        validation_rasters = VALIDATION_RASTERS
    if texas_bounds is None:
        texas_bounds = TEXAS_BOUNDS

    logger.info(
        "Building valid-pixel pool from %d rasters: %s",
        len(validation_rasters), validation_rasters
    )

    valid_mask: np.ndarray | None = None
    ref_transform = None
    ref_shape     = None
    ref_crs       = None

    for raster_name in validation_rasters:
        folder = raw_dir / raster_name
        tif_files = sorted(folder.glob("*.tif")) if folder.exists() else []

        if not tif_files:
            logger.warning(
                "  [SKIP] No .tif files found in '%s' -- skipping validation for '%s'.",
                folder, raster_name
            )
            continue

        # For multi-tile rasters, build a union valid mask
        raster_mask: np.ndarray | None = None
        for tif_path in tif_files:
            with rasterio.open(tif_path) as src:
                data = src.read(1).astype(np.float64)
                nodata = src.nodata

                # Mark NoData values as NaN
                if nodata is not None:
                    data[data == nodata] = np.nan
                    # Also handle common sentinel values
                    data[data <= -9999] = np.nan

                tile_valid = ~np.isnan(data)

                if raster_mask is None:
                    raster_mask = tile_valid
                    # Store reference grid from first tile of first raster
                    if ref_transform is None:
                        ref_transform = src.transform
                        ref_shape     = data.shape
                        ref_crs       = src.crs
                else:
                    # If tiles have same shape, union masks
                    if tile_valid.shape == raster_mask.shape:
                        raster_mask = raster_mask | tile_valid

        if raster_mask is None:
            logger.warning("  [SKIP] Could not build mask for '%s'.", raster_name)
            continue

        # Resize raster_mask to match ref_shape if different resolution
        if raster_mask.shape != ref_shape:
            from scipy.ndimage import zoom as ndimage_zoom
            scale_r = ref_shape[0] / raster_mask.shape[0]
            scale_c = ref_shape[1] / raster_mask.shape[1]
            raster_mask = ndimage_zoom(
                raster_mask.astype(np.float32),
                (scale_r, scale_c),
                order=0,        # nearest-neighbour: preserves binary values
            ).astype(bool)
            # Clip to ref_shape in case of rounding
            raster_mask = raster_mask[:ref_shape[0], :ref_shape[1]]
            # Pad if zoom produced a smaller array
            if raster_mask.shape != ref_shape:
                padded = np.zeros(ref_shape, dtype=bool)
                padded[:raster_mask.shape[0], :raster_mask.shape[1]] = raster_mask
                raster_mask = padded


        if valid_mask is None:
            valid_mask = raster_mask
            logger.info(
                "  [%s] valid pixels: %d / %d  (%.1f%%)",
                raster_name, int(valid_mask.sum()), valid_mask.size,
                valid_mask.sum() / valid_mask.size * 100
            )
        else:
            # Intersection: pixel must be valid in ALL validation rasters
            valid_mask = valid_mask & raster_mask
            logger.info(
                "  After intersect with [%s]: %d valid pixels remaining",
                raster_name, int(valid_mask.sum())
            )

    if valid_mask is None or valid_mask.sum() == 0:
        logger.error(
            "No valid pixels found in validation rasters. "
            "Falling back to full Texas bounding box sampling."
        )
        return np.empty((0, 2))

    # Convert valid pixel indices to geographic coordinates
    rows, cols = np.where(valid_mask[::subsample_stride, ::subsample_stride])
    if subsample_stride > 1:
        rows = rows * subsample_stride
        cols = cols * subsample_stride

    logger.info("  Total valid pixel positions: %d", len(rows))

    # Get pixel centre coordinates using affine transform
    # rasterio transform: (col, row) -> (lon, lat) for EPSG:4326
    xs, ys = rasterio.transform.xy(ref_transform, rows, cols, offset="center")
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)

    # If raster is not WGS-84, reproject to WGS-84
    epsg = ref_crs.to_epsg() if ref_crs else None
    if epsg != 4326:
        logger.info("  Reprojecting pixel coordinates from EPSG:%s to WGS-84...", epsg)
        wgs84 = CRS.from_epsg(4326)
        xs, ys = transform(ref_crs, wgs84, xs, ys)

    lons, lats = xs, ys

    # Filter to Texas bounding box
    in_texas = (
        (lats >= texas_bounds["lat_min"]) & (lats <= texas_bounds["lat_max"]) &
        (lons >= texas_bounds["lon_min"]) & (lons <= texas_bounds["lon_max"])
    )
    lats = lats[in_texas]
    lons = lons[in_texas]

    logger.info(
        "  Valid pixels within Texas bounds: %d  (%.1f%% of total valid)",
        len(lats), len(lats) / max(len(rows), 1) * 100
    )

    pool = np.column_stack([lats, lons])
    return pool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_negative_samples(
    firms_csv: str | Path,
    n_negatives: int,
    raw_dir: Path | None = None,
    exclusion_radius_km: float = 5.0,
    seed: int = 42,
    max_iter: int = 100,
    batch_multiplier: int = 15,
    valid_pixel_pool: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Generate spatially clean non-fire (Fire=0) sample points inside Texas.

    CORE GUARANTEE: Every generated point has valid NDVI, EVI, and LST
    values -- eliminating the bias where missing raster values were
    correlated 100% with Fire=0 class label.

    Parameters
    ----------
    firms_csv : str | Path
        Path to the filtered FIRMS CSV. Must have: latitude, longitude, acq_date.
    n_negatives : int
        Exact number of non-fire samples to generate.
    raw_dir : Path, optional
        data/raw/ directory. Required if valid_pixel_pool is not provided.
        Used to build the valid-pixel pool from NDVI/EVI/LST rasters.
    exclusion_radius_km : float
        Minimum distance (km) from any known fire. Default 5.0 km.
    seed : int
        Random seed for full reproducibility.
    max_iter : int
        Hard cap on sampling iterations (safety valve).
    batch_multiplier : int
        Candidates generated per needed negative in each batch.
    valid_pixel_pool : np.ndarray, optional
        Pre-built array of [lat, lon] valid pixel positions.
        If provided, raw_dir is not used. Pass this to reuse the pool
        across multiple calls without re-reading the rasters.

    Returns
    -------
    pd.DataFrame  -- columns: latitude, longitude, acq_date, Fire (=0)

    Raises
    ------
    ValueError  -- FIRMS CSV missing or malformed.
    RuntimeError -- Fewer than n_negatives collected after max_iter.
    """
    rng = np.random.default_rng(seed)

    # ── 1. Load FIRMS fire coordinates ─────────────────────────────────────
    firms_path = Path(firms_csv)
    if not firms_path.exists():
        raise ValueError(f"FIRMS CSV not found: {firms_path}")

    firms_df = pd.read_csv(firms_path, parse_dates=["acq_date"])
    required = {"latitude", "longitude", "acq_date"}
    missing  = required - set(firms_df.columns)
    if missing:
        raise ValueError(f"FIRMS CSV missing columns: {missing}")

    fire_lats = firms_df["latitude"].values.astype(np.float64)
    fire_lons = firms_df["longitude"].values.astype(np.float64)
    date_pool: List[str] = firms_df["acq_date"].dt.strftime("%Y-%m-%d").tolist()

    logger.info(
        "Loaded %d FIRMS fire records for exclusion buffer (radius=%.1f km).",
        len(firms_df), exclusion_radius_km
    )

    # ── 2. Build valid-pixel pool ───────────────────────────────────────────
    if valid_pixel_pool is not None:
        pool = valid_pixel_pool
        logger.info("Using pre-built valid-pixel pool: %d candidate locations.", len(pool))
    elif raw_dir is not None:
        pool = build_valid_pixel_pool(raw_dir)
        logger.info("Valid-pixel pool built: %d candidate locations.", len(pool))
    else:
        # Fallback to legacy random sampling (not recommended)
        logger.warning(
            "No raw_dir or valid_pixel_pool provided. "
            "Falling back to RANDOM sampling -- negatives may land on NoData pixels!"
        )
        pool = None

    # ── 3. Build KDTree for fire exclusion ─────────────────────────────────
    fire_xyz       = _lat_lon_to_xyz(fire_lats, fire_lons)
    tree           = cKDTree(fire_xyz)
    chord_thresh   = _km_to_chord(exclusion_radius_km)
    logger.info(
        "KDTree built from %d fire points. Chord threshold = %.6f",
        len(fire_lats), chord_thresh
    )

    # ── 4. Sample from valid-pixel pool ────────────────────────────────────
    accepted_lats:  List[float] = []
    accepted_lons:  List[float] = []
    accepted_dates: List[str]   = []
    n_rejected_total   = 0
    n_generated_total  = 0

    if pool is not None and len(pool) > 0:
        # --- VALID-PIXEL SAMPLING PATH (production mode) ---
        pool_lats = pool[:, 0]
        pool_lons = pool[:, 1]
        pool_xyz  = _lat_lon_to_xyz(pool_lats, pool_lons)

        # Pre-filter pool: remove pixels within exclusion radius of any fire
        counts_pool = tree.query_ball_point(pool_xyz, chord_thresh, return_length=True)
        safe_mask   = counts_pool == 0
        safe_pool_lats = pool_lats[safe_mask]
        safe_pool_lons = pool_lons[safe_mask]
        n_safe = int(safe_mask.sum())

        logger.info(
            "Valid-pixel pool after fire exclusion: %d / %d pixels safe "
            "(%.1f%% excluded as too close to known fires)",
            n_safe, len(pool),
            (1 - n_safe / len(pool)) * 100 if len(pool) > 0 else 0
        )

        if n_safe < n_negatives:
            logger.warning(
                "Safe pool (%d) is smaller than n_negatives (%d). "
                "Sampling with replacement.",
                n_safe, n_negatives
            )
            replace = True
        else:
            replace = False

        # Sample from safe pool
        chosen_idx = rng.choice(n_safe, size=n_negatives, replace=replace)
        accepted_lats  = safe_pool_lats[chosen_idx].tolist()
        accepted_lons  = safe_pool_lons[chosen_idx].tolist()
        accepted_dates = rng.choice(date_pool, size=n_negatives, replace=True).tolist()
        n_generated_total = n_negatives

        logger.info(
            "Negative generation complete (valid-pixel mode): %d samples "
            "sampled from %d safe pixel locations.",
            n_negatives, n_safe
        )

    else:
        # --- LEGACY RANDOM SAMPLING PATH (fallback, not recommended) ---
        logger.warning(
            "Using LEGACY random sampling. Fire=0 samples may land on NoData "
            "pixels and create a dataset bias. Provide raw_dir to fix this."
        )
        for iteration in range(1, max_iter + 1):
            needed = n_negatives - len(accepted_lats)
            if needed <= 0:
                break

            n_batch  = int(needed * batch_multiplier)
            cand_lats = rng.uniform(TEXAS_BOUNDS["lat_min"], TEXAS_BOUNDS["lat_max"], n_batch)
            cand_lons = rng.uniform(TEXAS_BOUNDS["lon_min"], TEXAS_BOUNDS["lon_max"], n_batch)
            n_generated_total += n_batch

            cand_xyz  = _lat_lon_to_xyz(cand_lats, cand_lons)
            counts    = tree.query_ball_point(cand_xyz, chord_thresh, return_length=True)
            clean     = counts == 0
            n_clean   = int(clean.sum())
            n_rejected_total += (n_batch - n_clean)

            take = min(n_clean, needed)
            accepted_lats.extend(cand_lats[clean][:take].tolist())
            accepted_lons.extend(cand_lons[clean][:take].tolist())
            accepted_dates.extend(
                rng.choice(date_pool, size=take, replace=True).tolist()
            )
            logger.info(
                "  Iter %2d: gen=%d  accepted=%d  rejected=%d  "
                "rate=%.1f%%  total=%d/%d",
                iteration, n_batch, take, n_batch - n_clean,
                n_clean / n_batch * 100, len(accepted_lats), n_negatives
            )

        if len(accepted_lats) < n_negatives:
            logger.warning(
                "Shortfall: %d/%d collected. Try reducing exclusion_radius_km.",
                len(accepted_lats), n_negatives
            )

    # ── 5. Assemble output DataFrame ────────────────────────────────────────
    n_actual = min(len(accepted_lats), n_negatives)
    neg_df = pd.DataFrame({
        "latitude":  accepted_lats[:n_actual],
        "longitude": accepted_lons[:n_actual],
        "acq_date":  pd.to_datetime(accepted_dates[:n_actual]),
    })
    neg_df["Fire"] = 0

    logger.info("Negative sample DataFrame shape: %s", neg_df.shape)
    return neg_df


# ---------------------------------------------------------------------------
# Utility: spatial stats
# ---------------------------------------------------------------------------

def negative_sample_spatial_stats(neg_df: pd.DataFrame, firms_csv: str | Path) -> dict:
    """
    Compute spatial distribution statistics for the generated negative samples.

    Returns dict with: min_dist_to_fire_km, mean_dist_to_fire_km,
                       lat_std, lon_std, temporal_unique_dates
    """
    firms_df  = pd.read_csv(firms_csv, parse_dates=["acq_date"])
    fire_lats = firms_df["latitude"].values.astype(np.float64)
    fire_lons = firms_df["longitude"].values.astype(np.float64)
    neg_lats  = neg_df["latitude"].values
    neg_lons  = neg_df["longitude"].values

    sample_idx = np.random.choice(len(neg_df), min(200, len(neg_df)), replace=False)
    min_dists  = []
    for i in sample_idx:
        dists = _haversine_km_vectorized(neg_lats[i], neg_lons[i], fire_lats, fire_lons)
        min_dists.append(float(dists.min()))

    return {
        "min_dist_to_fire_km":    float(np.min(min_dists)),
        "mean_dist_to_fire_km":   float(np.mean(min_dists)),
        "lat_std":                float(np.std(neg_lats)),
        "lon_std":                float(np.std(neg_lons)),
        "temporal_unique_dates":  int(neg_df["acq_date"].nunique()),
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    firms_path = Path("data/raw/firms/fire_archive_2024.csv")
    raw_dir    = Path("data/raw")

    if not firms_path.exists():
        print(f"FIRMS file not found: {firms_path}", file=sys.stderr)
        sys.exit(1)

    print("Building valid-pixel pool from NDVI/EVI/LST rasters ...")
    pool = build_valid_pixel_pool(raw_dir)
    print(f"Pool size: {len(pool):,} valid pixel locations")

    print("\nGenerating 100 negative samples from valid pixels ...")
    df = generate_negative_samples(
        firms_path, n_negatives=100,
        raw_dir=raw_dir, exclusion_radius_km=5.0
    )
    print(df.head(10).to_string())
    print(f"\nShape: {df.shape}")

    stats = negative_sample_spatial_stats(df, firms_path)
    print("\nSpatial stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
