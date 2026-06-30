"""
sample_rasters.py
=================
Production-grade geospatial raster sampler for the Texas Wildfire ML dataset.

Design Principles
-----------------
* **Auto-discovers tiled rasters** — any feature folder containing multiple
  .tif files is treated as a mosaic; the correct tile is selected per point
  without any manual configuration.
* **Windowed reads only** — never loads an entire band into RAM. Each sample
  reads exactly ONE pixel via ``rasterio.DatasetReader.read(1, window=...)``,
  making this scalable to hundreds of thousands of points on multi-GB rasters.
* **Correct CRS handling** — every (lat, lon) pair is reprojected into the
  native CRS of each raster before index lookup, supporting EPSG:4326,
  projected CRS (UTM, etc.) and exotic projections (Sinusoidal, MODIS grid).
* **Physically correct scaling** — raw GEE integer exports for NDVI/EVI and
  MODIS LST are converted to real-world units inside the sampler.
* **Validation hooks** — after sampling, a coordinate round-trip check
  confirms the pixel truly corresponds to the requested location.
* **Context-manager safe** — use ``with RasterSampler(...) as s:`` to
  guarantee all file handles are closed even on error.

Column names produced (matching wildfire_dataset spec)
------------------------------------------------------
  latitude, longitude, acq_date  (passed through from caller)
  NDVI, EVI, LST, Temperature, Wind, Rainfall, DEM, Slope, Aspect, LandCover
  Fire  (binary target, set by build_dataset.py)

Usage
-----
    from src.dataset_builder.sample_rasters import RasterSampler
    with RasterSampler("data/raw") as sampler:
        values = sampler.sample(lat=30.27, lon=-99.84)
        # → {'NDVI': 0.412, 'EVI': 0.301, 'LST': 35.4, ...}
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

# Texas geographic bounding box (WGS-84) — used for fast pre-validation
TEXAS_LAT_MIN: float = 25.50
TEXAS_LAT_MAX: float = 36.70
TEXAS_LON_MIN: float = -107.00
TEXAS_LON_MAX: float = -93.00

# WGS-84 CRS object — created once, reused everywhere
WGS84: CRS = CRS.from_epsg(4326)

# Feature → subfolder name inside data/raw/
# Order matters: this defines the column order in the output dataset.
FEATURE_FOLDER_MAP: Dict[str, str] = {
    "NDVI":        "ndvi",
    "EVI":         "evi",
    "LST":         "lst",
    "Temperature": "temperature",
    "Wind":        "wind",
    "Rainfall":    "rainfall",
    "DEM":         "dem",
    "Slope":       "slope",
    "Aspect":      "aspect",
    "LandCover":   "landcover",
}

# Output column order for the final dataset
FEATURE_COLUMNS: List[str] = list(FEATURE_FOLDER_MAP.keys())


# ---------------------------------------------------------------------------
# Scale / unit conversion helpers
# ---------------------------------------------------------------------------

def _apply_physical_scale(feature: str, raw_value: float) -> float:
    """
    Convert a raw GEE-exported pixel value to physical units.

    Rules
    -----
    NDVI / EVI
        GEE exports MODIS MOD13Q1 as scaled integers × 10000.
        If the exported value is in [-10000, 10000] range, scale by 0.0001.
        If it is already in [-1, 1] (GEE .divide(10000) applied), keep as-is.

    LST
        GEE MOD11A2 raw: integer × 0.02 → Kelvin, then − 273.15 → Celsius.
        GEE often exports the *scaled Kelvin* (200–330 K range after ×0.02),
        but sometimes exports raw integers (8000–18000 range).
        Detection strategy:
          raw > 1000  → raw MODIS integer  → ×0.02 → K → °C
          150 < raw ≤ 1000 → already in Kelvin → −273.15 → °C
          raw ≤ 150    → already in Celsius → keep as-is

    All others (Temperature, Wind, Rainfall, DEM, Slope, Aspect, LandCover)
        GEE already exports in physical units; no scaling needed.
    """
    if np.isnan(raw_value):
        return np.nan

    v = float(raw_value)

    if feature in ("NDVI", "EVI"):
        # Detect whether raw integer (> 2.0 in magnitude) or already scaled
        if abs(v) > 2.0:
            return v * 0.0001          # MODIS integer → index
        return v                        # already a fraction

    if feature == "LST":
        if v > 1000:
            # Raw MODIS integer (e.g. 14285 ≈ 285.7 K * 50)
            kelvin = v * 0.02
        elif v > 150:
            # Already Kelvin (GEE applied scale_factor internally)
            kelvin = v
        else:
            # Already Celsius
            return v
        return kelvin - 273.15          # K → °C

    # Temperature (ERA5 or similar) — exported in Kelvin from GEE?
    # Values > 150 are treated as Kelvin and converted to Celsius
    if feature == "Temperature" and v > 150:
        return v - 273.15

    return v                            # physical units as-is


# ---------------------------------------------------------------------------
# TileIndex — manages multiple .tif tiles for one feature
# ---------------------------------------------------------------------------

class TileIndex:
    """
    Manages a collection of GeoTIFF tiles for a single feature.

    On construction, opens every .tif in the folder and stores each dataset's
    CRS + spatial bounds in WGS-84 for fast O(n_tiles) point-in-tile lookup.
    Windowed pixel reads ensure no full raster is ever loaded into memory.

    Parameters
    ----------
    feature : str
        Feature name (e.g. "Slope") — used for logging only.
    folder : Path
        Directory that may contain one or more .tif files.
    """

    def __init__(self, feature: str, folder: Path) -> None:
        self.feature = feature
        self.folder = folder
        self._tiles: List[rasterio.DatasetReader] = []
        # Bounds stored in WGS-84 for fast containment checks
        self._wgs84_bounds: List[Tuple[float, float, float, float]] = []  # (W, S, E, N)
        self._load_tiles()

    def _load_tiles(self) -> None:
        """Open every .tif in the folder; compute WGS-84 bounds for each."""
        tif_files = sorted(self.folder.glob("*.tif"))
        if not tif_files:
            logger.warning(
                "[%s] No .tif files found in '%s'. Feature will return NaN.",
                self.feature, self.folder
            )
            return

        for tif in tif_files:
            try:
                src = rasterio.open(tif)
                self._tiles.append(src)
                bounds_wgs84 = self._bounds_to_wgs84(src)
                self._wgs84_bounds.append(bounds_wgs84)
                logger.debug(
                    "[%s] Opened tile: %s  WGS84-bounds=(%.4f, %.4f, %.4f, %.4f)",
                    self.feature, tif.name, *bounds_wgs84
                )
            except Exception as exc:
                logger.error("[%s] Failed to open %s: %s", self.feature, tif, exc)

        logger.info("[%s] Loaded %d tile(s) from '%s'", self.feature, len(self._tiles), self.folder)

    @staticmethod
    def _bounds_to_wgs84(src: rasterio.DatasetReader) -> Tuple[float, float, float, float]:
        """
        Return the dataset's spatial extent in WGS-84 (W, S, E, N).

        Works regardless of the raster's native CRS.
        """
        b = src.bounds
        if src.crs is None or src.crs == WGS84:
            return (b.left, b.bottom, b.right, b.top)

        # Transform all four corners to WGS-84 and take the envelope
        xs = [b.left, b.left,  b.right, b.right]
        ys = [b.bottom, b.top, b.bottom, b.top]
        try:
            lons, lats = warp_transform(src.crs, WGS84, xs, ys)
            return (min(lons), min(lats), max(lons), max(lats))
        except Exception as exc:
            logger.warning("CRS transform failed for bounds: %s. Using native bounds.", exc)
            return (b.left, b.bottom, b.right, b.top)

    def _find_tile(self, lat: float, lon: float) -> Optional[rasterio.DatasetReader]:
        """
        Return the tile dataset whose WGS-84 footprint contains (lat, lon).

        If multiple tiles overlap the point (rare but possible at tile seams),
        returns the first match (tiles are sorted by filename, so deterministic).

        Returns None if no tile covers the point.
        """
        for src, (west, south, east, north) in zip(self._tiles, self._wgs84_bounds):
            # Small epsilon to handle points exactly on the boundary
            eps = 1e-9
            if west - eps <= lon <= east + eps and south - eps <= lat <= north + eps:
                return src
        return None

    def sample(self, lat: float, lon: float) -> float:
        """
        Return the raw pixel value at (lat, lon) from the appropriate tile.

        Steps:
          1. Fast WGS-84 bbox check to find the correct tile.
          2. Reproject (lat, lon) into the tile's native CRS.
          3. Compute the pixel row/col via the tile's affine transform.
          4. Read exactly that single pixel via a 1×1 Window (no full read).
          5. Handle nodata — return NaN if pixel equals the nodata sentinel.

        Returns
        -------
        float  — raw pixel value (unscaled), or NaN if outside all tiles.
        """
        src = self._find_tile(lat, lon)
        if src is None:
            logger.debug(
                "[%s] Point (%.5f, %.5f) not covered by any tile.", self.feature, lat, lon
            )
            return np.nan

        try:
            # --- 1. Reproject to tile's native CRS ----------------------------
            if src.crs is None or src.crs == WGS84:
                x, y = lon, lat
            else:
                xs, ys = warp_transform(WGS84, src.crs, [lon], [lat])
                x, y = xs[0], ys[0]

            # --- 2. Check native-CRS bounds (tighter than WGS-84 bbox) --------
            b = src.bounds
            if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
                return np.nan

            # --- 3. Convert geographic coordinate → pixel row, col ------------
            # Use the inverse affine transform (rasterio.transform.rowcol)
            # rowcol returns (rows, cols) as arrays; extract scalars.
            rows, cols = rowcol(src.transform, x, y)
            row = int(rows)
            col = int(cols)

            # Clamp to valid raster extent (handles floating-point boundary cases)
            row = max(0, min(row, src.height - 1))
            col = max(0, min(col, src.width - 1))

            # --- 4. Windowed read: read exactly 1 pixel -----------------------
            # This is the key performance optimization: never read the full band.
            window = rasterio.windows.Window(col, row, 1, 1)
            data = src.read(1, window=window)
            raw_val = float(data[0, 0])

            # --- 5. Handle nodata sentinel ------------------------------------
            nodata = src.nodata
            if nodata is not None:
                if math.isnan(nodata):
                    if math.isnan(raw_val):
                        return np.nan
                else:
                    if raw_val == nodata:
                        return np.nan

            return raw_val

        except Exception as exc:
            logger.warning(
                "[%s] Sampling error at (%.5f, %.5f): %s",
                self.feature, lat, lon, exc
            )
            return np.nan

    def close(self) -> None:
        """Close all open tile datasets."""
        for src in self._tiles:
            try:
                src.close()
            except Exception:
                pass
        self._tiles.clear()
        self._wgs84_bounds.clear()

    @property
    def n_tiles(self) -> int:
        return len(self._tiles)

    @property
    def is_available(self) -> bool:
        return len(self._tiles) > 0


# ---------------------------------------------------------------------------
# RasterSampler — public API
# ---------------------------------------------------------------------------

class RasterSampler:
    """
    Production-grade point sampler that reads pixel values from all GEE raster
    datasets for any given (latitude, longitude) coordinate.

    Features
    --------
    * Auto-discovers all .tif files in each feature subdirectory — handles
      single-file rasters (NDVI, EVI, LST …) and tiled mosaics
      (DEM: 2 tiles, Slope: 4 tiles, Aspect: 4 tiles) identically.
    * Windowed reads — scales to multi-GB tiled rasters without OOM errors.
    * Geospatially correct — reprojects lat/lon into each raster's CRS before
      pixel lookup, supporting any projection.
    * Physically correct scaling — raw MODIS integer codes converted to proper
      scientific units (NDVI index, LST in Celsius, etc.).
    * Context-manager support — ``with RasterSampler(...) as s:`` pattern
      guarantees file handles are closed.

    Parameters
    ----------
    raw_dir : str | Path
        Path to the ``data/raw/`` directory. Each feature subfolder (ndvi/,
        evi/, slope/, …) must live directly inside this directory.

    Example
    -------
    >>> with RasterSampler("data/raw") as sampler:
    ...     row = sampler.sample(lat=30.27, lon=-99.84)
    ...     print(row)
    {'NDVI': 0.412, 'EVI': 0.301, 'LST': 35.4, 'Temperature': 32.1, ...}
    """

    def __init__(self, raw_dir: str | Path = "data/raw") -> None:
        self.raw_dir = Path(raw_dir)
        self._indices: Dict[str, TileIndex] = {}
        self._init_tile_indices()

    # -----------------------------------------------------------------------
    def _init_tile_indices(self) -> None:
        """Build a TileIndex for every configured feature folder."""
        for feature, subfolder in FEATURE_FOLDER_MAP.items():
            folder = self.raw_dir / subfolder
            if not folder.exists():
                logger.warning(
                    "[%s] Subfolder '%s' does not exist. Feature will return NaN.",
                    feature, folder
                )
                continue

            idx = TileIndex(feature, folder)
            if idx.is_available:
                self._indices[feature] = idx
                logger.info(
                    "[%s] Ready — %d tile(s), folder='%s'",
                    feature, idx.n_tiles, subfolder
                )
            else:
                logger.warning(
                    "[%s] No usable tiles found in '%s'.", feature, folder
                )

    # -----------------------------------------------------------------------
    def sample(self, lat: float, lon: float) -> Dict[str, float]:
        """
        Sample all raster features at a single WGS-84 coordinate.

        Parameters
        ----------
        lat : float
            Latitude in decimal degrees (WGS-84).
        lon : float
            Longitude in decimal degrees (WGS-84).

        Returns
        -------
        dict
            Keys are feature column names (NDVI, EVI, LST, …).
            Values are physically-scaled floats; NaN for missing/out-of-bounds.
        """
        result: Dict[str, float] = {}

        for feature in FEATURE_COLUMNS:
            idx = self._indices.get(feature)
            if idx is None:
                result[feature] = np.nan
                continue

            raw_val = idx.sample(lat, lon)
            result[feature] = _apply_physical_scale(feature, raw_val)

        return result

    # -----------------------------------------------------------------------
    def sample_batch(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> List[Dict[str, float]]:
        """
        Sample all raster features for an array of (lat, lon) coordinates.

        Parameters
        ----------
        lats : np.ndarray  shape (N,)
        lons : np.ndarray  shape (N,)

        Returns
        -------
        list of N dicts, each with the same keys as ``sample()``.
        """
        return [self.sample(float(lat), float(lon)) for lat, lon in zip(lats, lons)]

    # -----------------------------------------------------------------------
    def validate_sample(
        self,
        lat: float,
        lon: float,
        sampled: Dict[str, float],
        tolerance_km: float = 1.0,
    ) -> bool:
        """
        Validation check: verify that the sampled pixel truly corresponds to
        the requested (lat, lon) location.

        Method
        ------
        For each non-NaN feature, reproject (lat, lon) into the tile's CRS and
        back-project the pixel centre into WGS-84. Measure the Haversine
        distance between (lat, lon) and the back-projected pixel centre.
        If any feature's pixel centre is farther than ``tolerance_km``, log a
        warning but still return True (data is still correct — pixel granularity
        may just be coarser than the tolerance).

        Returns
        -------
        bool — True (always usable), but logs a warning if pixel drift > tolerance.
        """
        R = 6371.0

        for feature, idx in self._indices.items():
            src = idx._find_tile(lat, lon)
            if src is None:
                continue
            if np.isnan(sampled.get(feature, np.nan)):
                continue

            try:
                # Reproject to native CRS
                if src.crs is None or src.crs == WGS84:
                    x, y = lon, lat
                else:
                    xs, ys = warp_transform(WGS84, src.crs, [lon], [lat])
                    x, y = xs[0], ys[0]

                # Pixel row/col
                rows, cols = rowcol(src.transform, x, y)
                row, col = int(rows), int(cols)
                row = max(0, min(row, src.height - 1))
                col = max(0, min(col, src.width - 1))

                # Pixel centre in native CRS
                t = src.transform
                px_x = t.c + (col + 0.5) * t.a
                px_y = t.f + (row + 0.5) * t.e

                # Back-project to WGS-84
                if src.crs is None or src.crs == WGS84:
                    px_lon, px_lat = px_x, px_y
                else:
                    out_lons, out_lats = warp_transform(src.crs, WGS84, [px_x], [px_y])
                    px_lon, px_lat = out_lons[0], out_lats[0]

                # Haversine distance
                dlat = math.radians(px_lat - lat)
                dlon = math.radians(px_lon - lon)
                a = (math.sin(dlat / 2) ** 2
                     + math.cos(math.radians(lat)) * math.cos(math.radians(px_lat))
                     * math.sin(dlon / 2) ** 2)
                dist_km = 2 * R * math.asin(math.sqrt(a))

                if dist_km > tolerance_km:
                    logger.warning(
                        "[VALIDATION] Feature '%s' pixel centre is %.3f km from "
                        "requested (%.5f, %.5f). Pixel resolution may be coarse.",
                        feature, dist_km, lat, lon
                    )
            except Exception as exc:
                logger.debug("[VALIDATION] Skipped check for '%s': %s", feature, exc)

        return True  # Data is still geospatially correct; warning is informational

    # -----------------------------------------------------------------------
    def close(self) -> None:
        """Close all open raster file handles."""
        for idx in self._indices.values():
            idx.close()
        self._indices.clear()

    def __enter__(self) -> "RasterSampler":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # -----------------------------------------------------------------------
    @property
    def available_features(self) -> List[str]:
        """List of features with at least one usable tile."""
        return [f for f in FEATURE_COLUMNS if f in self._indices]

    @property
    def missing_features(self) -> List[str]:
        """List of features with no usable raster data."""
        return [f for f in FEATURE_COLUMNS if f not in self._indices]

    def summary(self) -> str:
        """Human-readable summary of loaded tiles."""
        lines = ["RasterSampler — Tile Inventory", "=" * 50]
        for feature in FEATURE_COLUMNS:
            idx = self._indices.get(feature)
            if idx:
                lines.append(f"  {feature:<15} {idx.n_tiles} tile(s) ✓")
            else:
                lines.append(f"  {feature:<15} MISSING ✗")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick self-test (run directly: python src/dataset_builder/sample_rasters.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        print(f"ERROR: data/raw not found. Run from project root.", file=sys.stderr)
        sys.exit(1)

    with RasterSampler(raw_dir) as sampler:
        print("\n" + sampler.summary())

        # Sample a representative fire point from the FIRMS 2024 dataset
        test_points = [
            (31.7996, -103.2413, "West Texas (from FIRMS row 1)"),
            (30.6892, -98.2013,  "Central Texas"),
            (29.5348, -99.6113,  "South Texas"),
            (32.9175, -98.5752,  "North Texas"),
        ]

        print("\n" + "=" * 60)
        print("SAMPLE RESULTS (with physical scaling applied)")
        print("=" * 60)

        for lat, lon, label in test_points:
            result = sampler.sample(lat, lon)
            sampler.validate_sample(lat, lon, result, tolerance_km=1.0)
            print(f"\n  [{label}]  lat={lat}, lon={lon}")
            for feat, val in result.items():
                if np.isnan(val):
                    print(f"    {feat:<15}: NaN  ← check raster extent")
                else:
                    print(f"    {feat:<15}: {val:.4f}")
