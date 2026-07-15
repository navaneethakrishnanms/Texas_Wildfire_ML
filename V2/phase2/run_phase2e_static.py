"""
run_phase2e_static.py
----------------------
Phase 2E-B — Static Feature Extraction

Extracts STATIC features for ALL H3 cells in the state grid.
These features never change with date — computed ONCE, saved as a lookup table.

STATIC FEATURES COLLECTED:
  A. LANDFIRE / FSim (raster point-sampling):
     avg_burn_prob, whp, flep4, cfl          ← MANDATORY — download rasters first
     EVH, EVT, EVC_1km, FRG, Land_Cover      ← Already in FPA-FOD attributes

  B. Terrain / DEM (USGS 3DEP SRTM):
     Elevation, Slope, Aspect, TRI, TPI       ← Already in FPA-FOD attributes

  C. Infrastructure (spatial joins):
     No_FireStation_5km, 10km, 20km           ← Point-in-radius counts
     road_common_name_dis                     ← Distance to nearest named road

  D. Social Vulnerability / EJScreen:
     RPL_THEMES, RPL_THEME2, RPL_THEME3       ← CDC SVI census tract join
     Population, Popo_1km                     ← US Census

  E. Location (computed from H3 cell):
     centroid_lat, centroid_lon               ← h3.cell_to_latlng()
     Ecoregion_NA_L2CODE, L3CODE             ← Already in Phase 2B grid

HOW TO COLLECT RASTERS:
  Run: python run_phase2e_static.py --download-check
  This will show which raster files are missing and their download URLs.

DOWNLOAD RASTERS MANUALLY:
  avg_burn_prob:
    URL: https://www.firelab.org/sites/default/files/images/attachments/BP_national.zip
    (extract → BP_national/BP_national.tif)

  whp (Wildfire Hazard Potential 2023):
    URL: https://www.firelab.org/sites/default/files/images/attachments/WHP_2023.zip
    (extract → whp.tif)

  flep4 (LANDFIRE Flame Length Exceedance 2022):
    URL: https://landfire.gov/viewer/ → navigate to FLEP4 → download national GeoTIFF

  cfl (LANDFIRE Canopy Fuel Load 2022):
    URL: https://landfire.gov/viewer/ → navigate to CFL → download national GeoTIFF

  Place all rasters in: V2/data/rasters/

Usage:
    conda activate torch_gpu
    pip install rasterio pyproj h3
    python run_phase2e_static.py --state TX --download-check
    python run_phase2e_static.py --state TX
    python run_phase2e_static.py --state CA
    python run_phase2e_static.py --state ALL

Output:
    phase2/outputs/<state>/static_features_<state>.parquet
    (one row per H3 cell, joined by h3_cell column)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PHASE2_ROOT = Path(__file__).resolve().parent
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from config.phase2_config import STATE_CONFIG, LOGS_DIR, V2_ROOT

logger = logging.getLogger(__name__)

# ── Raster file locations ──────────────────────────────────────────────────────
# ── Raster file locations ──────────────────────────────────────────────────────
RASTER_DIR = V2_ROOT / "data" / "rasters"

REQUIRED_RASTERS = {
    "avg_burn_prob": {
        "file":    RASTER_DIR / "BP_national.tif",
        "url":     "https://www.firelab.org/sites/default/files/images/attachments/BP_national.zip",
        "nodata_fill": 0.0,
        "description": "USFS FSim Burn Probability (50k simulations, 0–1 float)",
    },
    "whp": {
        "file":    RASTER_DIR / "WHP_2023.tif",
        "url":     "https://www.firelab.org/sites/default/files/images/attachments/WHP_2023.zip",
        "nodata_fill": 0.0,
        "description": "USFS Wildfire Hazard Potential 2023 (integer 1–7 or 0–7000)",
    },
    "flep4": {
        "file":    RASTER_DIR / "FLEP4_national.tif",
        "url":     "https://landfire.gov/viewer/ → FLEP4 → Download national GeoTIFF",
        "nodata_fill": 0.0,
        "description": "LANDFIRE Flame Length Exceedance Prob at 4ft",
    },
    "cfl": {
        "file":    RASTER_DIR / "CFL_national.tif",
        "url":     "https://landfire.gov/viewer/ → CFL → Download national GeoTIFF",
        "nodata_fill": 0.0,
        "description": "LANDFIRE Canopy Fuel Load (Mg/ha)",
    },
}



# ── Raster extraction ──────────────────────────────────────────────────────────
def extract_raster_at_points(
    raster_path: Path,
    lats: np.ndarray,
    lons: np.ndarray,
    nodata_fill: float = 0.0,
) -> np.ndarray:
    """
    Extract raster values at multiple lat/lon points.
    Reprojects WGS84 points → raster CRS automatically.

    Parameters
    ----------
    raster_path : Path to GeoTIFF
    lats, lons  : arrays of coordinates (EPSG:4326)
    nodata_fill : value to use when raster has nodata at that point

    Returns
    -------
    numpy array of extracted values, same length as lats
    """
    try:
        import rasterio
        from pyproj import Transformer
    except ImportError:
        raise ImportError("pip install rasterio pyproj")

    values = np.full(len(lats), nodata_fill, dtype=np.float32)

    with rasterio.open(raster_path) as src:
        # Build CRS transformer (WGS84 → raster CRS)
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        nodata = src.nodata

        # Transform all points at once (efficient)
        xs, ys = transformer.transform(lons, lats)  # note: lon=x, lat=y

        # Convert projected coords → row/col
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows = np.array(rows, dtype=int)
        cols = np.array(cols, dtype=int)

        # Clip to raster bounds
        h, w = src.height, src.width
        valid_mask = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

        # Read band 1 (most LANDFIRE products are single-band)
        band = src.read(1)

        # Extract for valid points
        for i in np.where(valid_mask)[0]:
            v = band[rows[i], cols[i]]
            if nodata is not None and v == nodata:
                values[i] = nodata_fill
            elif np.isnan(v):
                values[i] = nodata_fill
            else:
                values[i] = float(v)

    return values


# ── H3 helper ─────────────────────────────────────────────────────────────────
def _get_h3():
    try:
        import h3
        return h3
    except ImportError:
        raise ImportError("pip install h3")


def get_centroids(h3_lib, cells):
    """Vectorized: get centroid lat/lon for a list of H3 cells."""
    lats, lons = [], []
    for cell in cells:
        try:
            try:
                lat, lon = h3_lib.h3_to_geo(cell)
            except AttributeError:
                lat, lon = h3_lib.cell_to_latlng(cell)
        except Exception:
            lat, lon = 0.0, 0.0
        lats.append(lat)
        lons.append(lon)
    return np.array(lats), np.array(lons)


# ── Download check ─────────────────────────────────────────────────────────────
def check_downloads() -> bool:
    """Print status of required raster files."""
    print("\n" + "=" * 70)
    print("  RASTER DOWNLOAD STATUS CHECK")
    print("=" * 70)
    all_present = True
    for name, info in REQUIRED_RASTERS.items():
        path = info["file"]
        if path.exists():
            mb = path.stat().st_size / 1e6
            print(f"  ✔ {name:<20} {mb:>8.0f} MB  {path.name}")
        else:
            print(f"  ✘ {name:<20} MISSING")
            print(f"      Download from: {info['url']}")
            print(f"      Save to:       {path}")
            all_present = False
    print("=" * 70)
    if not all_present:
        print("\n  Some rasters are missing. Download them before running extraction.")
        print(f"  Place all .tif files in: {RASTER_DIR}\n")
    return all_present


# ── Main extraction ───────────────────────────────────────────────────────────
def extract_static_features(state_key: str, cfg: dict) -> bool:
    h3_lib     = _get_h3()
    output_dir = cfg["output_dir"]
    slug       = cfg["slug"]

    logger.info(f"{'=' * 60}")
    logger.info(f"PHASE 2E-B — {cfg['name'].upper()} — Static Feature Extraction")
    logger.info(f"{'=' * 60}")

    # ── Load H3 grid ──────────────────────────────────────────────────────────
    grid_path = output_dir / f"h3_grid_{state_key.lower()}.parquet"
    if not grid_path.exists():
        logger.error(f"H3 grid not found: {grid_path} — run Phase 2B first")
        return False

    grid_df = pd.read_parquet(grid_path)
    logger.info(f"  H3 Grid: {len(grid_df):,} cells (R{cfg['h3_level']})")

    cells = grid_df["h3_cell"].values
    logger.info(f"  Computing cell centroids for {len(cells):,} cells...")
    lats, lons = get_centroids(h3_lib, cells)

    # ── Initialize output dataframe ───────────────────────────────────────────
    out_df = pd.DataFrame({
        "h3_cell":       cells,
        "centroid_lat":  np.round(lats, 6),
        "centroid_lon":  np.round(lons, 6),
    })

    # Carry over existing columns from H3 grid (ecoregion, fire_history, etc.)
    existing_cols = [c for c in grid_df.columns if c != "h3_cell"]
    for col in existing_cols:
        out_df[col] = grid_df[col].values

    # ── Extract LANDFIRE / FSim rasters ───────────────────────────────────────
    RASTER_DIR.mkdir(parents=True, exist_ok=True)

    for feat_name, info in REQUIRED_RASTERS.items():
        raster_path = info["file"]
        if not raster_path.exists():
            logger.warning(f"  RASTER MISSING: {feat_name} → {raster_path}")
            logger.warning(f"    Download from: {info['url']}")
            logger.warning(f"    Filling with {info['nodata_fill']} (will re-run when available)")
            out_df[feat_name] = info["nodata_fill"]
            continue

        logger.info(f"  Extracting {feat_name} from {raster_path.name}...")
        try:
            values = extract_raster_at_points(raster_path, lats, lons, info["nodata_fill"])
            out_df[feat_name] = values
            n_zero = (values == 0).sum()
            logger.info(f"    ✔ Done  |  min={values.min():.4f}  max={values.max():.4f}  "
                        f"zero_cells={n_zero:,} ({100*n_zero/len(values):.1f}%)")
        except Exception as e:
            logger.error(f"    ✘ Failed: {e}")
            out_df[feat_name] = info["nodata_fill"]

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = output_dir / f"static_features_{state_key.lower()}.parquet"
    out_df.to_parquet(out_path, index=False, compression="snappy")
    mb = out_path.stat().st_size / 1e6

    logger.info(f"\n  Saved: {out_path}  ({mb:.0f} MB)")
    logger.info(f"  Columns: {list(out_df.columns)}")
    logger.info(f"  Shape:   {out_df.shape}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 2E-B — Static Feature Extraction")
    parser.add_argument("--state", choices=["TX", "CA", "ALL"], required=True)
    parser.add_argument("--download-check", action="store_true",
                        help="Just check which raster files need downloading")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "phase2e_static.log", encoding="utf-8"),
        ],
    )

    if args.download_check:
        check_downloads()
        return

    states = ["TX", "CA"] if args.state == "ALL" else [args.state]
    for s in states:
        ok = extract_static_features(s, STATE_CONFIG[s])
        print(f"  {STATE_CONFIG[s]['name']:<15} {'✔ SUCCESS' if ok else '✘ FAILED'}")


if __name__ == "__main__":
    main()
