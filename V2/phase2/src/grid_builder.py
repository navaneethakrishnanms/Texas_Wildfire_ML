"""
src/grid_builder.py
---------------------
Phase 2B — H3 Grid Construction

Builds the spatial foundation for the entire wildfire prediction system.
For each state, generates every H3 cell at the target resolution that
covers the state boundary, then assigns:
  - Centroid lat/lon  (for API data collection)
  - EPA Level-3 Ecoregion  (for DAY-ECO-MATCHED negative sampling)
  - Fire count (from historical data — helps identify active fire zones)
  - Burnable flag (placeholder = True; refined in Phase 2D with LANDFIRE)

Output columns:
  h3_cell         : H3 cell ID (string)
  centroid_lat    : Cell centroid latitude
  centroid_lon    : Cell centroid longitude
  h3_resolution   : H3 resolution (7 for TX, 8 for CA)
  state           : State abbreviation
  ecoregion_l3    : EPA Level-3 Ecoregion code (nearest-neighbor from fire data)
  ecoregion_l2    : EPA Level-2 Ecoregion code
  fire_count      : Number of historical fires mapped to this cell (0 = no fire history)
  has_fire_history: bool — True if any fire ever occurred in this cell (2014-2020)
  burnable        : bool — True by default; updated in Phase 2D with LANDFIRE EVT
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from shapely.geometry import mapping, shape

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# H3 API Compatibility (supports both h3-py v3 and v4)
# ─────────────────────────────────────────────────────────────────────────────

def _get_h3():
    """Import h3 and return version-aware wrapper functions."""
    try:
        import h3
        return h3
    except ImportError:
        raise ImportError(
            "h3 library not installed.\n"
            "Run:  pip install h3\n"
            "Then re-run this script."
        )


def _polyfill(h3, geojson: dict, resolution: int) -> set[str]:
    """Polyfill a GeoJSON polygon with H3 cells — handles v3 and v4 API."""
    try:
        # h3 v3 API
        cells = h3.polyfill(geojson, resolution, geo_json_conformant=True)
        return set(cells)
    except (AttributeError, TypeError):
        pass
    try:
        # h3 v4 API
        geo = shape(geojson)
        return set(h3.geo_to_cells(geo, resolution))
    except AttributeError:
        raise RuntimeError("h3 version not supported. Install h3>=3.7 or h3>=4.0")


def _cell_to_latlng(h3, cell: str) -> tuple[float, float]:
    """Get centroid (lat, lng) of H3 cell — handles v3 and v4."""
    try:
        return h3.h3_to_geo(cell)          # v3 → (lat, lng)
    except AttributeError:
        return h3.cell_to_latlng(cell)     # v4 → (lat, lng)


def _cell_to_boundary(h3, cell: str) -> list:
    """Get boundary of H3 cell — handles v3 and v4."""
    try:
        return h3.h3_to_geo_boundary(cell)  # v3
    except AttributeError:
        return h3.cell_to_boundary(cell)    # v4


# ─────────────────────────────────────────────────────────────────────────────
# State Boundary Config
# ─────────────────────────────────────────────────────────────────────────────

# Bounding boxes [min_lon, min_lat, max_lon, max_lat]
STATE_BBOX = {
    "TX": {
        "min_lon": -106.65, "min_lat": 25.84,
        "max_lon": -93.51,  "max_lat": 36.50,
        # Approximate polygon (key border vertices — more accurate than bbox)
        # Texas outline: clockwise from NW corner
        "polygon_coords": [
            [-106.65, 36.50],  # NW (New Mexico border)
            [-100.00, 36.50],  # N (Oklahoma panhandle W)
            [-100.00, 36.50],
            [-94.43,  33.94],  # NE (Arkansas/Louisiana border)
            [-93.51,  31.00],  # E (Louisiana border)
            [-93.82,  29.75],  # SE coast
            [-96.50,  25.84],  # S tip (Brownsville)
            [-100.00, 28.00],  # S border (Rio Grande)
            [-104.02, 29.57],  # SW (Big Bend)
            [-106.65, 31.90],  # W (El Paso)
            [-106.65, 36.50],  # Back to NW
        ],
    },
    "CA": {
        "min_lon": -124.41, "min_lat": 32.53,
        "max_lon": -114.13, "max_lat": 42.01,
        "polygon_coords": [
            [-124.41, 42.01],  # NW (Oregon border)
            [-120.00, 42.01],  # N border
            [-119.99, 39.00],  # NE (Nevada border)
            [-114.63, 35.00],  # SE (Arizona border)
            [-114.13, 32.62],  # SE corner
            [-117.13, 32.53],  # S (San Diego)
            [-120.50, 34.45],  # W coast (Santa Barbara)
            [-124.41, 38.00],  # W coast (San Francisco)
            [-124.41, 42.01],  # Back to NW
        ],
    },
}


def _build_state_polygon(state_key: str) -> dict:
    """
    Build GeoJSON polygon for a state.
    Uses approximate polygon first; falls back to bbox.
    """
    cfg = STATE_BBOX[state_key]
    coords = cfg["polygon_coords"]

    geojson = {
        "type": "Polygon",
        "coordinates": [coords],
    }
    return geojson


def _build_bbox_polygon(state_key: str) -> dict:
    """Build simple bounding-box GeoJSON polygon for a state."""
    cfg = STATE_BBOX[state_key]
    min_lon, min_lat = cfg["min_lon"], cfg["min_lat"]
    max_lon, max_lat = cfg["max_lon"], cfg["max_lat"]

    coords = [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]
    return {"type": "Polygon", "coordinates": [coords]}


# ─────────────────────────────────────────────────────────────────────────────
# Ecoregion Assignment
# ─────────────────────────────────────────────────────────────────────────────

def _assign_ecoregions_kdtree(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    fire_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign EPA Level-2 and Level-3 ecoregion codes to each H3 cell centroid
    using nearest-neighbor assignment from historical fire data.

    Since fire points span the entire state (every ecoregion has fires),
    nearest-neighbor gives accurate ecoregion assignment for all cells.

    Parameters
    ----------
    grid_lats : centroid latitudes of all H3 cells
    grid_lons : centroid longitudes of all H3 cells
    fire_df   : fire records with LATITUDE, LONGITUDE, ecoregion columns

    Returns
    -------
    (ecoregion_l3_array, ecoregion_l2_array)
    """
    # Determine ecoregion column names
    l3_col = None
    l2_col = None

    for col in fire_df.columns:
        cl = col.lower()
        if "l3code" in cl or "l3_code" in cl:
            l3_col = col
        elif "l2code" in cl or "l2_code" in cl:
            l2_col = col

    if l3_col is None:
        logger.warning("  Ecoregion L3 column not found in fire data — using zeros")
        return np.zeros(len(grid_lats), dtype=float), np.zeros(len(grid_lats), dtype=float)

    # Drop NaN ecoregion rows
    eco_df = fire_df[["LATITUDE", "LONGITUDE", l3_col]].dropna()
    if l2_col:
        eco_df = fire_df[["LATITUDE", "LONGITUDE", l3_col, l2_col]].dropna()

    if eco_df.empty:
        logger.warning("  No valid ecoregion data found — using zeros")
        return np.zeros(len(grid_lats), dtype=float), np.zeros(len(grid_lats), dtype=float)

    logger.info(f"  Building KD-tree from {len(eco_df):,} fire points for ecoregion assignment...")

    # Build KD-tree on fire point coordinates
    fire_coords = eco_df[["LATITUDE", "LONGITUDE"]].values
    tree = KDTree(fire_coords)

    # Query nearest fire point for each grid cell centroid
    grid_coords = np.stack([grid_lats, grid_lons], axis=1)
    _, indices = tree.query(grid_coords, k=1, workers=-1)

    l3_values = eco_df[l3_col].values[indices]
    l2_values = eco_df[l2_col].values[indices] if l2_col else np.zeros(len(grid_lats))

    logger.info(f"  Assigned {len(set(l3_values))} unique L3 ecoregion codes to {len(grid_lats):,} cells")

    return l3_values, l2_values


# ─────────────────────────────────────────────────────────────────────────────
# Fire Point → H3 Cell Mapping
# ─────────────────────────────────────────────────────────────────────────────

def _map_fires_to_h3_cells(
    h3_lib: Any,
    fire_df: pd.DataFrame,
    resolution: int,
) -> pd.Series:
    """
    Map each fire point to its H3 cell and count fires per cell.

    Returns pd.Series mapping h3_cell → fire_count.
    """
    lat_col = "LATITUDE"
    lon_col = "LONGITUDE"

    if lat_col not in fire_df.columns or lon_col not in fire_df.columns:
        logger.warning("  LATITUDE/LONGITUDE not found in fire data — skipping fire count")
        return pd.Series(dtype=int)

    fire_cells = []
    for lat, lon in zip(fire_df[lat_col], fire_df[lon_col]):
        try:
            try:
                cell = h3_lib.geo_to_h3(lat, lon, resolution)   # v3
            except AttributeError:
                cell = h3_lib.latlng_to_cell(lat, lon, resolution)  # v4
            fire_cells.append(cell)
        except Exception:
            continue

    return pd.Series(fire_cells).value_counts()


# ─────────────────────────────────────────────────────────────────────────────
# Main Grid Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_h3_grid(
    state_key: str,
    resolution: int,
    parquet_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Build the complete H3 grid for a state.

    Steps:
      1. Generate all H3 cells covering the state bounding polygon
      2. Compute centroid lat/lon for each cell
      3. Assign ecoregion from fire data (KD-tree nearest-neighbor)
      4. Count historical fires per cell
      5. Save parquet + summary CSV

    Parameters
    ----------
    state_key    : 'TX' or 'CA'
    resolution   : H3 resolution (7 for TX, 8 for CA)
    parquet_path : Path to processed fire parquet (from Phase 1)
    output_dir   : Where to save outputs

    Returns
    -------
    pd.DataFrame  The complete H3 grid table
    """
    state_name = {"TX": "Texas", "CA": "California"}[state_key]
    logger.info(f"Phase 2B — H3 Grid Construction [{state_name}]")
    logger.info(f"  H3 Resolution : R{resolution}")
    output_dir.mkdir(parents=True, exist_ok=True)

    h3 = _get_h3()
    logger.info(f"  h3 version: {getattr(h3, '__version__', 'unknown')}")

    # ── Step 1: Generate H3 cells ─────────────────────────────────────────────
    logger.info(f"  Generating H3 cells over {state_name} boundary...")

    geojson = _build_state_polygon(state_key)

    try:
        cells = _polyfill(h3, geojson, resolution)
        logger.info(f"  H3 polyfill with state polygon: {len(cells):,} cells")
    except Exception as e:
        logger.warning(f"  State polygon polyfill failed ({e}) — falling back to bounding box")
        geojson = _build_bbox_polygon(state_key)
        cells = _polyfill(h3, geojson, resolution)
        logger.info(f"  H3 polyfill with bbox: {len(cells):,} cells")

    if len(cells) == 0:
        raise RuntimeError(
            f"H3 polyfill returned 0 cells for {state_name}. "
            f"Check h3 installation and version compatibility."
        )

    # ── Step 2: Build centroid DataFrame ─────────────────────────────────────
    logger.info(f"  Computing centroids for {len(cells):,} cells...")

    rows = []
    for cell in cells:
        lat, lon = _cell_to_latlng(h3, cell)
        rows.append({
            "h3_cell":       cell,
            "centroid_lat":  round(lat, 6),
            "centroid_lon":  round(lon, 6),
        })

    grid_df = pd.DataFrame(rows)
    logger.info(f"  Grid shape: {grid_df.shape}")

    # ── Step 3: Load fire data ────────────────────────────────────────────────
    fire_df = pd.DataFrame()
    if parquet_path.exists():
        logger.info(f"  Loading fire data: {parquet_path}")
        try:
            cols_needed = ["LATITUDE", "LONGITUDE", "Ecoregion_NA_L3CODE",
                           "Ecoregion_NA_L2CODE", "Ecoregion_NA_L1CODE"]
            available = pd.read_parquet(parquet_path, columns=["LATITUDE"]).columns
            # Load only what's available
            load_cols = [c for c in cols_needed if True]  # load all, filter after
            fire_df = pd.read_parquet(parquet_path)
            logger.info(f"  Fire records loaded: {len(fire_df):,}")
        except Exception as e:
            logger.warning(f"  Could not load fire parquet: {e}")
    else:
        logger.warning(f"  Fire parquet not found: {parquet_path}")
        logger.warning("  Proceeding without fire data — ecoregion will be UNKNOWN")

    # ── Step 4: Assign ecoregions ─────────────────────────────────────────────
    if not fire_df.empty:
        logger.info("  Assigning ecoregions via KD-tree...")
        l3_vals, l2_vals = _assign_ecoregions_kdtree(
            grid_df["centroid_lat"].values,
            grid_df["centroid_lon"].values,
            fire_df,
        )
        grid_df["ecoregion_l3"] = l3_vals
        grid_df["ecoregion_l2"] = l2_vals
    else:
        grid_df["ecoregion_l3"] = -1
        grid_df["ecoregion_l2"] = -1

    # ── Step 5: Count fires per H3 cell ──────────────────────────────────────
    if not fire_df.empty:
        logger.info("  Mapping fire points to H3 cells...")
        fire_counts = _map_fires_to_h3_cells(h3, fire_df, resolution)
        grid_df["fire_count"]      = grid_df["h3_cell"].map(fire_counts).fillna(0).astype(int)
        grid_df["has_fire_history"] = grid_df["fire_count"] > 0
        n_fire_cells = grid_df["has_fire_history"].sum()
        logger.info(f"  Cells with fire history: {n_fire_cells:,} / {len(grid_df):,} "
                    f"({100*n_fire_cells/len(grid_df):.1f}%)")
    else:
        grid_df["fire_count"]      = 0
        grid_df["has_fire_history"] = False

    # ── Step 6: Add metadata columns ─────────────────────────────────────────
    grid_df["state"]         = state_key
    grid_df["h3_resolution"] = resolution
    grid_df["burnable"]      = True   # Placeholder — refined in Phase 2D with LANDFIRE

    # Reorder columns cleanly
    col_order = [
        "h3_cell", "state", "h3_resolution",
        "centroid_lat", "centroid_lon",
        "ecoregion_l3", "ecoregion_l2",
        "fire_count", "has_fire_history",
        "burnable",
    ]
    grid_df = grid_df[[c for c in col_order if c in grid_df.columns]]

    # ── Step 7: Save outputs ──────────────────────────────────────────────────
    parquet_out = output_dir / f"h3_grid_{state_key.lower()}.parquet"
    grid_df.to_parquet(parquet_out, index=False, compression="snappy")
    logger.info(f"  ✔ Saved grid parquet: {parquet_out} ({parquet_out.stat().st_size / 1e6:.1f} MB)")

    # ── Step 8: Save summary CSV ──────────────────────────────────────────────
    _save_summary(grid_df, state_key, state_name, resolution, output_dir)

    # ── Step 9: Print console report ──────────────────────────────────────────
    _print_report(grid_df, state_key, state_name, resolution, parquet_out)

    return grid_df


# ─────────────────────────────────────────────────────────────────────────────
# Summary Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_summary(
    grid_df: pd.DataFrame,
    state_key: str,
    state_name: str,
    resolution: int,
    output_dir: Path,
) -> None:
    """Save ecoregion breakdown and grid summary CSV."""
    # Ecoregion distribution
    eco_counts = (
        grid_df.groupby("ecoregion_l3")
        .agg(
            total_cells=("h3_cell", "count"),
            fire_cells=("has_fire_history", "sum"),
        )
        .reset_index()
        .sort_values("total_cells", ascending=False)
    )
    eco_counts["fire_pct"] = (
        eco_counts["fire_cells"] / eco_counts["total_cells"] * 100
    ).round(2)

    eco_out = output_dir / f"h3_ecoregion_breakdown_{state_key.lower()}.csv"
    eco_counts.to_csv(eco_out, index=False)
    logger.info(f"  ✔ Saved ecoregion breakdown: {eco_out}")

    # High-level summary CSV
    summary = {
        "State":              [state_name],
        "H3_Resolution":      [resolution],
        "Total_Cells":        [len(grid_df)],
        "Fire_History_Cells": [int(grid_df["has_fire_history"].sum())],
        "No_Fire_Cells":      [int((~grid_df["has_fire_history"]).sum())],
        "Unique_L3_Ecoregions": [grid_df["ecoregion_l3"].nunique()],
        "Lat_Min":            [round(grid_df["centroid_lat"].min(), 4)],
        "Lat_Max":            [round(grid_df["centroid_lat"].max(), 4)],
        "Lon_Min":            [round(grid_df["centroid_lon"].min(), 4)],
        "Lon_Max":            [round(grid_df["centroid_lon"].max(), 4)],
    }
    summary_df = pd.DataFrame(summary)
    summary_out = output_dir / f"h3_grid_summary_{state_key.lower()}.csv"
    summary_df.to_csv(summary_out, index=False)
    logger.info(f"  ✔ Saved summary: {summary_out}")


def _print_report(
    grid_df: pd.DataFrame,
    state_key: str,
    state_name: str,
    resolution: int,
    parquet_out: Path,
) -> None:
    """Print formatted console report."""
    fire_cells   = int(grid_df["has_fire_history"].sum())
    nofire_cells = len(grid_df) - fire_cells
    n_ecoregions = grid_df["ecoregion_l3"].nunique()

    print(f"\n{'=' * 65}")
    print(f"  PHASE 2B — H3 GRID COMPLETE [{state_name.upper()}]")
    print(f"{'=' * 65}")
    print(f"  H3 Resolution          : R{resolution}")
    print(f"  Total grid cells       : {len(grid_df):>10,}")
    print(f"  Cells with fire history: {fire_cells:>10,}  ({100*fire_cells/len(grid_df):.1f}%)")
    print(f"  Cells with no fires    : {nofire_cells:>10,}  ({100*nofire_cells/len(grid_df):.1f}%)")
    print(f"  Unique L3 ecoregions   : {n_ecoregions:>10}")
    print(f"  Lat range              : {grid_df['centroid_lat'].min():.2f} → {grid_df['centroid_lat'].max():.2f}")
    print(f"  Lon range              : {grid_df['centroid_lon'].min():.2f} → {grid_df['centroid_lon'].max():.2f}")

    if "ecoregion_l3" in grid_df.columns:
        print(f"\n  Top Ecoregions by cell count:")
        eco_top = (
            grid_df.groupby("ecoregion_l3")["h3_cell"]
            .count()
            .sort_values(ascending=False)
            .head(8)
        )
        for eco, cnt in eco_top.items():
            bar = "█" * int(cnt / eco_top.max() * 20)
            print(f"    L3-{eco:<6} {cnt:>7,} cells  {bar}")

    print(f"\n  Saved: {parquet_out.name}")
    print(f"{'=' * 65}\n")
