"""
reorganize_data.py
==================
Organizes all raw GEE-exported rasters into the correct data/raw/ subfolder layout.

What it does:
  1. Creates all required subdirectory folders
  2. Moves NDVI, EVI, LST if still in root
  3. Merges GEE Slope tile exports into data/raw/slope/Texas_Slope.tif
  4. Merges GEE Aspect tile exports into data/raw/aspect/Texas_Aspect.tif
  5. Moves LandCover_2024.tif to data/raw/landcover/
  6. Verifies DEM is already in data/raw/dem/
  7. Creates a placeholder FIRMS CSV if not present (replace with real one from NASA FIRMS)
  8. Prints a checklist of what still needs to be downloaded from GEE
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
import rasterio.windows

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
ROOT = Path(".")
RAW  = ROOT / "data" / "raw"

SUBDIRS = [
    "ndvi", "evi", "lst", "temperature", "wind",
    "rainfall", "dem", "slope", "aspect", "landcover", "firms"
]

# Files to move from root to subfolder
SIMPLE_MOVES = {
    "Texas_NDVI_2024.tif":      "ndvi/Texas_NDVI_2024.tif",
    "Texas_EVI_2024.tif":       "evi/Texas_EVI_2024.tif",
    "Texas_LST_2024.tif":       "lst/Texas_LST_2024.tif",
    "Texas_LandCover_2024.tif": "landcover/Texas_LandCover.tif",
    "Texas_Rainfall_2024.tif":      "rainfall/Texas_Rainfall_2024.tif",
    "Texas_Temperature_2024.tif": "temperature/Texas_Temperature_2024.tif",
    "Texas_Wind_2024.tif":        "wind/Texas_Wind_2024.tif",
}

# Tile groups to merge (prefix -> output path)
# Kept commented out for now to keep DEM/Slope/Aspect files exactly as they are.
TILE_MERGES = {
    # "Texas_Slope":  "slope/Texas_Slope.tif",
    # "Texas_Aspect": "aspect/Texas_Aspect.tif",
}

# -------------------------------------------------
# STEP 1 - Create folders
# -------------------------------------------------
def create_folders():
    for d in SUBDIRS:
        (RAW / d).mkdir(parents=True, exist_ok=True)
    print("[OK] Subdirectory structure created.")


# -------------------------------------------------
# STEP 2 - Move single-file rasters
# -------------------------------------------------
def move_single_files():
    for src_name, dst_rel in SIMPLE_MOVES.items():
        src = ROOT / src_name
        dst = RAW / dst_rel
        if dst.exists():
            print(f"  [OK] Already in place: {dst}")
        elif src.exists():
            shutil.move(str(src), str(dst))
            print(f"  [MOVED] {src_name} -> {dst}")
        else:
            print(f"  [MISSING] {src_name}  (download from GEE)")


# -------------------------------------------------
# STEP 3 - Merge GEE tile exports (streaming, low-memory)
# -------------------------------------------------
STRIP_ROWS = 256   # read/write this many rows at a time (~32 MB per strip)

def merge_tiles(prefix: str, out_rel: str):
    """
    Merge GEE tile exports into one GeoTIFF using a row-strip streaming
    approach.  Peak memory usage = STRIP_ROWS * tile_width * 4 bytes.
    """
    out_path = RAW / out_rel
    if out_path.exists():
        print(f"  [OK] Already merged: {out_path}")
        return

    tiles = sorted(ROOT.glob(f"{prefix}-*.tif"))
    if not tiles:
        print(f"  [MISSING] No tiles found for prefix '{prefix}'  (download from GEE)")
        return

    print(f"  [MERGING] {len(tiles)} tiles for {prefix} (streaming) ...")
    opened = [rasterio.open(t) for t in tiles]

    # Compute full mosaic extent from individual tile bounds
    left   = min(s.bounds.left   for s in opened)
    bottom = min(s.bounds.bottom for s in opened)
    right  = max(s.bounds.right  for s in opened)
    top    = max(s.bounds.top    for s in opened)

    res_x = opened[0].transform.a          # pixel width  (degrees)
    res_y = abs(opened[0].transform.e)     # pixel height (degrees)

    total_width  = int(round((right - left)  / res_x))
    total_height = int(round((top   - bottom) / res_y))

    out_transform = rasterio.transform.from_origin(left, top, res_x, res_y)
    out_meta = opened[0].meta.copy()
    out_meta.update({
        "driver":    "GTiff",
        "height":    total_height,
        "width":     total_width,
        "transform": out_transform,
        "compress":  "lzw",
        "tiled":     True,
        "blockxsize": 512,
        "blockysize": 512,
    })

    with rasterio.open(out_path, "w", **out_meta) as dst:
        for src in opened:
            # Pixel offset of this tile inside the full mosaic
            col_off = int(round((src.bounds.left - left) / res_x))
            row_off = int(round((top - src.bounds.top)   / res_y))

            # Stream in strips of STRIP_ROWS rows
            for r_start in range(0, src.height, STRIP_ROWS):
                rows = min(STRIP_ROWS, src.height - r_start)
                win_src = rasterio.windows.Window(0, r_start, src.width, rows)
                win_dst = rasterio.windows.Window(col_off, row_off + r_start,
                                                  src.width, rows)
                data = src.read(window=win_src)
                dst.write(data, window=win_dst)

            print(f"    [WRITTEN] {src.name.split(os.sep)[-1]}")

    for s in opened:
        s.close()

    # Delete source tiles to free disk space
    for t in tiles:
        t.unlink()
        print(f"    [DELETED] tile: {t.name}")

    print(f"  [OK] Merged -> {out_path}")


# -------------------------------------------------
# STEP 4 - Generate placeholder FIRMS CSV
# -------------------------------------------------
def create_firms_placeholder():
    firms_path = RAW / "firms" / "Texas_FIRMS_2024.csv"
    if firms_path.exists():
        print(f"  [OK] FIRMS CSV already exists: {firms_path}")
        return

    print("  [GENERATING] Placeholder FIRMS CSV ...")
    np.random.seed(42)
    n = 300
    lats = np.random.uniform(25.84, 36.50, n)
    lons = np.random.uniform(-106.65, -93.51, n)
    dates = pd.date_range("2024-01-01", "2024-12-31", periods=n)

    df = pd.DataFrame({
        "latitude":   lats,
        "longitude":  lons,
        "acq_date":   [d.strftime("%Y-%m-%d") for d in dates],
        "acq_time":   ["1830"] * n,
        "confidence": np.random.randint(40, 100, n),
        "frp":        np.round(np.random.exponential(15.0, n) + 5.0, 2),
        "satellite":  ["VIIRS"] * n,
    })
    df.to_csv(firms_path, index=False)
    print(f"  [OK] Placeholder FIRMS CSV -> {firms_path}")
    print("  [WARN] Replace with real data from: https://firms.modaps.eosdis.nasa.gov/")


# -------------------------------------------------
# STEP 5 - Checklist
# -------------------------------------------------
def print_checklist():
    expected = {
        "NDVI":        RAW / "ndvi"        / "Texas_NDVI_2024.tif",
        "EVI":         RAW / "evi"         / "Texas_EVI_2024.tif",
        "LST":         RAW / "lst"         / "Texas_LST_2024.tif",
        "DEM":         RAW / "dem"         / "Texas_DEM.tif",
        "Slope":       RAW / "slope"       / "Texas_Slope.tif",
        "Aspect":      RAW / "aspect"      / "Texas_Aspect.tif",
        "LandCover":   RAW / "landcover"   / "Texas_LandCover.tif",
        "Temperature": RAW / "temperature" / "Texas_Temperature_2024.tif",
        "Wind":        RAW / "wind"        / "Texas_Wind_2024.tif",
        "Rainfall":    RAW / "rainfall"    / "Texas_Rainfall_2024.tif",
        "FIRMS":       RAW / "firms"       / "Texas_FIRMS_2024.csv",
    }
    print("\n" + "="*55)
    print("  DATA CHECKLIST")
    print("="*55)
    all_ok = True
    for name, path in expected.items():
        status = "[OK]     " if path.exists() else "[MISSING]"
        if not path.exists():
            all_ok = False
        print(f"  {status}  {name:15s}  {path}")
    print("="*55)
    if not all_ok:
        print("\n  ⚠️  Download missing GEE layers and place in correct folders.")
    else:
        print("\n  🚀  All data present! Run: python src/dataset_builder/build_dataset.py")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("="*55)
    print("  TEXAS WILDFIRE — DATA REORGANIZATION")
    print("="*55)

    create_folders()
    move_single_files()

    for prefix, out_rel in TILE_MERGES.items():
        merge_tiles(prefix, out_rel)

    create_firms_placeholder()
    print_checklist()
    print("\n✅  REORGANIZATION COMPLETE")

if __name__ == "__main__":
    main()
