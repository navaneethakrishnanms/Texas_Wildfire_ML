"""
deep_diagnose.py
-----------------
Deep dive into each missing value cause.
"""
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("outputs/texas")
final = pd.read_parquet(OUT / "final_training_dataset_tx.parquet")
labels = pd.read_parquet(OUT / "full_training_labels.parquet")
gridmet = pd.read_parquet(OUT / "gridmet_features_tx.parquet")

print("=" * 70)
print("ISSUE 1 — 5-day rolling stats (erc_5D_mean etc): WHY 84.4% NaN?")
print("=" * 70)

# How many times does each h3_cell appear in gridmet output?
cell_counts = gridmet.groupby("h3_cell").size()
print(f"\nGridMET rows per unique h3_cell:")
print(f"  Total unique cells:   {len(cell_counts):,}")
print(f"  Cells with 1 date:    {(cell_counts == 1).sum():,}  ({100*(cell_counts==1).mean():.1f}%)")
print(f"  Cells with 2-5 dates: {((cell_counts > 1) & (cell_counts <= 5)).sum():,}  ({100*((cell_counts>1)&(cell_counts<=5)).mean():.1f}%)")
print(f"  Cells with >5 dates:  {(cell_counts > 5).sum():,}  ({100*(cell_counts>5).mean():.1f}%)")
print(f"  Max dates per cell:   {cell_counts.max()}")
print(f"  Mean dates per cell:  {cell_counts.mean():.2f}")
print(f"\n  --> Rolling window needs >=2 rows per cell to produce ANY non-NaN result.")
print(f"      If {100*(cell_counts==1).mean():.1f}% of cells only appear ONCE, rolling = NaN for them.")

print("\n" + "=" * 70)
print("ISSUE 2 — gridMET values hitting 32767 (fill value contamination)")
print("=" * 70)

for col in ['erc', 'fm100', 'bi', 'vpd', 'vs', 'rmax', 'rmin', 'pr', 'sph']:
    if col in gridmet.columns:
        vals = gridmet[col]
        n_32767 = (vals >= 32000).sum()
        pct = 100 * n_32767 / len(vals)
        print(f"  {col:<10} values >= 32000: {n_32767:>8,}  ({pct:.2f}%)  max={vals.max():.1f}")

print("\n  --> 32767 is the NetCDF int16 fill value (used for ocean / outside-CONUS cells).")
print("      Means: some H3 cells fall just outside the gridMET 4km CONUS boundary.")
print("      These cells got fill value instead of real weather data.")

print("\n" + "=" * 70)
print("ISSUE 3 — tmmx: 11.6% NaN (43,670 rows)")
print("=" * 70)

if 'tmmx' in final.columns:
    tmmx_null = final[final['tmmx'].isna()]
    if len(tmmx_null) > 0:
        year_counts = tmmx_null.groupby('fire_year').size()
        print(f"\n  NaN distribution by year:")
        for yr, cnt in year_counts.items():
            print(f"    {yr}: {cnt:,} NaN rows")
        print(f"\n  --> Likely cause: tmmx_2019.nc file may be incomplete (116 MB vs ~147 MB expected).")
        print(f"      Or: tmmx has a different variable name inside the NC file for some years.")

print("\n" + "=" * 70)
print("ISSUE 4 — LANDFIRE rasters zero (avg_burn_prob, whp, flep4, cfl)")
print("=" * 70)
RASTER_DIR = Path("../../data/rasters")
for name, fname in [("avg_burn_prob","BP_national.tif"), ("whp","WHP_2023.tif"),
                    ("flep4","FLEP4_national.tif"), ("cfl","CFL_national.tif")]:
    path = RASTER_DIR / fname
    if path.exists():
        mb = path.stat().st_size / 1e6
        print(f"  {name:<15} ✅ EXISTS ({mb:.0f} MB)")
    else:
        print(f"  {name:<15} ❌ MISSING — {path}")

print("\n  --> Rasters not downloaded. Every cell gets nodata_fill=0.0.")
print("      These are your 4 strongest predictors — model will be crippled without them.")

print("\n" + "=" * 70)
print("ISSUE 5 — Static join miss: 1,594 fire rows have NaN in static cols")
print("=" * 70)
miss = final[final['h3_resolution'].isna() & (final['label'] == 1)]
print(f"\n  {len(miss):,} fire rows couldn't join to static_features_tx.parquet")
print(f"  Likely: fire cells outside the Texas H3 grid boundary (border fires, lat/lon snapping)")
print(f"  This is 4.7% of fire rows = {len(miss):,} / {(final.label==1).sum():,}")

print("\n" + "=" * 70)
print("SUMMARY: What to do for each issue")
print("=" * 70)
print("""
Issue 1 (5D rolling — 84.4% NaN)
  CAUSE:  Code bug — rolling computed on sparse training rows, not consecutive daily data
  FIX:    Rewrite Phase 2F to fetch 5 preceding days from NC files for each (cell,date)
  ACTION: Run fixed run_phase2f_gridmet.py --state TX  (re-running current code won't fix it)

Issue 2 (32767 fill value contamination)
  CAUSE:  Code bug — fill value not masked BEFORE scale_factor multiplication
  FIX:    Mask data[data == _FillValue] = NaN BEFORE scale_factor step in extractor
  ACTION: Run fixed run_phase2f_gridmet.py --state TX  (needs code fix first)

Issue 3 (tmmx 11.6% NaN)
  CAUSE:  tmmx_2019.nc possibly incomplete download (116 MB vs ~147 MB for other years)
  FIX:    Delete tmmx_2019.nc and re-download it; OR zero-fill NaN for tmmx
  ACTION: Re-download tmmx_2019.nc, then re-run Phase 2F

Issue 4 (LANDFIRE rasters all zeros)
  CAUSE:  Data not downloaded — .tif files missing from V2/data/rasters/
  FIX:    Download 4 rasters, re-run Phase 2E --state TX, then Phase 2G --state TX
  ACTION: Cannot fix by re-running code alone — must download files first

Issue 5 (1,594 fire rows static NaN)
  CAUSE:  Border fire cells not in H3 grid (minor — 4.7% of fires)
  FIX:    Zero-fill or median-fill these specific rows
  ACTION: Optional — acceptable to drop or fill
""")
