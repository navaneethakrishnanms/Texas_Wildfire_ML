# Missing Data Diagnostic — Texas Final Dataset

## Summary of Problems Found

| Problem | Columns Affected | Missing Rate | Severity |
|---------|-----------------|-------------|----------|
| 5-day rolling computed on sparse rows | `erc_5D_mean`, `fm100_5D_mean`, etc (14 cols) | **84.4%** | 🔴 Critical |
| gridMET fill value 32767 leaking through | `erc`, `fm100`, `bi`, `vpd`, `vs`, `rmax`, `rmin`, `pr`, `sph`, `tmmx` | Values hitting 32767 max | 🔴 Critical |
| tmmx NaN | `tmmx` only | 11.6% | 🟡 Medium |
| LANDFIRE rasters not downloaded | `avg_burn_prob`, `whp`, `flep4`, `cfl` | 99.6% zero | 🔴 Critical |
| Fire rows not matching H3 grid | static cols | 4.7% of fire rows | 🟢 Minor |

---

## Problem 1 — 84.4% Missing in 5-Day Rolling Stats

### Root Cause
In `run_phase2f_gridmet.py`, the 5-day rolling is computed like this:

```python
gridmet_df.groupby("h3_cell")[var].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
```

This groups by `h3_cell` and rolls over whatever rows that cell has in the training table.
**But each non-fire cell only appears 1–2 times total** (it was sampled for one specific date).
With only 1 row per cell, `shift(1)` gives NaN for that row's value,
and `rolling(5)` of a single NaN = NaN.

### Fix Required
For each `(h3_cell, date)` in training, we need to fetch gridMET values for the
**5 preceding calendar days** directly from the NetCDF files and compute mean/max from those.
Cannot be computed from the training table alone.

---

## Problem 2 — gridMET Values Hitting 32767 (Fill Value Contamination)

### Root Cause
gridMET NetCDF files store data as **int16** packed with `scale_factor` and `add_offset`.
The fill value for missing data is `32767` (max int16).
The extraction code reads `scale_factor` and `add_offset`, but the fill value check
happens AFTER multiplication — by then `32767 × scale_factor` is a different number,
but the code checks for original `_FillValue` which no longer matches.

Result: nodata cells (ocean, outside CONUS) get their fill value `32767` kept as-is.

### Fix Required
Mask `data == _FillValue` BEFORE applying `scale_factor` and `add_offset`.

---

## Problem 3 — LANDFIRE Rasters (avg_burn_prob, whp, flep4, cfl all zeros)

All 4 raster files are missing from `V2/data/rasters/`.
The extractor falls back to `nodata_fill=0.0` for every cell.

### Fix Required
Download the 4 rasters, then re-run Phase 2E + 2G.

| Raster | Download URL |
|--------|-------------|
| `BP_national.tif` | https://www.firelab.org/sites/default/files/images/attachments/BP_national.zip |
| `WHP_2023.tif` | https://www.firelab.org/sites/default/files/images/attachments/WHP_2023.zip |
| `FLEP4_national.tif` | https://landfire.gov/viewer/ → FLEP4 |
| `CFL_national.tif` | https://landfire.gov/viewer/ → CFL |

---

## Fix Plan (Execution Order)

```
Step 1 — Fix Phase 2F:
  - Mask fill values BEFORE scale_factor multiplication
  - Recompute 5-day rolling by fetching 5 preceding days from NC files

Step 2 — Re-run Phase 2F:
  python run_phase2f_gridmet.py --state TX

Step 3 — Download LANDFIRE rasters
  Place in V2/data/rasters/

Step 4 — Re-run Phase 2E:
  python run_phase2e_static.py --state TX

Step 5 — Re-run Phase 2G:
  python run_phase2g_assemble.py --state TX
```
