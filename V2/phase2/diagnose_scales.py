"""
diagnose_scales.py
-------------------
Quick diagnostic: check what the raw and decoded values are in gridMET NC files
for tmmx, vpd, rmax to understand the scale issue.
Run from: V2/phase2/
"""
import netCDF4 as nc
import numpy as np
from pathlib import Path

GRIDMET_DIR = Path(r"..\data\gridmet")

def check_var(var, year=2015):
    fp = GRIDMET_DIR / f"{var}_{year}.nc"
    if not fp.exists():
        print(f"  NOT FOUND: {fp}")
        return

    with nc.Dataset(fp) as ds:
        # Find variable name
        vname = None
        for v in ds.variables:
            if v not in ("day","lat","lon","crs"):
                vname = v
                break

        var_obj = ds.variables[vname]

        # Mirror what read_band does: disable auto-decode first
        var_obj.set_auto_maskandscale(False)
        raw_slice = np.array(var_obj[180, :, :], dtype=np.float64).ravel()  # day 180 ~ July

        # Attributes
        fill   = getattr(var_obj, "_FillValue", None)
        scale  = getattr(var_obj, "scale_factor", None)
        offset = getattr(var_obj, "add_offset", None)
        units  = getattr(var_obj, "units", "?")
        valid_min = getattr(var_obj, "valid_min", None)
        valid_max = getattr(var_obj, "valid_max", None)

        print(f"\n{'='*55}")
        print(f"  Variable : {var} ({vname})")
        print(f"  NC file  : {fp.name}")
        print(f"  units    : {units}")
        print(f"  _FillValue  : {fill}")
        print(f"  scale_factor: {scale}")
        print(f"  add_offset  : {offset}")
        print(f"  valid_min   : {valid_min}")
        print(f"  valid_max   : {valid_max}")
        print(f"\n  RAW (before decode):")
        masked = raw_slice[raw_slice != fill] if fill else raw_slice
        masked = masked[~np.isnan(masked)]
        print(f"    min={masked.min():.2f}  max={masked.max():.2f}  mean={masked.mean():.2f}")

        # After decode
        decoded = raw_slice.copy()
        if fill is not None:
            decoded[decoded == float(fill)] = np.nan
        if scale  is not None: decoded *= float(scale)
        if offset is not None: decoded += float(offset)
        valid_dec = decoded[~np.isnan(decoded)]
        print(f"\n  DECODED (scale + offset applied):")
        print(f"    min={valid_dec.min():.4f}  max={valid_dec.max():.4f}  mean={valid_dec.mean():.4f}")

        if var in ("tmmx","tmmn"):
            if valid_dec.mean() > 200:
                celcius = valid_dec - 273.15
                print(f"  → Values in Kelvin! After -273.15: mean={celcius.mean():.2f}°C  "
                      f"min={celcius.min():.2f}  max={celcius.max():.2f}")
            else:
                print(f"  → Already in °C range: mean={valid_dec.mean():.2f}°C")

check_var("tmmx")
check_var("vpd")
check_var("rmax")
check_var("rmin")
check_var("erc")
check_var("fm100")
