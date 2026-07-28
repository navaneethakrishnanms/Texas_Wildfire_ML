"""
diagnose_hrrr.py
-----------------
Quick test to find out WHY 2015 HRRR files are failing.
Shows the actual error message instead of silently swallowing it.
Run: python diagnose_hrrr.py
"""
import traceback
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

print("Testing HRRR downloads for different years...")
print("=" * 60)

# Test dates — one from each problem year and one known-good year
test_cases = [
    ("2015-03-15 12:00", "2015 (should work)"),
    ("2016-06-20 18:00", "2016 (should work)"),
    ("2017-04-10 12:00", "2017 (should work)"),
    ("2018-07-04 12:00", "2018 (known good)"),
]

# Small set of TX centroids for testing
test_lats = np.array([30.2, 31.5, 29.7, 32.8, 33.1])
test_lons = np.array([-97.5, -98.2, -95.3, -96.7, -100.1])

HRRR_VARS = [
    (":TMP:2 m above ground",   "temp"),
    (":RH:2 m above ground",    "rh"),
    (":UGRD:10 m above ground", "u"),
    (":VGRD:10 m above ground", "v"),
    (":HPBL:surface",           "hpbl"),
    (":DSWRF:surface",          "dswrf"),
]

from herbie import Herbie

for dt_str, label in test_cases:
    print(f"\n--- {dt_str} ({label}) ---")
    try:
        H = Herbie(
            dt_str,
            model="hrrr",
            product="sfc",
            fxx=0,
            verbose=False,
            priority=["aws"],
        )
        print(f"  Herbie object created OK")
        print(f"  GRIB path: {H.grib}")

        # Try each variable individually
        for search_str, name in HRRR_VARS:
            try:
                ds = H.xarray(search_str, remove_grib=True)
                var_name = list(ds.data_vars)[0]
                vals = ds[var_name].values
                lats = ds["latitude"].values
                lons = ds["longitude"].values
                tree = cKDTree(np.column_stack([lats.ravel(), lons.ravel()]))
                _, idx = tree.query(np.column_stack([test_lats, test_lons]), k=1)
                sample = vals.ravel()[idx]
                print(f"    {name:<8} OK  shape={vals.shape}  sample={sample[0]:.2f}")
            except Exception as e:
                print(f"    {name:<8} FAIL: {e}")

    except Exception as e:
        print(f"  FAILED to create Herbie object:")
        print(f"  Error: {e}")
        traceback.print_exc()

print("\n" + "=" * 60)
print("Diagnosis complete. Share this output to debug the issue.")
