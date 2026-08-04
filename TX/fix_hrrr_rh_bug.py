"""
fix_hrrr_rh_bug.py
-------------------
Fixes a data-quality bug in hrrr_tx_all.parquet.

Root cause (fetch_hrrr_tx.py RH_FALLBACKS):
  Some 2014-2016 HRRR surface files don't have a ':RH:2 m above ground'
  field. The fallback chain tried ':SPFH:2 m above ground' (specific
  humidity, units kg/kg, magnitude ~0.001-0.02) and stored it directly
  into rh_pw as if it were a 0-100% relative humidity value.

  compute_vpd() then used that near-zero "RH" to compute vpd_pw_hrrr,
  producing a value that looks like a normal VPD reading (0.9-2.1 kPa)
  but is actually just saturation vapor pressure with no real humidity
  correction — a silent corruption, not a missing-data gap.

Detection: true RH% in this dataset is always > 1.0 (min observed
51.3%). Any rh_pw < 1.0 is specific humidity mislabeled as RH.
Affected: 100,981 rows (33% of matched HRRR rows), all in 2014-2016.

Fix: null out rh_pw and vpd_pw_hrrr for affected rows, add a
hrrr_rh_valid flag so the model (and anything else reading this file)
knows RH/VPD-from-HRRR aren't usable for those rows. temp_pw, wind_pw,
hpbl_pw, dswrf_pw are untouched — they come from different GRIB fields
(TMP/UGRD/VGRD/HPBL/DSWRF) and aren't affected by this bug.

Usage:
  python fix_hrrr_rh_bug.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PQ = ROOT / "data" / "hrrr" / "hrrr_tx_all.parquet"

RH_MIN_VALID = 1.0   # true RH% always > 1.0 in this dataset; below this = SPFH mislabeled as RH


def main():
    df = pd.read_parquet(PQ)

    bad = df["rh_pw"].notna() & (df["rh_pw"] < RH_MIN_VALID)
    n_bad = int(bad.sum())
    print(f"Total rows:            {len(df):,}")
    print(f"Rows with HRRR RH:     {df['rh_pw'].notna().sum():,}")
    print(f"Corrupted (SPFH->RH):  {n_bad:,}")
    print(f"By year:")
    print(df.loc[bad, "date_utc"].dt.year.value_counts().sort_index())

    df["hrrr_rh_valid"] = (~bad & df["rh_pw"].notna()).astype(np.int8)
    df.loc[bad, "rh_pw"] = np.nan
    df.loc[bad, "vpd_pw_hrrr"] = np.nan

    df.to_parquet(PQ, index=False, compression="snappy")
    print(f"\nFixed and saved: {PQ}")
    print(f"hrrr_rh_valid distribution:\n{df['hrrr_rh_valid'].value_counts()}")


if __name__ == "__main__":
    main()
