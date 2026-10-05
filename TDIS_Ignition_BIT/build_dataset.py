"""
Build the training dataset for the BIT-methodology retrain of TDIS's newest
(TDIS_Ignition) package.

Source of truth: TDIS_Ignition/model_tristate/data/train_tristate_v2.parquet
This is TDIS's own most-corrected, bug-fixed training table (the one where
the TX/NM/OK ecoregion-scale incommensurability bug was already fixed, the
one their own rev5/control_v2 models were trained from). We filter it to
TX-only rows, keep TDIS's own temporal split (train 2018-2020 / val 2021 /
test 2022-2026 embedded in the `split` column), and then apply our own
methodology on top of their data:

  1. Derive real lat/lon from the h3_cell index (100% coverage; TDIS's own
     lat/lon columns are 100% null for TX rows in this file).
  2. Seasonal (month-of-year) median imputation for the legacy gridMET daily
     weather columns (erc, fm100, vpd, vs, rmax, rmin, tmmx, pr), which are
     ~76% null in the raw table but carried real signal in our prior TEXAS
     models (rmin ranked #4 feature importance in TEXAS/model_32feat_tuned).
  3. Median/mode imputation for ecoregion_id (~11% null) and fm100_5d_min
     (~10% null) so no rows are dropped.

Output: TDIS_Ignition_BIT/data/bit_train_tx.parquet
"""
import pandas as pd
import numpy as np
import h3
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "TDIS_Ignition" / "model_tristate" / "data" / "train_tristate_v2.parquet"
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "bit_train_tx.parquet"

GRIDMET_COLS = ["erc", "fm100", "vpd", "vs", "rmax", "rmin", "tmmx", "pr"]

def main():
    print(f"Loading {SRC} ...")
    df = pd.read_parquet(SRC)
    print(f"Raw tri-state shape: {df.shape}")

    tx = df[df["state"] == "tx"].copy()
    print(f"TX-only rows: {len(tx):,}")

    # --- 1. Derive real lat/lon from h3_cell (TDIS's own lat/lon are 100% null here) ---
    print("Deriving lat/lon from h3_cell ...")
    latlon = tx["h3_cell"].apply(lambda c: h3.cell_to_latlng(c))
    tx["lat"] = latlon.apply(lambda t: t[0])
    tx["lon"] = latlon.apply(lambda t: t[1])

    # --- 2. Seasonal median imputation for sparse legacy gridMET columns ---
    tx["month"] = tx["date"].dt.month
    for col in GRIDMET_COLS:
        null_before = tx[col].isnull().sum()
        seasonal_median = tx.groupby("month")[col].transform("median")
        tx[col] = tx[col].fillna(seasonal_median)
        # any residual null (whole month missing) -> global median
        tx[col] = tx[col].fillna(tx[col].median())
        print(f"  {col}: imputed {null_before:,} nulls ({null_before/len(tx):.2%}) via seasonal median")

    # --- 3. Remaining sparse columns: ecoregion_id (mode), fm100_5d_min (median) ---
    eco_null = tx["ecoregion_id"].isnull().sum()
    tx["ecoregion_id"] = tx["ecoregion_id"].fillna(tx["ecoregion_id"].mode().iloc[0])
    print(f"  ecoregion_id: imputed {eco_null:,} nulls ({eco_null/len(tx):.2%}) via mode")

    fm_null = tx["fm100_5d_min"].isnull().sum()
    tx["fm100_5d_min"] = tx["fm100_5d_min"].fillna(tx["fm100_5d_min"].median())
    print(f"  fm100_5d_min: imputed {fm_null:,} nulls ({fm_null/len(tx):.2%}) via median")

    # Drop unusable 100%-null/identifier columns from the source table
    tx = tx.drop(columns=["fips"])  # 100% null for TX in source table, unused
    tx = tx.drop(columns=["month"])  # helper column only

    final_null = tx.isnull().sum()
    assert final_null.sum() == 0, f"Unexpected remaining nulls:\n{final_null[final_null>0]}"

    # --- 4. Truncate at the last real observed label (2026-07-29) ---
    # Our own prior audit (DATA_QUALITY_REVIEW.md) flagged exactly this class of
    # defect in the original handoff: a residual of 1,042 rows dated 2026-07-30/31
    # (all label=0, no event could yet be observed) sits inside the `test` split
    # here too, at far smaller scale than the original 86,072-row defect but the
    # same root cause. We apply our own recommended fix (truncate, don't silently keep).
    LAST_LABEL = pd.Timestamp("2026-07-29")
    before = len(tx)
    phantom = tx[tx["date"] > LAST_LABEL]
    tx = tx[tx["date"] <= LAST_LABEL].copy()
    print(f"\nTruncated {before - len(tx):,} phantom future-dated rows (> {LAST_LABEL.date()}), "
          f"all label={phantom['label'].unique().tolist()}")

    print(f"\nFinal shape: {tx.shape}")
    print(f"Label positive rate: {tx['label'].mean():.4f}")
    print("Split counts:\n", tx["split"].value_counts())
    print("Split date ranges:")
    print(tx.groupby("split")["date"].agg(["min", "max"]))

    tx.to_parquet(OUT, index=False)
    print(f"\nSaved -> {OUT}")

if __name__ == "__main__":
    main()
