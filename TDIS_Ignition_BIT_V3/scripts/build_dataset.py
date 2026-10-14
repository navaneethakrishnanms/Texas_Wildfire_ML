"""
TDIS_Ignition_BIT_V3 -- HRRR-only AND no-raw-coordinates dataset build.

Combines both corrections received:
  1. (Haripriya, internal feedback, 2026-10-14): "Use HRRR data only...
     Gridmet is the same day data and we are focused on a forecasting
     model. HRRR is forecast data so please just have one weather source."
     gridMET (erc, fm100, vpd, vs, rmax, rmin, tmmx, pr, and the derived
     fm100_5d_min) are all SAME-DAY OBSERVED weather products -- not
     knowable at real forecast-issuance time. Only
     hrrr_tmp/hrrr_vpd/hrrr_wind/hrrr_mstav are true forecasts.
  2. (Texas team feedback, relayed 2026-10-14): remove lat/lon -- do not
     proceed with raw coordinates as model features.

`bp_x_vpd` and `whp_x_vpd` are VERIFIED (by direct correlation check,
r=0.9999999) to already be built from hrrr_vpd, not gridmet vpd -- kept.

Source: TDIS_Ignition_BIT/data/bit_train_tx.parquet (v1's cleaned,
imputed, phantom-row-truncated 38-feature dataset).

Output: 27 features. (This replaces the earlier 29-feature V3 build, which
kept lat/lon -- that version is superseded now that the Texas team has
weighed in against raw coordinates.)
"""
import pandas as pd
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent.parent / "TDIS_Ignition_BIT" / "data" / "bit_train_tx.parquet"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# gridMET-sourced columns to drop (same-day OBSERVED weather, not a forecast)
DROP_GRIDMET = ["erc", "fm100", "vpd", "vs", "rmax", "rmin", "tmmx", "pr", "fm100_5d_min"]
# raw coordinates -- Texas team feedback: do not use
DROP_COORDS = ["lat", "lon"]
DROP_COLS = DROP_GRIDMET + DROP_COORDS

def main():
    print(f"Loading {SRC} ...")
    df = pd.read_parquet(SRC)
    print(f"Input shape: {df.shape}")

    assert all(c in df.columns for c in DROP_COLS), "expected columns not found"
    df27 = df.drop(columns=DROP_COLS)
    print(f"Dropped {DROP_COLS} -> shape: {df27.shape}")

    null_check = df27.isnull().sum().sum()
    assert null_check == 0, f"Unexpected nulls after drop: {null_check}"

    meta_cols = ["h3_cell", "date", "label", "year", "split", "state"]
    feature_cols = [c for c in df27.columns if c not in meta_cols]
    print(f"\nFeature count: {len(feature_cols)}")
    print("Features:", feature_cols)

    print("\nSplit counts:\n", df27["split"].value_counts())
    print(f"Label positive rate: {df27['label'].mean():.4f}")

    parquet_path = OUT_DIR / "bit_v3_train_tx.parquet"
    csv_path = OUT_DIR / "bit_v3_train_tx.csv"
    print(f"\nWriting parquet -> {parquet_path}")
    df27.to_parquet(parquet_path, index=False)
    print(f"Writing csv -> {csv_path}")
    df27.to_csv(csv_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
