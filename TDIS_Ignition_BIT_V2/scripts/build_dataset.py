"""
TDIS_Ignition_BIT_V2 — dataset build (36-feature variant).

Starts from the already-cleaned, already-imputed, already-phantom-truncated
BIT v1 dataset (TDIS_Ignition_BIT/data/bit_train_tx.parquet), and removes the
2 location features (`lat`, `lon`) to test whether the model's skill depends
on them or holds up with only the other 36 features.

No re-imputation needed here -- v1's build_dataset.py already did all of
that (seasonal-median gridMET imputation, ecoregion_id mode-fill,
fm100_5d_min median-fill, phantom future-row truncation). This script's only
job is the feature-set change, and writing the result out in both parquet
and csv so the folder is self-contained for downstream deployment/testing
work.

Output:
  data/processed/bit_v2_train_tx.parquet
  data/processed/bit_v2_train_tx.csv
  data/raw/bit_v2_test_sample.csv   (small human-readable sample for quick inspection)
"""
import pandas as pd
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent.parent / "TDIS_Ignition_BIT" / "data" / "bit_train_tx.parquet"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLS = ["lat", "lon"]

def main():
    print(f"Loading {SRC} ...")
    df = pd.read_parquet(SRC)
    print(f"Input shape: {df.shape}")

    assert all(c in df.columns for c in DROP_COLS), "lat/lon not found in source file"
    df36 = df.drop(columns=DROP_COLS)
    print(f"Dropped {DROP_COLS} -> shape: {df36.shape}")

    null_check = df36.isnull().sum().sum()
    assert null_check == 0, f"Unexpected nulls after drop: {null_check}"

    print("Split counts:\n", df36["split"].value_counts())
    print(f"Label positive rate: {df36['label'].mean():.4f}")

    parquet_path = OUT_DIR / "bit_v2_train_tx.parquet"
    csv_path = OUT_DIR / "bit_v2_train_tx.csv"

    print(f"\nWriting parquet -> {parquet_path}")
    df36.to_parquet(parquet_path, index=False)

    print(f"Writing csv -> {csv_path} (this is a ~2.2M-row file, will take a minute)")
    df36.to_csv(csv_path, index=False)

    # small human-readable sample for quick inspection without loading the full file
    sample_path = RAW_DIR / "bit_v2_test_sample.csv"
    df36[df36.split == "test"].sample(2000, random_state=42).to_csv(sample_path, index=False)
    print(f"Wrote 2,000-row inspection sample -> {sample_path}")

    print(f"\nFinal feature count (excl. h3_cell/date/label/year/split/state): "
          f"{len(df36.columns) - 6}")
    print("Final columns:", list(df36.columns))

if __name__ == "__main__":
    main()
