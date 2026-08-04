"""
export_csv.py
--------------
Convert data/hrrr/hrrr_tx_all.parquet to CSV for viewing in Excel/Sheets.

CSV is ~5-10x larger than the snappy-compressed parquet and slower to
load — use it for manual inspection only, not as a pipeline input.

Usage:
  python export_csv.py
  python export_csv.py --input data/hrrr/hrrr_tx_all.parquet --output data/hrrr/hrrr_tx_all.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, default="data/hrrr/hrrr_tx_all.parquet")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    in_path = ROOT / args.input
    out_path = ROOT / args.output if args.output else in_path.with_suffix(".csv")

    if not in_path.exists():
        raise SystemExit(f"Not found: {in_path}")

    print(f"Reading {in_path} ({in_path.stat().st_size/1e6:.1f} MB)...")
    df = pd.read_parquet(in_path)
    print(f"  {len(df):,} rows x {len(df.columns)} columns")

    print(f"Writing {out_path}...")
    df.to_csv(out_path, index=False)
    print(f"Done: {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
