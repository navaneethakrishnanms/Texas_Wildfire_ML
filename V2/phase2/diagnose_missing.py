"""
diagnose_missing.py
--------------------
Checks exactly which columns have missing values, how many, and why.
"""
import pandas as pd
from pathlib import Path

OUT = Path("outputs/texas")

final = pd.read_parquet(OUT / "final_training_dataset_tx.parquet")

print(f"\nShape: {final.shape}  ({len(final):,} rows x {len(final.columns)} cols)")
print(f"\nAll columns: {list(final.columns)}")

print("\n" + "="*70)
print("MISSING VALUE REPORT (per column)")
print("="*70)
print(f"{'Column':<35} {'Missing':>10} {'% Missing':>12} {'dtype':>10}")
print("-"*70)

for col in final.columns:
    n_miss = final[col].isna().sum()
    pct = 100 * n_miss / len(final)
    print(f"  {col:<33} {n_miss:>10,} {pct:>11.1f}%  {str(final[col].dtype):>10}")

# Summarise by group
print("\n" + "="*70)
print("SUMMARY: columns with ANY missing values")
print("="*70)
missing_cols = [(c, final[c].isna().sum()) for c in final.columns if final[c].isna().any()]
if missing_cols:
    for c, n in missing_cols:
        print(f"  {c:<35} {n:,} missing ({100*n/len(final):.1f}%)")
else:
    print("  None — all columns fully populated!")

# Check fire vs non-fire missing rates separately
print("\n" + "="*70)
print("MISSING RATES: Fire (label=1) vs Non-fire (label=0)")
print("="*70)
fire  = final[final.label == 1]
nofire = final[final.label == 0]
for col in missing_cols[:10]:  # top 10 missing
    c = col[0]
    f_miss = 100 * fire[c].isna().mean()
    n_miss = 100 * nofire[c].isna().mean()
    print(f"  {c:<35} fire={f_miss:.1f}%  non-fire={n_miss:.1f}%")

# Check the static feature columns specifically
print("\n" + "="*70)
print("STATIC FEATURES (LANDFIRE rasters) — are they all zero?")
print("="*70)
for col in ['avg_burn_prob', 'whp', 'flep4', 'cfl']:
    if col in final.columns:
        vals = final[col]
        print(f"  {col:<20} min={vals.min():.4f}  max={vals.max():.4f}  "
              f"mean={vals.mean():.4f}  zeros={100*(vals==0).mean():.1f}%  "
              f"NaN={vals.isna().sum():,}")
    else:
        print(f"  {col:<20} *** COLUMN NOT PRESENT ***")

# Check gridMET columns
print("\n" + "="*70)
print("GRIDMET FEATURES — value ranges")
print("="*70)
gridmet_cols = ['erc', 'fm100', 'bi', 'vpd', 'vs', 'rmax', 'rmin', 'tmmx', 'tmmn', 'pr', 'sph']
for col in gridmet_cols:
    if col in final.columns:
        vals = final[col].dropna()
        print(f"  {col:<10} min={vals.min():.2f}  max={vals.max():.2f}  "
              f"NaN={final[col].isna().sum():,} ({100*final[col].isna().mean():.1f}%)")
    else:
        print(f"  {col:<10} *** COLUMN NOT PRESENT ***")
