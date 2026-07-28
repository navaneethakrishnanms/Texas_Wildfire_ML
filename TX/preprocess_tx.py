"""
preprocess_tx.py
-----------------
Preprocessing for final_training_dataset_tx_22.07.2026_landfire.xlsx
before XGBoost training — Texas standalone model.

Steps applied (per README_tx_dataset.md):
  1. Drop 454 duplicate rows  (README line 194-198)
  2. Drop 24,954 rows where gridmet_missing = 1  (README line 200-201)
  3. Zero-fill 1,594 NaN burnable/fire_count cells  (README line 203-204)
  4. avg_burn_prob kept on 0-11 scale — NO normalization
     (README line 183: "normalize only if combining with CA data")
     (README line 219: line is commented out — skip for TX-only training)
  5. Chronological train/val/test split
  6. Save clean parquets ready for train_tx.py

Usage:
    python preprocess_tx.py

Output:
    data/train_tx_clean.parquet
    data/val_tx_clean.parquet
    data/test_tx_clean.parquet
    data/full_tx_clean.parquet
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
EXCEL_PATH  = Path("final_training_dataset_tx_22.07.2026_landfire.xlsx")
OUT_DIR     = Path("data")
OUT_DIR.mkdir(exist_ok=True)

TRAIN_YEARS = [2014, 2015, 2016, 2017]
VAL_YEARS   = [2018]
TEST_YEARS  = [2019, 2020]

# ── Step 0: Load ──────────────────────────────────────────────────────────────
print("Loading Excel file... (may take ~30–60 seconds for 94 MB)")
df = pd.read_excel(EXCEL_PATH)
print(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")
print(f"  Fire rows (label=1): {(df['label']==1).sum():,}  "
      f"({100*(df['label']==1).mean():.2f}%)")
print(f"  Non-fire rows (label=0): {(df['label']==0).sum():,}")

# ── Step 1: Drop duplicates ───────────────────────────────────────────────────
# Source: README_tx_dataset.md line 194-198
# "Exact duplicates on (h3_cell, date_utc, window_hour) were present"
before = len(df)
df.drop_duplicates(subset=['h3_cell', 'date_utc', 'window_hour'], inplace=True)
dropped = before - len(df)
print(f"\n[Step 1] Drop duplicates")
print(f"  Removed: {dropped:,} rows  →  {len(df):,} rows remaining")

# ── Step 2: Drop rows with missing weather ────────────────────────────────────
# Source: README_tx_dataset.md line 200-201
# "Rows with gridmet_missing = 1 have NaN for all weather features"
# Recommendation: exclude rather than rely on XGBoost NaN handling
# (6.6% loss is acceptable — 350K rows is still very large)
before = len(df)
df = df[df['gridmet_missing'] != 1].copy()
dropped = before - len(df)
print(f"\n[Step 2] Drop missing weather rows (gridmet_missing=1)")
print(f"  Removed: {dropped:,} rows  →  {len(df):,} rows remaining")
print(f"  Fire rate after: {100*(df['label']==1).mean():.2f}%  (should stay ~9.09%)")

# ── Step 3: Zero-fill boundary cells ─────────────────────────────────────────
# Source: README_tx_dataset.md line 203-204
# "1,594 rows have NaN fire_count / burnable — edge/boundary cells"
for col in ['burnable', 'fire_count']:
    if col in df.columns:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            df[col] = df[col].fillna(0)
            print(f"\n[Step 3] Zero-filled '{col}': {n_nan:,} NaN → 0")

# ── Step 4: (OPTIONAL) Normalize avg_burn_prob 0-11 → 0-1 ───────────────────
# WHY: The TX archive uses a 0-11 integer scale (not 0-1 probability).
# FOR XGBOOST: NOT required — trees are scale-invariant.
# WHEN to do it: only if combining TX and CA data in the same training run.
#
# Uncomment the line below ONLY if mixing with California data:
# df['avg_burn_prob'] = df['avg_burn_prob'] / 11.0
# print("\n[Step 4] Normalized avg_burn_prob from 0-11 → 0-1")

# ── Step 5: Feature inventory ─────────────────────────────────────────────────
FEATURE_COLS = [
    # Landscape (LANDFIRE + TxWRAP — NOW REAL VALUES, not zeros)
    'avg_burn_prob',  # 0-11 scale (TX WRC archive) — tree-safe as-is
    'whp',            # 0-9 class
    'flep4',          # 0-0.9 probability (discretized class midpoints)
    'cfl',            # 0-110 ft (discretized class midpoints)
    'cbd',            # kg/m³ — NEW in this dataset (85% zeros = correct for TX grassland)
    'cbh',            # meters — NEW in this dataset (85% zeros = correct)
    'burnable',       # binary: 1=burnable, 0=not

    # gridMET daily weather
    'erc',            # Energy Release Component [BTU/ft²]
    'fm100',          # 100-hr fuel moisture [%]
    'vpd',            # Vapor pressure deficit [kPa]
    'vs',             # Wind speed [m/s]
    'rmax',           # Max relative humidity [%]
    'rmin',           # Min relative humidity [%]
    'tmmx',           # Max temperature [°C]
    'pr',             # Precipitation [mm]

    # 5-day rolling stats
    'erc_5D_mean',  'erc_5D_max',
    'fm100_5D_mean','fm100_5D_min',
    'vpd_5D_mean',  'vpd_5D_max',
    'vs_5D_mean',   'vs_5D_max',
    'rmax_5D_mean', 'rmax_5D_min',
    'tmmx_5D_mean', 'tmmx_5D_max',

    # Temporal (already in dataset)
    'sin_month', 'cos_month',
    'sin_hour',  'cos_hour',

    # Location
    'centroid_lat', 'centroid_lon',
]

# Validate all expected features are present
present   = [c for c in FEATURE_COLS if c in df.columns]
missing_f = [c for c in FEATURE_COLS if c not in df.columns]

print(f"\n[Step 5] Feature inventory")
print(f"  Present: {len(present)}/{len(FEATURE_COLS)} features")
if missing_f:
    print(f"  MISSING from dataset: {missing_f}")
    print("  → These won't be used in training (check column names)")

# DO NOT include these as features — leakage:
#   fire_count        → computed from full 2014-2020 FPA-FOD (includes test years)
#   has_fire_history  → derived from fire_count → same leakage
#   gridmet_missing   → flag column, not a feature
#   h3_cell, date_utc, window_6h_utc, window_hour → identifiers

# ── Step 6: Chronological split ───────────────────────────────────────────────
df['year'] = pd.to_datetime(df['date_utc']).dt.year
train = df[df['year'].isin(TRAIN_YEARS)].reset_index(drop=True)
val   = df[df['year'].isin(VAL_YEARS)].reset_index(drop=True)
test  = df[df['year'].isin(TEST_YEARS)].reset_index(drop=True)

print(f"\n[Step 6] Chronological split")
for name, split, years in [
    ("TRAIN", train, "2014–2017"),
    ("VAL",   val,   "2018"),
    ("TEST",  test,  "2019–2020"),
]:
    n_pos = (split['label'] == 1).sum()
    n_neg = (split['label'] == 0).sum()
    rate  = 100 * n_pos / len(split)
    print(f"  {name:<6} ({years}): {len(split):>8,} rows  "
          f"fire={n_pos:,}  non-fire={n_neg:,}  rate={rate:.1f}%")

# ── Step 7: Save ──────────────────────────────────────────────────────────────
df.to_parquet(OUT_DIR / "full_tx_clean.parquet",  index=False, compression="snappy")
train.to_parquet(OUT_DIR / "train_tx_clean.parquet", index=False, compression="snappy")
val.to_parquet(  OUT_DIR / "val_tx_clean.parquet",   index=False, compression="snappy")
test.to_parquet( OUT_DIR / "test_tx_clean.parquet",  index=False, compression="snappy")

print(f"\n[Step 7] Saved to {OUT_DIR}/")
print(f"  full_tx_clean.parquet")
print(f"  train_tx_clean.parquet")
print(f"  val_tx_clean.parquet")
print(f"  test_tx_clean.parquet")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print(f"  Final rows: {len(df):,}")
print(f"  Features ready: {len(present)}")
print(f"  avg_burn_prob: scale 0-11 (NOT normalized — XGBoost safe)")
print(f"  Missing weather rows: EXCLUDED (drop_duplicates + gridmet filter)")
print(f"\nNext: run your XGBoost training on train_tx_clean.parquet")
print("  Expected TEST AUROC: ~0.90-0.93 (vs current 0.857 with zeros)")
