"""
inspect_datasets.py
--------------------
Inspects the Texas parquet files and exports CSV previews + summaries.
"""
import pandas as pd
from pathlib import Path

OUT = Path("outputs/texas")
CSV_OUT = OUT / "csv_exports"
CSV_OUT.mkdir(exist_ok=True)

# ── 1. Non-fire (negatives) ───────────────────────────────────────────────────
neg = pd.read_parquet(OUT / "negatives_labels.parquet")
print("\n" + "="*60)
print("NON-FIRE DATASET (negatives_labels.parquet)")
print("="*60)
print(f"  Rows    : {len(neg):,}")
print(f"  Columns : {len(neg.columns)}")
print(f"  Column names: {list(neg.columns)}")
print(neg.head(3).to_string())

# ── 2. Fire (positives) ───────────────────────────────────────────────────────
pos = pd.read_parquet(OUT / "positives_labels.parquet")
print("\n" + "="*60)
print("FIRE DATASET (positives_labels.parquet)")
print("="*60)
print(f"  Rows    : {len(pos):,}")
print(f"  Columns : {len(pos.columns)}")
print(f"  Column names: {list(pos.columns)}")
print(pos.head(3).to_string())

# ── 3. Full labels (fire + non-fire joined, no features yet) ─────────────────
full = pd.read_parquet(OUT / "full_training_labels.parquet")
print("\n" + "="*60)
print("FULL LABELS — fire + non-fire (full_training_labels.parquet)")
print("="*60)
print(f"  Rows    : {len(full):,}")
print(f"  Columns : {len(full.columns)}")
print(f"  Column names: {list(full.columns)}")
print(f"  Label counts: {full['label'].value_counts().to_dict()}")
print(full.head(3).to_string())

# ── 4. Final training dataset (all features) ──────────────────────────────────
final = pd.read_parquet(OUT / "final_training_dataset_tx.parquet")
print("\n" + "="*60)
print("FINAL TRAINING DATASET — all features (final_training_dataset_tx.parquet)")
print("="*60)
print(f"  Rows    : {len(final):,}")
print(f"  Columns : {len(final.columns)}")
print(f"  Column names: {list(final.columns)}")
print(f"  Label counts: {final['label'].value_counts().to_dict()}")
print(final.head(3).to_string())

# ── Export CSV samples ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Exporting CSV files...")

# Non-fire: first 10,000 rows
neg_csv = CSV_OUT / "nonfire_sample_10k.csv"
neg.head(10000).to_csv(neg_csv, index=False)
print(f"  ✔ Non-fire sample (10k rows): {neg_csv}")

# Fire only: all rows (34k)
pos_csv = CSV_OUT / "fire_all.csv"
pos.to_csv(pos_csv, index=False)
print(f"  ✔ Fire all rows ({len(pos):,} rows): {pos_csv}")

# Full labels (fire + non-fire): first 10,000 rows
full_csv = CSV_OUT / "full_labels_sample_10k.csv"
full.head(10000).to_csv(full_csv, index=False)
print(f"  ✔ Full labels sample (10k rows): {full_csv}")

# Final dataset (all features): first 10,000 rows
final_csv = CSV_OUT / "final_dataset_sample_10k.csv"
final.head(10000).to_csv(final_csv, index=False)
print(f"  ✔ Final dataset sample (10k rows): {final_csv}")

# Full final dataset — ALL rows
final_full_csv = CSV_OUT / "final_dataset_ALL.csv"
final.to_csv(final_full_csv, index=False)
mb = final_full_csv.stat().st_size / 1e6
print(f"  ✔ Final dataset FULL ({len(final):,} rows): {final_full_csv}  ({mb:.0f} MB)")

print("\n  All CSV exports saved to:", CSV_OUT)
print("="*60)
