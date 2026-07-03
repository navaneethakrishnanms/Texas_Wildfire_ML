"""
src/feature_analysis.py
------------------------
Analysis 5: Feature Quality Analysis.

Detects
-------
- Constant columns (zero variance)
- Near-constant columns (one value dominates ≥ NEAR_CONSTANT_FRAC)
- Duplicate columns (byte-identical content)
- Duplicate information (columns with identical value distributions)
- Zero variance
- High cardinality categorical columns
- Infinite values
- Invalid numeric values (NaN within float columns already handled in missing)
- Unexpected negative values in columns that should be non-negative
- Columns where all values are 0

Saves
-----
tables/feature_quality.csv
tables/constant_columns.csv
tables/near_constant_columns.csv
tables/duplicate_columns.csv
tables/high_cardinality_columns.csv
tables/negative_value_columns.csv
tables/infinite_value_columns.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import (
    CARDINALITY_HIGH,
    LOG_FILE,
    NEAR_CONSTANT_FRAC,
    TABLES_DIR,
    ZERO_VARIANCE_TOL,
)
from src.utils import ensure_dirs, save_csv, setup_logger

logger = setup_logger(__name__, LOG_FILE)

# Columns that are expected to have non-negative values
_EXPECTED_NON_NEGATIVE_PATTERNS = [
    "FIRE_SIZE", "ELEV", "SLOPE", "POP", "NDVI", "EVI", "PRCP",
    "No_FireStation", "road_", "AREA", "COUNT", "DIST", "DENSITY",
]


def _is_expected_non_negative(col: str) -> bool:
    col_l = col.lower()
    return any(p.lower() in col_l for p in _EXPECTED_NON_NEGATIVE_PATTERNS)


def generate_feature_quality_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analysis 5: Feature Quality Analysis.

    Parameters
    ----------
    df : The merged dataset (never modified).

    Returns
    -------
    pd.DataFrame  Master quality table (one row per column).
    """
    logger.info("Analysis 5 — Feature Quality Analysis")
    ensure_dirs(TABLES_DIR)

    n_rows = len(df)
    records: list[dict] = []

    # Pre-compute
    numeric_df = df.select_dtypes(include="number")
    inf_counts  = np.isinf(numeric_df).sum()
    std_series  = numeric_df.std()
    all_zeros   = (numeric_df == 0).all()

    # Duplicate columns detection (hash-based for speed)
    logger.debug("  Computing column fingerprints for duplicate detection...")
    col_hash: dict[str, list[str]] = {}
    for col in df.columns:
        h = pd.util.hash_pandas_object(df[col].fillna("__NaN__"), index=False).sum()
        col_hash.setdefault(str(h), []).append(col)
    duplicate_groups = {k: v for k, v in col_hash.items() if len(v) > 1}
    dup_col_set = {col for group in duplicate_groups.values() for col in group}

    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        is_numeric = pd.api.types.is_numeric_dtype(s)

        # Missing
        missing_count = s.isna().sum()
        missing_pct   = round(missing_count / n_rows * 100, 4)

        # Non-null series
        s_nn = s.dropna()
        n_unique = s_nn.nunique()

        # Constant check
        is_constant = (n_unique <= 1)

        # Near-constant check
        near_constant = False
        dominant_value = None
        dominant_frac = 0.0
        if not is_constant and n_unique >= 1 and len(s_nn) > 0:
            top_count = s_nn.value_counts().iloc[0]
            dominant_frac = top_count / len(s_nn)
            dominant_value = s_nn.value_counts().index[0]
            near_constant = (dominant_frac >= NEAR_CONSTANT_FRAC)

        # Duplicate
        is_duplicate = col in dup_col_set

        # High cardinality
        high_cardinality = (not is_numeric) and (n_unique > CARDINALITY_HIGH)

        # Infinite values
        inf_count = int(inf_counts.get(col, 0))
        has_inf = inf_count > 0

        # Zero variance
        zero_var = False
        std_val = None
        if is_numeric and col in std_series.index:
            std_val = float(std_series[col])
            zero_var = (std_val < ZERO_VARIANCE_TOL)

        # All zeros
        all_zero = bool(all_zeros.get(col, False))

        # Unexpected negatives
        has_unexpected_negatives = False
        neg_count = 0
        if is_numeric and _is_expected_non_negative(col):
            neg_count = int((s_nn < 0).sum())
            has_unexpected_negatives = neg_count > 0

        # Quality flags (comma-separated)
        flags = []
        if is_constant:          flags.append("CONSTANT")
        if near_constant:        flags.append(f"NEAR_CONSTANT({dominant_frac:.1%})")
        if is_duplicate:         flags.append("DUPLICATE")
        if zero_var:             flags.append("ZERO_VARIANCE")
        if all_zero:             flags.append("ALL_ZEROS")
        if high_cardinality:     flags.append(f"HIGH_CARD({n_unique})")
        if has_inf:              flags.append(f"HAS_INF({inf_count})")
        if has_unexpected_negatives:  flags.append(f"NEG({neg_count})")

        quality_score = "PASS" if not flags else "REVIEW"

        records.append({
            "Column":                   col,
            "Data Type":                dtype,
            "Unique Values":            n_unique,
            "Missing %":                missing_pct,
            "Is Constant":              is_constant,
            "Is Near Constant":         near_constant,
            "Dominant Value":           str(dominant_value) if dominant_value is not None else "",
            "Dominant Fraction":        round(dominant_frac, 4),
            "Is Duplicate Column":      is_duplicate,
            "Zero Variance":            zero_var,
            "All Zeros":                all_zero,
            "High Cardinality":         high_cardinality,
            "Infinite Value Count":     inf_count,
            "Unexpected Negative Count": neg_count,
            "Quality Flags":            "; ".join(flags) if flags else "PASS",
            "Quality Score":            quality_score,
        })

    quality_df = pd.DataFrame(records)
    save_csv(quality_df, TABLES_DIR / "feature_quality.csv")

    # Save sub-tables
    constant_df     = quality_df[quality_df["Is Constant"]]
    near_const_df   = quality_df[quality_df["Is Near Constant"]]
    duplicate_df    = quality_df[quality_df["Is Duplicate Column"]]
    high_card_df    = quality_df[quality_df["High Cardinality"]]
    neg_df          = quality_df[quality_df["Unexpected Negative Count"] > 0]
    inf_df          = quality_df[quality_df["Infinite Value Count"] > 0]

    save_csv(constant_df,   TABLES_DIR / "constant_columns.csv")
    save_csv(near_const_df, TABLES_DIR / "near_constant_columns.csv")
    save_csv(duplicate_df,  TABLES_DIR / "duplicate_columns.csv")
    save_csv(high_card_df,  TABLES_DIR / "high_cardinality_columns.csv")
    save_csv(neg_df,        TABLES_DIR / "negative_value_columns.csv")
    save_csv(inf_df,        TABLES_DIR / "infinite_value_columns.csv")

    # Duplicate groups table
    dup_group_records = []
    for group_cols in duplicate_groups.values():
        dup_group_records.append({
            "Group Size": len(group_cols),
            "Duplicate Columns": ", ".join(group_cols),
        })
    if dup_group_records:
        save_csv(pd.DataFrame(dup_group_records), TABLES_DIR / "duplicate_groups.csv")

    # Console summary
    n_issues = (quality_df["Quality Score"] == "REVIEW").sum()
    print("\n" + "=" * 60)
    print("  FEATURE QUALITY ANALYSIS")
    print("=" * 60)
    print(f"  Total Columns              : {len(df.columns)}")
    print(f"  PASS (no issues)           : {(quality_df['Quality Score'] == 'PASS').sum()}")
    print(f"  REVIEW (has issues)        : {n_issues}")
    print(f"  Constant columns           : {len(constant_df)}")
    print(f"  Near-constant columns      : {len(near_const_df)}")
    print(f"  Duplicate columns          : {len(duplicate_df)}  ({len(duplicate_groups)} groups)")
    print(f"  High-cardinality columns   : {len(high_card_df)}")
    print(f"  Columns with negatives     : {len(neg_df)}")
    print(f"  Columns with inf values    : {len(inf_df)}")
    print("=" * 60 + "\n")

    logger.info(f"  ✔ Feature quality analysis complete. {n_issues} columns flagged for review.")
    return quality_df
