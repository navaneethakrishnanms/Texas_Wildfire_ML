"""
src/load_data.py
----------------
Data loading module for the Wildfire Dataset Analysis Pipeline.

Loads the pre-processed Texas or California dataset directly from:
  V2/data/processed/texas/texas_fire_2014_2020.parquet
  V2/data/processed/california/california_fire_2014_2020.parquet

These files are produced by the Phase-1 preprocessing pipeline (run_phase1.py).
The dataset is NEVER modified — strictly read-only analysis.

Analysis 1 — Dataset Overview
    Generates tables/<state>/dataset_overview.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.config import (
    STATE_DATASETS,
    LOG_FILE,
)
from src.utils import (
    classify_dtype,
    ensure_dirs,
    memory_usage_mb,
    save_csv,
    setup_logger,
    timer,
)

logger = setup_logger(__name__, LOG_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_state_dataset(state_code: str) -> pd.DataFrame:
    """
    Load the pre-processed dataset for a given state.

    Tries Parquet first (fast), falls back to CSV if Parquet is missing.

    Parameters
    ----------
    state_code : 'TX' for Texas, 'CA' for California.

    Returns
    -------
    pd.DataFrame  The state dataset (read-only — never modified).

    Raises
    ------
    ValueError  If state_code is not in STATE_DATASETS.
    FileNotFoundError  If neither Parquet nor CSV exists.
    """
    state_code = state_code.upper()
    if state_code not in STATE_DATASETS:
        raise ValueError(
            f"Unknown state code '{state_code}'. "
            f"Valid options: {list(STATE_DATASETS.keys())}"
        )

    cfg = STATE_DATASETS[state_code]
    state_name = cfg["name"]
    parquet_path = Path(cfg["parquet"])
    csv_path     = Path(cfg["csv"])

    logger.info(f"Loading dataset for: {state_name} ({state_code})")

    # Try Parquet (preferred — much faster)
    if parquet_path.exists():
        logger.info(f"  Reading Parquet: {parquet_path}")
        with timer(f"Load {state_name} Parquet", logger):
            df = pd.read_parquet(parquet_path, engine="pyarrow")
        logger.info(f"  ✔ Loaded → {len(df):,} rows × {len(df.columns)} columns")
        return df

    # Fallback to CSV
    if csv_path.exists():
        logger.info(f"  Parquet not found. Reading CSV: {csv_path}")
        with timer(f"Load {state_name} CSV", logger):
            df = pd.read_csv(csv_path, low_memory=False)
        logger.info(f"  ✔ Loaded → {len(df):,} rows × {len(df.columns)} columns")
        return df

    raise FileNotFoundError(
        f"Neither Parquet nor CSV found for {state_name}.\n"
        f"  Expected Parquet: {parquet_path}\n"
        f"  Expected CSV    : {csv_path}\n"
        f"  Run 'run_phase1.py' first to produce the processed files."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1 — Dataset Overview
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset_overview(
    df: pd.DataFrame,
    tables_dir: Path,
    state_name: str = "",
) -> pd.DataFrame:
    """
    Analysis 1: Generate a high-level dataset overview table.

    Computes
    --------
    - Row / column counts
    - Memory usage (MB)
    - Dtype breakdown
    - Semantic type counts

    Saves
    -----
    tables/<state>/dataset_overview.csv

    Parameters
    ----------
    df         : The state dataset (read-only).
    tables_dir : State-specific tables directory.
    state_name : Human-readable state name (for display).

    Returns
    -------
    pd.DataFrame  Overview table (Property | Value).
    """
    logger.info(f"Analysis 1 — Dataset Overview [{state_name}]")
    ensure_dirs(tables_dir)

    dtype_counts = df.dtypes.value_counts().to_dict()
    dtype_str    = "; ".join(f"{str(k)}: {v}" for k, v in dtype_counts.items())

    type_series      = df.apply(classify_dtype)
    numeric_cols     = (type_series == "Numeric").sum()
    categorical_cols = (type_series == "Categorical").sum()
    datetime_cols    = (type_series == "Datetime").sum()
    boolean_cols     = (type_series == "Boolean").sum()
    text_cols        = (type_series == "Text").sum()
    object_cols      = (df.dtypes == object).sum()

    mem_mb = memory_usage_mb(df)

    rows = [
        ("State",                    state_name),
        ("Total Rows",               f"{len(df):,}"),
        ("Total Columns",            f"{len(df.columns):,}"),
        ("Memory Usage (MB)",        f"{mem_mb:.2f} MB"),
        ("Dtype Breakdown",          dtype_str),
        ("Numeric Columns",          f"{numeric_cols}"),
        ("Categorical Columns",      f"{categorical_cols}"),
        ("Datetime Columns",         f"{datetime_cols}"),
        ("Boolean Columns",          f"{boolean_cols}"),
        ("Object Columns (raw)",     f"{object_cols}"),
        ("Text Columns",             f"{text_cols}"),
        ("Float64 Columns",          f"{(df.dtypes == 'float64').sum()}"),
        ("Int64 Columns",            f"{(df.dtypes == 'int64').sum()}"),
        ("Year Range",               "2014 – 2020"),
        ("Data Source",              "FPA-FOD Pre-processed (Phase-1 Pipeline)"),
    ]

    overview_df = pd.DataFrame(rows, columns=["Property", "Value"])
    out_path = tables_dir / "dataset_overview.csv"
    save_csv(overview_df, out_path)
    logger.info(f"  ✔ Saved: {out_path}")

    # Console print
    print(f"\n{'=' * 60}")
    print(f"  DATASET OVERVIEW — {state_name.upper()}")
    print(f"{'=' * 60}")
    for prop, val in rows:
        print(f"  {prop:<30} {val}")
    print(f"{'=' * 60}\n")

    return overview_df
