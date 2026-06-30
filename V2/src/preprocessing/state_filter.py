"""
state_filter.py
---------------
Step 5: Split the master DataFrame into per-state DataFrames and persist them.

Public API
----------
filter_by_state(master, state_code)          ->  pd.DataFrame
save_state_dataset(df, state_cfg)            ->  None
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .config import COL_STATE, TARGET_STATES
from .logger import get_logger

log = get_logger(__name__)


def filter_by_state(master: pd.DataFrame, state_code: str) -> pd.DataFrame:
    """
    Return rows whose STATE column equals state_code.

    Parameters
    ----------
    master : pd.DataFrame
        The merged master dataset.
    state_code : str
        Two-letter state abbreviation, e.g. "TX".

    Returns
    -------
    pd.DataFrame
        Filtered subset, index reset.

    Raises
    ------
    KeyError
        If the STATE column is absent from the master DataFrame.
    ValueError
        If state_code is not recognised in TARGET_STATES.
    """
    if COL_STATE not in master.columns:
        raise KeyError(f"Column '{COL_STATE}' not found in master DataFrame.")
    if state_code not in TARGET_STATES:
        raise ValueError(
            f"Unknown state code '{state_code}'. "
            f"Supported: {list(TARGET_STATES.keys())}"
        )

    state_name = TARGET_STATES[state_code]["name"]
    log.info("Filtering for state: %s (%s) ...", state_name, state_code)

    mask = master[COL_STATE].astype(str).str.strip().str.upper() == state_code.upper()
    df   = master.loc[mask].reset_index(drop=True)

    log.info(
        "  %s -> %d rows (%.1f%% of master)",
        state_code,
        len(df),
        len(df) / len(master) * 100 if len(master) else 0,
    )
    return df


def save_state_dataset(df: pd.DataFrame, state_cfg: Dict[str, Any]) -> None:
    """
    Persist a state DataFrame as both Parquet and CSV.

    Parameters
    ----------
    df : pd.DataFrame
        State-filtered DataFrame.
    state_cfg : dict
        Entry from config.TARGET_STATES.
    """
    state_cfg["out_dir"].mkdir(parents=True, exist_ok=True)

    parquet_path: Path = state_cfg["parquet_out"]
    csv_path:     Path = state_cfg["csv_out"]

    # Parquet (efficient, preserves dtypes)
    df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")
    size_mb = parquet_path.stat().st_size / 1024 ** 2
    log.info("Saved Parquet -> %s  (%.1f MB)", parquet_path, size_mb)

    # CSV (human-readable)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    size_mb_csv = csv_path.stat().st_size / 1024 ** 2
    log.info("Saved CSV    -> %s  (%.1f MB)", csv_path, size_mb_csv)
