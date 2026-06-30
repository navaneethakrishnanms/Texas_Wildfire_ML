"""
merger.py
---------
Step 4: Merge all yearly DataFrames into one master DataFrame and
print a comprehensive pre-filter quality summary.

Public API
----------
merge_datasets(datasets, schema_report)  ->  pd.DataFrame
print_master_summary(master_df)          ->  None
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .config import COL_FIRE_YEAR
from .logger import get_logger
from .schema_checker import SchemaReport

log = get_logger(__name__)


def merge_datasets(
    datasets: Dict[int, pd.DataFrame],
    schema_report: SchemaReport,
) -> pd.DataFrame:
    """
    Concatenate all yearly DataFrames into one master DataFrame.

    Columns present in only some years will be NaN for missing years -
    they are never silently dropped.

    Parameters
    ----------
    datasets : dict[int, pd.DataFrame]
    schema_report : SchemaReport

    Returns
    -------
    pd.DataFrame
        Master DataFrame with a source_year column for traceability.
    """
    log.info("=== Step 4: Merging %d yearly datasets ===", len(datasets))

    frames = []
    for year, df in sorted(datasets.items()):
        df = df.copy()
        df["source_year"] = year
        frames.append(df)
        log.debug("  Year %d -> %d rows", year, len(df))

    master = pd.concat(frames, axis=0, ignore_index=True)
    log.info("Master DataFrame shape: %d rows x %d columns", *master.shape)
    return master


def print_master_summary(master: pd.DataFrame) -> None:
    """
    Print a comprehensive quality summary of the merged master DataFrame.

    Covers: total rows/columns, rows per year, duplicate rows,
    missing values (top 30), memory usage.
    """
    sep = "-" * 72

    print(f"\n{sep}")
    print("  MASTER DATASET QUALITY SUMMARY (pre-filter)")
    print(sep)

    n_rows, n_cols = master.shape
    print(f"\n  Total rows       : {n_rows:>15,}")
    print(f"  Total columns    : {n_cols:>15,}")

    # Rows per year
    print(f"\n  --- Rows per year ---")
    year_col = COL_FIRE_YEAR if COL_FIRE_YEAR in master.columns else "source_year"
    if year_col in master.columns:
        vc = master[year_col].value_counts().sort_index()
        for yr, cnt in vc.items():
            pct = cnt / n_rows * 100
            print(f"    {yr}  ->  {cnt:>9,}  ({pct:5.1f}%)")
    else:
        print("    [FIRE_YEAR column not found]")

    # Duplicate rows
    print(f"\n  --- Duplicate rows ---")
    dup_count = master.duplicated().sum()
    dup_pct   = dup_count / n_rows * 100 if n_rows else 0
    print(f"    Exact duplicates: {dup_count:>9,}  ({dup_pct:.2f}%)")

    # Missing values
    print(f"\n  --- Missing values (top 30 columns by % missing) ---")
    null_counts = master.isnull().sum()
    null_pct    = null_counts / n_rows * 100
    missing_df  = (
        pd.DataFrame({"missing_count": null_counts, "missing_pct": null_pct})
        .query("missing_count > 0")
        .sort_values("missing_pct", ascending=False)
        .head(30)
    )
    if missing_df.empty:
        print("    No missing values.")
    else:
        print(f"    {'Column':<45}  {'Missing':>9}  {'Pct':>7}")
        print(f"    {'-'*45}  {'-'*9}  {'-'*7}")
        for col, row in missing_df.iterrows():
            print(f"    {col:<45}  {int(row.missing_count):>9,}  {row.missing_pct:>6.1f}%")

    cols_no_missing = (null_counts == 0).sum()
    print(f"\n    Columns with NO missing values: {cols_no_missing}/{n_cols}")

    # Memory usage
    print(f"\n  --- Memory usage ---")
    mem_bytes = master.memory_usage(deep=True).sum()
    mem_mb    = mem_bytes / 1024 ** 2
    mem_gb    = mem_bytes / 1024 ** 3
    if mem_gb >= 1:
        print(f"    Total : {mem_gb:.2f} GB  ({mem_bytes:,} bytes)")
    else:
        print(f"    Total : {mem_mb:.1f} MB  ({mem_bytes:,} bytes)")

    # Per-dtype breakdown
    dtype_mem = master.memory_usage(deep=True).groupby(master.dtypes).sum()
    print("    Breakdown by dtype:")
    for dtype, mem in dtype_mem.items():
        print(f"      {str(dtype):<15} {mem / 1024**2:>8.1f} MB")

    print(f"\n{sep}\n")

    log.info(
        "Master summary: %d rows x %d cols | %d dups | %.1f MB",
        n_rows, n_cols, dup_count, mem_mb,
    )
