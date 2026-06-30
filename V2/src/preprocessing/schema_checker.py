"""
schema_checker.py
-----------------
Step 3: Verify that every yearly DataFrame shares the same column schema
and produce a detailed schema-comparison report.

Public API
----------
verify_schema(datasets)           ->  SchemaReport
save_schema_report(report, path)  ->  None
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .logger import get_logger

log = get_logger(__name__)


@dataclass
class SchemaReport:
    """Container for schema-comparison results."""

    # Column union across all years
    all_columns: List[str]

    # Columns present in every single year
    common_columns: List[str]

    # Columns missing in at least one year: {col_name: [missing_years]}
    missing_per_year: Dict[str, List[int]]

    # Dtype mismatches: {col_name: {year: dtype_str}}
    dtype_mismatches: Dict[str, Dict[int, str]]

    # Per-year column counts
    col_counts: Dict[int, int]

    # Per-year shape
    shapes: Dict[int, tuple]

    # Extra columns found in some years but not others
    extra_cols_per_year: Dict[int, List[str]] = field(default_factory=dict)

    @property
    def is_consistent(self) -> bool:
        """True when all years share the same columns with the same dtypes."""
        return (
            len(self.missing_per_year) == 0
            and len(self.dtype_mismatches) == 0
        )


def verify_schema(datasets: Dict[int, pd.DataFrame]) -> SchemaReport:
    """
    Compare column names and dtypes across all yearly DataFrames.

    Parameters
    ----------
    datasets : dict[int, pd.DataFrame]
        Year-keyed DataFrames (from the loader).

    Returns
    -------
    SchemaReport
        Detailed mismatch information - never silently drops columns.
    """
    log.info("=== Step 3: Schema Verification ===")

    years = sorted(datasets.keys())
    col_sets: Dict[int, set] = {yr: set(df.columns) for yr, df in datasets.items()}
    dtype_maps: Dict[int, Dict[str, str]] = {
        yr: {c: str(df[c].dtype) for c in df.columns}
        for yr, df in datasets.items()
    }

    all_cols    = sorted(set().union(*col_sets.values()))
    common_cols = sorted(set(all_cols).intersection(*col_sets.values()))
    col_counts  = {yr: len(col_sets[yr]) for yr in years}
    shapes      = {yr: datasets[yr].shape for yr in years}

    log.info("Column union : %d   |  Column intersection : %d", len(all_cols), len(common_cols))

    # Columns missing per year
    missing_per_year: Dict[str, List[int]] = {}
    for col in all_cols:
        absent_in = [yr for yr in years if col not in col_sets[yr]]
        if absent_in:
            missing_per_year[col] = absent_in

    if missing_per_year:
        log.warning(
            "%d column(s) are not present in all years - see report for details.",
            len(missing_per_year),
        )
        for col, absent in missing_per_year.items():
            log.warning("  Column %-40s  absent in year(s): %s", repr(col), absent)
    else:
        log.info("[OK] All years share the same %d columns.", len(all_cols))

    # Dtype mismatches for common columns
    dtype_mismatches: Dict[str, Dict[int, str]] = {}
    for col in common_cols:
        dtypes_seen = {yr: dtype_maps[yr][col] for yr in years}
        unique_dtypes = set(dtypes_seen.values())
        if len(unique_dtypes) > 1:
            dtype_mismatches[col] = dtypes_seen

    if dtype_mismatches:
        log.warning(
            "%d column(s) have inconsistent dtypes across years.", len(dtype_mismatches)
        )
        for col, dtypes in dtype_mismatches.items():
            log.warning("  Column %-40s  dtypes: %s", repr(col), dtypes)
    else:
        log.info("[OK] All common columns have consistent dtypes.")

    # Extra columns per year
    extra_cols_per_year: Dict[int, List[str]] = {}
    for yr in years:
        extra = sorted(col_sets[yr] - set(common_cols))
        if extra:
            extra_cols_per_year[yr] = extra

    # Per-year summary
    log.info("Per-year column counts:")
    for yr in years:
        log.info("  %d  ->  %d columns  |  shape %s", yr, col_counts[yr], shapes[yr])

    report = SchemaReport(
        all_columns         = all_cols,
        common_columns      = common_cols,
        missing_per_year    = missing_per_year,
        dtype_mismatches    = dtype_mismatches,
        col_counts          = col_counts,
        shapes              = {yr: list(v) for yr, v in shapes.items()},
        extra_cols_per_year = extra_cols_per_year,
    )

    verdict = "CONSISTENT [OK]" if report.is_consistent else "INCONSISTENT [!]"
    log.info("Schema status: %s", verdict)
    return report


def save_schema_report(report: SchemaReport, report_dir: Path, state_code: str) -> None:
    """
    Serialise the SchemaReport to JSON for the given state.

    Parameters
    ----------
    report : SchemaReport
    report_dir : Path
        Directory where the report is saved.
    state_code : str
        E.g. "TX" or "CA".
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "schema.json"

    payload = {
        "state":                  state_code,
        "schema_consistent":      report.is_consistent,
        "total_columns_union":    len(report.all_columns),
        "total_columns_common":   len(report.common_columns),
        "col_counts_per_year":    {str(k): v for k, v in report.col_counts.items()},
        "shapes_per_year":        {str(k): v for k, v in report.shapes.items()},
        "missing_per_year":       {col: yrs for col, yrs in report.missing_per_year.items()},
        "dtype_mismatches":       {
            col: {str(yr): dt for yr, dt in dtypes.items()}
            for col, dtypes in report.dtype_mismatches.items()
        },
        "extra_cols_per_year":    {str(yr): cols for yr, cols in report.extra_cols_per_year.items()},
        "all_columns":            report.all_columns,
        "common_columns":         report.common_columns,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log.info("Schema report saved -> %s", out_path)
