"""
loader.py
---------
Step 1 & 2: Discover, read, and column-standardise raw FPA-FOD yearly files.

Public API
----------
load_raw_datasets(raw_dir)  →  dict[int, pd.DataFrame]
    Reads every  YYYY_FPA_FOD_cons.csv  (or .xlsx) found in raw_dir,
    applies column-name standardisation, and returns a year-keyed dict.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import pandas as pd

from .config import (
    COL_FIRE_YEAR,
    RAW_FILE_GLOB,
    RAW_FILE_GLOB_XLSX,
    YEAR_RANGE,
)
from .logger import get_logger

log = get_logger(__name__)

# ── Column-name standardisation map ──────────────────────────────────────
# Keys are alternative spellings/cases found in older exports;
# values are the canonical names used throughout the pipeline.
_COL_RENAME_MAP: dict[str, str] = {
    # lowercase variants
    "fod_id":              "FOD_ID",
    "fpa_id":              "FPA_ID",
    "fire_year":           "FIRE_YEAR",
    "fire_name":           "FIRE_NAME",
    "fire_size":           "FIRE_SIZE",
    "fire_size_class":     "FIRE_SIZE_CLASS",
    "state":               "STATE",
    "county":              "COUNTY",
    "latitude":            "LATITUDE",
    "longitude":           "LONGITUDE",
    "discovery_date":      "DISCOVERY_DATE",
    "discovery_doy":       "DISCOVERY_DOY",
    "discovery_time":      "DISCOVERY_TIME",
    "cont_date":           "CONT_DATE",
    "cont_doy":            "CONT_DOY",
    "cont_time":           "CONT_TIME",
    "owner_descr":         "OWNER_DESCR",
    "nwcg_reporting_agency":       "NWCG_REPORTING_AGENCY",
    "nwcg_reporting_unit_id":      "NWCG_REPORTING_UNIT_ID",
    "nwcg_reporting_unit_name":    "NWCG_REPORTING_UNIT_NAME",
    "nwcg_cause_classification":   "NWCG_CAUSE_CLASSIFICATION",
    "nwcg_general_cause":          "NWCG_GENERAL_CAUSE",
    "nwcg_cause_age_category":     "NWCG_CAUSE_AGE_CATEGORY",
    "source_system_type":          "SOURCE_SYSTEM_TYPE",
    "source_system":               "SOURCE_SYSTEM",
    "source_reporting_unit":       "SOURCE_REPORTING_UNIT",
    "source_reporting_unit_name":  "SOURCE_REPORTING_UNIT_NAME",
    "local_fire_report_id":        "LOCAL_FIRE_REPORT_ID",
    "local_incident_id":           "LOCAL_INCIDENT_ID",
    "fire_code":                   "FIRE_CODE",
}


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to their canonical UPPER_SNAKE_CASE equivalents.
    Any column not in the rename map is kept as-is (preserving
    new/unknown columns rather than silently dropping them).
    """
    rename = {
        col: _COL_RENAME_MAP[col.lower()]
        for col in df.columns
        if col.lower() in _COL_RENAME_MAP
    }
    if rename:
        log.debug("  Renaming %d columns: %s", len(rename), list(rename.keys()))
    return df.rename(columns=rename)


def _extract_year_from_path(path: Path) -> int | None:
    """Parse the 4-digit year from a filename like 2014_FPA_FOD_cons.csv."""
    match = re.match(r"(\d{4})_", path.name)
    return int(match.group(1)) if match else None


def _read_file(path: Path) -> pd.DataFrame:
    """Read CSV or XLSX into a DataFrame with low-memory=False for dtype safety."""
    suffix = path.suffix.lower()
    log.info("  Reading %s …", path.name)
    if suffix == ".csv":
        df = pd.read_csv(path, low_memory=False, encoding="utf-8", on_bad_lines="warn")
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    log.info("  → %d rows × %d columns loaded.", len(df), len(df.columns))
    return df


def load_raw_datasets(raw_dir: Path) -> Dict[int, pd.DataFrame]:
    """
    Discover and load all yearly FPA-FOD files from *raw_dir*.

    Parameters
    ----------
    raw_dir : Path
        Directory containing  YYYY_FPA_FOD_cons.csv  (or .xlsx) files.

    Returns
    -------
    dict[int, pd.DataFrame]
        Year → DataFrame mapping, sorted ascending by year.

    Raises
    ------
    FileNotFoundError
        If no matching files are found in raw_dir.
    """
    raw_dir = Path(raw_dir)

    # Discover files (CSV preferred, fall back to XLSX)
    files: list[Path] = sorted(raw_dir.glob(RAW_FILE_GLOB))
    if not files:
        files = sorted(raw_dir.glob(RAW_FILE_GLOB_XLSX))
    if not files:
        raise FileNotFoundError(
            f"No FPA-FOD files found in {raw_dir}.\n"
            f"Expected pattern: {RAW_FILE_GLOB}"
        )

    log.info("Found %d raw file(s) in %s", len(files), raw_dir)

    datasets: Dict[int, pd.DataFrame] = {}
    missing_years: list[int] = []

    for path in files:
        year = _extract_year_from_path(path)
        if year is None:
            log.warning("Cannot parse year from filename '%s' – skipping.", path.name)
            continue

        df = _read_file(path)
        df = _standardise_columns(df)

        # Ensure FIRE_YEAR column is present; inject from filename if not
        if COL_FIRE_YEAR not in df.columns:
            log.warning("  '%s' missing FIRE_YEAR column – injecting from filename.", path.name)
            df[COL_FIRE_YEAR] = year

        datasets[year] = df

    # Warn about any expected years that were not found
    for yr in YEAR_RANGE:
        if yr not in datasets:
            missing_years.append(yr)
    if missing_years:
        log.warning("Missing data for year(s): %s", missing_years)

    log.info("Successfully loaded years: %s", sorted(datasets.keys()))
    return dict(sorted(datasets.items()))
