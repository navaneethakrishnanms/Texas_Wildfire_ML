"""
validator.py
------------
Data validation for the state-filtered DataFrames.

Rules validated
---------------
LATITUDE       : must be in [-90, 90]
LONGITUDE      : must be in [-180, 180]
DISCOVERY_DATE : must be non-null and parseable as a date
FIRE_SIZE      : must be >= 0  (and <= 1,000,000 acres)
STATE          : must match expected two-letter abbreviation
COUNTY         : should be non-null (warning only)

Invalid rows are reported but NEVER removed.

Public API
----------
validate_dataframe(df, state_code)           ->  ValidationReport
save_validation_report(report, report_dir)   ->  None
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import (
    COL_COUNTY,
    COL_DISCOVERY_DATE,
    COL_FIRE_SIZE,
    COL_FIRE_YEAR,
    COL_LATITUDE,
    COL_LONGITUDE,
    COL_STATE,
    VALID_FIRE_SIZE,
    VALID_LAT_RANGE,
    VALID_LON_RANGE,
    VALID_YEARS,
)
from .logger import get_logger

log = get_logger(__name__)


@dataclass
class ValidationReport:
    """Summary of validation results for one state dataset."""
    state_code: str
    total_rows: int

    # Maps rule_name -> list of row indices that failed
    violations: Dict[str, List[int]] = field(default_factory=dict)

    # Maps rule_name -> human-readable description
    rule_descriptions: Dict[str, str] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return all(len(v) == 0 for v in self.violations.values())

    def summary_lines(self) -> List[str]:
        lines = [
            f"Validation Report - {self.state_code}",
            f"Total rows : {self.total_rows:,}",
            "",
        ]
        for rule, idxs in self.violations.items():
            pct = len(idxs) / self.total_rows * 100 if self.total_rows else 0
            desc = self.rule_descriptions.get(rule, "")
            lines.append(
                f"  [{rule}]  {len(idxs):>8,} invalid rows  ({pct:.2f}%)  - {desc}"
            )
        if self.is_clean:
            lines.append("  [OK] No validation violations found.")
        return lines


def _check_numeric_range(
    series: pd.Series,
    lo: float,
    hi: float,
    allow_null: bool = True,
) -> List[int]:
    """Return indices where series is outside [lo, hi]."""
    numeric = pd.to_numeric(series, errors="coerce")
    mask = numeric.notna() & ((numeric < lo) | (numeric > hi))
    if not allow_null:
        mask = mask | numeric.isna()
    return list(series.index[mask])


def _check_date_parseable(series: pd.Series) -> List[int]:
    """Return indices where series cannot be parsed as a date."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return list(series.index[series.isna()])
    coerced = pd.to_datetime(series, errors="coerce")
    return list(series.index[coerced.isna()])


def validate_dataframe(df: pd.DataFrame, state_code: str) -> ValidationReport:
    """
    Run all validation rules against df and return a ValidationReport.
    Invalid rows are flagged only - never dropped.

    Parameters
    ----------
    df : pd.DataFrame
        State-filtered DataFrame.
    state_code : str
        Two-letter state abbreviation.

    Returns
    -------
    ValidationReport
    """
    log.info("=== Data Validation: %s ===", state_code)

    report = ValidationReport(state_code=state_code, total_rows=len(df))

    # 1. Latitude
    if COL_LATITUDE in df.columns:
        lo, hi = VALID_LAT_RANGE
        bad = _check_numeric_range(df[COL_LATITUDE], lo, hi, allow_null=False)
        report.violations["LATITUDE"] = bad
        report.rule_descriptions["LATITUDE"] = f"Must be numeric in [{lo}, {hi}]"
        if bad:
            log.warning("  LATITUDE  : %d invalid rows", len(bad))
        else:
            log.info("  LATITUDE  : [OK] all valid")
    else:
        log.warning("  LATITUDE column not found - skipping.")

    # 2. Longitude
    if COL_LONGITUDE in df.columns:
        lo, hi = VALID_LON_RANGE
        bad = _check_numeric_range(df[COL_LONGITUDE], lo, hi, allow_null=False)
        report.violations["LONGITUDE"] = bad
        report.rule_descriptions["LONGITUDE"] = f"Must be numeric in [{lo}, {hi}]"
        if bad:
            log.warning("  LONGITUDE : %d invalid rows", len(bad))
        else:
            log.info("  LONGITUDE : [OK] all valid")
    else:
        log.warning("  LONGITUDE column not found - skipping.")

    # 3. Discovery Date
    if COL_DISCOVERY_DATE in df.columns:
        bad = _check_date_parseable(df[COL_DISCOVERY_DATE])
        report.violations["DISCOVERY_DATE"] = bad
        report.rule_descriptions["DISCOVERY_DATE"] = "Must be a parseable, non-null date"
        if bad:
            log.warning("  DISCOVERY_DATE : %d invalid/null rows", len(bad))
        else:
            log.info("  DISCOVERY_DATE : [OK] all valid")
    else:
        log.warning("  DISCOVERY_DATE column not found - skipping.")

    # 4. Fire Size
    if COL_FIRE_SIZE in df.columns:
        lo, hi = VALID_FIRE_SIZE
        bad = _check_numeric_range(df[COL_FIRE_SIZE], lo, hi, allow_null=False)
        report.violations["FIRE_SIZE"] = bad
        report.rule_descriptions["FIRE_SIZE"] = f"Must be numeric in [{lo}, {hi:,}] acres"
        if bad:
            log.warning("  FIRE_SIZE : %d invalid rows", len(bad))
        else:
            log.info("  FIRE_SIZE : [OK] all valid")
    else:
        log.warning("  FIRE_SIZE column not found - skipping.")

    # 5. State value matches expected
    if COL_STATE in df.columns:
        bad = list(
            df.index[df[COL_STATE].astype(str).str.strip().str.upper() != state_code]
        )
        report.violations["STATE_MISMATCH"] = bad
        report.rule_descriptions["STATE_MISMATCH"] = f"STATE must equal '{state_code}'"
        if bad:
            log.warning("  STATE mismatch : %d rows with unexpected state value", len(bad))
        else:
            log.info("  STATE : [OK] all rows are %s", state_code)

    # 6. County non-null (warning only)
    if COL_COUNTY in df.columns:
        null_counties = list(df.index[df[COL_COUNTY].isna()])
        report.violations["COUNTY_NULL"] = null_counties
        report.rule_descriptions["COUNTY_NULL"] = "COUNTY should be non-null (warning only)"
        if null_counties:
            log.warning("  COUNTY : %d null values (rows not removed)", len(null_counties))
        else:
            log.info("  COUNTY : [OK] no nulls")

    # 7. Fire Year in expected range
    if COL_FIRE_YEAR in df.columns:
        numeric_yr = pd.to_numeric(df[COL_FIRE_YEAR], errors="coerce")
        bad = list(df.index[~numeric_yr.isin(VALID_YEARS) | numeric_yr.isna()])
        report.violations["FIRE_YEAR_OOB"] = bad
        report.rule_descriptions["FIRE_YEAR_OOB"] = (
            f"FIRE_YEAR must be in {VALID_YEARS}"
        )
        if bad:
            log.warning("  FIRE_YEAR out-of-bounds : %d rows", len(bad))
        else:
            log.info("  FIRE_YEAR : [OK] all valid")

    for line in report.summary_lines():
        log.info(line)

    return report


def save_validation_report(report: ValidationReport, report_dir: Path) -> None:
    """
    Save the validation report as a JSON file.

    Parameters
    ----------
    report : ValidationReport
    report_dir : Path
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "validation_report.json"

    payload = {
        "state_code": report.state_code,
        "total_rows": report.total_rows,
        "is_clean":   report.is_clean,
        "violations": {
            rule: {
                "description":    report.rule_descriptions.get(rule, ""),
                "invalid_count":  len(idxs),
                "invalid_pct":    round(len(idxs) / report.total_rows * 100, 4)
                                  if report.total_rows else 0,
                "sample_indices": idxs[:50],
            }
            for rule, idxs in report.violations.items()
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log.info("Validation report saved -> %s", out_path)
