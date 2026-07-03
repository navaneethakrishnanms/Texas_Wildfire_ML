"""
src/missing_root_cause.py
--------------------------
Analysis B — Missing Value Root Cause Analysis

For every column with missing values, determines WHY it is missing
and what the correct treatment is.

Root cause types:
  1. structural       — format artifact (geometry column)
  2. event_conditional— only populated for qualifying events (MTBS = large fires only)
  3. geographic_coverage — spatial join found no feature within search radius
  4. administrative   — agency reporting varies, not universal
  5. sensor_gap       — weather station coverage gap
  6. unknown          — needs manual investigation

Input:
  tables/<state>/missing_summary.csv     (from Phase 1)
  tables/<state>/schema_analysis.csv     (from Phase 1)

Output:
  outputs/<state>/missing_root_cause.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.phase2_config import MISSING_ROOT_CAUSE_RULES, MISSING_TREATMENT_MAP


logger = logging.getLogger(__name__)


# Sensor gap keywords — weather columns in remote/sparse areas
SENSOR_GAP_KEYWORDS = [
    "tmmx", "tmmn", "vs", "sph", "rmin", "rmax", "pr_",
    "vpd", "station", "raws",
]


def _classify_root_cause(col: str, missing_pct: float) -> tuple[str, str]:
    """
    Classify why a column has missing values.

    Returns
    -------
    (root_cause_type, treatment)
    """
    col_lower = col.lower()

    # Check explicit rule lists first (highest priority)
    for cause_type, col_list in MISSING_ROOT_CAUSE_RULES.items():
        if col in col_list:
            return cause_type, MISSING_TREATMENT_MAP[cause_type]

    # Sensor gap heuristic — weather feature + moderate missingness
    if any(kw in col_lower for kw in SENSOR_GAP_KEYWORDS) and 5 < missing_pct < 60:
        return "sensor_gap", MISSING_TREATMENT_MAP["sensor_gap"]

    # Near-complete missingness in features that look like geographic joins
    if missing_pct > 85 and ("_dis" in col_lower or "station" in col_lower):
        return "geographic_coverage", MISSING_TREATMENT_MAP["geographic_coverage"]

    # Very high missingness (>95%) in non-spatial features → event-conditional
    if missing_pct > 95:
        return "event_conditional", MISSING_TREATMENT_MAP["event_conditional"]

    return "unknown", MISSING_TREATMENT_MAP["unknown"]


def generate_missing_root_cause(
    missing_csv: Path,
    schema_csv: Path,
    output_dir: Path,
    state_name: str,
) -> pd.DataFrame:
    """
    Analysis B: Classify WHY each column has missing values.

    Parameters
    ----------
    missing_csv  : Path to Phase 1 missing_summary.csv
    schema_csv   : Path to Phase 1 schema_analysis.csv
    output_dir   : Where to save outputs
    state_name   : 'Texas' or 'California'

    Returns
    -------
    pd.DataFrame  Root cause table (one row per column with any missing values)
    """
    logger.info(f"Analysis B — Missing Root Cause Analysis [{state_name}]")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not missing_csv.exists():
        logger.warning(f"  Missing summary CSV not found: {missing_csv}")
        return pd.DataFrame()

    missing_df = pd.read_csv(missing_csv)

    # Normalize column names
    col_field = "Column" if "Column" in missing_df.columns else missing_df.columns[0]
    pct_field = "Missing %" if "Missing %" in missing_df.columns else missing_df.columns[1]

    # Filter to only columns with missing values
    has_missing = missing_df[missing_df[pct_field] > 0].copy()

    logger.info(f"  Columns with missing values: {len(has_missing)}")

    rows = []
    cause_counts: dict[str, int] = {}

    for _, row in has_missing.iterrows():
        col = row[col_field]
        pct = float(row[pct_field])
        cause, treatment = _classify_root_cause(col, pct)

        cause_counts[cause] = cause_counts.get(cause, 0) + 1

        rows.append({
            "Column":         col,
            "Missing_%":      round(pct, 4),
            "Root_Cause":     cause,
            "Treatment":      treatment,
            "Action":         _treatment_to_action(cause, col),
        })

    df = pd.DataFrame(rows).sort_values("Missing_%", ascending=False).reset_index(drop=True)

    # Print summary
    print(f"\n{'=' * 65}")
    print(f"  ANALYSIS B — MISSING ROOT CAUSE [{state_name.upper()}]")
    print(f"{'=' * 65}")
    print(f"  {'Root Cause Type':<30} {'Count':>6}  {'Treatment'}")
    print(f"  {'-' * 62}")
    for cause, cnt in sorted(cause_counts.items(), key=lambda x: -x[1]):
        treatment_short = MISSING_TREATMENT_MAP.get(cause, "REVIEW")[:30]
        print(f"  {cause:<30} {cnt:>6}  {treatment_short}")
    print(f"  {'-' * 62}")
    print(f"  {'TOTAL':<30} {len(df):>6}")
    print(f"{'=' * 65}\n")

    # Log root cause breakdown
    for cause, cnt in cause_counts.items():
        logger.info(f"  {cause}: {cnt} columns")

    # Save
    out_path = output_dir / "missing_root_cause.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"  ✔ Saved: {out_path}")

    return df


def _treatment_to_action(cause: str, col: str) -> str:
    """Convert root cause + column into a short action label."""
    action_map = {
        "structural":          "EXCLUDE",
        "event_conditional":   "BINARY_FLAG" if "mtbs" not in col.lower() else "EXCLUDE",
        "geographic_coverage": "SENTINEL_999",
        "administrative":      "EXCLUDE",
        "sensor_gap":          "GRIDMET_FALLBACK",
        "unknown":             "REVIEW",
    }
    return action_map.get(cause, "REVIEW")
