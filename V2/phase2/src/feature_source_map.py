"""
src/feature_source_map.py
--------------------------
Analysis A — Feature Source Mapping

For every column in the dataset, identifies:
  - Which data source it came from (GridMET, MODIS, LANDFIRE, SVI, etc.)
  - The original API or download URL
  - Native spatial resolution
  - Native temporal resolution
  - How frequently it must be updated in production

Input:
  tables/<state>/schema_analysis.csv   (from Phase 1)

Output:
  outputs/<state>/feature_source_map.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.phase2_config import SOURCE_KEYWORD_MAP


logger = logging.getLogger(__name__)


def _match_source(col_name: str) -> dict:
    """
    Match a column name to its source system using keyword rules.
    Returns the best matching source entry, or UNKNOWN if no match.
    """
    col_lower = col_name.lower()

    best_match = None
    best_score = 0

    for rule in SOURCE_KEYWORD_MAP:
        for kw in rule["keywords"]:
            if kw.lower() in col_lower:
                # Score by keyword length — longer match = more specific
                score = len(kw)
                if score > best_score:
                    best_score = score
                    best_match = rule

    if best_match:
        return best_match

    # No match found
    return {
        "source":     "UNKNOWN",
        "api_url":    "UNKNOWN — requires manual investigation",
        "spatial_res":"UNKNOWN",
        "temporal_res":"UNKNOWN",
        "update_freq":"UNKNOWN",
    }


def generate_feature_source_map(
    schema_csv: Path,
    output_dir: Path,
    state_name: str,
) -> pd.DataFrame:
    """
    Analysis A: Map every column to its original data source.

    Parameters
    ----------
    schema_csv   : Path to Phase 1 schema_analysis.csv
    output_dir   : Where to save outputs
    state_name   : 'Texas' or 'California'

    Returns
    -------
    pd.DataFrame  Feature source map (one row per column)
    """
    logger.info(f"Analysis A — Feature Source Mapping [{state_name}]")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load schema from Phase 1
    if not schema_csv.exists():
        logger.warning(f"  Schema CSV not found: {schema_csv}")
        return pd.DataFrame()

    schema_df = pd.read_csv(schema_csv)
    col_name_field = "Column Name" if "Column Name" in schema_df.columns else "Column"
    columns = schema_df[col_name_field].tolist()

    logger.info(f"  Processing {len(columns)} columns...")

    rows = []
    unknown_count = 0

    for col in columns:
        match = _match_source(col)
        source = match["source"]
        if source == "UNKNOWN":
            unknown_count += 1

        rows.append({
            "Column":           col,
            "Source_System":    source,
            "API_URL":          match["api_url"],
            "Spatial_Res":      match["spatial_res"],
            "Temporal_Res":     match["temporal_res"],
            "Update_Frequency": match["update_freq"],
        })

    df = pd.DataFrame(rows)

    # Summary by source
    src_counts = df["Source_System"].value_counts()
    logger.info(f"  Source system breakdown:")
    for src, cnt in src_counts.items():
        logger.info(f"    {src:<45} {cnt:>4} columns")
    logger.info(f"  UNKNOWN sources: {unknown_count} columns (require manual review)")

    # Save
    out_path = output_dir / "feature_source_map.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"  ✔ Saved: {out_path}")

    # Print summary table
    print(f"\n{'=' * 60}")
    print(f"  ANALYSIS A — FEATURE SOURCE MAP [{state_name.upper()}]")
    print(f"{'=' * 60}")
    print(f"  {'Source System':<45} {'Count':>6}")
    print(f"  {'-' * 52}")
    for src, cnt in src_counts.items():
        print(f"  {src:<45} {cnt:>6}")
    print(f"  {'-' * 52}")
    print(f"  {'TOTAL':<45} {len(df):>6}")
    print(f"  UNKNOWN (need review): {unknown_count}")
    print(f"{'=' * 60}\n")

    return df
