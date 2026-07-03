"""
src/schema_analysis.py
-----------------------
Analysis 2 & 3: Schema analysis and feature categorization.

Analysis 2 — Schema Analysis
    For every feature: dtype, example value, unique count, missing count/%, 
    semantic type, short description.
    Saves → tables/schema_analysis.csv

Analysis 3 — Feature Categorization
    Auto-classify every column into domain categories using keyword matching.
    Saves → tables/feature_category.csv
             tables/category_summary.csv
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from config.config import (
    FEATURE_CATEGORY_KEYWORDS,
    LOG_FILE,
    TABLES_DIR,
)
from src.utils import (
    classify_dtype,
    ensure_dirs,
    save_csv,
    setup_logger,
)

logger = setup_logger(__name__, LOG_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# Short Description Generator
# ─────────────────────────────────────────────────────────────────────────────

_KNOWN_DESCRIPTIONS: dict[str, str] = {
    "FOD_ID":                     "Fire occurrence database unique identifier",
    "OBJECTID":                   "GIS object identifier",
    "FIRE_YEAR":                  "Calendar year the fire was discovered",
    "STATE":                      "Two-letter US state abbreviation",
    "COUNTY":                     "County name where fire occurred",
    "LATITUDE":                   "Geographic latitude of fire origin (WGS84)",
    "LONGITUDE":                  "Geographic longitude of fire origin (WGS84)",
    "FIRE_NAME":                  "Name assigned to the fire",
    "FIRE_SIZE":                  "Final fire size in acres",
    "FIRE_SIZE_CLASS":            "Categorical fire size class (A–G)",
    "DISCOVERY_DATE":             "Date fire was first reported",
    "DISCOVERY_DOY":              "Discovery day-of-year (1–366)",
    "DISCOVERY_TIME":             "Time fire was first reported (HHMM)",
    "CONT_DATE":                  "Date fire was declared contained",
    "CONT_DOY":                   "Containment day-of-year",
    "CONT_TIME":                  "Time fire was contained (HHMM)",
    "NWCG_GENERAL_CAUSE":        "General cause category per NWCG standard",
    "NWCG_CAUSE_AGE_CATEGORY":  "Age category of responsible person (if human cause)",
    "NWCG_REPORTING_AGENCY":    "Agency that reported the fire",
    "OWNER_DESCR":               "Land ownership description",
    "COMPLEX_NAME":              "Name of fire complex (if part of multi-fire event)",
    "ICS_209_PLUS_INCIDENT_JOIN_ID": "Join key to ICS-209+ incident database",
    "ICS_209_PLUS_COMPLEX_JOIN_ID":  "Join key to ICS-209+ complex database",
    "MTBS_ID":                   "Monitoring Trends in Burn Severity fire ID",
    "MTBS_FIRE_NAME":            "Fire name in MTBS dataset",
    "NDVI":                      "Normalized Difference Vegetation Index",
    "ERC":                       "Energy Release Component (NFDRS fire weather index)",
    "BI":                        "Burning Index (NFDRS)",
    "SC":                        "Spread Component (NFDRS)",
    "IC":                        "Ignition Component (NFDRS)",
    "KBDI":                      "Keetch-Byram Drought Index",
    "TMAX":                      "Maximum daily temperature",
    "TMIN":                      "Minimum daily temperature",
    "PRCP":                      "Daily precipitation",
    "AWND":                      "Average daily wind speed",
    "No_FireStation_1.0km":     "Number of fire stations within 1 km",
    "No_FireStation_5.0km":     "Number of fire stations within 5 km",
    "geometry":                  "GIS geometry (WKT/WKB) — typically empty in tabular exports",
}


def _infer_description(col: str, series: pd.Series) -> str:
    """Attempt to generate a short column description from name and statistics."""
    if col in _KNOWN_DESCRIPTIONS:
        return _KNOWN_DESCRIPTIONS[col]

    col_up = col.upper()
    # Generic pattern matching
    if re.search(r"LAT", col_up):
        return "Latitude coordinate"
    if re.search(r"LON", col_up):
        return "Longitude coordinate"
    if re.search(r"DATE", col_up):
        return "Date field"
    if re.search(r"DOY", col_up):
        return "Day of year"
    if re.search(r"ELEV", col_up):
        return "Elevation (meters)"
    if re.search(r"SLOPE", col_up):
        return "Terrain slope"
    if re.search(r"PDSI|DROUGHT", col_up):
        return "Drought index"
    if re.search(r"NDVI|EVI|NBR", col_up):
        return "Vegetation index"
    if re.search(r"TMAX|TMIN|TAVG", col_up):
        return "Temperature variable"
    if re.search(r"PRCP|RAIN|PRECIP", col_up):
        return "Precipitation variable"
    if re.search(r"WIND|AWND|WSF", col_up):
        return "Wind variable"
    if re.search(r"POP|POPDEN", col_up):
        return "Population metric"
    if re.search(r"ROAD|HIGHWAY", col_up):
        return "Road/infrastructure distance"
    if re.search(r"NLCD|LANDCOVER", col_up):
        return "Land cover classification"
    if re.search(r"SOIL", col_up):
        return "Soil property"
    if re.search(r"DEM|SRTM", col_up):
        return "Digital elevation model"
    if re.search(r"ECOREGION|ECO3|ECO4", col_up):
        return "Ecoregion classification"
    if re.search(r"FIPS|COUNTY", col_up):
        return "Administrative geographic code"
    return "No description available"


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 2 — Schema Analysis
# ─────────────────────────────────────────────────────────────────────────────

def generate_schema_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analysis 2: Per-column schema table.

    Columns in output
    -----------------
    Column Name | Data Type | Semantic Type | Example Value |
    Unique Values | Missing Count | Missing % | Description

    Parameters
    ----------
    df : The merged dataset (read-only).

    Returns
    -------
    pd.DataFrame  Schema table (one row per feature).
    """
    logger.info("Analysis 2 — Schema Analysis")
    ensure_dirs(TABLES_DIR)

    records: list[dict] = []
    n_rows = len(df)

    for col in df.columns:
        s = df[col]
        missing_count = s.isna().sum()
        missing_pct   = round(missing_count / n_rows * 100, 4)
        n_unique      = s.nunique(dropna=True)
        semantic_type = classify_dtype(s)

        # Example value: first non-null value
        non_null = s.dropna()
        example  = str(non_null.iloc[0]) if not non_null.empty else "N/A"
        if len(example) > 80:
            example = example[:77] + "..."

        description = _infer_description(col, s)

        records.append({
            "Column Name":      col,
            "Data Type":        str(s.dtype),
            "Semantic Type":    semantic_type,
            "Example Value":    example,
            "Unique Values":    n_unique,
            "Missing Count":    missing_count,
            "Missing %":        missing_pct,
            "Description":      description,
        })

    schema_df = pd.DataFrame(records)
    out_path = TABLES_DIR / "schema_analysis.csv"
    save_csv(schema_df, out_path)
    logger.info(f"  ✔ Schema table saved: {out_path}  ({len(schema_df)} features)")
    return schema_df


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 3 — Feature Categorization
# ─────────────────────────────────────────────────────────────────────────────

def _match_category(col: str) -> str:
    """
    Assign a domain category to *col* using keyword substring matching.

    The FEATURE_CATEGORY_KEYWORDS dict is iterated in insertion order;
    the first match wins (more-specific patterns are listed first in config).
    Falls back to 'Other' if no keywords match.
    """
    col_upper = col.upper()
    col_lower = col.lower()
    for category, keywords in FEATURE_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.upper() in col_upper or kw.lower() in col_lower:
                return category
    return "Other"


def generate_feature_categorization(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analysis 3: Assign domain categories to every feature.

    Saves
    -----
    tables/feature_category.csv    — per-column category assignments
    tables/category_summary.csv    — count and list of columns per category

    Parameters
    ----------
    df : The merged dataset.

    Returns
    -------
    tuple[feature_category_df, category_summary_df]
    """
    logger.info("Analysis 3 — Feature Categorization")
    ensure_dirs(TABLES_DIR)

    rows: list[dict] = []
    for col in df.columns:
        category = _match_category(col)
        rows.append({"Column Name": col, "Category": category})

    feat_cat_df = pd.DataFrame(rows)

    # Category summary
    summary_records: list[dict] = []
    for cat, grp in feat_cat_df.groupby("Category", sort=True):
        cols_list = grp["Column Name"].tolist()
        summary_records.append({
            "Category":     cat,
            "Column Count": len(cols_list),
            "Columns":      ", ".join(cols_list),
        })
    cat_summary_df = pd.DataFrame(summary_records).sort_values(
        "Column Count", ascending=False
    ).reset_index(drop=True)

    save_csv(feat_cat_df,    TABLES_DIR / "feature_category.csv")
    save_csv(cat_summary_df, TABLES_DIR / "category_summary.csv")
    logger.info(f"  ✔ Feature category saved: {len(feat_cat_df)} columns → {feat_cat_df['Category'].nunique()} categories")

    # Console summary
    print("\n  Feature Category Distribution:")
    for _, row in cat_summary_df.iterrows():
        print(f"    {row['Category']:<30} {row['Column Count']:>4} columns")

    return feat_cat_df, cat_summary_df
