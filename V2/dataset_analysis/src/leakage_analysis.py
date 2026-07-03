"""
src/leakage_analysis.py
------------------------
Analysis 11 & 12: Leakage Detection and Predictive Readiness.

Analysis 11 — Leakage Analysis
    For every feature classify as:
        - Definite Leakage   : info only available after fire is detected/contained
        - Possible Leakage   : uncertain; may or may not be available at prediction time
        - Definitely Safe    : info available before fire (weather, terrain, etc.)

Analysis 12 — Predictive Readiness
    For every feature recommend:
        - Candidate Feature  : strong predictive signal, available pre-fire
        - Review Later       : may be useful but needs further investigation
        - Administrative     : ID/admin fields not useful for ML
        - Likely Remove      : leakage, constant, or entirely missing

Both analyses are annotation-only — no data is modified or removed.

Saves
-----
tables/leakage_analysis.csv
tables/predictive_readiness.csv
tables/leakage_summary.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.config import (
    LEAKAGE_DEFINITE_KEYWORDS,
    LEAKAGE_POSSIBLE_KEYWORDS,
    LOG_FILE,
    MISSING_HIGH,
    NEAR_CONSTANT_FRAC,
    READINESS_ADMINISTRATIVE,
    READINESS_CANDIDATE,
    READINESS_LIKELY_REMOVE,
    READINESS_REVIEW,
    TABLES_DIR,
)
from src.utils import ensure_dirs, save_csv, setup_logger

logger = setup_logger(__name__, LOG_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# Leakage Classifier
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that are definitely available BEFORE fire ignition
_PRE_FIRE_PATTERNS = [
    # Terrain / elevation
    "ELEV", "SLOPE", "ASPECT", "TPI", "TRI", "VRM", "DEM", "SRTM", "ROUGHNESS",
    # Weather (measured before/at time of detection)
    "TMAX", "TMIN", "TAVG", "PRCP", "SNOW", "AWND", "WSFG", "WSFI", "EVAP",
    "ERC", "BI", "SC", "IC", "KBDI", "FM", "NFDRS", "FWI", "BUI", "DSR", "RH", "FFWI",
    # Vegetation indices
    "NDVI", "EVI", "LAI", "FPAR", "NDWI", "SAVI", "NBR", "MSAVI",
    # Vegetation types
    "VEG", "GRASS", "SHRUB", "TREE", "FOREST", "CANOPY",
    # Land cover
    "NLCD", "LANDCOVER", "LAND_COVER", "LULC", "IMPERV", "URBAN",
    # Socioeconomic / static
    "POP", "POPDEN", "POPULATION", "CENSUS",
    "INCOME", "POVERTY", "UNEMPLOYMENT",
    "road", "ROAD", "HIGHWAY", "RAIL",
    "FireStation", "FIRE_STATION",
    # Geographic
    "LATITUDE", "LONGITUDE", "ECOREGION", "ECO3", "ECO4",
    # Climate
    "NORMAL", "CLIM", "ARIDITY", "PET",
    # Soil / hydrology
    "SOIL", "PDSI", "DROUGHT", "SPI", "SPEI",
    # Admin (available in GIS layers)
    "COUNTY", "STATE", "FIPS", "OWNER_DESCR",
    # Temporal
    "FIRE_YEAR",
]

_ADMIN_PATTERNS = [
    "FOD_ID", "OBJECTID", "SOURCE_SYSTEM",
    "NWCG_REPORTING_AGENCY", "NWCG_REPORTING_UNIT",
    "LOCAL_INCIDENT_ID", "GACC",
]


def _classify_leakage(col: str) -> str:
    """Return 'Definite Leakage', 'Possible Leakage', or 'Definitely Safe'."""
    col_u = col.upper()
    col_l = col.lower()

    # Check definite leakage
    for kw in LEAKAGE_DEFINITE_KEYWORDS:
        if kw.upper() in col_u:
            return "Definite Leakage"

    # Check possible leakage
    for kw in LEAKAGE_POSSIBLE_KEYWORDS:
        if kw.upper() in col_u:
            return "Possible Leakage"

    # Check known pre-fire patterns
    for kw in _PRE_FIRE_PATTERNS:
        if kw.upper() in col_u or kw.lower() in col_l:
            return "Definitely Safe"

    return "Definitely Safe"   # default: assume safe unless flagged


# ─────────────────────────────────────────────────────────────────────────────
# Predictive Readiness Classifier
# ─────────────────────────────────────────────────────────────────────────────

def _classify_readiness(
    col: str,
    leakage: str,
    missing_pct: float,
    is_constant: bool,
    n_unique: int,
) -> str:
    """Return the predictive readiness label for a column."""
    # Administrative / ID columns → Administrative
    col_u = col.upper()
    for pat in _ADMIN_PATTERNS + ["FOD_ID", "OBJECTID", "GEOMETRY", "GEOM"]:
        if pat.upper() in col_u:
            return READINESS_ADMINISTRATIVE

    # Definite leakage or extremely high missing → Likely Remove
    if leakage == "Definite Leakage":
        return READINESS_LIKELY_REMOVE

    if missing_pct >= MISSING_HIGH:
        return READINESS_REVIEW  # too much missing to decide

    if is_constant or n_unique <= 1:
        return READINESS_LIKELY_REMOVE

    # Possible leakage → Review Later
    if leakage == "Possible Leakage":
        return READINESS_REVIEW

    # Weather, terrain, vegetation (pre-fire) → Candidate
    _candidate_patterns = [
        "TMAX", "TMIN", "PRCP", "AWND", "ERC", "BI", "KBDI", "NDVI", "EVI",
        "SLOPE", "ELEV", "ASPECT", "POP", "road", "FireStation", "PDSI",
        "SOIL", "VEG", "FOREST", "SHRUB", "GRASS",
        "LATITUDE", "LONGITUDE", "FIRE_YEAR",
    ]
    for pat in _candidate_patterns:
        if pat.upper() in col_u or pat.lower() in col.lower():
            return READINESS_CANDIDATE

    return READINESS_REVIEW


# ─────────────────────────────────────────────────────────────────────────────
# Public Entry Points
# ─────────────────────────────────────────────────────────────────────────────

def generate_leakage_analysis(
    df: pd.DataFrame,
    quality_df: pd.DataFrame | None = None,
    missing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Analysis 11: Leakage Analysis.

    Parameters
    ----------
    df         : The merged dataset.
    quality_df : Output of feature_analysis.generate_feature_quality_analysis().
    missing_df : Output of missing_analysis.generate_missing_analysis().

    Returns
    -------
    pd.DataFrame  leakage_analysis.csv
    """
    logger.info("Analysis 11 — Leakage Analysis")
    ensure_dirs(TABLES_DIR)

    n_rows = len(df)

    # Build lookup maps from pre-computed tables
    missing_lookup: dict[str, float] = {}
    if missing_df is not None and "Column" in missing_df.columns and "Missing %" in missing_df.columns:
        missing_lookup = dict(zip(missing_df["Column"], missing_df["Missing %"]))

    is_constant_lookup: dict[str, bool] = {}
    n_unique_lookup: dict[str, int] = {}
    if quality_df is not None:
        for _, row in quality_df.iterrows():
            is_constant_lookup[row["Column"]] = bool(row.get("Is Constant", False))
            n_unique_lookup[row["Column"]]    = int(row.get("Unique Values", 0))

    records: list[dict] = []
    for col in df.columns:
        leakage    = _classify_leakage(col)
        missing_pct = missing_lookup.get(col, round(df[col].isna().sum() / n_rows * 100, 4))
        is_constant = is_constant_lookup.get(col, df[col].nunique() <= 1)
        n_unique    = n_unique_lookup.get(col, df[col].nunique())

        readiness = _classify_readiness(col, leakage, missing_pct, is_constant, n_unique)

        # Rationale
        if leakage == "Definite Leakage":
            rationale = "Contains information only known after fire occurrence or containment."
        elif leakage == "Possible Leakage":
            rationale = "May contain post-fire information; requires domain verification."
        else:
            rationale = "Information is typically available before fire ignition."

        records.append({
            "Column":            col,
            "Leakage Label":     leakage,
            "Readiness Label":   readiness,
            "Missing %":         missing_pct,
            "Is Constant":       is_constant,
            "Unique Values":     n_unique,
            "Rationale":         rationale,
        })

    leakage_df = pd.DataFrame(records)
    save_csv(leakage_df, TABLES_DIR / "leakage_analysis.csv")

    # Summary
    leakage_summary = leakage_df.groupby("Leakage Label")["Column"].count().reset_index()
    leakage_summary.columns = ["Leakage Label", "Column Count"]
    save_csv(leakage_summary, TABLES_DIR / "leakage_summary.csv")

    # Console
    print("\n" + "=" * 60)
    print("  LEAKAGE ANALYSIS")
    print("=" * 60)
    for label in ["Definite Leakage", "Possible Leakage", "Definitely Safe"]:
        cnt = (leakage_df["Leakage Label"] == label).sum()
        print(f"  {label:<25}: {cnt} columns")

    if (leakage_df["Leakage Label"] == "Definite Leakage").any():
        print("\n  Definite Leakage columns:")
        for col in leakage_df[leakage_df["Leakage Label"] == "Definite Leakage"]["Column"]:
            print(f"    - {col}")
    print("=" * 60 + "\n")

    logger.info(f"  ✔ Leakage analysis complete.")
    return leakage_df


def generate_predictive_readiness(leakage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analysis 12: Predictive Readiness classification.

    Uses the leakage_df produced by generate_leakage_analysis().

    Returns
    -------
    pd.DataFrame  predictive_readiness.csv
    """
    logger.info("Analysis 12 — Predictive Readiness")
    ensure_dirs(TABLES_DIR)

    save_csv(leakage_df[["Column", "Readiness Label", "Leakage Label", "Missing %", "Rationale"]],
             TABLES_DIR / "predictive_readiness.csv")

    # Summary by readiness label
    readiness_summary = leakage_df.groupby("Readiness Label")["Column"].count().reset_index()
    readiness_summary.columns = ["Readiness Label", "Column Count"]
    save_csv(readiness_summary, TABLES_DIR / "predictive_readiness_summary.csv")

    print("\n  Predictive Readiness Summary:")
    for _, r in readiness_summary.sort_values("Column Count", ascending=False).iterrows():
        print(f"    {r['Readiness Label']:<25}: {r['Column Count']:>4} columns")

    logger.info("  ✔ Predictive readiness analysis complete.")
    return leakage_df
