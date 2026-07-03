"""
src/feature_group_analysis.py
------------------------------
Analysis 10, 13, 14: Categorical Analysis, Feature Dependency Groups,
and Source Readiness Classification.

Analysis 10 — Categorical Analysis
    For every categorical/object column: unique count, top categories,
    frequency, missing %. Plots for top features.

Analysis 13 — Feature Dependency Groups
    Groups related features together automatically by domain.

Analysis 14 — Source Readiness
    Classifies every column by update frequency:
    Static / Daily / Hourly / Monthly / Event Based / Administrative / Unknown

Saves
-----
tables/categorical_summary.csv
tables/feature_groups.csv
tables/source_readiness.csv
plots/categorical/<column>_distribution.png  (top features)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config.config import (
    FEATURE_CATEGORY_KEYWORDS,
    FIGURE_DPI,
    LOG_FILE,
    MATPLOTLIB_STYLE,
    PLOTS_CATEGORICAL_DIR,
    SOURCE_ADMIN,
    SOURCE_DAILY,
    SOURCE_EVENT,
    SOURCE_HOURLY,
    SOURCE_MONTHLY,
    SOURCE_STATIC,
    SOURCE_UNKNOWN,
    TABLES_DIR,
)
from src.utils import ensure_dirs, save_csv, save_figure, setup_logger

logger = setup_logger(__name__, LOG_FILE)

# Top N bars to plot per categorical column
_TOP_K_BARS = 20
# Number of categorical columns to generate individual plots for
_MAX_PLOTS  = 30


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 10 — Categorical Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _plot_cat_column(col: str, value_counts: pd.Series, plot_dir: Path) -> None:
    """Bar chart of top categories for a single categorical column."""
    ensure_dirs(plot_dir)
    top = value_counts.head(_TOP_K_BARS)
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(5, len(top) * 0.35)))
    ax.barh(top.index.astype(str), top.values, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Count", fontsize=10)
    ax.set_title(f"{col} — Top {_TOP_K_BARS} Categories", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8)
    safe = re.sub(r"[^\w]", "_", col)[:50]
    save_figure(fig, plot_dir / f"{safe}_distribution.png", dpi=FIGURE_DPI)


def generate_categorical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analysis 10: Per-column categorical analysis.

    Parameters
    ----------
    df : The merged dataset.

    Returns
    -------
    pd.DataFrame  Categorical summary table.
    """
    logger.info("Analysis 10 — Categorical Analysis")
    ensure_dirs(TABLES_DIR, PLOTS_CATEGORICAL_DIR)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    # Also include integer/float columns with low cardinality that may be codes
    for col in df.select_dtypes(include="number").columns:
        if df[col].nunique() <= 30 and df[col].nunique() > 0:
            if col not in cat_cols:
                cat_cols.append(col)

    n_rows = len(df)
    records: list[dict] = []
    plot_count = 0

    for col in cat_cols:
        s = df[col]
        vc = s.value_counts(dropna=True)
        n_unique      = vc.shape[0]
        missing_count = s.isna().sum()
        missing_pct   = round(missing_count / n_rows * 100, 4)
        top5          = vc.head(5)
        top5_str      = "; ".join([f"{k}: {v}" for k, v in top5.items()])
        top1_frac     = round(vc.iloc[0] / n_rows * 100, 4) if not vc.empty else 0.0

        records.append({
            "Column":            col,
            "Unique Count":      n_unique,
            "Missing %":         missing_pct,
            "Top Category":      str(vc.index[0]) if not vc.empty else "N/A",
            "Top Category Count": int(vc.iloc[0]) if not vc.empty else 0,
            "Top Category %":    top1_frac,
            "Top 5 Categories":  top5_str,
        })

        # Generate plot for top features (limited count)
        if plot_count < _MAX_PLOTS and n_unique >= 2:
            try:
                _plot_cat_column(col, vc, PLOTS_CATEGORICAL_DIR)
                plot_count += 1
            except Exception as exc:
                logger.warning(f"  Could not plot {col}: {exc}")

    cat_df = pd.DataFrame(records).sort_values("Unique Count", ascending=False).reset_index(drop=True)
    save_csv(cat_df, TABLES_DIR / "categorical_summary.csv")
    logger.info(f"  ✔ Categorical summary saved ({len(cat_df)} columns, {plot_count} plots)")
    return cat_df


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 13 — Feature Dependency Groups
# ─────────────────────────────────────────────────────────────────────────────

_DEPENDENCY_GROUPS: dict[str, list[str]] = {
    "Fire Identity":         ["FIRE_NAME", "FOD_ID", "OBJECTID", "FIRE_YEAR", "FIRE_CODE",
                              "LOCAL_FIRE_REPORT", "SOURCE_SYSTEM", "ICS_209", "MTBS"],
    "Fire Outcome":          ["FIRE_SIZE", "FIRE_SIZE_CLASS", "CONT_DATE", "CONT_DOY",
                              "CONT_TIME", "DURATION", "FIRE_MAG"],
    "Discovery":             ["DISCOVERY_DATE", "DISCOVERY_DOY", "DISCOVERY_TIME"],
    "Cause":                 ["NWCG_GENERAL_CAUSE", "NWCG_CAUSE_AGE", "CAUSE", "IGNIT"],
    "Administrative":        ["COUNTY", "STATE", "OWNER", "ADMIN", "UNIT", "AGENCY",
                              "DISTRICT", "GACC", "FIPS", "JURISDICTION", "NWCG_REPORTING"],
    "Geographic Coordinates":["LATITUDE", "LONGITUDE", "GEOMETRY", "GEOM", "COORD"],
    "Ecoregion":             ["ECOREGION", "ECO3", "ECO4", "NA_L3", "NA_L2"],
    "Weather":               ["TMAX", "TMIN", "TAVG", "PRCP", "SNOW", "AWND", "WSFG",
                              "WSFI", "EVAP", "TSUN"],
    "Fire Weather Index":    ["ERC", "BI", "SC", "IC", "KBDI", "FM", "NFDRS", "FWI", "BUI",
                              "DSR", "RH", "FFWI"],
    "Vegetation Index":      ["NDVI", "EVI", "LAI", "FPAR", "NDWI", "NDRE", "SAVI",
                              "NBR", "NBR2"],
    "Vegetation Types":      ["VEG", "GRASS", "SHRUB", "TREE", "FOREST", "CANOPY",
                              "HERB", "ExoticAnnualGrass", "PoaSecunda", "Medusahead"],
    "Terrain":               ["ELEV", "SLOPE", "ASPECT", "TPI", "TRI", "VRM", "ROUGHNESS", "DEM"],
    "Soil / Hydrology":      ["SOIL", "SWE", "PDSI", "DROUGHT", "SPI", "SPEI", "RUNOFF",
                              "WETLAND", "STREAM"],
    "Land Cover":            ["NLCD", "LANDCOVER", "LAND_COVER", "LULC", "IMPERV",
                              "URBAN", "CROP", "BARE"],
    "Population":            ["POP", "POPDEN", "POPULATION", "CENSUS", "HOUSEHOLD"],
    "Infrastructure":        ["road", "ROAD", "HIGHWAY", "RAIL", "POWER", "PIPELINE"],
    "Fire Stations":         ["FireStation", "FIRE_STATION", "STATION"],
    "Climate / EJ":          ["CLIMATE", "ARIDITY", "EJ", "MINORITY", "LOW_INCOME",
                              "JUSTICE", "EQUITY"],
    "Socioeconomic":         ["INCOME", "POVERTY", "UNEMPLOYMENT", "EDUCATION"],
    "Historical Fire":       ["HIST_FIRE", "FIRE_HIST", "BURN_AREA", "FIRE_FREQ",
                              "FIRE_RETURN"],
}


def _assign_dependency_group(col: str) -> str:
    """Return the dependency group for a column."""
    col_u = col.upper()
    col_l = col.lower()
    for group, keywords in _DEPENDENCY_GROUPS.items():
        for kw in keywords:
            if kw.upper() in col_u or kw.lower() in col_l:
                return group
    return "Other"


def generate_feature_dependency_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analysis 13: Group features into dependency clusters.

    Returns
    -------
    pd.DataFrame  feature_groups.csv — one row per column with group assignment.
    """
    logger.info("Analysis 13 — Feature Dependency Analysis")
    ensure_dirs(TABLES_DIR)

    rows = [
        {"Column": col, "Feature Group": _assign_dependency_group(col)}
        for col in df.columns
    ]
    fg_df = pd.DataFrame(rows)
    save_csv(fg_df, TABLES_DIR / "feature_groups.csv")

    # Summary
    grp_summary_rows = []
    for group, grp in fg_df.groupby("Feature Group"):
        cols_list = grp["Column"].tolist()
        grp_summary_rows.append({
            "Feature Group": group,
            "Count":         len(cols_list),
            "Columns":       ", ".join(cols_list),
        })
    grp_summary = pd.DataFrame(grp_summary_rows).sort_values(
        "Count", ascending=False
    ).reset_index(drop=True)
    save_csv(grp_summary, TABLES_DIR / "feature_groups_summary.csv")

    logger.info(f"  ✔ Feature groups saved ({len(fg_df)} columns → {fg_df['Feature Group'].nunique()} groups)")
    return fg_df


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 14 — Source Readiness
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE_RULES: list[tuple[list[str], str]] = [
    # Keyword patterns → update frequency label
    (["CONT_DATE", "CONT_DOY", "CONT_TIME",
      "DISCOVERY_DATE", "DISCOVERY_DOY", "DISCOVERY_TIME",
      "FIRE_NAME", "FIRE_CODE", "ICS_209", "MTBS",
      "LOCAL_FIRE_REPORT", "SOURCE_SYSTEM",
      "FOD_ID", "OBJECTID",
      "NWCG_CAUSE_AGE", "COMPLEX_NAME"],    SOURCE_EVENT),

    (["COUNTY", "STATE", "OWNER", "ADMIN", "UNIT", "AGENCY",
      "DISTRICT", "GACC", "FIPS", "JURISDICTION",
      "NWCG_REPORTING"],                    SOURCE_ADMIN),

    (["LATITUDE", "LONGITUDE", "GEOMETRY", "GEOM",
      "ECOREGION", "ECO3", "ECO4", "NA_L3", "NA_L2",
      "ELEV", "SLOPE", "ASPECT", "TPI", "TRI", "VRM",
      "DEM", "SRTM"],                       SOURCE_STATIC),

    (["TMAX", "TMIN", "TAVG", "PRCP", "SNOW", "AWND",
      "WSFG", "WSFI", "EVAP",
      "ERC", "BI", "SC", "IC", "KBDI", "FM", "NFDRS",
      "FWI", "BUI", "DSR", "RH", "FFWI",
      "KBDI"],                              SOURCE_DAILY),

    (["NDVI", "EVI", "LAI", "FPAR", "NDWI",
      "SAVI", "NBR", "MSAVI"],              SOURCE_MONTHLY),

    (["NORMAL", "CLIM", "LTAV", "LTA",
      "POP", "POPDEN", "POPULATION", "CENSUS",
      "NLCD", "LANDCOVER", "LAND_COVER",
      "LULC", "IMPERV", "URBAN",
      "INCOME", "POVERTY", "UNEMPLOYMENT"],  SOURCE_STATIC),
]


def _assign_source_readiness(col: str) -> str:
    col_u = col.upper()
    col_l = col.lower()
    for keywords, label in _SOURCE_RULES:
        for kw in keywords:
            if kw.upper() in col_u or kw.lower() in col_l:
                return label
    return SOURCE_UNKNOWN


def generate_source_readiness_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analysis 14: Classify columns by data update frequency.

    Returns
    -------
    pd.DataFrame  source_readiness.csv
    """
    logger.info("Analysis 14 — Source Readiness Analysis")
    ensure_dirs(TABLES_DIR)

    rows = [
        {"Column": col, "Update Frequency": _assign_source_readiness(col)}
        for col in df.columns
    ]
    sr_df = pd.DataFrame(rows)
    save_csv(sr_df, TABLES_DIR / "source_readiness.csv")

    # Summary
    sr_summary = sr_df.groupby("Update Frequency")["Column"].count().reset_index()
    sr_summary.columns = ["Update Frequency", "Column Count"]
    save_csv(sr_summary, TABLES_DIR / "source_readiness_summary.csv")

    print("\n  Source Readiness:")
    for _, r in sr_summary.sort_values("Column Count", ascending=False).iterrows():
        print(f"    {r['Update Frequency']:<20}: {r['Column Count']:>4} columns")

    logger.info(f"  ✔ Source readiness analysis complete ({len(sr_df)} columns)")
    return sr_df
