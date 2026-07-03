"""
config/config.py
----------------
Central configuration for the Wildfire Dataset Analysis Pipeline.

All paths, thresholds, and constants live here so every module
stays free of hard-coded literals.
"""

from __future__ import annotations
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Root Paths
# ─────────────────────────────────────────────────────────────────────────────
# This file lives at:  dataset_analysis/config/config.py
# Project root  →      dataset_analysis/
# V2 root       →      V2/

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]   # dataset_analysis/
V2_ROOT       = ANALYSIS_ROOT.parent                  # V2/

# ─────────────────────────────────────────────────────────────────────────────
# Pre-processed State Datasets
# Located at V2/data/processed/<state>/<state>_fire_2014_2020.parquet
# These are produced by the Phase-1 preprocessing pipeline (run_phase1.py).
# ─────────────────────────────────────────────────────────────────────────────
PROCESSED_DIR = V2_ROOT / "data" / "processed"

STATE_DATASETS = {
    "TX": {
        "name":    "Texas",
        "parquet": PROCESSED_DIR / "texas"      / "texas_fire_2014_2020.parquet",
        "csv":     PROCESSED_DIR / "texas"      / "texas_fire_2014_2020.csv",
    },
    "CA": {
        "name":    "California",
        "parquet": PROCESSED_DIR / "california" / "california_fire_2014_2020.parquet",
        "csv":     PROCESSED_DIR / "california" / "california_fire_2014_2020.csv",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Output Directories (per-state sub-folders are created at runtime)
# ─────────────────────────────────────────────────────────────────────────────
REPORTS_DIR = ANALYSIS_ROOT / "reports"
TABLES_DIR  = ANALYSIS_ROOT / "tables"
PLOTS_DIR   = ANALYSIS_ROOT / "plots"
LOGS_DIR    = ANALYSIS_ROOT / "logs"

# Sub-directories for plots (per-state paths are built dynamically in run_analysis.py)
PLOTS_MISSING_DIR     = PLOTS_DIR / "missing"
PLOTS_STATS_DIR       = PLOTS_DIR / "statistics"
PLOTS_TEMPORAL_DIR    = PLOTS_DIR / "temporal"
PLOTS_GEOGRAPHIC_DIR  = PLOTS_DIR / "geographic"
PLOTS_CORRELATION_DIR = PLOTS_DIR / "correlation"
PLOTS_CATEGORICAL_DIR = PLOTS_DIR / "categorical"

# ─────────────────────────────────────────────────────────────────────────────
# Key Column Names  (as they appear in the merged dataset)
# ─────────────────────────────────────────────────────────────────────────────
COL_STATE          = "STATE"
COL_COUNTY         = "COUNTY"
COL_LATITUDE       = "LATITUDE"
COL_LONGITUDE      = "LONGITUDE"
COL_DISCOVERY_DATE = "DISCOVERY_DATE"
COL_CONT_DATE      = "CONT_DATE"
COL_FIRE_SIZE      = "FIRE_SIZE"
COL_FIRE_YEAR      = "FIRE_YEAR"
COL_FIRE_NAME      = "FIRE_NAME"
COL_CAUSE          = "NWCG_GENERAL_CAUSE"
COL_AGENCY         = "NWCG_REPORTING_AGENCY"
COL_OWNER          = "OWNER_DESCR"
COL_SOURCE_SYSTEM  = "SOURCE_SYSTEM"
COL_DISCOVERY_DOY  = "DISCOVERY_DOY"

# ─────────────────────────────────────────────────────────────────────────────
# Analysis Thresholds
# ─────────────────────────────────────────────────────────────────────────────
# Missing value thresholds
MISSING_CRITICAL    = 90.0   # %
MISSING_HIGH        = 75.0
MISSING_MODERATE    = 50.0
MISSING_LOW         = 25.0

# Statistical thresholds
SKEWNESS_HIGH       = 2.0    # |skew| > this → highly skewed
KURTOSIS_HIGH       = 7.0    # |kurtosis| > this → heavy-tailed
IQR_OUTLIER_FACTOR  = 3.0    # multiplier for IQR outlier fence (3×IQR from Q1/Q3)

# Correlation thresholds
CORR_HIGH           = 0.90   # |r| > this → highly correlated
CORR_MODERATE       = 0.70

# Cardinality thresholds
CARDINALITY_HIGH    = 100    # unique values > this → high cardinality
NEAR_CONSTANT_FRAC  = 0.98   # dominant-class fraction > this → near-constant

# Variance
ZERO_VARIANCE_TOL   = 1e-10

# ─────────────────────────────────────────────────────────────────────────────
# Feature Category Keywords
# Ordered: more specific patterns first to avoid mis-classification.
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Fire Metadata": [
        "FIRE_NAME", "FIRE_CODE", "FOD_ID", "LOCAL_FIRE_REPORT", "SOURCE_SYSTEM",
        "NWCG_REPORTING", "COMPLEX_NAME", "ICS_209", "MTBS_ID", "MTBS_FIRE",
        "OBJECTID", "FIRE_YEAR", "FIRE_ID",
    ],
    "Fire Outcome": [
        "FIRE_SIZE", "FIRE_SIZE_CLASS", "CONT_DATE", "CONT_DOY", "CONT_TIME",
        "DURATION", "FIRE_MAG", "DAMAGE", "STRUCTURE",
    ],
    "Fire Weather": [
        "ERC", "BI", "SC", "KBDI", "IC", "FM", "RH", "FFWI", "NFDRS",
        "FIRE_WEATHER", "FWI", "DSR", "BUI",
    ],
    "Weather": [
        "TMAX", "TMIN", "TAVG", "PRCP", "SNOW", "AWND", "WSFI", "PGTM",
        "WSFG", "WSFM", "WT", "WV", "TSUN", "WESD", "MDPR", "DAPR",
        "EVAP", "MXPN", "MNPN", "THIC", "PSUN",
        "temp", "precip", "wind", "humid", "dew", "pressure", "cloud",
        "ACMH", "ACSH", "TSUN",
    ],
    "Weather Normal": [
        "NORMAL", "CLIM", "LTAV", "LTA",
    ],
    "Vegetation Index": [
        "NDVI", "EVI", "LAI", "FPAR", "NDWI", "NDRE", "SAVI", "MSAVI",
        "NBR", "NBR2",
    ],
    "Vegetation": [
        "VEG", "COVER", "GRASS", "SHRUB", "TREE", "FOREST", "CANOPY",
        "HERB", "FORB", "LITTER", "ORGANIC",
        "ExoticAnnualGrass", "PoaSecunda", "Medusahead",
    ],
    "Terrain": [
        "ELEV", "SLOPE", "ASPECT", "TPI", "TRI", "VRM", "HEAT", "ROUGHNESS",
        "elevation", "terrain",
    ],
    "Topography": [
        "DEM", "SRTM", "CURVATURE", "FLOW",
    ],
    "Hydrology": [
        "SOIL_MOISTURE", "SWE", "RUNOFF", "STREAM", "WETLAND", "LAKE",
        "RIVER", "FLOOD", "DROUGHT", "PDSI", "SPI", "SPEI",
    ],
    "Land Cover": [
        "NLCD", "LANDCOVER", "LAND_COVER", "LULC", "IMPERV", "IMPERVIOUS",
        "URBAN", "CROP", "PASTURE", "BARE", "WATER_BODY",
    ],
    "Population": [
        "POP", "POPDEN", "POPULATION", "CENSUS", "HOUSEHOLD", "DENSITY",
    ],
    "Infrastructure": [
        "road", "ROAD", "HIGHWAY", "INTERSTATE", "RAIL", "POWER", "GAS",
        "PIPELINE", "UTILITY", "BUILDING",
    ],
    "Fire Stations": [
        "FireStation", "FIRE_STATION", "STATION", "FIREHOUSE",
    ],
    "Administrative": [
        "COUNTY", "STATE", "OWNER", "ADMIN", "UNIT", "AGENCY", "DISTRICT",
        "GACC", "FIPS", "CONGRESSIONAL", "JURISDICTION",
    ],
    "Geographic": [
        "LATITUDE", "LONGITUDE", "GEOM", "GEOMETRY", "COORD", "LON", "LAT",
        "ECOREGION", "ECO3", "ECO4", "REGION", "ZONE",
    ],
    "Climate": [
        "CLIMATE", "ZONE", "KOPPEN", "DRYNESS", "ARIDITY", "PET",
    ],
    "Environmental Justice": [
        "EJ", "MINORITY", "LOW_INCOME", "VULNER", "JUSTICE", "EQUITY",
        "PARTICUL", "OZONE", "CANCER",
    ],
    "Socioeconomic": [
        "INCOME", "POVERTY", "UNEMPLOYMENT", "EDUCATION", "MEDIAN_AGE",
        "GDPPC", "HDI",
    ],
    "Historical Fire": [
        "HIST_FIRE", "FIRE_HIST", "PREV_FIRE", "FIRE_FREQ", "BURN_AREA",
        "FIRE_RETURN", "FIRE_PROB",
    ],
    "Temporal": [
        "DATE", "DOY", "MONTH", "YEAR", "SEASON", "HOUR", "TIME", "WEEK",
    ],
    "Discovery": [
        "DISCOVERY",
    ],
    "Cause": [
        "CAUSE", "IGNIT", "ORIGIN", "HUMAN", "LIGHTNING", "ARSON",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Leakage Keywords
# ─────────────────────────────────────────────────────────────────────────────
LEAKAGE_DEFINITE_KEYWORDS = [
    "CONT_DATE", "CONT_DOY", "CONT_TIME",          # containment
    "MTBS_ID", "MTBS_FIRE_NAME",                   # post-fire satellite mapping
    "ICS_209_PLUS",                                 # post-incident command system
    "FIRE_SIZE", "FIRE_SIZE_CLASS",                 # outcome — known only after fire
    "FIRE_MAG",                                     # outcome
    "DURATION",                                     # computed after containment
]

LEAKAGE_POSSIBLE_KEYWORDS = [
    "FIRE_CODE", "LOCAL_FIRE_REPORT_ID", "COMPLEX_NAME",
    "FIRE_NAME",                                    # sometimes assigned after detection
    "DISCOVERY_TIME", "DISCOVERY_DOY",             # known at detection
    "REPORT_DATE",
]

# ─────────────────────────────────────────────────────────────────────────────
# Predictive Readiness Tags
# ─────────────────────────────────────────────────────────────────────────────
# Labels used in Analysis 12
READINESS_CANDIDATE     = "Candidate Feature"
READINESS_REVIEW        = "Review Later"
READINESS_ADMINISTRATIVE = "Administrative"
READINESS_LIKELY_REMOVE = "Likely Remove"

# ─────────────────────────────────────────────────────────────────────────────
# Source Update Frequency Tags (Analysis 14)
# ─────────────────────────────────────────────────────────────────────────────
SOURCE_STATIC       = "Static"
SOURCE_DAILY        = "Daily"
SOURCE_HOURLY       = "Hourly"
SOURCE_MONTHLY      = "Monthly"
SOURCE_EVENT        = "Event Based"
SOURCE_ADMIN        = "Administrative"
SOURCE_UNKNOWN      = "Unknown"

# ─────────────────────────────────────────────────────────────────────────────
# Plot Styling
# ─────────────────────────────────────────────────────────────────────────────
MATPLOTLIB_STYLE    = "seaborn-v0_8-whitegrid"
FIGURE_DPI          = 150
FIGURE_SIZE_WIDE    = (18, 8)
FIGURE_SIZE_SQUARE  = (12, 12)
FIGURE_SIZE_TALL    = (12, 20)
HEATMAP_MAX_COLS    = 60    # clip correlation matrix for readability

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = "DEBUG"
LOG_FILE  = LOGS_DIR / "analysis.log"

# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
FINAL_REPORT_MD  = REPORTS_DIR / "final_summary_report.md"
FINAL_REPORT_PDF = REPORTS_DIR / "final_summary_report.pdf"
