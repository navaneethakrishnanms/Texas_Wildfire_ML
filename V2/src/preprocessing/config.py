"""
config.py
---------
Central configuration for the Phase-1 preprocessing pipeline.

All paths, constants, and validation rules live here so that the rest of
the code stays free of hard-coded literals.  Adding a new year is as simple
as dropping the CSV into DATA_RAW_DIR – the loader discovers it automatically.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Project Root  (this file lives in  V2/src/preprocessing/)
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # → V2/

# ─────────────────────────────────────────────────────────────
# Directory Layout
# ─────────────────────────────────────────────────────────────
DATA_RAW_DIR        = PROJECT_ROOT / "data"
DATA_PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"

TEXAS_DIR       = DATA_PROCESSED_DIR / "texas"
CALIFORNIA_DIR  = DATA_PROCESSED_DIR / "california"

REPORTS_DIR         = PROJECT_ROOT / "reports"
TEXAS_REPORTS_DIR   = REPORTS_DIR / "texas"
CALI_REPORTS_DIR    = REPORTS_DIR / "california"

LOGS_DIR = PROJECT_ROOT / "logs"

# ─────────────────────────────────────────────────────────────
# Raw Data File Pattern
# ─────────────────────────────────────────────────────────────
# Matches files like  2014_FPA_FOD_cons.csv  (or .xlsx)
RAW_FILE_GLOB = "*_FPA_FOD_cons.csv"
RAW_FILE_GLOB_XLSX = "*_FPA_FOD_cons.xlsx"

YEAR_RANGE = range(2014, 2021)   # 2014 → 2020 inclusive

# ─────────────────────────────────────────────────────────────
# State Configuration
# ─────────────────────────────────────────────────────────────
TARGET_STATES = {
    "TX": {
        "name": "Texas",
        "out_dir": TEXAS_DIR,
        "report_dir": TEXAS_REPORTS_DIR,
        "parquet_out": TEXAS_DIR / "texas_fire_2014_2020.parquet",
        "csv_out":     TEXAS_DIR / "texas_fire_2014_2020.csv",
    },
    "CA": {
        "name": "California",
        "out_dir": CALIFORNIA_DIR,
        "report_dir": CALI_REPORTS_DIR,
        "parquet_out": CALIFORNIA_DIR / "california_fire_2014_2020.parquet",
        "csv_out":     CALIFORNIA_DIR / "california_fire_2014_2020.csv",
    },
}

# ─────────────────────────────────────────────────────────────
# Key Column Names  (exactly as they appear after standardisation)
# ─────────────────────────────────────────────────────────────
COL_STATE          = "STATE"
COL_COUNTY         = "COUNTY"
COL_LATITUDE       = "LATITUDE"
COL_LONGITUDE      = "LONGITUDE"
COL_DISCOVERY_DATE = "DISCOVERY_DATE"
COL_FIRE_SIZE      = "FIRE_SIZE"
COL_FIRE_YEAR      = "FIRE_YEAR"
COL_FIRE_NAME      = "FIRE_NAME"
COL_CAUSE          = "NWCG_GENERAL_CAUSE"
COL_AGENCY         = "NWCG_REPORTING_AGENCY"
COL_OWNER          = "OWNER_DESCR"
COL_SOURCE_SYSTEM  = "SOURCE_SYSTEM"

# ─────────────────────────────────────────────────────────────
# Validation Bounds
# ─────────────────────────────────────────────────────────────
VALID_LAT_RANGE  = (-90.0,   90.0)
VALID_LON_RANGE  = (-180.0, 180.0)
VALID_FIRE_SIZE  = (0.0, 1_000_000.0)   # acres
VALID_YEARS      = list(YEAR_RANGE)

# US State abbreviations (for quick validation)
VALID_US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY","DC","PR","VI","GU","AS","MP",
}

# ─────────────────────────────────────────────────────────────
# EDA / Report Settings
# ─────────────────────────────────────────────────────────────
CORR_HIGH_THRESHOLD   = 0.95   # pairs above this are flagged as highly correlated
TOP_N_COUNTIES        = 20
TOP_N_AGENCIES        = 15
CATEGORICAL_HIGH_CARD = 50     # categories above this are flagged as high-cardinality

# Columns likely to be leakage in a predictive model
LEAKAGE_KEYWORDS = [
    "CONT_DATE", "CONT_DOY", "CONT_TIME",          # containment info
    "MTBS_ID", "MTBS_FIRE_NAME",                   # post-fire mapping IDs
    "ICS_209_PLUS",                                 # post-incident system IDs
]
