"""
config/phase2_config.py
-----------------------
Central configuration for Phase 2A — Feature Finalization Pipeline.

Reads outputs from Phase 1 (dataset_analysis/) and produces
the master feature_schema.csv for each state.
"""

from __future__ import annotations
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Root Paths
# ─────────────────────────────────────────────────────────────────────────────
PHASE2_ROOT = Path(__file__).resolve().parents[1]   # V2/phase2/
V2_ROOT     = PHASE2_ROOT.parent                    # V2/

# Phase 1 analysis outputs (read-only inputs)
PHASE1_ROOT       = V2_ROOT / "dataset_analysis"
PHASE1_TABLES_TX  = PHASE1_ROOT / "tables" / "texas"
PHASE1_TABLES_CA  = PHASE1_ROOT / "tables" / "california"

# Original processed datasets (read-only)
PROCESSED_DIR = V2_ROOT / "data" / "processed"
TX_PARQUET    = PROCESSED_DIR / "texas"      / "texas_fire_2014_2020.parquet"
CA_PARQUET    = PROCESSED_DIR / "california" / "california_fire_2014_2020.parquet"

# Phase 2A outputs
OUTPUTS_ROOT = PHASE2_ROOT / "outputs"
OUTPUTS_TX   = OUTPUTS_ROOT / "texas"
OUTPUTS_CA   = OUTPUTS_ROOT / "california"

# Logs
LOGS_DIR = PHASE2_ROOT / "logs"
LOG_FILE = LOGS_DIR / "phase2a.log"

# ─────────────────────────────────────────────────────────────────────────────
# State Configuration
# ─────────────────────────────────────────────────────────────────────────────
STATE_CONFIG = {
    "TX": {
        "name":        "Texas",
        "slug":        "texas",
        "parquet":     TX_PARQUET,
        "tables_dir":  PHASE1_TABLES_TX,
        "output_dir":  OUTPUTS_TX,
        "h3_level":    8,            # H3-R8 (~0.73 km edge, ~0.74 km²) — same as CA per scope doc
        "grid_cells_est": 317_142,   # confirmed from actual dataset (317,142 unique cells)
    },
    "CA": {
        "name":        "California",
        "slug":        "california",
        "parquet":     CA_PARQUET,
        "tables_dir":  PHASE1_TABLES_CA,
        "output_dir":  OUTPUTS_CA,
        "h3_level":    8,            # H3-R8 (~0.73 km edge, ~0.74 km²)
        "grid_cells_est": 345_300,   # estimated burnable cells
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Gate 1 — Definite Removal Rules (Production Feasibility)
# ─────────────────────────────────────────────────────────────────────────────

# Columns that are definitively post-fire — cannot exist at prediction time
DEFINITE_LEAKAGE_COLS = [
    "ICS_209_PLUS_INCIDENT_JOIN_ID",
    "ICS_209_PLUS_COMPLEX_JOIN_ID",
    "MTBS_ID",
    "MTBS_FIRE_NAME",
    "CONT_DATE",
    "CONT_DOY",
    "CONT_TIME",
    "FIRE_SIZE",
    "FIRE_SIZE_CLASS",
]

# Columns that are administrative identifiers — never used as model features
ADMIN_ID_COLS = [
    "FOD_ID",
    "FPA_ID",
    "LOCAL_FIRE_REPORT_ID",
    "LOCAL_INCIDENT_ID",
    "FIRE_CODE",
    "FIRE_NAME",
    "COMPLEX_NAME",
    "NWCG_REPORTING_UNIT_ID",
    "NWCG_REPORTING_UNIT_NAME",
    "SOURCE_REPORTING_UNIT",
    "SOURCE_REPORTING_UNIT_NAME",
    "FIPS_NAME",
]

# Columns that are structural artifacts (format/join artifacts, never real data)
STRUCTURAL_ARTIFACT_COLS = [
    "geometry",        # GeoDataFrame artifact stored as text
    "source_year",     # Duplicate of FIRE_YEAR from join artifact
]

# Confirmed corrupted columns
CORRUPTED_COLS = [
    "SDI",             # Values of 10^34 — numerical overflow during feature engineering
]

# ─────────────────────────────────────────────────────────────────────────────
# Gate 2 — Confirmed Duplicate Pairs (Correlation r = 1.000)
# These were confirmed in both TX and CA correlation analysis
# ─────────────────────────────────────────────────────────────────────────────
# Format: (keep, drop) — keep the more physically interpretable name
EXACT_DUPLICATE_PAIRS = [
    ("FIRE_YEAR",    "source_year"),      # source_year is join artifact
    ("WDLI",         "M_WTR"),            # WDLI = water deficit, more standard
    ("WD_ET",        "M_WTR_EOMI"),       # WD_ET = water deficit ET
    ("SM_C",         "SM_PFS"),           # SM_C = soil moisture, keep primary
    ("M_WKFC_105",   "LHE"),              # Keep M_WKFC_105 (field capacity)
]

# Near-duplicate groups (r > 0.999 from same source — keep composite, drop subspecies)
NEAR_DUPLICATE_GROUPS = {
    # Rangeland Analysis Platform — highly overlapping invasive grass species
    # These are virtually identical in Texas/California fire zones
    "invasive_grass": {
        "keep":  "ExoticAnnualGrass",    # broadest composite
        "drop":  ["CheatGrass", "Medusahead", "PoaSecunda"],
    },
    # SVI composite vs sub-themes (r > 0.999 pairs)
    # Keep RPL_THEMES (overall composite) and one representative per domain
    "svi_composites": {
        "keep":  ["RPL_THEMES", "RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"],
        "drop":  [
            # Socioeconomic (THEME1) — keep RPL_THEME1, drop sub-components
            "EPL_POV", "EPL_UNEMP", "EPL_NOHSDP", "EPL_PCI",
            # Household/Disability (THEME2) — keep RPL_THEME2, drop sub-components
            "EPL_AGE65", "EPL_AGE17", "EPL_DISABL", "EPL_SNGPNT", "EPL_LIMENG",
            # Minority/Language (THEME3)
            "EPL_MINRTY", "EPL_LIMENG",
            # Housing/Transportation (THEME4)
            "EPL_MUNIT", "EPL_MOBILE", "EPL_CROWD", "EPL_NOVEH", "EPL_GROUPQ",
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature Source Mapping — Keyword → Source System
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: keywords (lowercase match), source_system, api_url,
#             spatial_resolution, temporal_resolution, update_frequency

SOURCE_KEYWORD_MAP = [
    # ── GridMET daily weather ─────────────────────────────────────────────────
    {
        "keywords":   ["erc", "bi", "sc", "ic", "fm100", "fm1000", "tmmx", "tmmn",
                       "vs", "sph", "rmin", "rmax", "vpd", "pet", "aet",
                       "pr_", "pr5d", "pr_5d", "erc_5d", "bi_5d", "fm100_5d",
                       "erc_normal", "bi_normal", "fm100_normal", "fm1000_normal",
                       "erc_percentile", "bi_percentile", "fm100_percentile",
                       "fm1000_percentile", "sph_percentile", "vs_percentile",
                       "tmmx_percentile", "tmmn_percentile"],
        "source":     "GridMET",
        "api_url":    "https://www.climatologylab.org/wget-gridmet.html",
        "spatial_res":"4 km",
        "temporal_res":"Daily",
        "update_freq":"Daily",
    },
    # ── MODIS Vegetation Indices ──────────────────────────────────────────────
    {
        "keywords":   ["ndvi", "evi", "mod_ndvi", "mod_evi", "ndvi_max",
                       "ndvi_min", "ndvi_mean", "lswi", "nbr"],
        "source":     "MODIS MOD13A1 (Terra)",
        "api_url":    "https://appeears.earthdatacloud.nasa.gov/",
        "spatial_res":"500 m",
        "temporal_res":"16-day composite",
        "update_freq":"16-day",
    },
    # ── LANDFIRE ──────────────────────────────────────────────────────────────
    {
        "keywords":   ["evt", "evc", "evh", "frg", "fbfm", "cbd", "cbh",
                       "evt_1km", "evc_1km", "evh_1km", "frg_1km",
                       "land_cover", "land_cover_1km"],
        "source":     "LANDFIRE",
        "api_url":    "https://www.landfire.gov/viewer/",
        "spatial_res":"30 m → 1 km aggregate in FPA-FOD",
        "temporal_res":"~2-year release cycle",
        "update_freq":"~2 years",
    },
    # ── USGS 3DEP / SRTM Terrain ──────────────────────────────────────────────
    {
        "keywords":   ["elevation", "slope", "aspect", "tri", "tpi", "roughness",
                       "northness", "eastness", "elevation_1km", "slope_1km",
                       "aspect_1km", "tri_1km", "tpi_1km"],
        "source":     "USGS 3DEP / SRTM",
        "api_url":    "https://elevation.nationalmap.gov/arcgis/rest/services/",
        "spatial_res":"30 m → 1 km aggregate",
        "temporal_res":"Static",
        "update_freq":"Static",
    },
    # ── CDC Social Vulnerability Index ────────────────────────────────────────
    {
        "keywords":   ["epl_", "rpl_", "spl_", "f_", "ep_", "e_pov", "e_unemp",
                       "rpl_theme"],
        "source":     "CDC Social Vulnerability Index (SVI)",
        "api_url":    "https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html",
        "spatial_res":"Census tract",
        "temporal_res":"Static (ACS 5-year)",
        "update_freq":"5 years",
    },
    # ── EPA EJScreen ──────────────────────────────────────────────────────────
    {
        "keywords":   ["ia_lmi", "ia_un", "ia_pov", "iaulhse", "iaplhse",
                       "ialmilhse", "ialmil_87", "iaplhs_88", "iaulhs_89",
                       "ialhe", "iahsef", "lilhse", "liso_et", "tsdf_et",
                       "hwli", "ui_exp", "thrhld"],
        "source":     "EPA EJScreen",
        "api_url":    "https://www.epa.gov/ejscreen/download-ejscreen-data",
        "spatial_res":"Census block group",
        "temporal_res":"Static",
        "update_freq":"Annual",
    },
    # ── Road Network (TIGER/OSM) ──────────────────────────────────────────────
    {
        "keywords":   ["road_interstate_dis", "road_us_dis", "road_state_dis",
                       "road_county_dis", "road_other_dis", "road_common_name_dis"],
        "source":     "US Census TIGER Road Network",
        "api_url":    "https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html",
        "spatial_res":"Vector (computed distance)",
        "temporal_res":"Static",
        "update_freq":"Annual",
    },
    # ── Fire Stations (HSIP) ─────────────────────────────────────────────────
    {
        "keywords":   ["no_firestation", "firestation"],
        "source":     "HSIP / TIGER Fire Stations",
        "api_url":    "https://geoplatform.gov/metadata/",
        "spatial_res":"Point (count within radius)",
        "temporal_res":"Static",
        "update_freq":"Annual",
    },
    # ── Population (Census / LandScan) ───────────────────────────────────────
    {
        "keywords":   ["population", "popo_1km", "popo", "popden"],
        "source":     "US Census ACS / LandScan",
        "api_url":    "https://www.census.gov/data/developers/data-sets/acs-5year.html",
        "spatial_res":"Census tract / 1 km grid",
        "temporal_res":"Static (ACS 5-year)",
        "update_freq":"5 years",
    },
    # ── Rangeland Analysis Platform ──────────────────────────────────────────
    {
        "keywords":   ["cheatgrass", "medusahead", "poasecunda", "exoticannualgr",
                       "annual_forb", "perennial_grass", "shrub", "tree_cover"],
        "source":     "Rangeland Analysis Platform (RAP)",
        "api_url":    "https://rangelands.app/products/",
        "spatial_res":"30 m",
        "temporal_res":"Annual",
        "update_freq":"Annual",
    },
    # ── Water Budget / ET (NLDAS / TerraClimate) ─────────────────────────────
    {
        "keywords":   ["m_wtr", "wdli", "wd_et", "m_wtr_eomi", "m_wkfc",
                       "sm_c", "sm_pfs", "lhe", "tc", "cc", "ca", "nca",
                       "ws_", "wue", "ro_"],
        "source":     "NLDAS / TerraClimate (Water Budget)",
        "api_url":    "https://ldas.gsfc.nasa.gov/nldas/",
        "spatial_res":"4 km / 0.125°",
        "temporal_res":"Monthly",
        "update_freq":"Monthly",
    },
    # ── Drought Indices ───────────────────────────────────────────────────────
    {
        "keywords":   ["pdsi", "spi", "spei", "drought", "aridity_index",
                       "annual_precipitation"],
        "source":     "NOAA CPC / TerraClimate",
        "api_url":    "https://www.climatologylab.org/terraclimate.html",
        "spatial_res":"0.5° / 4 km",
        "temporal_res":"Monthly",
        "update_freq":"Monthly",
    },
    # ── Ecoregion / GACC ─────────────────────────────────────────────────────
    {
        "keywords":   ["ecoregion", "gacc", "nwcg_gacc"],
        "source":     "EPA Level-3 Ecoregions / NIFC GACC",
        "api_url":    "https://www.epa.gov/eco-research/ecoregions",
        "spatial_res":"Vector polygon",
        "temporal_res":"Static",
        "update_freq":"Static",
    },
    # ── FPA-FOD Administrative / Temporal ────────────────────────────────────
    {
        "keywords":   ["fod_id", "fpa_id", "source_system", "nwcg_reporting",
                       "fire_year", "discovery_date", "discovery_doy",
                       "discovery_time", "stat_cause", "owner", "county",
                       "fips", "latlong", "latitude", "longitude",
                       "nwcg_cause"],
        "source":     "FPA-FOD v6 (Administrative/Temporal)",
        "api_url":    "https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009",
        "spatial_res":"Point record",
        "temporal_res":"Event-based",
        "update_freq":"Event-based",
    },
    # ── FPA-FOD Post-Fire Outcomes ────────────────────────────────────────────
    {
        "keywords":   ["fire_size", "fire_size_class", "cont_date", "cont_doy",
                       "cont_time", "mtbs_id", "mtbs_fire_name",
                       "ics_209_plus", "complex_name", "evacuation"],
        "source":     "FPA-FOD v6 (Post-fire outcome — LEAKAGE)",
        "api_url":    "N/A — post-fire only",
        "spatial_res":"N/A",
        "temporal_res":"Post-fire",
        "update_freq":"Event-based",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Missing Root Cause Rules
# ─────────────────────────────────────────────────────────────────────────────
MISSING_ROOT_CAUSE_RULES = {
    # Structural — format artifact
    "structural": [
        "geometry",         # GeoDataFrame column stored as text in tabular export
    ],
    # Event-conditional — only populated for qualifying events
    "event_conditional": [
        "MTBS_ID", "MTBS_FIRE_NAME",          # Only fires ≥ 1,000 acres
        "ICS_209_PLUS_INCIDENT_JOIN_ID",       # Only Type 1/2 incidents
        "ICS_209_PLUS_COMPLEX_JOIN_ID",        # Only multi-fire complexes
        "COMPLEX_NAME",                        # Only multi-fire complexes
        "CONT_DATE", "CONT_DOY", "CONT_TIME",  # Only formally contained fires
        "Evacuation",                          # Only major incidents
        "GACC_Fire Use Teams",                 # Only GACC-managed large fires
        "NWCG_CAUSE_AGE_CATEGORY",             # Only when age/cause known
    ],
    # Geographic coverage — spatial join found nothing within search radius
    "geographic_coverage": [
        "road_interstate_dis",  # No interstate within search radius
        "road_US_dis",          # No US highway within search radius
        "road_state_dis",       # No state road within search radius
        "road_county_dis",      # No county road within search radius
        "road_other_dis",       # No other road within search radius
        "No_FireStation_1.0km", # No fire station within 1 km (rural areas)
    ],
    # Administrative — agency reporting varies; not all agencies submit
    "administrative": [
        "LOCAL_FIRE_REPORT_ID",  # Many agencies don't assign/report
        "FIRE_CODE",             # Only federal/large fires use codes
        "LOCAL_INCIDENT_ID",     # Agency-specific; not universal
    ],
}

# Treatment mapping per root cause
MISSING_TREATMENT_MAP = {
    "structural":           "EXCLUDE — format artifact, never a real feature",
    "event_conditional":    "BINARY_FLAG or EXCLUDE — missing means 'small fire', informative",
    "geographic_coverage":  "SENTINEL_FILL — 999 means 'no such infrastructure nearby'",
    "administrative":       "EXCLUDE — administrative ID, not a predictive feature",
    "sensor_gap":           "INTERPOLATE or GRIDMET_FALLBACK",
    "unknown":              "REVIEW — investigate before deciding",
}

# ─────────────────────────────────────────────────────────────────────────────
# Production Availability Rules
# ─────────────────────────────────────────────────────────────────────────────
AVAILABILITY_RULES = {
    "POST_FIRE": [
        # These only exist after a fire is discovered/contained
        "FIRE_SIZE", "FIRE_SIZE_CLASS",
        "CONT_DATE", "CONT_DOY", "CONT_TIME",
        "MTBS_ID", "MTBS_FIRE_NAME",
        "ICS_209_PLUS_INCIDENT_JOIN_ID", "ICS_209_PLUS_COMPLEX_JOIN_ID",
        "COMPLEX_NAME", "Evacuation",
        "GACC_Fire Use Teams",
        "geometry",          # post-processing artifact
    ],
    "PRE_FIRE_STATIC": [
        # Available always — static or slow-changing
        "LATITUDE", "LONGITUDE",
        "Elevation", "Slope", "Aspect", "Elevation_1km", "Slope_1km", "Aspect_1km",
        "TRI_1km", "TPI_1km",
        "Population", "Popo_1km",
        "No_FireStation_5.0km", "No_FireStation_10.0km", "No_FireStation_20.0km",
        "road_common_name_dis",
        "ExoticAnnualGrass",
    ],
    "PRE_FIRE_DYNAMIC": [
        # Available pre-fire but updated regularly
        "bi", "erc", "fm100", "fm1000",
        "bi_5D_mean", "erc_5D_mean", "bi_5D_max", "erc_5D_max",
        "bi_Percentile", "erc_Percentile", "fm100_Percentile",
        "sph_Percentile", "vs_Percentile",
        "FIRE_YEAR",
    ],
    "SAME_DAY_RISK": [
        # Available same day but may include same-day fire-influenced observations
        "DISCOVERY_DOY",    # day of fire — not before fire
        "DISCOVERY_DATE",   # date of fire — not before fire
        "DISCOVERY_TIME",   # time of fire — not before fire
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Static vs Dynamic Classification
# ─────────────────────────────────────────────────────────────────────────────
STATIC_DYNAMIC_RULES = {
    "STATIC": {
        "description": "Download once; refresh every 2–5 years",
        "keywords": ["elevation", "slope", "aspect", "tri", "tpi",
                     "evt", "evc", "evh", "frg", "land_cover",
                     "road_", "firestation", "ecoregion", "gacc",
                     "epl_", "rpl_", "population", "popo",
                     "exoticannualgr", "cheatgrass", "medusahead",
                     "poasecunda"],
    },
    "ANNUAL": {
        "description": "Update once per year",
        "keywords": ["rangeland", "rap_", "nlcd", "landfire"],
    },
    "MONTHLY": {
        "description": "Update monthly",
        "keywords": ["pdsi", "spi", "spei", "drought", "m_wtr", "wdli",
                     "wd_et", "sm_c", "sm_pfs", "tc", "aridity",
                     "annual_prec"],
    },
    "DAILY": {
        "description": "Download every day for operational prediction",
        "keywords": ["erc", "bi", "fm100", "fm1000", "tmmx", "tmmn",
                     "vs", "sph", "pr_", "vpd", "rmin", "rmax",
                     "ndvi", "evi", "_5d_", "_7d_", "_percentile",
                     "_normal", "kbdi"],
    },
    "EVENT_BASED": {
        "description": "Only exists after a fire event — not for production",
        "keywords": ["fire_size", "cont_date", "mtbs", "ics_209",
                     "evacuation", "complex_name"],
    },
}
