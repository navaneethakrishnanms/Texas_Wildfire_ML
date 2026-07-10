"""
run_phase2e_schema_fix.py
--------------------------
Phase 2E-A — Feature Schema Cleanup

Reads the current 251-column feature_schema.csv and:
  1. REMOVES all leakage/event-based/unusable columns
  2. ADDS the 4 mandatory LANDFIRE/FSim columns that are missing
  3. ADDS temporal encodings and H3 location features
  4. PRODUCES: cleaned_feature_schema.csv — the safe contract for data collection

Run this BEFORE starting any data collection.

Usage:
    conda activate torch_gpu
    python run_phase2e_schema_fix.py --state TX
    python run_phase2e_schema_fix.py --state ALL

Output:
    phase2/outputs/<state>/cleaned_feature_schema.csv
    phase2/outputs/<state>/schema_fix_report.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PHASE2_ROOT = Path(__file__).resolve().parent
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from config.phase2_config import STATE_CONFIG, LOGS_DIR

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMNS TO REMOVE — with exact reason codes
# ══════════════════════════════════════════════════════════════════════════════

# CATEGORY 1: Event-based leakage (exist only for fire records, not H3 cells)
REMOVE_EVENT_BASED = [
    "COUNTY",                    # from fire record only
    "FIPS_CODE",                 # from fire record, 51% missing
    "FIRE_YEAR",                 # USE FOR SPLIT ONLY — not a model feature
    "LatLong_County",            # fire record spatial field
    "LatLong_State",             # fire record spatial field
    "LATITUDE",                  # use H3 centroid lat (centroid_lat) instead
    "LONGITUDE",                 # use H3 centroid lon (centroid_lon) instead
    "NWCG_CAUSE_AGE_CATEGORY",   # 99.4% missing, post-investigation
    "NWCG_CAUSE_CLASSIFICATION", # POST-FIRE LEAKAGE — cause assigned weeks after fire
    "NWCG_GENERAL_CAUSE",        # POST-FIRE LEAKAGE — same issue
    "NWCG_REPORTING_AGENCY",     # fire record metadata only
    "OWNER_DESCR",               # recorded at fire event only
    "SOURCE_SYSTEM",             # fire record metadata
    "SOURCE_SYSTEM_TYPE",        # fire record metadata
]

# CATEGORY 2: 100% missing — no data exists
REMOVE_ALL_MISSING = [
    "Evacuation",                # 98.5% missing — effectively all missing
    "GACC_Fire Use Teams",       # 100% missing
    "IAHSEF",                    # 100% missing
    "IALMIL_87",                 # 100% missing
    "IAPLHS_88",                 # 100% missing
    "IAULHS_89",                 # 100% missing
]

# CATEGORY 3: Road distances with >98% missing — not collectible for non-fire cells
REMOVE_HIGH_MISSING_ROADS = [
    "road_US_dis",               # 99.8% missing
    "road_interstate_dis",       # 99.9% missing
    "road_county_dis",           # 98.1% missing
    "road_other_dis",            # 99.5% missing
    "road_state_dis",            # 98.5% missing
    # Note: road_common_name_dis (34% missing) → KEEP
    # Note: No_FireStation_10km/20km → KEEP
]

# CATEGORY 4: GACC operational fields — categorical labels with no ML value
REMOVE_GACC_OPERATIONAL = [
    "GACC_Area Command Teams",
    "GACC_NIMO Teams",
    "GACC_New LF",
    "GACC_New fire",
    "GACC_Type 1 IMTs",
    "GACC_Type 2 IMTs",
    "GACC_Uncont LF",
    "GACC_PL",
    # Note: GACCAbbrev → KEEP (useful ecoregion proxy)
]

ALL_REMOVE = list(set(
    REMOVE_EVENT_BASED
    + REMOVE_ALL_MISSING
    + REMOVE_HIGH_MISSING_ROADS
    + REMOVE_GACC_OPERATIONAL
))

# ══════════════════════════════════════════════════════════════════════════════
# MANDATORY COLUMNS TO ADD
# ══════════════════════════════════════════════════════════════════════════════

ADD_LANDFIRE_FSIM = [
    {
        "Column": "avg_burn_prob",
        "Source_System": "USFS FSim (50,000 stochastic fire simulations)",
        "API_URL": "https://www.firelab.org/project/wildfire-hazard-potential",
        "Spatial_Res": "270m raster",
        "Temporal_Res": "Static",
        "Update_Category": "STATIC",
        "Availability_Label": "PRE_FIRE_STATIC",
        "Missing_%": 0.0,
        "Missing_Treatment": "zero-fill (0 = never burned in simulation)",
        "Gate1_Status": "RETAINED",
        "Gate2_Status": "RETAINED",
        "Notes": "MANDATORY — strongest single landscape predictor (Cohen's d >> 1.0)",
    },
    {
        "Column": "whp",
        "Source_System": "USFS Wildfire Hazard Potential",
        "API_URL": "https://www.firelab.org/project/wildfire-hazard-potential",
        "Spatial_Res": "270m raster",
        "Temporal_Res": "Static",
        "Update_Category": "STATIC",
        "Availability_Label": "PRE_FIRE_STATIC",
        "Missing_%": 0.0,
        "Missing_Treatment": "zero-fill",
        "Gate1_Status": "RETAINED",
        "Gate2_Status": "RETAINED",
        "Notes": "MANDATORY — Wildfire Hazard Potential index 0-7000",
    },
    {
        "Column": "flep4",
        "Source_System": "LANDFIRE LF2022",
        "API_URL": "https://landfire.gov/viewer/",
        "Spatial_Res": "30m raster",
        "Temporal_Res": "Static (~2yr release)",
        "Update_Category": "STATIC",
        "Availability_Label": "PRE_FIRE_STATIC",
        "Missing_%": 0.0,
        "Missing_Treatment": "zero-fill",
        "Gate1_Status": "RETAINED",
        "Gate2_Status": "RETAINED",
        "Notes": "MANDATORY — Flame Length Exceedance Prob at 4ft (Cohen's d > 1.0 in team EDA)",
    },
    {
        "Column": "cfl",
        "Source_System": "LANDFIRE LF2022",
        "API_URL": "https://landfire.gov/viewer/",
        "Spatial_Res": "30m raster",
        "Temporal_Res": "Static (~2yr release)",
        "Update_Category": "STATIC",
        "Availability_Label": "PRE_FIRE_STATIC",
        "Missing_%": 0.0,
        "Missing_Treatment": "zero-fill",
        "Gate1_Status": "RETAINED",
        "Gate2_Status": "RETAINED",
        "Notes": "MANDATORY — Canopy Fuel Load Mg/ha (2nd strongest landscape feature)",
    },
]

ADD_TEMPORAL = [
    {
        "Column": "sin_month",
        "Source_System": "Computed from window_6h_utc timestamp",
        "API_URL": "N/A",
        "Spatial_Res": "N/A",
        "Temporal_Res": "Per-window",
        "Update_Category": "DAILY",
        "Availability_Label": "PRE_FIRE_DYNAMIC",
        "Missing_%": 0.0,
        "Missing_Treatment": "none",
        "Gate1_Status": "RETAINED",
        "Gate2_Status": "RETAINED",
        "Notes": "sin(2π × month / 12) — cyclic month encoding",
    },
    {
        "Column": "cos_month",
        "Source_System": "Computed",
        "API_URL": "N/A", "Spatial_Res": "N/A", "Temporal_Res": "Per-window",
        "Update_Category": "DAILY", "Availability_Label": "PRE_FIRE_DYNAMIC",
        "Missing_%": 0.0, "Missing_Treatment": "none",
        "Gate1_Status": "RETAINED", "Gate2_Status": "RETAINED",
        "Notes": "cos(2π × month / 12)",
    },
    {
        "Column": "sin_hour",
        "Source_System": "Computed",
        "API_URL": "N/A", "Spatial_Res": "N/A", "Temporal_Res": "Per-window",
        "Update_Category": "DAILY", "Availability_Label": "PRE_FIRE_DYNAMIC",
        "Missing_%": 0.0, "Missing_Treatment": "none",
        "Gate1_Status": "RETAINED", "Gate2_Status": "RETAINED",
        "Notes": "sin(2π × window_hour / 24) — cyclic 6hr window encoding",
    },
    {
        "Column": "cos_hour",
        "Source_System": "Computed",
        "API_URL": "N/A", "Spatial_Res": "N/A", "Temporal_Res": "Per-window",
        "Update_Category": "DAILY", "Availability_Label": "PRE_FIRE_DYNAMIC",
        "Missing_%": 0.0, "Missing_Treatment": "none",
        "Gate1_Status": "RETAINED", "Gate2_Status": "RETAINED",
        "Notes": "cos(2π × window_hour / 24)",
    },
]

ADD_LOCATION = [
    {
        "Column": "centroid_lat",
        "Source_System": "H3 centroid (h3.cell_to_latlng)",
        "API_URL": "N/A", "Spatial_Res": "H3-R8 cell centroid",
        "Temporal_Res": "Static", "Update_Category": "STATIC",
        "Availability_Label": "PRE_FIRE_STATIC",
        "Missing_%": 0.0, "Missing_Treatment": "none",
        "Gate1_Status": "RETAINED", "Gate2_Status": "RETAINED",
        "Notes": "H3 cell centroid latitude — NOT the fire lat/lon from FPA-FOD",
    },
    {
        "Column": "centroid_lon",
        "Source_System": "H3 centroid (h3.cell_to_latlng)",
        "API_URL": "N/A", "Spatial_Res": "H3-R8 cell centroid",
        "Temporal_Res": "Static", "Update_Category": "STATIC",
        "Availability_Label": "PRE_FIRE_STATIC",
        "Missing_%": 0.0, "Missing_Treatment": "none",
        "Gate1_Status": "RETAINED", "Gate2_Status": "RETAINED",
        "Notes": "H3 cell centroid longitude",
    },
]

ADD_HRRR_FUTURE = [
    {
        "Column": "rh_pw",
        "Source_System": "NOAA HRRR (AWS S3 via Herbie)",
        "API_URL": "s3://noaa-hrrr-bdp-pds",
        "Spatial_Res": "3km analysis grid", "Temporal_Res": "6-hourly",
        "Update_Category": "DAILY", "Availability_Label": "PRE_FIRE_DYNAMIC",
        "Missing_%": 24.4, "Missing_Treatment": "zero-fill + hrrr_available flag",
        "Gate1_Status": "RETAINED", "Gate2_Status": "RETAINED",
        "Notes": "Phase 2F — add AFTER daily baseline model is working",
    },
    {
        "Column": "temp_pw",
        "Source_System": "NOAA HRRR", "API_URL": "s3://noaa-hrrr-bdp-pds",
        "Spatial_Res": "3km", "Temporal_Res": "6-hourly",
        "Update_Category": "DAILY", "Availability_Label": "PRE_FIRE_DYNAMIC",
        "Missing_%": 24.4, "Missing_Treatment": "zero-fill + hrrr_available flag",
        "Gate1_Status": "RETAINED", "Gate2_Status": "RETAINED",
        "Notes": "Phase 2F",
    },
    {
        "Column": "wind_speed_pw",
        "Source_System": "NOAA HRRR (UGRD+VGRD → √(U²+V²))",
        "API_URL": "s3://noaa-hrrr-bdp-pds",
        "Spatial_Res": "3km", "Temporal_Res": "6-hourly",
        "Update_Category": "DAILY", "Availability_Label": "PRE_FIRE_DYNAMIC",
        "Missing_%": 24.4, "Missing_Treatment": "zero-fill + hrrr_available flag",
        "Gate1_Status": "RETAINED", "Gate2_Status": "RETAINED",
        "Notes": "Phase 2F",
    },
    {
        "Column": "vpd_pw",
        "Source_System": "NOAA HRRR (derived TMP+RH)",
        "API_URL": "s3://noaa-hrrr-bdp-pds",
        "Spatial_Res": "3km", "Temporal_Res": "6-hourly",
        "Update_Category": "DAILY", "Availability_Label": "PRE_FIRE_DYNAMIC",
        "Missing_%": 24.4, "Missing_Treatment": "zero-fill + hrrr_available flag",
        "Gate1_Status": "RETAINED", "Gate2_Status": "RETAINED",
        "Notes": "Phase 2F",
    },
    {
        "Column": "hpbl_pw",
        "Source_System": "NOAA HRRR", "API_URL": "s3://noaa-hrrr-bdp-pds",
        "Spatial_Res": "3km", "Temporal_Res": "6-hourly",
        "Update_Category": "DAILY", "Availability_Label": "PRE_FIRE_DYNAMIC",
        "Missing_%": 24.4, "Missing_Treatment": "zero-fill + hrrr_available flag",
        "Gate1_Status": "RETAINED", "Gate2_Status": "RETAINED",
        "Notes": "Phase 2F — Planetary Boundary Layer Height",
    },
    {
        "Column": "dswrf_pw",
        "Source_System": "NOAA HRRR", "API_URL": "s3://noaa-hrrr-bdp-pds",
        "Spatial_Res": "3km", "Temporal_Res": "6-hourly",
        "Update_Category": "DAILY", "Availability_Label": "PRE_FIRE_DYNAMIC",
        "Missing_%": 24.4, "Missing_Treatment": "zero-fill + hrrr_available flag",
        "Gate1_Status": "RETAINED", "Gate2_Status": "RETAINED",
        "Notes": "Phase 2F — Downwelling Solar Radiation",
    },
    {
        "Column": "hrrr_available",
        "Source_System": "Computed (1 if HRRR available for this year, else 0)",
        "API_URL": "N/A", "Spatial_Res": "N/A", "Temporal_Res": "Per-year",
        "Update_Category": "DAILY", "Availability_Label": "PRE_FIRE_DYNAMIC",
        "Missing_%": 0.0, "Missing_Treatment": "none — always 0 or 1",
        "Gate1_Status": "RETAINED", "Gate2_Status": "RETAINED",
        "Notes": "Phase 2F — binary flag: 2017+ = 1, 2014-2015 = 0, 2016 partial",
    },
]

# Combine all additions
ALL_ADD = ADD_LANDFIRE_FSIM + ADD_TEMPORAL + ADD_LOCATION + ADD_HRRR_FUTURE

# (ALL_REMOVE already defined above — no duplicate needed)


def fix_schema(state_key: str, cfg: dict) -> None:
    output_dir = cfg["output_dir"]
    schema_path = output_dir / "feature_schema.csv"

    logger.info(f"{'=' * 60}")
    logger.info(f"PHASE 2E-A — SCHEMA FIX [{cfg['name'].upper()}]")
    logger.info(f"{'=' * 60}")

    if not schema_path.exists():
        logger.error(f"feature_schema.csv not found: {schema_path}")
        logger.error("Run Phase 2A first.")
        return

    df = pd.read_csv(schema_path)
    logger.info(f"  Loaded schema: {len(df)} features")

    original_cols = set(df["Column"].tolist())

    # ── Remove bad columns ────────────────────────────────────────────────────
    removed = []
    actually_removed = []
    for col in ALL_REMOVE:
        if col in df["Column"].values:
            df = df[df["Column"] != col]
            actually_removed.append(col)
        removed.append(col)

    logger.info(f"  Removed {len(actually_removed)} columns (of {len(ALL_REMOVE)} targeted)")

    # ── Add mandatory missing columns ─────────────────────────────────────────
    added = []
    for col_info in ALL_ADD:
        if col_info["Column"] not in df["Column"].values:
            new_row = {c: col_info.get(c, "") for c in df.columns}
            new_row["Column"] = col_info["Column"]
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            added.append(col_info["Column"])

    logger.info(f"  Added {len(added)} mandatory columns: {added}")

    # ── Save cleaned schema ───────────────────────────────────────────────────
    cleaned_path = output_dir / "cleaned_feature_schema.csv"
    df.to_csv(cleaned_path, index=False)
    logger.info(f"\n  Cleaned schema: {len(df)} features -> {cleaned_path}")

    # ── Print category breakdown ───────────────────────────────────────────────
    if "Update_Category" in df.columns:
        cat_counts = df["Update_Category"].value_counts()
        logger.info(f"\n  Feature breakdown by update category:")
        for cat, cnt in cat_counts.items():
            logger.info(f"    {cat:<15} : {cnt:>4} features")

    # ── Generate markdown report ───────────────────────────────────────────────
    report_lines = [
        f"# Schema Fix Report — {cfg['name']}",
        "",
        "## Summary",
        f"| Stage | Count |",
        f"|---|---|",
        f"| Original (Phase 2A) | {len(original_cols)} |",
        f"| Removed (leakage + missing + operational) | {len(actually_removed)} |",
        f"| Added (mandatory + temporal + location + HRRR) | {len(added)} |",
        f"| **Final cleaned schema** | **{len(df)}** |",
        "",
        "## Removed Columns",
        "",
        "### Event-Based Leakage (FPA-FOD post-discovery fields)",
        "| Column | Reason |",
        "|---|---|",
    ]
    for col in REMOVE_EVENT_BASED:
        report_lines.append(f"| `{col}` | Post-discovery / fire record only |")

    report_lines += [
        "",
        "### Added Mandatory Features",
        "| Column | Source | Notes |",
        "|---|---|---|",
    ]
    for col_info in ALL_ADD:
        report_lines.append(
            f"| `{col_info['Column']}` | {col_info['Source_System'][:40]} | {col_info['Notes'][:60]} |"
        )

    report_path = output_dir / "schema_fix_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"  Report saved: {report_path}")

    logger.info(f"\n  [OK] Schema fix complete for {cfg['name']}")
    logger.info(f"  Use cleaned_feature_schema.csv for all downstream data collection")


def main():
    parser = argparse.ArgumentParser(description="Phase 2E-A — Feature Schema Cleanup")
    parser.add_argument("--state", choices=["TX", "CA", "ALL"], required=True)
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "phase2e_schema.log", encoding="utf-8"),
        ],
    )

    states = ["TX", "CA"] if args.state == "ALL" else [args.state]
    for s in states:
        fix_schema(s, STATE_CONFIG[s])


if __name__ == "__main__":
    main()
