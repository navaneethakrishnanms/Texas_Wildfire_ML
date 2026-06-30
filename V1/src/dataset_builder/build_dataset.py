"""
build_dataset.py
================
Production-grade orchestrator for the Texas Wildfire ML dataset construction pipeline.

╔══════════════════════════════════════════════════════════════════════╗
║              VERSION ARCHITECTURE: V1 → V2 UPGRADE PATH             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  VERSION 1 (current — this file)                                     ║
║  ─────────────────────────────────────────────────────────────────   ║
║  Environmental features: ANNUAL COMPOSITE rasters from GEE           ║
║    NDVI_2024.tif, EVI_2024.tif, LST_2024.tif, etc.                   ║
║  All pixels represent the yearly average — a fire on Jan 15 and      ║
║  Aug 20 at the same location get IDENTICAL raster values.            ║
║                                                                      ║
║  Compensation strategy: 7 temporal features derived from acq_date    ║
║  (month, day_of_year, season_code, sin_month, cos_month, sin_doy,    ║
║  cos_doy) teach the model seasonal patterns.                         ║
║                                                                      ║
║  is_peak_fire_season: EXPERIMENTAL feature (domain knowledge from    ║
║  Texas A&M Forest Service). Run with and without to validate via     ║
║  --no-peak-feature flag. Only retain if it improves AUC/F1.          ║
║                                                                      ║
║  Static layers (safe to keep as annual):                             ║
║    DEM, Slope, Aspect, LandCover — these change very little over     ║
║    time and annual composites are geospatially accurate for them.    ║
║                                                                      ║
║  VERSION 2 (future upgrade — see FutureGEEInterface below)           ║
║  ─────────────────────────────────────────────────────────────────   ║
║  Environmental features: DATE-SPECIFIC rasters queried from GEE      ║
║  For each FIRMS event (lat, lon, acq_date):                          ║
║    → Query GEE for NDVI/EVI/LST/Temperature/Wind/Rainfall            ║
║       within a ±N day window around acq_date                         ║
║    → Preserve DEM/Slope/Aspect/LandCover as static annual rasters    ║
║    → Same output schema → zero changes to model training code        ║
║                                                                      ║
║  The FutureGEEInterface class at the bottom of this file defines     ║
║  the exact API contract for the V2 extractor.                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

Pipeline Steps (V1)
-------------------
  1.  Load FIRMS 2024 fire detections          → positive samples (Fire = 1)
  2.  Generate spatially clean random points   → negative samples (Fire = 0)
  3.  Combine positives + negatives
  4.  Sample ALL raster features via windowed reads (correct CRS, auto-tiling)
  5.  Quality control: dedup, bounds check, NaN rows
  6.  Engineer 7 (+1 optional) temporal features from acq_date
  7.  Enforce output schema column order
  8.  Save wildfire_dataset.csv + wildfire_dataset.parquet
  9.  Chronological split: Jan–Aug=train | Sep=val | Oct–Dec=test
  10. Print comprehensive statistics

Output Schema (V1 — 22 columns)
--------------------------------
  latitude | longitude | acq_date
  NDVI | EVI | LST | Temperature | Wind | Rainfall       ← dynamic in V2
  DEM | Slope | Aspect | LandCover                       ← always static
  month | day_of_year | season_code
  sin_month | cos_month | sin_doy | cos_doy              ← always included
  is_peak_fire_season                                    ← OPTIONAL (--no-peak-feature)
  Fire

Usage
-----
  # Default run (includes is_peak_fire_season):
  python src/dataset_builder/build_dataset.py

  # Ablation: exclude is_peak_fire_season to validate its contribution:
  python src/dataset_builder/build_dataset.py --no-peak-feature

  # Full custom:
  python src/dataset_builder/build_dataset.py \\
      --firms data/raw/firms/fire_archive_2024.csv \\
      --neg-ratio 3 --exclusion-km 5.0 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Make the project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset_builder.sample_rasters import RasterSampler, FEATURE_COLUMNS
from src.dataset_builder.generate_negatives import (
    generate_negative_samples,
    negative_sample_spatial_stats,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------
RAW_DIR:   Path = Path("data/raw")
PROC_DIR:  Path = Path("data/processed")
FIRMS_CSV: Path = RAW_DIR / "firms" / "fire_archive_2024.csv"

NEG_POS_RATIO:   int   = 3     # 1:3  (pos:neg)
EXCLUSION_KM:    float = 5.0   # km around fire points excluded for negatives
RANDOM_SEED:     int   = 42

# Chronological split boundaries (calendar months, inclusive)
TRAIN_END_MONTH: int = 8   # Jan–Aug → train
VAL_MONTH:       int = 9   # Sep     → validation
# Oct–Dec → test

# Texas wildfire peak fire season months (Feb–Apr + Oct–Nov)
# Source: Texas A&M Forest Service historical fire data
PEAK_FIRE_MONTHS: set = {2, 3, 4, 10, 11}

# Season definitions (meteorological seasons for Northern Hemisphere)
# 0=Winter (Dec-Feb), 1=Spring (Mar-May), 2=Summer (Jun-Aug), 3=Fall (Sep-Nov)
SEASON_MAP: dict = {
    1: 0, 2: 0,               # Jan-Feb  → Winter
    3: 1, 4: 1, 5: 1,         # Mar-May  → Spring
    6: 2, 7: 2, 8: 2,         # Jun-Aug  → Summer
    9: 3, 10: 3, 11: 3,       # Sep-Nov  → Fall
    12: 0,                    # Dec      → Winter
}

# Core temporal features — ALWAYS included (derived from acq_date)
# These are scientifically sound, require no domain assumptions,
# and allow the model to learn seasonal fire risk from training data.
CORE_TEMPORAL_COLUMNS = [
    "month",            # Calendar month (1–12)
    "day_of_year",      # Day of year (1–366)
    "season_code",      # 0=Winter, 1=Spring, 2=Summer, 3=Fall
    "sin_month",        # Cyclical sin encoding of month
    "cos_month",        # Cyclical cos encoding of month
    "sin_doy",          # Cyclical sin encoding of day-of-year
    "cos_doy",          # Cyclical cos encoding of day-of-year
]

# Experimental feature — OPTIONAL (Texas A&M domain knowledge)
# Include by default. Use --no-peak-feature to exclude and compare models.
# Only retain in production if it consistently improves AUC/F1 over the
# model trained on CORE_TEMPORAL_COLUMNS alone.
PEAK_FEATURE_COL = "is_peak_fire_season"

# All temporal columns (core + optional peak flag) — used when peak IS included
TEMPORAL_FEATURE_COLUMNS = CORE_TEMPORAL_COLUMNS + [PEAK_FEATURE_COL]

# Base schema (without is_peak_fire_season) — used when --no-peak-feature is set
SCHEMA_BASE = [
    "latitude", "longitude", "acq_date",
    *FEATURE_COLUMNS,
    *CORE_TEMPORAL_COLUMNS,
    "Fire",
]

# Full schema (default — includes peak fire season flag)
SCHEMA_FULL = [
    "latitude", "longitude", "acq_date",
    *FEATURE_COLUMNS,
    *TEMPORAL_FEATURE_COLUMNS,
    "Fire",
]


# ---------------------------------------------------------------------------
# Step 1 — Load positive samples
# ---------------------------------------------------------------------------

def load_positive_samples(firms_csv: Path) -> pd.DataFrame:
    """
    Load FIRMS fire detections as positive samples.

    Every row in the FIRMS dataset is a confirmed fire event.
    We extract latitude, longitude, and acquisition date.
    The ``Fire`` label is assigned here as 1 (positive class).

    Parameters
    ----------
    firms_csv : Path
        Path to the filtered 2024 FIRMS CSV file.

    Returns
    -------
    pd.DataFrame
        Columns: latitude, longitude, acq_date, Fire
        All rows have Fire = 1.

    Raises
    ------
    FileNotFoundError  — if the file does not exist.
    ValueError         — if required columns are absent.
    """
    if not firms_csv.exists():
        raise FileNotFoundError(
            f"FIRMS CSV not found: {firms_csv}\n"
            f"Run firms_2024.py first to generate the filtered 2024 file."
        )

    logger.info("Loading FIRMS data from: %s", firms_csv)
    df = pd.read_csv(firms_csv, parse_dates=["acq_date"])

    required = {"latitude", "longitude", "acq_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FIRMS CSV missing required columns: {missing}")

    # Validate coordinate sanity
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude", "acq_date"])
    # Remove impossible coordinates (outside Earth)
    df = df[(df["latitude"].between(-90.0, 90.0)) & (df["longitude"].between(-180.0, 180.0))]
    if len(df) < before:
        logger.warning("Dropped %d FIRMS rows with invalid coordinates.", before - len(df))

    # Filter strictly to 2024
    df = df[df["acq_date"].dt.year == 2024].copy()

    pos_df = df[["latitude", "longitude", "acq_date"]].copy()
    pos_df["Fire"] = 1

    logger.info("Positive samples loaded: %d fire events (after QC)", len(pos_df))
    logger.info(
        "  Date range: %s → %s",
        pos_df["acq_date"].min().date(),
        pos_df["acq_date"].max().date(),
    )
    return pos_df


# ---------------------------------------------------------------------------
# Step 2 — Negative sample generation (delegated to generate_negatives.py)
# ---------------------------------------------------------------------------

# (see generate_negatives.generate_negative_samples — called by build_dataset)


# ---------------------------------------------------------------------------
# Step 3 — Raster feature sampling
# ---------------------------------------------------------------------------

def sample_all_features(
    points_df: pd.DataFrame,
    sampler: RasterSampler,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Sample all raster features for every (latitude, longitude) in points_df.

    Uses tqdm progress bar for visibility on large datasets.
    Skipped points (all NaN) are counted and logged but NOT silently dropped
    here — the caller decides whether to retain or discard them.

    Parameters
    ----------
    points_df : pd.DataFrame
        Must contain columns: latitude, longitude (and any other metadata).
        Does NOT need to contain feature columns yet.
    sampler : RasterSampler
        An already-opened RasterSampler context.
    validate : bool
        If True, run coordinate round-trip validation on every 500th point.

    Returns
    -------
    pd.DataFrame
        Original columns + one column per feature (NDVI, EVI, LST, …).
    """
    lats = points_df["latitude"].values
    lons = points_df["longitude"].values
    n    = len(points_df)

    feature_records = []
    skipped = 0

    for i, (lat, lon) in enumerate(
        tqdm(zip(lats, lons), total=n, desc="Sampling rasters", unit="pts", dynamic_ncols=True)
    ):
        feat = sampler.sample(float(lat), float(lon))

        # Coordinate validation on every 500th point
        if validate and (i % 500 == 0):
            sampler.validate_sample(float(lat), float(lon), feat, tolerance_km=1.0)

        # Track fully-NaN rows (point completely outside all rasters)
        if all(np.isnan(v) for v in feat.values()):
            skipped += 1
            logger.debug(
                "Point %d (%.5f, %.5f) returned all-NaN features — outside raster extents?",
                i, lat, lon
            )

        feature_records.append(feat)

    if skipped > 0:
        logger.warning(
            "%d / %d points returned all-NaN features (fully outside raster extents). "
            "These rows will be dropped during QC.",
            skipped, n
        )

    feat_df = pd.DataFrame(feature_records, index=points_df.index)
    result  = pd.concat([points_df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    logger.info(
        "Feature sampling complete. Shape: %s  Available features: %s",
        result.shape,
        sampler.available_features,
    )
    return result


# ---------------------------------------------------------------------------
# Step 4 — Dataset quality control
# ---------------------------------------------------------------------------

def quality_control(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Apply data quality checks and produce a clean DataFrame.

    Checks performed:
    -----------------
    1. Remove rows where ALL feature columns are NaN (outside every raster).
    2. [STRICT MODE] Remove rows where >50% of raster features are NaN.
       This removes points outside the GEE study area (see diagnose_coverage.py).
    3. Remove exact duplicate (latitude, longitude, Fire) rows.
    4. Validate coordinate bounds (Texas: lat 25-37, lon -107.5 to -92.5).

    Parameters
    ----------
    df : pd.DataFrame
        Full merged dataset (positives + negatives, features sampled).
    strict : bool
        If True, also drops rows with >50% missing raster features.
        Default False preserves all rows (XGBoost handles NaN natively).

    Returns
    -------
    pd.DataFrame -- clean dataset ready for splitting and saving.
    """
    logger.info("-" * 60)
    logger.info("QUALITY CONTROL  (strict=%s)", strict)
    logger.info("-" * 60)

    n_start   = len(df)
    feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    n_feats   = len(feat_cols)

    # ── 1. Remove rows with ALL-NaN features ──────────────────────────────
    all_nan_mask = df[feat_cols].isna().all(axis=1)
    n_all_nan    = int(all_nan_mask.sum())
    if n_all_nan > 0:
        logger.warning("Dropping %d rows with ALL-NaN features.", n_all_nan)
        df = df[~all_nan_mask].copy()

    # ── 2. [STRICT] Remove rows with >50% missing raster features ─────────
    n_strict_drop = 0
    if strict:
        missing_count  = df[feat_cols].isna().sum(axis=1)
        strict_mask    = missing_count > (n_feats / 2)  # more than half missing
        n_strict_drop  = int(strict_mask.sum())
        if n_strict_drop > 0:
            # Report Fire=1 vs Fire=0 among dropped rows
            dropped_fire = int((df[strict_mask]["Fire"] == 1).sum()) if "Fire" in df.columns else "?"
            dropped_neg  = int((df[strict_mask]["Fire"] == 0).sum()) if "Fire" in df.columns else "?"
            logger.warning(
                "[STRICT QC] Dropping %d rows with >50%% missing raster features "
                "(Fire=1: %s, Fire=0: %s). Root cause: GEE rasters do not cover "
                "full Texas extent. Run diagnose_coverage.py for full analysis.",
                n_strict_drop, dropped_fire, dropped_neg
            )
            df = df[~strict_mask].copy()
        else:
            logger.info("[STRICT QC] No rows with >50%% missing features found.")

    # ── 3. Remove duplicate rows ───────────────────────────────────────────
    dup_mask  = df.duplicated(subset=["latitude", "longitude", "Fire"], keep="first")
    n_dups    = int(dup_mask.sum())
    if n_dups > 0:
        logger.info("Removing %d duplicate (lat, lon, Fire) rows.", n_dups)
        df = df[~dup_mask].copy()

    # ── 4. Validate coordinate bounds ─────────────────────────────────────
    bad_coords = ~(
        df["latitude"].between(25.0, 37.0) &
        df["longitude"].between(-107.5, -92.5)
    )
    n_bad = int(bad_coords.sum())
    if n_bad > 0:
        logger.warning(
            "Dropping %d rows with coordinates outside Texas bounding box.", n_bad
        )
        df = df[~bad_coords].copy()

    n_end = len(df)
    logger.info(
        "QC summary: started=%d -> all_nan=%d, strict_drop=%d, dups=%d, "
        "bad_coords=%d -> final=%d  (removed %.1f%%)",
        n_start, n_all_nan, n_strict_drop, n_dups, n_bad, n_end,
        (n_start - n_end) / n_start * 100 if n_start > 0 else 0.0
    )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 5 — Temporal / Seasonal Feature Engineering
# ---------------------------------------------------------------------------

def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive 8 temporal features from the ``acq_date`` column.

    WHY this is critical
    --------------------
    The GEE raster exports (NDVI, EVI, LST, etc.) are annual composites.
    A fire on Jan 15 and a fire on Aug 20 at the same coordinate will receive
    IDENTICAL raster values. Without temporal features, XGBoost cannot learn
    seasonal fire risk patterns.

    Texas seasonal fire risk profile
    ---------------------------------
    * Feb–Apr  : PEAK season — dry cured grass, low humidity, strong winds
    * May–Jun  : Moderate — vegetation greening, occasional storms
    * Jul–Aug  : Hot but more rain (East Texas); drought risk in West Texas
    * Sep      : Second build-up — vegetation drying after summer
    * Oct–Nov  : Second PEAK — post-hurricane dry spells, senescent grass
    * Dec–Jan  : Lower but non-zero — cold fronts with dry winds

    Features engineered
    -------------------
    month (1–12)
        Raw calendar month. XGBoost can split on this directly.

    day_of_year (1–366)
        Higher resolution than month. Captures within-month risk gradients.

    season_code (0=Winter, 1=Spring, 2=Summer, 3=Fall)
        Coarse 4-class season label. Useful as a categorical split.

    sin_month, cos_month
        Cyclical encoding: sin(2π × month/12), cos(2π × month/12).
        Ensures December (month=12) and January (month=1) are neighbours
        in feature space — critical for tree-based models.

    sin_doy, cos_doy
        Cyclical encoding of day_of_year on a 365-day cycle.
        Finer-grained than month cyclical encoding.

    is_peak_fire_season (0 or 1)
        Binary flag: 1 if month ∈ {Feb, Mar, Apr, Oct, Nov}.
        Direct signal for the two Texas peak fire windows.
        XGBoost will weight this heavily as it correlates strongly with Fire=1.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with ``acq_date`` column (datetime or string parseable).

    Returns
    -------
    pd.DataFrame — same df with 8 new temporal feature columns appended.
    """
    df = df.copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"])

    month      = df["acq_date"].dt.month
    doy        = df["acq_date"].dt.day_of_year

    # Raw integer features
    df["month"]         = month.astype(np.int8)
    df["day_of_year"]   = doy.astype(np.int16)
    df["season_code"]   = month.map(SEASON_MAP).astype(np.int8)

    # Cyclical encoding — prevents discontinuity at year boundary
    df["sin_month"] = np.sin(2.0 * np.pi * month / 12.0)
    df["cos_month"] = np.cos(2.0 * np.pi * month / 12.0)
    df["sin_doy"]   = np.sin(2.0 * np.pi * doy / 365.0)
    df["cos_doy"]   = np.cos(2.0 * np.pi * doy / 365.0)

    # Texas peak fire season binary indicator
    df["is_peak_fire_season"] = month.isin(PEAK_FIRE_MONTHS).astype(np.int8)

    # Validate: no NaNs should appear in temporal features
    temporal_null_check = df[TEMPORAL_FEATURE_COLUMNS].isna().sum()
    if temporal_null_check.any():
        logger.error(
            "NaN values found in temporal features — check acq_date column!\n%s",
            temporal_null_check[temporal_null_check > 0]
        )
    else:
        logger.info(
            "Temporal features engineered successfully: %s",
            TEMPORAL_FEATURE_COLUMNS
        )

    # Log feature distribution for QA
    peak_pct = (df["is_peak_fire_season"] == 1).mean() * 100
    fire_in_peak = 0.0
    if "Fire" in df.columns:
        fire_peak_df = df[df["Fire"] == 1]
        fire_in_peak = (fire_peak_df["is_peak_fire_season"] == 1).mean() * 100

    logger.info(
        "  Seasonal breakdown: %.1f%% of all rows in peak fire season | "
        "%.1f%% of FIRE=1 rows in peak season",
        peak_pct, fire_in_peak
    )
    logger.info(
        "  Month distribution (fire events):\n%s",
        df[df["Fire"] == 1]["month"].value_counts().sort_index().to_string()
        if "Fire" in df.columns else "N/A"
    )

    return df


# ---------------------------------------------------------------------------
# Step 6 — Chronological train/val/test split
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset chronologically by acquisition month to prevent data leakage.

    Split strategy (2024 calendar year):
    ├── Train : Jan–Aug  (months 1–8)  ← ~67% of annual fire season
    ├── Val   : Sep      (month  9)    ← shoulder season
    └── Test  : Oct–Dec  (months 10–12) ← late-season / post-season

    Parameters
    ----------
    df : pd.DataFrame  — full cleaned dataset with acq_date column.

    Returns
    -------
    (train_df, val_df, test_df) — three DataFrames without the temporary
    ``month`` column.
    """
    df = df.copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    month = df["acq_date"].dt.month

    train = df[month <= TRAIN_END_MONTH].drop(columns=[], errors="ignore")
    val   = df[month == VAL_MONTH]
    test  = df[month >  VAL_MONTH]

    logger.info(
        "Chronological split — Train: %d rows | Val: %d rows | Test: %d rows",
        len(train), len(val), len(test)
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Step 6 — Statistics & Reporting
# ---------------------------------------------------------------------------

def print_dataset_statistics(df: pd.DataFrame, split_dfs: dict) -> None:
    """
    Print comprehensive dataset statistics to the console and log.

    Covers:
    -------
    * Dataset shape
    * Class balance (positive / negative samples)
    * Missing values per feature column (count and %)
    * Feature summary statistics (min, max, mean, std, median)
    * Split sizes and class balance per split
    """
    sep = "=" * 70

    print(f"\n{sep}")
    print("  WILDFIRE DATASET — CREATION STATISTICS")
    print(sep)

    # ── Overall shape ─────────────────────────────────────────────────────
    n_pos = int((df["Fire"] == 1).sum())
    n_neg = int((df["Fire"] == 0).sum())
    print(f"\n  Dataset shape      : {df.shape}")
    print(f"  Positive samples   : {n_pos:>6d}  (Fire = 1, confirmed FIRMS events)")
    print(f"  Negative samples   : {n_neg:>6d}  (Fire = 0, synthetic non-fire points)")
    print(f"  Total rows         : {len(df):>6d}")
    print(f"  Pos:Neg ratio      : 1 : {n_neg // n_pos if n_pos > 0 else '?'}")

    # ── Feature columns ───────────────────────────────────────────────────
    feat_cols    = [c for c in FEATURE_COLUMNS if c in df.columns]
    temp_cols    = [c for c in TEMPORAL_FEATURE_COLUMNS if c in df.columns]
    all_feat     = feat_cols + temp_cols
    print(f"\n  Raster features  ({len(feat_cols)}): {feat_cols}")
    print(f"  Temporal features({len(temp_cols)}): {temp_cols}")

    # ── Seasonal breakdown ────────────────────────────────────────────────
    if "is_peak_fire_season" in df.columns:
        print(f"\n{'─' * 70}")
        print("  SEASONAL FIRE RISK BREAKDOWN")
        print(f"{'─' * 70}")
        if "month" in df.columns and "Fire" in df.columns:
            monthly = df.groupby("month")["Fire"].agg(["sum", "count"])
            monthly.columns = ["fires", "total"]
            monthly["fire_rate%"] = (monthly["fires"] / monthly["total"] * 100).round(2)
            month_names = {
                1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
            }
            print(f"  {'Month':<6} {'Fires':>7} {'Total':>7} {'Fire%':>9}  {'Peak?':>5}")
            print(f"  {'-'*6} {'-'*7} {'-'*7} {'-'*9}  {'-'*5}")
            for m, row in monthly.iterrows():
                peak_flag = " ★" if m in PEAK_FIRE_MONTHS else ""
                print(f"  {month_names.get(m, m):<6} {int(row['fires']):>7,d} "
                      f"{int(row['total']):>7,d} {row['fire_rate%']:>8.2f}%{peak_flag}")
        peak_pct      = (df["is_peak_fire_season"] == 1).mean() * 100
        fire_peak_pct = 0.0
        if "Fire" in df.columns:
            fire_df = df[df["Fire"] == 1]
            fire_peak_pct = (fire_df["is_peak_fire_season"] == 1).mean() * 100
        print(f"\n  Peak season rows    : {peak_pct:.1f}% of all rows")
        print(f"  FIRE=1 in peak season: {fire_peak_pct:.1f}% of fire events")

    # -- Missing values (with per-class bias check) --
    print("\n" + chr(9472)*70)
    print("  MISSING VALUES PER FEATURE  (Fire=1 vs Fire=0 bias check)")
    print(chr(9472)*70)
    print(f"  {'Feature':<20} {'Total%':>7}  {'Fire=1%':>9}  {'Fire=0%':>9}  {'Bias?':>10}")
    print(f"  {'-'*20} {'-'*7}  {'-'*9}  {'-'*9}  {'-'*10}")

    fire1_df      = df[df['Fire'] == 1]
    fire0_df      = df[df['Fire'] == 0]
    bias_detected = False

    for col in feat_cols:
        pct_all = (df[col].isna().sum()       / len(df)       * 100) if len(df)       > 0 else 0.0
        pct1    = (fire1_df[col].isna().sum() / len(fire1_df) * 100) if len(fire1_df) > 0 else 0.0
        pct0    = (fire0_df[col].isna().sum() / len(fire0_df) * 100) if len(fire0_df) > 0 else 0.0
        biased  = (pct0 > pct1 + 10)
        if biased:
            bias_detected = True
        badge = "CLASS BIAS!" if biased else "OK"
        print(f"  {col:<20} {pct_all:>6.1f}%  {pct1:>8.1f}%  {pct0:>8.1f}%  {badge:>10}")

    if bias_detected:
        print("\n  *** CLASS BIAS: Fire=0 has much higher NaN rates than Fire=1.")
        print("  *** XGBoost may learn 'NaN = Fire=0'. This is a DATA ARTIFACT, not a pattern.")
        print("  *** The valid-pixel sampling fix in generate_negatives.py resolves this.")
        print("  *** Rerun: python src/dataset_builder/build_dataset.py")
    else:
        print("\n  [BIAS CHECK PASSED] Fire=1 and Fire=0 have similar NaN rates.")
        print("  Valid-pixel negative sampling is working correctly.")

    # ── Summary statistics ────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  RASTER FEATURE SUMMARY STATISTICS")
    print(f"{'─' * 70}")
    desc = df[feat_cols].describe().T[["min", "mean", "50%", "max", "std"]]
    desc.columns = ["min", "mean", "median", "max", "std"]
    print(desc.to_string(float_format=lambda x: f"{x:>10.4f}"))

    print(f"\n{'─' * 70}")
    print("  TEMPORAL FEATURE SUMMARY STATISTICS")
    print(f"{'─' * 70}")
    if temp_cols:
        tdesc = df[temp_cols].describe().T[["min", "mean", "50%", "max", "std"]]
        tdesc.columns = ["min", "mean", "median", "max", "std"]
        print(tdesc.to_string(float_format=lambda x: f"{x:>10.4f}"))

    # ── Fire distribution by season ───────────────────────────────────────
    if "season_code" in df.columns and "Fire" in df.columns:
        print(f"\n{'─' * 70}")
        print("  FIRE DISTRIBUTION BY SEASON")
        print(f"{'─' * 70}")
        season_names = {0: "Winter(Dec-Feb)", 1: "Spring(Mar-May)",
                        2: "Summer(Jun-Aug)", 3: "Fall(Sep-Nov)"}
        by_season = df.groupby("season_code")["Fire"].agg(["sum", "count"])
        by_season.columns = ["fires", "total"]
        by_season["fire_rate%"] = (by_season["fires"] / by_season["total"] * 100).round(2)
        print(f"  {'Season':<20} {'Fires':>7} {'Total':>7} {'Fire%':>9}")
        print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*9}")
        for sc, row in by_season.iterrows():
            print(f"  {season_names.get(sc, sc):<20} {int(row['fires']):>7,d}"
                  f" {int(row['total']):>7,d} {row['fire_rate%']:>8.2f}%")

    # ── Split class balance with percentages ──────────────────────────────
    print(f"\n{'─' * 70}")
    print("  TRAIN / VAL / TEST SPLIT — CLASS BALANCE")
    print(f"{'─' * 70}")
    print(f"  {'Split':<8} {'Total':>7} {'Fire=1':>8} {'Fire%':>8}"
          f" {'Fire=0':>8} {'NonFire%':>10} {'Ratio':>8}")
    print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for name, sdf in split_dfs.items():
        sp      = int((sdf["Fire"] == 1).sum())
        sn      = int((sdf["Fire"] == 0).sum())
        total   = len(sdf)
        fire_p  = (sp / total * 100) if total > 0 else 0.0
        nonf_p  = (sn / total * 100) if total > 0 else 0.0
        rat     = f"1:{sn // sp}" if sp > 0 else "N/A"
        print(f"  {name:<8} {total:>7,d} {sp:>8,d} {fire_p:>7.1f}%"
              f" {sn:>8,d} {nonf_p:>9.1f}% {rat:>8}")

    print(f"\n{sep}\n")

    # Also log to logger so it appears in log files
    logger.info("Dataset shape: %s  Positive: %d  Negative: %d", df.shape, n_pos, n_neg)


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------

def build_dataset(
    firms_csv:           Path  = FIRMS_CSV,
    raw_dir:             Path  = RAW_DIR,
    proc_dir:            Path  = PROC_DIR,
    neg_ratio:           int   = NEG_POS_RATIO,
    exclusion_km:        float = EXCLUSION_KM,
    seed:                int   = RANDOM_SEED,
    include_peak_feature: bool = True,
    strict_qc:           bool = False,
) -> pd.DataFrame:
    """
    Full end-to-end dataset construction pipeline (V1 — annual composites).

    Parameters
    ----------
    firms_csv : Path
        Path to the filtered 2024 FIRMS CSV file.
    raw_dir : Path
        data/raw/ directory containing raster subfolders (ndvi/, slope/, …).
    proc_dir : Path
        Output directory for processed dataset files.
    neg_ratio : int
        Negative-to-positive sample ratio (1=balanced, 3=3× more negatives).
    exclusion_km : float
        Safety buffer (km) around known fire points for negative exclusion.
    seed : int
        Random seed for full reproducibility.
    include_peak_feature : bool
        If True (default), include ``is_peak_fire_season`` in the dataset.
        This is Texas A&M domain-knowledge: Feb/Mar/Apr/Oct/Nov = peak fire months.
        Set False to build an ablation dataset WITHOUT this feature so you can
        compare model performance (AUC, F1, Precision, Recall) with and without it.
        Recommendation: only retain in production if it consistently improves metrics.

    Returns
    -------
    pd.DataFrame — the complete wildfire_dataset (positives + negatives + features).
    """
    proc_dir.mkdir(parents=True, exist_ok=True)
    t_pipeline_start = time.perf_counter()

    # Select correct schema based on flag
    schema_cols = SCHEMA_FULL if include_peak_feature else SCHEMA_BASE

    logger.info("=" * 70)
    logger.info("  WILDFIRE DATASET BUILDER  —  V1 PIPELINE (annual composites)")
    logger.info("=" * 70)
    logger.info("  FIRMS CSV          : %s", firms_csv)
    logger.info("  Raw rasters        : %s", raw_dir)
    logger.info("  Output dir         : %s", proc_dir)
    logger.info("  Neg ratio          : 1:%d", neg_ratio)
    logger.info("  Exclusion          : %.1f km", exclusion_km)
    logger.info("  Seed               : %d", seed)
    logger.info("  Peak feature       : %s  (--no-peak-feature to toggle)",
                "INCLUDED" if include_peak_feature else "EXCLUDED (ablation)")
    logger.info("  Strict QC          : %s  (--strict-qc to enable)",
                "ON -- dropping >50%% missing rows" if strict_qc else "OFF -- NaN rows kept")
    logger.info("  Neg sampling mode  : VALID-PIXEL POOL (no NaN bias)")
    logger.info("=" * 70)

    # ── STEP 1: Positive samples ─────────────────────────────────────────
    logger.info("\n[STEP 1/8] Loading positive samples from FIRMS ...")
    t0 = time.perf_counter()
    pos_df = load_positive_samples(firms_csv)
    n_pos  = len(pos_df)
    logger.info("  → %d positive samples  (%.2f s)", n_pos, time.perf_counter() - t0)

    # ── STEP 2: Negative sample generation (valid-pixel mode) ───────────────
    n_neg = n_pos * neg_ratio
    logger.info(
        "\n[STEP 2/8] Generating %d negative samples (ratio 1:%d) ...",
        n_neg, neg_ratio
    )
    logger.info(
        "  BIAS FIX: Sampling negatives from valid (non-NaN) raster pixels only.\n"
        "  This eliminates the original bias where missing values were 100%%\n"
        "  correlated with Fire=0, which would let XGBoost use NaN as a class proxy."
    )
    t0 = time.perf_counter()
    neg_df = generate_negative_samples(
        firms_csv           = firms_csv,
        n_negatives         = n_neg,
        raw_dir             = raw_dir,          # <-- CORE FIX: valid-pixel pool
        exclusion_radius_km = exclusion_km,
        seed                = seed,
    )
    logger.info("  -> %d negative samples generated  (%.2f s)", len(neg_df), time.perf_counter() - t0)

    # Spatial stats for the negatives (sampled check)
    try:
        stats = negative_sample_spatial_stats(neg_df, firms_csv)
        logger.info(
            "  Negative spatial stats — min_dist_to_fire=%.2f km, mean_dist=%.2f km",
            stats["min_dist_to_fire_km"], stats["mean_dist_to_fire_km"]
        )
    except Exception as exc:
        logger.warning("Could not compute spatial stats: %s", exc)

    # ── STEP 3: Combine positives + negatives ────────────────────────────
    logger.info("\n[STEP 3/8] Combining positive and negative sample sets ...")
    all_points = pd.concat([pos_df, neg_df], ignore_index=True)
    logger.info(
        "  → Total points to sample: %d  (%d fire, %d non-fire)",
        len(all_points), n_pos, len(neg_df)
    )

    # ── STEP 4: Open rasters and sample features ─────────────────────────
    logger.info("\n[STEP 4/8] Opening rasters and sampling features ...")
    t0 = time.perf_counter()
    with RasterSampler(raw_dir) as sampler:
        logger.info("  %s", sampler.summary())

        if sampler.missing_features:
            logger.warning(
                "  WARNING — These features will be all-NaN: %s",
                sampler.missing_features
            )

        dataset = sample_all_features(all_points, sampler, validate=True)

    logger.info("  → Raster sampling done  (%.2f s)", time.perf_counter() - t0)

    # ── STEP 5: Quality control ──────────────────────────────────────────
    logger.info("\n[STEP 5/9] Applying quality control checks ...")
    dataset = quality_control(dataset, strict=strict_qc)

    # ── STEP 6: Temporal / seasonal feature engineering ───────────────────
    logger.info("\n[STEP 6/9] Engineering temporal and seasonal features from acq_date ...")
    logger.info(
        "  WHY: Raster files are annual composites — a Jan fire and Aug fire "
        "at the same location get identical raster values. Temporal features "
        "teach XGBoost seasonal fire risk patterns (Texas peak: Feb-Apr & Oct-Nov)."
    )
    t0 = time.perf_counter()
    dataset = engineer_temporal_features(dataset)
    logger.info("  → Temporal features added  (%.2f s)", time.perf_counter() - t0)

    # ── STEP 6b: Drop is_peak_fire_season if ablation mode ───────────────
    if not include_peak_feature and PEAK_FEATURE_COL in dataset.columns:
        dataset = dataset.drop(columns=[PEAK_FEATURE_COL])
        logger.info(
            "  [ABLATION] Dropped '%s'. This build is for model comparison "
            "(WITHOUT peak season domain-knowledge feature).",
            PEAK_FEATURE_COL
        )
    elif include_peak_feature:
        logger.info(
            "  [PEAK FEATURE] '%s' INCLUDED. To compare without it, run: "
            "--no-peak-feature",
            PEAK_FEATURE_COL
        )

    # ── STEP 7: Enforce output schema column order ────────────────────────
    logger.info("\n[STEP 7/9] Enforcing output schema ...")
    # Only include columns that actually exist in the dataset
    ordered_cols = [c for c in schema_cols if c in dataset.columns]
    # Append any unexpected extra columns at the end (don't lose data)
    extra_cols   = [c for c in dataset.columns if c not in schema_cols]
    if extra_cols:
        logger.debug("Extra columns preserved: %s", extra_cols)
    dataset = dataset[ordered_cols + extra_cols]
    logger.info(
        "  → Final schema (%d cols): %s",
        len(dataset.columns), list(dataset.columns)
    )

    # ── STEP 8: Save wildfire_dataset.csv and .parquet ───────────────────
    logger.info("\n[STEP 8/9] Saving wildfire_dataset.csv and .parquet ...")
    t0 = time.perf_counter()

    out_csv     = proc_dir / "wildfire_dataset.csv"
    out_parquet = proc_dir / "wildfire_dataset.parquet"

    dataset.to_csv(out_csv, index=False)
    logger.info("  → CSV saved: %s  (%.2f MB)", out_csv, out_csv.stat().st_size / 1e6)

    try:
        dataset.to_parquet(out_parquet, index=False, engine="pyarrow", compression="snappy")
        logger.info(
            "  → Parquet saved: %s  (%.2f MB)",
            out_parquet, out_parquet.stat().st_size / 1e6
        )
    except ImportError:
        logger.warning(
            "pyarrow not installed — saving parquet with fastparquet fallback ..."
        )
        try:
            dataset.to_parquet(out_parquet, index=False, engine="fastparquet")
            logger.info("  → Parquet saved (fastparquet): %s", out_parquet)
        except ImportError:
            logger.error(
                "No parquet engine available. Install: conda install -c conda-forge pyarrow"
            )

    logger.info("  → Save complete  (%.2f s)", time.perf_counter() - t0)

    # ── STEP 9: Chronological split → train / val / test ─────────────────
    logger.info("\n[STEP 9/9] Chronological split (Jan–Aug=train, Sep=val, Oct–Dec=test) ...")
    logger.info(
        "  NOTE: Seasonal features (sin_month, cos_month, is_peak_fire_season, etc.) "
        "are included in EVERY split — the model learns seasonal context from the "
        "TRAINING set and can apply it to Sep–Dec test/val correctly."
    )
    train, val, test = chronological_split(dataset)

    train.to_csv(proc_dir / "train.csv", index=False)
    val.to_csv(  proc_dir / "val.csv",   index=False)
    test.to_csv( proc_dir / "test.csv",  index=False)
    logger.info("  → Split CSVs saved: train.csv, val.csv, test.csv")

    # ── STEP 10: Statistics ───────────────────────────────────────────────
    print_dataset_statistics(
        dataset,
        {"TRAIN": train, "VAL": val, "TEST": test},
    )

    elapsed = time.perf_counter() - t_pipeline_start
    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE — total elapsed: %.1f s (%.1f min)", elapsed, elapsed / 60)
    logger.info("=" * 70)
    logger.info("Output files:")
    logger.info("  %s", proc_dir / "wildfire_dataset.csv")
    logger.info("  %s", proc_dir / "wildfire_dataset.parquet")
    logger.info("  %s", proc_dir / "train.csv")
    logger.info("  %s", proc_dir / "val.csv")
    logger.info("  %s", proc_dir / "test.csv")

    return dataset


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Texas wildfire ML dataset from GEE rasters + NASA FIRMS 2024.\n"
            "V1 pipeline: annual composite rasters + temporal features.\n"
            "Use --no-peak-feature for ablation comparison."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--firms",
        type=str,
        default=str(FIRMS_CSV),
        help="Path to the filtered 2024 FIRMS CSV file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(RAW_DIR),
        help="data/raw directory containing raster subfolders.",
    )
    parser.add_argument(
        "--proc-dir",
        type=str,
        default=str(PROC_DIR),
        help="Output directory for processed dataset files.",
    )
    parser.add_argument(
        "--neg-ratio",
        type=int,
        default=NEG_POS_RATIO,
        choices=[1, 2, 3, 4, 5],
        help="Negative-to-positive sample ratio (default: 3).",
    )
    parser.add_argument(
        "--exclusion-km",
        type=float,
        default=EXCLUSION_KM,
        help="Safety buffer (km) around fire points for negative exclusion.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--strict-qc",
        action="store_true",
        default=False,
        help=(
            "Drop rows with >50%% missing raster features (strict QC mode). "
            "Use when GEE rasters don't cover all of Texas. "
            "Run diagnose_coverage.py first to understand missing value root cause."
        ),
    )
    parser.add_argument(
        "--no-peak-feature",
        action="store_true",
        default=False,
        help=(
            "EXCLUDE is_peak_fire_season from the dataset (ablation mode).\n"
            "Use this to build a second dataset WITHOUT the Texas-specific domain-\n"
            "knowledge feature, then compare model AUC/F1/Precision/Recall.\n"
            "Only include the feature in production if it consistently improves metrics."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# FutureGEEInterface — V2 upgrade contract
# ---------------------------------------------------------------------------

class FutureGEEInterface:
    """
    Abstract interface documenting the V2 upgrade path.

    PURPOSE
    -------
    V1 uses annual composite rasters (e.g. NDVI_2024.tif = yearly mean).
    V2 will query GEE on a per-event basis, extracting environmental conditions
    that match the EXACT DATE (±N days) of each FIRMS fire detection.

    This class defines the method signatures that the V2 extractor must
    implement. Because the OUTPUT SCHEMA is identical to V1, the downstream
    model training pipeline requires ZERO changes when upgrading V1 → V2.

    V2 UPGRADE STEPS
    ----------------
    1.  Implement GEEEventExtractor(FutureGEEInterface)
    2.  Replace RasterSampler calls in sample_all_features() with
        GEEEventExtractor.extract(lat, lon, date) calls.
    3.  DEM / Slope / Aspect / LandCover remain static — keep RasterSampler
        for those four features only.
    4.  All other code (QC, temporal features, split, stats) is unchanged.

    STATIC vs DYNAMIC LAYER CLASSIFICATION
    ----------------------------------------
    Static (annual rasters always correct):     Dynamic (need date-specific in V2):
      DEM         — elevation never changes       NDVI      — vegetation greenness
      Slope       — topography never changes      EVI       — vegetation index
      Aspect      — topography never changes      LST       — surface temperature
      LandCover   — changes very slowly           Temperature — air temperature
                                                  Wind       — highly variable
                                                  Rainfall   — highly variable
    """

    def extract(
        self,
        lat: float,
        lon: float,
        date: str,
        window_days: int = 8,
    ) -> dict:
        """
        Extract environmental conditions at (lat, lon) around a specific date.

        Parameters
        ----------
        lat : float        WGS-84 latitude
        lon : float        WGS-84 longitude
        date : str         ISO date string, e.g. '2024-03-15'
        window_days : int  ±days around the event date to aggregate (e.g. ±8 for MODIS)

        Returns
        -------
        dict with keys matching FEATURE_COLUMNS:
            {'NDVI': float, 'EVI': float, 'LST': float,
             'Temperature': float, 'Wind': float, 'Rainfall': float,
             'DEM': float, 'Slope': float, 'Aspect': float, 'LandCover': float}

        Notes
        -----
        V2 implementation should use:
            import ee
            ee.Initialize()
            # Query MODIS MOD13Q1 (NDVI/EVI), MOD11A2 (LST),
            #       ERA5-Land (Temperature, Wind, Rainfall)
            # around the target date within window_days.
        """
        raise NotImplementedError(
            "V2 not yet implemented. "
            "See FutureGEEInterface docstring for upgrade instructions."
        )

    def extract_batch(
        self,
        events: pd.DataFrame,
        window_days: int = 8,
    ) -> pd.DataFrame:
        """
        Batch version of extract() for all rows in events DataFrame.

        Parameters
        ----------
        events : pd.DataFrame  with columns: latitude, longitude, acq_date
        window_days : int      GEE temporal aggregation window

        Returns
        -------
        pd.DataFrame  with same index + dynamic feature columns
        """
        raise NotImplementedError("V2 not yet implemented.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    # Reconfigure logging level from CLI argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    build_dataset(
        firms_csv            = Path(args.firms),
        raw_dir              = Path(args.raw_dir),
        proc_dir             = Path(args.proc_dir),
        neg_ratio            = args.neg_ratio,
        exclusion_km         = args.exclusion_km,
        seed                 = args.seed,
        include_peak_feature = not args.no_peak_feature,
        strict_qc            = args.strict_qc,
    )
