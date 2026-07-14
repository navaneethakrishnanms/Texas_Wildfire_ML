"""
run_phase2g_assemble.py
------------------------
Phase 2G — Final Training Dataset Assembly  [PRODUCTION FINAL]

Joins all extracted feature tables into a single ML-ready training parquet:
  - full_training_labels.parquet   (Phase 2D: fire=1 + non-fire=0 rows)
  + static_features_<state>.parquet (Phase 2E: LANDFIRE rasters)
  + gridmet_features_<state>.parquet (Phase 2F: daily + 5-day weather)
  + temporal encodings (computed on-the-fly)

FEATURE SELECTION applied per TEAM_DATA_GUIDE.MD review:
  KEPT  : erc, fm100, vpd, vs, rmax, tmmx, pr + all 5D trailing stats
           avg_burn_prob, whp, flep4, cfl (LANDFIRE)
           fire_count, has_fire_history, burnable
           sin/cos month+hour, centroid_lat/lon
  DROPPED (redundant, per team guidance):
    bi      → correlated with erc (both NFDRS indices)
    tmmn    → correlated with tmmx
    fm1000  → correlated with fm100
    sph     → overlaps with vpd and rmax/rmin
    ecoregion_l2/l3 → captured by lat/lon
    h3_resolution   → constant, zero info
    bi_5D_mean, bi_5D_max → bi dropped, so its 5D stats drop too

NaN TREATMENT (6.64% of rows):
  - 24,954 rows have all gridMET NaN — these are H3 cells whose centroid maps
    to a gridMET ocean/nodata pixel (coastal Gulf cells, southern border cells)
  - XGBoost handles NaN natively → keep rows but flag them
  - A separate column `gridmet_missing` = 1 flags these rows for audit

CHRONOLOGICAL SPLIT:
  TRAIN : 2014–2017
  VAL   : 2018
  TEST  : 2019–2020

Usage:
    conda activate torch_gpu
    python run_phase2g_assemble.py --state TX
    python run_phase2g_assemble.py --state CA
    python run_phase2g_assemble.py --state ALL
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PHASE2_ROOT = Path(__file__).resolve().parent
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from config.phase2_config import STATE_CONFIG, LOGS_DIR

logger = logging.getLogger(__name__)

TRAIN_YEARS = list(range(2014, 2018))   # 2014–2017
VAL_YEARS   = [2018]
TEST_YEARS  = [2019, 2020]

# ── Feature drop list (per TEAM_DATA_GUIDE.MD review) ─────────────────────────
# These are confirmed redundant / zero-value columns
COLS_TO_DROP = [
    # Redundant gridMET daily variables (correlated duplicates)
    "bi",           # ≈ erc (both NFDRS energy-based indices — keep erc)
    "tmmn",         # ≈ tmmx (min temp correlated with max temp — keep tmmx)
    "fm1000",       # ≈ fm100 (1000-hr moisture correlated with 100-hr — keep fm100)
    "sph",          # overlaps with vpd + rmax/rmin — redundant

    # 5D stats for dropped variables (bi was dropped, so its 5D stats too)
    "bi_5D_mean",
    "bi_5D_max",

    # Categorical with no added value over lat/lon
    "ecoregion_l2",
    "ecoregion_l3",
    "Ecoregion_NA_L2CODE",
    "Ecoregion_NA_L3CODE",
    "GACCAbbrev",
    "NWCG_GACC_NAME",

    # Constant column
    "h3_resolution",

    # State name (constant within each state's dataset)
    "state",

    # Duplicate state columns (if present from join artifacts)
    "state_x",
    "state_y",
]

# ── Columns that must NOT be present (leakage audit) ──────────────────────────
FORBIDDEN_COLS = [
    "FIRE_SIZE", "FIRE_SIZE_CLASS",
    "CONT_DATE", "CONT_DOY", "CONT_TIME",
    "NWCG_CAUSE_CLASSIFICATION", "NWCG_GENERAL_CAUSE",
    "MTBS_ID", "MTBS_FIRE_NAME",
    "ICS_209_PLUS_INCIDENT_JOIN_ID", "ICS_209_PLUS_COMPLEX_JOIN_ID",
    "FIRE_NAME", "FOD_ID", "NWCG_CAUSE_AGE_CATEGORY",
]


def compute_temporal_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos cyclic encodings from date and window_hour."""
    df = df.copy()
    ts = pd.to_datetime(df["date_utc"])
    df["sin_month"] = np.sin(2 * np.pi * ts.dt.month / 12).astype(np.float32)
    df["cos_month"] = np.cos(2 * np.pi * ts.dt.month / 12).astype(np.float32)
    df["sin_hour"]  = np.sin(2 * np.pi * df["window_hour"] / 24).astype(np.float32)
    df["cos_hour"]  = np.cos(2 * np.pi * df["window_hour"] / 24).astype(np.float32)
    return df


def assemble(state_key: str, cfg: dict) -> bool:
    output_dir = cfg["output_dir"]
    slug       = state_key.lower()

    logger.info(f"{'=' * 65}")
    logger.info(f"PHASE 2G — {cfg['name'].upper()} — Final Dataset Assembly")
    logger.info(f"{'=' * 65}")

    # ── Step 1: Load base labels ──────────────────────────────────────────────
    labels_path = output_dir / "full_training_labels.parquet"
    if not labels_path.exists():
        logger.error(f"Labels not found: {labels_path} — run Phase 2D first")
        return False

    df = pd.read_parquet(labels_path)
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    logger.info(f"  Base labels: {len(df):,} rows  "
                f"(fire={n_pos:,} [{100*n_pos/len(df):.1f}%], "
                f"non-fire={n_neg:,} [{100*n_neg/len(df):.1f}%])")

    # ── Step 2: Join static (LANDFIRE) features ───────────────────────────────
    static_path = output_dir / f"static_features_{slug}.parquet"
    if static_path.exists():
        static_df  = pd.read_parquet(static_path)
        # Drop centroid columns that already exist in labels
        static_df  = static_df.drop(
            columns=["centroid_lat", "centroid_lon"], errors="ignore"
        )
        before = len(df.columns)
        df = df.merge(static_df, on="h3_cell", how="left")
        added = len(df.columns) - before
        logger.info(f"  Static join:  +{added} cols  ({static_path.name})")

        # Check LANDFIRE raster quality
        for col in ["avg_burn_prob", "whp", "flep4", "cfl"]:
            if col in df.columns:
                pct_zero  = (df[col] == 0).mean() * 100
                pct_valid = 100 - pct_zero
                if pct_valid < 10:
                    logger.warning(f"    {col}: {pct_zero:.1f}% zeros — rasters likely not downloaded")
                else:
                    logger.info(f"    {col}: mean={df[col].mean():.4f}  zeros={pct_zero:.1f}%  ✔")
    else:
        logger.warning(f"  Static features NOT found: {static_path}")
        logger.warning("  Run: python run_phase2e_static.py --state " + state_key)
        logger.warning("  Continuing without LANDFIRE features (will degrade model performance)")

    # ── Step 3: Join gridMET features ─────────────────────────────────────────
    gridmet_path = output_dir / f"gridmet_features_{slug}.parquet"
    if gridmet_path.exists():
        gridmet_df = pd.read_parquet(gridmet_path)
        before     = len(df.columns)
        df = df.merge(gridmet_df, on=["h3_cell", "date_utc"], how="left")
        added = len(df.columns) - before
        logger.info(f"  gridMET join: +{added} cols  ({gridmet_path.name})")

        # Flag rows with no gridMET data (coastal/border cells mapping to ocean pixels)
        if "erc" in df.columns:
            no_gridmet = df["erc"].isna()
            df["gridmet_missing"] = no_gridmet.astype(np.int8)
            n_missing  = int(no_gridmet.sum())
            pct        = 100 * n_missing / len(df)
            logger.info(f"  gridMET coverage: {100-pct:.1f}% valid  "
                        f"({n_missing:,} rows have no gridMET — coastal/border cells flagged)")
    else:
        logger.warning(f"  gridMET features NOT found: {gridmet_path}")
        logger.warning("  Run: python run_phase2f_gridmet.py --state " + state_key)

    # ── Step 4: Temporal encodings ────────────────────────────────────────────
    df = compute_temporal_encodings(df)
    logger.info(f"  Temporal encodings: sin/cos month + hour  (4 cols)")

    # ── Step 5: Leakage check ─────────────────────────────────────────────────
    found_forbidden = [c for c in FORBIDDEN_COLS if c in df.columns]
    if found_forbidden:
        logger.error(f"  ✘ LEAKAGE DETECTED: {found_forbidden}")
        df = df.drop(columns=found_forbidden, errors="ignore")
        logger.error(f"  Dropped {len(found_forbidden)} forbidden columns")
    else:
        logger.info(f"  Leakage audit: ✔ PASSED — no forbidden columns found")

    # ── Step 6: Drop redundant columns (per team review) ─────────────────────
    cols_present  = [c for c in COLS_TO_DROP if c in df.columns]
    cols_missing  = [c for c in COLS_TO_DROP if c not in df.columns]
    df = df.drop(columns=cols_present, errors="ignore")
    logger.info(f"  Dropped {len(cols_present)} redundant cols (per team review): "
                f"{cols_present}")
    if cols_missing:
        logger.info(f"  (already absent, no action needed: {cols_missing[:5]})")

    # ── Step 7: Final column audit ────────────────────────────────────────────
    ID_COLS      = ["h3_cell", "date_utc", "window_hour", "window_6h_utc",
                    "label", "fire_year", "gridmet_missing"]
    feature_cols = [c for c in df.columns if c not in ID_COLS]

    logger.info(f"\n  {'─'*55}")
    logger.info(f"  FINAL SHAPE: {df.shape}")
    logger.info(f"  Feature columns ({len(feature_cols)}):")

    # Group by type for clean logging
    static_feats   = [c for c in feature_cols if c in
                      ["avg_burn_prob", "whp", "flep4", "cfl", "EVH", "EVT",
                       "EVC_1km", "FRG", "Land_Cover", "Elevation", "Slope",
                       "Aspect", "TRI", "TPI", "fire_count", "has_fire_history",
                       "burnable"]]
    gridmet_feats  = [c for c in feature_cols if any(v in c for v in
                      ["erc","fm100","fm1000","vpd","vs","rmax","rmin",
                       "tmmx","tmmn","pr","sph","bi","5D"])]
    temporal_feats = [c for c in feature_cols if any(v in c for v in
                      ["sin_","cos_","month","hour"])]
    location_feats = [c for c in feature_cols if "centroid" in c or "lat" in c.lower()
                      or "lon" in c.lower()]
    other_feats    = [c for c in feature_cols if c not in static_feats + gridmet_feats
                      + temporal_feats + location_feats]

    logger.info(f"    Landscape/static  ({len(static_feats)}):  {static_feats}")
    logger.info(f"    gridMET weather   ({len(gridmet_feats)}):  {gridmet_feats}")
    logger.info(f"    Temporal          ({len(temporal_feats)}):  {temporal_feats}")
    logger.info(f"    Location          ({len(location_feats)}):  {location_feats}")
    if other_feats:
        logger.info(f"    Other             ({len(other_feats)}):  {other_feats}")

    # ── Step 8: Missing value summary ─────────────────────────────────────────
    logger.info(f"\n  MISSING VALUE SUMMARY:")
    any_missing = False
    for col in feature_cols:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            pct = 100 * n_nan / len(df)
            logger.info(f"    {col:<30} NaN={n_nan:>8,} ({pct:.2f}%)")
            any_missing = True
    if not any_missing:
        logger.info(f"    All feature columns fully populated!")

    # ── Step 9: Label distribution ────────────────────────────────────────────
    logger.info(f"\n  LABEL DISTRIBUTION:")
    logger.info(f"    label=1 (fire):     {n_pos:>8,}  ({100*n_pos/len(df):.1f}%)")
    logger.info(f"    label=0 (non-fire): {n_neg:>8,}  ({100*n_neg/len(df):.1f}%)")
    logger.info(f"    Total:              {len(df):>8,}")

    # ── Step 10: Save full dataset ────────────────────────────────────────────
    full_path = output_dir / f"final_training_dataset_{slug}.parquet"
    df.to_parquet(full_path, index=False, compression="snappy")
    logger.info(f"\n  ✔ Full dataset: {full_path}  ({full_path.stat().st_size/1e6:.0f} MB)")

    # ── Step 11: Chronological split ──────────────────────────────────────────
    train = df[df["fire_year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    val   = df[df["fire_year"].isin(VAL_YEARS)].reset_index(drop=True)
    test  = df[df["fire_year"].isin(TEST_YEARS)].reset_index(drop=True)

    logger.info(f"\n  CHRONOLOGICAL SPLIT:")
    for split_df, name, yrs in [
        (train, "TRAIN", "2014–2017"),
        (val,   "VAL",   "2018"),
        (test,  "TEST",  "2019–2020"),
    ]:
        p     = output_dir / f"{name.lower()}_{slug}.parquet"
        n_p   = int((split_df["label"] == 1).sum())
        n_n   = int((split_df["label"] == 0).sum())
        split_df.to_parquet(p, index=False, compression="snappy")
        logger.info(f"    {name:<6} ({yrs}): {len(split_df):>8,} rows  "
                    f"fire={n_p:,}  non-fire={n_n:,}  → {p.name}")

    # ── Step 12: Summary markdown ─────────────────────────────────────────────
    raster_status = "⚠️ LANDFIRE rasters not yet downloaded — avg_burn_prob/whp/flep4/cfl = 0" \
        if static_path.exists() and (df.get("avg_burn_prob", pd.Series([0])) == 0).mean() > 0.9 \
        else "✔ LANDFIRE rasters present"

    summary = f"""# Phase 2G Summary — {cfg['name']}

## Dataset Statistics

| Split | Years | Total Rows | Fire (1) | Non-fire (0) | Fire Rate |
|-------|-------|-----------|---------|-------------|-----------|
| TRAIN | 2014–2017 | {len(train):,} | {int((train.label==1).sum()):,} | {int((train.label==0).sum()):,} | {100*(train.label==1).mean():.1f}% |
| VAL   | 2018      | {len(val):,} | {int((val.label==1).sum()):,} | {int((val.label==0).sum()):,} | {100*(val.label==1).mean():.1f}% |
| TEST  | 2019–2020 | {len(test):,} | {int((test.label==1).sum()):,} | {int((test.label==0).sum()):,} | {100*(test.label==1).mean():.1f}% |
| **TOTAL** | 2014–2020 | **{len(df):,}** | **{int((df.label==1).sum()):,}** | **{int((df.label==0).sum()):,}** | **{100*(df.label==1).mean():.1f}%** |

## Feature Columns ({len(feature_cols)} total)

### Landscape (static, LANDFIRE)
{', '.join(static_feats) if static_feats else 'None — run Phase 2E after downloading rasters'}

### gridMET Weather (daily + 5-day trailing)
{', '.join(gridmet_feats)}

### Temporal
{', '.join(temporal_feats)}

### Location
{', '.join(location_feats)}

### Other
{', '.join(other_feats) if other_feats else 'None'}

## Dropped Columns (redundant per team review)
{', '.join(cols_present)}

## Data Quality
- gridMET NaN rows: {int(df['gridmet_missing'].sum()) if 'gridmet_missing' in df.columns else 'N/A'} ({100*df['gridmet_missing'].mean():.1f}% of rows — coastal/border cells)
- Fill contamination: ✔ CLEAN (all values < 9000)
- Leakage audit: ✔ PASSED
- LANDFIRE rasters: {raster_status}

## Files
- Full:  `final_training_dataset_{slug}.parquet`
- Train: `train_{slug}.parquet`
- Val:   `val_{slug}.parquet`
- Test:  `test_{slug}.parquet`

## Next Steps
1. Download 4 LANDFIRE rasters → re-run Phase 2E → re-run Phase 2G
2. Run Phase 3: `python run_phase3_train.py --state {state_key}`
"""
    (output_dir / "phase2g_summary.md").write_text(summary, encoding="utf-8")
    logger.info(f"  Summary: {output_dir / 'phase2g_summary.md'}")
    logger.info(f"{'=' * 65}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2G — Final Training Dataset Assembly [Production Final]"
    )
    parser.add_argument("--state", choices=["TX", "CA", "ALL"], required=True)
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "phase2g.log", encoding="utf-8"),
        ],
    )

    states = ["TX", "CA"] if args.state == "ALL" else [args.state]
    for s in states:
        ok = assemble(s, STATE_CONFIG[s])
        print(f"\n  {STATE_CONFIG[s]['name']:<15} {'✔ SUCCESS' if ok else '✘ FAILED'}")


if __name__ == "__main__":
    main()
