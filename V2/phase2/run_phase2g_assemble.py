"""
run_phase2g_assemble.py
------------------------
Phase 2G — Final Training Dataset Assembly

Joins all extracted feature tables into a single unified training parquet:
  - full_training_labels.parquet   (Phase 2D: fire=1 + non-fire=0 rows)
  + static_features_<state>.parquet (Phase 2E: LANDFIRE, terrain, infra)
  + gridmet_features_<state>.parquet (Phase 2F: daily weather)
  + temporal encodings (computed on-the-fly)
  = final_training_dataset_<state>.parquet

THE FINAL TABLE STRUCTURE:
  Each row = (H3 cell, date, 6-hour UTC window)
  label = 1 if fire discovered in this cell+window, else 0

  Feature groups:
    STATIC (15 cols): avg_burn_prob, whp, flep4, cfl, EVH, EVT, EVC_1km,
                      FRG, Land_Cover, Elevation, Slope, Aspect, TRI, TPI,
                      GACCAbbrev
    GRIDMET (35+ cols): erc, erc_5D_mean, erc_5D_max, fm100, fm100_5D_mean,
                        fm100_5D_min, bi, bi_5D_mean, vpd, vpd_5D_mean, vs,
                        vs_5D_mean, rmax, rmin, tmmx, tmmn, pr, sph, ...
    TEMPORAL (4 cols):  sin_month, cos_month, sin_hour, cos_hour
    LOCATION (2 cols):  centroid_lat, centroid_lon
    ECOREGION (2 cols): Ecoregion_NA_L2CODE, Ecoregion_NA_L3CODE
    LABEL (1 col):      label (0 or 1)

CHRONOLOGICAL SPLIT (applied automatically):
    TRAIN : fire_year 2014–2017
    VAL   : fire_year 2018
    TEST  : fire_year 2019–2020

Usage:
    conda activate torch_gpu
    python run_phase2g_assemble.py --state TX
    python run_phase2g_assemble.py --state CA
    python run_phase2g_assemble.py --state ALL

Output:
    phase2/outputs/<state>/final_training_dataset_<state>.parquet
    phase2/outputs/<state>/train_<state>.parquet
    phase2/outputs/<state>/val_<state>.parquet
    phase2/outputs/<state>/test_<state>.parquet
    phase2/outputs/<state>/phase2g_summary.md
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

# Chronological split years
TRAIN_YEARS = list(range(2014, 2018))   # 2014–2017
VAL_YEARS   = [2018]
TEST_YEARS  = [2019, 2020]


def compute_temporal_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos temporal encodings from window timestamp."""
    df = df.copy()
    ts = pd.to_datetime(df["date_utc"])
    df["sin_month"] = np.sin(2 * np.pi * ts.dt.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * ts.dt.month / 12)
    df["sin_hour"]  = np.sin(2 * np.pi * df["window_hour"] / 24)
    df["cos_hour"]  = np.cos(2 * np.pi * df["window_hour"] / 24)
    return df


def assemble(state_key: str, cfg: dict) -> bool:
    output_dir = cfg["output_dir"]
    slug       = state_key.lower()

    logger.info(f"{'=' * 60}")
    logger.info(f"PHASE 2G — {cfg['name'].upper()} — Dataset Assembly")
    logger.info(f"{'=' * 60}")

    # ── Load base labels ──────────────────────────────────────────────────────
    labels_path = output_dir / "full_training_labels.parquet"
    if not labels_path.exists():
        logger.error(f"Labels not found: {labels_path} — run Phase 2D first")
        return False

    df = pd.read_parquet(labels_path)
    logger.info(f"  Base labels: {len(df):,} rows  (pos={int((df.label==1).sum()):,}, neg={int((df.label==0).sum()):,})")

    # ── Join static features ──────────────────────────────────────────────────
    static_path = output_dir / f"static_features_{slug}.parquet"
    if static_path.exists():
        static_df = pd.read_parquet(static_path)
        static_df = static_df.drop(columns=["centroid_lat", "centroid_lon"], errors="ignore")
        before = len(df.columns)
        df = df.merge(static_df, on="h3_cell", how="left")
        added = len(df.columns) - before
        logger.info(f"  After static join: +{added} columns from {static_path.name}")
    else:
        logger.warning(f"  Static features not found: {static_path}")
        logger.warning("  Run Phase 2E first: python run_phase2e_static.py --state " + state_key)

    # ── Join gridMET features ─────────────────────────────────────────────────
    gridmet_path = output_dir / f"gridmet_features_{slug}.parquet"
    if gridmet_path.exists():
        gridmet_df = pd.read_parquet(gridmet_path)
        before = len(df.columns)
        df = df.merge(gridmet_df, on=["h3_cell", "date_utc"], how="left")
        added = len(df.columns) - before
        logger.info(f"  After gridMET join: +{added} columns from {gridmet_path.name}")
        # Report missing gridMET coverage
        pct_missing = df["erc"].isna().mean() * 100 if "erc" in df.columns else 100.0
        logger.info(f"  gridMET coverage: {100-pct_missing:.1f}% rows have valid erc")
    else:
        logger.warning(f"  gridMET features not found: {gridmet_path}")
        logger.warning("  Run Phase 2F first: python run_phase2f_gridmet.py --state " + state_key)

    # ── Add temporal encodings ────────────────────────────────────────────────
    df = compute_temporal_encodings(df)
    logger.info(f"  Temporal encodings added: sin_month, cos_month, sin_hour, cos_hour")

    # ── Leakage check — ensure no post-fire columns ───────────────────────────
    FORBIDDEN_COLS = [
        "FIRE_SIZE", "FIRE_SIZE_CLASS", "CONT_DATE", "CONT_DOY", "CONT_TIME",
        "NWCG_CAUSE_CLASSIFICATION", "NWCG_GENERAL_CAUSE",
        "MTBS_ID", "MTBS_FIRE_NAME", "ICS_209_PLUS_INCIDENT_JOIN_ID",
        "FIRE_NAME", "FOD_ID", "NWCG_CAUSE_AGE_CATEGORY",
    ]
    found_forbidden = [c for c in FORBIDDEN_COLS if c in df.columns]
    if found_forbidden:
        logger.error(f"  LEAKAGE DETECTED: {found_forbidden}")
        logger.error("  Dropping forbidden columns immediately...")
        df = df.drop(columns=found_forbidden, errors="ignore")
    else:
        logger.info(f"  Leakage check: PASSED (no forbidden columns found)")

    # ── Final column summary ──────────────────────────────────────────────────
    n_features = len(df.columns) - len(["h3_cell", "date_utc", "window_hour",
                                         "window_6h_utc", "label", "fire_year", "state"])
    logger.info(f"\n  Final shape: {df.shape}")
    logger.info(f"  Approx. feature columns: ~{n_features}")
    logger.info(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    # ── Save full dataset ─────────────────────────────────────────────────────
    full_path = output_dir / f"final_training_dataset_{slug}.parquet"
    df.to_parquet(full_path, index=False, compression="snappy")
    logger.info(f"\n  Full dataset saved: {full_path}  ({full_path.stat().st_size/1e6:.0f} MB)")

    # ── Chronological split ───────────────────────────────────────────────────
    train = df[df["fire_year"].isin(TRAIN_YEARS)]
    val   = df[df["fire_year"].isin(VAL_YEARS)]
    test  = df[df["fire_year"].isin(TEST_YEARS)]

    for split_df, name in [(train, "train"), (val, "val"), (test, "test")]:
        p = output_dir / f"{name}_{slug}.parquet"
        split_df.to_parquet(p, index=False, compression="snappy")
        n_pos = (split_df["label"] == 1).sum()
        n_neg = (split_df["label"] == 0).sum()
        logger.info(f"  {name.upper():<6}: {len(split_df):>8,} rows  "
                    f"(fire={n_pos:,}, non-fire={n_neg:,})  → {p.name}")

    # ── Generate summary ──────────────────────────────────────────────────────
    summary = f"""# Phase 2G Summary — {cfg['name']}

## Dataset Statistics

| Split | Years | Total Rows | Fire (label=1) | Non-fire (label=0) |
|---|---|---|---|---|
| TRAIN | 2014–2017 | {len(train):,} | {int((train.label==1).sum()):,} | {int((train.label==0).sum()):,} |
| VAL   | 2018      | {len(val):,} | {int((val.label==1).sum()):,} | {int((val.label==0).sum()):,} |
| TEST  | 2019–2020 | {len(test):,} | {int((test.label==1).sum()):,} | {int((test.label==0).sum()):,} |
| **TOTAL** | 2014–2020 | **{len(df):,}** | **{int((df.label==1).sum()):,}** | **{int((df.label==0).sum()):,}** |

## Feature Columns ({n_features} total)
{', '.join([c for c in df.columns if c not in ['h3_cell','label','fire_year','state','date_utc','window_hour','window_6h_utc']][:40])}...

## Leakage Audit
- Forbidden columns found: {found_forbidden if found_forbidden else 'NONE ✔'}

## Files
- Full: `final_training_dataset_{slug}.parquet`
- Train: `train_{slug}.parquet`
- Val: `val_{slug}.parquet`
- Test: `test_{slug}.parquet`
"""
    (output_dir / "phase2g_summary.md").write_text(summary, encoding="utf-8")
    logger.info(f"  Summary saved: phase2g_summary.md")

    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 2G — Dataset Assembly")
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
        print(f"  {STATE_CONFIG[s]['name']:<15} {'✔ SUCCESS' if ok else '✘ FAILED'}")


if __name__ == "__main__":
    main()
