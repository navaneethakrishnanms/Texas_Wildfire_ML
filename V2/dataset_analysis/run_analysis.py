"""
run_analysis.py
---------------
Main entry point for the Wildfire Dataset Analysis Pipeline.

Runs all 15 analyses separately for Texas and/or California using
the pre-processed datasets at:
  V2/data/processed/texas/texas_fire_2014_2020.parquet
  V2/data/processed/california/california_fire_2014_2020.parquet

All outputs are written to per-state sub-directories:
  tables/texas/     tables/california/
  plots/texas/      plots/california/
  reports/texas/    reports/california/

Usage
-----
  # Run for BOTH states (default)
  conda run -n torch_gpu python run_analysis.py

  # Run for Texas only
  conda run -n torch_gpu python run_analysis.py --state TX

  # Run for California only
  conda run -n torch_gpu python run_analysis.py --state CA

  # Skip the slow correlation step
  conda run -n torch_gpu python run_analysis.py --skip-correlation

  # Quick test on 5,000 rows
  conda run -n torch_gpu python run_analysis.py --sample 5000

  # Run only specific analysis steps
  conda run -n torch_gpu python run_analysis.py --only overview,schema,missing,quality

  # Show help
  conda run -n torch_gpu python run_analysis.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

# ── Ensure dataset_analysis/ is on sys.path ───────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# ── Imports ───────────────────────────────────────────────────────────────────
from config.config import (
    ANALYSIS_ROOT,
    LOGS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    STATE_DATASETS,
    LOG_FILE,
)
from src.utils import ensure_dirs, setup_logger, timer

# Create top-level output dirs
ensure_dirs(LOGS_DIR, TABLES_DIR, REPORTS_DIR, PLOTS_DIR)

logger = setup_logger("run_analysis", LOG_FILE)

# ─────────────────────────────────────────────────────────────────────────────
# Available Analysis Steps
# ─────────────────────────────────────────────────────────────────────────────

ALL_STEPS = [
    "overview",       # 1  — Dataset Overview
    "schema",         # 2  — Schema Analysis
    "categorize",     # 3  — Feature Categorization
    "missing",        # 4  — Missing Value Analysis
    "quality",        # 5  — Feature Quality Analysis
    "statistics",     # 6  — Statistical Analysis
    "temporal",       # 7  — Temporal Analysis
    "geographic",     # 8  — Geographic Analysis
    "correlation",    # 9  — Correlation Analysis
    "categorical",    # 10 — Categorical Analysis
    "leakage",        # 11 — Leakage Analysis
    "readiness",      # 12 — Predictive Readiness
    "groups",         # 13 — Feature Dependency Groups
    "source",         # 14 — Source Readiness
    "report",         # 15 — Final Report
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_analysis.py",
        description=(
            "Wildfire Dataset Analysis Pipeline — "
            "EDA for Texas and California (2014–2020)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
States:
  TX  →  V2/data/processed/texas/texas_fire_2014_2020.parquet
  CA  →  V2/data/processed/california/california_fire_2014_2020.parquet

Steps (--only):
  {', '.join(ALL_STEPS)}

Examples:
  conda run -n torch_gpu python run_analysis.py
  conda run -n torch_gpu python run_analysis.py --state TX
  conda run -n torch_gpu python run_analysis.py --state CA
  conda run -n torch_gpu python run_analysis.py --skip-correlation
  conda run -n torch_gpu python run_analysis.py --sample 5000
  conda run -n torch_gpu python run_analysis.py --only overview,schema,missing,quality,leakage,report
""",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="BOTH",
        choices=["TX", "CA", "BOTH"],
        help="Which state dataset to analyze. Default: BOTH (TX then CA).",
    )
    parser.add_argument(
        "--skip-correlation",
        action="store_true",
        help="Skip the correlation step (can be slow for 250+ numeric columns).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="STEPS",
        help=(
            "Comma-separated list of steps to run. All others are skipped. "
            f"Available: {', '.join(ALL_STEPS)}"
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N rows (for quick testing).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Per-State Directory Builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_state_dirs(state_slug: str) -> dict[str, Path]:
    """
    Return a dict of per-state output directory paths.

    state_slug : lowercase state name, e.g. 'texas' or 'california'.
    """
    tables  = TABLES_DIR  / state_slug
    plots   = PLOTS_DIR   / state_slug
    reports = REPORTS_DIR / state_slug

    dirs = {
        "tables":      tables,
        "reports":     reports,
        "plots_miss":  plots / "missing",
        "plots_stats": plots / "statistics",
        "plots_time":  plots / "temporal",
        "plots_geo":   plots / "geographic",
        "plots_corr":  plots / "correlation",
        "plots_cat":   plots / "categorical",
    }
    for d in dirs.values():
        ensure_dirs(d)
    return dirs


# ─────────────────────────────────────────────────────────────────────────────
# Step Runner (fault-tolerant)
# ─────────────────────────────────────────────────────────────────────────────

def _step(name: str, active_steps: set[str], fn, *args, **kwargs):
    """
    Execute analysis step *fn* if *name* is active.
    Exceptions are caught and logged — pipeline continues regardless.
    """
    if name not in active_steps:
        logger.info(f"[SKIP] {name}")
        return None

    print(f"\n{'━' * 58}")
    print(f"  ► Step {ALL_STEPS.index(name) + 1:02d}: {name.upper()}")
    print(f"{'━' * 58}")

    try:
        with timer(f"Step '{name}'", logger):
            result = fn(*args, **kwargs)
        logger.info(f"[DONE] {name}")
        return result
    except Exception:
        logger.error(f"[ERROR] Step '{name}' failed:\n{traceback.format_exc()}")
        print(f"\n  ⚠️  Step '{name}' failed (see log). Continuing...\n")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Single-State Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_for_state(
    state_code: str,
    active_steps: set[str],
    sample_size: int | None,
) -> None:
    """
    Run the complete EDA pipeline for one state.

    Parameters
    ----------
    state_code   : 'TX' or 'CA'
    active_steps : Set of step names to execute.
    sample_size  : If set, only the first N rows are used.
    """
    state_cfg  = STATE_DATASETS[state_code]
    state_name = state_cfg["name"]
    state_slug = state_name.lower()      # 'texas' or 'california'

    print(f"\n{'═' * 58}")
    print(f"  STATE: {state_name.upper()} ({state_code})")
    print(f"{'═' * 58}")
    logger.info(f"{'=' * 60}")
    logger.info(f"  STARTING ANALYSIS — {state_name.upper()} ({state_code})")
    logger.info(f"{'=' * 60}")

    # Build per-state output paths and inject them into sub-modules
    dirs = _build_state_dirs(state_slug)

    # Monkey-patch config-level path variables used by sub-modules
    # Each module reads from config at import time, but paths are passed
    # as arguments wherever possible, so we patch only the fallbacks.
    import config.config as cfg_mod
    cfg_mod.TABLES_DIR          = dirs["tables"]
    cfg_mod.REPORTS_DIR         = dirs["reports"]
    cfg_mod.PLOTS_MISSING_DIR   = dirs["plots_miss"]
    cfg_mod.PLOTS_STATS_DIR     = dirs["plots_stats"]
    cfg_mod.PLOTS_TEMPORAL_DIR  = dirs["plots_time"]
    cfg_mod.PLOTS_GEOGRAPHIC_DIR= dirs["plots_geo"]
    cfg_mod.PLOTS_CORRELATION_DIR = dirs["plots_corr"]
    cfg_mod.PLOTS_CATEGORICAL_DIR = dirs["plots_cat"]
    cfg_mod.FINAL_REPORT_MD     = dirs["reports"] / "final_summary_report.md"
    cfg_mod.FINAL_REPORT_PDF    = dirs["reports"] / "final_summary_report.pdf"

    # ── Load Dataset ──────────────────────────────────────────────────────────
    from src.load_data import load_state_dataset
    with timer(f"Load {state_name}", logger):
        df = load_state_dataset(state_code)

    if sample_size and sample_size < len(df):
        logger.info(f"  Using sample of {sample_size:,} rows.")
        df = df.head(sample_size).copy()

    print(f"\n  ✔ Dataset loaded: {len(df):,} rows × {len(df.columns)} columns")

    # ── Step 1: Overview ──────────────────────────────────────────────────────
    from src.load_data import generate_dataset_overview
    _step("overview", active_steps,
          generate_dataset_overview, df, dirs["tables"], state_name)

    # ── Step 2: Schema ────────────────────────────────────────────────────────
    from src.schema_analysis import generate_schema_analysis
    schema_df = _step("schema", active_steps, generate_schema_analysis, df)

    # ── Step 3: Feature Categorization ────────────────────────────────────────
    from src.schema_analysis import generate_feature_categorization
    _step("categorize", active_steps, generate_feature_categorization, df)

    # ── Step 4: Missing Value Analysis ────────────────────────────────────────
    from src.missing_analysis import generate_missing_analysis
    missing_df = _step("missing", active_steps, generate_missing_analysis, df)

    # ── Step 5: Feature Quality ───────────────────────────────────────────────
    from src.feature_analysis import generate_feature_quality_analysis
    quality_df = _step("quality", active_steps, generate_feature_quality_analysis, df)

    # ── Step 6: Statistical Analysis ──────────────────────────────────────────
    from src.statistical_analysis import generate_statistical_analysis
    _step("statistics", active_steps, generate_statistical_analysis, df)

    # ── Step 7: Temporal Analysis ─────────────────────────────────────────────
    from src.temporal_analysis import generate_temporal_analysis
    _step("temporal", active_steps, generate_temporal_analysis, df)

    # ── Step 8: Geographic Analysis ───────────────────────────────────────────
    from src.geographic_analysis import generate_geographic_analysis
    _step("geographic", active_steps, generate_geographic_analysis, df)

    # ── Step 9: Correlation Analysis ──────────────────────────────────────────
    from src.correlation_analysis import generate_correlation_analysis
    _step("correlation", active_steps, generate_correlation_analysis, df)

    # ── Step 10: Categorical Analysis ─────────────────────────────────────────
    from src.feature_group_analysis import generate_categorical_analysis
    _step("categorical", active_steps, generate_categorical_analysis, df)

    # ── Step 11: Leakage Analysis ─────────────────────────────────────────────
    from src.leakage_analysis import generate_leakage_analysis
    leakage_df = _step(
        "leakage", active_steps, generate_leakage_analysis,
        df,
        quality_df if quality_df is not None else None,
        missing_df if missing_df is not None else None,
    )

    # ── Step 12: Predictive Readiness ─────────────────────────────────────────
    from src.leakage_analysis import generate_predictive_readiness
    if leakage_df is not None:
        _step("readiness", active_steps, generate_predictive_readiness, leakage_df)

    # ── Step 13: Feature Dependency Groups ────────────────────────────────────
    from src.feature_group_analysis import generate_feature_dependency_analysis
    _step("groups", active_steps, generate_feature_dependency_analysis, df)

    # ── Step 14: Source Readiness ─────────────────────────────────────────────
    from src.feature_group_analysis import generate_source_readiness_analysis
    _step("source", active_steps, generate_source_readiness_analysis, df)

    # ── Step 15: Final Report ─────────────────────────────────────────────────
    from src.report_generator import generate_final_report
    report_path = _step("report", active_steps, generate_final_report)

    # ── State Complete Banner ─────────────────────────────────────────────────
    print(f"\n{'═' * 58}")
    print(f"  ✅  {state_name.upper()} ANALYSIS COMPLETE")
    print(f"{'═' * 58}")
    print(f"  📁 Tables  → {dirs['tables']}")
    print(f"  📊 Plots   → {PLOTS_DIR / state_slug}")
    print(f"  📝 Report  → {dirs['reports']}")
    if report_path:
        print(f"  📄 Report  → {report_path}")
    print(f"{'═' * 58}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # ── Determine states to run ───────────────────────────────────────────────
    if args.state == "BOTH":
        states = list(STATE_DATASETS.keys())   # ['TX', 'CA']
    else:
        states = [args.state.upper()]

    # ── Determine active steps ────────────────────────────────────────────────
    if args.only:
        requested = {s.strip().lower() for s in args.only.split(",")}
        invalid   = requested - set(ALL_STEPS)
        if invalid:
            print(f"ERROR: Unknown steps: {invalid}\nValid: {ALL_STEPS}")
            return 1
        active_steps = requested
    else:
        active_steps = set(ALL_STEPS)

    if args.skip_correlation:
        active_steps.discard("correlation")
        logger.info("Correlation step disabled via --skip-correlation.")

    # ── Header ────────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("  WILDFIRE DATASET ANALYSIS PIPELINE — START")
    logger.info("=" * 70)
    logger.info(f"  States       : {states}")
    logger.info(f"  Active steps : {sorted(active_steps)}")
    logger.info(f"  Sample size  : {args.sample or 'Full dataset'}")

    print(f"\n{'#' * 58}")
    print(f"  WILDFIRE DATASET ANALYSIS PIPELINE")
    print(f"  States: {' + '.join(states)}")
    print(f"  Steps : {len(active_steps)} active")
    print(f"{'#' * 58}")

    # ── Run each state ────────────────────────────────────────────────────────
    for state_code in states:
        try:
            run_for_state(state_code, active_steps, args.sample)
        except FileNotFoundError as exc:
            print(f"\n  ❌ ERROR: {exc}\n")
            logger.error(str(exc))
        except Exception:
            logger.error(
                f"Unhandled error for state {state_code}:\n{traceback.format_exc()}"
            )
            print(f"\n  ❌ Unexpected error for {state_code} (see log). Skipping.\n")

    # ── Final Banner ──────────────────────────────────────────────────────────
    print(f"\n{'#' * 58}")
    print(f"  🏁  ALL STATES COMPLETE")
    print(f"{'#' * 58}")
    print(f"  📁 All outputs → {ANALYSIS_ROOT}")
    print(f"  📋 Log file    → {LOG_FILE}")
    print(f"{'#' * 58}\n")

    logger.info("=" * 70)
    logger.info("  WILDFIRE DATASET ANALYSIS PIPELINE — COMPLETE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
