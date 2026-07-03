"""
run_phase2a.py
---------------
Phase 2A — Feature Finalization Pipeline Entry Point

Runs all 4 missing Phase 1 analyses + Gate 1 + Gate 2 for a given state.

Usage:
    python run_phase2a.py --state TX
    python run_phase2a.py --state CA
    python run_phase2a.py --state ALL

Outputs (per state):
    phase2/outputs/<state>/feature_source_map.csv
    phase2/outputs/<state>/missing_root_cause.csv
    phase2/outputs/<state>/production_availability.csv
    phase2/outputs/<state>/static_dynamic_features.csv
    phase2/outputs/<state>/gate1_removed.csv
    phase2/outputs/<state>/gate2_removed.csv
    phase2/outputs/<state>/feature_schema.csv        ← MASTER CONTRACT
    phase2/outputs/<state>/phase2a_summary.md
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
PHASE2_ROOT = Path(__file__).resolve().parent
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

# ── Imports ───────────────────────────────────────────────────────────────────
from config.phase2_config import STATE_CONFIG, LOGS_DIR

from src.feature_source_map        import generate_feature_source_map
from src.missing_root_cause         import generate_missing_root_cause
from src.production_availability    import generate_production_availability
from src.static_dynamic_classifier  import generate_static_dynamic_classification
from src.feature_finalizer          import generate_feature_schema


# ── Logging setup ─────────────────────────────────────────────────────────────
def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "phase2a.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.info(f"Log file: {log_file}")


logger = logging.getLogger(__name__)


# ── Per-state runner ───────────────────────────────────────────────────────────
def run_state(state_key: str) -> bool:
    """Run the full Phase 2A pipeline for one state. Returns True on success."""
    cfg        = STATE_CONFIG[state_key]
    state_name = cfg["name"]
    tables_dir = cfg["tables_dir"]
    output_dir = cfg["output_dir"]

    logger.info("=" * 60)
    logger.info(f"PHASE 2A — {state_name.upper()}")
    logger.info("=" * 60)

    # ── Input CSV paths (from Phase 1 tables/) ────────────────────────────────
    schema_csv        = tables_dir / "schema_analysis.csv"
    missing_csv       = tables_dir / "missing_summary.csv"
    leakage_csv       = tables_dir / "leakage_analysis.csv"
    quality_csv       = tables_dir / "feature_quality.csv"
    corr_pearson_csv  = tables_dir / "highly_correlated_pairs_pearson.csv"
    # Additional Phase 1 tables available for richer analysis
    constant_csv      = tables_dir / "constant_columns.csv"
    duplicate_csv     = tables_dir / "duplicate_columns.csv"
    readiness_csv     = tables_dir / "predictive_readiness.csv"

    # Check which inputs exist
    for label, path in [
        ("schema_analysis", schema_csv),
        ("missing_summary", missing_csv),
        ("leakage_analysis", leakage_csv),
        ("feature_quality",  quality_csv),
        ("corr_pearson",     corr_pearson_csv),
    ]:
        status = "✔" if path.exists() else "✘ NOT FOUND"
        logger.info(f"  Input [{label}]: {status}")

    steps_ok = []

    # ── Analysis A: Feature Source Mapping ───────────────────────────────────
    t0 = time.time()
    try:
        logger.info("\n[Step 1/6] Analysis A — Feature Source Mapping")
        source_map_df = generate_feature_source_map(
            schema_csv  = schema_csv,
            output_dir  = output_dir,
            state_name  = state_name,
        )
        steps_ok.append(("Analysis A", True, time.time() - t0))
    except Exception as exc:
        logger.error(f"  Analysis A FAILED: {exc}", exc_info=True)
        steps_ok.append(("Analysis A", False, time.time() - t0))

    # ── Analysis B: Missing Root Cause ────────────────────────────────────────
    t0 = time.time()
    try:
        logger.info("\n[Step 2/6] Analysis B — Missing Root Cause")
        missing_cause_df = generate_missing_root_cause(
            missing_csv = missing_csv,
            schema_csv  = schema_csv,
            output_dir  = output_dir,
            state_name  = state_name,
        )
        steps_ok.append(("Analysis B", True, time.time() - t0))
    except Exception as exc:
        logger.error(f"  Analysis B FAILED: {exc}", exc_info=True)
        steps_ok.append(("Analysis B", False, time.time() - t0))

    # ── Analysis C: Production Availability ───────────────────────────────────
    t0 = time.time()
    try:
        logger.info("\n[Step 3/6] Analysis C — Production Availability")
        avail_df = generate_production_availability(
            leakage_csv    = leakage_csv,
            source_map_csv = output_dir / "feature_source_map.csv",
            output_dir     = output_dir,
            state_name     = state_name,
        )
        steps_ok.append(("Analysis C", True, time.time() - t0))
    except Exception as exc:
        logger.error(f"  Analysis C FAILED: {exc}", exc_info=True)
        steps_ok.append(("Analysis C", False, time.time() - t0))

    # ── Analysis D: Static vs Dynamic ─────────────────────────────────────────
    t0 = time.time()
    try:
        logger.info("\n[Step 4/6] Analysis D — Static vs Dynamic Classification")
        sd_df = generate_static_dynamic_classification(
            source_map_csv   = output_dir / "feature_source_map.csv",
            availability_csv = output_dir / "production_availability.csv",
            output_dir       = output_dir,
            state_name       = state_name,
        )
        steps_ok.append(("Analysis D", True, time.time() - t0))
    except Exception as exc:
        logger.error(f"  Analysis D FAILED: {exc}", exc_info=True)
        steps_ok.append(("Analysis D", False, time.time() - t0))

    # ── Gate 1 + Gate 2 → feature_schema.csv ─────────────────────────────────
    t0 = time.time()
    try:
        logger.info("\n[Step 5/6] Gate 1 — Production Feasibility Filter")
        logger.info("[Step 6/6] Gate 2 — Correlation Redundancy Filter")
        schema_df = generate_feature_schema(
            source_map_csv    = output_dir / "feature_source_map.csv",
            missing_cause_csv = output_dir / "missing_root_cause.csv",
            availability_csv  = output_dir / "production_availability.csv",
            static_dynamic_csv= output_dir / "static_dynamic_features.csv",
            corr_pearson_csv  = corr_pearson_csv,
            quality_csv       = quality_csv,
            output_dir        = output_dir,
            state_name        = state_name,
        )
        steps_ok.append(("Gate 1+2 Finalizer", True, time.time() - t0))
    except Exception as exc:
        logger.error(f"  Feature Finalizer FAILED: {exc}", exc_info=True)
        steps_ok.append(("Gate 1+2 Finalizer", False, time.time() - t0))

    # ── Final step summary ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"PHASE 2A STEP SUMMARY [{state_name.upper()}]")
    logger.info("=" * 60)
    all_ok = True
    for step, ok, elapsed in steps_ok:
        status = "✔ PASS" if ok else "✘ FAIL"
        logger.info(f"  {status}  {step:<30}  ({elapsed:.2f}s)")
        if not ok:
            all_ok = False

    logger.info("=" * 60)
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  feature_schema.csv: {output_dir / 'feature_schema.csv'}")
    logger.info("=" * 60)

    return all_ok


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2A — Feature Finalization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_phase2a.py --state TX      # Run for Texas only
  python run_phase2a.py --state CA      # Run for California only
  python run_phase2a.py --state ALL     # Run both states
        """,
    )
    parser.add_argument(
        "--state",
        choices=["TX", "CA", "ALL"],
        required=True,
        help="State to process: TX (Texas), CA (California), or ALL (both)",
    )
    args = parser.parse_args()

    setup_logging(LOGS_DIR)

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║     WILDFIRE FORECASTING SYSTEM — PHASE 2A              ║")
    logger.info("║     Feature Finalization + Missing Analysis              ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    states_to_run = ["TX", "CA"] if args.state == "ALL" else [args.state]

    results = {}
    total_start = time.time()

    for state_key in states_to_run:
        if state_key not in STATE_CONFIG:
            logger.error(f"Unknown state: {state_key}")
            continue
        ok = run_state(state_key)
        results[state_key] = ok

    total_elapsed = time.time() - total_start

    logger.info("\n" + "═" * 60)
    logger.info("  PHASE 2A COMPLETE")
    logger.info("═" * 60)
    for state_key, ok in results.items():
        status = "✔ SUCCESS" if ok else "✘ PARTIAL FAILURE"
        logger.info(f"  {STATE_CONFIG[state_key]['name']:<15} {status}")
    logger.info(f"  Total time: {total_elapsed:.1f}s")
    logger.info("═" * 60)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
