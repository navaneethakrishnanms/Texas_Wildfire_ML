"""
pipeline.py
-----------
Phase-1 preprocessing pipeline orchestrator.

Wires together all modules:
    loader -> schema_checker -> merger -> state_filter
    -> validator -> quality_reporter -> eda_reporter

Run directly:
    conda run -n torch_gpu python -m src.preprocessing.pipeline

Or import and call:
    from src.preprocessing.pipeline import run_pipeline
    run_pipeline()
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Force UTF-8 for stdout/stderr on Windows to avoid cp1252 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from .config import DATA_RAW_DIR, LOGS_DIR, TARGET_STATES
from .eda_reporter import generate_eda_report
from .loader import load_raw_datasets
from .logger import get_logger, setup_logging
from .merger import merge_datasets, print_master_summary
from .quality_reporter import generate_quality_report
from .schema_checker import save_schema_report, verify_schema
from .state_filter import filter_by_state, save_state_dataset
from .validator import save_validation_report, validate_dataframe

log = get_logger(__name__)


def run_pipeline(raw_dir: Path | None = None) -> None:
    """
    Execute the full Phase-1 preprocessing pipeline.

    Parameters
    ----------
    raw_dir : Path | None
        Override the default raw data directory.
        If None, uses config.DATA_RAW_DIR.
    """
    t0 = time.perf_counter()

    setup_logging(LOGS_DIR)
    log.info("=" * 52)
    log.info("   Wildfire Ignition Prediction System - Phase 1")
    log.info("=" * 52)

    raw_dir = Path(raw_dir) if raw_dir else DATA_RAW_DIR

    # -- Step 1 & 2: Load + standardise
    log.info("-" * 60)
    log.info("STEP 1 & 2 - Load and standardise raw datasets")
    log.info("-" * 60)
    try:
        datasets = load_raw_datasets(raw_dir)
    except FileNotFoundError as exc:
        log.critical("Cannot find raw data: %s", exc)
        sys.exit(1)

    # -- Step 3: Schema verification
    log.info("-" * 60)
    log.info("STEP 3 - Schema verification")
    log.info("-" * 60)
    schema_report = verify_schema(datasets)

    # -- Step 4: Merge
    log.info("-" * 60)
    log.info("STEP 4 - Merge all years into master DataFrame")
    log.info("-" * 60)
    master = merge_datasets(datasets, schema_report)
    print_master_summary(master)

    # Free yearly datasets from memory
    del datasets

    # -- Step 5 + per-state processing
    for state_code, state_cfg in TARGET_STATES.items():
        state_name = state_cfg["name"]
        report_dir = state_cfg["report_dir"]

        log.info("=" * 60)
        log.info("PROCESSING STATE: %s (%s)", state_name, state_code)
        log.info("=" * 60)

        # 5a. Filter
        state_df = filter_by_state(master, state_code)

        if len(state_df) == 0:
            log.warning(
                "No records found for %s - skipping report generation.", state_name
            )
            continue

        # 5b. Save datasets
        save_state_dataset(state_df, state_cfg)

        # Save schema report per state
        save_schema_report(schema_report, report_dir, state_code)

        # Data Validation
        log.info("-" * 50)
        log.info("DATA VALIDATION - %s", state_name)
        log.info("-" * 50)
        val_report = validate_dataframe(state_df, state_code)
        save_validation_report(val_report, report_dir)

        # Quality Report
        log.info("-" * 50)
        log.info("QUALITY REPORT - %s", state_name)
        log.info("-" * 50)
        generate_quality_report(state_df, state_code, state_name, report_dir)

        # EDA Report
        log.info("-" * 50)
        log.info("EDA REPORT - %s", state_name)
        log.info("-" * 50)
        generate_eda_report(state_df, state_code, state_name, report_dir)

    # -- Done
    elapsed = time.perf_counter() - t0
    log.info("=" * 60)
    log.info("[DONE] Phase-1 complete in %.1f seconds (%.1f min).", elapsed, elapsed / 60)
    log.info("=" * 60)
    _print_output_summary()


def _print_output_summary() -> None:
    """Print a table of all generated output files."""
    from .config import (
        CALI_REPORTS_DIR,
        CALIFORNIA_DIR,
        TEXAS_DIR,
        TEXAS_REPORTS_DIR,
    )

    print("\n" + "-" * 72)
    print("  OUTPUT FILES GENERATED")
    print("-" * 72)

    output_groups = {
        "Texas - Processed Data":     TEXAS_DIR,
        "California - Processed Data": CALIFORNIA_DIR,
        "Texas - Reports":             TEXAS_REPORTS_DIR,
        "California - Reports":        CALI_REPORTS_DIR,
    }

    for group, directory in output_groups.items():
        dir_path = Path(directory)
        if dir_path.exists():
            files = sorted(dir_path.iterdir())
            print(f"\n  {group}/")
            for f in files:
                size_mb = f.stat().st_size / 1024 ** 2
                print(f"    [OK]  {f.name:<55} {size_mb:>8.2f} MB")
        else:
            print(f"\n  {group}/ -> (not created)")

    print("\n" + "-" * 72)


if __name__ == "__main__":
    run_pipeline()
