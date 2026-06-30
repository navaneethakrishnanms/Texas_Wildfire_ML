"""
main.py — Central CLI Orchestration Entrypoint
===============================================
Short-Term Wildfire Ignition Risk Prediction POC

Usage
-----
Run the full pipeline end-to-end:
    python main.py --config configs/config.yaml --phase all

Run individual phases:
    python main.py --config configs/config.yaml --phase ingest
    python main.py --config configs/config.yaml --phase harmonize
    python main.py --config configs/config.yaml --phase features
    python main.py --config configs/config.yaml --phase prepare
    python main.py --config configs/config.yaml --phase train
    python main.py --config configs/config.yaml --phase evaluate
    python main.py --config configs/config.yaml --phase explain

Skip simulation (use real raw data already in data/raw/):
    python main.py --config configs/config.yaml --phase ingest --no-simulate
"""

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

from src.utils.config_utils import load_config
from src.utils.logging_utils import setup_logging

# ---------------------------------------------------------------------------
# Pipeline phase imports
# ---------------------------------------------------------------------------
from src.pipelines.data_ingestion_pipeline import run_ingestion_pipeline
from src.pipelines.data_harmonization_pipeline import run_harmonization_pipeline
from src.pipelines.feature_engineering_pipeline import run_feature_engineering_pipeline
from src.pipelines.data_preparation_pipeline import run_preparation_pipeline
from src.pipelines.model_training_pipeline import run_training_pipeline
from src.pipelines.model_evaluation_pipeline import run_evaluation_pipeline
from src.pipelines.explainability_pipeline import run_explainability_pipeline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wildfire Ignition Risk Prediction — Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "ingest", "harmonize", "features", "prepare", "train", "evaluate", "explain"],
        help="Which pipeline phase to execute (default: all)",
    )
    parser.add_argument(
        "--no-simulate",
        action="store_true",
        default=False,
        help="Skip geospatial simulation; rely on pre-existing raw data files.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def run_phase(phase: str, config: dict, simulate: bool) -> None:
    """Dispatch a single named phase or the full end-to-end pipeline."""

    phases = {
        "ingest":    lambda: run_ingestion_pipeline(config, force_simulate=simulate),
        "harmonize": lambda: run_harmonization_pipeline(config),
        "features":  lambda: run_feature_engineering_pipeline(config),
        "prepare":   lambda: run_preparation_pipeline(config),
        "train":     lambda: run_training_pipeline(config),
        "evaluate":  lambda: run_evaluation_pipeline(config),
        "explain":   lambda: run_explainability_pipeline(config),
    }

    if phase == "all":
        ordered = ["ingest", "harmonize", "features", "prepare", "train", "evaluate", "explain"]
        for p in ordered:
            t0 = time.perf_counter()
            logger.info(f"\n{'='*55}")
            logger.info(f"  STARTING PHASE: {p.upper()}")
            logger.info(f"{'='*55}")
            phases[p]()
            elapsed = time.perf_counter() - t0
            logger.info(f"  Phase '{p}' completed in {elapsed:.1f}s")
    else:
        phases[phase]()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load config and initialise logging
    config = load_config(args.config)
    setup_logging(config["paths"]["log_file"])

    simulate = not args.no_simulate

    logger.info("=" * 55)
    logger.info("  WILDFIRE IGNITION RISK PREDICTION — POC")
    logger.info("=" * 55)
    logger.info(f"  Config  : {args.config}")
    logger.info(f"  Phase   : {args.phase}")
    logger.info(f"  Simulate: {simulate}")
    logger.info("=" * 55)

    t_start = time.perf_counter()
    try:
        run_phase(args.phase, config, simulate)
    except Exception as exc:
        logger.exception(f"Pipeline failed with error: {exc}")
        sys.exit(1)

    total = time.perf_counter() - t_start
    logger.info(f"\nAll requested phases completed in {total:.1f}s.")
    logger.info("Artifacts saved:")
    logger.info(f"  Models   → {config['paths']['models_dir']}/")
    logger.info(f"  Outputs  → {config['paths']['outputs_dir']}/")
    logger.info(f"  Logs     → {config['paths']['log_file']}")


if __name__ == "__main__":
    main()
