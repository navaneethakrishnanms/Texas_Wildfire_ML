"""
run_phase2b.py
---------------
Phase 2B — H3 Grid Construction Entry Point

Builds the spatial H3 grid for Texas (R7) and California (R8).
This grid is the spatial foundation for all downstream phases:
  Phase 2C → Fire=1 cell-days are mapped onto this grid
  Phase 2D → Fire=0 cells are sampled from this grid (DAY-ECO-MATCHED)
  Phase 2E → Final training table references this grid
  Phase 6  → Operational prediction runs over this grid daily

Usage:
    conda activate torch_gpu
    python run_phase2b.py --state TX
    python run_phase2b.py --state CA
    python run_phase2b.py --state ALL

Outputs (per state):
    outputs/<state>/h3_grid_<state>.parquet           ← Master grid (main output)
    outputs/<state>/h3_ecoregion_breakdown_<state>.csv
    outputs/<state>/h3_grid_summary_<state>.csv

Requirements:
    pip install h3 scipy

Note:
    h3 is not installed by default in the conda environment.
    Run:  conda activate torch_gpu && pip install h3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Ensure phase2 root is on path ─────────────────────────────────────────────
PHASE2_ROOT = Path(__file__).resolve().parent
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from config.phase2_config import STATE_CONFIG, LOGS_DIR
from src.grid_builder import build_h3_grid


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "phase2b.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("fiona").setLevel(logging.WARNING)
    logging.info(f"Log file: {log_file}")


logger = logging.getLogger(__name__)


# ── Per-state runner ───────────────────────────────────────────────────────────
def run_state(state_key: str) -> bool:
    """Run Phase 2B for one state. Returns True on success."""
    cfg        = STATE_CONFIG[state_key]
    state_name = cfg["name"]
    resolution = cfg["h3_level"]
    parquet    = cfg["parquet"]
    output_dir = cfg["output_dir"]

    logger.info("=" * 60)
    logger.info(f"PHASE 2B — {state_name.upper()} — H3 Grid Construction")
    logger.info("=" * 60)
    logger.info(f"  H3 resolution : R{resolution}")
    logger.info(f"  Fire parquet  : {parquet}")
    logger.info(f"  Output dir    : {output_dir}")

    if not parquet.exists():
        logger.error(
            f"  Fire parquet not found: {parquet}\n"
            f"  Make sure Phase 1 data processing is complete before running Phase 2B."
        )
        return False

    t0 = time.time()
    try:
        grid_df = build_h3_grid(
            state_key    = state_key,
            resolution   = resolution,
            parquet_path = parquet,
            output_dir   = output_dir,
        )
        elapsed = time.time() - t0
        logger.info(f"  ✔ Phase 2B [{state_name}] complete in {elapsed:.1f}s")
        logger.info(f"  Grid size: {len(grid_df):,} cells")
        return True

    except ImportError as e:
        logger.error(f"\n  {'=' * 50}")
        logger.error(f"  MISSING DEPENDENCY: {e}")
        logger.error(f"  {'=' * 50}")
        logger.error("  Install required packages and re-run:")
        logger.error("    conda activate torch_gpu")
        logger.error("    pip install h3 scipy")
        return False

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"  Phase 2B FAILED after {elapsed:.1f}s: {e}", exc_info=True)
        return False


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2B — H3 Grid Construction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_phase2b.py --state TX     # Texas only (H3-R7)
  python run_phase2b.py --state CA     # California only (H3-R8)
  python run_phase2b.py --state ALL    # Both states

Required packages (install once):
  pip install h3 scipy
        """,
    )
    parser.add_argument(
        "--state",
        choices=["TX", "CA", "ALL"],
        required=True,
        help="State to process",
    )
    args = parser.parse_args()

    setup_logging(LOGS_DIR)

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║     WILDFIRE FORECASTING SYSTEM — PHASE 2B              ║")
    logger.info("║     H3 Grid Construction                                 ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    states = ["TX", "CA"] if args.state == "ALL" else [args.state]

    results = {}
    total_start = time.time()

    for state_key in states:
        results[state_key] = run_state(state_key)

    total_elapsed = time.time() - total_start

    logger.info("\n" + "═" * 60)
    logger.info("  PHASE 2B RESULTS")
    logger.info("═" * 60)
    for sk, ok in results.items():
        icon = "✔ SUCCESS" if ok else "✘ FAILED"
        logger.info(f"  {STATE_CONFIG[sk]['name']:<15} {icon}")
    logger.info(f"  Total time: {total_elapsed:.1f}s")
    logger.info("═" * 60)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
