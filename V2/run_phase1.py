"""
run_phase1.py
-------------
Entry-point script for the Phase-1 preprocessing pipeline.

Usage
-----
    conda run -n torch_gpu python run_phase1.py
    conda run -n torch_gpu python run_phase1.py --raw-dir /path/to/data
    conda run -n torch_gpu python run_phase1.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_phase1.py",
        description=(
            "Phase-1 Preprocessing Pipeline - "
            "Wildfire Ignition Prediction System (Texas & California)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run with default data directory (V2/data/):
      conda run -n torch_gpu python run_phase1.py

  Run with a custom raw data directory:
      conda run -n torch_gpu python run_phase1.py --raw-dir C:/data/wildfire
""",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Directory containing the raw FPA-FOD yearly CSV/XLSX files. "
            "Defaults to V2/data/ relative to this script."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure the package is importable when run from V2/
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.preprocessing.pipeline import run_pipeline

    run_pipeline(raw_dir=args.raw_dir)


if __name__ == "__main__":
    main()
