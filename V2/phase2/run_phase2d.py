"""
run_phase2d.py
--------------
Phase 2D — DAY_MATCHED Negative Sampling

HOW IT WORKS (THE CORE QUESTION ANSWERED):
=========================================

The 4 UTC windows explained:
  Each day has 4 time slots: 00Z, 06Z, 12Z, 18Z
  A FIRE event is assigned to ONE of these slots based on its discovery time.
  NON-FIRE cells inherit that SAME date + SAME slot.

Example:
  FIRE: Texas, 2018-11-08, 14:30 CST → 20:30 UTC → 18Z window
  → This fire gets label=1 in slot (date=2018-11-08, window=18Z)

  10 NON-FIRE CELLS:
  → Also assigned to (date=2018-11-08, window=18Z)
  → But their coordinates are 10 DIFFERENT Texas H3 cells that had no fire
  → Their gridMET weather is extracted at THEIR OWN locations on 2018-11-08
  → Their HRRR weather is extracted at THEIR OWN locations for 18Z 2018-11-08
  → label=0 for all of them

WHY THIS WORKS:
  Both fire and non-fire cells experience the SAME ambient weather on that day.
  The model can't just learn "18Z is fire time" — both fire and non-fire are AT 18Z.
  It must learn: "which CELL LOCATIONS are most dangerous given today's weather?"
  This is exactly what a dispatch system needs.

SCALE (answering the "too big?" doubt):
  Texas fires 2014–2020: ~36,182 events
  Non-fire cells (10×): ~361,820 rows
  TOTAL TRAINING TABLE: ~397,000 rows — very manageable
  (The 2.2M cell grid is just the sampling POOL, not the training set)

Usage:
    conda activate torch_gpu
    python run_phase2d.py --state TX --ratio 10
    python run_phase2d.py --state CA --ratio 10
    python run_phase2d.py --state ALL

Outputs:
    phase2/outputs/<state>/negatives_labels.parquet
    phase2/outputs/<state>/full_training_labels.parquet   ← fire + non-fire combined
    phase2/outputs/<state>/phase2d_summary.csv
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


def _get_h3():
    try:
        import h3
        return h3
    except ImportError:
        raise ImportError("pip install h3")


def _cell_to_latlng(h3_lib, cell):
    try:
        try:
            return h3_lib.h3_to_geo(cell)
        except AttributeError:
            return h3_lib.cell_to_latlng(cell)
    except Exception:
        return (0.0, 0.0)


def build_day_matched_negatives(
    positives_df: pd.DataFrame,
    all_state_cells: list,
    ratio: int = 10,
    seed: int = 42,
    state_key: str = "TX",
) -> pd.DataFrame:
    """
    For each positive fire event (h3_cell, date_utc, window_hour):
      - Find all state H3 cells that did NOT have a fire on that (date, window)
      - Sample `ratio` of them at random → label=0

    Parameters
    ----------
    positives_df    : Output from Phase 2C (label=1 rows)
    all_state_cells : List of ALL H3 cell IDs in the state (from h3_grid parquet)
    ratio           : Number of non-fire cells per fire cell (default 10)
    seed            : Random seed for reproducibility

    Returns
    -------
    DataFrame with same columns as positives_df, label=0
    """
    h3_lib = _get_h3()
    rng = np.random.default_rng(seed)
    all_cells_arr = np.array(all_state_cells)
    all_cells_set = set(all_state_cells)

    neg_rows = []
    groups = list(positives_df.groupby(["date_utc", "window_hour"]))

    logger.info(f"  Processing {len(groups):,} unique (date, window) slots...")
    logger.info(f"  Total positive events: {len(positives_df):,}")
    logger.info(f"  Target negative ratio: 1:{ratio}")
    logger.info(f"  Target negatives: ~{len(positives_df) * ratio:,}")

    n_slots_processed = 0
    n_slots_skipped   = 0

    for (date_d, window_h), group in groups:
        # Which cells had fires in this (date, window)?
        fire_cells_today = set(group["h3_cell"].values)

        # Eligible = all state cells that had NO fire today in this window
        eligible = [c for c in all_cells_arr if c not in fire_cells_today]

        n_needed = len(group) * ratio
        if len(eligible) < ratio:
            n_slots_skipped += 1
            continue  # extremely rare — almost all cells are fire-free

        # Sample without replacement (cap at available)
        n_sample = min(n_needed, len(eligible))
        sampled_cells = rng.choice(eligible, size=n_sample, replace=False)

        # Build the window timestamp
        window_ts = pd.Timestamp(date_d) + pd.Timedelta(hours=window_h)

        for cell in sampled_cells:
            clat, clon = _cell_to_latlng(h3_lib, cell)
            neg_rows.append({
                "h3_cell":       cell,
                "date_utc":      str(date_d),
                "window_hour":   window_h,
                "window_6h_utc": window_ts,
                "label":         0,
                "centroid_lat":  round(clat, 6),
                "centroid_lon":  round(clon, 6),
                "fire_year":     pd.Timestamp(date_d).year,
                "state":         state_key,
            })

        n_slots_processed += 1
        if n_slots_processed % 500 == 0:
            logger.info(f"    Processed {n_slots_processed:,}/{len(groups):,} slots "
                        f"({100*n_slots_processed/len(groups):.0f}%)")

    logger.info(f"\n  Slots processed: {n_slots_processed:,}")
    logger.info(f"  Slots skipped (too few eligible cells): {n_slots_skipped:,}")
    logger.info(f"  Total negative rows generated: {len(neg_rows):,}")

    return pd.DataFrame(neg_rows)


def run_state(state_key: str, ratio: int) -> bool:
    cfg        = STATE_CONFIG[state_key]
    output_dir = cfg["output_dir"]

    logger.info("=" * 60)
    logger.info(f"PHASE 2D — {cfg['name'].upper()} — Negative Sampling (1:{ratio})")
    logger.info("=" * 60)

    # ── Load positives (Phase 2C output) ──────────────────────────────────────
    pos_path = output_dir / "positives_labels.parquet"
    if not pos_path.exists():
        logger.error(f"Positives parquet not found: {pos_path}")
        logger.error("Run Phase 2C first: python run_phase2c.py --state " + state_key)
        return False

    pos_df = pd.read_parquet(pos_path)
    logger.info(f"  Loaded {len(pos_df):,} positive fire events")

    # ── Load H3 grid (Phase 2B output) ────────────────────────────────────────
    grid_path = output_dir / f"h3_grid_{state_key.lower()}.parquet"
    if not grid_path.exists():
        logger.error(f"H3 grid not found: {grid_path}")
        logger.error("Run Phase 2B first: python run_phase2b.py --state " + state_key)
        return False

    grid_df = pd.read_parquet(grid_path)
    all_cells = grid_df["h3_cell"].tolist()
    logger.info(f"  Loaded H3 grid: {len(all_cells):,} cells (R{cfg['h3_level']})")

    # ── Generate negatives ────────────────────────────────────────────────────
    try:
        neg_df = build_day_matched_negatives(
            positives_df    = pos_df,
            all_state_cells = all_cells,
            ratio           = ratio,
            seed            = 42,
            state_key       = state_key,
        )
    except Exception as e:
        logger.error(f"Negative sampling failed: {e}", exc_info=True)
        return False

    if len(neg_df) == 0:
        logger.error("No negative rows generated!")
        return False

    # ── Save negatives ────────────────────────────────────────────────────────
    neg_path = output_dir / "negatives_labels.parquet"
    neg_df.to_parquet(neg_path, index=False, compression="snappy")
    logger.info(f"\n  Saved negatives: {neg_path}  ({neg_path.stat().st_size/1e6:.1f} MB)")

    # ── Combine fire + non-fire → full training labels table ──────────────────
    combined = pd.concat([pos_df, neg_df], ignore_index=True)
    combined = combined.sort_values(["fire_year", "date_utc", "window_hour"]).reset_index(drop=True)

    full_path = output_dir / "full_training_labels.parquet"
    combined.to_parquet(full_path, index=False, compression="snappy")
    logger.info(f"  Saved combined: {full_path}  ({full_path.stat().st_size/1e6:.1f} MB)")

    # ── Stats ──────────────────────────────────────────────────────────────────
    n_pos    = (combined["label"] == 1).sum()
    n_neg    = (combined["label"] == 0).sum()
    pos_rate = 100 * n_pos / len(combined)

    logger.info(f"\n  === SAMPLING SUMMARY ===")
    logger.info(f"  Positive (fire)     : {n_pos:>10,} ({pos_rate:.1f}%)")
    logger.info(f"  Negative (non-fire) : {n_neg:>10,} ({100-pos_rate:.1f}%)")
    logger.info(f"  TOTAL rows          : {len(combined):>10,}")
    logger.info(f"  Actual ratio        : 1:{n_neg//n_pos:.0f}")

    # Window distribution in combined set
    wd = combined.groupby(["window_hour", "label"]).size().unstack(fill_value=0)
    logger.info(f"\n  Rows by UTC window:")
    logger.info(f"    {'Window':<10} {'Fire=1':>8} {'Fire=0':>10} {'Total':>8}")
    for wh in sorted(combined["window_hour"].unique()):
        sub = combined[combined["window_hour"] == wh]
        n1  = (sub["label"] == 1).sum()
        n0  = (sub["label"] == 0).sum()
        logger.info(f"    {wh:02d}Z       {n1:>8,} {n0:>10,} {n1+n0:>8,}")

    # Year distribution
    yd = combined.groupby(["fire_year", "label"]).size().unstack(fill_value=0)
    logger.info(f"\n  Rows by year (chronological split reference):")
    logger.info(f"    {'Year':<8} {'Fire=1':>8} {'Fire=0':>10} {'Split':>10}")
    for yr in sorted(combined["fire_year"].unique()):
        sub   = combined[combined["fire_year"] == yr]
        n1    = (sub["label"] == 1).sum()
        n0    = (sub["label"] == 0).sum()
        split = "TRAIN" if yr <= 2017 else ("VAL" if yr == 2018 else "TEST")
        logger.info(f"    {yr:<8} {n1:>8,} {n0:>10,} {split:>10}")

    # Summary CSV
    pd.DataFrame({
        "state":      [state_key],
        "n_positive": [int(n_pos)],
        "n_negative": [int(n_neg)],
        "n_total":    [len(combined)],
        "pos_rate_%": [round(pos_rate, 2)],
        "ratio":      [ratio],
        "h3_resolution": [cfg["h3_level"]],
    }).to_csv(output_dir / "phase2d_summary.csv", index=False)

    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 2D — DAY_MATCHED Negative Sampling")
    parser.add_argument("--state", choices=["TX", "CA", "ALL"], required=True)
    parser.add_argument("--ratio", type=int, default=10,
                        help="Negative:positive ratio (default=10)")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "phase2d.log", encoding="utf-8"),
        ],
    )

    states  = ["TX", "CA"] if args.state == "ALL" else [args.state]
    results = {s: run_state(s, args.ratio) for s in states}

    print("\n" + "═" * 60)
    for s, ok in results.items():
        print(f"  {STATE_CONFIG[s]['name']:<15} {'✔ SUCCESS' if ok else '✘ FAILED'}")
    print("═" * 60)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
