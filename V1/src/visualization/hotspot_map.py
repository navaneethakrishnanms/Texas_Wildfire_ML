"""
hotspot_map.py
==============
Visualizes wildfire risk predictions on a Texas base map.

Generates:
  - outputs/plots/hotspot_map.png   (static risk/prediction map)

Usage:
    python src/visualization/hotspot_map.py
    python src/visualization/hotspot_map.py --predictions outputs/predictions/predictions.csv
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from loguru import logger

# ---------------------------------------------------------------------------
PROC_DIR     = Path("data/processed")
PREDS_DIR    = Path("outputs/predictions")
PLOTS_DIR    = Path("outputs/plots")
FIRMS_CSV    = Path("data/raw/firms/Texas_FIRMS_2024.csv")

# Texas approximate bounds (WGS-84)
TX_BOUNDS = dict(west=-106.65, east=-93.51, south=25.84, north=36.50)


# ---------------------------------------------------------------------------
def load_data(predictions_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load fire detections (FIRMS) and model predictions.
    Returns (firms_df, pred_df).  pred_df may be empty if predictions not yet run.
    """
    # FIRMS actual fires
    firms_df = pd.DataFrame()
    if FIRMS_CSV.exists():
        firms_df = pd.read_csv(FIRMS_CSV)
        logger.info(f"Loaded {len(firms_df)} FIRMS fire points.")
    else:
        logger.warning("FIRMS CSV not found — fire points will not be plotted.")

    # Model predictions
    pred_df = pd.DataFrame()
    pred_path = predictions_path or (PREDS_DIR / "predictions.csv")
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        logger.info(f"Loaded {len(pred_df)} prediction rows.")
    else:
        # Fall back to test set fire_label if predictions don't exist yet
        test_path = PROC_DIR / "test.csv"
        if test_path.exists():
            pred_df = pd.read_csv(test_path)
            if "fire_risk_score" not in pred_df.columns and "fire_label" in pred_df.columns:
                pred_df["fire_risk_score"] = pred_df["fire_label"].astype(float)
                pred_df["fire_predicted"] = pred_df["fire_label"]
            logger.info(f"Using test.csv as fallback ({len(pred_df)} rows).")

    return firms_df, pred_df


# ---------------------------------------------------------------------------
def make_hotspot_map(
    firms_df: pd.DataFrame,
    pred_df:  pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Draw the hotspot map with two panels:
      Left  — actual FIRMS fire detections
      Right — model predicted risk scores
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#0e1117")

    # Custom risk colormap: green → yellow → red
    risk_cmap = LinearSegmentedColormap.from_list(
        "wildfire_risk",
        ["#00b09b", "#ffcc00", "#ff4e00", "#8b0000"],
        N=256,
    )

    for ax in axes:
        ax.set_facecolor("#1a1f2e")
        ax.set_xlim(TX_BOUNDS["west"], TX_BOUNDS["east"])
        ax.set_ylim(TX_BOUNDS["south"], TX_BOUNDS["north"])
        ax.set_xlabel("Longitude", color="white", fontsize=10)
        ax.set_ylabel("Latitude",  color="white", fontsize=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333d57")

        # Texas state outline (rough polygon)
        tx_lon = [-106.65, -103.00, -100.00, -94.00, -93.51, -94.48,
                  -96.50, -97.00, -100.00, -104.00, -106.65]
        tx_lat = [31.80, 36.50, 36.50, 33.55, 29.75, 26.20,
                  25.84, 26.00, 28.00, 29.00, 31.80]
        ax.plot(tx_lon, tx_lat, color="#334466", linewidth=1.2, alpha=0.6)

    # ── LEFT PANEL: FIRMS actual fire detections ──────────────────────────
    ax_left = axes[0]
    ax_left.set_title("FIRMS Active Fire Detections (2024)", color="white",
                       fontsize=13, fontweight="bold", pad=12)

    if not firms_df.empty and "longitude" in firms_df.columns:
        sc = ax_left.scatter(
            firms_df["longitude"], firms_df["latitude"],
            c=firms_df.get("frp", np.ones(len(firms_df))) ,
            cmap="hot",
            s=12, alpha=0.7, linewidths=0,
        )
        cbar = plt.colorbar(sc, ax=ax_left, fraction=0.03, pad=0.02)
        cbar.set_label("Fire Radiative Power (MW)", color="white", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    else:
        ax_left.text(
            0.5, 0.5, "No FIRMS data available",
            transform=ax_left.transAxes,
            ha="center", va="center", color="#888", fontsize=12
        )

    # ── RIGHT PANEL: Model predictions ───────────────────────────────────
    ax_right = axes[1]
    ax_right.set_title("Model Predicted Fire Risk Score", color="white",
                        fontsize=13, fontweight="bold", pad=12)

    if not pred_df.empty and "fire_risk_score" in pred_df.columns:
        # Background scatter: all predictions by risk score
        sc2 = ax_right.scatter(
            pred_df["longitude"], pred_df["latitude"],
            c=pred_df["fire_risk_score"],
            cmap=risk_cmap,
            vmin=0, vmax=1,
            s=20, alpha=0.75, linewidths=0,
        )
        cbar2 = plt.colorbar(sc2, ax=ax_right, fraction=0.03, pad=0.02)
        cbar2.set_label("Risk Score (0=Safe, 1=High)", color="white", fontsize=9)
        cbar2.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="white")

        # Overlay: predicted fires highlighted with rings
        if "fire_predicted" in pred_df.columns:
            high_risk = pred_df[pred_df["fire_predicted"] == 1]
            ax_right.scatter(
                high_risk["longitude"], high_risk["latitude"],
                facecolors="none", edgecolors="#ff4e00",
                s=60, linewidths=0.8, alpha=0.9, label=f"Predicted Fire ({len(high_risk)})"
            )
            legend = ax_right.legend(facecolor="#1a1f2e", edgecolor="#666",
                                      labelcolor="white", fontsize=9)
    else:
        ax_right.text(
            0.5, 0.5,
            "No predictions available.\nRun: python src/inference/predict.py --test-set",
            transform=ax_right.transAxes,
            ha="center", va="center", color="#888", fontsize=11
        )

    # ── Figure-level styling ──────────────────────────────────────────────
    fig.suptitle(
        "Texas Wildfire Risk — POC Hotspot Map",
        color="white", fontsize=16, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    logger.info(f"Hotspot map saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate wildfire hotspot map")
    parser.add_argument(
        "--predictions", type=str, default=None,
        help="Path to predictions CSV (default: outputs/predictions/predictions.csv)"
    )
    parser.add_argument(
        "--output", type=str, default=str(PLOTS_DIR / "hotspot_map.png"),
        help="Output image path"
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    pred_path = Path(args.predictions) if args.predictions else None
    firms_df, pred_df = load_data(pred_path)
    make_hotspot_map(firms_df, pred_df, Path(args.output))
    print(f"\nMap saved to: {args.output}")


if __name__ == "__main__":
    main()
