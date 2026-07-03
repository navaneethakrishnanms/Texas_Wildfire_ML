"""
src/missing_analysis.py
------------------------
Analysis 4: Missing Value Analysis.

Computes per-column missing counts/percentages, identifies
critical/high/moderate/low-missing columns, and generates:
  - Missing value heatmap      → plots/missing/missing_heatmap.png
  - Missing value matrix       → plots/missing/missing_matrix.png  (missingno)
  - Missing bar chart          → plots/missing/missing_bar.png
  - tables/missing_summary.csv

IMPORTANT: No values are filled. No columns are removed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config.config import (
    LOG_FILE,
    MISSING_CRITICAL,
    MISSING_HIGH,
    MISSING_LOW,
    MISSING_MODERATE,
    PLOTS_MISSING_DIR,
    TABLES_DIR,
    FIGURE_DPI,
    FIGURE_SIZE_WIDE,
    FIGURE_SIZE_TALL,
    MATPLOTLIB_STYLE,
)
from src.utils import (
    ensure_dirs,
    save_csv,
    save_figure,
    setup_logger,
)

logger = setup_logger(__name__, LOG_FILE)

try:
    import missingno as msno
    HAS_MISSINGNO = True
except ImportError:
    HAS_MISSINGNO = False
    logger.warning("missingno not installed — skipping matrix plot.")


# ─────────────────────────────────────────────────────────────────────────────
# Core Computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with one row per column showing missing statistics."""
    n_rows = len(df)
    missing_count = df.isna().sum()
    missing_pct   = (missing_count / n_rows * 100).round(4)

    def _tier(pct: float) -> str:
        if pct >= MISSING_CRITICAL:
            return f"Critical (≥{MISSING_CRITICAL}%)"
        if pct >= MISSING_HIGH:
            return f"High (≥{MISSING_HIGH}%)"
        if pct >= MISSING_MODERATE:
            return f"Moderate (≥{MISSING_MODERATE}%)"
        if pct >= MISSING_LOW:
            return f"Low (≥{MISSING_LOW}%)"
        if pct > 0:
            return "Minimal (<25%)"
        return "Complete (0%)"

    summary = pd.DataFrame({
        "Column":          df.columns,
        "Missing Count":   missing_count.values,
        "Missing %":       missing_pct.values,
        "Present Count":   (n_rows - missing_count).values,
        "Present %":       (100 - missing_pct).round(4).values,
        "Tier":            [_tier(p) for p in missing_pct.values],
    })
    return summary.sort_values("Missing %", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_missing_bar(summary: pd.DataFrame) -> None:
    """Bar chart showing top-N columns with highest missing %."""
    ensure_dirs(PLOTS_MISSING_DIR)

    # Show top 60 columns with any missing values
    top = summary[summary["Missing %"] > 0].head(60).copy()
    if top.empty:
        logger.info("  No missing values found — skipping bar chart.")
        return

    n = len(top)
    fig, ax = plt.subplots(figsize=(18, max(8, n * 0.28)))

    colors = []
    for pct in top["Missing %"]:
        if pct >= MISSING_CRITICAL:
            colors.append("#D32F2F")  # red
        elif pct >= MISSING_HIGH:
            colors.append("#F57C00")  # orange
        elif pct >= MISSING_MODERATE:
            colors.append("#FBC02D")  # yellow
        else:
            colors.append("#388E3C")  # green

    bars = ax.barh(top["Column"], top["Missing %"], color=colors, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, pct in zip(bars, top["Missing %"]):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", ha="left", fontsize=7, color="#333"
        )

    ax.set_xlabel("Missing %", fontsize=11)
    ax.set_title("Missing Value Percentage — Top Columns", fontsize=14, fontweight="bold", pad=16)
    ax.set_xlim(0, 110)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#D32F2F", label=f"Critical ≥{MISSING_CRITICAL}%"),
        Patch(facecolor="#F57C00", label=f"High ≥{MISSING_HIGH}%"),
        Patch(facecolor="#FBC02D", label=f"Moderate ≥{MISSING_MODERATE}%"),
        Patch(facecolor="#388E3C", label=f"Low <{MISSING_MODERATE}%"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.style.use(MATPLOTLIB_STYLE)
    out = PLOTS_MISSING_DIR / "missing_bar.png"
    save_figure(fig, out, dpi=FIGURE_DPI)
    logger.info(f"  ✔ Missing bar chart saved: {out}")


def _plot_missing_heatmap(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Heatmap of missingness pattern across a subset of columns."""
    ensure_dirs(PLOTS_MISSING_DIR)

    # Select top 50 most-missing columns for the heatmap
    top_cols = summary[summary["Missing %"] > 0].head(50)["Column"].tolist()
    if not top_cols:
        logger.info("  No missing columns for heatmap.")
        return

    # Sample rows for performance
    sub = df[top_cols].head(5000)

    # Build binary matrix (1=missing, 0=present)
    mat = sub.isna().astype(int)

    fig, ax = plt.subplots(figsize=(min(len(top_cols) * 0.35 + 2, 24), 8))
    im = ax.imshow(mat.T, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest")

    ax.set_yticks(range(len(top_cols)))
    ax.set_yticklabels(top_cols, fontsize=6)
    ax.set_xlabel("Sample Row Index (first 5,000)", fontsize=9)
    ax.set_title("Missing Value Pattern Heatmap (Top-50 Missing Columns)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="1=Missing / 0=Present")

    out = PLOTS_MISSING_DIR / "missing_heatmap.png"
    save_figure(fig, out, dpi=FIGURE_DPI)
    logger.info(f"  ✔ Missing heatmap saved: {out}")


def _plot_missingno_matrix(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Use missingno to create a sparse-matrix visualization."""
    if not HAS_MISSINGNO:
        return
    ensure_dirs(PLOTS_MISSING_DIR)

    top_cols = summary[summary["Missing %"] > 0].head(50)["Column"].tolist()
    if not top_cols:
        return

    sub = df[top_cols].head(5000)
    fig, ax = plt.subplots(figsize=(16, 8))
    msno.matrix(sub, ax=ax, sparkline=False, fontsize=7, color=(0.25, 0.45, 0.85))
    ax.set_title("Missing Value Matrix (missingno) — Top-50 Columns", fontsize=12, fontweight="bold")
    out = PLOTS_MISSING_DIR / "missing_matrix.png"
    save_figure(fig, out, dpi=FIGURE_DPI)
    logger.info(f"  ✔ Missingno matrix saved: {out}")


def _plot_missingno_bar(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Use missingno bar chart showing completeness per column (top 50)."""
    if not HAS_MISSINGNO:
        return
    ensure_dirs(PLOTS_MISSING_DIR)

    top_cols = summary[summary["Missing %"] > 0].head(50)["Column"].tolist()
    if not top_cols:
        return

    sub = df[top_cols].head(5000)
    fig, ax = plt.subplots(figsize=(16, 7))
    msno.bar(sub, ax=ax, fontsize=7, color="steelblue")
    ax.set_title("Data Completeness (missingno bar) — Top-50 Missing Columns", fontsize=12, fontweight="bold")
    out = PLOTS_MISSING_DIR / "missing_bar_msno.png"
    save_figure(fig, out, dpi=FIGURE_DPI)
    logger.info(f"  ✔ Missingno bar chart saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 4 — Public Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def generate_missing_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analysis 4: Missing Value Analysis.

    Steps
    -----
    1. Compute per-column missing statistics.
    2. Generate tiered missing summary.
    3. Save tables/missing_summary.csv.
    4. Generate all three plots (bar, heatmap, missingno matrix).
    5. Print a human-readable console report.

    Parameters
    ----------
    df : The merged dataset (never modified).

    Returns
    -------
    pd.DataFrame  Missing summary table.
    """
    logger.info("Analysis 4 — Missing Value Analysis")
    ensure_dirs(TABLES_DIR, PLOTS_MISSING_DIR)

    summary = _compute_missing_summary(df)
    save_csv(summary, TABLES_DIR / "missing_summary.csv")
    logger.info(f"  ✔ Missing summary saved: {TABLES_DIR / 'missing_summary.csv'}")

    # Tier breakdown
    tiers = summary.groupby("Tier")["Column"].count().reset_index()
    tiers.columns = ["Tier", "Column Count"]
    save_csv(tiers, TABLES_DIR / "missing_tier_breakdown.csv")

    # Console report
    total_cols = len(df.columns)
    cols_with_missing = (summary["Missing %"] > 0).sum()
    cols_complete     = total_cols - cols_with_missing

    print("\n" + "=" * 60)
    print("  MISSING VALUE ANALYSIS")
    print("=" * 60)
    print(f"  Total Columns        : {total_cols}")
    print(f"  Fully Complete       : {cols_complete}")
    print(f"  Has Any Missing      : {cols_with_missing}")

    for threshold, label in [
        (MISSING_CRITICAL, f">{MISSING_CRITICAL}% missing"),
        (MISSING_HIGH,     f">{MISSING_HIGH}% missing"),
        (MISSING_MODERATE, f">{MISSING_MODERATE}% missing"),
        (MISSING_LOW,      f">{MISSING_LOW}% missing"),
    ]:
        cnt = (summary["Missing %"] >= threshold).sum()
        print(f"  Columns {label:<18}: {cnt}")

    print("\n  Top 20 Columns by Missing %:")
    for _, row in summary.head(20).iterrows():
        print(f"    {row['Column']:<50} {row['Missing %']:>7.2f}%")
    print("=" * 60 + "\n")

    # Generate plots
    _plot_missing_bar(summary)
    _plot_missing_heatmap(df, summary)
    _plot_missingno_matrix(df, summary)
    _plot_missingno_bar(df, summary)

    return summary
