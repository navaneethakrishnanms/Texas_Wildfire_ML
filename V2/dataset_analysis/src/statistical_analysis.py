"""
src/statistical_analysis.py
----------------------------
Analysis 6: Statistical Analysis of all numeric features.

For every numeric column computes:
  Mean, Median, Std, Variance, Min, Max, Q1, Q3, IQR,
  Skewness, Kurtosis, # Outliers (IQR method), # Infinites

Identifies:
  - Highly skewed columns (|skew| > SKEWNESS_HIGH)
  - Heavy-tailed columns (|kurtosis| > KURTOSIS_HIGH)
  - Outlier-rich columns

Saves
-----
tables/statistical_summary.csv
tables/skewed_columns.csv
tables/heavy_tail_columns.csv
tables/outlier_summary.csv
plots/statistics/skewness_distribution.png
plots/statistics/top_outlier_columns.png
plots/statistics/distribution_<column>.png  (top N most interesting)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config.config import (
    FIGURE_DPI,
    FIGURE_SIZE_WIDE,
    IQR_OUTLIER_FACTOR,
    KURTOSIS_HIGH,
    LOG_FILE,
    MATPLOTLIB_STYLE,
    PLOTS_STATS_DIR,
    SKEWNESS_HIGH,
    TABLES_DIR,
)
from src.utils import ensure_dirs, save_csv, save_figure, setup_logger

logger = setup_logger(__name__, LOG_FILE)

# Number of individual distribution plots to generate
_TOP_N_DIST_PLOTS = 30


def _compute_stats_row(col: str, s: pd.Series) -> dict:
    """Compute all statistics for a single numeric series."""
    s_clean = s.dropna()
    s_finite = s_clean[np.isfinite(s_clean)]

    if s_finite.empty:
        return {
            "Column": col, "Count Non-Null": 0, "Count Finite": 0,
            "Mean": np.nan, "Median": np.nan, "Std": np.nan, "Variance": np.nan,
            "Min": np.nan, "Max": np.nan, "Q1": np.nan, "Q3": np.nan, "IQR": np.nan,
            "Skewness": np.nan, "Kurtosis": np.nan,
            "Outlier Count (IQR)": 0, "Outlier %": 0.0,
            "Infinite Count": 0,
            "Highly Skewed": False, "Heavy Tailed": False,
        }

    q1  = float(s_finite.quantile(0.25))
    q3  = float(s_finite.quantile(0.75))
    iqr = q3 - q1
    fence_lo = q1 - IQR_OUTLIER_FACTOR * iqr
    fence_hi = q3 + IQR_OUTLIER_FACTOR * iqr
    outliers = ((s_finite < fence_lo) | (s_finite > fence_hi)).sum()
    outlier_pct = round(outliers / len(s_finite) * 100, 4) if len(s_finite) > 0 else 0.0

    skew  = float(s_finite.skew())
    kurt  = float(s_finite.kurtosis())  # excess kurtosis (Fisher)
    inf_c = int(np.isinf(s_clean).sum())

    return {
        "Column":               col,
        "Count Non-Null":       int(s_clean.shape[0]),
        "Count Finite":         int(s_finite.shape[0]),
        "Mean":                 round(float(s_finite.mean()), 6),
        "Median":               round(float(s_finite.median()), 6),
        "Std":                  round(float(s_finite.std()), 6),
        "Variance":             round(float(s_finite.var()), 6),
        "Min":                  round(float(s_finite.min()), 6),
        "P5":                   round(float(s_finite.quantile(0.05)), 6),
        "Q1 (25%)":             round(q1, 6),
        "Q3 (75%)":             round(q3, 6),
        "P95":                  round(float(s_finite.quantile(0.95)), 6),
        "Max":                  round(float(s_finite.max()), 6),
        "IQR":                  round(iqr, 6),
        "Skewness":             round(skew, 6),
        "Kurtosis (excess)":    round(kurt, 6),
        "Outlier Count (IQR)":  int(outliers),
        "Outlier %":            outlier_pct,
        "Infinite Count":       inf_c,
        "Highly Skewed":        abs(skew) > SKEWNESS_HIGH,
        "Heavy Tailed":         abs(kurt) > KURTOSIS_HIGH,
    }


def _plot_skewness_distribution(stats_df: pd.DataFrame) -> None:
    """Histogram of skewness values across all numeric columns."""
    ensure_dirs(PLOTS_STATS_DIR)
    valid = stats_df["Skewness"].dropna()
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(valid.clip(-10, 10), bins=60, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(SKEWNESS_HIGH,  color="red",    linestyle="--", linewidth=1.5, label=f"+{SKEWNESS_HIGH} threshold")
    ax.axvline(-SKEWNESS_HIGH, color="red",    linestyle="--", linewidth=1.5, label=f"-{SKEWNESS_HIGH} threshold")
    ax.axvline(0,              color="black",  linestyle="-",  linewidth=1.0, label="Zero skewness")
    ax.set_xlabel("Skewness (clipped to ±10)", fontsize=11)
    ax.set_ylabel("Number of Columns", fontsize=11)
    ax.set_title("Skewness Distribution Across All Numeric Features", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    out = PLOTS_STATS_DIR / "skewness_distribution.png"
    save_figure(fig, out, dpi=FIGURE_DPI)
    logger.info(f"  ✔ Skewness distribution plot saved: {out}")


def _plot_top_outlier_columns(stats_df: pd.DataFrame) -> None:
    """Bar chart of top 30 columns with highest outlier %."""
    ensure_dirs(PLOTS_STATS_DIR)
    top = stats_df.sort_values("Outlier %", ascending=False).head(30)
    top = top[top["Outlier %"] > 0]
    if top.empty:
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(top) * 0.35)))
    ax.barh(top["Column"], top["Outlier %"], color="tomato", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Outlier % (IQR method)", fontsize=11)
    ax.set_title(f"Top Columns by Outlier Percentage (IQR × {IQR_OUTLIER_FACTOR})", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=7)
    out = PLOTS_STATS_DIR / "top_outlier_columns.png"
    save_figure(fig, out, dpi=FIGURE_DPI)
    logger.info(f"  ✔ Outlier column chart saved: {out}")


def _plot_individual_distributions(df: pd.DataFrame, cols_to_plot: list[str]) -> None:
    """Generate histogram + KDE for each column in *cols_to_plot*."""
    ensure_dirs(PLOTS_STATS_DIR / "distributions")

    for col in cols_to_plot:
        try:
            s = df[col].dropna()
            s = s[np.isfinite(s)]
            if s.empty or s.nunique() < 2:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Distribution: {col}", fontsize=12, fontweight="bold")

            # Histogram
            ax = axes[0]
            ax.hist(s, bins=60, color="steelblue", edgecolor="white", linewidth=0.4, density=True)
            ax.set_xlabel(col, fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.set_title("Histogram", fontsize=10)

            # KDE
            ax2 = axes[1]
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(s.values)
                x_grid = np.linspace(float(s.min()), float(s.max()), 300)
                ax2.plot(x_grid, kde(x_grid), color="darkorange", linewidth=2)
                ax2.fill_between(x_grid, kde(x_grid), alpha=0.3, color="darkorange")
            except Exception:
                ax2.hist(s, bins=60, color="darkorange", edgecolor="white", linewidth=0.4, density=True)
            ax2.set_xlabel(col, fontsize=9)
            ax2.set_ylabel("Density", fontsize=9)
            ax2.set_title("KDE", fontsize=10)

            safe_name = col.replace("/", "_").replace("\\", "_").replace(" ", "_")[:60]
            out = PLOTS_STATS_DIR / "distributions" / f"dist_{safe_name}.png"
            save_figure(fig, out, dpi=FIGURE_DPI)
        except Exception as exc:
            logger.warning(f"  Could not plot distribution for {col}: {exc}")


def generate_statistical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analysis 6: Full statistical analysis of all numeric features.

    Parameters
    ----------
    df : The merged dataset (never modified).

    Returns
    -------
    pd.DataFrame  Statistical summary table.
    """
    logger.info("Analysis 6 — Statistical Analysis")
    ensure_dirs(TABLES_DIR, PLOTS_STATS_DIR)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    logger.info(f"  Processing {len(numeric_cols)} numeric columns...")

    rows = []
    for i, col in enumerate(numeric_cols):
        rows.append(_compute_stats_row(col, df[col]))
        if (i + 1) % 50 == 0:
            logger.debug(f"  ... {i + 1}/{len(numeric_cols)}")

    stats_df = pd.DataFrame(rows)
    save_csv(stats_df, TABLES_DIR / "statistical_summary.csv")

    # Sub-tables
    skewed_df    = stats_df[stats_df["Highly Skewed"]].sort_values("Skewness", key=abs, ascending=False)
    heavy_df     = stats_df[stats_df["Heavy Tailed"]].sort_values("Kurtosis (excess)", key=abs, ascending=False)
    outlier_df   = stats_df[stats_df["Outlier %"] > 0].sort_values("Outlier %", ascending=False)

    save_csv(skewed_df,  TABLES_DIR / "skewed_columns.csv")
    save_csv(heavy_df,   TABLES_DIR / "heavy_tail_columns.csv")
    save_csv(outlier_df, TABLES_DIR / "outlier_summary.csv")

    # Console summary
    print("\n" + "=" * 60)
    print("  STATISTICAL ANALYSIS")
    print("=" * 60)
    print(f"  Numeric columns analyzed  : {len(numeric_cols)}")
    print(f"  Highly skewed (|skew|>{SKEWNESS_HIGH}) : {stats_df['Highly Skewed'].sum()}")
    print(f"  Heavy-tailed (|kurt|>{KURTOSIS_HIGH})  : {stats_df['Heavy Tailed'].sum()}")
    print(f"  Columns with outliers     : {(stats_df['Outlier %'] > 0).sum()}")
    print("=" * 60 + "\n")

    # Plots
    _plot_skewness_distribution(stats_df)
    _plot_top_outlier_columns(stats_df)

    # Plot top N most-skewed columns' distributions
    top_cols_to_plot = (
        skewed_df["Column"].head(_TOP_N_DIST_PLOTS).tolist()
    )
    # Also add a few key fire columns
    priority_cols = ["FIRE_SIZE", "LATITUDE", "LONGITUDE", "ERC", "NDVI", "KBDI", "TMAX", "PRCP"]
    for p in priority_cols:
        if p in numeric_cols and p not in top_cols_to_plot:
            top_cols_to_plot.insert(0, p)

    logger.info(f"  Generating individual distribution plots for {len(top_cols_to_plot)} columns...")
    _plot_individual_distributions(df, top_cols_to_plot[:_TOP_N_DIST_PLOTS])

    logger.info(f"  ✔ Statistical analysis complete.")
    return stats_df
