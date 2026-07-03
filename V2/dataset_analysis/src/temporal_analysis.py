"""
src/temporal_analysis.py
------------------------
Analysis 7: Temporal Analysis.

Analyzes the time dimensions of the wildfire dataset:
  - Fire counts by year
  - Fire counts by month
  - Fire counts by season
  - Fire counts by day-of-week
  - Discovery date distribution
  - Containment date distribution
  - Duration (days between discovery and containment) distribution

Saves
-----
tables/temporal_year.csv
tables/temporal_month.csv
tables/temporal_season.csv
tables/temporal_dow.csv
tables/temporal_duration.csv
plots/temporal/fires_by_year.png
plots/temporal/fires_by_month.png
plots/temporal/fires_by_season.png
plots/temporal/fires_by_dow.png
plots/temporal/duration_distribution.png
plots/temporal/fire_size_by_year.png
plots/temporal/discovery_heatmap.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.config import (
    COL_CONT_DATE,
    COL_DISCOVERY_DATE,
    COL_FIRE_SIZE,
    COL_FIRE_YEAR,
    FIGURE_DPI,
    FIGURE_SIZE_WIDE,
    LOG_FILE,
    MATPLOTLIB_STYLE,
    PLOTS_TEMPORAL_DIR,
    TABLES_DIR,
)
from src.utils import ensure_dirs, save_csv, save_figure, setup_logger

logger = setup_logger(__name__, LOG_FILE)

_MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

_SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3:  "Spring", 4: "Spring", 5: "Spring",
    6:  "Summer", 7: "Summer", 8: "Summer",
    9:  "Fall",   10: "Fall",  11: "Fall",
}


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to parse DISCOVERY_DATE and CONT_DATE columns to datetime.
    Returns a COPY of df with parsed datetime columns added.
    """
    work = df.copy()
    for col in [COL_DISCOVERY_DATE, COL_CONT_DATE]:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], errors="coerce")
    return work


def generate_temporal_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Analysis 7: Temporal Analysis.

    Parameters
    ----------
    df : The merged dataset (never modified).

    Returns
    -------
    dict[str, pd.DataFrame]  All temporal summary tables keyed by name.
    """
    logger.info("Analysis 7 — Temporal Analysis")
    ensure_dirs(TABLES_DIR, PLOTS_TEMPORAL_DIR)

    work = _parse_dates(df)
    results: dict[str, pd.DataFrame] = {}

    # ── Year ──────────────────────────────────────────────────────────────────
    if COL_FIRE_YEAR in work.columns:
        year_cnt = (
            work[COL_FIRE_YEAR]
            .value_counts()
            .sort_index()
            .reset_index(name="Fire Count")
        )
        year_cnt.columns = ["Year", "Fire Count"]
        year_cnt["% of Total"] = (year_cnt["Fire Count"] / len(work) * 100).round(2)

        if COL_FIRE_SIZE in work.columns:
            year_size = work.groupby(COL_FIRE_YEAR)[COL_FIRE_SIZE].agg(
                ["sum", "mean", "median"]
            ).reset_index()
            year_size.columns = ["Year", "Total Acres", "Mean Acres", "Median Acres"]
            year_cnt = year_cnt.merge(year_size, on="Year", how="left")

        save_csv(year_cnt, TABLES_DIR / "temporal_year.csv")
        results["year"] = year_cnt

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].bar(year_cnt["Year"], year_cnt["Fire Count"], color="steelblue", edgecolor="white")
        axes[0].set_xlabel("Year", fontsize=11)
        axes[0].set_ylabel("Number of Fires", fontsize=11)
        axes[0].set_title("Fire Count by Year", fontsize=13, fontweight="bold")
        axes[0].tick_params(axis="x", rotation=45)

        if "Total Acres" in year_cnt.columns:
            axes[1].bar(year_cnt["Year"], year_cnt["Total Acres"] / 1e3, color="tomato", edgecolor="white")
            axes[1].set_xlabel("Year", fontsize=11)
            axes[1].set_ylabel("Total Area Burned (1,000 acres)", fontsize=11)
            axes[1].set_title("Total Burned Area by Year", fontsize=13, fontweight="bold")
            axes[1].tick_params(axis="x", rotation=45)

        fig.suptitle("Temporal Analysis — By Year", fontsize=14, fontweight="bold")
        save_figure(fig, PLOTS_TEMPORAL_DIR / "fires_by_year.png", dpi=FIGURE_DPI)
        logger.info(f"  ✔ Year plot saved")

    # ── Month ─────────────────────────────────────────────────────────────────
    disc_col_parsed = None
    if COL_DISCOVERY_DATE in work.columns:
        disc = work[COL_DISCOVERY_DATE]
        if pd.api.types.is_datetime64_any_dtype(disc):
            disc_col_parsed = disc

    if disc_col_parsed is not None:
        month_cnt = disc_col_parsed.dt.month.value_counts().sort_index().reset_index()
        month_cnt.columns = ["Month", "Fire Count"]
        month_cnt["Month Name"] = month_cnt["Month"].map(lambda m: _MONTH_NAMES[m] if 1 <= m <= 12 else "?")
        month_cnt["% of Total"] = (month_cnt["Fire Count"] / len(work) * 100).round(2)
        save_csv(month_cnt, TABLES_DIR / "temporal_month.csv")
        results["month"] = month_cnt

        # Season
        month_cnt["Season"] = month_cnt["Month"].map(_SEASON_MAP)
        season_cnt = month_cnt.groupby("Season")["Fire Count"].sum().reset_index()
        save_csv(season_cnt, TABLES_DIR / "temporal_season.csv")
        results["season"] = season_cnt

        # Day of week
        dow_cnt = disc_col_parsed.dt.day_name().value_counts().reset_index()
        dow_cnt.columns = ["Day of Week", "Fire Count"]
        _dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_cnt["DOW_Order"] = dow_cnt["Day of Week"].map({d: i for i, d in enumerate(_dow_order)})
        dow_cnt = dow_cnt.sort_values("DOW_Order").drop(columns="DOW_Order")
        save_csv(dow_cnt, TABLES_DIR / "temporal_dow.csv")
        results["dow"] = dow_cnt

        # Plots
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Month
        ax = axes[0]
        ax.bar(month_cnt["Month Name"], month_cnt["Fire Count"], color="darkorange", edgecolor="white")
        ax.set_xlabel("Month", fontsize=10)
        ax.set_ylabel("Fire Count", fontsize=10)
        ax.set_title("Fires by Month", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)

        # Season
        _season_colors = {"Winter": "#6495ED", "Spring": "#90EE90", "Summer": "#FF6347", "Fall": "#D2691E"}
        ax = axes[1]
        ax.bar(
            season_cnt["Season"],
            season_cnt["Fire Count"],
            color=[_season_colors.get(s, "grey") for s in season_cnt["Season"]],
            edgecolor="white",
        )
        ax.set_xlabel("Season", fontsize=10)
        ax.set_ylabel("Fire Count", fontsize=10)
        ax.set_title("Fires by Season", fontsize=12, fontweight="bold")

        # DOW
        ax = axes[2]
        ax.bar(dow_cnt["Day of Week"], dow_cnt["Fire Count"], color="mediumpurple", edgecolor="white")
        ax.set_xlabel("Day of Week", fontsize=10)
        ax.set_ylabel("Fire Count", fontsize=10)
        ax.set_title("Fires by Day of Week", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)

        fig.suptitle("Temporal Analysis — Month / Season / Day of Week", fontsize=14, fontweight="bold")
        save_figure(fig, PLOTS_TEMPORAL_DIR / "fires_by_month.png", dpi=FIGURE_DPI)
        logger.info(f"  ✔ Month/Season/DOW plot saved")

        # Season standalone
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.bar(
            season_cnt["Season"],
            season_cnt["Fire Count"],
            color=[_season_colors.get(s, "grey") for s in season_cnt["Season"]],
            edgecolor="white",
        )
        ax2.set_title("Fire Count by Season", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Fire Count")
        save_figure(fig2, PLOTS_TEMPORAL_DIR / "fires_by_season.png", dpi=FIGURE_DPI)

        # Heatmap: year × month
        if COL_FIRE_YEAR in work.columns:
            work["_month"] = disc_col_parsed.dt.month
            pivot = work.groupby([COL_FIRE_YEAR, "_month"]).size().unstack(fill_value=0)
            pivot.columns = [_MONTH_NAMES[c] if 1 <= c <= 12 else str(c) for c in pivot.columns]
            fig3, ax3 = plt.subplots(figsize=(14, 7))
            im = ax3.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
            ax3.set_xticks(range(len(pivot.columns)))
            ax3.set_xticklabels(pivot.columns, fontsize=9)
            ax3.set_yticks(range(len(pivot.index)))
            ax3.set_yticklabels(pivot.index.astype(str), fontsize=9)
            ax3.set_xlabel("Month", fontsize=10)
            ax3.set_ylabel("Year", fontsize=10)
            ax3.set_title("Fire Count Heatmap: Year × Month", fontsize=12, fontweight="bold")
            plt.colorbar(im, ax=ax3, label="Fire Count")
            save_figure(fig3, PLOTS_TEMPORAL_DIR / "discovery_heatmap.png", dpi=FIGURE_DPI)
            logger.info(f"  ✔ Discovery heatmap saved")

    # ── Duration ──────────────────────────────────────────────────────────────
    if COL_DISCOVERY_DATE in work.columns and COL_CONT_DATE in work.columns:
        disc = work[COL_DISCOVERY_DATE]
        cont = work[COL_CONT_DATE]
        if pd.api.types.is_datetime64_any_dtype(disc) and pd.api.types.is_datetime64_any_dtype(cont):
            duration = (cont - disc).dt.days
            valid_dur = duration[(duration >= 0) & (duration < 3650)]  # cap at 10 years

            dur_stats = pd.DataFrame([{
                "Mean Duration (days)":   round(float(valid_dur.mean()), 2),
                "Median Duration (days)": round(float(valid_dur.median()), 2),
                "Max Duration (days)":    round(float(valid_dur.max()), 2),
                "P90 Duration (days)":    round(float(valid_dur.quantile(0.90)), 2),
                "P95 Duration (days)":    round(float(valid_dur.quantile(0.95)), 2),
                "Fires with Duration":    int(valid_dur.count()),
                "Fires Missing Duration": int(duration.isna().sum()),
            }])
            save_csv(dur_stats, TABLES_DIR / "temporal_duration.csv")
            results["duration"] = dur_stats

            fig4, ax4 = plt.subplots(figsize=(12, 5))
            ax4.hist(valid_dur.dropna().clip(0, 60), bins=60, color="teal", edgecolor="white", linewidth=0.5)
            ax4.set_xlabel("Duration (days, clipped at 60)", fontsize=11)
            ax4.set_ylabel("Count", fontsize=11)
            ax4.set_title("Fire Duration Distribution (Discovery → Containment)", fontsize=12, fontweight="bold")
            save_figure(fig4, PLOTS_TEMPORAL_DIR / "duration_distribution.png", dpi=FIGURE_DPI)
            logger.info(f"  ✔ Duration distribution saved")

    # Console
    print("\n" + "=" * 60)
    print("  TEMPORAL ANALYSIS")
    print("=" * 60)
    if "year" in results:
        print("  Fire counts by year:")
        for _, r in results["year"].iterrows():
            print(f"    {int(r['Year'])}: {int(r['Fire Count']):>8,}")
    if "season" in results:
        print("\n  Fire counts by season:")
        for _, r in results["season"].iterrows():
            print(f"    {r['Season']:<10}: {int(r['Fire Count']):>8,}")
    print("=" * 60 + "\n")

    logger.info("  ✔ Temporal analysis complete.")
    return results
