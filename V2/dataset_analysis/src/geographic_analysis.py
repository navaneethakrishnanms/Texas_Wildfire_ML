"""
src/geographic_analysis.py
---------------------------
Analysis 8: Geographic Analysis.

Analyzes the spatial dimensions of the dataset:
  - Latitude distribution
  - Longitude distribution
  - County distribution (top N)
  - State distribution
  - Ecoregion distribution
  - Land cover distribution

Saves
-----
tables/geo_lat_stats.csv
tables/geo_lon_stats.csv
tables/geo_county_counts.csv
tables/geo_state_counts.csv
tables/geo_ecoregion_counts.csv
tables/geo_landcover_counts.csv
plots/geographic/lat_distribution.png
plots/geographic/lon_distribution.png
plots/geographic/lat_lon_scatter.png
plots/geographic/county_top20.png
plots/geographic/state_distribution.png
plots/geographic/ecoregion_distribution.png
plots/geographic/fire_size_by_state.png
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
    COL_COUNTY,
    COL_FIRE_SIZE,
    COL_LATITUDE,
    COL_LONGITUDE,
    COL_STATE,
    FIGURE_DPI,
    FIGURE_SIZE_WIDE,
    LOG_FILE,
    MATPLOTLIB_STYLE,
    PLOTS_GEOGRAPHIC_DIR,
    TABLES_DIR,
)
from src.utils import ensure_dirs, save_csv, save_figure, setup_logger

logger = setup_logger(__name__, LOG_FILE)

_TOP_N_COUNTIES = 30
_TOP_N_STATES   = 55


def _plot_lat_lon_distributions(df: pd.DataFrame) -> None:
    """Side-by-side histograms for latitude and longitude."""
    ensure_dirs(PLOTS_GEOGRAPHIC_DIR)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if COL_LATITUDE in df.columns:
        lat = df[COL_LATITUDE].dropna()
        axes[0].hist(lat, bins=80, color="steelblue", edgecolor="white", linewidth=0.5)
        axes[0].set_xlabel("Latitude (°N)", fontsize=11)
        axes[0].set_ylabel("Fire Count", fontsize=11)
        axes[0].set_title("Latitude Distribution", fontsize=12, fontweight="bold")
        axes[0].axvline(float(lat.mean()), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {lat.mean():.2f}°")
        axes[0].legend(fontsize=9)

    if COL_LONGITUDE in df.columns:
        lon = df[COL_LONGITUDE].dropna()
        axes[1].hist(lon, bins=80, color="darkorange", edgecolor="white", linewidth=0.5)
        axes[1].set_xlabel("Longitude (°W)", fontsize=11)
        axes[1].set_ylabel("Fire Count", fontsize=11)
        axes[1].set_title("Longitude Distribution", fontsize=12, fontweight="bold")
        axes[1].axvline(float(lon.mean()), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {lon.mean():.2f}°")
        axes[1].legend(fontsize=9)

    fig.suptitle("Geographic Distribution — Lat / Lon", fontsize=14, fontweight="bold")
    save_figure(fig, PLOTS_GEOGRAPHIC_DIR / "lat_distribution.png", dpi=FIGURE_DPI)
    logger.info("  ✔ Lat/Lon histogram saved")


def _plot_lat_lon_scatter(df: pd.DataFrame) -> None:
    """Scatter plot of fire locations (lat vs lon)."""
    ensure_dirs(PLOTS_GEOGRAPHIC_DIR)

    if COL_LATITUDE not in df.columns or COL_LONGITUDE not in df.columns:
        return

    sub = df[[COL_LATITUDE, COL_LONGITUDE]].dropna()
    if len(sub) > 50_000:
        sub = sub.sample(50_000, random_state=42)

    fig, ax = plt.subplots(figsize=(14, 8))
    sc = ax.scatter(
        sub[COL_LONGITUDE], sub[COL_LATITUDE],
        s=1, alpha=0.15, c="crimson", rasterized=True,
    )
    ax.set_xlabel("Longitude (°)", fontsize=11)
    ax.set_ylabel("Latitude (°)", fontsize=11)
    ax.set_title("Fire Locations — Lat/Lon Scatter (sampled 50k)", fontsize=13, fontweight="bold")
    save_figure(fig, PLOTS_GEOGRAPHIC_DIR / "lat_lon_scatter.png", dpi=FIGURE_DPI)
    logger.info("  ✔ Lat/Lon scatter plot saved")


def _plot_top_counties(county_df: pd.DataFrame) -> None:
    ensure_dirs(PLOTS_GEOGRAPHIC_DIR)
    top = county_df.head(30)
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(6, len(top) * 0.38)))
    ax.barh(top["County"], top["Fire Count"], color="teal", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Fire Count", fontsize=11)
    ax.set_title("Top 30 Counties by Fire Count", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8)
    save_figure(fig, PLOTS_GEOGRAPHIC_DIR / "county_top20.png", dpi=FIGURE_DPI)
    logger.info("  ✔ County plot saved")


def _plot_state_distribution(state_df: pd.DataFrame) -> None:
    ensure_dirs(PLOTS_GEOGRAPHIC_DIR)
    top = state_df.head(25)
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(top["State"], top["Fire Count"], color="mediumpurple", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("State", fontsize=11)
    ax.set_ylabel("Fire Count", fontsize=11)
    ax.set_title("Fire Count by State (Top 25)", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    save_figure(fig, PLOTS_GEOGRAPHIC_DIR / "state_distribution.png", dpi=FIGURE_DPI)
    logger.info("  ✔ State distribution plot saved")


def _plot_fire_size_by_state(df: pd.DataFrame, state_df: pd.DataFrame) -> None:
    ensure_dirs(PLOTS_GEOGRAPHIC_DIR)
    if COL_FIRE_SIZE not in df.columns or COL_STATE not in df.columns:
        return
    top_states = state_df.head(20)["State"].tolist()
    sub = df[df[COL_STATE].isin(top_states)]
    size_by_state = sub.groupby(COL_STATE)[COL_FIRE_SIZE].median().sort_values(ascending=False).reset_index()
    size_by_state.columns = ["State", "Median Fire Size (acres)"]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(size_by_state["State"], size_by_state["Median Fire Size (acres)"], color="darksalmon", edgecolor="white")
    ax.set_xlabel("State", fontsize=11)
    ax.set_ylabel("Median Fire Size (acres)", fontsize=11)
    ax.set_title("Median Fire Size by State (Top 20 by Count)", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    save_figure(fig, PLOTS_GEOGRAPHIC_DIR / "fire_size_by_state.png", dpi=FIGURE_DPI)
    logger.info("  ✔ Fire size by state plot saved")


def generate_geographic_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Analysis 8: Geographic Analysis.

    Parameters
    ----------
    df : The merged dataset (never modified).

    Returns
    -------
    dict[str, pd.DataFrame]  All geographic summary tables.
    """
    logger.info("Analysis 8 — Geographic Analysis")
    ensure_dirs(TABLES_DIR, PLOTS_GEOGRAPHIC_DIR)
    results: dict[str, pd.DataFrame] = {}

    # ── Lat stats ─────────────────────────────────────────────────────────────
    if COL_LATITUDE in df.columns:
        lat = df[COL_LATITUDE].dropna()
        lat_stats = pd.DataFrame([{
            "Mean": round(float(lat.mean()), 6),
            "Median": round(float(lat.median()), 6),
            "Std": round(float(lat.std()), 6),
            "Min": round(float(lat.min()), 6),
            "Max": round(float(lat.max()), 6),
            "Q1": round(float(lat.quantile(0.25)), 6),
            "Q3": round(float(lat.quantile(0.75)), 6),
            "Non-Null Count": int(lat.count()),
        }])
        save_csv(lat_stats, TABLES_DIR / "geo_lat_stats.csv")
        results["lat"] = lat_stats

    if COL_LONGITUDE in df.columns:
        lon = df[COL_LONGITUDE].dropna()
        lon_stats = pd.DataFrame([{
            "Mean": round(float(lon.mean()), 6),
            "Median": round(float(lon.median()), 6),
            "Std": round(float(lon.std()), 6),
            "Min": round(float(lon.min()), 6),
            "Max": round(float(lon.max()), 6),
            "Q1": round(float(lon.quantile(0.25)), 6),
            "Q3": round(float(lon.quantile(0.75)), 6),
            "Non-Null Count": int(lon.count()),
        }])
        save_csv(lon_stats, TABLES_DIR / "geo_lon_stats.csv")
        results["lon"] = lon_stats

    # ── County ────────────────────────────────────────────────────────────────
    if COL_COUNTY in df.columns:
        county_cnt = (
            df[COL_COUNTY]
            .value_counts()
            .reset_index()
        )
        county_cnt.columns = ["County", "Fire Count"]
        county_cnt["% of Total"] = (county_cnt["Fire Count"] / len(df) * 100).round(4)
        save_csv(county_cnt, TABLES_DIR / "geo_county_counts.csv")
        results["county"] = county_cnt
        _plot_top_counties(county_cnt)

    # ── State ─────────────────────────────────────────────────────────────────
    if COL_STATE in df.columns:
        state_cnt = (
            df[COL_STATE]
            .value_counts()
            .reset_index()
        )
        state_cnt.columns = ["State", "Fire Count"]
        state_cnt["% of Total"] = (state_cnt["Fire Count"] / len(df) * 100).round(4)
        save_csv(state_cnt, TABLES_DIR / "geo_state_counts.csv")
        results["state"] = state_cnt
        _plot_state_distribution(state_cnt)
        _plot_fire_size_by_state(df, state_cnt)

    # ── Ecoregion ─────────────────────────────────────────────────────────────
    for eco_col in ["ECOREGION", "ECO3", "ECO4", "NA_L3NAME", "NA_L2NAME"]:
        if eco_col in df.columns:
            eco_cnt = df[eco_col].value_counts().reset_index()
            eco_cnt.columns = ["Ecoregion", "Fire Count"]
            eco_cnt["% of Total"] = (eco_cnt["Fire Count"] / len(df) * 100).round(4)
            save_csv(eco_cnt, TABLES_DIR / "geo_ecoregion_counts.csv")
            results["ecoregion"] = eco_cnt

            top_eco = eco_cnt.head(20)
            fig, ax = plt.subplots(figsize=(14, max(6, len(top_eco) * 0.4)))
            ax.barh(top_eco["Ecoregion"], top_eco["Fire Count"], color="olivedrab", edgecolor="white")
            ax.set_xlabel("Fire Count", fontsize=11)
            ax.set_title(f"Fire Count by Ecoregion ({eco_col})", fontsize=12, fontweight="bold")
            ax.invert_yaxis()
            ax.tick_params(axis="y", labelsize=8)
            save_figure(fig, PLOTS_GEOGRAPHIC_DIR / "ecoregion_distribution.png", dpi=FIGURE_DPI)
            logger.info(f"  ✔ Ecoregion plot saved (using {eco_col})")
            break

    # ── Land Cover ────────────────────────────────────────────────────────────
    for lc_col in ["NLCD_CLASS", "LANDCOVER", "LAND_COVER", "LULC"]:
        if lc_col in df.columns:
            lc_cnt = df[lc_col].value_counts().reset_index()
            lc_cnt.columns = ["Land Cover", "Fire Count"]
            save_csv(lc_cnt, TABLES_DIR / "geo_landcover_counts.csv")
            results["landcover"] = lc_cnt
            break

    # Plots
    _plot_lat_lon_distributions(df)
    _plot_lat_lon_scatter(df)

    # Console
    print("\n" + "=" * 60)
    print("  GEOGRAPHIC ANALYSIS")
    print("=" * 60)
    if "lat" in results:
        lat_row = results["lat"].iloc[0]
        print(f"  Latitude  →  Min: {lat_row['Min']:.2f}°  Max: {lat_row['Max']:.2f}°  Mean: {lat_row['Mean']:.2f}°")
    if "lon" in results:
        lon_row = results["lon"].iloc[0]
        print(f"  Longitude →  Min: {lon_row['Min']:.2f}°  Max: {lon_row['Max']:.2f}°  Mean: {lon_row['Mean']:.2f}°")
    if "state" in results:
        print(f"\n  States with fires: {len(results['state'])}")
        for _, r in results["state"].head(5).iterrows():
            print(f"    {r['State']}: {int(r['Fire Count']):,} fires ({r['% of Total']:.1f}%)")
    if "county" in results:
        print(f"\n  Counties with fires: {len(results['county'])}")
    print("=" * 60 + "\n")

    logger.info("  ✔ Geographic analysis complete.")
    return results
