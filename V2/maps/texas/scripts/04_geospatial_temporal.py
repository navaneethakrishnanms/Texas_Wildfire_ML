"""
04_geospatial_temporal.py -- Texas Wildfire Geospatial & Temporal Analysis
===========================================================================
Outputs:
  geo_scatter_by_cause.png
  geo_scatter_by_size.png
  geo_hexbin_density.png
  temporal_fires_per_year.png
  temporal_fires_per_month.png
  temporal_month_year_heatmap.png
  temporal_avg_erc_per_year.png
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

# ─── CONFIG ──────────────────────────────────────────────────────
STATE_CODE = "TX"
STATE_NAME = "Texas"
BASE_DIR   = Path(__file__).resolve().parents[3]
DATA_PATH  = BASE_DIR / "data" / "processed" / "texas" / "texas_fire_2014_2020.parquet"
OUT_DIR    = BASE_DIR / "maps" / "texas" / "eda_outputs"

BG     = "#1a1a2e"
FG     = "#e0e0e0"
GRID   = "#333355"
FIRE_C = "#FF6B35"
NEG_C  = "#004E89"
ACCENT = "#F7C548"
DPI    = 150

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID,
    "axes.labelcolor":  FG, "xtick.color":    FG, "ytick.color":    FG,
    "text.color":       FG, "grid.color":     GRID,
})

# ─── LOAD ────────────────────────────────────────────────────────
print(f"[04] Loading {STATE_NAME} dataset ...")
OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(DATA_PATH)
df["large_fire"] = (df["FIRE_SIZE"] >= 100).astype(int)

# Parse date
if "DISCOVERY_DATE" in df.columns:
    df["disc_date"] = pd.to_datetime(df["DISCOVERY_DATE"], errors="coerce")
    df["disc_month"] = df["disc_date"].dt.month
    df["disc_year"]  = df["disc_date"].dt.year
elif "FIRE_YEAR" in df.columns:
    df["disc_year"] = df["FIRE_YEAR"]

lat_col  = "LATITUDE"
lon_col  = "LONGITUDE"
cause_col = "NWCG_GENERAL_CAUSE"

geo_df = df.dropna(subset=[lat_col, lon_col]).copy()
print(f"     Rows with valid lat/lon: {len(geo_df):,}")

# ─── CAUSE COLOR MAP ─────────────────────────────────────────────
CAUSE_COLORS = {
    "Lightning":         "#00B4D8",
    "Human":             FIRE_C,
    "Equipment Use":     ACCENT,
    "Debris Burning":    "#2ECC71",
    "Arson/Incendiary":  "#E74C3C",
    "Recreation":        "#9B59B6",
    "Children":          "#F39C12",
    "Railroad":          "#1ABC9C",
    "Smoking":           "#BDC3C7",
    "Miscellaneous":     "#7F8C8D",
}
DEFAULT_CAUSE_COLOR = "#888888"

# ─── 1. GEO SCATTER BY CAUSE ─────────────────────────────────────
print("[04] Plotting geo_scatter_by_cause.png ...")
if cause_col in geo_df.columns:
    top_causes = geo_df[cause_col].value_counts().head(8).index.tolist()
    palette    = [CAUSE_COLORS.get(c, DEFAULT_CAUSE_COLOR) for c in top_causes]

    fig, ax = plt.subplots(figsize=(18, 12), facecolor=BG)
    ax.set_facecolor(BG)
    # Background: all points
    ax.scatter(geo_df[lon_col], geo_df[lat_col], s=1, c=GRID, alpha=0.15, rasterized=True)
    # Overlay each cause
    for cause, color in zip(top_causes, palette):
        sub = geo_df[geo_df[cause_col] == cause]
        ax.scatter(sub[lon_col], sub[lat_col], s=4, c=color, alpha=0.5,
                   label=f"{cause} ({len(sub):,})", rasterized=True)
    ax.set_xlabel("Longitude", color=FG, fontsize=11)
    ax.set_ylabel("Latitude", color=FG, fontsize=11)
    ax.set_title(f"{STATE_NAME} -- Fire Locations by Cause (2014-2020)", color=FG, fontsize=14)
    ax.tick_params(colors=FG)
    ax.grid(True, alpha=0.2)
    leg = ax.legend(fontsize=8, labelcolor=FG, facecolor=BG, edgecolor=GRID,
                    markerscale=3, loc="lower left")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "geo_scatter_by_cause.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("     Saved -> geo_scatter_by_cause.png")

# ─── 2. GEO SCATTER BY FIRE SIZE ─────────────────────────────────
print("[04] Plotting geo_scatter_by_size.png ...")
fig, ax = plt.subplots(figsize=(18, 12), facecolor=BG)
ax.set_facecolor(BG)
size_vals = np.log1p(geo_df["FIRE_SIZE"].clip(lower=0))
sc = ax.scatter(geo_df[lon_col], geo_df[lat_col], s=3, c=size_vals,
                cmap="plasma", alpha=0.5, rasterized=True)
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label("log(1 + FIRE_SIZE) [acres]", color=FG, fontsize=10)
cbar.ax.tick_params(colors=FG, labelsize=8)
ax.set_xlabel("Longitude", color=FG, fontsize=11)
ax.set_ylabel("Latitude", color=FG, fontsize=11)
ax.set_title(f"{STATE_NAME} -- Fire Locations by Size (log scale)", color=FG, fontsize=14)
ax.tick_params(colors=FG)
ax.grid(True, alpha=0.2)
plt.tight_layout()
fig.savefig(OUT_DIR / "geo_scatter_by_size.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("     Saved -> geo_scatter_by_size.png")

# ─── 3. HEXBIN DENSITY ───────────────────────────────────────────
print("[04] Plotting geo_hexbin_density.png ...")
fig, ax = plt.subplots(figsize=(18, 12), facecolor=BG)
ax.set_facecolor(BG)
hb = ax.hexbin(geo_df[lon_col], geo_df[lat_col], gridsize=60, cmap="YlOrRd",
               mincnt=1, linewidths=0.1)
cbar = plt.colorbar(hb, ax=ax, shrink=0.6)
cbar.set_label("Fire Count per Hex Cell", color=FG, fontsize=10)
cbar.ax.tick_params(colors=FG, labelsize=8)
ax.set_xlabel("Longitude", color=FG, fontsize=11)
ax.set_ylabel("Latitude", color=FG, fontsize=11)
ax.set_title(f"{STATE_NAME} -- Fire Density Hexbin Map (2014-2020)", color=FG, fontsize=14)
ax.tick_params(colors=FG)
plt.tight_layout()
fig.savefig(OUT_DIR / "geo_hexbin_density.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("     Saved -> geo_hexbin_density.png")

# ─── 4. FIRES PER YEAR ───────────────────────────────────────────
print("[04] Plotting temporal_fires_per_year.png ...")
if "FIRE_YEAR" in df.columns:
    yr_counts = df["FIRE_YEAR"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.bar(yr_counts.index.astype(str), yr_counts.values,
                  color=FIRE_C, edgecolor=GRID, linewidth=0.5)
    ax.set_xlabel("Year", color=FG, fontsize=12)
    ax.set_ylabel("Number of Fires", color=FG, fontsize=12)
    ax.set_title(f"{STATE_NAME} -- Fire Events per Year", color=FG, fontsize=14)
    ax.tick_params(colors=FG)
    ax.grid(axis="y", color=GRID, alpha=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{int(bar.get_height()):,}", ha="center", color=FG, fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "temporal_fires_per_year.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("     Saved -> temporal_fires_per_year.png")

# ─── 5. FIRES PER MONTH ──────────────────────────────────────────
print("[04] Plotting temporal_fires_per_month.png ...")
if "disc_month" in df.columns:
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    mo_counts = df["disc_month"].value_counts().sort_index()
    mo_labels = [month_names.get(int(m), str(m)) for m in mo_counts.index]
    palette   = plt.cm.plasma(np.linspace(0.2, 0.9, len(mo_counts)))
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.bar(mo_labels, mo_counts.values, color=palette, edgecolor=GRID, linewidth=0.5)
    ax.set_xlabel("Month", color=FG, fontsize=12)
    ax.set_ylabel("Number of Fires", color=FG, fontsize=12)
    ax.set_title(f"{STATE_NAME} -- Fire Events per Month (2014-2020 aggregate)", color=FG, fontsize=14)
    ax.tick_params(colors=FG)
    ax.grid(axis="y", color=GRID, alpha=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{int(bar.get_height()):,}", ha="center", color=FG, fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "temporal_fires_per_month.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("     Saved -> temporal_fires_per_month.png")

# ─── 6. MONTH x YEAR HEATMAP ─────────────────────────────────────
print("[04] Plotting temporal_month_year_heatmap.png ...")
if "disc_month" in df.columns and "disc_year" in df.columns:
    pivot = df.groupby(["disc_year", "disc_month"]).size().unstack(fill_value=0)
    pivot.columns = [month_names.get(int(c), str(c)) for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG)
    ax.set_facecolor(BG)
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.3,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title(f"{STATE_NAME} -- Fire Count Heatmap (Year x Month)", color=FG, fontsize=14)
    ax.set_xlabel("Month", color=FG, fontsize=11)
    ax.set_ylabel("Year", color=FG, fontsize=11)
    ax.tick_params(colors=FG, labelsize=9)
    ax.figure.axes[-1].tick_params(colors=FG, labelsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "temporal_month_year_heatmap.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("     Saved -> temporal_month_year_heatmap.png")

# ─── 7. AVG ERC PER YEAR (small vs large) ────────────────────────
print("[04] Plotting temporal_avg_erc_per_year.png ...")
if "erc" in df.columns and "FIRE_YEAR" in df.columns:
    erc_yr = df.groupby(["FIRE_YEAR", "large_fire"])["erc"].mean().unstack()
    erc_yr.columns = ["Small Fire (<100ac)", "Large Fire (>=100ac)"]
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_facecolor(BG)
    if "Small Fire (<100ac)" in erc_yr.columns:
        ax.plot(erc_yr.index, erc_yr["Small Fire (<100ac)"], marker="o",
                color=NEG_C, linewidth=2, markersize=7, label="Small Fire (<100ac)")
    if "Large Fire (>=100ac)" in erc_yr.columns:
        ax.plot(erc_yr.index, erc_yr["Large Fire (>=100ac)"], marker="s",
                color=FIRE_C, linewidth=2, markersize=7, label="Large Fire (>=100ac)")
    ax.set_xlabel("Year", color=FG, fontsize=12)
    ax.set_ylabel("Mean ERC", color=FG, fontsize=12)
    ax.set_title(f"{STATE_NAME} -- Average ERC per Year by Fire Size Class", color=FG, fontsize=14)
    ax.tick_params(colors=FG)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, labelcolor=FG, facecolor=BG, edgecolor=GRID)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "temporal_avg_erc_per_year.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("     Saved -> temporal_avg_erc_per_year.png")

print(f"\n[04] DONE -- All outputs in {OUT_DIR}")
