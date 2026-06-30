"""
02_distributions.py -- Texas Wildfire Feature Distributions
=============================================================
Outputs:
  numeric_histograms_weather.png
  numeric_histograms_terrain.png
  boxplots_by_fire_size.png
  violin_top_features.png
  cause_distribution.png
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# â”€â”€â”€ CONFIG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
STATE_CODE = "CA"
STATE_NAME = "California"
BASE_DIR   = Path(__file__).resolve().parents[3]
DATA_PATH  = BASE_DIR / "data" / "processed" / "california" / "california_fire_2014_2020.parquet"
OUT_DIR    = BASE_DIR / "maps" / "california" / "eda_outputs"

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

WEATHER_FEATURES = [
    "erc", "bi", "vpd", "fm100", "fm1000", "tmmx", "tmmn",
    "vs", "rmin", "rmax", "srad", "pr", "etr", "sph", "th",
]
TERRAIN_FEATURES = [
    "Elevation", "Slope", "TRI", "TPI", "Aspect", "GHM",
    "NDVI_mean", "NDVI_min", "NDVI_max",
]

# â”€â”€â”€ LOAD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"[02] Loading {STATE_NAME} dataset ...")
OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(DATA_PATH)
df["large_fire"] = (df["FIRE_SIZE"] >= 100).astype(int)
small_df = df[df["large_fire"] == 0]
large_df = df[df["large_fire"] == 1]
print(f"     Small fires: {len(small_df):,}  |  Large fires: {len(large_df):,}")

# â”€â”€â”€ HELPER: feature histogram panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def plot_hist_panel(features, title_suffix, fname):
    # Keep only features that are present AND numeric (or coercible to numeric)
    available = []
    for f in features:
        if f not in df.columns:
            continue
        if pd.to_numeric(df[f], errors="coerce").dropna().empty:
            continue
        available.append(f)
    if not available:
        print(f"     No numeric features available for {title_suffix} -- skipping")
        return
    ncols = 4
    nrows = int(np.ceil(len(available) / ncols))
    fig = plt.figure(figsize=(22, nrows * 4 + 1), facecolor=BG)
    fig.suptitle(f"{STATE_NAME} -- {title_suffix} Distributions", color=FG, fontsize=15, y=0.98)

    for i, feat in enumerate(available, 1):
        ax = fig.add_subplot(nrows, ncols, i)
        ax.set_facecolor(BG)
        all_num = pd.to_numeric(df[feat], errors="coerce")
        s_data  = pd.to_numeric(small_df[feat], errors="coerce").dropna()
        l_data  = pd.to_numeric(large_df[feat], errors="coerce").dropna()
        bins    = min(60, max(20, int(all_num.dropna().shape[0] ** 0.4)))
        lo      = float(all_num.quantile(0.01))
        hi      = float(all_num.quantile(0.99))
        rng     = (lo, hi) if lo < hi else None
        if not s_data.empty:
            ax.hist(s_data, bins=bins, range=rng, color=NEG_C, alpha=0.6,
                    density=True, label="Small (<100ac)")
        if not l_data.empty:
            ax.hist(l_data, bins=bins, range=rng, color=FIRE_C, alpha=0.75,
                    density=True, label="Large (>=100ac)")
        ax.set_title(feat, color=ACCENT, fontsize=9)
        ax.tick_params(labelsize=7, colors=FG)
        ax.grid(True, alpha=0.3)
        if i == 1:
            ax.legend(fontsize=7, labelcolor=FG, facecolor=BG, edgecolor=GRID)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / fname, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"     Saved -> {fname}")

# â”€â”€â”€ 1. WEATHER HISTOGRAMS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[02] Plotting weather histograms ...")
plot_hist_panel(WEATHER_FEATURES, "Weather/Climate Feature", "numeric_histograms_weather.png")

# â”€â”€â”€ 2. TERRAIN HISTOGRAMS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[02] Plotting terrain histograms ...")
plot_hist_panel(TERRAIN_FEATURES, "Terrain/Landscape Feature", "numeric_histograms_terrain.png")

# â”€â”€â”€ 3. BOX PLOTS SPLIT BY FIRE SIZE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[02] Plotting boxplots_by_fire_size.png ...")
box_feats = [f for f in WEATHER_FEATURES[:12] if f in df.columns]
if box_feats:
    fig, axes = plt.subplots(3, 4, figsize=(22, 14), facecolor=BG)
    fig.suptitle(f"{STATE_NAME} -- Feature Box Plots by Fire Size Class", color=FG, fontsize=14, y=0.98)
    axes = axes.flatten()
    for i, feat in enumerate(box_feats):
        ax = axes[i]
        ax.set_facecolor(BG)
        data = [small_df[feat].dropna(), large_df[feat].dropna()]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color=ACCENT, linewidth=2),
                        whiskerprops=dict(color=FG), capprops=dict(color=FG),
                        flierprops=dict(marker=".", markersize=2, color=FG, alpha=0.3))
        bp["boxes"][0].set_facecolor(NEG_C)
        bp["boxes"][0].set_alpha(0.7)
        if len(bp["boxes"]) > 1:
            bp["boxes"][1].set_facecolor(FIRE_C)
            bp["boxes"][1].set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Small\n(<100ac)", "Large\n(>=100ac)"], color=FG, fontsize=8)
        ax.set_title(feat, color=ACCENT, fontsize=9)
        ax.tick_params(colors=FG, labelsize=7)
        ax.grid(True, alpha=0.3)
    for j in range(len(box_feats), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "boxplots_by_fire_size.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("     Saved -> boxplots_by_fire_size.png")

# â”€â”€â”€ 4. VIOLIN PLOTS (top discriminating features) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[02] Plotting violin_top_features.png ...")
violin_feats = [f for f in ["erc", "bi", "vpd", "fm100", "tmmx", "vs"] if f in df.columns]
if violin_feats:
    fig, axes = plt.subplots(1, len(violin_feats), figsize=(22, 7), facecolor=BG)
    fig.suptitle(f"{STATE_NAME} -- Violin Plots: Top Discriminating Features", color=FG, fontsize=14, y=1.02)
    if len(violin_feats) == 1:
        axes = [axes]
    for ax, feat in zip(axes, violin_feats):
        ax.set_facecolor(BG)
        s_data = small_df[feat].dropna().values
        l_data = large_df[feat].dropna().values
        vp = ax.violinplot([s_data, l_data], positions=[1, 2], showmedians=True, showextrema=False)
        vp["bodies"][0].set_facecolor(NEG_C);  vp["bodies"][0].set_alpha(0.7)
        vp["bodies"][1].set_facecolor(FIRE_C); vp["bodies"][1].set_alpha(0.7)
        vp["cmedians"].set_color(ACCENT)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Small", "Large"], color=FG, fontsize=9)
        ax.set_title(feat, color=ACCENT, fontsize=10)
        ax.tick_params(colors=FG, labelsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "violin_top_features.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("     Saved -> violin_top_features.png")

# â”€â”€â”€ 5. CAUSE DISTRIBUTION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[02] Plotting cause_distribution.png ...")
cause_col = "NWCG_GENERAL_CAUSE"
if cause_col in df.columns:
    vc = df[cause_col].value_counts(dropna=False).head(15)
    palette = plt.cm.plasma(np.linspace(0.15, 0.95, len(vc)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor=BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
    # Horizontal bar
    bars = ax1.barh(vc.index.astype(str), vc.values, color=palette, edgecolor=GRID)
    ax1.invert_yaxis()
    ax1.set_xlabel("Fire Count", color=FG, fontsize=11)
    ax1.set_title(f"{STATE_NAME} -- Fire Cause Distribution", color=FG, fontsize=13)
    ax1.tick_params(colors=FG, labelsize=9)
    for bar in bars:
        ax1.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                 f"{int(bar.get_width()):,}", va="center", color=FG, fontsize=8)
    # Pie
    ax2.pie(vc.values, labels=vc.index.astype(str), autopct="%1.1f%%",
            colors=palette, textprops={"color": FG, "fontsize": 8})
    ax2.set_title(f"{STATE_NAME} -- Cause Share (%)", color=FG, fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "cause_distribution.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("     Saved -> cause_distribution.png")

print(f"\n[02] DONE -- All outputs in {OUT_DIR}")

