"""
03_correlation_analysis.py -- Texas Wildfire Correlation Analysis
==================================================================
Outputs:
  full_correlation_matrix.csv
  correlation_heatmap_top30.png
  correlation_with_fire_size.png
  subgroup_heatmaps.png
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import matplotlib.pyplot as plt
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
    "text.color":       FG,
})

WEATHER_COLS = ["erc", "bi", "vpd", "fm100", "fm1000", "tmmx", "tmmn",
                "vs", "rmin", "rmax", "srad", "pr", "etr", "sph"]
TERRAIN_COLS = ["Elevation", "Slope", "TRI", "TPI", "Aspect", "GHM"]
NDVI_COLS    = ["NDVI_mean", "NDVI_min", "NDVI_max", "MOD_NDVI_12m"]
SOCIAL_COLS  = ["Population", "GDP", "RPL_THEMES", "RPL_THEME1"]

# ─── LOAD ────────────────────────────────────────────────────────
print(f"[03] Loading {STATE_NAME} dataset ...")
OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(DATA_PATH)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Exclude ID-like and source_year
exclude  = {"FOD_ID", "source_year", "DISCOVERY_DOY", "CONT_DOY"}
num_cols = [c for c in num_cols if c not in exclude]
num_df   = df[num_cols].copy()

# ─── 1. FULL CORRELATION MATRIX CSV ──────────────────────────────
print("[03] Computing full correlation matrix ...")
corr_full = num_df.corr(method="pearson", numeric_only=True)
corr_full.to_csv(OUT_DIR / "full_correlation_matrix.csv", encoding="utf-8")
print("     Saved -> full_correlation_matrix.csv")

# ─── 2. TOP-30 HEATMAP (by variance, lower triangle) ─────────────
print("[03] Plotting correlation_heatmap_top30.png ...")
top30 = num_df.var().nlargest(30).index.tolist()
corr30 = num_df[top30].corr()
mask   = np.triu(np.ones_like(corr30, dtype=bool))

fig, ax = plt.subplots(figsize=(22, 18), facecolor=BG)
ax.set_facecolor(BG)
sns.heatmap(
    corr30, mask=mask, annot=True, fmt=".2f", linewidths=0.3,
    cmap="coolwarm", vmin=-1, vmax=1,
    annot_kws={"size": 6, "color": FG},
    cbar_kws={"shrink": 0.7},
    ax=ax
)
ax.set_title(f"{STATE_NAME} -- Correlation Heatmap (Top 30 Features by Variance)",
             color=FG, fontsize=14, pad=15)
ax.tick_params(colors=FG, labelsize=8)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.setp(ax.get_yticklabels(), rotation=0)
ax.figure.axes[-1].tick_params(colors=FG, labelsize=8)
plt.tight_layout()
fig.savefig(OUT_DIR / "correlation_heatmap_top30.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("     Saved -> correlation_heatmap_top30.png")

# ─── 3. CORRELATION WITH FIRE_SIZE (horizontal bar) ──────────────
print("[03] Plotting correlation_with_fire_size.png ...")
if "FIRE_SIZE" in num_df.columns:
    corr_fs = (
        num_df.drop(columns=["FIRE_SIZE"], errors="ignore")
              .corrwith(num_df["FIRE_SIZE"])
              .dropna()
              .sort_values(key=abs, ascending=False)
              .head(40)
    )
    colors = [FIRE_C if v > 0 else NEG_C for v in corr_fs.values]
    fig, ax = plt.subplots(figsize=(16, 14), facecolor=BG)
    ax.set_facecolor(BG)
    ax.barh(corr_fs.index, corr_fs.values, color=colors, edgecolor=GRID, linewidth=0.4)
    ax.axvline(0, color=FG, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Pearson r with FIRE_SIZE", color=FG, fontsize=12)
    ax.set_title(f"{STATE_NAME} -- Feature Correlation with FIRE_SIZE (Top 40)", color=FG, fontsize=14)
    ax.tick_params(colors=FG, labelsize=9)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    # Annotations
    for val, name in zip(corr_fs.values, corr_fs.index):
        offset = 0.003 if val >= 0 else -0.003
        ha     = "left" if val >= 0 else "right"
        ax.text(val + offset, name, f"{val:.3f}", va="center", ha=ha, color=FG, fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "correlation_with_fire_size.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("     Saved -> correlation_with_fire_size.png")

# ─── 4. SUBGROUP HEATMAPS (2x2 grid) ─────────────────────────────
print("[03] Plotting subgroup_heatmaps.png ...")
groups = {
    "Weather/Climate": [c for c in WEATHER_COLS if c in num_df.columns],
    "Terrain/Landscape": [c for c in TERRAIN_COLS + NDVI_COLS if c in num_df.columns],
    "5-Day Weather Means": [c for c in num_df.columns if "_5D_mean" in c],
    "Percentile Features": [c for c in num_df.columns if "_Percentile" in c],
}
groups = {k: v for k, v in groups.items() if len(v) >= 2}

if groups:
    n = len(groups)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 10), facecolor=BG)
    fig.suptitle(f"{STATE_NAME} -- Correlation Sub-Heatmaps by Feature Group", color=FG, fontsize=14)
    if n == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    flat_axes = [ax for row in axes for ax in (row if hasattr(row, "__iter__") else [row])]

    for idx, (grp_name, cols) in enumerate(groups.items()):
        ax = flat_axes[idx]
        ax.set_facecolor(BG)
        grp_corr = num_df[cols].corr()
        mask_grp  = np.triu(np.ones_like(grp_corr, dtype=bool))
        sns.heatmap(
            grp_corr, mask=mask_grp, annot=True, fmt=".2f", linewidths=0.4,
            cmap="coolwarm", vmin=-1, vmax=1,
            annot_kws={"size": 8},
            cbar_kws={"shrink": 0.7},
            ax=ax
        )
        ax.set_title(f"{grp_name}", color=ACCENT, fontsize=11)
        ax.tick_params(colors=FG, labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

    for idx in range(len(groups), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "subgroup_heatmaps.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("     Saved -> subgroup_heatmaps.png")

print(f"\n[03] DONE -- All outputs in {OUT_DIR}")
