"""
05_advanced_eda.py -- Texas Wildfire Advanced EDA
==================================================
Outputs:
  pairplot_weather_features.png
  pairplot_terrain_features.png
  scatter_feature_pairs.png
  outlier_analysis.png
  skewness_kurtosis.png
  feature_discriminative_power.png
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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

WEATHER_FEATS = ["erc", "bi", "vpd", "fm100", "tmmx", "vs", "rmin", "srad"]
TERRAIN_FEATS = ["Elevation", "Slope", "TRI", "GHM", "NDVI_mean"]

# â”€â”€â”€ LOAD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"[05] Loading {STATE_NAME} dataset ...")
OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(DATA_PATH)
df["large_fire"] = (df["FIRE_SIZE"] >= 100).astype(int)

small_df = df[df["large_fire"] == 0]
large_df = df[df["large_fire"] == 1]

# â”€â”€â”€ 1. PAIR PLOT: WEATHER FEATURES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[05] Plotting pairplot_weather_features.png ...")
import warnings
warnings.filterwarnings("ignore")
try:
    import seaborn as sns

    # Only keep features that are truly numeric in the dataframe
    w_feats = [f for f in WEATHER_FEATS if f in df.columns
               and pd.api.types.is_numeric_dtype(df[f])]
    SAMPLE  = 2500
    sample_s = small_df[w_feats + ["large_fire"]].dropna().sample(min(SAMPLE, len(small_df)), random_state=42)
    sample_l = large_df[w_feats + ["large_fire"]].dropna().sample(min(SAMPLE, len(large_df)), random_state=42)
    pair_df  = pd.concat([sample_s, sample_l])

    palette  = {0: NEG_C, 1: FIRE_C}
    pg = sns.pairplot(
        pair_df, vars=w_feats, hue="large_fire", palette=palette,
        diag_kind="kde", plot_kws={"alpha": 0.3, "s": 8},
        diag_kws={"linewidth": 1.5},
    )
    pg.fig.set_facecolor(BG)
    for ax in pg.axes.flatten():
        if ax:
            ax.set_facecolor(BG)
            ax.tick_params(colors=FG, labelsize=7)
            ax.xaxis.label.set_color(FG)
            ax.yaxis.label.set_color(FG)
    pg._legend.get_frame().set_facecolor(BG)
    pg._legend.get_frame().set_edgecolor(GRID)
    for text in pg._legend.get_texts():
        text.set_color(FG)
    pg.fig.suptitle(f"{STATE_NAME} -- Pair Plot: Weather Features (sample {SAMPLE} per class)",
                    color=FG, fontsize=14, y=1.01)
    pg.fig.savefig(OUT_DIR / "pairplot_weather_features.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close("all")
    print("     Saved -> pairplot_weather_features.png")

    # Terrain pair plot -- numeric only
    t_feats  = [f for f in TERRAIN_FEATS if f in df.columns
                and pd.api.types.is_numeric_dtype(df[f])]
    sample_s2 = small_df[t_feats + ["large_fire"]].dropna().sample(min(SAMPLE, len(small_df)), random_state=42)
    sample_l2 = large_df[t_feats + ["large_fire"]].dropna().sample(min(SAMPLE, len(large_df)), random_state=42)
    pair_df2  = pd.concat([sample_s2, sample_l2])
    pg2 = sns.pairplot(
        pair_df2, vars=t_feats, hue="large_fire", palette=palette,
        diag_kind="kde", plot_kws={"alpha": 0.3, "s": 8},
        diag_kws={"linewidth": 1.5},
    )
    pg2.fig.set_facecolor(BG)
    for ax in pg2.axes.flatten():
        if ax:
            ax.set_facecolor(BG)
            ax.tick_params(colors=FG, labelsize=7)
            ax.xaxis.label.set_color(FG)
            ax.yaxis.label.set_color(FG)
    pg2._legend.get_frame().set_facecolor(BG)
    for text in pg2._legend.get_texts():
        text.set_color(FG)
    pg2.fig.suptitle(f"{STATE_NAME} -- Pair Plot: Terrain Features (sample {SAMPLE} per class)",
                     color=FG, fontsize=14, y=1.01)
    pg2.fig.savefig(OUT_DIR / "pairplot_terrain_features.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close("all")
    print("     Saved -> pairplot_terrain_features.png")
except Exception as e:
    print(f"     Pair plots skipped: {e}")

# â”€â”€â”€ 2. 6 FEATURE PAIR SCATTER PLOTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[05] Plotting scatter_feature_pairs.png ...")
pair_list = [
    ("erc",       "FIRE_SIZE"),
    ("bi",        "erc"),
    ("vpd",       "tmmx"),
    ("fm100",     "rmin"),
    ("Elevation", "FIRE_SIZE"),
    ("NDVI_mean", "erc"),
]
pair_list = [(a, b) for a, b in pair_list if a in df.columns and b in df.columns]
if pair_list:
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), facecolor=BG)
    fig.suptitle(f"{STATE_NAME} -- Feature Pair Scatter Plots", color=FG, fontsize=14, y=0.98)
    axes = axes.flatten()
    for i, (fx, fy) in enumerate(pair_list[:6]):
        ax = axes[i]
        ax.set_facecolor(BG)
        s_sub = small_df[[fx, fy]].dropna().sample(min(2000, len(small_df)), random_state=42)
        l_sub = large_df[[fx, fy]].dropna().sample(min(2000, len(large_df)), random_state=42)
        ax.scatter(s_sub[fx], s_sub[fy], s=4, color=NEG_C, alpha=0.4,
                   label="Small (<100ac)", rasterized=True)
        ax.scatter(l_sub[fx], l_sub[fy], s=6, color=FIRE_C, alpha=0.6,
                   label="Large (>=100ac)", rasterized=True)
        ax.set_xlabel(fx, color=FG, fontsize=9)
        ax.set_ylabel(fy, color=FG, fontsize=9)
        ax.set_title(f"{fx} vs {fy}", color=ACCENT, fontsize=10)
        ax.tick_params(colors=FG, labelsize=7)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8, labelcolor=FG, facecolor=BG, edgecolor=GRID)
    for j in range(len(pair_list), 6):
        axes[j].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "scatter_feature_pairs.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("     Saved -> scatter_feature_pairs.png")

# â”€â”€â”€ 3. OUTLIER ANALYSIS (IQR method) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[05] Plotting outlier_analysis.png ...")
# Build numeric-only key_feats list for all downstream analyses
key_feats = [f for f in WEATHER_FEATS + TERRAIN_FEATS
             if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
outlier_pct = {}
for feat in key_feats:
    vals = pd.to_numeric(df[feat], errors="coerce").dropna()
    if len(vals) == 0:
        continue
    Q1, Q3 = vals.quantile(0.25), vals.quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out  = ((vals < lo) | (vals > hi)).sum()
    outlier_pct[feat] = n_out / len(vals) * 100

if outlier_pct:
    out_s  = pd.Series(outlier_pct).sort_values(ascending=False)
    colors = plt.cm.Oranges(out_s.values / max(out_s.values) * 0.8 + 0.2)
    fig, ax = plt.subplots(figsize=(16, 8), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.bar(out_s.index, out_s.values, color=colors, edgecolor=GRID, linewidth=0.4)
    ax.set_xlabel("Feature", color=FG, fontsize=11)
    ax.set_ylabel("Outlier Rows (%)", color=FG, fontsize=11)
    ax.set_title(f"{STATE_NAME} -- Outlier Analysis (IQR Method, 1.5x)", color=FG, fontsize=14)
    ax.tick_params(colors=FG, labelsize=8)
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, out_s.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:.1f}%", ha="center", color=FG, fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "outlier_analysis.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("     Saved -> outlier_analysis.png")

# â”€â”€â”€ 4. SKEWNESS AND KURTOSIS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[05] Plotting skewness_kurtosis.png ...")
num_df   = df[key_feats].apply(pd.to_numeric, errors="coerce").dropna(how="all")
skewness = num_df.skew().sort_values(key=abs, ascending=False)
kurtosis = num_df.kurt().sort_values(key=abs, ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8), facecolor=BG)
fig.suptitle(f"{STATE_NAME} -- Skewness and Kurtosis", color=FG, fontsize=14, y=0.98)
for ax in (ax1, ax2):
    ax.set_facecolor(BG)

colors_sk = [FIRE_C if v > 0 else NEG_C for v in skewness.values]
ax1.barh(skewness.index, skewness.values, color=colors_sk, edgecolor=GRID, linewidth=0.4)
ax1.axvline(0, color=FG, linewidth=0.8, linestyle="--")
ax1.set_xlabel("Skewness", color=FG, fontsize=11)
ax1.set_title("Skewness", color=ACCENT, fontsize=12)
ax1.tick_params(colors=FG, labelsize=8)
ax1.grid(True, alpha=0.3, axis="x")
ax1.invert_yaxis()

colors_ku = [FIRE_C if v > 0 else NEG_C for v in kurtosis.values]
ax2.barh(kurtosis.index, kurtosis.values, color=colors_ku, edgecolor=GRID, linewidth=0.4)
ax2.axvline(0, color=FG, linewidth=0.8, linestyle="--")
ax2.set_xlabel("Kurtosis (excess)", color=FG, fontsize=11)
ax2.set_title("Kurtosis", color=ACCENT, fontsize=12)
ax2.tick_params(colors=FG, labelsize=8)
ax2.grid(True, alpha=0.3, axis="x")
ax2.invert_yaxis()

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_DIR / "skewness_kurtosis.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("     Saved -> skewness_kurtosis.png")

# â”€â”€â”€ 5. FEATURE DISCRIMINATIVE POWER (Cohen's d + Mann-Whitney U) â”€
print("[05] Computing feature discriminative power ...")
results = []
for feat in key_feats:
    s_vals = pd.to_numeric(small_df[feat], errors="coerce").dropna().values
    l_vals = pd.to_numeric(large_df[feat], errors="coerce").dropna().values
    if len(s_vals) < 5 or len(l_vals) < 5:
        continue
    # Cohen's d
    pooled_std = np.sqrt((np.std(s_vals, ddof=1)**2 + np.std(l_vals, ddof=1)**2) / 2)
    cohens_d   = abs(np.mean(l_vals) - np.mean(s_vals)) / (pooled_std + 1e-10)
    # Mann-Whitney U
    try:
        _, p_val = stats.mannwhitneyu(s_vals, l_vals, alternative="two-sided")
    except Exception:
        p_val = 1.0
    results.append({"feature": feat, "cohens_d": cohens_d, "mwu_pvalue": p_val})

if results:
    disc_df = pd.DataFrame(results).sort_values("cohens_d", ascending=False)
    disc_df.to_csv(OUT_DIR / "feature_discriminative_power.csv", index=False, encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8), facecolor=BG)
    fig.suptitle(f"{STATE_NAME} -- Feature Discriminative Power (Small vs Large Fire)",
                 color=FG, fontsize=14, y=0.98)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)

    # Cohen's d bar chart
    top = disc_df.head(20)
    pal = plt.cm.plasma(np.linspace(0.2, 0.9, len(top)))
    ax1.barh(top["feature"], top["cohens_d"], color=pal, edgecolor=GRID, linewidth=0.4)
    ax1.set_xlabel("Cohen's d (effect size)", color=FG, fontsize=11)
    ax1.set_title("Cohen's d (sorted by effect size)", color=ACCENT, fontsize=12)
    ax1.tick_params(colors=FG, labelsize=9)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis="x")
    for _, row in top.iterrows():
        ax1.text(row["cohens_d"] + 0.005, row["feature"], f"{row['cohens_d']:.3f}",
                 va="center", color=FG, fontsize=7)

    # Mann-Whitney p-value (log scale)
    sig_df = disc_df[disc_df["mwu_pvalue"] < 0.05].head(20)
    neg_log_p = -np.log10(sig_df["mwu_pvalue"].clip(lower=1e-300))
    pal2 = plt.cm.plasma(np.linspace(0.2, 0.9, len(sig_df)))
    ax2.barh(sig_df["feature"], neg_log_p.values, color=pal2, edgecolor=GRID, linewidth=0.4)
    ax2.set_xlabel("-log10(p-value)  [Mann-Whitney U]", color=FG, fontsize=11)
    ax2.set_title("Statistical Significance", color=ACCENT, fontsize=12)
    ax2.tick_params(colors=FG, labelsize=9)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.axvline(1.301, color=ACCENT, linestyle="--", linewidth=1, label="p=0.05")
    ax2.legend(fontsize=9, labelcolor=FG, facecolor=BG, edgecolor=GRID)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "feature_discriminative_power.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("     Saved -> feature_discriminative_power.png")
    print("     Saved -> feature_discriminative_power.csv")

print(f"\n[05] DONE -- All outputs in {OUT_DIR}")

