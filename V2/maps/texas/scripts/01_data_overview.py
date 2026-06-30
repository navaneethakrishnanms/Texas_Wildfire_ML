"""
01_data_overview.py -- Texas Wildfire Dataset Overview
=======================================================
Outputs:
  data_overview.txt
  summary_statistics.csv
  missing_values_report.csv
  missing_values_heatmap.png
  fire_size_class_distribution.png
  data_types_chart.png
"""

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─── CONFIG ──────────────────────────────────────────────────────
STATE_CODE = "TX"
STATE_NAME = "Texas"
BASE_DIR   = Path(__file__).resolve().parents[3]          # -> V2/
DATA_PATH  = BASE_DIR / "data" / "processed" / "texas" / "texas_fire_2014_2020.parquet"
OUT_DIR    = BASE_DIR / "maps" / "texas" / "eda_outputs"

# ─── STYLE ───────────────────────────────────────────────────────
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
print(f"[01] Loading {STATE_NAME} dataset from {DATA_PATH.name} ...")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)
df["large_fire"] = (df["FIRE_SIZE"] >= 100).astype(int)

print(f"     Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"     Large fires (>=100 ac): {df['large_fire'].sum():,}")
print(f"     Small fires (<100 ac):  {(df['large_fire']==0).sum():,}")

# ─── 1. DATA OVERVIEW TXT ────────────────────────────────────────
print("[01] Writing data_overview.txt ...")
lines = []
lines.append("=" * 72)
lines.append(f"  {STATE_NAME} Wildfire Dataset -- Data Overview")
lines.append("=" * 72)
lines.append(f"  Rows           : {df.shape[0]:,}")
lines.append(f"  Columns        : {df.shape[1]}")
lines.append(f"  Large fires    : {df['large_fire'].sum():,}  (FIRE_SIZE >= 100 acres)")
lines.append(f"  Small fires    : {(df['large_fire']==0).sum():,}  (FIRE_SIZE < 100 acres)")
lines.append(f"  Years          : {sorted(df['FIRE_YEAR'].dropna().unique().astype(int).tolist())}")
lines.append("")
lines.append(f"  {'Column':<45}  {'Dtype':<12}  {'Unique':>8}  {'Nulls':>8}  {'Null%':>7}")
lines.append(f"  {'-'*45}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*7}")
for col in df.columns:
    d   = str(df[col].dtype)
    u   = int(df[col].nunique(dropna=False))
    n   = int(df[col].isnull().sum())
    pct = n / len(df) * 100
    lines.append(f"  {col:<45}  {d:<12}  {u:>8,}  {n:>8,}  {pct:>6.1f}%")
Path(OUT_DIR / "data_overview.txt").write_text("\n".join(lines), encoding="utf-8")
print("     Saved -> data_overview.txt")

# ─── 2. SUMMARY STATISTICS CSV ───────────────────────────────────
print("[01] Writing summary_statistics.csv ...")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_cols].describe(percentiles=[.05, .25, .5, .75, .95]).T.to_csv(
    OUT_DIR / "summary_statistics.csv", encoding="utf-8"
)
print("     Saved -> summary_statistics.csv")

# ─── 3. MISSING VALUES REPORT CSV ────────────────────────────────
print("[01] Writing missing_values_report.csv ...")
mv = pd.DataFrame({
    "column":       df.columns,
    "dtype":        [str(df[c].dtype) for c in df.columns],
    "null_count":   [int(df[c].isnull().sum()) for c in df.columns],
    "null_pct":     [round(df[c].isnull().sum() / len(df) * 100, 2) for c in df.columns],
    "unique_count": [int(df[c].nunique(dropna=False)) for c in df.columns],
}).sort_values("null_pct", ascending=False)
mv.to_csv(OUT_DIR / "missing_values_report.csv", index=False, encoding="utf-8")
print("     Saved -> missing_values_report.csv")

# ─── 4. MISSING VALUES HEATMAP (horizontal bar, red gradient) ────
print("[01] Plotting missing_values_heatmap.png ...")
mv_nz = mv[mv["null_count"] > 0].head(50).copy()
if not mv_nz.empty:
    fig_h = max(8, len(mv_nz) * 0.30)
    fig, ax = plt.subplots(figsize=(18, fig_h), facecolor=BG)
    ax.set_facecolor(BG)
    norm_vals = mv_nz["null_pct"] / mv_nz["null_pct"].max()
    colors    = plt.cm.Reds(norm_vals.values * 0.8 + 0.2)
    bars = ax.barh(mv_nz["column"].values, mv_nz["null_pct"].values, color=colors, edgecolor=GRID, linewidth=0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Missing Values (%)", color=FG, fontsize=12)
    ax.set_title(f"{STATE_NAME} -- Missing Values by Column (top 50)", color=FG, fontsize=14, pad=15)
    ax.tick_params(colors=FG, labelsize=8)
    for bar, pct in zip(bars, mv_nz["null_pct"].values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", ha="left", color=FG, fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "missing_values_heatmap.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("     Saved -> missing_values_heatmap.png")

# ─── 5. FIRE SIZE CLASS DISTRIBUTION ─────────────────────────────
print("[01] Plotting fire_size_class_distribution.png ...")
if "FIRE_SIZE_CLASS" in df.columns:
    class_order  = ["A", "B", "C", "D", "E", "F", "G"]
    class_labels = {"A": "A (<0.25ac)", "B": "B (0.25-10ac)", "C": "C (10-100ac)",
                    "D": "D (100-300ac)", "E": "E (300-1k ac)", "F": "F (1k-5k ac)", "G": "G (>5k ac)"}
    vc = df["FIRE_SIZE_CLASS"].value_counts().reindex(
        [c for c in class_order if c in df["FIRE_SIZE_CLASS"].values]
    ).dropna()
    palette = [plt.cm.plasma(i / max(len(vc)-1, 1)) for i in range(len(vc))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor=BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
    # Pie
    wedges, texts, autotexts = ax1.pie(
        vc.values, labels=[class_labels.get(k, k) for k in vc.index],
        autopct="%1.1f%%", colors=palette, textprops={"color": FG, "fontsize": 10},
    )
    for at in autotexts:
        at.set_color(BG); at.set_fontsize(9)
    ax1.set_title(f"{STATE_NAME} -- Fire Size Class Distribution", color=FG, fontsize=13)
    # Bar
    bars = ax2.bar([class_labels.get(k, k) for k in vc.index], vc.values,
                   color=palette, edgecolor=GRID, linewidth=0.5)
    ax2.set_xlabel("Fire Size Class", color=FG, fontsize=11)
    ax2.set_ylabel("Number of Fires", color=FG, fontsize=11)
    ax2.set_title(f"{STATE_NAME} -- Fire Count by Size Class", color=FG, fontsize=13)
    ax2.tick_params(colors=FG)
    ax2.grid(axis="y", color=GRID, alpha=0.5)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f"{int(bar.get_height()):,}", ha="center", color=FG, fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fire_size_class_distribution.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("     Saved -> fire_size_class_distribution.png")

# ─── 6. DATA TYPES CHART ─────────────────────────────────────────
print("[01] Plotting data_types_chart.png ...")
dtype_counts = df.dtypes.value_counts()
dtype_labels = [str(d) for d in dtype_counts.index]
type_palette = [FIRE_C, ACCENT, NEG_C, "#9B59B6", "#2ECC71"][:len(dtype_counts)]
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
ax.set_facecolor(BG)
bars = ax.bar(dtype_labels, dtype_counts.values, color=type_palette, edgecolor=GRID)
ax.set_xlabel("Data Type", color=FG, fontsize=12)
ax.set_ylabel("Column Count", color=FG, fontsize=12)
ax.set_title(f"{STATE_NAME} -- Column Data Types", color=FG, fontsize=14)
ax.tick_params(colors=FG)
ax.grid(axis="y", color=GRID, alpha=0.5)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(int(bar.get_height())), ha="center", color=FG, fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "data_types_chart.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("     Saved -> data_types_chart.png")

print(f"\n[01] DONE -- All outputs in {OUT_DIR}")
