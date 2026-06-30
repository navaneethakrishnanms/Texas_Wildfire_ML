"""
06_summary_report.py -- Texas Wildfire Executive Dashboard & Text Report
=========================================================================
Outputs:
  executive_dashboard.png  (28x20 inch composite figure)
  eda_summary_report.txt   (comprehensive analyst text report)
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
KEY_FEATS     = WEATHER_FEATS + ["Elevation", "Slope", "TRI", "NDVI_mean"]
MONTH_NAMES   = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# â”€â”€â”€ LOAD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"[06] Loading {STATE_NAME} dataset ...")
OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(DATA_PATH)
df["large_fire"] = (df["FIRE_SIZE"] >= 100).astype(int)
df["disc_date"]  = pd.to_datetime(df.get("DISCOVERY_DATE", None), errors="coerce")
df["disc_month"] = df["disc_date"].dt.month
df["disc_year"]  = df["disc_date"].dt.year.fillna(df.get("FIRE_YEAR", np.nan))

small_df = df[df["large_fire"] == 0]
large_df = df[df["large_fire"] == 1]

# â”€â”€â”€ COMPUTE COHEN'S D FOR KEY FEATURES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
disc_results = []
for feat in KEY_FEATS:
    if feat not in df.columns:
        continue
    if not pd.api.types.is_numeric_dtype(df[feat]):
        continue
    s = pd.to_numeric(small_df[feat], errors="coerce").dropna().values
    l = pd.to_numeric(large_df[feat], errors="coerce").dropna().values
    if len(s) < 5 or len(l) < 5:
        continue
    pooled = np.sqrt((np.std(s, ddof=1)**2 + np.std(l, ddof=1)**2) / 2)
    d      = abs(np.mean(l) - np.mean(s)) / (pooled + 1e-10)
    disc_results.append({"feature": feat, "cohens_d": d})
disc_df = pd.DataFrame(disc_results).sort_values("cohens_d", ascending=False) if disc_results else pd.DataFrame(columns=["feature", "cohens_d"])

# â”€â”€â”€ EXECUTIVE DASHBOARD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[06] Building executive_dashboard.png ...")

fig = plt.figure(figsize=(28, 20), facecolor=BG)
fig.suptitle(f"{STATE_NAME} Wildfire Dataset -- Executive EDA Dashboard",
             color=FG, fontsize=20, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.40,
                       left=0.05, right=0.97, top=0.94, bottom=0.04)

# â”€â”€ Row 1: KPI CARDS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
kpi_data = [
    ("Total Records",     f"{len(df):,}",                    FIRE_C),
    ("Large Fires",       f"{df['large_fire'].sum():,}",     ACCENT),
    ("Features",          f"{df.shape[1]}",                  NEG_C),
    ("Missing Data %",    f"{df.isnull().mean().mean()*100:.1f}%", "#9B59B6"),
]
for col, (label, value, color) in enumerate(kpi_data):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor(color + "22")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    rect = plt.Rectangle((0.05, 0.05), 0.90, 0.90, transform=ax.transAxes,
                          color=color, alpha=0.15, linewidth=2,
                          edgecolor=color, fill=True)
    ax.add_patch(rect)
    ax.text(0.5, 0.62, value, transform=ax.transAxes, ha="center", va="center",
            fontsize=28, color=color, fontweight="bold")
    ax.text(0.5, 0.28, label, transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=FG)

# â”€â”€ Row 2: Cohen's d bar + Monthly distribution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax_cd = fig.add_subplot(gs[1, :2])
ax_cd.set_facecolor(BG)
if not disc_df.empty:
    top_d   = disc_df.head(10)
    pal_d   = plt.cm.plasma(np.linspace(0.2, 0.9, len(top_d)))
    bars_d  = ax_cd.barh(top_d["feature"], top_d["cohens_d"], color=pal_d, edgecolor=GRID)
    ax_cd.invert_yaxis()
    ax_cd.set_xlabel("Cohen's d (effect size)", color=FG, fontsize=10)
    ax_cd.set_title("Top Features: Discriminative Power (Cohen's d)", color=ACCENT, fontsize=11)
    ax_cd.tick_params(colors=FG, labelsize=8)
    ax_cd.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars_d, top_d["cohens_d"]):
        ax_cd.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                   f"{val:.3f}", va="center", color=FG, fontsize=7)

ax_mo = fig.add_subplot(gs[1, 2:])
ax_mo.set_facecolor(BG)
if "disc_month" in df.columns:
    mo_cnt = df["disc_month"].value_counts().sort_index()
    mo_lbl = [MONTH_NAMES.get(int(m), str(m)) for m in mo_cnt.index]
    pal_mo = plt.cm.plasma(np.linspace(0.2, 0.9, len(mo_cnt)))
    bars_m = ax_mo.bar(mo_lbl, mo_cnt.values, color=pal_mo, edgecolor=GRID, linewidth=0.5)
    ax_mo.set_xlabel("Month", color=FG, fontsize=10)
    ax_mo.set_ylabel("Fire Count", color=FG, fontsize=10)
    ax_mo.set_title("Monthly Fire Distribution", color=ACCENT, fontsize=11)
    ax_mo.tick_params(colors=FG, labelsize=9)
    ax_mo.grid(axis="y", alpha=0.3)

# â”€â”€ Row 3: Compact correlation heatmap + Geographic hexbin â”€â”€â”€â”€â”€â”€â”€â”€
import seaborn as sns
ax_corr = fig.add_subplot(gs[2, :2])
ax_corr.set_facecolor(BG)
key_present = [f for f in KEY_FEATS
               if f in df.columns and pd.api.types.is_numeric_dtype(df[f])][:10]
if len(key_present) >= 2:
    corr_mini = df[key_present].corr()
    mask_m    = np.triu(np.ones_like(corr_mini, dtype=bool))
    sns.heatmap(corr_mini, mask=mask_m, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, linewidths=0.3, annot_kws={"size": 7},
                cbar_kws={"shrink": 0.6}, ax=ax_corr)
    ax_corr.set_title("Compact Correlation Heatmap (Key Features)", color=ACCENT, fontsize=11)
    ax_corr.tick_params(colors=FG, labelsize=8)
    plt.setp(ax_corr.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax_corr.get_yticklabels(), rotation=0)

ax_geo = fig.add_subplot(gs[2, 2:])
ax_geo.set_facecolor(BG)
geo = df.dropna(subset=["LATITUDE", "LONGITUDE"])
if not geo.empty:
    hb = ax_geo.hexbin(geo["LONGITUDE"], geo["LATITUDE"], gridsize=40, cmap="YlOrRd",
                       mincnt=1, linewidths=0.1)
    cb = plt.colorbar(hb, ax=ax_geo, shrink=0.5)
    cb.set_label("Fire Count", color=FG, fontsize=8)
    cb.ax.tick_params(colors=FG, labelsize=7)
    ax_geo.set_xlabel("Longitude", color=FG, fontsize=10)
    ax_geo.set_ylabel("Latitude", color=FG, fontsize=10)
    ax_geo.set_title("Geographic Fire Density", color=ACCENT, fontsize=11)
    ax_geo.tick_params(colors=FG, labelsize=8)

fig.savefig(OUT_DIR / "executive_dashboard.png", dpi=DPI, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("     Saved -> executive_dashboard.png")

# â”€â”€â”€ TEXT REPORT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[06] Writing eda_summary_report.txt ...")

null_counts = df.isnull().sum()
null_pct    = null_counts / len(df) * 100
top_missing = null_pct[null_pct > 0].sort_values(ascending=False).head(10)
top_features_disc = disc_df.head(5)["feature"].tolist() if not disc_df.empty else []

yr_cnt  = df["FIRE_YEAR"].value_counts().sort_index() if "FIRE_YEAR" in df.columns else pd.Series()
mo_cnt2 = df["disc_month"].value_counts().sort_index() if "disc_month" in df.columns else pd.Series()
peak_mo = MONTH_NAMES.get(int(mo_cnt2.idxmax()), "N/A") if not mo_cnt2.empty else "N/A"
peak_yr = int(yr_cnt.idxmax()) if not yr_cnt.empty else "N/A"

cause_col = "NWCG_GENERAL_CAUSE"
cause_vc  = df[cause_col].value_counts() if cause_col in df.columns else pd.Series()

lines = []
lines.append("=" * 72)
lines.append(f"  {STATE_NAME} Wildfire Dataset -- EDA Summary Report")
lines.append("=" * 72)
lines.append("")

lines.append("--- 1. DATASET OVERVIEW ---")
lines.append(f"  State           : {STATE_NAME} ({STATE_CODE})")
lines.append(f"  Total records   : {len(df):,}")
lines.append(f"  Columns         : {df.shape[1]}")
lines.append(f"  Years covered   : {sorted(df['FIRE_YEAR'].dropna().unique().astype(int).tolist())}")
lines.append(f"  Numeric cols    : {len(df.select_dtypes(include=[np.number]).columns)}")
lines.append(f"  Categorical cols: {len(df.select_dtypes(include='object').columns)}")
lines.append(f"  Memory usage    : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
lines.append("")

lines.append("--- 2. DATA QUALITY ---")
lines.append(f"  Exact duplicates     : {df.duplicated().sum():,}")
lines.append(f"  Columns with nulls   : {(null_pct > 0).sum()} / {df.shape[1]}")
lines.append(f"  Overall missing rate : {null_pct.mean():.1f}%")
lines.append("  Top 10 most missing columns:")
for col, pct in top_missing.items():
    lines.append(f"    {col:<40}  {pct:>5.1f}% missing")
lines.append("")

lines.append("--- 3. FIRE SIZE SUMMARY ---")
fs = df["FIRE_SIZE"].dropna()
lines.append(f"  Count    : {len(fs):,}")
lines.append(f"  Mean     : {fs.mean():.2f} acres")
lines.append(f"  Median   : {fs.median():.2f} acres")
lines.append(f"  Std Dev  : {fs.std():.2f} acres")
lines.append(f"  Max      : {fs.max():.2f} acres")
lines.append(f"  Small (<100ac) : {(df['large_fire']==0).sum():,}  ({(df['large_fire']==0).mean()*100:.1f}%)")
lines.append(f"  Large (>=100ac): {df['large_fire'].sum():,}  ({df['large_fire'].mean()*100:.1f}%)")
lines.append("")

lines.append("--- 4. KEY FINDINGS ---")
lines.append(f"  Top discriminating features (Cohen's d, Large vs Small fire):")
for _, row in disc_df.head(8).iterrows():
    lines.append(f"    {row['feature']:<30}  d = {row['cohens_d']:.3f}")
lines.append("")

lines.append("--- 5. TEMPORAL PATTERNS ---")
lines.append(f"  Peak year  : {peak_yr}  ({yr_cnt.max() if not yr_cnt.empty else 'N/A':,} fires)")
lines.append(f"  Peak month : {peak_mo}  ({mo_cnt2.max() if not mo_cnt2.empty else 'N/A':,} fires)")
lines.append("  Fires per year:")
for yr, cnt in yr_cnt.items():
    lines.append(f"    {int(yr)}  :  {cnt:>6,} fires")
lines.append("")

lines.append("--- 6. GEOGRAPHIC PATTERNS ---")
if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
    geo = df.dropna(subset=["LATITUDE", "LONGITUDE"])
    lines.append(f"  Records with lat/lon : {len(geo):,}")
    lines.append(f"  Latitude  range : {geo['LATITUDE'].min():.3f}  to  {geo['LATITUDE'].max():.3f}")
    lines.append(f"  Longitude range : {geo['LONGITUDE'].min():.3f}  to  {geo['LONGITUDE'].max():.3f}")
lines.append("")

lines.append("--- 7. CAUSE BREAKDOWN ---")
for cause, cnt in cause_vc.head(10).items():
    lines.append(f"  {str(cause):<35}  {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")
lines.append("")

lines.append("--- 8. MODELLING RECOMMENDATIONS ---")
lines.append("  [Class Imbalance]")
lines.append(f"    Large vs Small ratio: 1 : {(df['large_fire']==0).sum() // max(df['large_fire'].sum(),1)}")
lines.append("    Use SMOTE, class_weight='balanced', or threshold tuning.")
lines.append("")
lines.append("  [Multicollinearity]")
lines.append("    ERC, BI, VPD, TMMX are highly correlated.")
lines.append("    Consider PCA or dropping correlated pairs before tree-based models.")
lines.append("")
lines.append("  [Leakage Columns]")
lines.append("    CONT_DATE, CONT_DOY, CONT_TIME, MTBS_ID describe post-fire outcomes.")
lines.append("    Exclude from all predictive models.")
lines.append("")
lines.append("  [Feature Engineering Ideas]")
lines.append("    - Interaction: erc * vs (fire weather index proxy)")
lines.append("    - Rolling means: 7-day / 30-day weather windows")
lines.append("    - NDVI anomaly relative to seasonal normal")
lines.append("    - County-level historical fire frequency")
lines.append("")
lines.append("  [Train/Test Split]")
lines.append("    Use temporal split: train on 2014-2018, validate on 2019, test on 2020.")
lines.append("    Avoid random split -- spatial autocorrelation causes leakage.")
lines.append("")
lines.append("=" * 72)
lines.append("  END OF REPORT")
lines.append("=" * 72)

(OUT_DIR / "eda_summary_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("     Saved -> eda_summary_report.txt")

print(f"\n[06] DONE -- All outputs in {OUT_DIR}")

