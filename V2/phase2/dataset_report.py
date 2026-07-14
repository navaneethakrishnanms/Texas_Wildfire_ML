"""
dataset_report.py
------------------
Generates a full markdown report about the final training dataset.
Run from: V2/phase2/
    python dataset_report.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

OUT_DIR = Path("outputs/texas")
df = pd.read_parquet(OUT_DIR / "final_training_dataset_tx.parquet")

lines = []
A = lines.append

A("# Final Training Dataset Report — Texas")
A("")
A("---")
A("")

# ── Overview ──────────────────────────────────────────────────────────────────
A("## 1. Overview")
A("")
A(f"| Item | Value |")
A(f"|------|-------|")
A(f"| Total rows | **{len(df):,}** |")
A(f"| Total columns | **{len(df.columns)}** |")
A(f"| File | `final_training_dataset_tx.parquet` |")
A(f"| Size on disk | ~19 MB (parquet) |")
A(f"| Study period | 2014–2020 (7 years) |")
A(f"| State | Texas |")
A(f"| H3 Resolution | 7 (~1.9 km cell width) |")
A("")

# ── Label Distribution ─────────────────────────────────────────────────────────
n_fire    = int((df["label"] == 1).sum())
n_nofire  = int((df["label"] == 0).sum())
A("## 2. Label Distribution")
A("")
A("| Label | Meaning | Count | Percentage |")
A("|-------|---------|-------|------------|")
A(f"| 1 | Fire discovered | **{n_fire:,}** | {100*n_fire/len(df):.1f}% |")
A(f"| 0 | No fire (negative sample) | **{n_nofire:,}** | {100*n_nofire/len(df):.1f}% |")
A(f"| **Total** | | **{len(df):,}** | 100% |")
A("")

# ── Chronological Split ───────────────────────────────────────────────────────
A("## 3. Chronological Train / Val / Test Split")
A("")
A("| Split | Years | Total Rows | Fire Rows | Non-fire Rows | Fire Rate |")
A("|-------|-------|-----------|---------|-------------|-----------|")
splits = [
    ("TRAIN", [2014,2015,2016,2017], "2014–2017"),
    ("VAL",   [2018],                "2018"),
    ("TEST",  [2019,2020],           "2019–2020"),
]
for name, yrs, yr_str in splits:
    sub = df[df["fire_year"].isin(yrs)]
    n1  = int((sub["label"] == 1).sum())
    n0  = int((sub["label"] == 0).sum())
    A(f"| {name} | {yr_str} | {len(sub):,} | {n1:,} | {n0:,} | {100*n1/len(sub):.1f}% |")
A("")

# ── All Columns ───────────────────────────────────────────────────────────────
A("## 4. All Columns")
A("")

GROUPS = {
    "🆔 Identifiers (not used as features)": [
        "h3_cell","date_utc","window_hour","window_6h_utc","fire_year","gridmet_missing"
    ],
    "🎯 Target": ["label"],
    "🌿 Landscape / Static (LANDFIRE + FSim)": [
        "avg_burn_prob","whp","flep4","cfl",
        "fire_count","has_fire_history","burnable"
    ],
    "🌦️ gridMET Daily Weather": [
        "erc","fm100","vpd","vs","rmax","rmin","tmmx","pr"
    ],
    "📅 5-Day Trailing Stats (gridMET)": [
        "erc_5D_mean","erc_5D_max",
        "fm100_5D_mean","fm100_5D_min",
        "vpd_5D_mean","vpd_5D_max",
        "vs_5D_mean","vs_5D_max",
        "rmax_5D_mean","rmax_5D_min",
        "tmmx_5D_mean","tmmx_5D_max",
    ],
    "⏱️ Temporal Encodings": [
        "sin_month","cos_month","sin_hour","cos_hour"
    ],
    "📍 Location": ["centroid_lat","centroid_lon"],
}

DESCRIPTIONS = {
    "h3_cell":           "H3 hexagonal cell ID (resolution-7)",
    "date_utc":          "UTC date of the 6-hour window",
    "window_hour":       "UTC hour (0, 6, 12, or 18)",
    "window_6h_utc":     "Full UTC timestamp string",
    "fire_year":         "Year of the event (used for split)",
    "gridmet_missing":   "1 = this cell has no gridMET data (coastal/border)",
    "label":             "1 = fire discovered, 0 = no fire",
    "avg_burn_prob":     "FSim annual burn probability [0–1] (⚠️ = 0, rasters missing)",
    "whp":               "Wildfire Hazard Potential index [0–7000] (⚠️ = 0, rasters missing)",
    "flep4":             "Flame Length Exceedance Prob ≥4ft [0–1] (⚠️ = 0, rasters missing)",
    "cfl":               "Canopy Fuel Load [Mg ha⁻¹] (⚠️ = 0, rasters missing)",
    "fire_count":        "Historical fire count in this H3 cell (2014–2020)",
    "has_fire_history":  "1 if cell has ever burned before",
    "burnable":          "1 = burnable land cover, 0 = non-burnable",
    "erc":               "Energy Release Component [BTU ft⁻²] — top daily predictor",
    "fm100":             "100-hr dead fuel moisture [%]",
    "vpd":               "Vapor pressure deficit [kPa] — from gridMET (not FPA-FOD, no leakage)",
    "vs":                "Wind speed [m s⁻¹]",
    "rmax":              "Max relative humidity [%]",
    "rmin":              "Min relative humidity [%]",
    "tmmx":              "Max temperature [°C]",
    "pr":                "Precipitation [mm]",
    "erc_5D_mean":       "5-day trailing mean of ERC (days D-5 to D-1)",
    "erc_5D_max":        "5-day trailing max of ERC",
    "fm100_5D_mean":     "5-day trailing mean of 100-hr fuel moisture",
    "fm100_5D_min":      "5-day trailing min of fuel moisture (lower = drier)",
    "vpd_5D_mean":       "5-day trailing mean of VPD",
    "vpd_5D_max":        "5-day trailing max of VPD",
    "vs_5D_mean":        "5-day trailing mean of wind speed",
    "vs_5D_max":         "5-day trailing max of wind speed",
    "rmax_5D_mean":      "5-day trailing mean of max relative humidity",
    "rmax_5D_min":       "5-day trailing min of max relative humidity",
    "tmmx_5D_mean":      "5-day trailing mean of max temperature",
    "tmmx_5D_max":       "5-day trailing max of max temperature",
    "sin_month":         "sin(2π × month / 12) — cyclic fire season encoding",
    "cos_month":         "cos(2π × month / 12) — cyclic fire season encoding",
    "sin_hour":          "sin(2π × window_hour / 24) — cyclic time-of-day",
    "cos_hour":          "cos(2π × window_hour / 24) — cyclic time-of-day",
    "centroid_lat":      "H3 cell centroid latitude (WGS84)",
    "centroid_lon":      "H3 cell centroid longitude (WGS84)",
}

for group_name, cols in GROUPS.items():
    present = [c for c in cols if c in df.columns]
    A(f"### {group_name}  ({len(present)} columns)")
    A("")
    A("| Column | Type | Non-null | NaN Count | NaN % | Min | Max | Mean | Description |")
    A("|--------|------|---------|-----------|-------|-----|-----|------|-------------|")
    for c in present:
        s       = df[c]
        n_nan   = int(s.isna().sum())
        pct_nan = 100 * n_nan / len(df)
        n_valid = len(df) - n_nan
        dtype   = str(s.dtype)
        desc    = DESCRIPTIONS.get(c, "—")
        if pd.api.types.is_numeric_dtype(s) and n_valid > 0:
            mn  = f"{s.min():.3f}" if not pd.isna(s.min()) else "—"
            mx  = f"{s.max():.3f}" if not pd.isna(s.max()) else "—"
            avg = f"{s.mean():.3f}" if not pd.isna(s.mean()) else "—"
        else:
            mn = mx = avg = "—"
        nan_str  = f"{n_nan:,} ({pct_nan:.1f}%)" if n_nan > 0 else "0 ✔"
        A(f"| `{c}` | {dtype} | {n_valid:,} | {nan_str} | {pct_nan:.1f}% | {mn} | {mx} | {avg} | {desc} |")
    A("")

# ── Missing Value Summary ─────────────────────────────────────────────────────
A("## 5. Missing Value Summary")
A("")
A("### Root Causes")
A("")
A("| Root Cause | Columns Affected | NaN Count | NaN % | Action |")
A("|-----------|-----------------|-----------|-------|--------|")
A("| LANDFIRE rasters not downloaded | `avg_burn_prob`, `whp`, `flep4`, `cfl` | 1,594 | 0.42% | Fill with 0 until rasters downloaded |")
A("| H3 cells not in static grid | `fire_count`, `has_fire_history`, `burnable` | 1,594 | 0.42% | Fill with 0 |")
A("| gridMET ocean/border pixels | All 20 gridMET cols | 24,954 | 6.63% | XGBoost handles natively (NaN = coastal/border cell) |")
A("| 5D trailing stats (same root) | All 12 5D stat cols | 24,965 | 6.64% | Same as above (11 extra = Jan 1–5 boundary) |")
A("")
A("### Missing Values by Column")
A("")
A("| Column | NaN Count | NaN % | Status |")
A("|--------|-----------|-------|--------|")
for c in df.columns:
    n   = int(df[c].isna().sum())
    pct = 100 * n / len(df)
    if n == 0:
        status = "✅ Complete"
    elif pct < 1:
        status = "🟡 Minor — fill with 0"
    elif pct < 7:
        status = "🟠 Coastal/border cells — XGBoost handles"
    else:
        status = "🔴 Check"
    A(f"| `{c}` | {n:,} | {pct:.2f}% | {status} |")
A("")

# ── Dropped Columns ────────────────────────────────────────────────────────────
A("## 6. Columns Dropped (per Team Review)")
A("")
A("| Column | Reason |")
A("|--------|--------|")
dropped = {
    "bi":          "Correlated with `erc` (both NFDRS indices) — keep `erc`",
    "bi_5D_mean":  "`bi` was dropped, so its 5D stats also dropped",
    "bi_5D_max":   "`bi` was dropped, so its 5D stats also dropped",
    "tmmn":        "Correlated with `tmmx` — keep max temperature",
    "fm1000":      "Correlated with `fm100` — keep 100-hr moisture",
    "sph":         "Overlaps with `vpd` and `rmax`/`rmin`",
    "ecoregion_l2": "Geography captured by `centroid_lat`/`lon`",
    "ecoregion_l3": "Geography captured by `centroid_lat`/`lon`",
    "h3_resolution": "Constant value — zero information",
    "state_x":     "Duplicate/constant column",
    "state_y":     "Duplicate/constant column",
}
for col, reason in dropped.items():
    A(f"| `{col}` | {reason} |")
A("")

# ── Weather Stats ─────────────────────────────────────────────────────────────
A("## 7. Weather Feature Statistics")
A("")
wcols = ["erc","fm100","vpd","vs","rmax","rmin","tmmx","pr",
         "erc_5D_mean","fm100_5D_mean","vpd_5D_mean","tmmx_5D_mean"]
wcols = [c for c in wcols if c in df.columns]
desc  = df[wcols].describe().round(3)
A("| Stat | " + " | ".join(wcols) + " |")
A("|------|" + "|".join(["---"]*len(wcols)) + "|")
for stat in ["count","mean","std","min","25%","50%","75%","max"]:
    vals = [str(desc.loc[stat, c]) for c in wcols]
    A(f"| {stat} | " + " | ".join(vals) + " |")
A("")

# ── Next Steps ─────────────────────────────────────────────────────────────────
A("## 8. Status and Next Steps")
A("")
A("| Step | Status | Action |")
A("|------|--------|--------|")
A("| Phase 2F gridMET extraction | ✅ Complete | `gridmet_features_tx.parquet` (15 MB) |")
A("| Phase 2G dataset assembly | ✅ Complete | `final_training_dataset_tx.parquet` (19 MB) |")
A("| LANDFIRE rasters (4 files) | ⚠️ Missing | Download from doi.org/10.2737/RDS-2020-0016-2 and RDS-2015-0047-4 |")
A("| Phase 3 XGBoost baseline | 🔜 Ready to run | `python run_phase3_train.py --state TX` |")
A("| Phase 3 full model (+ LANDFIRE) | 🔜 After rasters | Re-run Phase 2E → 2G → 3 |")
A("")
A("---")
A("*Report generated automatically from `final_training_dataset_tx.parquet`*")

# ── Write report ──────────────────────────────────────────────────────────────
report = "\n".join(lines)
out    = OUT_DIR / "dataset_report_tx.md"
out.write_text(report, encoding="utf-8")
print(f"\n  ✔ Report saved: {out}")
print(f"  {len(df):,} rows  x  {len(df.columns)} cols")
print(f"  Total missing cells: {df.isna().sum().sum():,} out of {df.size:,} "
      f"({100*df.isna().sum().sum()/df.size:.2f}%)")
