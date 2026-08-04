"""
generate_dataset_report.py
---------------------------
Profiles data/hrrr/hrrr_tx_all.parquet (the merged gridMET+landscape+HRRR
training dataset) and writes a markdown report: shape, missingness,
label balance by year, HRRR coverage, per-feature stats, and fire vs
no-fire feature comparison (signal check).

Usage:
  python generate_dataset_report.py
  python generate_dataset_report.py --input data/hrrr/hrrr_tx_all.parquet --output outputs/texas_landfire/DATASET_REPORT.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

NUMERIC_FEATURES = [
    "avg_burn_prob", "whp", "flep4", "cfl", "cbd", "cbh",
    "erc", "fm100", "vpd", "vs", "rmax", "rmin", "tmmx", "pr",
    "erc_5D_mean", "erc_5D_max", "fm100_5D_mean", "fm100_5D_min",
    "vpd_5D_mean", "vpd_5D_max", "vs_5D_mean", "vs_5D_max",
    "rmax_5D_mean", "rmax_5D_min", "tmmx_5D_mean", "tmmx_5D_max",
    "temp_pw", "rh_pw", "wind_pw", "vpd_pw_hrrr", "hpbl_pw", "dswrf_pw",
]

FLAG_COLS = ["burnable", "gridmet_missing", "hrrr_pw", "hrrr_rh_valid"]


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/hrrr/hrrr_tx_all.parquet")
    parser.add_argument("--output", type=str, default="outputs/texas_landfire/DATASET_REPORT.md")
    args = parser.parse_args()

    in_path = ROOT / args.input
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    df["date_utc"] = pd.to_datetime(df["date_utc"])
    df["yr"] = df["date_utc"].dt.year

    lines = []
    lines.append(f"# Dataset Report — {in_path.name}\n")
    lines.append(f"Generated from `{args.input}`\n")

    # ── Overview ──────────────────────────────────────────────────────────
    lines.append("## Overview\n")
    lines.append(md_table(
        ["Property", "Value"],
        [
            ["Rows", f"{len(df):,}"],
            ["Columns", len(df.columns)],
            ["Unique H3 cells", f"{df['h3_cell'].nunique():,}"],
            ["Date range", f"{df['date_utc'].min().date()} to {df['date_utc'].max().date()}"],
            ["Fire rows (label=1)", f"{(df['label']==1).sum():,} ({100*(df['label']==1).mean():.2f}%)"],
            ["Non-fire rows", f"{(df['label']==0).sum():,} ({100*(df['label']==0).mean():.2f}%)"],
        ]
    ))
    lines.append("")

    # ── Label + HRRR coverage by year ────────────────────────────────────
    lines.append("## By Year — Label Rate & HRRR Coverage\n")
    rows = []
    for yr, g in df.groupby("yr"):
        n = len(g)
        fire = int((g["label"] == 1).sum())
        hrrr = int((g["hrrr_pw"] == 1).sum())
        rh_ok = int((g["hrrr_rh_valid"] == 1).sum())
        rows.append([
            yr, f"{n:,}", f"{fire:,} ({100*fire/n:.1f}%)",
            f"{hrrr:,} ({100*hrrr/n:.1f}%)",
            f"{rh_ok:,} ({100*rh_ok/n:.1f}%)",
        ])
    lines.append(md_table(
        ["Year", "Rows", "Fire rows", "HRRR coverage", "HRRR RH/VPD valid"], rows
    ))
    lines.append("")

    # ── Split distribution ────────────────────────────────────────────────
    if "_split" in df.columns:
        lines.append("## Chronological Split\n")
        rows = []
        for split, g in df.groupby("_split"):
            n = len(g)
            fire = int((g["label"] == 1).sum())
            rows.append([split, f"{n:,}", f"{fire:,} ({100*fire/n:.1f}%)"])
        lines.append(md_table(["Split", "Rows", "Fire rows"], rows))
        lines.append("")

    # ── Missingness ────────────────────────────────────────────────────────
    lines.append("## Missingness (NaN counts)\n")
    na = df.isna().sum()
    na = na[na > 0].sort_values(ascending=False)
    rows = [[c, f"{n:,}", f"{100*n/len(df):.1f}%"] for c, n in na.items()]
    if rows:
        lines.append(md_table(["Column", "NaN count", "NaN %"], rows))
    else:
        lines.append("No missing values.")
    lines.append("")

    # ── Flags ────────────────────────────────────────────────────────────
    lines.append("## Binary Flags\n")
    rows = []
    for col in FLAG_COLS:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False).to_dict()
            rows.append([col, str(vc)])
    lines.append(md_table(["Flag", "Value counts"], rows))
    lines.append("")

    # ── Numeric feature stats ─────────────────────────────────────────────
    lines.append("## Numeric Feature Statistics\n")
    rows = []
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        s = df[col]
        rows.append([
            col, f"{s.mean():.3f}", f"{s.std():.3f}", f"{s.min():.3f}",
            f"{s.median():.3f}", f"{s.max():.3f}", f"{s.isna().sum():,}",
        ])
    lines.append(md_table(["Feature", "Mean", "Std", "Min", "Median", "Max", "NaN"], rows))
    lines.append("")

    # ── Fire vs no-fire signal check ──────────────────────────────────────
    lines.append("## Fire vs No-Fire — Feature Means (signal check)\n")
    fire = df[df["label"] == 1]
    nofire = df[df["label"] == 0]
    rows = []
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        fm, nm = fire[col].mean(), nofire[col].mean()
        if pd.isna(fm) or pd.isna(nm) or nm == 0:
            ratio = "n/a"
        else:
            ratio = f"{fm/nm:.2f}x"
        rows.append([col, f"{fm:.3f}", f"{nm:.3f}", ratio])
    lines.append(md_table(["Feature", "Fire mean", "No-fire mean", "Ratio"], rows))
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {out_path}")
    print(f"  {len(df):,} rows, {len(df.columns)} columns profiled")


if __name__ == "__main__":
    main()
