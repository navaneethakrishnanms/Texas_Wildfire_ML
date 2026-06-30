"""
eda_reporter.py
---------------
Exploratory Data Analysis (EDA) report generator.

Produces
--------
- reports/<state>/eda_report.md
- reports/<state>/correlation.csv

Public API
----------
generate_eda_report(df, state_code, state_name, report_dir)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_HIGH_CARD,
    COL_AGENCY,
    COL_CAUSE,
    COL_COUNTY,
    COL_DISCOVERY_DATE,
    COL_FIRE_SIZE,
    COL_FIRE_YEAR,
    CORR_HIGH_THRESHOLD,
    LEAKAGE_KEYWORDS,
    TOP_N_AGENCIES,
    TOP_N_COUNTIES,
)
from .logger import get_logger

log = get_logger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _h(text: str, level: int = 2) -> str:
    return f"\n{'#' * level} {text}\n"


def _table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a Markdown table string."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep    = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows   = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, sep] + rows)


def _series_table(series: pd.Series, val_col: str = "Count") -> str:
    df = series.reset_index()
    df.columns = ["Value", val_col]
    return _table(df)


def generate_eda_report(
    df: pd.DataFrame,
    state_code: str,
    state_name: str,
    report_dir: Path,
) -> None:
    """
    Full EDA report for df saved to report_dir.

    Parameters
    ----------
    df : pd.DataFrame
    state_code : str
    state_name : str
    report_dir : Path
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    log.info("Generating EDA report for %s ...", state_name)

    n_rows, n_cols = df.shape
    lines: list[str] = []

    lines.append(f"# {state_name} - Exploratory Data Analysis Report")
    lines.append(f"\n_Phase-1 Pipeline | State: {state_code}_\n")

    # 1. Dataset Overview
    lines.append(_h("1. Dataset Overview"))
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| State | {state_name} ({state_code}) |")
    lines.append(f"| Rows | {n_rows:,} |")
    lines.append(f"| Columns | {n_cols:,} |")
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    lines.append(f"| Memory usage | {mem_mb:.1f} MB |")

    # 2. Column Types
    lines.append(_h("2. Column Data Types"))
    dtype_counts = df.dtypes.value_counts()
    lines.append(f"| Dtype | Column Count |")
    lines.append(f"|-------|-------------|")
    for dtype, cnt in dtype_counts.items():
        lines.append(f"| `{dtype}` | {cnt} |")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categ_cols   = df.select_dtypes(include=["object", "category"]).columns.tolist()
    lines.append(f"\n- Numeric columns    : **{len(numeric_cols)}**")
    lines.append(f"- Categorical/object : **{len(categ_cols)}**")

    # 3. Missing Values
    lines.append(_h("3. Missing Values"))
    null_pct = (df.isnull().sum() / n_rows * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]
    lines.append(f"- Columns with >=1 missing value : **{len(null_pct)}** / {n_cols}")
    lines.append(f"- Columns fully populated        : **{n_cols - len(null_pct)}** / {n_cols}")

    if not null_pct.empty:
        lines.append("\n**Top 30 columns by % missing:**\n")
        top_missing = null_pct.head(30).reset_index()
        top_missing.columns = ["Column", "Missing %"]
        top_missing["Missing %"] = top_missing["Missing %"].round(2)
        lines.append(_table(top_missing))

    # 4. Fire Size Distribution
    lines.append(_h("4. Fire Size Distribution (acres)"))
    if COL_FIRE_SIZE in df.columns:
        fs = pd.to_numeric(df[COL_FIRE_SIZE], errors="coerce").dropna()
        bins   = [0, 0.25, 1, 10, 100, 300, 1000, 5000, float("inf")]
        labels = ["<0.25", "0.25-1", "1-10", "10-100", "100-300", "300-1k", "1k-5k", ">5k"]
        cut = pd.cut(fs, bins=bins, labels=labels, right=True)
        vc  = cut.value_counts().sort_index()
        lines.append(f"| Size Class (acres) | Count | Pct |")
        lines.append(f"|--------------------|-------|-----|")
        for lbl, cnt in vc.items():
            lines.append(f"| {lbl} | {cnt:,} | {cnt/len(fs)*100:.1f}% |")

        percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        lines.append(f"\n**Percentiles:**\n")
        lines.append(f"| Percentile | Value (acres) |")
        lines.append(f"|-----------|--------------|")
        for p in percentiles:
            lines.append(f"| {p}th | {np.percentile(fs, p):.4f} |")

    # 5. Fire Causes
    lines.append(_h("5. Distribution of Fire Causes"))
    if COL_CAUSE in df.columns:
        vc = df[COL_CAUSE].value_counts(dropna=False)
        lines.append(_series_table(vc))
    else:
        lines.append("_Column not present._")

    # 6. Fires per Year
    lines.append(_h("6. Fires per Year"))
    if COL_FIRE_YEAR in df.columns:
        vc = df[COL_FIRE_YEAR].value_counts().sort_index()
        lines.append(_series_table(vc, "Fire Count"))

    # 7. Fires per Month
    lines.append(_h("7. Fires per Month"))
    if COL_DISCOVERY_DATE in df.columns:
        dates = pd.to_datetime(df[COL_DISCOVERY_DATE], errors="coerce")
        months = dates.dt.month.value_counts().sort_index()
        month_names = {
            1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
            7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec",
        }
        months.index = months.index.map(lambda x: f"{x} ({month_names.get(x,'?')})")
        lines.append(_series_table(months, "Fire Count"))
    else:
        lines.append("_DISCOVERY_DATE column not present._")

    # 8. Top Counties
    lines.append(_h(f"8. Top {TOP_N_COUNTIES} Counties by Fire Count"))
    if COL_COUNTY in df.columns:
        vc = df[COL_COUNTY].value_counts().head(TOP_N_COUNTIES)
        lines.append(_series_table(vc))

    # 9. Top Reporting Agencies
    lines.append(_h(f"9. Top {TOP_N_AGENCIES} Reporting Agencies"))
    if COL_AGENCY in df.columns:
        vc = df[COL_AGENCY].value_counts().head(TOP_N_AGENCIES)
        lines.append(_series_table(vc))

    # 10. Correlation Matrix
    lines.append(_h("10. Correlation Matrix (numeric columns)"))
    if len(numeric_cols) >= 2:
        corr_df   = df[numeric_cols].corr(method="pearson", numeric_only=True)
        corr_path = report_dir / "correlation.csv"
        corr_df.to_csv(corr_path)
        log.info("Correlation matrix saved -> %s", corr_path)
        lines.append(f"_Full matrix saved to `{corr_path.name}`. Preview (first 15x15):_\n")
        preview = corr_df.iloc[:15, :15].round(3)
        lines.append(_table(preview.reset_index().rename(columns={"index": ""})))
    else:
        lines.append("_Fewer than 2 numeric columns - correlation skipped._")
        (report_dir / "correlation.csv").write_text("", encoding="utf-8")

    # 11. Constant Columns
    lines.append(_h("11. Constant Columns (single unique value)"))
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if const_cols:
        lines.append(f"Found **{len(const_cols)}** constant column(s):\n")
        for c in const_cols:
            val = df[c].iloc[0] if n_rows else "N/A"
            lines.append(f"- `{c}` = `{val}`")
    else:
        lines.append("_No constant columns._")

    # 12. Duplicate Columns
    lines.append(_h("12. Duplicate Columns"))
    dup_col_groups: list[list[str]] = []
    checked: set[str] = set()
    for i, c1 in enumerate(df.columns):
        if c1 in checked:
            continue
        group = [c1]
        for c2 in df.columns[i + 1:]:
            if c2 in checked:
                continue
            try:
                if df[c1].equals(df[c2]):
                    group.append(c2)
                    checked.add(c2)
            except Exception:
                pass
        if len(group) > 1:
            dup_col_groups.append(group)
            checked.add(c1)
    if dup_col_groups:
        lines.append(f"Found **{len(dup_col_groups)}** group(s) of duplicate columns:\n")
        for grp in dup_col_groups:
            lines.append(f"- {grp}")
    else:
        lines.append("_No duplicate columns found._")

    # 13. Highly Correlated Columns
    lines.append(_h(f"13. Highly Correlated Columns (|r| >= {CORR_HIGH_THRESHOLD})"))
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr(method="pearson", numeric_only=True).abs()
        upper   = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
        high_pairs = [
            (c1, c2, round(float(upper.loc[c1, c2]), 4))
            for c1 in upper.columns
            for c2 in upper.columns
            if pd.notna(upper.loc[c1, c2]) and upper.loc[c1, c2] >= CORR_HIGH_THRESHOLD
        ]
        if high_pairs:
            lines.append(f"Found **{len(high_pairs)}** highly correlated pair(s):\n")
            lines.append(f"| Column A | Column B | |r| |")
            lines.append(f"|----------|----------|------|")
            for c1, c2, r in sorted(high_pairs, key=lambda x: -x[2])[:50]:
                lines.append(f"| {c1} | {c2} | {r} |")
        else:
            lines.append(f"_No pairs with |r| >= {CORR_HIGH_THRESHOLD}._")
    else:
        lines.append("_Not enough numeric columns._")

    # 14. Categorical Cardinality
    lines.append(_h("14. Categorical Column Cardinality"))
    if categ_cols:
        card_df = pd.DataFrame({
            "Column": categ_cols,
            "Unique Values": [df[c].nunique(dropna=False) for c in categ_cols],
        }).sort_values("Unique Values", ascending=False)
        lines.append(f"| Column | Unique Values | High Cardinality |")
        lines.append(f"|--------|--------------|-----------------|")
        for _, row in card_df.iterrows():
            flag = "[HIGH]" if row["Unique Values"] > CATEGORICAL_HIGH_CARD else "No"
            lines.append(f"| {row['Column']} | {row['Unique Values']:,} | {flag} |")

    # 15. Potential Useless Columns
    lines.append(_h("15. Potential Useless Columns"))
    useless: list[str] = []
    for c in df.columns:
        n_unique = df[c].nunique(dropna=False)
        pct_null = df[c].isnull().mean()
        if n_unique <= 1:
            useless.append(f"`{c}` - constant (1 unique value)")
        elif pct_null > 0.95:
            useless.append(f"`{c}` - {pct_null*100:.0f}% missing")
        elif n_unique == n_rows:
            useless.append(f"`{c}` - all values unique (likely ID/index)")
    if useless:
        for u in useless[:40]:
            lines.append(f"- {u}")
    else:
        lines.append("_No obviously useless columns detected._")

    # 16. Potential Leakage Columns
    lines.append(_h("16. Potential Leakage Columns"))
    lines.append(
        "_These columns describe post-ignition outcomes and should be excluded from "
        "any predictive model trained to predict ignition._\n"
    )
    leakage: list[str] = []
    for c in df.columns:
        for kw in LEAKAGE_KEYWORDS:
            if kw.upper() in c.upper():
                leakage.append(c)
                break
    if leakage:
        for lk in leakage:
            lines.append(f"- `{lk}`")
    else:
        lines.append("_No known leakage columns detected._")

    # Write EDA report
    eda_path = report_dir / "eda_report.md"
    eda_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("EDA report saved -> %s", eda_path)
