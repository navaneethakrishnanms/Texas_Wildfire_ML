"""
src/report_generator.py
------------------------
Analysis 15: Final Summary Report Generator.

Generates a professional Markdown report consolidating all analysis results,
and optionally a PDF version via weasyprint or pdfkit (if available).

Saves
-----
reports/final_summary_report.md
reports/final_summary_report.pdf  (if weasyprint/pdfkit available)
"""

from __future__ import annotations

import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.config import (
    FINAL_REPORT_MD,
    FINAL_REPORT_PDF,
    LOG_FILE,
    MISSING_CRITICAL,
    MISSING_HIGH,
    MISSING_MODERATE,
    REPORTS_DIR,
    TABLES_DIR,
)
from src.utils import ensure_dirs, setup_logger

logger = setup_logger(__name__, LOG_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# Safe Table Loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_table(filename: str) -> pd.DataFrame | None:
    """Load a CSV from TABLES_DIR. Returns None if file does not exist."""
    path = TABLES_DIR / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:
            logger.warning(f"Could not load {filename}: {exc}")
    return None


def _df_to_md(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Convert a DataFrame to a Markdown table string (truncated to max_rows)."""
    if df is None or df.empty:
        return "_No data available._\n"
    sub = df.head(max_rows)
    header = "| " + " | ".join(sub.columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(sub.columns)) + " |"
    rows   = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for _, row in sub.iterrows()
    )
    note = f"\n\n_Showing top {min(len(df), max_rows)} of {len(df)} rows._" if len(df) > max_rows else ""
    return f"{header}\n{sep}\n{rows}{note}\n"


# ─────────────────────────────────────────────────────────────────────────────
# Section Builders
# ─────────────────────────────────────────────────────────────────────────────

def _section_overview() -> str:
    df = _load_table("dataset_overview.csv")
    if df is None:
        return ""
    rows = "\n".join(f"| **{r['Property']}** | {r['Value']} |" for _, r in df.iterrows())
    return f"""
## 1. Dataset Overview

| Property | Value |
|----------|-------|
{rows}

"""


def _section_schema() -> str:
    df = _load_table("schema_analysis.csv")
    if df is None:
        return ""
    type_counts = df["Semantic Type"].value_counts() if "Semantic Type" in df.columns else pd.Series()
    type_summary = "\n".join(f"- **{t}**: {c} columns" for t, c in type_counts.items())
    return f"""
## 2. Schema & Feature Summary

- Total features analyzed: **{len(df)}**

### Semantic Type Breakdown
{type_summary}

### Sample — First 20 Features
{_df_to_md(df[['Column Name', 'Data Type', 'Semantic Type', 'Unique Values', 'Missing %', 'Description']], max_rows=20)}
"""


def _section_feature_categories() -> str:
    df = _load_table("category_summary.csv")
    if df is None:
        return ""
    return f"""
## 3. Feature Category Distribution

{_df_to_md(df[['Category', 'Column Count']], max_rows=30)}
"""


def _section_missing() -> str:
    df = _load_table("missing_summary.csv")
    tier_df = _load_table("missing_tier_breakdown.csv")
    if df is None:
        return ""

    total = len(df)
    has_missing = (df["Missing %"] > 0).sum() if "Missing %" in df.columns else "N/A"
    complete    = total - has_missing if isinstance(has_missing, int) else "N/A"

    critical_cols = df[df["Missing %"] >= MISSING_CRITICAL]["Column"].tolist() if "Missing %" in df.columns else []
    high_cols     = df[(df["Missing %"] >= MISSING_HIGH) & (df["Missing %"] < MISSING_CRITICAL)]["Column"].tolist() if "Missing %" in df.columns else []
    mod_cols      = df[(df["Missing %"] >= MISSING_MODERATE) & (df["Missing %"] < MISSING_HIGH)]["Column"].tolist() if "Missing %" in df.columns else []

    crit_list = "\n".join(f"  - `{c}`" for c in critical_cols[:20]) or "  _None_"
    high_list = "\n".join(f"  - `{c}`" for c in high_cols[:20])     or "  _None_"
    mod_list  = "\n".join(f"  - `{c}`" for c in mod_cols[:20])      or "  _None_"

    tier_md = _df_to_md(tier_df) if tier_df is not None else ""

    return f"""
## 4. Missing Value Summary

| Metric | Value |
|--------|-------|
| Total Columns | {total} |
| Fully Complete | {complete} |
| Has Any Missing | {has_missing} |

### Missing Tier Breakdown
{tier_md}

### Critical (≥{MISSING_CRITICAL}% missing)
{crit_list}

### High (≥{MISSING_HIGH}% missing)
{high_list}

### Moderate (≥{MISSING_MODERATE}% missing)
{mod_list}

> **Note:** No values were imputed. No columns were removed. This is a read-only analysis.

"""


def _section_quality() -> str:
    df = _load_table("feature_quality.csv")
    const_df  = _load_table("constant_columns.csv")
    dup_df    = _load_table("duplicate_columns.csv")
    inf_df    = _load_table("infinite_value_columns.csv")
    if df is None:
        return ""

    total  = len(df)
    review = (df["Quality Score"] == "REVIEW").sum() if "Quality Score" in df.columns else "N/A"

    const_list = ", ".join(f"`{c}`" for c in const_df["Column"].tolist()[:20]) if const_df is not None and not const_df.empty else "_None_"
    dup_list   = ", ".join(f"`{c}`" for c in dup_df["Column"].tolist()[:20])   if dup_df is not None and not dup_df.empty else "_None_"
    inf_list   = ", ".join(f"`{c}`" for c in inf_df["Column"].tolist()[:20])   if inf_df is not None and not inf_df.empty else "_None_"

    return f"""
## 5. Feature Quality Summary

| Metric | Value |
|--------|-------|
| Total Columns | {total} |
| PASS (no issues) | {total - review if isinstance(review, int) else "N/A"} |
| REVIEW (has issues) | {review} |

- **Constant columns**: {const_list}
- **Duplicate columns**: {dup_list}
- **Columns with infinite values**: {inf_list}

"""


def _section_statistics() -> str:
    df = _load_table("statistical_summary.csv")
    skew_df = _load_table("skewed_columns.csv")
    if df is None:
        return ""

    n_skewed = len(skew_df) if skew_df is not None else "N/A"
    return f"""
## 6. Statistical Summary

- Numeric columns analyzed: **{len(df)}**
- Highly skewed columns (|skew| > 2): **{n_skewed}**

### Top 15 by Absolute Skewness
{_df_to_md(skew_df[['Column', 'Skewness', 'Mean', 'Median', 'Std', 'Outlier %']].head(15) if skew_df is not None else pd.DataFrame(), max_rows=15)}
"""


def _section_temporal() -> str:
    year_df  = _load_table("temporal_year.csv")
    month_df = _load_table("temporal_month.csv")
    dur_df   = _load_table("temporal_duration.csv")
    return f"""
## 7. Temporal Analysis

### Fires by Year
{_df_to_md(year_df, max_rows=10)}

### Fires by Month
{_df_to_md(month_df, max_rows=12)}

### Duration Statistics
{_df_to_md(dur_df, max_rows=5)}
"""


def _section_geographic() -> str:
    state_df  = _load_table("geo_state_counts.csv")
    county_df = _load_table("geo_county_counts.csv")
    eco_df    = _load_table("geo_ecoregion_counts.csv")
    return f"""
## 8. Geographic Analysis

### Top States by Fire Count
{_df_to_md(state_df.head(15) if state_df is not None else pd.DataFrame(), max_rows=15)}

### Top 15 Counties
{_df_to_md(county_df.head(15) if county_df is not None else pd.DataFrame(), max_rows=15)}

### Top Ecoregions
{_df_to_md(eco_df.head(10) if eco_df is not None else pd.DataFrame(), max_rows=10)}
"""


def _section_correlation() -> str:
    p_pairs = _load_table("highly_correlated_pairs_pearson.csv")
    s_pairs = _load_table("highly_correlated_pairs_spearman.csv")
    return f"""
## 9. Correlation Summary

### Top 20 Highly Correlated Pairs (Pearson)
{_df_to_md(p_pairs[['Feature A', 'Feature B', 'Correlation']].head(20) if p_pairs is not None else pd.DataFrame(), max_rows=20)}

### Top 20 Highly Correlated Pairs (Spearman)
{_df_to_md(s_pairs[['Feature A', 'Feature B', 'Correlation']].head(20) if s_pairs is not None else pd.DataFrame(), max_rows=20)}
"""


def _section_categorical() -> str:
    df = _load_table("categorical_summary.csv")
    return f"""
## 10. Categorical Analysis

{_df_to_md(df.head(20) if df is not None else pd.DataFrame(), max_rows=20)}
"""


def _section_leakage() -> str:
    df = _load_table("leakage_analysis.csv")
    summary = _load_table("leakage_summary.csv")
    if df is None:
        return ""

    def_cols = df[df["Leakage Label"] == "Definite Leakage"]["Column"].tolist() if "Leakage Label" in df.columns else []
    pos_cols = df[df["Leakage Label"] == "Possible Leakage"]["Column"].tolist() if "Leakage Label" in df.columns else []
    def_list = "\n".join(f"- `{c}`" for c in def_cols) or "_None_"
    pos_list = "\n".join(f"- `{c}`" for c in pos_cols[:20]) or "_None_"

    return f"""
## 11. Leakage Analysis

### Summary
{_df_to_md(summary, max_rows=5)}

### Definite Leakage Columns
{def_list}

> These columns contain information that would only be available **after** a fire event, and must be excluded from any predictive model.

### Possible Leakage Columns
{pos_list}

> These columns require domain expert review before inclusion.

"""


def _section_readiness() -> str:
    df = _load_table("predictive_readiness_summary.csv")
    full_df = _load_table("predictive_readiness.csv")

    candidate_cols = []
    if full_df is not None and "Readiness Label" in full_df.columns:
        candidate_cols = full_df[full_df["Readiness Label"] == "Candidate Feature"]["Column"].tolist()

    cand_list = "\n".join(f"- `{c}`" for c in candidate_cols[:30]) or "_None identified_"

    return f"""
## 12. Predictive Readiness

### Summary by Label
{_df_to_md(df, max_rows=10)}

### Candidate Features (pre-fire, not leakage)
{cand_list}

> **Important:** This is a recommendation only. Final feature selection requires domain expert review.

"""


def _section_feature_groups() -> str:
    df = _load_table("feature_groups_summary.csv")
    return f"""
## 13. Feature Dependency Groups

{_df_to_md(df[['Feature Group', 'Count']].rename(columns={'Count': 'Column Count'}) if df is not None and 'Count' in df.columns else df, max_rows=25)}
"""


def _section_source_readiness() -> str:
    df = _load_table("source_readiness_summary.csv")
    return f"""
## 14. Source Readiness (Update Frequency)

{_df_to_md(df, max_rows=10)}
"""


def _section_next_steps() -> str:
    return """
## 15. Recommended Next Steps

Based on the exploratory analysis, the following actions are recommended:

### 🔴 Immediate Actions (Pre-Modeling)
1. **Remove Definite Leakage Columns** — Exclude `CONT_DATE`, `CONT_DOY`, `CONT_TIME`, 
   `MTBS_ID`, `MTBS_FIRE_NAME`, `ICS_209_PLUS_*`, `FIRE_SIZE`, `FIRE_SIZE_CLASS` 
   from input features (they are post-fire outcomes, not predictors).
2. **Review High-Missing Columns** — Columns with >90% missing data (e.g., `geometry`, 
   `road_interstate_dis`, `GACC_Fire Use Teams`) should be evaluated for drop or imputation.
3. **Resolve Constant/Near-Constant Columns** — Zero-variance columns contribute no 
   information and can be safely dropped after confirmation.
4. **Investigate Duplicate Columns** — Review duplicate-content groups and keep only 
   the most interpretable representative.

### 🟡 Feature Engineering Decisions
5. **Address Skewed Features** — Highly skewed numeric columns (|skew| > 2) may benefit 
   from log or Box-Cox transformations in the preprocessing phase.
6. **Resolve High-Cardinality Categoricals** — Columns with >100 unique categories may 
   need target encoding, hashing, or grouping of rare levels.
7. **Verify Temporal Features** — Confirm `DISCOVERY_DOY`, `DISCOVERY_DATE` month/season 
   encoding strategy before training.

### 🟢 Modeling Readiness
8. **Define Target Variable** — Confirm whether target is binary ignition (Yes/No), 
   fire size class (A–G), or continuous acres burned.
9. **Finalize Feature Set** — After addressing leakage and quality issues, compile the 
   final feature matrix from Candidate Feature columns.
10. **Train/Test Split Strategy** — Use temporal splitting (train on 2014–2018, 
    validate on 2019, test on 2020) to prevent temporal leakage.

### 📊 Additional Analysis
11. **Spatial Autocorrelation** — Run Moran's I on fire occurrence to understand 
    spatial clustering patterns.
12. **Class Imbalance Assessment** — Analyze the ratio of fire vs. non-fire grid cells 
    if framing as a classification problem.

"""


# ─────────────────────────────────────────────────────────────────────────────
# Main Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_final_report() -> Path:
    """
    Analysis 15: Generate the final Markdown summary report.

    Reads all CSV outputs from TABLES_DIR and assembles a professional
    Markdown document.

    Returns
    -------
    Path  Path to the generated Markdown report.
    """
    logger.info("Analysis 15 — Generating Final Summary Report")
    ensure_dirs(REPORTS_DIR)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = [
        f"# Wildfire Dataset — Complete Exploratory Analysis Report\n",
        f"> **Generated:** {timestamp}  \n"
        f"> **Dataset:** Historical Wildfire Records 2014–2020 (FPA-FOD)  \n"
        f"> **Pipeline:** Wildfire Prediction System — Dataset Analysis Module  \n",
        "---\n",
        "## Table of Contents\n"
        "1. [Dataset Overview](#1-dataset-overview)\n"
        "2. [Schema & Feature Summary](#2-schema--feature-summary)\n"
        "3. [Feature Categories](#3-feature-category-distribution)\n"
        "4. [Missing Value Summary](#4-missing-value-summary)\n"
        "5. [Feature Quality Summary](#5-feature-quality-summary)\n"
        "6. [Statistical Summary](#6-statistical-summary)\n"
        "7. [Temporal Analysis](#7-temporal-analysis)\n"
        "8. [Geographic Analysis](#8-geographic-analysis)\n"
        "9. [Correlation Summary](#9-correlation-summary)\n"
        "10. [Categorical Analysis](#10-categorical-analysis)\n"
        "11. [Leakage Analysis](#11-leakage-analysis)\n"
        "12. [Predictive Readiness](#12-predictive-readiness)\n"
        "13. [Feature Dependency Groups](#13-feature-dependency-groups)\n"
        "14. [Source Readiness](#14-source-readiness-update-frequency)\n"
        "15. [Recommended Next Steps](#15-recommended-next-steps)\n",
        "---\n",
        _section_overview(),
        _section_schema(),
        _section_feature_categories(),
        _section_missing(),
        _section_quality(),
        _section_statistics(),
        _section_temporal(),
        _section_geographic(),
        _section_correlation(),
        _section_categorical(),
        _section_leakage(),
        _section_readiness(),
        _section_feature_groups(),
        _section_source_readiness(),
        _section_next_steps(),
        "---\n",
        f"_Report generated by the Wildfire Dataset Analysis Pipeline — {timestamp}_\n",
    ]

    report_text = "\n".join(sections)

    with open(FINAL_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"  ✔ Markdown report saved: {FINAL_REPORT_MD}")

    # PDF Export (optional — requires weasyprint or pdfkit)
    _export_pdf(report_text)

    return FINAL_REPORT_MD


def _export_pdf(md_text: str) -> None:
    """Attempt to export PDF. Silently skips if no converter is available."""
    # Try weasyprint (preferred)
    try:
        import markdown as md_lib
        from weasyprint import HTML, CSS

        html_body = md_lib.markdown(md_text, extensions=["tables", "fenced_code", "toc"])
        html_full = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
  body {{ font-family: 'Arial', sans-serif; margin: 40px; font-size: 12px; line-height: 1.6; }}
  h1 {{ color: #1a237e; }} h2 {{ color: #283593; border-bottom: 1px solid #ccc; }}
  h3 {{ color: #3949ab; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10px; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
  th {{ background: #e8eaf6; }}
  code {{ background: #f5f5f5; padding: 1px 4px; border-radius: 3px; }}
  blockquote {{ border-left: 4px solid #3949ab; margin: 0; padding: 8px 16px; background: #e8eaf6; }}
</style>
</head><body>{html_body}</body></html>"""
        HTML(string=html_full).write_pdf(str(FINAL_REPORT_PDF))
        logger.info(f"  ✔ PDF report saved: {FINAL_REPORT_PDF}")
        return
    except ImportError:
        pass
    except Exception as exc:
        logger.warning(f"  weasyprint PDF export failed: {exc}")

    # Try pdfkit (wkhtmltopdf)
    try:
        import markdown as md_lib
        import pdfkit

        html_body = md_lib.markdown(md_text, extensions=["tables", "fenced_code"])
        html_full = f"<html><body style='font-family:Arial;margin:40px'>{html_body}</body></html>"
        pdfkit.from_string(html_full, str(FINAL_REPORT_PDF))
        logger.info(f"  ✔ PDF report saved (pdfkit): {FINAL_REPORT_PDF}")
        return
    except ImportError:
        pass
    except Exception as exc:
        logger.warning(f"  pdfkit PDF export failed: {exc}")

    logger.info("  PDF export skipped (install weasyprint + markdown, or pdfkit + wkhtmltopdf).")
