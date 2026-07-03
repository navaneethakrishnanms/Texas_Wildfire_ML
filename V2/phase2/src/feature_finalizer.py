"""
src/feature_finalizer.py
--------------------------
Gate 1 + Gate 2 — Feature Finalization

Applies the two mechanical gates to produce the master feature_schema.csv.
This is the CONTRACT for all downstream phases.

Gate 1 — Production Feasibility (Rule-Based):
  Remove: leakage, post-fire, structural artifacts, corrupted, admin IDs

Gate 2 — Correlation Redundancy:
  Remove: exact duplicates (r = 1.000), near-duplicates from same source

Input:
  outputs/<state>/feature_source_map.csv
  outputs/<state>/missing_root_cause.csv
  outputs/<state>/production_availability.csv
  outputs/<state>/static_dynamic_features.csv
  tables/<state>/highly_correlated_pairs_pearson.csv   (from Phase 1)
  tables/<state>/feature_quality.csv                    (from Phase 1)

Output:
  outputs/<state>/feature_schema.csv   ← THE MASTER CONTRACT
  outputs/<state>/gate1_removed.csv    ← What Gate 1 removed and why
  outputs/<state>/gate2_removed.csv    ← What Gate 2 removed and why
  outputs/<state>/phase2a_summary.md   ← Human-readable summary
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.phase2_config import (
    DEFINITE_LEAKAGE_COLS,
    ADMIN_ID_COLS,
    STRUCTURAL_ARTIFACT_COLS,
    CORRUPTED_COLS,
    EXACT_DUPLICATE_PAIRS,
    NEAR_DUPLICATE_GROUPS,
)


logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Gate 1 Removal Logic
# ─────────────────────────────────────────────────────────────────────────────

def apply_gate1(
    avail_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> tuple[set[str], dict[str, str]]:
    """
    Apply Gate 1 — Production Feasibility.

    Returns
    -------
    removed_cols  : Set of column names to remove
    removal_reasons : {col: reason}
    """
    removed: dict[str, str] = {}

    # From availability analysis
    if not avail_df.empty:
        for _, row in avail_df.iterrows():
            col   = row["Column"]
            label = row.get("Availability_Label", "")
            if label == "POST_FIRE_ONLY":
                removed[col] = "POST_FIRE_ONLY — only exists after fire event (leakage)"
            elif label == "EXCLUDE":
                removed[col] = "EXCLUDE — artifact, corrupted, or zero-variance"
            elif label == "ADMIN_ONLY":
                removed[col] = "ADMIN_ONLY — administrative identifier, not predictive"
            elif label == "SAME_DAY_RISK":
                removed[col] = "SAME_DAY_RISK — temporal contamination risk"

    # Explicit lists (belt-and-suspenders)
    for col in DEFINITE_LEAKAGE_COLS:
        if col not in removed:
            removed[col] = "POST_FIRE_ONLY — confirmed leakage (config list)"

    for col in ADMIN_ID_COLS:
        if col not in removed:
            removed[col] = "ADMIN_ONLY — administrative ID (config list)"

    for col in STRUCTURAL_ARTIFACT_COLS:
        if col not in removed:
            removed[col] = "EXCLUDE — structural artifact (config list)"

    for col in CORRUPTED_COLS:
        if col not in removed:
            removed[col] = "EXCLUDE — corrupted values (config list)"

    # Constant / zero-variance columns from quality report
    if not quality_df.empty:
        const_field = None
        for f in ["Constant", "Is_Constant", "constant"]:
            if f in quality_df.columns:
                const_field = f
                break
        col_field = "Column" if "Column" in quality_df.columns else quality_df.columns[0]
        if const_field:
            for _, row in quality_df.iterrows():
                if row.get(const_field, False):
                    col = row[col_field]
                    if col not in removed:
                        removed[col] = "EXCLUDE — zero-variance (constant in state dataset)"

    logger.info(f"  Gate 1 removed: {len(removed)} columns")
    return set(removed.keys()), removed


# ─────────────────────────────────────────────────────────────────────────────
# Gate 2 Removal Logic
# ─────────────────────────────────────────────────────────────────────────────

def apply_gate2(
    remaining_cols: set[str],
    corr_df: pd.DataFrame,
    corr_threshold: float = 0.990,
) -> tuple[set[str], dict[str, str]]:
    """
    Apply Gate 2 — Correlation Redundancy.

    Parameters
    ----------
    remaining_cols   : Columns that survived Gate 1
    corr_df          : Phase 1 highly_correlated_pairs_pearson.csv
    corr_threshold   : Drop one col from pair if |r| >= this value

    Returns
    -------
    removed_cols   : Set of column names to remove
    removal_reasons: {col: reason}
    """
    removed: dict[str, str] = {}

    # Exact duplicate pairs (config)
    for keep_col, drop_col in EXACT_DUPLICATE_PAIRS:
        if drop_col in remaining_cols:
            removed[drop_col] = f"EXACT_DUPLICATE of {keep_col} (r=1.000)"

    # Near-duplicate groups (config)
    for group_name, group in NEAR_DUPLICATE_GROUPS.items():
        for drop_col in group.get("drop", []):
            if drop_col in remaining_cols:
                keep_cols = group.get("keep", [])
                keep_str  = keep_cols if isinstance(keep_cols, str) else ", ".join(keep_cols)
                removed[drop_col] = f"NEAR_DUPLICATE in group '{group_name}' — keeping {keep_str}"

    # Data-driven: high correlation pairs from Phase 1 analysis
    if not corr_df.empty:
        fa_field = "Feature A" if "Feature A" in corr_df.columns else corr_df.columns[0]
        fb_field = "Feature B" if "Feature B" in corr_df.columns else corr_df.columns[1]
        cr_field = "Correlation" if "Correlation" in corr_df.columns else corr_df.columns[2]

        for _, row in corr_df.iterrows():
            try:
                corr_val = abs(float(row[cr_field]))
            except (ValueError, TypeError):
                continue

            if corr_val < corr_threshold:
                continue

            col_a = str(row[fa_field])
            col_b = str(row[fb_field])

            if col_a not in remaining_cols or col_b not in remaining_cols:
                continue

            # Skip if already handled
            if col_a in removed or col_b in removed:
                continue

            # Decide which to keep: prefer shorter, more standard name
            if len(col_a) <= len(col_b):
                drop_col = col_b
                keep_col = col_a
            else:
                drop_col = col_a
                keep_col = col_b

            removed[drop_col] = (
                f"HIGH_CORR with {keep_col} (r={corr_val:.4f} ≥ {corr_threshold})"
            )

    logger.info(f"  Gate 2 removed: {len(removed)} columns")
    return set(removed.keys()), removed


# ─────────────────────────────────────────────────────────────────────────────
# Master Schema Assembly
# ─────────────────────────────────────────────────────────────────────────────

def generate_feature_schema(
    source_map_csv:   Path,
    missing_cause_csv: Path,
    availability_csv:  Path,
    static_dynamic_csv: Path,
    corr_pearson_csv:  Path,
    quality_csv:       Path,
    output_dir:        Path,
    state_name:        str,
) -> pd.DataFrame:
    """
    Finalize the master feature_schema.csv.

    Applies Gate 1 and Gate 2, then assembles the schema from all 4 analyses.

    Parameters
    ----------
    All CSV paths from Phase 1 and Analyses A–D.
    output_dir  : Where to save all outputs.
    state_name  : 'Texas' or 'California'.

    Returns
    -------
    pd.DataFrame  Master feature schema (retained columns only)
    """
    logger.info(f"Feature Finalizer — Gate 1 + Gate 2 [{state_name}]")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all analysis outputs ────────────────────────────────────────────
    def safe_load(path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path)
        logger.warning(f"  File not found: {path}")
        return pd.DataFrame()

    source_df   = safe_load(source_map_csv)
    missing_df  = safe_load(missing_cause_csv)
    avail_df    = safe_load(availability_csv)
    sd_df       = safe_load(static_dynamic_csv)
    corr_df     = safe_load(corr_pearson_csv)
    quality_df  = safe_load(quality_csv)

    if source_df.empty:
        logger.error("  Source map is empty — cannot build schema")
        return pd.DataFrame()

    all_cols = set(source_df["Column"].tolist())
    logger.info(f"  Starting columns: {len(all_cols)}")

    # ── Gate 1 ───────────────────────────────────────────────────────────────
    g1_removed, g1_reasons = apply_gate1(avail_df, quality_df)
    after_g1 = all_cols - g1_removed
    logger.info(f"  After Gate 1: {len(after_g1)} columns ({len(g1_removed)} removed)")

    # ── Gate 2 ───────────────────────────────────────────────────────────────
    g2_removed, g2_reasons = apply_gate2(after_g1, corr_df)
    after_g2 = after_g1 - g2_removed
    logger.info(f"  After Gate 2: {len(after_g2)} columns ({len(g2_removed)} removed)")

    # ── Build master schema for retained columns ─────────────────────────────
    # Build lookup dicts from all analyses
    def col_lookup(df: pd.DataFrame, col_field: str, value_field: str) -> dict:
        if df.empty or col_field not in df.columns or value_field not in df.columns:
            return {}
        return dict(zip(df[col_field], df[value_field]))

    src_lookup  = col_lookup(source_df, "Column", "Source_System")
    api_lookup  = col_lookup(source_df, "Column", "API_URL")
    sres_lookup = col_lookup(source_df, "Column", "Spatial_Res")
    tres_lookup = col_lookup(source_df, "Column", "Temporal_Res")
    freq_lookup = col_lookup(source_df, "Column", "Update_Frequency")
    miss_lookup = col_lookup(missing_df, "Column", "Root_Cause") if not missing_df.empty else {}
    treat_lookup= col_lookup(missing_df, "Column", "Treatment") if not missing_df.empty else {}
    miss_pct    = col_lookup(missing_df, "Column", "Missing_%") if not missing_df.empty else {}
    avail_lookup= col_lookup(avail_df, "Column", "Availability_Label") if not avail_df.empty else {}
    sd_lookup   = col_lookup(sd_df, "Column", "Update_Category") if not sd_df.empty else {}
    usable_lookup = col_lookup(avail_df, "Column", "Usable_In_Model") if not avail_df.empty else {}

    schema_rows = []
    for col in sorted(after_g2):
        schema_rows.append({
            "Column":               col,
            "Source_System":        src_lookup.get(col, "UNKNOWN"),
            "API_URL":              api_lookup.get(col, "UNKNOWN"),
            "Spatial_Res":          sres_lookup.get(col, "UNKNOWN"),
            "Temporal_Res":         tres_lookup.get(col, "UNKNOWN"),
            "Update_Frequency":     freq_lookup.get(col, "UNKNOWN"),
            "Update_Category":      sd_lookup.get(col, "UNKNOWN"),
            "Availability_Label":   avail_lookup.get(col, "UNKNOWN"),
            "Missing_%":            miss_pct.get(col, 0.0),
            "Missing_Root_Cause":   miss_lookup.get(col, "none"),
            "Missing_Treatment":    treat_lookup.get(col, "none"),
            "Gate1_Status":         "RETAINED",
            "Gate2_Status":         "RETAINED",
        })

    schema_df = pd.DataFrame(schema_rows)

    # ── Save Gate 1 removal log ───────────────────────────────────────────────
    g1_log = pd.DataFrame([
        {"Column": c, "Gate": "Gate1", "Reason": r}
        for c, r in g1_reasons.items()
    ])
    g1_log.to_csv(output_dir / "gate1_removed.csv", index=False)
    logger.info(f"  Saved gate1_removed.csv ({len(g1_log)} rows)")

    # ── Save Gate 2 removal log ───────────────────────────────────────────────
    g2_log = pd.DataFrame([
        {"Column": c, "Gate": "Gate2", "Reason": r}
        for c, r in g2_reasons.items()
    ])
    g2_log.to_csv(output_dir / "gate2_removed.csv", index=False)
    logger.info(f"  Saved gate2_removed.csv ({len(g2_log)} rows)")

    # ── Save master schema ────────────────────────────────────────────────────
    schema_df.to_csv(output_dir / "feature_schema.csv", index=False)
    logger.info(f"  ✔ Saved feature_schema.csv — {len(schema_df)} retained columns")

    # ── Generate human-readable summary ─────────────────────────────────────
    _write_summary_report(
        state_name, schema_df, g1_log, g2_log,
        len(all_cols), len(after_g1), len(after_g2),
        output_dir,
    )

    # ── Print final console summary ───────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  PHASE 2A COMPLETE — FEATURE SCHEMA [{state_name.upper()}]")
    print(f"{'=' * 70}")
    print(f"  Starting columns        : {len(all_cols):>5}")
    print(f"  Gate 1 removed          : {len(g1_removed):>5}  (leakage, artifacts, admin)")
    print(f"  Gate 2 removed          : {len(g2_removed):>5}  (duplicates, near-duplicates)")
    print(f"  ─────────────────────────────────")
    print(f"  RETAINED in schema      : {len(schema_df):>5}  ← goes to SHAP/LOO in Phase 5")
    print(f"\n  Update Category Breakdown:")
    for cat, grp in schema_df.groupby("Update_Category"):
        print(f"    {cat:<15} {len(grp):>4} columns")
    print(f"\n  Output: {output_dir / 'feature_schema.csv'}")
    print(f"{'=' * 70}\n")

    return schema_df


def _write_summary_report(
    state_name: str,
    schema_df: pd.DataFrame,
    g1_log: pd.DataFrame,
    g2_log: pd.DataFrame,
    n_start: int,
    n_after_g1: int,
    n_after_g2: int,
    output_dir: Path,
) -> None:
    """Write a human-readable Phase 2A summary Markdown report."""
    lines = [
        f"# Phase 2A Summary — Feature Schema [{state_name}]",
        "",
        f"**Generated by:** Phase 2A Feature Finalization Pipeline",
        "",
        "---",
        "",
        "## Reduction Summary",
        "",
        f"| Stage | Columns | Removed |",
        f"|-------|---------|---------|",
        f"| Original dataset | {n_start} | — |",
        f"| After Gate 1 (production feasibility) | {n_after_g1} | {n_start - n_after_g1} |",
        f"| After Gate 2 (correlation redundancy) | {n_after_g2} | {n_after_g1 - n_after_g2} |",
        f"| **Final retained schema** | **{n_after_g2}** | **{n_start - n_after_g2} total** |",
        "",
        "---",
        "",
        "## Gate 1 Removals (Production Feasibility)",
        "",
        "| Column | Reason |",
        "|--------|--------|",
    ]
    for _, row in g1_log.iterrows():
        lines.append(f"| `{row['Column']}` | {row['Reason']} |")

    lines += [
        "",
        "---",
        "",
        "## Gate 2 Removals (Correlation Redundancy)",
        "",
        "| Column | Reason |",
        "|--------|--------|",
    ]
    for _, row in g2_log.iterrows():
        lines.append(f"| `{row['Column']}` | {row['Reason']} |")

    lines += [
        "",
        "---",
        "",
        "## Retained Feature Schema by Update Category",
        "",
        "| Update Category | Count | Description |",
        "|----------------|-------|-------------|",
    ]
    if not schema_df.empty and "Update_Category" in schema_df.columns:
        for cat, grp in schema_df.groupby("Update_Category"):
            from config.phase2_config import STATIC_DYNAMIC_RULES
            desc = STATIC_DYNAMIC_RULES.get(cat, {}).get("description", "")
            lines.append(f"| {cat} | {len(grp)} | {desc} |")

    lines += [
        "",
        "---",
        "",
        "## Next Steps",
        "",
        "The `feature_schema.csv` is the contract for all downstream phases:",
        "",
        "- **Phase 2B:** Build H3 grid (TX: R7, CA: R8) using only burnable cells",
        "- **Phase 2C:** Map FPA-FOD fires to H3 cells → Fire=1 cell-days",
        "- **Phase 2D:** DAY-ECO-MATCHED 1:10 negative sampling + API feature collection",
        "- **Phase 2E:** Assemble final training dataset (Fire=1 + Fire=0)",
        "- **Phase 5:** Train LightGBM/XGBoost → run SHAP/LOO → Gate 3 selection",
        "",
        "> Gate 3 (SHAP/LOO ablation) will further reduce from the retained schema",
        "> to the final model feature set. Feature count emerges from data, not pre-decided.",
    ]

    report_path = output_dir / "phase2a_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  ✔ Saved phase2a_summary.md")
