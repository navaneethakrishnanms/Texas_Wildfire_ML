"""
src/production_availability.py
--------------------------------
Analysis C — Production Feature Availability Report

For every feature, answers:
  "Can this feature be obtained BEFORE a fire occurs in production?"

Availability labels:
  PRE_FIRE_STATIC   — always available, static data
  PRE_FIRE_DYNAMIC  — available pre-fire, updated regularly (daily/16-day/monthly)
  SAME_DAY_RISK     — available same day but temporal contamination risk
  POST_FIRE_ONLY    — only exists after fire is discovered/contained → LEAKAGE
  ADMIN_ONLY        — administrative ID, not a predictive feature
  EXCLUDE           — artifact, corrupted, or zero-variance

Input:
  tables/<state>/leakage_analysis.csv    (from Phase 1)
  outputs/<state>/feature_source_map.csv  (from Analysis A)

Output:
  outputs/<state>/production_availability.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.phase2_config import (
    AVAILABILITY_RULES,
    DEFINITE_LEAKAGE_COLS,
    ADMIN_ID_COLS,
    STRUCTURAL_ARTIFACT_COLS,
    CORRUPTED_COLS,
)


logger = logging.getLogger(__name__)


# Features that should be available pre-fire (from source map update frequency)
_DYNAMIC_FREQ_KEYWORDS = {
    "Daily":        "PRE_FIRE_DYNAMIC",
    "16-day":       "PRE_FIRE_DYNAMIC",
    "Monthly":      "PRE_FIRE_DYNAMIC",
    "Annual":       "PRE_FIRE_STATIC",
    "~2 years":     "PRE_FIRE_STATIC",
    "5 years":      "PRE_FIRE_STATIC",
    "Static":       "PRE_FIRE_STATIC",
}


def _classify_availability(
    col: str,
    leakage_label: str | None,
    update_freq: str | None,
) -> tuple[str, str]:
    """
    Determine production availability for a column.

    Returns
    -------
    (availability_label, reason)
    """
    # Hard exclusions first
    if col in DEFINITE_LEAKAGE_COLS or leakage_label == "Definite Leakage":
        return "POST_FIRE_ONLY", "Confirmed leakage — only exists after fire event"

    if col in STRUCTURAL_ARTIFACT_COLS:
        return "EXCLUDE", "Format artifact — never a real data column"

    if col in CORRUPTED_COLS:
        return "EXCLUDE", "Corrupted values (numerical overflow) — cannot use"

    if col in ADMIN_ID_COLS:
        return "ADMIN_ONLY", "Administrative identifier — not a predictive feature"

    # Check explicit availability rules
    for label, col_list in AVAILABILITY_RULES.items():
        if col in col_list:
            reasons = {
                "POST_FIRE": "Only known after fire discovery/containment",
                "PRE_FIRE_STATIC": "Static data — always available",
                "PRE_FIRE_DYNAMIC": "Dynamic — available pre-fire from live API",
                "SAME_DAY_RISK": "Available same day but temporal contamination risk",
            }
            return label, reasons.get(label, "")

    # Use update frequency from source map
    if update_freq:
        for freq_key, avail_label in _DYNAMIC_FREQ_KEYWORDS.items():
            if freq_key.lower() in update_freq.lower():
                return avail_label, f"Based on source update frequency: {update_freq}"

    # Post-fire check from update_freq
    if update_freq and "post-fire" in update_freq.lower():
        return "POST_FIRE_ONLY", "Source only exists after fire event"

    return "PRE_FIRE_DYNAMIC", "Default — assumed available pre-fire (verify)"


def generate_production_availability(
    leakage_csv: Path,
    source_map_csv: Path,
    output_dir: Path,
    state_name: str,
) -> pd.DataFrame:
    """
    Analysis C: Classify production availability for every feature.

    Parameters
    ----------
    leakage_csv    : Phase 1 leakage_analysis.csv
    source_map_csv : Analysis A feature_source_map.csv
    output_dir     : Where to save output
    state_name     : 'Texas' or 'California'

    Returns
    -------
    pd.DataFrame  Availability table (one row per column)
    """
    logger.info(f"Analysis C — Production Feature Availability [{state_name}]")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load leakage analysis
    leakage_df = pd.DataFrame()
    if leakage_csv.exists():
        leakage_df = pd.read_csv(leakage_csv)
        logger.info(f"  Loaded leakage data: {len(leakage_df)} columns")
    else:
        logger.warning(f"  Leakage CSV not found: {leakage_csv}")

    # Load source map
    source_df = pd.DataFrame()
    if source_map_csv.exists():
        source_df = pd.read_csv(source_map_csv)
        logger.info(f"  Loaded source map: {len(source_df)} columns")
    else:
        logger.warning(f"  Source map CSV not found: {source_map_csv}")

    # Build lookup dicts
    leakage_lookup: dict[str, str] = {}
    if not leakage_df.empty:
        col_f = "Column" if "Column" in leakage_df.columns else leakage_df.columns[0]
        lbl_f = "Leakage Label" if "Leakage Label" in leakage_df.columns else leakage_df.columns[1]
        leakage_lookup = dict(zip(leakage_df[col_f], leakage_df[lbl_f]))

    freq_lookup: dict[str, str] = {}
    if not source_df.empty:
        freq_lookup = dict(zip(source_df["Column"], source_df.get("Update_Frequency", pd.Series(dtype=str))))

    # Get all columns from source map (or leakage)
    all_cols = source_df["Column"].tolist() if not source_df.empty else list(leakage_lookup.keys())

    rows = []
    label_counts: dict[str, int] = {}

    for col in all_cols:
        leakage_label = leakage_lookup.get(col)
        update_freq   = freq_lookup.get(col)
        avail_label, reason = _classify_availability(col, leakage_label, update_freq)

        label_counts[avail_label] = label_counts.get(avail_label, 0) + 1
        rows.append({
            "Column":               col,
            "Availability_Label":   avail_label,
            "Update_Frequency":     update_freq or "UNKNOWN",
            "Reason":               reason,
            "Usable_In_Model":      avail_label in {"PRE_FIRE_STATIC", "PRE_FIRE_DYNAMIC"},
        })

    df = pd.DataFrame(rows)

    # Print summary
    print(f"\n{'=' * 65}")
    print(f"  ANALYSIS C — PRODUCTION AVAILABILITY [{state_name.upper()}]")
    print(f"{'=' * 65}")
    usable = df["Usable_In_Model"].sum()
    print(f"  ✅  Usable in model (pre-fire): {usable}")
    print(f"  ❌  Post-fire only (leakage)  : {label_counts.get('POST_FIRE_ONLY', 0)}")
    print(f"  ⚠️   Same-day risk              : {label_counts.get('SAME_DAY_RISK', 0)}")
    print(f"  🗑️   Exclude (artifact/corrupt) : {label_counts.get('EXCLUDE', 0)}")
    print(f"  📋   Admin only (IDs)           : {label_counts.get('ADMIN_ONLY', 0)}")
    print(f"\n  Availability breakdown:")
    for label, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {label:<25} {cnt:>4} columns")
    print(f"{'=' * 65}\n")

    for label, cnt in label_counts.items():
        logger.info(f"  {label}: {cnt}")

    # Save
    out_path = output_dir / "production_availability.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"  ✔ Saved: {out_path}")

    return df
