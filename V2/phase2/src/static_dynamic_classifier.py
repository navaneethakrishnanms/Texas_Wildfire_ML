"""
src/static_dynamic_classifier.py
----------------------------------
Analysis D — Static vs Dynamic Feature Classification

Classifies every feature by how often it must be updated
in the production operational pipeline.

Categories:
  STATIC       — Download once, refresh every 2–5 years
  ANNUAL       — Update once per year (RAP, NLCD)
  MONTHLY      — Update monthly (drought indices, water budget)
  DAILY        — Download every day (weather, fire weather indices)
  EVENT_BASED  — Only exists after a fire event (exclude from production)

Input:
  outputs/<state>/feature_source_map.csv   (from Analysis A)
  outputs/<state>/production_availability.csv  (from Analysis C)

Output:
  outputs/<state>/static_dynamic_features.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.phase2_config import STATIC_DYNAMIC_RULES


logger = logging.getLogger(__name__)

# Priority order — more specific category wins
_CATEGORY_PRIORITY = ["EVENT_BASED", "DAILY", "MONTHLY", "ANNUAL", "STATIC"]


def _classify_update_frequency(
    col: str,
    update_freq: str | None,
    availability_label: str | None,
) -> tuple[str, str]:
    """
    Determine how often this feature must be updated in production.

    Returns
    -------
    (update_category, reason)
    """
    col_lower = col.lower()

    # Post-fire / excluded features → EVENT_BASED
    if availability_label in {"POST_FIRE_ONLY", "EXCLUDE", "ADMIN_ONLY"}:
        return "EVENT_BASED", "Not applicable — feature excluded from production"

    # Keyword-based classification (source map keywords take priority if source
    # update frequency is already known)
    matched_categories: list[str] = []

    for category, rule in STATIC_DYNAMIC_RULES.items():
        for kw in rule["keywords"]:
            if kw.lower() in col_lower:
                matched_categories.append(category)
                break

    if matched_categories:
        # Return highest-priority match
        for priority_cat in _CATEGORY_PRIORITY:
            if priority_cat in matched_categories:
                desc = STATIC_DYNAMIC_RULES[priority_cat]["description"]
                return priority_cat, f"Keyword match — {desc}"

    # Fallback — use update_freq from source map
    if update_freq:
        freq_lower = update_freq.lower()
        if "daily" in freq_lower:
            return "DAILY", f"Source update frequency: {update_freq}"
        if "16-day" in freq_lower or "16 day" in freq_lower:
            return "DAILY", "MODIS 16-day composite — check daily for new composite"
        if "monthly" in freq_lower:
            return "MONTHLY", f"Source update frequency: {update_freq}"
        if "annual" in freq_lower or "year" in freq_lower:
            return "ANNUAL", f"Source update frequency: {update_freq}"
        if "static" in freq_lower or "5 year" in freq_lower or "2 year" in freq_lower:
            return "STATIC", f"Source update frequency: {update_freq}"
        if "event" in freq_lower or "post-fire" in freq_lower:
            return "EVENT_BASED", "Event-based — not for production prediction"

    return "DAILY", "Default — conservative daily assumption (verify)"


def generate_static_dynamic_classification(
    source_map_csv: Path,
    availability_csv: Path,
    output_dir: Path,
    state_name: str,
) -> pd.DataFrame:
    """
    Analysis D: Classify every feature by production update frequency.

    Parameters
    ----------
    source_map_csv   : Analysis A feature_source_map.csv
    availability_csv : Analysis C production_availability.csv
    output_dir       : Where to save output
    state_name       : 'Texas' or 'California'

    Returns
    -------
    pd.DataFrame  Static/dynamic classification (one row per column)
    """
    logger.info(f"Analysis D — Static vs Dynamic Classification [{state_name}]")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source map
    source_df = pd.DataFrame()
    if source_map_csv.exists():
        source_df = pd.read_csv(source_map_csv)

    # Load availability
    avail_df = pd.DataFrame()
    if availability_csv.exists():
        avail_df = pd.read_csv(availability_csv)

    if source_df.empty:
        logger.warning("  Source map not found — cannot classify")
        return pd.DataFrame()

    # Build availability lookup
    avail_lookup: dict[str, str] = {}
    if not avail_df.empty:
        avail_lookup = dict(zip(avail_df["Column"], avail_df["Availability_Label"]))

    rows = []
    cat_counts: dict[str, int] = {}

    for _, row in source_df.iterrows():
        col         = row["Column"]
        update_freq = row.get("Update_Frequency", "UNKNOWN")
        avail_label = avail_lookup.get(col)

        category, reason = _classify_update_frequency(col, update_freq, avail_label)
        cat_counts[category] = cat_counts.get(category, 0) + 1

        rows.append({
            "Column":             col,
            "Update_Category":    category,
            "Category_Desc":      STATIC_DYNAMIC_RULES.get(category, {}).get("description", "N/A"),
            "Source_Update_Freq": update_freq,
            "Classification_Basis": reason,
        })

    df = pd.DataFrame(rows)

    # Print summary
    print(f"\n{'=' * 65}")
    print(f"  ANALYSIS D — STATIC vs DYNAMIC [{state_name.upper()}]")
    print(f"{'=' * 65}")
    print(f"  {'Category':<15}  {'Count':>6}  Description")
    print(f"  {'-' * 62}")
    icons = {"STATIC": "🏔️ ", "ANNUAL": "📅", "MONTHLY": "🗓️ ", "DAILY": "⚡",
             "EVENT_BASED": "❌"}
    for cat in _CATEGORY_PRIORITY:
        cnt  = cat_counts.get(cat, 0)
        desc = STATIC_DYNAMIC_RULES.get(cat, {}).get("description", "Not applicable")
        icon = icons.get(cat, "  ")
        print(f"  {icon} {cat:<13}  {cnt:>6}  {desc}")
    print(f"  {'-' * 62}")
    print(f"  {'TOTAL':<15}  {len(df):>6}")
    print(f"{'=' * 65}\n")

    for cat, cnt in cat_counts.items():
        logger.info(f"  {cat}: {cnt}")

    # Save
    out_path = output_dir / "static_dynamic_features.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"  ✔ Saved: {out_path}")

    return df
