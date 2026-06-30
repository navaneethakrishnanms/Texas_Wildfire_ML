"""
explainability.py
=================
SHAP-based model explainability for the Texas Wildfire ML pipeline.

Purpose
-------
SHAP (SHapley Additive exPlanations) verifies that the model is learning
REAL wildfire risk patterns from environmental features rather than
exploiting data artifacts or spurious correlations.

Expected top SHAP features (physically meaningful):
  1. NDVI / EVI       -- vegetation dryness (dry fuel = high fire risk)
  2. LST / Temperature -- heat (high LST = fire-prone conditions)
  3. Wind              -- fire spread driver
  4. sin/cos month     -- seasonal risk encoding
  5. Rainfall          -- fuel moisture proxy

Red flags (model may be cheating):
  - latitude/longitude in top features → model memorised locations
  - Temporal features dominate, NDVI near zero → season proxy only
  - LandCover dominates with no weather features → overly simplified

Outputs
-------
  outputs/plots/shap_summary_{model}.png      -- beeswarm (all features)
  outputs/plots/shap_bar_{model}.png          -- mean |SHAP| bar chart
  outputs/plots/shap_dep_{feature}_{model}.png -- top 5 dependence plots
  outputs/shap_importance_{model}.csv         -- SHAP importance table
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_shap_analysis(
    model: Any,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    model_name: str,
    out_dir: Path,
    background_samples: int = 500,
    val_samples: int = 500,
    expected_top: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run full SHAP analysis for any tree-based model.

    Supports: XGBClassifier, RandomForestClassifier, LGBMClassifier.
    Uses TreeExplainer (fast exact SHAP for tree models).

    Parameters
    ----------
    model            : fitted classifier
    X_train          : training features (for background distribution)
    X_val            : validation features (for SHAP computation)
    model_name       : name for file prefixes and titles
    out_dir          : directory to save plots and CSV
    background_samples: random sample of X_train for TreeExplainer background
    val_samples      : random sample of X_val for SHAP computation
    expected_top     : list of feature names expected to be in top SHAP features

    Returns
    -------
    pd.DataFrame: feature importance table sorted by mean |SHAP|
    """
    try:
        import shap
    except ImportError:
        logger.error(
            "shap not installed. Run: pip install shap\n"
            "Skipping SHAP analysis."
        )
        return pd.DataFrame()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    # ── 1. Sample background and explanation sets ──────────────────────────
    rng = np.random.default_rng(42)
    bg_idx  = rng.choice(len(X_train), size=min(background_samples, len(X_train)), replace=False)
    val_idx = rng.choice(len(X_val),   size=min(val_samples,        len(X_val)),   replace=False)

    X_bg  = X_train.iloc[bg_idx].reset_index(drop=True)
    X_exp = X_val.iloc[val_idx].reset_index(drop=True)

    logger.info(
        "SHAP TreeExplainer: background=%d samples, explain=%d val samples",
        len(X_bg), len(X_exp)
    )

    # ── 2. Build explainer ─────────────────────────────────────────────────
    # model_output="probability" works for XGBoost & LightGBM.
    # Random Forest: TreeExplainer must use default (raw margin/log-odds).
    model_type = type(model).__name__
    is_xgb_lgbm = any(n in model_type for n in ("XGB", "LGBM", "Booster"))
    if is_xgb_lgbm:
        explainer   = shap.TreeExplainer(model, data=X_bg, model_output="probability")
    else:
        # RandomForestClassifier: use default — returns raw SHAP values
        explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_exp, check_additivity=False)

    # For binary classification: SHAP values for class 1 (Fire)
    # shap_values.values shape: (n_samples, n_features) or (n_samples, n_features, 2)
    if shap_values.values.ndim == 3:
        sv = shap_values.values[:, :, 1]   # class 1 column
    else:
        sv = shap_values.values

    feature_names = list(X_exp.columns)

    # ── 3. Feature importance table ────────────────────────────────────────
    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = range(1, len(importance_df) + 1)

    csv_path = out_dir / f"shap_importance_{safe_name}.csv"
    importance_df.to_csv(csv_path, index=False)
    logger.info("SHAP importance saved: %s", csv_path)

    # ── 4. Summary beeswarm plot ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv, X_exp,
        feature_names=feature_names,
        show=False, plot_type="dot",
        max_display=18,
    )
    plt.title(f"SHAP Feature Impact — {model_name}", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    beeswarm_path = out_dir / f"shap_summary_{safe_name}.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP beeswarm saved: %s", beeswarm_path)

    # ── 5. Bar chart (mean |SHAP|) ─────────────────────────────────────────
    top_n = min(18, len(feature_names))
    top_df = importance_df.head(top_n)

    colors = []
    good_features = {"NDVI", "EVI", "LST", "Temperature", "Wind",
                     "Rainfall", "sin_month", "cos_month", "sin_doy", "cos_doy",
                     "DEM", "Slope", "month", "day_of_year", "is_peak_fire_season"}
    for feat in top_df["feature"]:
        colors.append("#2ECC71" if feat in good_features else "#E74C3C")

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        top_df["feature"][::-1],
        top_df["mean_abs_shap"][::-1],
        color=colors[::-1], edgecolor="white", height=0.7,
    )
    ax.set_xlabel("Mean |SHAP value|  (impact on model output)", fontsize=11)
    ax.set_title(f"Feature Importance (SHAP) — {model_name}", fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ECC71", label="Physically meaningful feature"),
        Patch(facecolor="#E74C3C", label="Potential data artifact"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    bar_path = out_dir / f"shap_bar_{safe_name}.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    logger.info("SHAP bar chart saved: %s", bar_path)

    # ── 6. Dependence plots for top 5 features ─────────────────────────────
    top5 = importance_df["feature"].head(5).tolist()
    for feat in top5:
        if feat not in X_exp.columns:
            continue
        feat_idx = list(X_exp.columns).index(feat)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            X_exp[feat], sv[:, feat_idx],
            alpha=0.4, s=10, c="#3498DB", edgecolors="none"
        )
        ax.axhline(0, color="gray", lw=1, linestyle="--")
        ax.set_xlabel(feat, fontsize=12)
        ax.set_ylabel(f"SHAP value for {feat}", fontsize=12)
        ax.set_title(
            f"SHAP Dependence: {feat} — {model_name}",
            fontsize=12, fontweight="bold"
        )
        ax.grid(alpha=0.3)
        plt.tight_layout()
        dep_path = out_dir / f"shap_dep_{feat.lower()}_{safe_name}.png"
        fig.savefig(dep_path, dpi=150)
        plt.close(fig)
        logger.info("SHAP dependence plot saved: %s", dep_path)

    # ── 7. Verdict ─────────────────────────────────────────────────────────
    print_shap_verdict(importance_df, model_name, expected_top)

    return importance_df


def print_shap_verdict(
    importance_df: pd.DataFrame,
    model_name: str,
    expected_top: Optional[List[str]] = None,
) -> None:
    """
    Print the SHAP feature ranking and a pass/fail verdict.

    Checks whether environmental features dominate the top positions,
    confirming the model learned real wildfire patterns.
    """
    if expected_top is None:
        expected_top = ["NDVI", "EVI", "LST", "Temperature", "Wind",
                        "Rainfall", "sin_month", "cos_month"]

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  SHAP ANALYSIS VERDICT — {model_name}")
    print(sep)
    print(f"\n  {'Rank':<6} {'Feature':<25} {'Mean |SHAP|':>12}")
    print(f"  {'-'*6} {'-'*25} {'-'*12}")

    top10 = importance_df.head(10)
    for _, row in top10.iterrows():
        mark = " ✓" if row["feature"] in expected_top else " ?"
        print(f"  {int(row['rank']):<6} {row['feature']:<25} {row['mean_abs_shap']:>12.5f}{mark}")

    # Check: are at least 3 of top 5 expected features?
    top5_features   = set(importance_df["feature"].head(5).tolist())
    expected_in_top5 = len(top5_features & set(expected_top))

    print()
    if expected_in_top5 >= 3:
        print(f"  [VERDICT: PASS] {expected_in_top5}/5 top features are physically meaningful.")
        print(f"  The model is learning real wildfire risk patterns from the environment.")
    elif expected_in_top5 >= 2:
        print(f"  [VERDICT: MARGINAL] {expected_in_top5}/5 top features are meaningful.")
        print(f"  Review dependence plots. Consider adding more environmental features.")
    else:
        print(f"  [VERDICT: WARNING] Only {expected_in_top5}/5 top features are meaningful.")
        print(f"  The model may be relying on temporal proxies or location memorization.")
        print(f"  Do NOT deploy until root cause is investigated.")
    print(sep)
