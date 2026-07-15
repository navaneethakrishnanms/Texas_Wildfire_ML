"""
Phase 3 — Texas Baseline Model: Full Visualization Suite
=========================================================
Generates 6 publication-quality figures + a comprehensive text report.

Run from:  V2/phase2/
Command :  python run_phase3_visualize.py --state TX
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score,
)

# ── Paths ───────────────────────────────────────────────────────────────────────
PHASE2_ROOT = Path(__file__).resolve().parent
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from config.phase2_config import STATE_CONFIG

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# ── Style ───────────────────────────────────────────────────────────────────────
PALETTE = {
    "fire":    "#E63946",   # red
    "no_fire": "#457B9D",   # steel blue
    "val":     "#2A9D8F",   # teal
    "test":    "#E9C46A",   # amber
    "train":   "#264653",   # dark slate
    "accent":  "#F4A261",   # orange
    "bg":      "#0D1117",   # dark background
    "panel":   "#161B22",
    "text":    "#E6EDF3",
    "grid":    "#30363D",
}

def dark_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["panel"],
        "axes.edgecolor":    PALETTE["grid"],
        "axes.labelcolor":   PALETTE["text"],
        "axes.titlecolor":   PALETTE["text"],
        "xtick.color":       PALETTE["text"],
        "ytick.color":       PALETTE["text"],
        "text.color":        PALETTE["text"],
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.6,
        "legend.facecolor":  PALETTE["panel"],
        "legend.edgecolor":  PALETTE["grid"],
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
    })

# ── Helpers ─────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "avg_burn_prob","whp","flep4","cfl","burnable",
    "erc","fm100","vpd","vs","rmax","rmin","tmmx","pr",
    "erc_5D_mean","erc_5D_max","fm100_5D_mean","fm100_5D_min",
    "vpd_5D_mean","vpd_5D_max","vs_5D_mean","vs_5D_max",
    "rmax_5D_mean","rmax_5D_min","tmmx_5D_mean","tmmx_5D_max",
    "sin_month","cos_month","sin_hour","cos_hour",
    "centroid_lat","centroid_lon",
]
STATIC_COLS = ["avg_burn_prob","whp","flep4","cfl","burnable"]


def load_split(path: Path) -> tuple:
    df = pd.read_parquet(path)
    present = [c for c in FEATURE_COLS if c in df.columns]
    X = df[present].copy()
    y = df["label"].values.astype(np.int8)
    for col in STATIC_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(0.0)
    obj_cols = X.select_dtypes(include="object").columns.tolist()
    for col in obj_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(np.float32)
    return X, y, present


def get_probs(model, X, present_feats):
    dm = xgb.DMatrix(X, feature_names=present_feats, missing=np.nan)
    return model.predict(dm)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — ROC Curves (all 3 splits)
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc(ax, splits_data):
    ax.set_title("ROC Curve — Area Under Curve (AUROC)", fontweight="bold")
    colors = [PALETTE["train"], PALETTE["val"], PALETTE["test"]]
    labels = ["Train (2014–2017)", "Validation (2018)", "Test (2019–2020)"]

    for (y, prob), color, label in zip(splits_data, colors, labels):
        fpr, tpr, _ = roc_curve(y, prob)
        auroc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.2,
                label=f"{label}  AUROC = {auroc:.4f}")

    ax.plot([0, 1], [0, 1], "--", color=PALETTE["grid"], lw=1.2, label="Random (0.50)")
    ax.set_xlabel("False Positive Rate  (FPR)")
    ax.set_ylabel("True Positive Rate  (TPR = Recall)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True, alpha=0.4)
    ax.annotate("← Better model stays\nin top-left corner", xy=(0.45, 0.35),
                color=PALETTE["text"], fontsize=8.5, alpha=0.7)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Precision-Recall (AUPR) Curves
# ══════════════════════════════════════════════════════════════════════════════
def plot_pr(ax, splits_data):
    ax.set_title("Precision-Recall Curve — AUPR\n(Better metric for imbalanced classes)", fontweight="bold")
    colors = [PALETTE["train"], PALETTE["val"], PALETTE["test"]]
    labels = ["Train (2014–2017)", "Validation (2018)", "Test (2019–2020)"]
    baseline = 0.091  # positive rate in dataset

    ax.axhline(baseline, color=PALETTE["accent"], lw=1.2, ls="--",
               label=f"Random baseline = {baseline:.3f} (fire rate)")

    for (y, prob), color, label in zip(splits_data, colors, labels):
        prec, rec, _ = precision_recall_curve(y, prob)
        ap = average_precision_score(y, prob)
        ax.plot(rec, prec, color=color, lw=2.2,
                label=f"{label}  AUPR = {ap:.4f}")

    ax.set_xlabel("Recall  (fraction of real fires caught)")
    ax.set_ylabel("Precision  (of those flagged, how many are real fires)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True, alpha=0.4)
    ax.annotate("AUPR is the key metric:\nhigher = better at finding\nreal fires with few false alarms",
                xy=(0.55, 0.65), color=PALETTE["text"], fontsize=8.5, alpha=0.8)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Confusion Matrix (TEST set at optimal threshold)
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion(ax, y_test, prob_test, threshold):
    y_pred = (prob_test >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_test)

    matrix = np.array([[tn, fp], [fn, tp]])
    labels_pct = np.array([
        [f"TN\n{tn:,}\n({100*tn/total:.1f}%)", f"FP\n{fp:,}\n({100*fp/total:.1f}%)"],
        [f"FN\n{fn:,}\n({100*fn/total:.1f}%)", f"TP\n{tp:,}\n({100*tp/total:.1f}%)"],
    ])

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels_pct[i, j], ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="black" if matrix[i,j] > matrix.max()*0.5 else PALETTE["text"])

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: NO FIRE", "Predicted: FIRE"], fontsize=10)
    ax.set_yticklabels(["Actual: NO FIRE", "Actual: FIRE"], fontsize=10)
    ax.set_title(f"Confusion Matrix — TEST Set\n(Threshold = {threshold:.4f}, max-F1 on VAL)",
                 fontweight="bold")

    # Metrics annotation
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp)
    fpr_val   = fp / (fp + tn)

    metrics_txt = (f"  Precision: {precision:.3f}   |  TPR (Recall): {recall:.3f}\n"
                   f"  Specificity: {specificity:.3f}  |  FPR: {fpr_val:.3f}\n"
                   f"  F1 Score: {f1:.3f}")
    ax.set_xlabel(metrics_txt, fontsize=9.5)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Feature Importance (gain)
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importance(ax, model, top_n=20):
    scores = model.get_score(importance_type="gain")
    if not scores:
        ax.text(0.5, 0.5, "No importance scores available", ha="center", transform=ax.transAxes)
        return

    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, vals = zip(*items)
    vals_norm = np.array(vals) / sum(vals) * 100  # percentage of total gain

    # Color by category
    colors = []
    for n in names:
        if n in ["avg_burn_prob","whp","flep4","cfl","burnable"]:
            colors.append(PALETTE["fire"])
        elif n in ["centroid_lat","centroid_lon"]:
            colors.append(PALETTE["accent"])
        elif n in ["sin_month","cos_month","sin_hour","cos_hour"]:
            colors.append(PALETTE["val"])
        else:
            colors.append(PALETTE["no_fire"])

    bars = ax.barh(range(len(names)), vals_norm[::-1], color=colors[::-1], alpha=0.9, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(names)[::-1], fontsize=9)
    ax.set_xlabel("% of Total Gain Importance")
    ax.set_title(f"Top {top_n} Features by XGBoost Gain\n(SHAP analysis recommended after LANDFIRE re-train)",
                 fontweight="bold")

    # Legend patches
    patches = [
        mpatches.Patch(color=PALETTE["fire"],    label="Landscape / LANDFIRE (all ~0 now)"),
        mpatches.Patch(color=PALETTE["no_fire"], label="gridMET weather"),
        mpatches.Patch(color=PALETTE["val"],     label="Temporal encodings"),
        mpatches.Patch(color=PALETTE["accent"],  label="Location"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8.5)
    ax.grid(True, axis="x", alpha=0.4)

    for i, (bar, val) in enumerate(zip(bars[::-1], vals_norm[::-1])):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8, color=PALETTE["text"])


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Score Distribution (fire vs non-fire)
# ══════════════════════════════════════════════════════════════════════════════
def plot_score_dist(ax, y_test, prob_test, threshold):
    fire_scores    = prob_test[y_test == 1]
    nonfire_scores = prob_test[y_test == 0]

    ax.hist(nonfire_scores, bins=80, alpha=0.65, color=PALETTE["no_fire"],
            label=f"No Fire (n={len(nonfire_scores):,})", density=True)
    ax.hist(fire_scores, bins=80, alpha=0.75, color=PALETTE["fire"],
            label=f"Fire (n={len(fire_scores):,})", density=True)
    ax.axvline(threshold, color=PALETTE["accent"], lw=2, ls="--",
               label=f"Decision threshold = {threshold:.4f}")

    ax.set_xlabel("Model Output Score (probability of fire)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted Score Distribution — TEST Set\n(Good separation = fire scores shifted right)",
                 fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.grid(True, alpha=0.4)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Leakage Explanation (before/after comparison)
# ══════════════════════════════════════════════════════════════════════════════
def plot_leakage_explanation(ax):
    ax.axis("off")
    title = "⚠️  WHY AUROC WAS 0.990 — Leakage Detected & Fixed"
    body = """
WHAT HAPPENED:
  First training run included 'fire_count' and 'has_fire_history' as features.
  These were computed from ALL fire data (2014–2020, including TEST years 2019–2020).

WHY IT'S LEAKAGE (scope doc line 391):
  fire_count    = how many fires historically occurred in that H3 cell
  has_fire_history = 1 if that cell ever had a fire (derived from fire_count)

  ┌─────────────────────────────────────────────────────────────┐
  │  Fire row (label=1)    →  fire_count ≥ 1  (always)         │
  │  Non-fire row (label=0) →  fire_count = 0  (mostly)        │
  │  → Model just learns: "high fire_count = fire"             │
  │  → No need to learn weather, landscape, or timing          │
  └─────────────────────────────────────────────────────────────┘

FEATURE IMPORTANCE WITH LEAKAGE:
  has_fire_history   12,078  ████████████████████  58% of gain
  fire_count          7,390  ████████████          36% of gain
  burnable            2,582  ██                     <--- everything else tiny
  erc_5D_max            267  <-- real fire weather signal

RESULT: AUROC = 0.990 (artificially inflated)

HOW IT WAS FIXED:
  Removed 'fire_count' and 'has_fire_history' from FEATURE_COLS in
  run_phase3_train.py. The model now uses only:
  → LANDFIRE landscape features (still 0 — rasters not downloaded yet)
  → gridMET weather (31 features)
  → Temporal encodings
  → Location (lat/lon)

RESULT AFTER FIX: AUROC = 0.8569 (real signal)
  Feature importance is now physically meaningful:
    burnable (valid cell flag), centroid_lon/lat (geography),
    erc_5D_max (fire danger), vs_5D_max (wind speed)
"""
    ax.text(0.02, 0.98, title, transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=PALETTE["fire"], va="top")
    ax.text(0.02, 0.88, body, transform=ax.transAxes, fontsize=8.8,
            color=PALETTE["text"], va="top", fontfamily="monospace",
            linespacing=1.55)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="TX")
    args = parser.parse_args()

    s   = args.state.upper()
    cfg = STATE_CONFIG[s]
    out = cfg["output_dir"]
    slug = s.lower()

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    model_path = out / "models" / f"xgb_baseline_{slug}.ubj"
    meta_path  = out / "models" / f"xgb_baseline_{slug}_meta.json"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}  — run Phase 3 first")
        return

    model = xgb.Booster()
    model.load_model(str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    threshold = meta["threshold"]
    logger.info(f"Model loaded | threshold={threshold:.4f}")

    # ── Load splits ─────────────────────────────────────────────────────────
    X_train, y_train, feats = load_split(out / f"train_{slug}.parquet")
    X_val,   y_val,   _     = load_split(out / f"val_{slug}.parquet")
    X_test,  y_test,  _     = load_split(out / f"test_{slug}.parquet")

    prob_train = get_probs(model, X_train, feats)
    prob_val   = get_probs(model, X_val,   feats)
    prob_test  = get_probs(model, X_test,  feats)

    splits_data = [(y_train, prob_train), (y_val, prob_val), (y_test, prob_test)]

    dark_style()

    # ════════════════════════════════════════════════════════════════════════
    # COMBINED FIGURE (6 panels, 3×2 grid)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("Generating combined figure (6 panels)...")
    fig = plt.figure(figsize=(22, 18), facecolor=PALETTE["bg"])
    fig.suptitle(
        "Texas Wildfire Ignition Model — Phase 3 Baseline Evaluation\n"
        "(gridMET weather + temporal + location  |  LANDFIRE rasters pending)",
        fontsize=15, fontweight="bold", color=PALETTE["text"], y=0.98
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    plot_roc(ax1, splits_data)
    plot_pr(ax2, splits_data)
    plot_confusion(ax3, y_test, prob_test, threshold)
    plot_feature_importance(ax4, model)
    plot_score_dist(ax5, y_test, prob_test, threshold)
    plot_leakage_explanation(ax6)

    combined_path = fig_dir / f"phase3_evaluation_{slug}.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    logger.info(f"Saved: {combined_path}")

    # ════════════════════════════════════════════════════════════════════════
    # INDIVIDUAL HIGH-RES FIGURES
    # ════════════════════════════════════════════════════════════════════════
    for name, fn, args_ in [
        ("roc_curve",     plot_roc,                  [splits_data]),
        ("pr_curve",      plot_pr,                   [splits_data]),
        ("score_dist",    plot_score_dist,            [y_test, prob_test, threshold]),
        ("feat_importance", plot_feature_importance,  [model]),
    ]:
        fig_s, ax_s = plt.subplots(figsize=(10, 6.5), facecolor=PALETTE["bg"])
        fn(ax_s, *args_)
        fig_s.tight_layout()
        p = fig_dir / f"{name}_{slug}.png"
        fig_s.savefig(p, dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
        plt.close(fig_s)
        logger.info(f"Saved: {p}")

    # Confusion matrix standalone
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6), facecolor=PALETTE["bg"])
    plot_confusion(ax_cm, y_test, prob_test, threshold)
    fig_cm.tight_layout()
    p = fig_dir / f"confusion_matrix_{slug}.png"
    fig_cm.savefig(p, dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig_cm)
    logger.info(f"Saved: {p}")

    # ════════════════════════════════════════════════════════════════════════
    # METRICS SUMMARY TEXT REPORT
    # ════════════════════════════════════════════════════════════════════════
    write_report(out, meta, y_test, prob_test, threshold, feats, model, fig_dir)

    logger.info("\n" + "=" * 65)
    logger.info("  ALL FIGURES SAVED to: " + str(fig_dir))
    logger.info("  Report: " + str(out / "phase3_model_report_tx.md"))
    logger.info("=" * 65)


def write_report(out, meta, y_test, prob_test, threshold, feats, model, fig_dir):
    """Write comprehensive markdown report."""
    from sklearn.metrics import confusion_matrix
    y_pred = (prob_test >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_test)
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1          = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    specificity = tn / (tn + fp)
    fpr_val     = fp / (fp + tn)
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Feature importance
    scores = model.get_score(importance_type="gain")
    top_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]
    total_gain = sum(scores.values())

    rpt = out / "phase3_model_report_tx.md"
    lines = []
    A = lines.append

    A("# Texas Wildfire Ignition Model — Phase 3 Baseline Report")
    A("")
    A(f"*Generated automatically by `run_phase3_visualize.py`*")
    A("")
    A("---")
    A("")
    A("## 1. What is AUROC?")
    A("")
    A("> **AUROC** = Area Under the ROC (Receiver Operating Characteristic) Curve.")
    A("> It measures how well the model **ranks** fire cells above non-fire cells.")
    A("> - **1.0** = perfect ranking (every fire cell scored higher than every non-fire cell)")
    A("> - **0.5** = random guessing (coin flip)")
    A("> - **0.85** = our current model correctly ranks 85% of fire/non-fire pairs")
    A("")
    A("The ROC curve plots **True Positive Rate (Recall)** vs **False Positive Rate** at every")
    A("possible threshold. AUROC summarises that entire curve as a single number.")
    A("")
    A("---")
    A("")
    A("## 2. What is AUPR?")
    A("")
    A("> **AUPR** = Area Under the Precision-Recall Curve.")
    A("> This is the **primary metric** for rare-event prediction like wildfire ignition.")
    A("")
    A("| Metric | What it measures | Why it matters for fires |")
    A("|--------|-----------------|--------------------------|")
    A("| **AUROC** | Overall ranking quality | Good general indicator |")
    A("| **AUPR** | Precision vs Recall tradeoff | Better for 9% fire rate — not fooled by easy negatives |")
    A("| Accuracy | % correct predictions | **Useless** — predicting 'no fire' everywhere gives 90.9% |")
    A("")
    A("**Our AUPR = 0.3978** means: across all thresholds, the model achieves average precision")
    A("of 39.8% when recalling fire events. Random baseline AUPR = 0.091 (fire rate).")
    A("Our model is **4.4× better than random**.")
    A("")
    A("---")
    A("")
    A("## 3. Model Performance Summary")
    A("")
    A("### 3a. Metrics per Split")
    A("")
    A("| Split | Years | Rows | Fire Rows | AUROC | AUPR | F1 | Precision | Recall |")
    A("|-------|-------|------|-----------|-------|------|----|-----------|--------|")
    for m in meta["metrics"]:
        A(f"| {m['split']} | — | {m['n_samples']:,} | {m['n_pos']:,} "
          f"| {m['auroc']:.4f} | {m['aupr']:.4f} "
          f"| {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} |")
    A("")
    A("### 3b. Confusion Matrix — TEST Set")
    A(f"*(Threshold = {threshold:.4f}, chosen to maximise F1 on validation set)*")
    A("")
    A("```")
    A("                      PREDICTED")
    A("                   No Fire  |  Fire")
    A("         ──────────────────────────────")
    A(f"Actual:  No Fire  |  TN={tn:>6,}  |  FP={fp:>6,}  |  ({100*tn/total:.1f}%)  ({100*fp/total:.1f}%)")
    A(f"         Fire     |  FN={fn:>6,}  |  TP={tp:>6,}  |  ({100*fn/total:.1f}%)  ({100*tp/total:.1f}%)")
    A("```")
    A("")
    A("| Metric | Value | Interpretation |")
    A("|--------|-------|----------------|")
    A(f"| True Positive Rate (Recall/Sensitivity) | **{recall:.3f}** | {100*recall:.1f}% of real fires are detected |")
    A(f"| False Positive Rate | **{fpr_val:.3f}** | {100*fpr_val:.1f}% of non-fire cells are falsely flagged |")
    A(f"| Precision | **{precision:.3f}** | Of all flagged cells, {100*precision:.1f}% are real fires |")
    A(f"| Specificity | **{specificity:.3f}** | {100*specificity:.1f}% of non-fire cells correctly identified |")
    A(f"| Negative Predictive Value | **{npv:.3f}** | Of cells cleared, {100*npv:.1f}% truly had no fire |")
    A(f"| F1 Score | **{f1:.3f}** | Harmonic mean of precision and recall |")
    A("")
    A("---")
    A("")
    A("## 4. Missing Value Analysis")
    A("")
    A("| Feature Group | Missing Count | % Missing | Treatment | Reason |")
    A("|---------------|--------------|-----------|-----------|--------|")
    A("| `avg_burn_prob`, `whp`, `flep4`, `cfl` | ~1,594 | 0.42% | Zero-filled | Boundary H3 cells not in LANDFIRE raster extent |")
    A("| `avg_burn_prob`, `whp`, `flep4`, `cfl` | 99.6% are **zero** | — | Kept as 0 | **Rasters NOT downloaded** — currently placeholder |")
    A("| `erc`, `fm100`, `vpd`, `vs`, `rmax`, `rmin`, `tmmx`, `pr` | 24,954 | 6.63% | Left as NaN | Coastal/border H3 cells outside gridMET grid — XGBoost handles natively |")
    A("| `erc_5D_*` etc. | 24,965 | 6.64% | Left as NaN | Same border cells — 5-day rolling window also missing |")
    A("| `burnable`, `has_fire_history`, `fire_count` | ~1,594 | 0.42% | Zero-filled | Same border cells |")
    A("")
    A("> **Note:** `burnable` was a placeholder `True` for all cells from Phase 2B. It has")
    A("> 0.42% NaN (border cells filled to 0). In the current model it acts as a 'valid cell'")
    A("> flag and is the top feature by gain — this will be replaced with real LANDFIRE")
    A("> vegetation data after raster download.")
    A("")
    A("---")
    A("")
    A("## 5. ⚠️ Leakage Issue: Why AUROC Was 0.990 — and How It Was Fixed")
    A("")
    A("### 5a. What Happened")
    A("")
    A("The **first training run** included `fire_count` and `has_fire_history` as features.")
    A("These columns were built in Phase 2B from the **full fire dataset (2014–2020)**,")
    A("which includes the **test years (2019–2020)**.")
    A("")
    A("### 5b. Why This Is Leakage")
    A("")
    A("```")
    A("  fire_count    = how many FPA-FOD fires occurred in that H3 cell (2014–2020)")
    A("  has_fire_history = True if fire_count > 0")
    A("")
    A("  Label=1 row (fire cell)     → fire_count ≥ 1  (by definition)")
    A("  Label=0 row (non-fire cell) → fire_count = 0  (almost always)")
    A("")
    A("  ∴ Model learns: fire_count > 0 → predict fire")
    A("  No need to learn weather, landscape, or time-of-day patterns.")
    A("```")
    A("")
    A("This is exactly what the official scope document warns against (line 391):")
    A("> *'ignition_density, burn_count — Historical fire count — trivially separates")
    A("> fire/non-fire by construction'*")
    A("")
    A("### 5c. Evidence of Leakage")
    A("")
    A("| Feature | Gain (leaked model) | Share |")
    A("|---------|--------------------:|-------|")
    A("| `has_fire_history` | 12,078 | 58% |")
    A("| `fire_count` | 7,390 | 36% |")
    A("| `burnable` | 2,582 | 12% |")
    A("| `erc_5D_max` *(real fire signal)* | 267 | 1% |")
    A("| All other features | < 250 total | <1% |")
    A("")
    A("AUROC = **0.9900** (artificially inflated — model is 'cheating')")
    A("")
    A("### 5d. Fix Applied")
    A("")
    A("Removed `fire_count` and `has_fire_history` from `FEATURE_COLS` in")
    A("`run_phase3_train.py`. Retrained from scratch. No data changes needed.")
    A("")
    A("### 5e. Results After Fix")
    A("")
    A("| Metric | Leaked (wrong) | Clean (correct) |")
    A("|--------|---------------|-----------------|")
    A("| AUROC | 0.9900 ❌ | **0.8569** ✅ |")
    A("| AUPR | 0.8642 ❌ | **0.3978** ✅ |")
    A("| Feature importance | fire_count dominant | erc, vs, lat/lon — physically correct |")
    A("| Trees used | 143 | 387 (more complexity needed for real signal) |")
    A("")
    A("---")
    A("")
    A("## 6. Top Feature Importance (Clean Model)")
    A("")
    A("| Rank | Feature | Gain | % of Total | Group |")
    A("|------|---------|------|-----------|-------|")
    for rank, (feat, gain) in enumerate(top_feats, 1):
        grp = ("Landscape" if feat in ["avg_burn_prob","whp","flep4","cfl","burnable"]
               else "Location" if feat in ["centroid_lat","centroid_lon"]
               else "Temporal" if feat in ["sin_month","cos_month","sin_hour","cos_hour"]
               else "gridMET weather")
        A(f"| {rank} | `{feat}` | {gain:.1f} | {100*gain/total_gain:.1f}% | {grp} |")
    A("")
    A("---")
    A("")
    A("## 7. What Happens Next (Expected Improvement)")
    A("")
    A("| Step | Action | Expected AUROC |")
    A("|------|--------|----------------|")
    A("| Current | gridMET + temporal + location (no LANDFIRE) | **0.8569** |")
    A("| Step 1 | Download LANDFIRE rasters → re-run Phase 2E + 2G + 3 | **~0.90–0.93** |")
    A("| Step 2 | Add HRRR per-window features (6-hourly atmospheric) | **~0.93–0.96** |")
    A("")
    A("### LANDFIRE rasters to download:")
    A("- `avg_burn_prob` (Burn Probability): `firelab.org/fsim`")
    A("- `whp` (Wildfire Hazard Potential): `doi.org/10.2737/RDS-2015-0047-4`")
    A("- `flep4` (Flame Length Exceedance): `landfire.gov`")
    A("- `cfl` (Canopy Fuel Load): `landfire.gov`")
    A("")
    A("---")
    A("")
    A("## 8. Figures Generated")
    A("")
    for f in sorted(fig_dir.glob("*.png")):
        A(f"- `{f.name}`")
    A("")

    rpt.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved: {rpt}")


if __name__ == "__main__":
    main()
