"""
evaluator.py
============
Evaluation utilities for the Texas Wildfire ML pipeline.

Covers:
  - AUC-ROC, AUC-PR, F1, Precision, Recall, Confusion Matrix
  - Threshold sweep to find optimal recall >= target
  - PR curve and ROC curve plots
  - Comparison table across all models
  - Final test-set evaluation for best model
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    label: str = "",
) -> Dict[str, float]:
    """
    Compute full classification metrics at a given probability threshold.

    Parameters
    ----------
    y_true   : ground truth labels (0/1)
    y_prob   : predicted probabilities for class 1
    threshold: decision cutoff (default 0.5)
    label    : name for logging

    Returns
    -------
    dict with: auc_roc, auc_pr, f1, precision, recall,
               accuracy, threshold, tp, fp, tn, fn
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score,
        accuracy_score, confusion_matrix,
    )

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "auc_roc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "auc_pr":    round(float(average_precision_score(y_true, y_prob)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "threshold": round(threshold, 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }

    if label:
        logger.info(
            "[%s] @ thresh=%.2f | AUC-ROC=%.4f | AUC-PR=%.4f | "
            "F1=%.4f | Prec=%.4f | Recall=%.4f | "
            "TP=%d FP=%d TN=%d FN=%d",
            label, threshold,
            metrics["auc_roc"], metrics["auc_pr"],
            metrics["f1"], metrics["precision"], metrics["recall"],
            tp, fp, tn, fn
        )
    return metrics


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.85,
    step: float = 0.01,
    label: str = "",
) -> Tuple[float, Dict[str, float]]:
    """
    Sweep thresholds from high to low and find the highest threshold
    where Recall(Fire=1) >= target_recall.

    This maximises precision (minimises false alarms) while guaranteeing
    the target recall (catching enough real fires).

    Parameters
    ----------
    y_true        : ground truth (0/1)
    y_prob        : predicted probabilities
    target_recall : minimum acceptable recall for Fire=1 (default 0.85)
    step          : threshold sweep step size

    Returns
    -------
    (optimal_threshold, metrics_at_optimal)
    """
    from sklearn.metrics import recall_score, precision_score, f1_score

    thresholds = np.arange(0.90, 0.05, -step)
    best_threshold = 0.50
    best_metrics   = None

    logger.info(
        "Threshold tuning for %s: target Recall(Fire=1) >= %.2f",
        label or "model", target_recall
    )

    results = []
    for thr in thresholds:
        y_pred   = (y_prob >= thr).astype(int)
        recall   = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1       = f1_score(y_true, y_pred, zero_division=0)
        results.append((thr, recall, precision, f1))

        if recall >= target_recall:
            best_threshold = thr
            break   # highest threshold that satisfies recall target

    # Print sweep summary (every 5th step)
    logger.info(
        "  %-10s %-10s %-12s %-10s",
        "Threshold", "Recall", "Precision", "F1"
    )
    for thr, rec, prec, f1 in results:
        flag = " <-- SELECTED" if abs(thr - best_threshold) < 1e-6 else ""
        logger.info("  %-10.2f %-10.4f %-12.4f %-10.4f%s", thr, rec, prec, f1, flag)

    best_metrics = compute_metrics(y_true, y_prob, threshold=best_threshold, label=f"{label} [tuned]")
    logger.info(
        "Optimal threshold: %.2f  ->  Recall=%.4f  Precision=%.4f  F1=%.4f",
        best_threshold, best_metrics["recall"], best_metrics["precision"], best_metrics["f1"]
    )
    return best_threshold, best_metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pr_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    out_dir: Path,
    threshold: float = 0.50,
) -> None:
    """Save PR curve and ROC curve plots to out_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        precision_recall_curve, roc_curve,
        average_precision_score, roc_auc_score,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── PR Curve ──────────────────────────────────────────────────────
    prec, rec, thr_pr = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, lw=2, color="#E74C3C", label=f"PR Curve (AUC-PR={ap:.4f})")
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Random baseline (25%)")
    ax.axvline(x=0.85, color="#3498DB", linestyle=":", alpha=0.7, label="Target Recall=0.85")

    # Mark the selected threshold
    if len(thr_pr) > 0:
        idx = np.searchsorted(thr_pr[::-1], threshold)
        idx = min(idx, len(thr_pr) - 1)
        ax.scatter(rec[-(idx+1)], prec[-(idx+1)], s=120, zorder=5,
                   color="#F39C12", label=f"Threshold={threshold:.2f}")

    ax.set_xlabel("Recall (Fire=1)", fontsize=12)
    ax.set_ylabel("Precision (Fire=1)", fontsize=12)
    ax.set_title(f"Precision–Recall Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = out_dir / f"pr_curve_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    logger.info("PR curve saved: %s", pr_path)

    # ── ROC Curve ─────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, color="#2ECC71", label=f"ROC Curve (AUC={auc_roc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = out_dir / f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved: %s", roc_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_dir: Path,
) -> None:
    """Save a styled confusion matrix plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred: No Fire", "Pred: Fire"],
        yticklabels=["True: No Fire", "True: Fire"],
        ax=ax, annot_kws={"size": 14}
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.tight_layout()
    path = out_dir / f"cm_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", path)


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print comparison table across all models."""
    sep = "=" * 95
    print(f"\n{sep}")
    print("  MODEL COMPARISON — VALIDATION SET")
    print(sep)
    print(f"  {'Model':<30} {'AUC-ROC':>9} {'AUC-PR':>8} {'F1':>8} {'Recall':>9} {'Prec':>9} {'Thresh':>8}")
    print(f"  {'-'*30} {'-'*9} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*8}")

    sorted_models = sorted(results.items(), key=lambda x: x[1].get("auc_roc", 0), reverse=True)
    best_name = sorted_models[0][0] if sorted_models else ""

    for name, m in sorted_models:
        star = " ★" if name == best_name else ""
        print(
            f"  {name+star:<30} "
            f"{m.get('auc_roc',0):>9.4f} "
            f"{m.get('auc_pr',0):>8.4f} "
            f"{m.get('f1',0):>8.4f} "
            f"{m.get('recall',0):>9.4f} "
            f"{m.get('precision',0):>9.4f} "
            f"{m.get('threshold',0.5):>8.2f}"
        )

    print(sep)
    if best_name:
        print(f"\n  Best model (by AUC-ROC): {best_name}")
        print(f"  This model proceeds to Model E (threshold tuning + SHAP).")
    print()


def final_test_report(
    model_name: str,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> None:
    """Print the final held-out test set evaluation report."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  FINAL TEST SET REPORT — {model_name}")
    print(sep)
    print(f"\n  {'Metric':<20} {'Val':>10}  {'Test':>10}  {'Gap':>10}")
    print(f"  {'-'*20} {'-'*10}  {'-'*10}  {'-'*10}")

    for key in ["auc_roc", "auc_pr", "f1", "recall", "precision"]:
        v = val_metrics.get(key, 0)
        t = test_metrics.get(key, 0)
        gap = t - v
        flag = "  <- OVERFIT" if abs(gap) > 0.05 else ""
        print(f"  {key:<20} {v:>10.4f}  {t:>10.4f}  {gap:>+10.4f}{flag}")

    print(f"\n  Threshold used  : {test_metrics.get('threshold', 0.5):.2f}")
    print(f"  TP={test_metrics.get('tp',0)}  FP={test_metrics.get('fp',0)}  "
          f"TN={test_metrics.get('tn',0)}  FN={test_metrics.get('fn',0)}")

    fn = test_metrics.get("fn", 0)
    tp = test_metrics.get("tp", 0)
    print(f"\n  In production terms:")
    print(f"    Fires CAUGHT  : {tp}  (evacuations triggered correctly)")
    print(f"    Fires MISSED  : {fn}  (missed detections — minimize this!)")
    print(f"    False alarms  : {test_metrics.get('fp',0)}")
    print(sep)
