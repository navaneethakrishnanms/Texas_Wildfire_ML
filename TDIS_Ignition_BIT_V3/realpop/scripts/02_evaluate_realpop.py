"""
TDIS_Ignition_BIT_V3/realpop -- concatenate all per-date scored panels from
01_pull_score_days.py and compute the TRUE real-population metrics:
AUROC, AUC-PR, F1, Precision, Recall, Lift, confusion matrices, and the
same 9-plot visualization suite used for the matched-sample model, but
now at the true ~1-3% base rate instead of the ~30% training rate.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score, brier_score_loss
)
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parent.parent.parent
REALPOP = ROOT / "realpop"
DAYS_DIR = REALPOP / "data" / "scored_days"
RESULTS_DIR = REALPOP / "results"
FIG_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

def metrics_block(y, p):
    aucpr = average_precision_score(y, p)
    auroc = roc_auc_score(y, p)
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    best_i = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
    base_rate = y.mean()
    return {
        "aucpr": float(aucpr), "auroc": float(auroc), "f1": float(f1[best_i]),
        "precision": float(prec[best_i]), "recall": float(rec[best_i]),
        "best_threshold": float(thr[best_i]) if best_i < len(thr) else 0.5,
        "base_rate": float(base_rate), "lift": float(aucpr / base_rate),
    }

def main():
    files = sorted(DAYS_DIR.glob("*.parquet"))
    assert files, f"No scored-day files found in {DAYS_DIR} -- run 01_pull_score_days.py first"
    print(f"Found {len(files)} scored-day files")

    panel = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    panel.to_parquet(RESULTS_DIR / "realpop_panel_all_days.parquet", index=False)
    print(f"Panel shape: {panel.shape}")
    print(f"Dates covered: {sorted(panel.date.dt.date.unique().tolist())}")
    print(f"Total positives: {panel.label.sum()}  base rate: {panel.label.mean():.5f}")

    y = panel["label"].values
    p_raw = panel["p_raw"].values
    p_cal = panel["p_cal"].values

    m_raw = metrics_block(y, p_raw)
    m_cal = metrics_block(y, p_cal)
    brier_raw = brier_score_loss(y, p_raw)
    brier_cal = brier_score_loss(y, p_cal)

    print("\n=== REAL-POPULATION METRICS ===")
    print(f"n = {len(y):,}   base rate = {y.mean():.5f}")
    print(f"RAW:        AUROC={m_raw['auroc']:.4f}  AUC-PR={m_raw['aucpr']:.4f}  "
          f"F1={m_raw['f1']:.4f}  Lift={m_raw['lift']:.2f}x")
    print(f"CALIBRATED: AUROC={m_cal['auroc']:.4f}  AUC-PR={m_cal['aucpr']:.4f}  "
          f"F1={m_cal['f1']:.4f}  Lift={m_cal['lift']:.2f}x")
    print(f"Brier raw={brier_raw:.5f} -> calibrated={brier_cal:.5f}")

    def cm_at(pp, thr):
        pred = (pp >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        return {"threshold": float(thr), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
                "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
                "accuracy": float((tp + tn) / len(y))}

    cm_raw_bestf1 = cm_at(p_raw, m_raw["best_threshold"])
    cm_cal_bestf1 = cm_at(p_cal, m_cal["best_threshold"])
    print("\nConfusion matrix (raw, best-F1 thr):", cm_raw_bestf1)
    print("Confusion matrix (calibrated, best-F1 thr):", cm_cal_bestf1)

    # threshold sweep (operationally meaningful at real base rate)
    sweep = []
    for t in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        c = cm_at(p_cal, t)
        f1 = 2 * c["precision"] * c["recall"] / (c["precision"] + c["recall"] + 1e-12)
        sweep.append({**c, "f1": f1, "flagged": c["tp"] + c["fp"]})
        print(f"  cal thr={t:.2f}  P={c['precision']:.4f}  R={c['recall']:.4f}  "
              f"F1={f1:.4f}  flagged={c['tp']+c['fp']:,}")

    results = {
        "built": pd.Timestamp.now().isoformat(),
        "n_dates": len(files), "dates": sorted([str(d) for d in panel.date.dt.date.unique()]),
        "n_rows": len(panel), "base_rate": float(y.mean()),
        "metrics_raw": m_raw, "metrics_calibrated": m_cal,
        "brier_raw": float(brier_raw), "brier_calibrated": float(brier_cal),
        "confusion_matrix_raw_best_f1": cm_raw_bestf1,
        "confusion_matrix_calibrated_best_f1": cm_cal_bestf1,
        "threshold_sweep_calibrated": sweep,
    }
    with open(RESULTS_DIR / "realpop_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ---------------- Visuals ----------------
    print("\nGenerating visuals ...")
    fpr, tpr, _ = roc_curve(y, p_raw)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, color="#1f77b4", label=f"AUROC = {m_raw['auroc']:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", ls=":", label="Random (0.5)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC -- REAL POPULATION (n={len(y):,}, base rate={y.mean():.3%})")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(FIG_DIR / "01_roc_curve_realpop.png"); plt.close(fig)

    prec, rec, _ = precision_recall_curve(y, p_raw)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec, prec, lw=2, color="#1f77b4", label=f"AUC-PR = {m_raw['aucpr']:.4f}")
    ax.axhline(y.mean(), color="gray", ls=":", label=f"Base rate = {y.mean():.3%}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall -- REAL POPULATION")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "02_pr_curve_realpop.png"); plt.close(fig)

    def plot_cm(cm, title, path):
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(mat, cmap="Blues")
        labels_ = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels_[i][j]}\n{mat[i,j]:,}", ha="center", va="center",
                        fontsize=12, color="white" if mat[i, j] > mat.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred No Fire", "Pred Fire"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual No Fire", "Actual Fire"])
        ax.set_title(f"{title}\nthr={cm['threshold']:.4f}  P={cm['precision']:.4f}  "
                     f"R={cm['recall']:.4f}  Acc={cm['accuracy']:.4f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout(); fig.savefig(path); plt.close(fig)

    plot_cm(cm_raw_bestf1, "Confusion Matrix -- REAL POPULATION (raw, best-F1)",
            FIG_DIR / "03_confusion_matrix_realpop_raw.png")
    plot_cm(cm_cal_bestf1, "Confusion Matrix -- REAL POPULATION (calibrated, best-F1)",
            FIG_DIR / "04_confusion_matrix_realpop_calibrated.png")

    frac_pos, mean_pred = calibration_curve(y, p_cal, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="gray", ls=":", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color="#ff7f0e", label="Calibrated (BIT V3)")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fire rate")
    ax.set_title("Reliability Diagram -- REAL POPULATION")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "05_reliability_realpop.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(p_raw[y == 0], bins=60, alpha=0.6, label="Actual No Fire", color="#1f77b4", density=True)
    ax.hist(p_raw[y == 1], bins=60, alpha=0.6, label="Actual Fire", color="#d62728", density=True)
    ax.set_xlabel("Raw predicted probability"); ax.set_ylabel("Density")
    ax.set_title("Prediction Distribution -- REAL POPULATION")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "06_prediction_distribution_realpop.png"); plt.close(fig)

    print(f"\nSaved real-population results and 6 figures to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
