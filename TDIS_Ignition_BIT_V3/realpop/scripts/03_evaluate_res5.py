"""
TDIS_Ignition_BIT_V3/realpop -- res-5 aggregated real-population evaluation.

TDIS reports its real-population numbers at H3 res-5 (mean predicted
probability per res-5 cell/date; label = 1 if ANY underlying res-8 cell had
a real ignition). This reproduces that convention on our scored panel so the
comparison against TDIS (AUROC 0.7968, AUC-PR 0.0878, lift 4.58x, base 1.92%)
is apples-to-apples. Run after 02_evaluate_realpop.py.
"""
import json
import numpy as np
import pandas as pd
import h3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (roc_curve, precision_recall_curve, average_precision_score,
                             roc_auc_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "realpop" / "results"
FIG = RES / "figures"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

TDIS = {"base_rate": 0.0192, "auroc": 0.7968, "aucpr": 0.0878, "lift": 4.58,
        "f1": 0.1692, "precision": 0.1493, "recall": 0.1851}


def block(y, p):
    aucpr = average_precision_score(y, p)
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    i = int(np.nanargmax(f1[:-1]))
    return {"aucpr": float(aucpr), "auroc": float(roc_auc_score(y, p)), "f1": float(f1[i]),
            "precision": float(prec[i]), "recall": float(rec[i]), "best_threshold": float(thr[i]),
            "base_rate": float(y.mean()), "lift": float(aucpr / y.mean())}


def main():
    panel = pd.read_parquet(RES / "realpop_panel_all_days.parquet")
    cells = panel["h3_cell"].unique()
    parent = pd.Series({c: h3.cell_to_parent(c, 5) for c in cells})
    panel["h3_5"] = panel["h3_cell"].map(parent)
    agg = panel.groupby(["h3_5", "date"], sort=False).agg(
        p_raw=("p_raw", "mean"), p_cal=("p_cal", "mean"), label=("label", "max")).reset_index()
    y = agg["label"].values
    print(f"res-5 cell-days: {len(agg):,}  positives: {int(y.sum())}  base rate: {y.mean():.4%}")

    m_raw, m_cal = block(y, agg["p_raw"].values), block(y, agg["p_cal"].values)

    def cm_at(p, t):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        return {"threshold": float(t), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)),
                "accuracy": float((tp + tn) / len(y))}

    cm = cm_at(agg["p_raw"].values, m_raw["best_threshold"])
    out = {"n_rows": len(agg), "n_positive": int(y.sum()), "base_rate": float(y.mean()),
           "metrics_raw": m_raw, "metrics_calibrated": m_cal,
           "confusion_matrix_raw_best_f1": cm, "tdis_served_model_realpop": TDIS}
    json.dump(out, open(RES / "realpop_res5_aggregated.json", "w"), indent=2)
    print(json.dumps(out, indent=1))

    p = agg["p_raw"].values
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"BIT V3 27-feat  AUROC = {m_raw['auroc']:.4f}")
    ax.plot([0, 1], [0, 1], ":", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC -- REAL POPULATION, res-5 (n={len(y):,}, base {y.mean():.2%})\n"
                 f"TDIS served model AUROC = {TDIS['auroc']}")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(FIG / "07_roc_curve_res5.png"); plt.close(fig)

    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec, prec, lw=2, label=f"BIT V3 27-feat  AUC-PR = {m_raw['aucpr']:.4f}")
    ax.axhline(y.mean(), color="gray", ls=":", label=f"Base rate = {y.mean():.2%}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall -- REAL POPULATION, res-5\nTDIS served model AUC-PR = {TDIS['aucpr']}")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "08_pr_curve_res5.png"); plt.close(fig)

    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(mat, cmap="Blues")
    names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i][j]}\n{mat[i, j]:,}", ha="center", va="center", fontsize=12,
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred No Fire", "Pred Fire"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual No Fire", "Actual Fire"])
    ax.set_title(f"Confusion Matrix -- REAL POPULATION, res-5 (raw, best-F1)\nthr={cm['threshold']:.4f}  "
                 f"P={cm['precision']:.4f}  R={cm['recall']:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(FIG / "09_confusion_matrix_res5.png"); plt.close(fig)

    # head-to-head bars vs TDIS served model
    names_ = ["AUROC", "AUC-PR", "Lift (x)", "Best F1"]
    ours = [m_raw["auroc"], m_raw["aucpr"], m_raw["lift"], m_raw["f1"]]
    theirs = [TDIS["auroc"], TDIS["aucpr"], TDIS["lift"], TDIS["f1"]]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.8))
    for ax, n, a, b in zip(axs, names_, ours, theirs):
        ax.bar(["BIT V3\n27-feat", "TDIS\n23-feat"], [a, b], color=["#1f77b4", "#7f7f7f"])
        for k, v in enumerate([a, b]):
            ax.text(k, v, f"{v:.3f}" if v < 10 else f"{v:.2f}", ha="center", va="bottom")
        ax.set_title(n); ax.grid(axis="x", alpha=0)
    fig.suptitle("Real-population comparison at res-5 (TDIS: 941 days; ours: sampled dates)")
    fig.tight_layout(); fig.savefig(FIG / "10_comparison_vs_tdis_res5.png"); plt.close(fig)
    print("saved res-5 results + 4 figures")


if __name__ == "__main__":
    main()
