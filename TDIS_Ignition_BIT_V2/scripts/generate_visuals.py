"""
TDIS_Ignition_BIT_V2 -- generates every standard model-evaluation visualization
for the 36-feature model, scored on the real test split.

Produces (all under results/figures/):
  01_roc_curve.png              -- ROC curve (raw + calibrated), AUROC annotated
  02_pr_curve.png                -- Precision-Recall curve (raw + calibrated), AUC-PR annotated
  03_confusion_matrix_raw.png    -- Confusion matrix heatmap, raw probs @ best-F1 threshold
  04_confusion_matrix_calibrated.png -- same, calibrated probs @ best-F1 threshold
  05_precision_recall_f1_vs_threshold.png -- P/R/F1 curves across thresholds
  06_feature_importance.png     -- Top-20 feature importance (gain) bar chart
  07_calibration_reliability.png -- Reliability diagram (predicted vs observed), raw vs calibrated
  08_prediction_distribution.png -- Histogram of predicted scores by true label
  09_calibration_curve_map.png  -- The isotonic calibrator's raw->calibrated mapping curve
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score
)
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "bit_v2_train_tx.parquet"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

def main():
    print("Loading model, calibrator, data ...")
    feat_info = json.load(open(MODEL_DIR / "feature_list.json"))
    FEATURES = feat_info["features"]
    results = json.load(open(ROOT / "results" / "training_results_bit_v2_36feat.json"))

    df = pd.read_parquet(DATA)
    test = df[df.split == "test"]
    X_test, y_test = test[FEATURES], test["label"].values

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_DIR / "model_bit_v2_36feat.json"))
    calibrator = joblib.load(MODEL_DIR / "calibrator_bit_v2_36feat.joblib")

    p_raw = model.predict_proba(X_test)[:, 1]
    p_cal = calibrator.predict(p_raw)

    # ---------------- 01. ROC curve ----------------
    print("Plotting ROC curve ...")
    fpr_r, tpr_r, _ = roc_curve(y_test, p_raw)
    fpr_c, tpr_c, _ = roc_curve(y_test, p_cal)
    auroc_r, auroc_c = auc(fpr_r, tpr_r), auc(fpr_c, tpr_c)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_r, tpr_r, label=f"Raw (AUROC = {auroc_r:.4f})", color="#1f77b4", lw=2)
    ax.plot(fpr_c, tpr_c, label=f"Calibrated (AUROC = {auroc_c:.4f})", color="#ff7f0e", lw=2, ls="--")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls=":", label="Random (AUROC = 0.5)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve -- BIT V2 (36 features), test split (n={:,})".format(len(y_test)))
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(FIG_DIR / "01_roc_curve.png"); plt.close(fig)

    # ---------------- 02. PR curve ----------------
    print("Plotting Precision-Recall curve ...")
    prec_r, rec_r, _ = precision_recall_curve(y_test, p_raw)
    prec_c, rec_c, _ = precision_recall_curve(y_test, p_cal)
    aupr_r, aupr_c = average_precision_score(y_test, p_raw), average_precision_score(y_test, p_cal)
    base_rate = y_test.mean()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec_r, prec_r, label=f"Raw (AUC-PR = {aupr_r:.4f})", color="#1f77b4", lw=2)
    ax.plot(rec_c, prec_c, label=f"Calibrated (AUC-PR = {aupr_c:.4f})", color="#ff7f0e", lw=2, ls="--")
    ax.axhline(base_rate, color="gray", lw=1, ls=":", label=f"Base rate = {base_rate:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve -- BIT V2, test split")
    ax.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(FIG_DIR / "02_pr_curve.png"); plt.close(fig)

    # ---------------- 03/04. Confusion matrices ----------------
    def plot_cm(y_true, p, thr, title, path):
        pred = (p >= thr).astype(int)
        cm = confusion_matrix(y_true, pred)
        tn, fp, fn, tp = cm.ravel()
        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(cm, cmap="Blues")
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}", ha="center", va="center",
                         fontsize=13, color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted No Fire", "Predicted Fire"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual No Fire", "Actual Fire"])
        precision = tp / (tp + fp); recall = tp / (tp + fn); acc = (tp + tn) / len(y_true)
        ax.set_title(f"{title}\nthr={thr:.4f}  P={precision:.3f}  R={recall:.3f}  Acc={acc:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout(); fig.savefig(path); plt.close(fig)

    print("Plotting confusion matrices ...")
    thr_raw = results["metrics_raw"]["test"]["best_threshold"]
    thr_cal = results["metrics_calibrated"]["test"]["best_threshold"]
    plot_cm(y_test, p_raw, thr_raw, "Confusion Matrix -- Raw probability, best-F1 threshold",
            FIG_DIR / "03_confusion_matrix_raw.png")
    plot_cm(y_test, p_cal, thr_cal, "Confusion Matrix -- Calibrated probability, best-F1 threshold",
            FIG_DIR / "04_confusion_matrix_calibrated.png")

    # ---------------- 05. Precision/Recall/F1 vs threshold ----------------
    print("Plotting P/R/F1 vs threshold ...")
    thresholds = np.linspace(0.01, 0.99, 99)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        pred = (p_raw >= t).astype(int)
        tp = ((pred == 1) & (y_test == 1)).sum()
        fp = ((pred == 1) & (y_test == 0)).sum()
        fn = ((pred == 0) & (y_test == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        precs.append(p); recs.append(r); f1s.append(f1)
    best_idx = int(np.argmax(f1s))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precs, label="Precision", color="#2ca02c", lw=2)
    ax.plot(thresholds, recs, label="Recall", color="#d62728", lw=2)
    ax.plot(thresholds, f1s, label="F1", color="#9467bd", lw=2)
    ax.axvline(thresholds[best_idx], color="gray", ls="--", lw=1,
               label=f"Best F1 @ thr={thresholds[best_idx]:.2f}")
    ax.set_xlabel("Decision threshold (raw probability)"); ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs. Threshold -- BIT V2, test split")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "05_precision_recall_f1_vs_threshold.png"); plt.close(fig)

    # ---------------- 06. Feature importance ----------------
    print("Plotting feature importance ...")
    fi = results["feature_importance_gain"][:20]
    names = [r["feature"] for r in fi][::-1]
    pcts = [r["pct"] * 100 for r in fi][::-1]
    colors = ["#d62728" if p == 0 else "#1f77b4" for p in pcts]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(names, pcts, color=colors)
    for i, v in enumerate(pcts):
        ax.text(v + 0.1, i, f"{v:.2f}%", va="center", fontsize=8)
    ax.set_xlabel("% of total gain"); ax.set_title("Top 20 Feature Importance (XGBoost gain) -- BIT V2")
    fig.tight_layout(); fig.savefig(FIG_DIR / "06_feature_importance.png"); plt.close(fig)

    # ---------------- 07. Calibration reliability diagram ----------------
    print("Plotting calibration reliability diagram ...")
    frac_pos_r, mean_pred_r = calibration_curve(y_test, p_raw, n_bins=15, strategy="quantile")
    frac_pos_c, mean_pred_c = calibration_curve(y_test, p_cal, n_bins=15, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1, label="Perfectly calibrated")
    ax.plot(mean_pred_r, frac_pos_r, "o-", color="#1f77b4", label="Raw")
    ax.plot(mean_pred_c, frac_pos_c, "o-", color="#ff7f0e", label="Calibrated")
    ax.set_xlabel("Mean predicted probability (per bin)"); ax.set_ylabel("Observed fire rate (per bin)")
    ax.set_title("Reliability Diagram -- BIT V2, test split (15 quantile bins)")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "07_calibration_reliability.png"); plt.close(fig)

    # ---------------- 08. Prediction distribution ----------------
    print("Plotting prediction distribution ...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(p_raw[y_test == 0], bins=60, alpha=0.6, label="Actual No Fire", color="#1f77b4", density=True)
    ax.hist(p_raw[y_test == 1], bins=60, alpha=0.6, label="Actual Fire", color="#d62728", density=True)
    ax.axvline(thr_raw, color="black", ls="--", lw=1, label=f"Best-F1 threshold ({thr_raw:.3f})")
    ax.set_xlabel("Raw predicted probability"); ax.set_ylabel("Density")
    ax.set_title("Predicted Score Distribution by True Label -- BIT V2, test split")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "08_prediction_distribution.png"); plt.close(fig)

    # ---------------- 09. Calibrator mapping curve ----------------
    print("Plotting calibrator mapping curve ...")
    x_grid = np.linspace(0, 1, 200)
    y_grid = calibrator.predict(x_grid)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1, label="Identity (no change)")
    ax.plot(x_grid, y_grid, color="#2ca02c", lw=2, label="Isotonic calibrator")
    ax.set_xlabel("Raw model score"); ax.set_ylabel("Calibrated probability")
    ax.set_title("Isotonic Calibrator Mapping -- BIT V2 (fit on validation fold)")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "09_calibration_curve_map.png"); plt.close(fig)

    print(f"\nSaved 9 figures to {FIG_DIR}")

if __name__ == "__main__":
    main()
