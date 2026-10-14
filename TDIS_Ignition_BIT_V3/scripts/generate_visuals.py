"""TDIS_Ignition_BIT_V3 -- same 9-plot evaluation suite as BIT V2, adapted to the 29-feature HRRR-only model."""
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
    confusion_matrix
)
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "bit_v3_train_tx.parquet"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

def main():
    feat_info = json.load(open(MODEL_DIR / "feature_list.json"))
    FEATURES = feat_info["features"]
    results = json.load(open(ROOT / "results" / "training_results_bit_v3_27feat.json"))

    df = pd.read_parquet(DATA)
    test = df[df.split == "test"]
    X_test, y_test = test[FEATURES], test["label"].values

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_DIR / "model_bit_v3_27feat.json"))
    calibrator = joblib.load(MODEL_DIR / "calibrator_bit_v3_27feat.joblib")

    p_raw = model.predict_proba(X_test)[:, 1]
    p_cal = calibrator.predict(p_raw)

    fpr_r, tpr_r, _ = roc_curve(y_test, p_raw)
    fpr_c, tpr_c, _ = roc_curve(y_test, p_cal)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_r, tpr_r, label=f"Raw (AUROC = {auc(fpr_r,tpr_r):.4f})", color="#1f77b4", lw=2)
    ax.plot(fpr_c, tpr_c, label=f"Calibrated (AUROC = {auc(fpr_c,tpr_c):.4f})", color="#ff7f0e", lw=2, ls="--")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls=":", label="Random (0.5)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve -- BIT V3 (29 feat, HRRR-only), test (n={len(y_test):,})")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(FIG_DIR / "01_roc_curve.png"); plt.close(fig)

    prec_r, rec_r, _ = precision_recall_curve(y_test, p_raw)
    prec_c, rec_c, _ = precision_recall_curve(y_test, p_cal)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec_r, prec_r, label=f"Raw (AUC-PR = {average_precision_score(y_test,p_raw):.4f})", color="#1f77b4", lw=2)
    ax.plot(rec_c, prec_c, label=f"Calibrated (AUC-PR = {average_precision_score(y_test,p_cal):.4f})", color="#ff7f0e", lw=2, ls="--")
    ax.axhline(y_test.mean(), color="gray", lw=1, ls=":", label=f"Base rate = {y_test.mean():.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve -- BIT V3, test split")
    ax.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(FIG_DIR / "02_pr_curve.png"); plt.close(fig)

    def plot_cm(y_true, p, thr, title, path):
        pred = (p >= thr).astype(int)
        cm = confusion_matrix(y_true, pred)
        tn, fp, fn, tp = cm.ravel()
        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(cm, cmap="Blues")
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]:,}", ha="center", va="center",
                        fontsize=13, color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted No Fire", "Predicted Fire"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual No Fire", "Actual Fire"])
        precision = tp/(tp+fp); recall = tp/(tp+fn); acc = (tp+tn)/len(y_true)
        ax.set_title(f"{title}\nthr={thr:.4f}  P={precision:.3f}  R={recall:.3f}  Acc={acc:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout(); fig.savefig(path); plt.close(fig)

    thr_raw = results["metrics_raw"]["test"]["best_threshold"]
    thr_cal = results["metrics_calibrated"]["test"]["best_threshold"]
    plot_cm(y_test, p_raw, thr_raw, "Confusion Matrix -- Raw probability, best-F1 threshold",
            FIG_DIR / "03_confusion_matrix_raw.png")
    plot_cm(y_test, p_cal, thr_cal, "Confusion Matrix -- Calibrated probability, best-F1 threshold",
            FIG_DIR / "04_confusion_matrix_calibrated.png")

    thresholds = np.linspace(0.01, 0.99, 99)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        pred = (p_raw >= t).astype(int)
        tp = ((pred==1)&(y_test==1)).sum(); fp = ((pred==1)&(y_test==0)).sum(); fn = ((pred==0)&(y_test==1)).sum()
        p = tp/(tp+fp) if (tp+fp) else 0; r = tp/(tp+fn) if (tp+fn) else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        precs.append(p); recs.append(r); f1s.append(f1)
    best_idx = int(np.argmax(f1s))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precs, label="Precision", color="#2ca02c", lw=2)
    ax.plot(thresholds, recs, label="Recall", color="#d62728", lw=2)
    ax.plot(thresholds, f1s, label="F1", color="#9467bd", lw=2)
    ax.axvline(thresholds[best_idx], color="gray", ls="--", lw=1, label=f"Best F1 @ thr={thresholds[best_idx]:.2f}")
    ax.set_xlabel("Decision threshold (raw probability)"); ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs. Threshold -- BIT V3, test split")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "05_precision_recall_f1_vs_threshold.png"); plt.close(fig)

    fi = results["feature_importance_gain"][:20]
    names = [r["feature"] for r in fi][::-1]; pcts = [r["pct"]*100 for r in fi][::-1]
    colors = ["#d62728" if p == 0 else "#1f77b4" for p in pcts]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(names, pcts, color=colors)
    for i, v in enumerate(pcts):
        ax.text(v + 0.1, i, f"{v:.2f}%", va="center", fontsize=8)
    ax.set_xlabel("% of total gain"); ax.set_title("Top 20 Feature Importance (gain) -- BIT V3 (HRRR-only)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "06_feature_importance.png"); plt.close(fig)

    frac_pos_r, mean_pred_r = calibration_curve(y_test, p_raw, n_bins=15, strategy="quantile")
    frac_pos_c, mean_pred_c = calibration_curve(y_test, p_cal, n_bins=15, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="gray", ls=":", label="Perfectly calibrated")
    ax.plot(mean_pred_r, frac_pos_r, "o-", color="#1f77b4", label="Raw")
    ax.plot(mean_pred_c, frac_pos_c, "o-", color="#ff7f0e", label="Calibrated")
    ax.set_xlabel("Mean predicted probability (per bin)"); ax.set_ylabel("Observed fire rate (per bin)")
    ax.set_title("Reliability Diagram -- BIT V3, test split (15 quantile bins)")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "07_calibration_reliability.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(p_raw[y_test==0], bins=60, alpha=0.6, label="Actual No Fire", color="#1f77b4", density=True)
    ax.hist(p_raw[y_test==1], bins=60, alpha=0.6, label="Actual Fire", color="#d62728", density=True)
    ax.axvline(thr_raw, color="black", ls="--", lw=1, label=f"Best-F1 threshold ({thr_raw:.3f})")
    ax.set_xlabel("Raw predicted probability"); ax.set_ylabel("Density")
    ax.set_title("Predicted Score Distribution by True Label -- BIT V3, test split")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "08_prediction_distribution.png"); plt.close(fig)

    x_grid = np.linspace(0, 1, 200); y_grid = calibrator.predict(x_grid)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="gray", ls=":", label="Identity (no change)")
    ax.plot(x_grid, y_grid, color="#2ca02c", lw=2, label="Isotonic calibrator")
    ax.set_xlabel("Raw model score"); ax.set_ylabel("Calibrated probability")
    ax.set_title("Isotonic Calibrator Mapping -- BIT V3 (fit on validation fold)")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "09_calibration_curve_map.png"); plt.close(fig)

    print(f"Saved 9 figures to {FIG_DIR}")

if __name__ == "__main__":
    main()
