"""
visualize_tx_hrrr.py
---------------------
Texas Wildfire Ignition Model — HRRR Evaluation Figure Suite

Generates publication-quality evaluation figures for the HRRR-enriched TX model
(the equivalent of V2/phase2/run_phase3_visualize.py, updated for the TX
LANDFIRE + HRRR pipeline and the models produced by train_tx_hrrr.py /
tune_tx_hrrr.py).

Reads:
  data/hrrr/hrrr_tx_all.parquet                      (features + _split + label)
  outputs/texas_landfire/models/<model>.ubj          (booster)
  outputs/texas_landfire/models/<model>_meta.json    (feature list + threshold)

Writes (outputs/texas_landfire/figures/):
  01_roc_curve.png              ROC for train/val/test
  02_pr_curve.png               Precision-Recall for train/val/test
  03_confusion_matrix.png       TEST confusion matrix @ tuned threshold
  04_threshold_sweep.png        Precision / Recall / F1 vs decision threshold
  05_feature_importance.png     Top-25 gain importance, colored by feature group
  06_feature_group_gain.png     Gain share per feature group (HRRR vs rest)
  07_score_distribution.png     Predicted score distribution, fire vs no-fire
  08_calibration_curve.png      Reliability diagram + Brier score
  09_precision_at_topk.png      Operational targeting curve (precision @ top K%)
  10_auroc_by_year.png          TEST-year breakdown vs HRRR coverage
  11_model_comparison.png       All trained TX models, AUROC + AUPR
  00_dashboard.png              All key panels on one sheet
  EVALUATION_REPORT.md          Metrics written out in markdown

Usage:
    python visualize_tx_hrrr.py                      # default: xgb_tx_hrrr_tuned
    python visualize_tx_hrrr.py --model xgb_tx_hrrr
    python visualize_tx_hrrr.py --model xgb_tx_tuned # gridMET-only baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score, brier_score_loss,
)
from sklearn.calibration import calibration_curve

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
HRRR_PQ = ROOT / "data" / "hrrr" / "hrrr_tx_all.parquet"
OUT_DIR = ROOT / "outputs" / "texas_landfire"
MODELS  = OUT_DIR / "models"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Palette ───────────────────────────────────────────────────────────────────
# Validated dark-mode categorical slots (blue / orange / aqua / magenta / violet).
# Yellow is deliberately skipped: it fails the all-pairs separation floor against
# orange, and several of these charts put non-adjacent groups side by side.
C = {
    "s1_blue":    "#3987e5",
    "s2_orange":  "#d95926",
    "s3_aqua":    "#199e70",
    "s5_magenta": "#d55181",
    "s7_violet":  "#9085e9",
    "s8_red":     "#e66767",
    "surface":    "#1a1a19",
    "panel":      "#232321",
    "text":       "#ffffff",
    "text2":      "#c3c2b7",
    "grid":       "#3a3a37",
    "good":       "#0ca30c",
    "critical":   "#d03b3b",
}
SPLIT_COLORS = {"TRAIN": C["s1_blue"], "VAL": C["s2_orange"], "TEST": C["s3_aqua"]}

# Feature-group colors. Every chart that uses these also names the group in a
# legend AND labels each bar directly, so identity is never carried by hue alone.
GROUP_COLORS = {
    "HRRR (sub-daily)":  C["s1_blue"],
    "gridMET weather":   C["s2_orange"],
    "Landscape/LANDFIRE": C["s3_aqua"],
    "Temporal":          C["s5_magenta"],
    "Location":          C["s7_violet"],
}

HRRR_FEATS = {"temp_pw", "rh_pw", "wind_pw", "vpd_pw_hrrr",
              "hpbl_pw", "dswrf_pw", "hrrr_pw", "hrrr_rh_valid"}
LANDSCAPE_FEATS = {"avg_burn_prob", "whp", "flep4", "cfl", "cbd", "cbh", "burnable"}
TEMPORAL_FEATS = {"sin_month", "cos_month", "sin_hour", "cos_hour"}
LOCATION_FEATS = {"centroid_lat", "centroid_lon"}


def feature_group(name: str) -> str:
    if name in HRRR_FEATS:
        return "HRRR (sub-daily)"
    if name in LANDSCAPE_FEATS:
        return "Landscape/LANDFIRE"
    if name in TEMPORAL_FEATS:
        return "Temporal"
    if name in LOCATION_FEATS:
        return "Location"
    return "gridMET weather"


def dark_style():
    plt.rcParams.update({
        "figure.facecolor":  C["surface"],
        "axes.facecolor":    C["panel"],
        "axes.edgecolor":    C["grid"],
        "axes.labelcolor":   C["text"],
        "axes.titlecolor":   C["text"],
        "xtick.color":       C["text2"],
        "ytick.color":       C["text2"],
        "text.color":        C["text"],
        "grid.color":        C["grid"],
        "grid.linewidth":    0.6,
        "legend.facecolor":  C["panel"],
        "legend.edgecolor":  C["grid"],
        "legend.labelcolor": C["text"],
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ── Load ──────────────────────────────────────────────────────────────────────
def load_everything(model_name: str):
    model_path = MODELS / f"{model_name}.ubj"
    meta_path  = MODELS / f"{model_name}_meta.json"
    if not model_path.exists():
        log.error(f"Model not found: {model_path}")
        log.error(f"Available: {[p.stem for p in MODELS.glob('*.ubj')]}")
        sys.exit(1)

    model = xgb.Booster()
    model.load_model(str(model_path))
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    threshold = meta["threshold"]
    features = meta.get("features")
    if not features:
        features = model.feature_names
    log.info(f"Model: {model_name}  |  {len(features)} features  |  threshold={threshold:.4f}")

    df = pd.read_parquet(HRRR_PQ)
    log.info(f"Data: {len(df):,} rows x {len(df.columns)} cols")

    splits = {}
    for key, name in [("train", "TRAIN"), ("val", "VAL"), ("test", "TEST")]:
        d = df[df["_split"] == key].reset_index(drop=True)
        X = d[features].copy()
        for col in X.select_dtypes(include="object").columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(np.float32)
        y = d["label"].values.astype(np.int8)
        dm = xgb.DMatrix(X, feature_names=features, missing=np.nan)
        prob = model.predict(dm)
        splits[name] = {"y": y, "prob": prob, "df": d}
        log.info(f"  {name:<6} n={len(y):,}  fires={int(y.sum()):,}  "
                 f"AUROC={roc_auc_score(y, prob):.4f}")

    return model, meta, features, threshold, splits, df


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc(ax, splits):
    ax.set_title("ROC Curve — ranking quality across splits", fontweight="bold")
    for name in ["TRAIN", "VAL", "TEST"]:
        y, prob = splits[name]["y"], splits[name]["prob"]
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, color=SPLIT_COLORS[name], lw=2,
                label=f"{name}   AUROC = {auc(fpr, tpr):.4f}")
    ax.plot([0, 1], [0, 1], "--", color=C["text2"], lw=1.2, alpha=0.6,
            label="Random = 0.5000")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(True, alpha=0.35)


def plot_pr(ax, splits, base_rate):
    ax.set_title("Precision-Recall Curve — the metric that matters at 9% fire rate",
                 fontweight="bold")
    for name in ["TRAIN", "VAL", "TEST"]:
        y, prob = splits[name]["y"], splits[name]["prob"]
        prec, rec, _ = precision_recall_curve(y, prob)
        ap = average_precision_score(y, prob)
        ax.plot(rec, prec, color=SPLIT_COLORS[name], lw=2,
                label=f"{name}   AUPR = {ap:.4f}")
    ax.axhline(base_rate, ls="--", lw=1.2, color=C["text2"], alpha=0.7,
               label=f"Random = {base_rate:.4f} (fire rate)")
    ax.set_xlabel("Recall  (share of real fires caught)")
    ax.set_ylabel("Precision  (share of flags that are real fires)")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True, alpha=0.35)


def plot_confusion(ax, y, prob, threshold):
    y_pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    total = len(y)
    m = np.array([[tn, fp], [fn, tp]])
    labels = np.array([
        [f"TN\n{tn:,}\n{100*tn/total:.1f}%", f"FP\n{fp:,}\n{100*fp/total:.1f}%"],
        [f"FN\n{fn:,}\n{100*fn/total:.1f}%", f"TP\n{tp:,}\n{100*tp/total:.1f}%"],
    ])
    # Sequential single-hue ramp (magnitude encoding), not a rainbow.
    ax.imshow(m, cmap="Blues", aspect="auto")
    # Ink follows cell darkness: "Blues" paints high values dark (needs light ink)
    # and low values near-white (needs dark ink). TN dwarfs the other three cells,
    # so normalise against the actual range rather than testing raw counts.
    norm = (m - m.min()) / (m.max() - m.min() + 1e-9)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i, j], ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color=C["text"] if norm[i, j] > 0.5 else "#0b0b0b")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: NO FIRE", "Predicted: FIRE"], fontsize=10)
    ax.set_yticklabels(["Actual: NO FIRE", "Actual: FIRE"], fontsize=10)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    spec = tn / (tn + fp)
    ax.set_title(f"Confusion Matrix — TEST @ threshold {threshold:.4f}",
                 fontweight="bold")
    ax.set_xlabel(f"Precision {p:.3f}   Recall {r:.3f}   F1 {f1:.3f}   "
                  f"Specificity {spec:.3f}", fontsize=10, color=C["text2"])
    ax.grid(False)
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                precision=p, recall=r, f1=f1, specificity=spec)


def plot_threshold_sweep(ax, y, prob, tuned_threshold):
    ax.set_title("Precision / Recall / F1 vs decision threshold — TEST",
                 fontweight="bold")
    prec, rec, thr = precision_recall_curve(y, prob)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    ax.plot(thr, prec, color=C["s1_blue"], lw=2, label="Precision")
    ax.plot(thr, rec, color=C["s2_orange"], lw=2, label="Recall")
    ax.plot(thr, f1, color=C["s3_aqua"], lw=2, label="F1")
    ax.axvline(tuned_threshold, ls="--", lw=1.6, color=C["text2"],
               label=f"Tuned threshold = {tuned_threshold:.4f}")
    best_i = int(np.argmax(f1))
    # Carried as a legend entry, not a floating annotation — the max-F1 point sits
    # in the middle of the three curves, where any callout collides with something.
    ax.plot(thr[best_i], f1[best_i], "o", ms=9, color=C["s3_aqua"],
            markeredgecolor=C["surface"], markeredgewidth=2,
            label=f"max F1 = {f1[best_i]:.3f} @ {thr[best_i]:.3f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center right", fontsize=9.5)
    ax.grid(True, alpha=0.35)


def plot_feature_importance(ax, model, top_n=25):
    scores = model.get_score(importance_type="gain")
    if not scores:
        ax.text(0.5, 0.5, "No importance scores", ha="center", transform=ax.transAxes)
        return {}
    total = sum(scores.values())
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [n for n, _ in items][::-1]
    vals  = np.array([100 * v / total for _, v in items])[::-1]
    colors = [GROUP_COLORS[feature_group(n)] for n in names]

    ax.barh(range(len(names)), vals, color=colors, height=0.72)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("% of total gain importance")
    ax.set_title(f"Top {top_n} features by XGBoost gain", fontweight="bold")
    ax.set_xlim(0, vals.max() * 1.16)
    for i, v in enumerate(vals):
        ax.text(v + vals.max() * 0.012, i, f"{v:.1f}%", va="center",
                fontsize=8, color=C["text2"])
    present = [g for g in GROUP_COLORS if any(feature_group(n) == g for n in names)]
    ax.legend(handles=[mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in present],
              loc="lower right", fontsize=8.5)
    ax.grid(True, axis="x", alpha=0.3)
    return {n: 100 * v / total for n, v in scores.items()}


def plot_group_gain(ax, model):
    scores = model.get_score(importance_type="gain")
    total = sum(scores.values())
    agg = {}
    for n, v in scores.items():
        agg[feature_group(n)] = agg.get(feature_group(n), 0) + v
    order = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in order]
    vals = np.array([100 * v / total for _, v in order])
    ax.bar(range(len(names)), vals, color=[GROUP_COLORS[n] for n in names], width=0.62)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=9)
    ax.set_ylabel("% of total gain importance")
    ax.set_title("Where the model gets its signal — gain share by feature group",
                 fontweight="bold")
    ax.set_ylim(0, vals.max() * 1.18)
    for i, v in enumerate(vals):
        ax.text(i, v + vals.max() * 0.025, f"{v:.1f}%", ha="center",
                fontsize=11, fontweight="bold", color=C["text"])
    ax.grid(True, axis="y", alpha=0.3)
    return dict(order)


def plot_score_dist(ax, y, prob, threshold):
    ax.set_title("Predicted score distribution — TEST", fontweight="bold")
    ax.hist(prob[y == 0], bins=80, alpha=0.75, color=C["s1_blue"], density=True,
            label=f"No fire (n={int((y==0).sum()):,})")
    ax.hist(prob[y == 1], bins=80, alpha=0.75, color=C["s2_orange"], density=True,
            label=f"Fire (n={int((y==1).sum()):,})")
    ax.axvline(threshold, ls="--", lw=1.8, color=C["text2"],
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Model output score (probability of fire)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9.5)
    ax.grid(True, alpha=0.35)


def plot_calibration(ax, y, prob):
    ax.set_title("Calibration (reliability) — TEST", fontweight="bold")
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=20, strategy="quantile")
    brier = brier_score_loss(y, prob)
    ax.plot([0, 1], [0, 1], "--", lw=1.2, color=C["text2"], alpha=0.7,
            label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", lw=2, ms=7, color=C["s1_blue"],
            markeredgecolor=C["surface"], markeredgewidth=1.5,
            label=f"Model  (Brier = {brier:.4f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fire fraction")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(True, alpha=0.35)
    ax.annotate("Above the line = model under-predicts risk\n"
                "Below = over-predicts (scale_pos_weight inflates scores)",
                xy=(0.03, 0.62), xycoords="axes fraction",
                fontsize=8.5, color=C["text2"])
    return brier


def plot_precision_at_topk(ax, y, prob):
    ax.set_title("Operational targeting — precision within the top-scoring cells",
                 fontweight="bold")
    order = np.argsort(prob)[::-1]
    y_sorted = y[order]
    n = len(y_sorted)
    ks = np.unique(np.linspace(0.001, 0.5, 300) * n).astype(int)
    ks = ks[ks > 0]
    prec = np.cumsum(y_sorted)[ks - 1] / ks
    base = y.mean()
    ax.plot(100 * ks / n, prec, lw=2, color=C["s1_blue"], label="Precision @ top K%")
    ax.axhline(base, ls="--", lw=1.2, color=C["text2"], alpha=0.8,
               label=f"Random = {base:.4f}")
    for pct in [1, 5, 10]:
        k = max(int(pct / 100 * n), 1)
        p = y_sorted[:k].sum() / k
        ax.plot(pct, p, "o", ms=8, color=C["s2_orange"],
                markeredgecolor=C["surface"], markeredgewidth=1.5)
        ax.annotate(f"top {pct}% → {p:.3f}\n({p/base:.1f}× random)",
                    xy=(pct, p), xytext=(10, 8), textcoords="offset points",
                    fontsize=8.5, color=C["text"])
    ax.set_xlabel("Top K% of cells by predicted risk")
    ax.set_ylabel("Precision within that K%")
    ax.set_xlim(0, 50)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True, alpha=0.35)


def plot_auroc_by_year(ax, splits, full_df):
    ax.set_title("TEST-set AUROC by year, with HRRR coverage", fontweight="bold")
    d = splits["TEST"]["df"].copy()
    d["_prob"] = splits["TEST"]["prob"]
    d["_year"] = pd.to_datetime(d["date_utc"]).dt.year
    years, aurocs, covs = [], [], []
    for yr, g in d.groupby("_year"):
        if g["label"].nunique() < 2:
            continue
        years.append(int(yr))
        aurocs.append(roc_auc_score(g["label"], g["_prob"]))
        covs.append(100 * (g["hrrr_pw"] == 1).mean())
    ax.bar(range(len(years)), aurocs, color=C["s3_aqua"], width=0.5)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls="--", lw=1.2, color=C["text2"], alpha=0.7, label="Random = 0.5")
    for i, (a, cv) in enumerate(zip(aurocs, covs)):
        ax.text(i, a + 0.02, f"{a:.4f}", ha="center", fontweight="bold",
                fontsize=11, color=C["text"])
        ax.text(i, 0.06, f"HRRR\n{cv:.0f}%", ha="center", fontsize=9,
                color=C["text2"])
    ax.legend(fontsize=9.5, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    return dict(zip(years, aurocs))


def plot_model_comparison(ax):
    """All trained TX models side by side, read from their saved meta JSONs."""
    catalog = [
        ("xgb_tx_landfire",     "V3 gridMET\n(untuned)"),
        ("xgb_tx_tuned",        "V3 gridMET\n(tuned)"),
        ("xgb_tx_hrrr",         "HRRR\n(borrowed cfg)"),
        ("xgb_tx_hrrr_tuned",   "HRRR\n(tuned)"),
        ("xgb_tx_hrrr_noflags", "HRRR\n(no flags)"),
    ]
    names, aurocs, auprs = [], [], []
    for stem, label in catalog:
        p = MODELS / f"{stem}_meta.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        key = "test_metrics" if "test_metrics" in m else "metrics"
        try:
            t = next(x for x in m[key] if x["split"] == "TEST")
        except (KeyError, StopIteration):
            continue
        names.append(label); aurocs.append(t["auroc"]); auprs.append(t["aupr"])

    if not names:
        ax.text(0.5, 0.5, "No model metadata found", ha="center", transform=ax.transAxes)
        return []

    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w/2, aurocs, w, color=C["s1_blue"], label="Test AUROC")
    ax.bar(x + w/2, auprs,  w, color=C["s2_orange"], label="Test AUPR")
    best = int(np.argmax(aurocs))
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Score")
    # Headroom for the value labels, the "best" callout and the legend band above
    # the bars — at ylim 1.0 the legend sat on top of the right-most bar's label.
    ax.set_ylim(0, 1.28)
    ax.set_title("Every trained TX model — HRRR does not beat gridMET-only",
                 fontweight="bold")
    for i, (a, p_) in enumerate(zip(aurocs, auprs)):
        ax.text(i - w/2, a + 0.015, f"{a:.4f}", ha="center", fontsize=8.5,
                fontweight="bold" if i == best else "normal", color=C["text"])
        ax.text(i + w/2, p_ + 0.015, f"{p_:.4f}", ha="center", fontsize=8.5,
                color=C["text2"])
    ax.annotate("best", xy=(best - w/2, aurocs[best] + 0.055),
                ha="center", fontsize=9, fontweight="bold", color=C["good"])
    ax.legend(fontsize=9.5, loc="upper center", ncol=2, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)
    return list(zip(names, aurocs, auprs))


# ══════════════════════════════════════════════════════════════════════════════
def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=C["surface"])
    plt.close(fig)
    log.info(f"  saved {path.name}")


def main():
    ap = argparse.ArgumentParser(description="TX HRRR model — evaluation figures")
    ap.add_argument("--model", default="xgb_tx_hrrr_tuned",
                    help="Model stem in outputs/texas_landfire/models/ "
                         "(default: xgb_tx_hrrr_tuned)")
    args = ap.parse_args()

    model, meta, features, threshold, splits, full_df = load_everything(args.model)
    dark_style()

    y_test, prob_test = splits["TEST"]["y"], splits["TEST"]["prob"]
    base_rate = float(full_df["label"].mean())

    log.info("Rendering figures...")

    fig, ax = plt.subplots(figsize=(9, 6.5)); plot_roc(ax, splits)
    save(fig, FIG_DIR / "01_roc_curve.png")

    fig, ax = plt.subplots(figsize=(9, 6.5)); plot_pr(ax, splits, base_rate)
    save(fig, FIG_DIR / "02_pr_curve.png")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    cm_stats = plot_confusion(ax, y_test, prob_test, threshold)
    save(fig, FIG_DIR / "03_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    plot_threshold_sweep(ax, y_test, prob_test, threshold)
    save(fig, FIG_DIR / "04_threshold_sweep.png")

    fig, ax = plt.subplots(figsize=(10, 9))
    plot_feature_importance(ax, model)
    save(fig, FIG_DIR / "05_feature_importance.png")

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    group_gain = plot_group_gain(ax, model)
    save(fig, FIG_DIR / "06_feature_group_gain.png")

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    plot_score_dist(ax, y_test, prob_test, threshold)
    save(fig, FIG_DIR / "07_score_distribution.png")

    fig, ax = plt.subplots(figsize=(8.5, 7))
    brier = plot_calibration(ax, y_test, prob_test)
    save(fig, FIG_DIR / "08_calibration_curve.png")

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    plot_precision_at_topk(ax, y_test, prob_test)
    save(fig, FIG_DIR / "09_precision_at_topk.png")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    year_auroc = plot_auroc_by_year(ax, splits, full_df)
    save(fig, FIG_DIR / "10_auroc_by_year.png")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    comparison = plot_model_comparison(ax)
    save(fig, FIG_DIR / "11_model_comparison.png")

    # ── Combined dashboard ────────────────────────────────────────────────────
    log.info("Rendering combined dashboard...")
    fig = plt.figure(figsize=(24, 26), facecolor=C["surface"])
    fig.suptitle(
        f"Texas Wildfire Ignition Model — HRRR Evaluation  ({args.model})\n"
        f"LANDFIRE + gridMET + HRRR  |  chronological split: "
        f"train 2014-2017 · val 2018 · test 2019-2020",
        fontsize=17, fontweight="bold", color=C["text"], y=0.982)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.32, wspace=0.24,
                           top=0.955, bottom=0.03, left=0.055, right=0.975)

    plot_roc(fig.add_subplot(gs[0, 0]), splits)
    plot_pr(fig.add_subplot(gs[0, 1]), splits, base_rate)
    plot_confusion(fig.add_subplot(gs[1, 0]), y_test, prob_test, threshold)
    plot_threshold_sweep(fig.add_subplot(gs[1, 1]), y_test, prob_test, threshold)
    plot_feature_importance(fig.add_subplot(gs[2, 0]), model, top_n=18)
    plot_group_gain(fig.add_subplot(gs[2, 1]), model)
    plot_score_dist(fig.add_subplot(gs[3, 0]), y_test, prob_test, threshold)
    plot_model_comparison(fig.add_subplot(gs[3, 1]))

    save(fig, FIG_DIR / "00_dashboard.png")

    # ── Report ────────────────────────────────────────────────────────────────
    write_report(args.model, meta, features, threshold, splits, base_rate,
                 cm_stats, group_gain, brier, year_auroc, comparison, model)

    log.info("=" * 65)
    log.info(f"  Figures: {FIG_DIR}")
    log.info(f"  Report:  {FIG_DIR / 'EVALUATION_REPORT.md'}")
    log.info("=" * 65)


def write_report(model_name, meta, features, threshold, splits, base_rate,
                 cm, group_gain, brier, year_auroc, comparison, model):
    L = []
    A = L.append
    A(f"# TX HRRR Model — Evaluation Report (`{model_name}`)")
    A("")
    A(f"Generated by `visualize_tx_hrrr.py`. Figures in `outputs/texas_landfire/figures/`.")
    A("")
    A("## Metrics by split")
    A("")
    A("| Split | Rows | Fires | Fire rate | AUROC | AUPR |")
    A("|---|---|---|---|---|---|")
    for name in ["TRAIN", "VAL", "TEST"]:
        y, prob = splits[name]["y"], splits[name]["prob"]
        A(f"| {name} | {len(y):,} | {int(y.sum()):,} | {100*y.mean():.2f}% | "
          f"{roc_auc_score(y, prob):.4f} | {average_precision_score(y, prob):.4f} |")
    A("")
    A(f"Random-baseline AUPR (dataset fire rate) = **{base_rate:.4f}**.")
    A("")
    A(f"## TEST confusion matrix @ threshold {threshold:.4f}")
    A("")
    A("| | Predicted NO FIRE | Predicted FIRE |")
    A("|---|---|---|")
    A(f"| **Actual NO FIRE** | TN {cm['tn']:,} | FP {cm['fp']:,} |")
    A(f"| **Actual FIRE** | FN {cm['fn']:,} | TP {cm['tp']:,} |")
    A("")
    A(f"- Precision **{cm['precision']:.4f}** — of all cells flagged, this share had a real fire")
    A(f"- Recall **{cm['recall']:.4f}** — of all real fires, this share was caught")
    A(f"- F1 **{cm['f1']:.4f}**, Specificity **{cm['specificity']:.4f}**")
    A(f"- Brier score **{brier:.4f}** (calibration; lower is better)")
    A("")
    A("## Gain share by feature group")
    A("")
    A("| Group | % of total gain |")
    A("|---|---|")
    total_g = sum(group_gain.values())
    for g, v in sorted(group_gain.items(), key=lambda x: -x[1]):
        A(f"| {g} | {100*v/total_g:.1f}% |")
    A("")
    A("## Full feature importance (gain %)")
    A("")
    A("| Rank | Feature | Group | % of total gain |")
    A("|---|---|---|---|")
    scores = model.get_score(importance_type="gain")
    tot = sum(scores.values())
    for i, (n, v) in enumerate(sorted(scores.items(), key=lambda x: -x[1]), 1):
        A(f"| {i} | `{n}` | {feature_group(n)} | {100*v/tot:.2f}% |")
    A("")
    A("## TEST AUROC by year")
    A("")
    A("| Year | AUROC |")
    A("|---|---|")
    for yr, a in year_auroc.items():
        A(f"| {yr} | {a:.4f} |")
    A("")
    if comparison:
        A("## All trained models")
        A("")
        A("| Model | Test AUROC | Test AUPR |")
        A("|---|---|---|")
        for nm, a, p in comparison:
            A(f"| {nm.replace(chr(10), ' ')} | {a:.4f} | {p:.4f} |")
        A("")
    A("## Figures")
    A("")
    for f in sorted(FIG_DIR.glob("*.png")):
        A(f"- `{f.name}`")
    A("")
    (FIG_DIR / "EVALUATION_REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
