"""
tune_tx.py
-----------
Hyperparameter tuning for Texas Wildfire XGBoost model.

Current baseline (train_tx.py result):
  TEST AUROC = 0.8637  (target: ~0.87-0.89)
  TEST AUPR  = 0.4106

Search strategy:
  Stage 1 — tree structure: max_depth x min_child_weight  (8 trials)
  Stage 2 — learning + sampling: learning_rate x subsample  (9 trials)
  Final   — retrain best config, evaluate on TEST set

All trials evaluate on VAL set only.
TEST set is touched only once at the very end (correct ML practice).

Loads from pre-saved parquets — no Excel re-read (fast).

Usage:
    python tune_tx.py

Output:
    outputs/texas_landfire/models/xgb_tx_tuned.ubj
    outputs/texas_landfire/models/xgb_tx_tuned_meta.json
    outputs/texas_landfire/tuning_results.csv
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "outputs" / "texas_landfire"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "models").mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "tune_tx.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Baselines to beat ─────────────────────────────────────────────────────────
BASELINE_AUROC = 0.8637
BASELINE_AUPR  = 0.4106

# ── Feature columns (must match train_tx.py) ──────────────────────────────────
FEATURE_COLS = [
    "avg_burn_prob", "whp", "flep4", "cfl", "cbd", "cbh", "burnable",
    "erc", "fm100", "vpd", "vs", "rmax", "rmin", "tmmx", "pr",
    "erc_5D_mean",  "erc_5D_max",
    "fm100_5D_mean","fm100_5D_min",
    "vpd_5D_mean",  "vpd_5D_max",
    "vs_5D_mean",   "vs_5D_max",
    "rmax_5D_mean", "rmax_5D_min",
    "tmmx_5D_mean", "tmmx_5D_max",
    "sin_month", "cos_month",
    "sin_hour",  "cos_hour",
    "centroid_lat", "centroid_lon",
    "gridmet_missing",   # added in train_tx.py
]


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    log.info("Loading pre-saved clean parquets from data/...")
    train_df = pd.read_parquet(DATA_DIR / "train_tx_clean.parquet")
    val_df   = pd.read_parquet(DATA_DIR / "val_tx_clean.parquet")
    test_df  = pd.read_parquet(DATA_DIR / "test_tx_clean.parquet")

    features = [c for c in FEATURE_COLS if c in train_df.columns]
    absent   = [c for c in FEATURE_COLS if c not in train_df.columns]
    if absent:
        log.warning(f"  Features not found in parquets: {absent}")

    def to_xy(d: pd.DataFrame):
        X = d[features].copy()
        for col in X.select_dtypes(include="object").columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(np.float32)
        return X, d["label"].values.astype(np.int8)

    X_train, y_train = to_xy(train_df)
    X_val,   y_val   = to_xy(val_df)
    X_test,  y_test  = to_xy(test_df)

    log.info(f"  TRAIN: {len(y_train):,} rows  fire={y_train.sum():,}  "
             f"rate={100*y_train.mean():.1f}%")
    log.info(f"  VAL:   {len(y_val):,} rows  fire={y_val.sum():,}")
    log.info(f"  TEST:  {len(y_test):,} rows  fire={y_test.sum():,}")
    log.info(f"  Features: {len(features)}")

    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    log.info(f"  scale_pos_weight = {spw:.2f}")

    return X_train, y_train, X_val, y_val, X_test, y_test, features, spw


# ── Single trial ──────────────────────────────────────────────────────────────
def run_trial(params: dict,
              dtrain: xgb.DMatrix,
              dval: xgb.DMatrix,
              y_val: np.ndarray,
              trial_name: str) -> dict:
    """Train one XGBoost config, return val AUROC + AUPR."""
    t0 = time.time()
    evals_result = {}
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=3000,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        early_stopping_rounds=60,
        verbose_eval=False,
    )
    elapsed = time.time() - t0
    best_round = model.best_iteration

    val_prob = model.predict(dval)
    val_auroc = roc_auc_score(y_val, val_prob)
    val_aupr  = average_precision_score(y_val, val_prob)
    val_auc_xgb = evals_result["val"]["auc"][best_round]

    depth = params["max_depth"]
    mcw   = params["min_child_weight"]
    lr    = params["learning_rate"]
    sub   = params["subsample"]
    cbt   = params["colsample_bytree"]

    marker = "  ✓ BEST" if val_auroc > BASELINE_AUROC else ""
    log.info(
        f"  {trial_name:<12}  "
        f"depth={depth}  mcw={mcw:>2}  lr={lr:.3f}  sub={sub}  cbt={cbt}  "
        f"| rounds={best_round+1:>4}  "
        f"val-AUROC={val_auroc:.4f}  val-AUPR={val_aupr:.4f}  "
        f"({elapsed:.0f}s){marker}"
    )

    return {
        "trial":     trial_name,
        "max_depth": depth,
        "min_child_weight": mcw,
        "learning_rate": lr,
        "subsample": sub,
        "colsample_bytree": cbt,
        "best_round": best_round + 1,
        "val_auroc": val_auroc,
        "val_aupr":  val_aupr,
        "elapsed_s": round(elapsed, 1),
        "model":     model,
    }


# ── Main tuning ───────────────────────────────────────────────────────────────
def tune():
    # ── Load ──────────────────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test, features, spw = load_data()

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features, missing=np.nan)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=features, missing=np.nan)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=features, missing=np.nan)

    # GPU check
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        gpu = r.stdout.strip()
        device = "cuda" if gpu else "cpu"
        if gpu:
            log.info(f"GPU: {gpu}")
    except Exception:
        device = "cpu"
    log.info(f"Device: {device.upper()}")

    # Shared base params (not being tuned in this search)
    base_params = {
        "objective":         "binary:logistic",
        "eval_metric":       ["logloss", "auc"],
        "tree_method":       "hist",
        "device":            device,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "gamma":             0.1,
        "colsample_bylevel": 0.8,
        "scale_pos_weight":  spw,
    }

    all_results = []

    # ── STAGE 1: Tree structure ────────────────────────────────────────────────
    # Fix LR=0.05, subsample=0.8, cbt=0.8
    # Vary: max_depth (6, 7, 8, 9) × min_child_weight (10, 20, 30)
    log.info(f"\n{'='*75}")
    log.info("STAGE 1 — Tree structure  (max_depth × min_child_weight)")
    log.info(f"  Baseline: depth=7  mcw=30  →  val-AUROC={BASELINE_AUROC:.4f}")
    log.info(f"{'='*75}")
    log.info(f"  {'Trial':<12}  {'Config':<45}  {'Results'}")
    log.info(f"  {'-'*72}")

    stage1_depths = [6, 7, 8, 9]
    stage1_mcws   = [10, 20, 30]

    for depth, mcw in product(stage1_depths, stage1_mcws):
        params = {
            **base_params,
            "max_depth":        depth,
            "min_child_weight": mcw,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
        }
        result = run_trial(params, dtrain, dval, y_val,
                           trial_name=f"d{depth}_mcw{mcw}")
        all_results.append(result)

    # Best stage 1 config
    best_s1 = max(all_results, key=lambda x: x["val_auroc"])
    best_depth = best_s1["max_depth"]
    best_mcw   = best_s1["min_child_weight"]
    log.info(f"\n  Stage 1 best:  depth={best_depth}  mcw={best_mcw}  "
             f"val-AUROC={best_s1['val_auroc']:.4f}  val-AUPR={best_s1['val_aupr']:.4f}")

    # ── STAGE 2: Learning rate + sampling ─────────────────────────────────────
    # Fix best depth + mcw from stage 1
    # Vary: learning_rate (0.01, 0.02, 0.05) × subsample (0.7, 0.8, 0.9)
    log.info(f"\n{'='*75}")
    log.info(f"STAGE 2 — Learning rate × subsample  "
             f"(depth={best_depth}  mcw={best_mcw})")
    log.info(f"{'='*75}")

    stage2_lrs  = [0.01, 0.02, 0.05]
    stage2_subs = [0.7, 0.8, 0.9]

    stage2_results = []
    for lr, sub in product(stage2_lrs, stage2_subs):
        params = {
            **base_params,
            "max_depth":        best_depth,
            "min_child_weight": best_mcw,
            "learning_rate":    lr,
            "subsample":        sub,
            "colsample_bytree": 0.8,
        }
        result = run_trial(params, dtrain, dval, y_val,
                           trial_name=f"lr{lr}_s{sub}")
        all_results.append(result)
        stage2_results.append(result)

    best_s2 = max(stage2_results, key=lambda x: x["val_auroc"])
    log.info(f"\n  Stage 2 best:  lr={best_s2['learning_rate']}  "
             f"sub={best_s2['subsample']}  "
             f"val-AUROC={best_s2['val_auroc']:.4f}  "
             f"val-AUPR={best_s2['val_aupr']:.4f}")

    # ── Best overall ──────────────────────────────────────────────────────────
    best = max(all_results, key=lambda x: x["val_auroc"])
    log.info(f"\n{'='*75}")
    log.info(f"BEST CONFIG OVERALL")
    log.info(f"{'='*75}")
    log.info(f"  max_depth        = {best['max_depth']}")
    log.info(f"  min_child_weight = {best['min_child_weight']}")
    log.info(f"  learning_rate    = {best['learning_rate']}")
    log.info(f"  subsample        = {best['subsample']}")
    log.info(f"  best_round       = {best['best_round']}")
    log.info(f"  val AUROC        = {best['val_auroc']:.4f}  "
             f"(vs baseline {BASELINE_AUROC:.4f}, "
             f"Δ={best['val_auroc']-BASELINE_AUROC:+.4f})")
    log.info(f"  val AUPR         = {best['val_aupr']:.4f}  "
             f"(vs baseline {BASELINE_AUPR:.4f}, "
             f"Δ={best['val_aupr']-BASELINE_AUPR:+.4f})")

    # ── Retrain best config with more rounds (final) ───────────────────────────
    # Lower LR often benefits from more trees — retrain with 5000 rounds
    log.info(f"\n{'='*75}")
    log.info("FINAL — Retrain best config on full train (up to 5000 rounds)")
    log.info(f"{'='*75}")

    final_params = {
        **base_params,
        "max_depth":        best["max_depth"],
        "min_child_weight": best["min_child_weight"],
        "learning_rate":    best["learning_rate"],
        "subsample":        best["subsample"],
        "colsample_bytree": 0.8,
    }
    log.info(f"  Config: {final_params}")

    evals_result_final = {}
    final_model = xgb.train(
        params=final_params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result_final,
        early_stopping_rounds=80,
        verbose_eval=100,
    )

    final_best_round = final_model.best_iteration
    log.info(f"  Final best round: {final_best_round+1}")

    # ── Final threshold (val) ─────────────────────────────────────────────────
    val_prob_final = final_model.predict(dval)
    prec, rec, thr = precision_recall_curve(y_val, val_prob_final)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    threshold = float(thr[np.argmax(f1s[:-1])])

    # ── Evaluate train/val/test ───────────────────────────────────────────────
    log.info(f"\n{'─'*75}")
    log.info("FINAL EVALUATION (tuned model)")
    log.info(f"{'─'*75}")
    final_metrics = []
    for y_true, dmat, sname in [
        (y_train, dtrain, "TRAIN"),
        (y_val,   dval,   "VAL  "),
        (y_test,  dtest,  "TEST "),
    ]:
        prob  = final_model.predict(dmat)
        auroc = roc_auc_score(y_true, prob)
        aupr  = average_precision_score(y_true, prob)
        ypred = (prob >= threshold).astype(int)
        tp = int(((ypred==1)&(y_true==1)).sum())
        fp = int(((ypred==1)&(y_true==0)).sum())
        fn = int(((ypred==0)&(y_true==1)).sum())
        tn = int(((ypred==0)&(y_true==0)).sum())
        p  = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
        m = {"split": sname.strip(), "auroc": round(auroc,4),
             "aupr": round(aupr,4), "f1": round(f1,4),
             "precision": round(p,4), "recall": round(r,4),
             "tp":tp,"fp":fp,"fn":fn,"tn":tn}
        final_metrics.append(m)
        log.info(f"  {sname}  AUROC={auroc:.4f}  AUPR={aupr:.4f}  "
                 f"F1={f1:.4f}  P={p:.3f}  R={r:.3f}")

    test_m = final_metrics[2]
    d_auroc = test_m["auroc"] - BASELINE_AUROC
    d_aupr  = test_m["aupr"]  - BASELINE_AUPR

    # ── Feature importance ────────────────────────────────────────────────────
    imp = final_model.get_score(importance_type="gain")
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    total_gain = sum(v for _, v in imp_sorted)
    log.info(f"\n  TOP 15 FEATURES (tuned model):")
    for feat, gain in imp_sorted[:15]:
        bar = chr(9608) * int(gain/imp_sorted[0][1]*25)
        log.info(f"  {feat:<20} {gain:>9.1f}  ({100*gain/total_gain:4.1f}%)  {bar}")

    # ── Save model + results ──────────────────────────────────────────────────
    model_path = OUT_DIR / "models" / "xgb_tx_tuned.ubj"
    meta_path  = OUT_DIR / "models" / "xgb_tx_tuned_meta.json"
    csv_path   = OUT_DIR / "tuning_results.csv"

    final_model.save_model(str(model_path))

    # Save tuning CSV (all trials, sorted by val AUROC)
    csv_rows = [{k: v for k, v in r.items() if k != "model"}
                for r in all_results]
    pd.DataFrame(csv_rows).sort_values("val_auroc", ascending=False).to_csv(
        csv_path, index=False
    )

    meta = {
        "tuning_baseline": {"auroc": BASELINE_AUROC, "aupr": BASELINE_AUPR},
        "best_config": {
            "max_depth":        best["max_depth"],
            "min_child_weight": best["min_child_weight"],
            "learning_rate":    best["learning_rate"],
            "subsample":        best["subsample"],
        },
        "final_best_round": final_best_round + 1,
        "threshold":        threshold,
        "test_metrics":     final_metrics,
        "vs_baseline": {"auroc_delta": round(d_auroc,4), "aupr_delta": round(d_aupr,4)},
        "feature_importance_top30": dict(imp_sorted[:30]),
        "n_trials": len(all_results),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"\n{'='*75}")
    log.info("  TUNING COMPLETE")
    log.info(f"{'='*75}")
    log.info(f"  Trials run:     {len(all_results)}")
    log.info(f"  Best config:    depth={best['max_depth']}  "
             f"mcw={best['min_child_weight']}  "
             f"lr={best['learning_rate']}  sub={best['subsample']}")
    log.info(f"")
    log.info(f"  {'Metric':<12}  {'Baseline':>10}  {'Tuned':>10}  {'Change':>10}")
    log.info(f"  {'─'*46}")
    log.info(f"  {'TEST AUROC':<12}  {BASELINE_AUROC:>10.4f}  "
             f"{test_m['auroc']:>10.4f}  {d_auroc:>+10.4f}")
    log.info(f"  {'TEST AUPR':<12}  {BASELINE_AUPR:>10.4f}  "
             f"{test_m['aupr']:>10.4f}  {d_aupr:>+10.4f}")
    log.info(f"  {'TEST F1':<12}  {'—':>10}  {test_m['f1']:>10.4f}  {'—':>10}")
    log.info(f"")
    log.info(f"  Model:   {model_path}")
    log.info(f"  Results: {csv_path}")
    log.info(f"{'='*75}")

    if test_m["auroc"] > 0.99:
        log.warning("  AUROC > 0.99 — check for leakage!")
    elif d_auroc > 0:
        log.info("  TEST AUROC improved — tuned model is better")
    else:
        log.warning("  TEST AUROC did not improve — keep baseline model")
        log.info("  Baseline model: outputs/texas_landfire/models/xgb_tx_landfire.ubj")


if __name__ == "__main__":
    tune()
