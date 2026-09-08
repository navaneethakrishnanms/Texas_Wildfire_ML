"""
tune_train_model_32feat.py
---------------------------
Hyperparameter-tuned 32-feature XGBoost wildfire ignition model on the TEXAS
v2 dataset (the original 30 TDIS features + centroid lat/lon added from
Supporting_files' tx_static_master.parquet -- see build_dataset_v2.py).

Baselines being compared against:
  TDIS shared model (Supporting_files/models/tdis_forecast_hrrr_filtered.json):
      TEST AUC-PR 0.4941, AUROC 0.7402, F1 0.4958   (22 features, no gridMET,
      no lat/lon, no tuning search -- fixed hyperparameters)
  Our own untuned 30-feat baseline (TEXAS_v1_backup/training_results_30feat.json):
      TEST AUC-PR 0.7384, AUROC 0.8249, F1 0.6440   (fixed hyperparameters,
      no lat/lon)

Search strategy (same two-stage design used in TX/tune_tx.py, which took the
TX model from AUROC 0.8637 -> 0.8687):
  Stage 1 -- tree structure:      max_depth x min_child_weight   (12 trials)
  Stage 2 -- learning + sampling: learning_rate x subsample       (9 trials)
  Final   -- retrain best config on TRAIN, evaluate on VAL + TEST once.
All 21 trials are scored on VAL only; TEST is touched exactly once, at the end.

Dataset : TEXAS/tdis_train_daily_imputed_v2.parquet
Split   : Train (<=2023-12-31) / Val (2024) / Test (2025-01-01 .. 2026-07-29)
          (same boundaries as the original 30-feat model; rows dated after
          2026-07-29 are fabricated placeholder rows per DATA_QUALITY_REVIEW.md
          and are excluded, exactly as the original script already did.)

Outputs:
  TEXAS/model_32feat_tuned.json
  TEXAS/calibrator_32feat_tuned.joblib
  TEXAS/tuning_results_32feat.csv
  TEXAS/training_results_32feat_tuned.json

Usage (run from the Texas ML Wildfire root directory):
    python TEXAS/tune_train_model_32feat.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
DATA = HERE / "tdis_train_daily_imputed_v2.parquet"
MODEL_OUT = HERE / "model_32feat_tuned.json"
CAL_OUT = HERE / "calibrator_32feat_tuned.joblib"
TUNING_CSV = HERE / "tuning_results_32feat.csv"
RESULTS_OUT = HERE / "training_results_32feat_tuned.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Baselines to beat ────────────────────────────────────────────────────────
BASELINE_30FEAT = {"aucpr": 0.7384, "auroc": 0.8249, "f1": 0.6440}
TDIS_SHARED = {"aucpr": 0.4941, "auroc": 0.7402, "f1": 0.4958}

# ── Split boundaries (identical to the original 30-feat script) ─────────────
TRAIN_END = "2023-12-31"
VAL_END = "2024-12-31"
TEST_END = "2026-07-29"

# ── 32 feature columns (original 30 + lat/lon) ───────────────────────────────
FEATURES = [
    "elevation_m", "slope_deg", "aspect_deg", "road_dist_km", "ecoregion_id",
    "avg_burn_prob", "whp", "flep4", "cfl", "cbd", "cbh",
    "powerline_dist_km",
    "hrrr_tmp", "hrrr_vpd", "hrrr_wind", "hrrr_mstav",
    "erc", "fm100", "vpd", "vs", "rmax", "rmin", "tmmx", "pr",
    "sin_month", "cos_month", "sin_dow", "cos_dow", "is_weekend", "is_holiday",
    "lat", "lon",  # NEW -- added in build_dataset_v2.py
]
TARGET = "label"


def best_f1_threshold(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 200)
    best_thr, best_f1 = 0.5, 0.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        if y_pred.sum() == 0:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def evaluate(y_true, y_prob, label=""):
    aucpr = float(average_precision_score(y_true, y_prob))
    auroc = float(roc_auc_score(y_true, y_prob))
    base = float(y_true.mean())
    lift = aucpr / base if base > 0 else 0.0
    thr, f1 = best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    log.info(f"  -- {label} --")
    log.info(f"     Rows       : {len(y_true):,}")
    log.info(f"     Base rate  : {base:.4f}")
    log.info(f"     AUC-PR     : {aucpr:.4f}")
    log.info(f"     AUROC      : {auroc:.4f}")
    log.info(f"     Lift       : {lift:.2f}x")
    log.info(f"     Best Thr   : {thr:.4f}")
    log.info(f"     F1         : {f1:.4f}")
    log.info(f"     Precision  : {prec:.4f}")
    log.info(f"     Recall     : {rec:.4f}")

    return {
        "rows": len(y_true), "base_rate": round(base, 6),
        "aucpr": round(aucpr, 4), "auroc": round(auroc, 4),
        "lift": round(lift, 4), "best_threshold": round(float(thr), 4),
        "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4),
    }


def run_trial(params, dtrain, dval, y_val, trial_name):
    t0 = time.time()
    model = xgb.train(
        params=params, dtrain=dtrain, num_boost_round=3000,
        evals=[(dval, "val")], early_stopping_rounds=60, verbose_eval=False,
    )
    elapsed = time.time() - t0
    best_round = model.best_iteration
    val_prob = model.predict(dval)
    val_auroc = roc_auc_score(y_val, val_prob)
    val_aupr = average_precision_score(y_val, val_prob)

    log.info(
        f"  {trial_name:<14} depth={params['max_depth']} mcw={params['min_child_weight']:>2} "
        f"lr={params['learning_rate']:.3f} sub={params['subsample']} "
        f"| rounds={best_round + 1:>4} val-AUROC={val_auroc:.4f} "
        f"val-AUPR={val_aupr:.4f} ({elapsed:.0f}s)"
    )
    return {
        "trial": trial_name, "max_depth": params["max_depth"],
        "min_child_weight": params["min_child_weight"],
        "learning_rate": params["learning_rate"], "subsample": params["subsample"],
        "best_round": best_round + 1, "val_auroc": val_auroc, "val_aupr": val_aupr,
        "elapsed_s": round(elapsed, 1),
    }


def run():
    t_start = time.time()
    log.info("=" * 70)
    log.info("  TEXAS 32-Feature Wildfire Ignition Model -- Tuned")
    log.info("=" * 70)

    log.info(f"[1/7] Loading dataset: {DATA}")
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"])
    log.info(f"      Total rows: {len(df):,}  Columns: {len(df.columns)}")

    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    log.info(f"      All {len(FEATURES)} features verified present.")

    n_future = (df["date"] > TEST_END).sum()
    log.info(f"      Excluding {n_future:,} rows dated after {TEST_END} "
              f"(fabricated placeholder rows per DATA_QUALITY_REVIEW.md)")

    log.info(f"[2/7] Splitting data by date ...")
    train_df = df[df["date"] <= TRAIN_END].copy()
    val_df = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
    test_df = df[(df["date"] > VAL_END) & (df["date"] <= TEST_END)].copy()
    log.info(f"      Train : {len(train_df):,} rows  pos_rate={train_df[TARGET].mean():.3f}")
    log.info(f"      Val   : {len(val_df):,} rows  pos_rate={val_df[TARGET].mean():.3f}")
    log.info(f"      Test  : {len(test_df):,} rows  pos_rate={test_df[TARGET].mean():.3f}")

    X_train, y_train = train_df[FEATURES].values, train_df[TARGET].values.astype(int)
    X_val, y_val = val_df[FEATURES].values, val_df[TARGET].values.astype(int)
    X_test, y_test = test_df[FEATURES].values, test_df[TARGET].values.astype(int)

    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    log.info(f"      scale_pos_weight = {spw:.4f}")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURES)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURES)

    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=5)
        device = "cuda" if r.stdout.strip() else "cpu"
        log.info(f"      GPU: {r.stdout.strip() or 'none detected'}")
    except Exception:
        device = "cpu"
    log.info(f"      Device: {device.upper()}")

    base_params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "tree_method": "hist", "device": device,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "gamma": 0.1,
        "colsample_bytree": 0.8, "colsample_bylevel": 0.8,
        "scale_pos_weight": spw,
    }

    # -- Stage 1: tree structure ----------------------------------------------
    log.info(f"\n[3/7] STAGE 1 -- max_depth x min_child_weight (12 trials)")
    all_results = []
    for depth, mcw in product([6, 7, 8, 9], [10, 20, 30]):
        params = {**base_params, "max_depth": depth, "min_child_weight": mcw,
                  "learning_rate": 0.05, "subsample": 0.8}
        all_results.append(run_trial(params, dtrain, dval, y_val, f"d{depth}_mcw{mcw}"))

    best_s1 = max(all_results, key=lambda x: x["val_auroc"])
    best_depth, best_mcw = best_s1["max_depth"], best_s1["min_child_weight"]
    log.info(f"      Stage 1 best: depth={best_depth} mcw={best_mcw} "
              f"val-AUROC={best_s1['val_auroc']:.4f}")

    # -- Stage 2: learning rate + subsample ------------------------------------
    log.info(f"\n[4/7] STAGE 2 -- learning_rate x subsample (9 trials, "
              f"depth={best_depth} mcw={best_mcw})")
    stage2 = []
    for lr, sub in product([0.01, 0.02, 0.05], [0.7, 0.8, 0.9]):
        params = {**base_params, "max_depth": best_depth, "min_child_weight": best_mcw,
                  "learning_rate": lr, "subsample": sub}
        r = run_trial(params, dtrain, dval, y_val, f"lr{lr}_s{sub}")
        all_results.append(r)
        stage2.append(r)

    best = max(all_results, key=lambda x: x["val_auroc"])
    log.info(f"\n      BEST CONFIG: depth={best['max_depth']} mcw={best['min_child_weight']} "
              f"lr={best['learning_rate']} sub={best['subsample']} "
              f"val-AUROC={best['val_auroc']:.4f} val-AUPR={best['val_aupr']:.4f}")

    # -- Final retrain ----------------------------------------------------------
    log.info(f"\n[5/7] FINAL -- retrain best config (up to 3000 rounds)")
    final_params = {**base_params, "max_depth": best["max_depth"],
                     "min_child_weight": best["min_child_weight"],
                     "learning_rate": best["learning_rate"], "subsample": best["subsample"]}
    final_model = xgb.train(
        params=final_params, dtrain=dtrain, num_boost_round=3000,
        evals=[(dtrain, "train"), (dval, "val")], early_stopping_rounds=80, verbose_eval=200,
    )
    best_iter = final_model.best_iteration
    log.info(f"      Best iteration: {best_iter}")

    # -- Evaluate raw ------------------------------------------------------------
    log.info(f"\n[6/7] Evaluating raw model scores ...")
    p_train = final_model.predict(dtrain)
    p_val = final_model.predict(dval)
    p_test = final_model.predict(dtest)
    res_train_raw = evaluate(y_train, p_train, "TRAIN (raw)")
    res_val_raw = evaluate(y_val, p_val, "VAL   (raw)")
    res_test_raw = evaluate(y_test, p_test, "TEST  (raw)")

    # -- Calibrate ----------------------------------------------------------------
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(p_val, y_val)
    p_val_cal = calibrator.predict(p_val)
    p_test_cal = calibrator.predict(p_test)
    res_val_cal = evaluate(y_val, p_val_cal, "VAL  (calibrated)")
    res_test_cal = evaluate(y_test, p_test_cal, "TEST (calibrated)")

    # -- Feature importance ---------------------------------------------------------
    log.info(f"\n[7/7] Computing feature importance (gain) ...")
    imp = final_model.get_score(importance_type="gain")
    imp = {f: imp.get(f, 0.0) for f in FEATURES}
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    total_gain = sum(v for _, v in imp_sorted) or 1.0
    log.info("      Top 15 features by gain:")
    for rank, (feat, gain) in enumerate(imp_sorted[:15], 1):
        log.info(f"       {rank:>2}. {feat:<20} gain={gain:>9.1f}  ({100*gain/total_gain:5.2f}%)")

    lat_lon_rank = [i for i, (f, _) in enumerate(imp_sorted, 1) if f in ("lat", "lon")]
    log.info(f"      NEW features (lat, lon) ranks: {lat_lon_rank} of {len(FEATURES)}")

    # -- Save ---------------------------------------------------------------------
    final_model.save_model(str(MODEL_OUT))
    joblib.dump(calibrator, str(CAL_OUT))
    pd.DataFrame(all_results).sort_values("val_auroc", ascending=False).to_csv(TUNING_CSV, index=False)

    elapsed = round(time.time() - t_start, 2)
    d_aucpr = res_test_raw["aucpr"] - BASELINE_30FEAT["aucpr"]
    d_auroc = res_test_raw["auroc"] - BASELINE_30FEAT["auroc"]

    results = {
        "dataset": str(DATA), "features": FEATURES, "n_features": len(FEATURES),
        "new_features_vs_30feat_baseline": ["lat", "lon"],
        "split": {
            "train": {"end": TRAIN_END, "rows": int(len(train_df))},
            "val": {"end": VAL_END, "rows": int(len(val_df))},
            "test": {"end": TEST_END, "rows": int(len(test_df))},
        },
        "n_trials": len(all_results),
        "best_config": {
            "max_depth": best["max_depth"], "min_child_weight": best["min_child_weight"],
            "learning_rate": best["learning_rate"], "subsample": best["subsample"],
            "colsample_bytree": 0.8, "best_iteration": int(best_iter),
            "scale_pos_weight": round(spw, 4),
        },
        "scores_raw": {"train": res_train_raw, "val": res_val_raw, "test": res_test_raw},
        "scores_calibrated": {"val": res_val_cal, "test": res_test_cal},
        "feature_importance_gain": {k: round(v, 4) for k, v in imp_sorted},
        "lat_lon_importance_rank": lat_lon_rank,
        "comparison": {
            "tdis_shared_model": TDIS_SHARED,
            "our_30feat_untuned_baseline": BASELINE_30FEAT,
            "this_32feat_tuned_model_test_raw": {
                "aucpr": res_test_raw["aucpr"], "auroc": res_test_raw["auroc"], "f1": res_test_raw["f1"],
            },
            "delta_vs_30feat_baseline": {"aucpr": round(d_aucpr, 4), "auroc": round(d_auroc, 4)},
            "delta_vs_tdis_shared": {
                "aucpr": round(res_test_raw["aucpr"] - TDIS_SHARED["aucpr"], 4),
                "auroc": round(res_test_raw["auroc"] - TDIS_SHARED["auroc"], 4),
            },
        },
        "elapsed_seconds": elapsed,
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)

    log.info("\n" + "=" * 70)
    log.info(f"  TRAINING COMPLETE in {elapsed:.1f}s  ({len(all_results)} tuning trials)")
    log.info(f"  {'Metric':<12} {'TDIS shared':>12} {'30feat base':>12} {'32feat tuned':>13} {'vs 30feat':>10}")
    log.info(f"  {'AUC-PR':<12} {TDIS_SHARED['aucpr']:>12.4f} {BASELINE_30FEAT['aucpr']:>12.4f} "
              f"{res_test_raw['aucpr']:>13.4f} {d_aucpr:>+10.4f}")
    log.info(f"  {'AUROC':<12} {TDIS_SHARED['auroc']:>12.4f} {BASELINE_30FEAT['auroc']:>12.4f} "
              f"{res_test_raw['auroc']:>13.4f} {d_auroc:>+10.4f}")
    log.info(f"  {'F1':<12} {TDIS_SHARED['f1']:>12.4f} {BASELINE_30FEAT['f1']:>12.4f} "
              f"{res_test_raw['f1']:>13.4f} {res_test_raw['f1']-BASELINE_30FEAT['f1']:>+10.4f}")
    log.info("=" * 70)


if __name__ == "__main__":
    run()
