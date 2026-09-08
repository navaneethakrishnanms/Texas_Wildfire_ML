"""
train_model_30feat.py
----------------------
Train a 30-feature XGBoost wildfire ignition model on the imputed TEXAS dataset.

Dataset  : TEXAS/tdis_train_daily_imputed.parquet
Split    : Train (2014-2023) / Val (2024) / Test (2025 - 2026-07-29)
Features : 30 (Static + HRRR Forecast + GridMET Observed + Calendar)
Outputs  : TEXAS/model_30feat.json
           TEXAS/calibrator_30feat.joblib
           TEXAS/training_results_30feat.json

Usage (run from the Texas ML Wildfire root directory):
    python TEXAS/train_model_30feat.py

Or from within TEXAS/:
    python train_model_30feat.py
"""

import json
import time
import warnings
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

# ─────────────────────────────────────────────
# PATHS — all relative to this script's location
# ─────────────────────────────────────────────
HERE = Path(__file__).resolve().parent          # TEXAS/
DATA = HERE / "tdis_train_daily_imputed.parquet"
MODEL_OUT = HERE / "model_30feat.json"
CAL_OUT = HERE / "calibrator_30feat.joblib"
RESULTS_OUT = HERE / "training_results_30feat.json"

# ─────────────────────────────────────────────
# SPLIT BOUNDARIES
# ─────────────────────────────────────────────
TRAIN_END = "2023-12-31"
VAL_END   = "2024-12-31"
TEST_END  = "2026-07-29"    # last date with real ground-truth labels

# ─────────────────────────────────────────────
# 30 FEATURE COLUMNS
# ─────────────────────────────────────────────
FEATURES = [
    # Static Topography & Access (5)
    "elevation_m", "slope_deg", "aspect_deg", "road_dist_km", "ecoregion_id",

    # Static Fuels & Wildfire Hazard (6)
    "avg_burn_prob", "whp", "flep4", "cfl", "cbd", "cbh",

    # Infrastructure / Ignition Source (1)
    "powerline_dist_km",

    # HRRR Forecast Weather & Soil Moisture (4)
    "hrrr_tmp", "hrrr_vpd", "hrrr_wind", "hrrr_mstav",

    # Observed gridMET Daily Weather (8)
    "erc", "fm100", "vpd", "vs", "rmax", "rmin", "tmmx", "pr",

    # Temporal & Calendar (6)
    "sin_month", "cos_month", "sin_dow", "cos_dow", "is_weekend", "is_holiday",
]

TARGET = "label"


# ─────────────────────────────────────────────
# HELPER: find best F1 threshold
# ─────────────────────────────────────────────
def best_f1_threshold(y_true, y_prob, thresholds=None):
    """Sweep probability thresholds and return the one with highest F1."""
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


# ─────────────────────────────────────────────
# HELPER: evaluate predictions and print results
# ─────────────────────────────────────────────
def evaluate(y_true, y_prob, label=""):
    aucpr = float(average_precision_score(y_true, y_prob))
    auroc = float(roc_auc_score(y_true, y_prob))
    base  = float(y_true.mean())
    lift  = aucpr / base if base > 0 else 0.0
    thr, f1 = best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))

    print(f"\n  -- {label} --")
    print(f"     Rows       : {len(y_true):,}")
    print(f"     Base rate  : {base:.4f}")
    print(f"     AUC-PR     : {aucpr:.4f}")
    print(f"     AUROC      : {auroc:.4f}")
    print(f"     Lift       : {lift:.2f}x")
    print(f"     Best Thr   : {thr:.4f}")
    print(f"     F1         : {f1:.4f}")
    print(f"     Precision  : {prec:.4f}")
    print(f"     Recall     : {rec:.4f}")

    return {
        "rows": len(y_true),
        "base_rate": round(base, 6),
        "aucpr": round(aucpr, 4),
        "auroc": round(auroc, 4),
        "lift": round(lift, 4),
        "best_threshold": round(float(thr), 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
    }


# ─────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────
def run():
    t_start = time.time()

    # -- 1. Load dataset ----------------------
    print("=" * 60)
    print("  TEXAS 30-Feature Wildfire Ignition Model")
    print("=" * 60)
    print(f"\n[1/6] Loading dataset: {DATA}")
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"])
    print(f"      Total rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # -- 2. Verify all 30 features are present
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in dataset: {missing_cols}")
    print("      All 30 features verified present.")

    # -- 3. Temporal split --------------------
    print(f"\n[2/6] Splitting data by date ...")
    train_df = df[df["date"] <= TRAIN_END].copy()
    val_df   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
    test_df  = df[(df["date"] > VAL_END)   & (df["date"] <= TEST_END)].copy()

    print(f"      Train  : {len(train_df):,} rows  (up to {TRAIN_END})  pos_rate={train_df[TARGET].mean():.3f}")
    print(f"      Val    : {len(val_df):,} rows  (up to {VAL_END})  pos_rate={val_df[TARGET].mean():.3f}")
    print(f"      Test   : {len(test_df):,} rows  (up to {TEST_END})  pos_rate={test_df[TARGET].mean():.3f}")

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values.astype(int)
    X_val   = val_df[FEATURES].values
    y_val   = val_df[TARGET].values.astype(int)
    X_test  = test_df[FEATURES].values
    y_test  = test_df[TARGET].values.astype(int)

    # -- 4. Train XGBoost model ---------------
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"\n[3/6] Training XGBoost model (scale_pos_weight={spw:.2f}) ...")
    print("      This may take several minutes ...")

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=9,
        min_child_weight=30,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        tree_method="hist",
        device="cuda",           # Change to "cpu" if no GPU is available
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    best_iter = model.best_iteration
    print(f"      Best iteration: {best_iter}")

    # -- 5. Evaluate raw model ----------------
    print(f"\n[4/6] Evaluating raw model scores ...")
    p_train = model.predict_proba(X_train)[:, 1]
    p_val   = model.predict_proba(X_val)[:, 1]
    p_test  = model.predict_proba(X_test)[:, 1]

    res_train_raw = evaluate(y_train, p_train, "TRAIN (raw)")
    res_val_raw   = evaluate(y_val,   p_val,   "VAL   (raw)")
    res_test_raw  = evaluate(y_test,  p_test,  "TEST  (raw)")

    # -- 6. Calibrate on validation set -------
    print(f"\n[5/6] Fitting isotonic calibrator on validation set ...")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(p_val, y_val)
    print("      Calibrator fitted.")

    p_val_cal  = calibrator.predict(p_val)
    p_test_cal = calibrator.predict(p_test)

    print(f"\n      Calibrated Val mean:  {p_val_cal.mean():.4f}  (true base rate: {y_val.mean():.4f})")
    print(f"      Calibrated Test mean: {p_test_cal.mean():.4f}  (true base rate: {y_test.mean():.4f})")

    res_val_cal  = evaluate(y_val,  p_val_cal,  "VAL  (calibrated)")
    res_test_cal = evaluate(y_test, p_test_cal, "TEST (calibrated)")

    # -- 7. Feature importance ----------------
    print(f"\n[6/6] Computing feature importance ...")
    imp = {
        feat: round(float(score), 6)
        for feat, score in zip(FEATURES, model.feature_importances_)
    }
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    print("\n      Top 10 Features by Gain:")
    for rank, (feat, score) in enumerate(imp_sorted[:10], 1):
        print(f"       {rank:>2}. {feat:<25}  {score:.4f}")

    # -- 8. Save outputs ----------------------
    model.save_model(str(MODEL_OUT))
    joblib.dump(calibrator, str(CAL_OUT))
    print(f"\n      Model saved      -> {MODEL_OUT}")
    print(f"      Calibrator saved -> {CAL_OUT}")

    elapsed = round(time.time() - t_start, 2)

    results = {
        "dataset": str(DATA),
        "features": FEATURES,
        "n_features": len(FEATURES),
        "split": {
            "train": {"end": TRAIN_END, "rows": int(len(train_df))},
            "val":   {"end": VAL_END,   "rows": int(len(val_df))},
            "test":  {"end": TEST_END,  "rows": int(len(test_df))},
        },
        "model_params": {
            "n_estimators": 1000,
            "best_iteration": int(best_iter),
            "max_depth": 9,
            "min_child_weight": 30,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": round(spw, 4),
        },
        "scores_raw": {
            "train": res_train_raw,
            "val":   res_val_raw,
            "test":  res_test_raw,
        },
        "scores_calibrated": {
            "val":  res_val_cal,
            "test": res_test_cal,
        },
        "feature_importance": dict(imp_sorted),
        "elapsed_seconds": elapsed,
    }

    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"      Results saved    -> {RESULTS_OUT}")

    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE in {elapsed:.1f} seconds")
    print(f"  Test AUC-PR (raw)  : {res_test_raw['aucpr']:.4f}")
    print(f"  Test AUROC  (raw)  : {res_test_raw['auroc']:.4f}")
    print(f"  Test AUC-PR (cal)  : {res_test_cal['aucpr']:.4f}")
    print(f"  Test AUROC  (cal)  : {res_test_cal['auroc']:.4f}")
    print(f"  Test F1     (cal)  : {res_test_cal['f1']:.4f}")
    print("=" * 60)


# ─────────────────────────────────────────────
# ENTRY POINT — only runs when called from terminal
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run()
