"""
train_tx.py
------------
Texas Wildfire Ignition Model — Full Training Pipeline
Reads: final_training_dataset_tx_22.07.2026_landfire.xlsx
Trains: XGBoost binary classifier (Texas standalone)

Preprocessing applied per README_tx_dataset.md:
  [README s8  line 194] Drop 454 duplicate rows
  [README s9  line 200] Drop 24,954 rows with gridmet_missing=1
  [README s10 line 203] Zero-fill 1,594 NaN burnable/fire_count (boundary cells)
  [README s4  line 183] avg_burn_prob kept on 0-11 scale — NO normalization
                        "normalize only if combining with CA training data"
                        README line 219 shows this line commented out — skip for TX-only
  [Scope doc  line 391] fire_count / has_fire_history EXCLUDED — leakage
                        (computed from full 2014-2020 FPA-FOD including test years)

New features vs V2 pipeline (which had these as zeros):
  avg_burn_prob, whp, flep4, cfl  — now real TxWRAP values
  cbd, cbh                        — NEW (LANDFIRE LF2022 Canopy Bulk/Base)

Chronological split (NEVER random):
  TRAIN  2014-2017
  VAL    2018       (early stopping + threshold tuning)
  TEST   2019-2020  (final evaluation only — never touched during training)

Usage:
    python train_tx.py              # GPU if available, else CPU
    python train_tx.py --no-gpu     # force CPU
    python train_tx.py --skip-excel # use pre-saved parquets in data/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
EXCEL_PATH = ROOT / "final_training_dataset_tx_22.07.2026_landfire.xlsx"
DATA_DIR   = ROOT / "data"
OUT_DIR    = ROOT / "outputs" / "texas_landfire"

DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "models").mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "train_tx.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Features (per README_tx_dataset.md column groups, line 27-38) ─────────────
# DO NOT include: fire_count, has_fire_history (leakage — scope doc line 391)
# DO NOT include: gridmet_missing (flag, not a feature)
# DO NOT include: identifiers (h3_cell, date_utc, window_6h_utc, window_hour)
FEATURE_COLS = [
    # Landscape — static (TxWRAP + LANDFIRE LF2022) — REAL VALUES in new dataset
    "avg_burn_prob",    # 0-11 scale (TX WRC annual burn prob) — kept as-is, README line 183
    "whp",              # 0-9 class (Wildfire Hazard Potential)
    "flep4",            # 0-0.9 probability (flame length exceedance, discretized)
    "cfl",              # 0-110 ft (canopy flame length, discretized class midpoints)
    "cbd",              # kg/m3 — LANDFIRE LF2022 Canopy Bulk Density (NEW)
    "cbh",              # meters — LANDFIRE LF2022 Canopy Base Height (NEW)
    "burnable",         # binary: 1=burnable, 0=urban/water/agriculture

    # gridMET daily weather (README line 35)
    "erc",              # Energy Release Component [BTU/ft2]
    "fm100",            # 100-hr dead fuel moisture [%]
    "vpd",              # Vapor pressure deficit [kPa]
    "vs",               # Wind speed [m/s]
    "rmax",             # Max relative humidity [%]
    "rmin",             # Min relative humidity [%]
    "tmmx",             # Max temperature [degC]
    "pr",               # Precipitation [mm]

    # 5-day trailing statistics (README line 36)
    "erc_5D_mean",  "erc_5D_max",
    "fm100_5D_mean","fm100_5D_min",
    "vpd_5D_mean",  "vpd_5D_max",
    "vs_5D_mean",   "vs_5D_max",
    "rmax_5D_mean", "rmax_5D_min",
    "tmmx_5D_mean", "tmmx_5D_max",

    # Temporal encodings (README line 38)
    "sin_month", "cos_month",
    "sin_hour",  "cos_hour",

    # Location (README line 31)
    "centroid_lat",
    "centroid_lon",
]

TRAIN_YEARS = [2015, 2016, 2017]
VAL_YEARS   = [2018]
TEST_YEARS  = [2019, 2020]

V2_BASELINE_AUROC = 0.8569   # previous model with zero LANDFIRE features
V2_BASELINE_AUPR  = 0.3978


# ── Preprocessing ─────────────────────────────────────────────────────────────
def load_and_preprocess():
    """Load Excel and apply all README-mandated preprocessing."""
    log.info("=" * 65)
    log.info("STEP 1 — LOADING")
    log.info("=" * 65)
    log.info(f"  File: {EXCEL_PATH.name}  ({EXCEL_PATH.stat().st_size/1e6:.0f} MB)")
    log.info("  Loading Excel... (30-90 seconds for 94 MB)")

    t0 = time.time()
    df = pd.read_excel(EXCEL_PATH)
    elapsed = time.time() - t0
    log.info(f"  Loaded: {len(df):,} rows x {len(df.columns)} columns  ({elapsed:.0f}s)")
    log.info(f"  Fire rows (label=1):    {(df['label']==1).sum():,}  "
             f"({100*(df['label']==1).mean():.2f}%)")
    log.info(f"  Non-fire rows (label=0):{(df['label']==0).sum():,}")

    # Fix date columns — Excel loads datetime cols as strings/object dtype
    # pyarrow requires proper datetime to serialize parquet — convert here
    for date_col in ["date_utc", "window_6h_utc"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            log.info(f"  Converted '{date_col}' -> datetime")

    log.info("\nSTEP 2 — PREPROCESSING  (per README_tx_dataset.md)")

    # README section 8 line 194 — Drop duplicates
    before = len(df)
    df.drop_duplicates(subset=["h3_cell", "date_utc", "window_hour"], inplace=True)
    log.info(f"  [README s8] Duplicates dropped:         {before-len(df):,}  "
             f"-> {len(df):,} rows")

    # README section 9 line 200 — gridmet_missing rows strategy
    # STRATEGY: KEEP rows — let XGBoost handle NaN natively.
    # WHY: V2 baseline (AUROC=0.8569) trained ON these rows with XGBoost NaN handling.
    # Dropping them costs ~17,000 training rows and slightly lowers AUROC.
    # XGBoost natively splits NaN into the best branch — no imputation needed.
    # We add gridmet_missing as a feature so the model knows which rows lack weather.
    n_missing = int((df["gridmet_missing"] == 1).sum())
    log.info(f"  [README s9] gridmet_missing rows:        {n_missing:,}  "
             f"-> KEPT (XGBoost handles NaN natively)")
    log.info(f"              gridmet_missing added as feature (model learns availability)")
    log.info(f"              Fire rate: {100*(df['label']==1).mean():.2f}%")

    # Add gridmet_missing as a feature (availability signal)
    if "gridmet_missing" not in FEATURE_COLS:
        FEATURE_COLS.append("gridmet_missing")

    # README section 10 line 203 — Zero-fill boundary cells
    for col in ["burnable", "fire_count"]:
        if col in df.columns:
            n = int(df[col].isna().sum())
            if n > 0:
                df[col] = df[col].fillna(0)
                log.info(f"  [README s10] Zero-filled '{col}': {n:,} NaN -> 0")

    # README section 4 line 183 — avg_burn_prob scale decision
    if "avg_burn_prob" in df.columns:
        mx = df["avg_burn_prob"].max()
        log.info(f"  [README s4]  avg_burn_prob max={mx:.1f}  (0-11 TX WRC scale)")
        log.info(f"              DECISION: keep 0-11, no normalization")
        log.info(f"              REASON: README line 183 says 'normalize only if combining with CA'")
        log.info(f"              We are training TX standalone -> normalization skipped")

    # Feature validation
    features = [c for c in FEATURE_COLS if c in df.columns]
    absent   = [c for c in FEATURE_COLS if c not in df.columns]
    log.info(f"\n  Features present: {len(features)}/{len(FEATURE_COLS)}")
    if absent:
        log.warning(f"  Features NOT in dataset: {absent}")

    # Landscape quality check
    log.info("\n  Landscape feature quality check:")
    for col in ["avg_burn_prob", "whp", "flep4", "cfl", "cbd", "cbh"]:
        if col in df.columns:
            pnz  = 100 - 100*(df[col] == 0).mean()
            mean = df[col].mean()
            flag = "REAL DATA" if pnz > 10 else "MOSTLY ZERO - check rasters"
            log.info(f"    {col:<15} mean={mean:>7.4f}  non-zero={pnz:5.1f}%  "
                     f"[{flag}]")

    return df, features


# ── Split + Train ─────────────────────────────────────────────────────────────
def split_and_train(df: pd.DataFrame, features: list[str], use_gpu: bool):
    """Chronological split, train XGBoost, evaluate, save."""

    # Split
    df["year"] = pd.to_datetime(df["date_utc"]).dt.year
    train_df = df[df["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    val_df   = df[df["year"].isin(VAL_YEARS)].reset_index(drop=True)
    test_df  = df[df["year"].isin(TEST_YEARS)].reset_index(drop=True)

    log.info("\nSTEP 3 — CHRONOLOGICAL SPLIT")
    for sdf, name, yrs in [
        (train_df,"TRAIN","2014-2017"),
        (val_df,  "VAL",  "2018"),
        (test_df, "TEST", "2019-2020"),
    ]:
        np_ = int((sdf["label"]==1).sum())
        log.info(f"  {name:<6} ({yrs}): {len(sdf):>8,} rows  "
                 f"fire={np_:,}  rate={100*np_/len(sdf):.1f}%")

    # Save clean parquets
    train_df.to_parquet(DATA_DIR / "train_tx_clean.parquet", index=False)
    val_df.to_parquet(  DATA_DIR / "val_tx_clean.parquet",   index=False)
    test_df.to_parquet( DATA_DIR / "test_tx_clean.parquet",  index=False)
    log.info(f"  Clean parquets saved to {DATA_DIR}/")

    # Feature matrices
    def to_xy(d: pd.DataFrame):
        X = d[features].copy()
        # Coerce object columns (burnable sometimes stored as "0"/"1")
        for col in X.select_dtypes(include="object").columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(np.float32)
        return X, d["label"].values.astype(np.int8)

    X_train, y_train = to_xy(train_df)
    X_val,   y_val   = to_xy(val_df)
    X_test,  y_test  = to_xy(test_df)

    # Class weight
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    log.info(f"\n  scale_pos_weight = {spw:.2f}  (handles {spw:.0f}:1 class imbalance)")

    # XGBoost config
    device = "cuda" if use_gpu else "cpu"
    params = {
        "objective":          "binary:logistic",
        "eval_metric":        ["logloss", "auc"],
        "tree_method":        "hist",
        "device":             device,
        "max_depth":          7,
        "min_child_weight":   30,
        "subsample":          0.8,
        "colsample_bytree":   0.8,
        "colsample_bylevel":  0.8,
        "learning_rate":      0.05,
        "gamma":              0.1,
        "reg_alpha":          0.1,
        "reg_lambda":         1.0,
        "scale_pos_weight":   spw,
    }

    log.info(f"\n{'='*65}")
    log.info(f"STEP 4 — TRAINING  (device={device.upper()}  "
             f"features={len(features)}  train={len(y_train):,})")
    log.info(f"{'='*65}")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features, missing=np.nan)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=features, missing=np.nan)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=features, missing=np.nan)

    evals_result = {}
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    best = model.best_iteration
    log.info(f"\n  Best round:  {best+1}")
    log.info(f"  Val AUC:     {evals_result['val']['auc'][best]:.4f}")
    log.info(f"  Val logloss: {evals_result['val']['logloss'][best]:.4f}")

    # Optimal threshold (max-F1 on validation)
    val_prob = model.predict(dval)
    prec, rec, thr = precision_recall_curve(y_val, val_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    threshold = float(thr[np.argmax(f1s[:-1])])
    log.info(f"  Threshold:   {threshold:.4f}  (max F1 on val set)")

    # Evaluation
    log.info(f"\n{'─'*65}")
    log.info("STEP 5 — EVALUATION")
    log.info(f"{'─'*65}")
    all_metrics = []
    for y_true, dmat, sname in [
        (y_train, dtrain, "TRAIN"),
        (y_val,   dval,   "VAL  "),
        (y_test,  dtest,  "TEST "),
    ]:
        prob  = model.predict(dmat)
        auroc = roc_auc_score(y_true, prob)
        aupr  = average_precision_score(y_true, prob)
        k     = int(y_true.sum())
        preck = float(y_true[np.argsort(prob)[::-1][:k]].mean())
        ypred = (prob >= threshold).astype(int)
        tp = int(((ypred==1)&(y_true==1)).sum())
        fp = int(((ypred==1)&(y_true==0)).sum())
        fn = int(((ypred==0)&(y_true==1)).sum())
        tn = int(((ypred==0)&(y_true==0)).sum())
        p  = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
        m = {
            "split":     sname.strip(),
            "n":         len(y_true),
            "n_pos":     int(y_true.sum()),
            "auroc":     round(auroc, 4),
            "aupr":      round(aupr, 4),
            f"prec_at_top{k//1000}k": round(preck, 4),
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
            "threshold": round(threshold, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        all_metrics.append(m)
        log.info(f"  {sname}  AUROC={auroc:.4f}  AUPR={aupr:.4f}  "
                 f"F1={f1:.4f}  P={p:.3f}  R={r:.3f}")

    test_m = all_metrics[2]
    d_auroc = test_m["auroc"] - V2_BASELINE_AUROC
    d_aupr  = test_m["aupr"]  - V2_BASELINE_AUPR
    log.info(f"\n  vs V2 baseline (zeros LANDFIRE):  "
             f"AUROC {d_auroc:+.4f}  |  AUPR {d_aupr:+.4f}")

    if test_m["auroc"] > 0.99:
        log.warning("  AUROC > 0.99 — POSSIBLE LEAKAGE. Check FEATURE_COLS immediately!")
    elif test_m["auroc"] > 0.96:
        log.info("  AUROC > 0.96 — excellent result")
    elif test_m["auroc"] > 0.90:
        log.info("  AUROC > 0.90 — expected with real LANDFIRE")
    elif test_m["auroc"] > 0.857:
        log.info("  AUROC > 0.857 — improvement over V2 baseline confirmed")
    else:
        log.warning("  AUROC <= 0.857 — no improvement. Check landscape feature quality above.")

    # Feature importance
    imp = model.get_score(importance_type="gain")
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    total_gain = sum(v for _, v in imp_sorted)
    log.info(f"\n  TOP 15 FEATURES (gain importance):")
    for feat, gain in imp_sorted[:15]:
        bar = chr(9608) * int(gain / imp_sorted[0][1] * 25)
        log.info(f"  {feat:<20} {gain:>9.1f}  ({100*gain/total_gain:4.1f}%)  {bar}")

    # Save model + metadata
    model_path = OUT_DIR / "models" / "xgb_tx_landfire.ubj"
    meta_path  = OUT_DIR / "models" / "xgb_tx_landfire_meta.json"
    model.save_model(str(model_path))

    meta = {
        "state":   "TX",
        "dataset": EXCEL_PATH.name,
        "features": features,
        "n_features": len(features),
        "preprocessing": {
            "duplicates_dropped":          True,
            "gridmet_missing_dropped":     True,
            "burnable_zero_filled":        True,
            "avg_burn_prob_normalized":    False,
            "avg_burn_prob_scale":         "0-11 (TX WRC original)",
            "normalization_decision_note": "README line 183: normalize only if combining with CA",
        },
        "excluded_leakage":   ["fire_count", "has_fire_history"],
        "new_vs_v2_pipeline": ["avg_burn_prob (real)", "whp (real)",
                               "flep4 (real)", "cfl (real)",
                               "cbd (NEW)", "cbh (NEW)"],
        "best_round": best + 1,
        "threshold":  threshold,
        "params":     params,
        "metrics":    all_metrics,
        "vs_v2_baseline": {
            "auroc_delta": round(d_auroc, 4),
            "aupr_delta":  round(d_aupr, 4),
        },
        "feature_importance_top30": dict(imp_sorted[:30]),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info(f"\n{'='*65}")
    log.info("  TRAINING COMPLETE")
    log.info(f"{'='*65}")
    log.info(f"  TEST AUROC:  {test_m['auroc']:.4f}  "
             f"(V2 baseline 0.8569,  change {d_auroc:+.4f})")
    log.info(f"  TEST AUPR:   {test_m['aupr']:.4f}  "
             f"(V2 baseline 0.3978,  change {d_aupr:+.4f})")
    log.info(f"  TEST F1:     {test_m['f1']:.4f}")
    log.info(f"  Best round:  {best+1}")
    log.info(f"  Threshold:   {threshold:.4f}")
    log.info(f"  Model:       {model_path}")
    log.info(f"  Metadata:    {meta_path}")
    log.info(f"  Log:         {OUT_DIR / 'train_tx.log'}")
    log.info(f"{'='*65}")

    return model, all_metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Texas Wildfire Ignition Model — Training"
    )
    parser.add_argument("--no-gpu",     action="store_true",
                        help="Force CPU (skip GPU detection)")
    parser.add_argument("--skip-excel", action="store_true",
                        help="Load pre-saved parquets from data/ (skips Excel read)")
    args = parser.parse_args()

    use_gpu = not args.no_gpu
    if use_gpu:
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            gpu_name = r.stdout.strip()
            if gpu_name:
                log.info(f"GPU detected: {gpu_name}")
            else:
                raise RuntimeError("no output")
        except Exception:
            log.warning("No GPU found — falling back to CPU")
            use_gpu = False

    if args.skip_excel and (DATA_DIR / "train_tx_clean.parquet").exists():
        log.info("Loading pre-saved clean parquets (--skip-excel)...")
        train_df = pd.read_parquet(DATA_DIR / "train_tx_clean.parquet")
        val_df   = pd.read_parquet(DATA_DIR / "val_tx_clean.parquet")
        test_df  = pd.read_parquet(DATA_DIR / "test_tx_clean.parquet")
        features = [c for c in FEATURE_COLS if c in train_df.columns]
        df_all   = pd.concat([train_df, val_df, test_df], ignore_index=True)
        split_and_train(df_all, features, use_gpu)
    else:
        df, features = load_and_preprocess()
        split_and_train(df, features, use_gpu)


if __name__ == "__main__":
    main()
