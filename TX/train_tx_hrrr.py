"""
train_tx_hrrr.py
-----------------
Texas Wildfire Ignition Model — HRRR-Enriched Training

Same pipeline as train_tx.py (identical preprocessing, chronological
split, XGBoost params) but reads data/hrrr/hrrr_tx_all.parquet, which
adds 7 HRRR sub-daily features on top of the gridMET-only feature set:

  temp_pw        2m temperature [C]              (analysis, all years)
  rh_pw          2m relative humidity [%]         (NaN where hrrr_rh_valid=0)
  wind_pw        10m wind speed [m/s]
  vpd_pw_hrrr    vapor pressure deficit [kPa]     (NaN where hrrr_rh_valid=0)
  hpbl_pw        planetary boundary layer height [m]
  dswrf_pw       downward solar radiation [W/m2]
  hrrr_pw        1 = HRRR available for this row, 0 = not (missing timestamp)
  hrrr_rh_valid  1 = rh_pw/vpd_pw_hrrr are real RH-based values, 0 = unavailable
                 (see fix_hrrr_rh_bug.py — some 2014-2016 files only had
                 specific humidity, not RH, and that bug is corrected upstream)

XGBoost handles NaN natively (same missing=np.nan strategy as train_tx.py),
so hrrr_pw=0 / hrrr_rh_valid=0 rows aren't dropped — the model learns to
fall back on gridMET features when HRRR is unavailable.

Run fix_hrrr_rh_bug.py once before this script (idempotent — safe to
run again, it's a no-op the second time).

Usage:
    python train_tx_hrrr.py              # GPU if available, else CPU
    python train_tx_hrrr.py --no-gpu     # force CPU
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
ROOT     = Path(__file__).resolve().parent
HRRR_PQ  = ROOT / "data" / "hrrr" / "hrrr_tx_all.parquet"
OUT_DIR  = ROOT / "outputs" / "texas_landfire"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "models").mkdir(exist_ok=True)

# Compare HRRR model against the TUNED gridMET-only model (V3), not the untuned baseline
BASELINE_META = OUT_DIR / "models" / "xgb_tx_tuned_meta.json"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "train_tx_hrrr.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Features ───────────────────────────────────────────────────────────────────
# Same base set as train_tx.py (gridMET + landscape + temporal), plus HRRR block.
BASE_FEATURE_COLS = [
    "avg_burn_prob", "whp", "flep4", "cfl", "cbd", "cbh", "burnable",
    "erc", "fm100", "vpd", "vs", "rmax", "rmin", "tmmx", "pr",
    "erc_5D_mean",  "erc_5D_max",
    "fm100_5D_mean","fm100_5D_min",
    "vpd_5D_mean",  "vpd_5D_max",
    "vs_5D_mean",   "vs_5D_max",
    "rmax_5D_mean", "rmax_5D_min",
    "tmmx_5D_mean", "tmmx_5D_max",
    "sin_month", "cos_month", "sin_hour", "cos_hour",
    "centroid_lat", "centroid_lon",
    "gridmet_missing",
]

HRRR_FEATURE_COLS = [
    "temp_pw", "rh_pw", "wind_pw", "vpd_pw_hrrr",
    "hpbl_pw", "dswrf_pw", "hrrr_pw", "hrrr_rh_valid",
]

# Availability flags — not real weather readings. Coverage climbs from 12% (2014)
# to 99% (2020), so these can act as a year proxy rather than genuine fire signal.
# --no-flags drops them to test whether they were actually contributing.
AVAILABILITY_FLAGS = ["hrrr_pw", "hrrr_rh_valid"]


# ── Load ───────────────────────────────────────────────────────────────────────
def load_data(drop_flags: bool):
    log.info("=" * 65)
    log.info("STEP 1 — LOADING")
    log.info("=" * 65)
    if not HRRR_PQ.exists():
        log.error(f"{HRRR_PQ} not found. Run merge_hrrr_duckdb.py first.")
        sys.exit(1)

    df = pd.read_parquet(HRRR_PQ)
    log.info(f"  Loaded: {len(df):,} rows x {len(df.columns)} columns")

    if "hrrr_rh_valid" not in df.columns:
        log.error("  'hrrr_rh_valid' column missing — run fix_hrrr_rh_bug.py first.")
        sys.exit(1)

    log.info(f"  hrrr_pw=1 (HRRR available):      {(df['hrrr_pw']==1).sum():,}  "
             f"({100*(df['hrrr_pw']==1).mean():.1f}%)")
    log.info(f"  hrrr_rh_valid=1 (RH/VPD usable):  {(df['hrrr_rh_valid']==1).sum():,}  "
             f"({100*(df['hrrr_rh_valid']==1).mean():.1f}%)")

    hrrr_cols = [c for c in HRRR_FEATURE_COLS if c not in AVAILABILITY_FLAGS] if drop_flags else HRRR_FEATURE_COLS
    if drop_flags:
        log.info(f"  --no-flags: dropping availability flags {AVAILABILITY_FLAGS} from feature set")
    feature_cols = BASE_FEATURE_COLS + hrrr_cols

    features = [c for c in feature_cols if c in df.columns]
    absent   = [c for c in feature_cols if c not in df.columns]
    log.info(f"  Features present: {len(features)}/{len(feature_cols)}")
    if absent:
        log.warning(f"  Features NOT in dataset: {absent}")

    return df, features, hrrr_cols


# ── Split + Train ─────────────────────────────────────────────────────────────
def split_and_train(df: pd.DataFrame, features: list[str], use_gpu: bool,
                     hrrr_cols: list[str], drop_flags: bool):
    log.info("\nSTEP 2 — CHRONOLOGICAL SPLIT  (using existing _split column)")
    train_df = df[df["_split"] == "train"].reset_index(drop=True)
    val_df   = df[df["_split"] == "val"].reset_index(drop=True)
    test_df  = df[df["_split"] == "test"].reset_index(drop=True)

    for sdf, name in [(train_df, "TRAIN"), (val_df, "VAL"), (test_df, "TEST")]:
        np_ = int((sdf["label"] == 1).sum())
        log.info(f"  {name:<6}: {len(sdf):>8,} rows  fire={np_:,}  rate={100*np_/len(sdf):.1f}%")

    def to_xy(d: pd.DataFrame):
        X = d[features].copy()
        for col in X.select_dtypes(include="object").columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(np.float32)
        return X, d["label"].values.astype(np.int8)

    X_train, y_train = to_xy(train_df)
    X_val,   y_val   = to_xy(val_df)
    X_test,  y_test  = to_xy(test_df)

    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    log.info(f"\n  scale_pos_weight = {spw:.2f}")

    device = "cuda" if use_gpu else "cpu"
    # Tuned hyperparameters from tune_tx.py Stage 1+2 winner:
    #   depth=9, mcw=30, lr=0.01, subsample=0.9  (Val AUROC 0.8856 on gridMET-only)
    # Applying the same config here to HRRR-enriched features for a fair comparison.
    params = {
        "objective":          "binary:logistic",
        "eval_metric":        ["logloss", "auc"],
        "tree_method":        "hist",
        "device":             device,
        "max_depth":          9,
        "min_child_weight":   30,
        "subsample":          0.9,
        "colsample_bytree":   0.8,
        "colsample_bylevel":  0.8,
        "learning_rate":      0.01,
        "gamma":              0.1,
        "reg_alpha":          0.1,
        "reg_lambda":         1.0,
        "scale_pos_weight":   spw,
    }

    log.info(f"\n{'='*65}")
    log.info(f"STEP 3 — TRAINING  (device={device.upper()}  "
             f"features={len(features)}  train={len(y_train):,})")
    log.info(f"{'='*65}")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features, missing=np.nan)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=features, missing=np.nan)
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=features, missing=np.nan)

    evals_result = {}
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    best = model.best_iteration
    log.info(f"\n  Best round:  {best+1}")
    log.info(f"  Val AUC:     {evals_result['val']['auc'][best]:.4f}")

    val_prob = model.predict(dval)
    prec, rec, thr = precision_recall_curve(y_val, val_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    threshold = float(thr[np.argmax(f1s[:-1])])
    log.info(f"  Threshold:   {threshold:.4f}  (max F1 on val set)")

    log.info(f"\n{'─'*65}")
    log.info("STEP 4 — EVALUATION")
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
            "split": sname.strip(), "n": len(y_true), "n_pos": int(y_true.sum()),
            "auroc": round(auroc, 4), "aupr": round(aupr, 4),
            f"prec_at_top{k//1000}k": round(preck, 4),
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "threshold": round(threshold, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        all_metrics.append(m)
        log.info(f"  {sname}  AUROC={auroc:.4f}  AUPR={aupr:.4f}  F1={f1:.4f}  P={p:.3f}  R={r:.3f}")

    test_m = all_metrics[2]

    # Compare vs existing gridMET-only baseline
    baseline_test_auroc = baseline_test_aupr = None
    if BASELINE_META.exists():
        with open(BASELINE_META, encoding="utf-8") as f:
            base_meta = json.load(f)
        # xgb_tx_tuned_meta.json uses "test_metrics"; xgb_tx_landfire_meta.json uses "metrics"
        metrics_key = "test_metrics" if "test_metrics" in base_meta else "metrics"
        base_test = next(m for m in base_meta[metrics_key] if m["split"] == "TEST")
        baseline_test_auroc = base_test["auroc"]
        baseline_test_aupr  = base_test["aupr"]
        d_auroc = test_m["auroc"] - baseline_test_auroc
        d_aupr  = test_m["aupr"]  - baseline_test_aupr
        log.info(f"\n  vs tuned gridMET-only V3 (xgb_tx_tuned):  "
                 f"AUROC {d_auroc:+.4f}  |  AUPR {d_aupr:+.4f}")

    # Feature importance
    imp = model.get_score(importance_type="gain")
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    total_gain = sum(v for _, v in imp_sorted)
    log.info(f"\n  TOP 15 FEATURES (gain importance):")
    for feat, gain in imp_sorted[:15]:
        tag = "  [HRRR]" if feat in hrrr_cols else ""
        bar = chr(9608) * int(gain / imp_sorted[0][1] * 25)
        log.info(f"  {feat:<20} {gain:>9.1f}  ({100*gain/total_gain:4.1f}%)  {bar}{tag}")

    hrrr_gain_share = sum(g for f, g in imp_sorted if f in hrrr_cols) / total_gain
    log.info(f"\n  HRRR features share of total gain: {100*hrrr_gain_share:.1f}%")

    # Save (distinct filenames when flags are dropped, so the original run isn't overwritten)
    suffix = "_noflags" if drop_flags else ""
    model_path = OUT_DIR / "models" / f"xgb_tx_hrrr{suffix}.ubj"
    meta_path  = OUT_DIR / "models" / f"xgb_tx_hrrr{suffix}_meta.json"
    model.save_model(str(model_path))

    meta = {
        "state": "TX",
        "dataset": str(HRRR_PQ.name),
        "features": features,
        "n_features": len(features),
        "hrrr_features": hrrr_cols,
        "availability_flags_dropped": drop_flags,
        "best_round": best + 1,
        "threshold": threshold,
        "params": params,
        "metrics": all_metrics,
        "hrrr_gain_share": round(hrrr_gain_share, 4),
        "vs_gridmet_baseline": {
            "baseline_test_auroc": baseline_test_auroc,
            "baseline_test_aupr":  baseline_test_aupr,
            "auroc_delta": round(test_m["auroc"] - baseline_test_auroc, 4) if baseline_test_auroc else None,
            "aupr_delta":  round(test_m["aupr"]  - baseline_test_aupr, 4) if baseline_test_aupr else None,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info(f"\n{'='*65}")
    log.info("  TRAINING COMPLETE")
    log.info(f"{'='*65}")
    log.info(f"  TEST AUROC:  {test_m['auroc']:.4f}"
             + (f"  (gridMET-only: {baseline_test_auroc:.4f})" if baseline_test_auroc else ""))
    log.info(f"  TEST AUPR:   {test_m['aupr']:.4f}"
             + (f"  (gridMET-only: {baseline_test_aupr:.4f})" if baseline_test_aupr else ""))
    log.info(f"  Model:       {model_path}")
    log.info(f"  Metadata:    {meta_path}")
    log.info(f"{'='*65}")

    return model, all_metrics


def main():
    parser = argparse.ArgumentParser(description="Texas Wildfire Model — HRRR-Enriched Training")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU")
    parser.add_argument("--no-flags", action="store_true",
                         help="Drop hrrr_pw/hrrr_rh_valid availability flags "
                              "(ablation: are they real signal or a year-coverage proxy?)")
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

    df, features, hrrr_cols = load_data(args.no_flags)
    split_and_train(df, features, use_gpu, hrrr_cols, args.no_flags)


if __name__ == "__main__":
    main()
