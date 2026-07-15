"""
run_phase3_train.py
--------------------
Phase 3 — XGBoost Baseline Training  [PRODUCTION]

Trains an XGBoost binary classifier on the final assembled training dataset.

Feature strategy (per TEAM_DATA_GUIDE.MD):
  - Landscape (LANDFIRE): avg_burn_prob, whp, flep4, cfl
    → If rasters not downloaded, these will be ~0 → model treats them as "low"
    → After downloading rasters, retrain for full model
  - gridMET daily: erc, fm100, vpd, vs, rmax, rmin, tmmx, pr
  - gridMET 5D stats: erc_5D_mean, erc_5D_max, fm100_5D_mean, fm100_5D_min,
                      vpd_5D_mean, vpd_5D_max, vs_5D_mean, vs_5D_max,
                      rmax_5D_mean, rmax_5D_min, tmmx_5D_mean, tmmx_5D_max
  - Temporal: sin_month, cos_month, sin_hour, cos_hour
  - Location: centroid_lat, centroid_lon
  - History: fire_count, has_fire_history, burnable

NaN handling:
  - XGBoost handles NaN natively (learns optimal split direction)
  - 6.64% coastal/border gridMET NaN → kept as NaN, XGBoost decides
  - 0.42% static NaN → filled with 0 (no landscape data = no hazard)

Split: chronological (never random)
  - TRAIN: 2014–2017
  - VAL  : 2018  (early stopping + threshold tuning)
  - TEST : 2019–2020  (final evaluation ONLY — never touched during training)

Usage:
    conda activate torch_gpu
    python run_phase3_train.py --state TX
    python run_phase3_train.py --state TX --no-gpu     # CPU fallback
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
)
import xgboost as xgb

PHASE2_ROOT = Path(__file__).resolve().parent
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from config.phase2_config import STATE_CONFIG, LOGS_DIR

logger = logging.getLogger(__name__)

# ── Feature columns used by the model ─────────────────────────────────────────
# Exclude: h3_cell, date_utc, window_hour, window_6h_utc, label,
#          fire_year, gridmet_missing (flag col, not a feature)
FEATURE_COLS = [
    # ── Landscape / static (LANDFIRE + FSim) ──────────────────────────────────
    "avg_burn_prob",    # FSim burn probability  [0–1]
    "whp",              # Wildfire Hazard Potential  [0–7000]
    "flep4",            # Flame Length Exceedance Prob ≥4ft  [0–1]
    "cfl",              # Canopy Fuel Load  [Mg ha⁻¹]
    # REMOVED: fire_count, has_fire_history — LEAKAGE per scope doc
    # fire_count is computed from full 2014-2020 data (including test years)
    # and trivially separates fire/non-fire by construction (scope doc line 391)
    "burnable",         # Binary: non-burnable mask

    # ── gridMET daily weather ──────────────────────────────────────────────────
    "erc",              # Energy Release Component  [BTU ft⁻²]
    "fm100",            # 100-hr dead fuel moisture  [%]
    "vpd",              # Vapor pressure deficit  [kPa]
    "vs",               # Wind speed  [m s⁻¹]
    "rmax",             # Max relative humidity  [%]
    "rmin",             # Min relative humidity  [%]
    "tmmx",             # Max temperature  [°C]
    "pr",               # Precipitation  [mm]

    # ── 5-day trailing stats ───────────────────────────────────────────────────
    "erc_5D_mean",
    "erc_5D_max",
    "fm100_5D_mean",
    "fm100_5D_min",
    "vpd_5D_mean",
    "vpd_5D_max",
    "vs_5D_mean",
    "vs_5D_max",
    "rmax_5D_mean",
    "rmax_5D_min",
    "tmmx_5D_mean",
    "tmmx_5D_max",

    # ── Temporal (cyclic encodings) ────────────────────────────────────────────
    "sin_month",
    "cos_month",
    "sin_hour",
    "cos_hour",

    # ── Location ───────────────────────────────────────────────────────────────
    "centroid_lat",
    "centroid_lon",
]

# Non-critical static cols — fill NaN with 0 (no landscape = low/unknown risk)
STATIC_COLS = ["avg_burn_prob", "whp", "flep4", "cfl", "burnable"]


def load_split(path: Path, split_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a parquet split, return (features_df, labels_array)."""
    df = pd.read_parquet(path)
    logger.info(f"  {split_name}: {len(df):,} rows  "
                f"(fire={int((df['label']==1).sum()):,}  "
                f"non-fire={int((df['label']==0).sum()):,})")

    # Validate all expected feature cols are present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"  Feature columns not found in {split_name}: {missing}")

    present_feats = [c for c in FEATURE_COLS if c in df.columns]
    X = df[present_feats].copy()
    y = df["label"].values.astype(np.int8)

    # Fill static NaN with 0 (safe — no landscape data = unknown/low hazard)
    for col in STATIC_COLS:
        if col in X.columns and X[col].isna().any():
            n = X[col].isna().sum()
            X[col] = X[col].fillna(0.0)
            logger.info(f"    {col}: filled {n:,} NaN → 0")

    # gridMET NaN: leave as NaN — XGBoost handles natively
    for col in [c for c in present_feats if c not in STATIC_COLS]:
        n_nan = X[col].isna().sum()
        if n_nan > 0:
            logger.debug(f"    {col}: {n_nan:,} NaN (XGBoost handles natively)")

    # ── Convert any object columns to numeric ────────────────────────────────
    # has_fire_history and burnable are stored as object dtype (e.g. "0"/"1")
    # XGBoost DMatrix requires int/float/bool — convert all object cols here
    obj_cols = X.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        for col in obj_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(np.float32)
            logger.info(f"    {col}: converted object → float32")

    return X, y



def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    split: str, threshold: float = 0.5) -> dict:
    """Compute AUROC, AUPR, Precision@TopK, F1 at threshold."""
    auroc = roc_auc_score(y_true, y_prob)
    aupr  = average_precision_score(y_true, y_prob)

    # Precision @ Top-K (where K = number of positives)
    k = int(y_true.sum())
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    prec_at_k = y_true[top_k_idx].mean()

    # F1 at threshold
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0.0

    metrics = {
        "split":       split,
        "n_samples":   int(len(y_true)),
        "n_pos":       int(y_true.sum()),
        "auroc":       round(auroc, 4),
        "aupr":        round(aupr, 4),
        f"prec@top{k//1000}k": round(float(prec_at_k), 4),
        "precision":   round(float(precision), 4),
        "recall":      round(float(recall), 4),
        "f1":          round(float(f1), 4),
        "threshold":   round(threshold, 4),
    }
    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold maximizing F1 on validation set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    return float(thresholds[best_idx])


def train(state_key: str, cfg: dict, use_gpu: bool) -> bool:
    output_dir  = cfg["output_dir"]
    slug        = state_key.lower()
    model_dir   = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'=' * 65}")
    logger.info(f"PHASE 3 — {cfg['name'].upper()} — XGBoost Baseline Training")
    logger.info(f"{'=' * 65}")

    # ── Load splits ───────────────────────────────────────────────────────────
    logger.info(f"\n  Loading data splits...")
    train_path = output_dir / f"train_{slug}.parquet"
    val_path   = output_dir / f"val_{slug}.parquet"
    test_path  = output_dir / f"test_{slug}.parquet"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            logger.error(f"  Not found: {p} — run Phase 2G first")
            return False

    X_train, y_train = load_split(train_path, "TRAIN")
    X_val,   y_val   = load_split(val_path,   "VAL  ")
    X_test,  y_test  = load_split(test_path,  "TEST ")

    present_feats = list(X_train.columns)
    logger.info(f"\n  Features used: {len(present_feats)}")
    logger.info(f"  Feature list: {present_feats}")

    # ── Compute class weight ──────────────────────────────────────────────────
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    logger.info(f"\n  Class imbalance ratio: {scale_pos_weight:.2f}:1 "
                f"(non-fire:fire) → scale_pos_weight={scale_pos_weight:.2f}")

    # ── XGBoost parameters ────────────────────────────────────────────────────
    device = "cuda" if use_gpu else "cpu"
    params = {
        "objective":         "binary:logistic",
        "eval_metric":       ["logloss", "auc"],
        "tree_method":       "hist",
        "device":            device,

        # Model capacity
        "n_estimators":      2000,          # max rounds (early stopping will cut)
        "max_depth":         7,             # enough for interaction effects
        "min_child_weight":  30,            # regularize small leaves
        "subsample":         0.8,           # row subsampling per tree
        "colsample_bytree":  0.8,           # feature subsampling per tree
        "colsample_bylevel": 0.8,

        # Learning rate + regularization
        "learning_rate":     0.05,
        "gamma":             0.1,           # min loss reduction for split
        "reg_alpha":         0.1,           # L1
        "reg_lambda":        1.0,           # L2

        # Class imbalance
        "scale_pos_weight":  scale_pos_weight,

        "random_state":      42,
        "n_jobs":            -1,
        "verbosity":         1,
    }

    logger.info(f"\n  XGBoost config:")
    logger.info(f"    device        = {device}")
    logger.info(f"    max_depth     = {params['max_depth']}")
    logger.info(f"    n_estimators  = {params['n_estimators']} (+ early stopping)")
    logger.info(f"    learning_rate = {params['learning_rate']}")

    # ── Build DMatrix ─────────────────────────────────────────────────────────
    logger.info(f"\n  Building DMatrix...")
    dtrain = xgb.DMatrix(X_train, label=y_train,
                          feature_names=present_feats,
                          missing=np.nan, enable_categorical=False)
    dval   = xgb.DMatrix(X_val,   label=y_val,
                          feature_names=present_feats, missing=np.nan)
    dtest  = xgb.DMatrix(X_test,  label=y_test,
                          feature_names=present_feats, missing=np.nan)

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info(f"\n  Training...")
    evals       = [(dtrain, "train"), (dval, "val")]
    evals_result = {}

    model = xgb.train(
        params           = {k: v for k, v in params.items()
                            if k not in ("n_estimators", "random_state", "n_jobs")},
        dtrain           = dtrain,
        num_boost_round  = params["n_estimators"],
        evals            = evals,
        evals_result     = evals_result,
        early_stopping_rounds = 50,
        verbose_eval     = 100,
    )

    best_round = model.best_iteration
    logger.info(f"\n  Best round: {best_round + 1}  "
                f"(val-auc={evals_result['val']['auc'][best_round]:.4f}  "
                f"val-logloss={evals_result['val']['logloss'][best_round]:.4f})")

    # ── Find optimal threshold on VAL ─────────────────────────────────────────
    val_prob  = model.predict(dval)
    threshold = find_optimal_threshold(y_val, val_prob)
    logger.info(f"  Optimal threshold (max-F1 on VAL): {threshold:.4f}")

    # ── Evaluate all splits ───────────────────────────────────────────────────
    logger.info(f"\n  ── Evaluation ──────────────────────────────────────────")
    all_metrics = []
    for y, prob, dmat, split_name in [
        (y_train, model.predict(dtrain), dtrain, "TRAIN"),
        (y_val,   val_prob,              dval,   "VAL"),
        (y_test,  model.predict(dtest),  dtest,  "TEST"),
    ]:
        m = compute_metrics(y, prob, split_name, threshold)
        all_metrics.append(m)
        logger.info(
            f"  {split_name:<6}  AUROC={m['auroc']:.4f}  "
            f"AUPR={m['aupr']:.4f}  "
            f"Prec@Top={list(m.values())[5]:.4f}  "   # prec@topK
            f"F1={m['f1']:.4f}  "
            f"P={m['precision']:.3f}  R={m['recall']:.3f}"
        )

    test_m = all_metrics[2]

    # Leakage guard
    if test_m["auroc"] > 0.99:
        logger.warning(f"\n  ⚠️  AUROC={test_m['auroc']} > 0.99 — possible leakage. "
                       f"Check feature list immediately!")
    elif test_m["auroc"] > 0.96:
        logger.info(f"\n  ✔ TEST AUROC={test_m['auroc']:.4f} — excellent (with LANDFIRE)")
    elif test_m["auroc"] > 0.90:
        logger.info(f"\n  ✔ TEST AUROC={test_m['auroc']:.4f} — good (expected for gridMET-only baseline)")
    else:
        logger.warning(f"\n  ⚠️  TEST AUROC={test_m['auroc']:.4f} below expected range")

    # ── Feature importance ────────────────────────────────────────────────────
    logger.info(f"\n  ── Feature Importance (gain) ────────────────────────────")
    importance = model.get_score(importance_type="gain")
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, gain in importance_sorted[:20]:
        bar = "█" * int(gain / max(importance.values()) * 30)
        logger.info(f"    {feat:<25} {gain:>10.1f}  {bar}")

    # ── Save model + metadata ─────────────────────────────────────────────────
    model_path = model_dir / f"xgb_baseline_{slug}.ubj"
    model.save_model(str(model_path))
    logger.info(f"\n  Model saved: {model_path}")

    meta = {
        "state":          state_key,
        "model_type":     "XGBoost baseline (gridMET + LANDFIRE placeholder)",
        "features":       present_feats,
        "n_features":     len(present_feats),
        "best_round":     best_round + 1,
        "threshold":      threshold,
        "params":         params,
        "metrics":        all_metrics,
        "landfire_rasters_downloaded": False,  # update after raster download
        "notes": (
            "LANDFIRE features (avg_burn_prob, whp, flep4, cfl) are ~0 — "
            "rasters not yet downloaded. Download rasters, re-run Phase 2E + 2G + 3 "
            "to get full model. Expected AUROC boost: ~+0.05-0.08."
        ),
    }
    meta_path = model_dir / f"xgb_baseline_{slug}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  Metadata: {meta_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(f"\n  ══════════════════════════════════════════════════════════")
    logger.info(f"  PHASE 3 COMPLETE — {state_key} XGBoost Baseline")
    logger.info(f"  ══════════════════════════════════════════════════════════")
    logger.info(f"  TEST  AUROC: {test_m['auroc']:.4f}")
    logger.info(f"  TEST  AUPR:  {test_m['aupr']:.4f}")
    logger.info(f"  Best round:  {best_round + 1} trees")
    logger.info(f"  Threshold:   {threshold:.4f} (max-F1 on val)")
    logger.info(f"  Model file:  {model_path.name}")
    if test_m["auroc"] < 0.90:
        logger.info(f"\n  NEXT: Download LANDFIRE rasters → Phase 2E → 2G → Phase 3 retrain")
        logger.info(f"  Expected AUROC after rasters: ~0.93–0.96")
    logger.info(f"  ══════════════════════════════════════════════════════════")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 — XGBoost Baseline Training [Production]"
    )
    parser.add_argument("--state",  choices=["TX", "CA", "ALL"], required=True)
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU — use CPU training")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "phase3.log", encoding="utf-8"),
        ],
    )

    # Detect GPU
    use_gpu = not args.no_gpu
    if use_gpu:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            gpu_name = result.stdout.strip()
            logger.info(f"  GPU detected: {gpu_name}")
        except Exception:
            logger.warning("  nvidia-smi not found — falling back to CPU")
            use_gpu = False

    states = ["TX", "CA"] if args.state == "ALL" else [args.state]
    for s in states:
        ok = train(s, STATE_CONFIG[s], use_gpu)
        print(f"\n  {STATE_CONFIG[s]['name']:<15} {'✔ SUCCESS' if ok else '✘ FAILED'}")


if __name__ == "__main__":
    main()
