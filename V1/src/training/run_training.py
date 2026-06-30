"""
run_training.py
===============
Main entry point for the Texas Wildfire ML training pipeline (V1).

Usage
-----
  python src/training/run_training.py

What this script does
---------------------
  [STEP 1] Data preparation
    - Load pre-split train/val/test CSVs
    - Drop latitude, longitude, acq_date (non-features)
    - Cast LandCover to int
    - Fit median imputer on train (for Random Forest only)

  [STEP 2] Temporal cross-validation (diagnostic)
    - 3-fold TimeSeriesSplit on training set
    - Confirms model stability across time

  [STEP 3] Train 4 base models
    - A: XGBoost  (full features, scale_pos_weight=3)
    - B: XGBoost  (ablation: no is_peak_fire_season)
    - C: Random Forest (class_weight='balanced', median-imputed)
    - D: LightGBM (is_unbalance=True, LandCover as category)

  [STEP 4] Evaluate all models on validation set
    - AUC-ROC, AUC-PR, F1, Precision, Recall at default threshold=0.50
    - Print comparison table, identify best model

  [STEP 5] Model E — Best model with threshold tuning + SHAP
    - Tune threshold for Recall >= 0.85 on validation set
    - Run full SHAP analysis
    - Generate all plots (PR, ROC, confusion matrix, SHAP beeswarm, bar, dependence)

  [STEP 6] Final test set evaluation
    - Evaluate best model (with tuned threshold) on held-out test set
    - Print final report with TP/FP/TN/FN breakdown

Outputs
-------
  models/
    xgb_a.json / .pkl      -- XGBoost Model A
    xgb_b.json / .pkl      -- XGBoost Model B (ablation)
    rf.pkl                 -- Random Forest
    lgbm.pkl               -- LightGBM
    best_model.pkl         -- Copy of best model
    rf_imputer.pkl         -- Median imputer (RF only)
    feature_cols.json      -- Feature names list
    optimal_threshold.json -- Tuned decision threshold

  outputs/plots/
    pr_curve_*.png         -- Precision-Recall curves
    roc_curve_*.png        -- ROC curves
    cm_*.png               -- Confusion matrices
    shap_summary_*.png     -- SHAP beeswarm
    shap_bar_*.png         -- SHAP bar chart
    shap_dep_*_*.png       -- SHAP dependence plots (top 5 features)

  outputs/
    shap_importance_*.csv  -- SHAP feature importance tables
    val_metrics.json       -- All model validation metrics
    test_metrics.json      -- Best model test metrics
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_prep import DataPreparer
from src.training.trainer import (
    train_xgboost,
    train_random_forest,
    train_lightgbm,
    run_temporal_cv,
    prepare_lgbm_features,
)
from src.evaluation.evaluator import (
    compute_metrics,
    tune_threshold,
    plot_pr_roc,
    plot_confusion_matrix,
    print_comparison_table,
    final_test_report,
)
from src.evaluation.explainability import run_shap_analysis

# ── Directory constants ──────────────────────────────────────────────────────
PROC_DIR      = PROJECT_ROOT / "data" / "processed"
ABLATION_DIR  = PROJECT_ROOT / "data" / "processed_ablation"
MODELS_DIR    = PROJECT_ROOT / "models"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
PLOTS_DIR     = OUTPUTS_DIR / "plots"

# Create ALL directories FIRST (before logging opens the log file)
for d in [MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, PROJECT_ROOT / "logs"]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "training.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 70)
    logger.info("  TEXAS WILDFIRE ML — TRAINING PIPELINE  (V1)")
    logger.info("=" * 70)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1 — Data Preparation
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n[STEP 1/6] Data preparation ...")

    # Model A/C/D — full features (with is_peak_fire_season)
    prep_full = DataPreparer(PROC_DIR, MODELS_DIR, use_peak_feature=True)
    data_full = prep_full.prepare()

    # Model B — ablation (without is_peak_fire_season)
    prep_abla = DataPreparer(ABLATION_DIR, MODELS_DIR / "ablation", use_peak_feature=False)
    data_abla = prep_abla.prepare()

    X_train     = data_full["X_train"]
    y_train     = data_full["y_train"]
    X_val       = data_full["X_val"]
    y_val       = data_full["y_val"]
    X_test      = data_full["X_test"]
    y_test      = data_full["y_test"]
    X_train_rf  = data_full["X_train_rf"]   # median-imputed for RF
    X_val_rf    = data_full["X_val_rf"]
    X_test_rf   = data_full["X_test_rf"]
    spw         = data_full["scale_pos_weight"]

    logger.info(
        "Features (Model A): %d | Features (Model B): %d",
        len(data_full["feature_cols"]), len(data_abla["feature_cols"])
    )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2 — Temporal Cross-Validation (diagnostic)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n[STEP 2/6] Temporal cross-validation (3-fold, diagnostic) ...")
    cv_scores = run_temporal_cv(X_train, y_train, scale_pos_weight=spw, n_splits=3)
    logger.info("CV PR-AUC scores: %s", [f"{s:.4f}" for s in cv_scores])

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3 — Train 4 Base Models
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n[STEP 3/6] Training 4 base models ...")

    # --- Model A: XGBoost (full features) ---
    logger.info("\n  --- Model A: XGBoost (full features, %d cols) ---", len(X_train.columns))
    model_a = train_xgboost(
        X_train, y_train, X_val, y_val,
        scale_pos_weight=spw,
        models_dir=MODELS_DIR,
        model_tag="xgb_a",
    )

    # --- Model B: XGBoost Ablation (no is_peak_fire_season) ---
    logger.info("\n  --- Model B: XGBoost Ablation (no peak feature, %d cols) ---",
                len(data_abla["X_train"].columns))
    model_b = train_xgboost(
        data_abla["X_train"], data_abla["y_train"],
        data_abla["X_val"],   data_abla["y_val"],
        scale_pos_weight=data_abla["scale_pos_weight"],
        models_dir=MODELS_DIR,
        model_tag="xgb_b",
    )

    # --- Model C: Random Forest ---
    logger.info("\n  --- Model C: Random Forest (class_weight='balanced') ---")
    model_c = train_random_forest(
        X_train_rf, y_train,
        models_dir=MODELS_DIR,
        model_tag="rf",
    )

    # --- Model D: LightGBM ---
    logger.info("\n  --- Model D: LightGBM (is_unbalance=True) ---")
    model_d = train_lightgbm(
        X_train, y_train, X_val, y_val,
        models_dir=MODELS_DIR,
        model_tag="lgbm",
    )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4 — Evaluate All Models on Validation Set
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n[STEP 4/6] Evaluating all models on validation set ...")

    val_results = {}

    # Model A
    probs_a = model_a.predict_proba(X_val)[:, 1]
    val_results["A: XGBoost (full)"] = compute_metrics(
        y_val.values, probs_a, threshold=0.50, label="Model A"
    )
    plot_pr_roc(y_val.values, probs_a, "Model_A_XGBoost", PLOTS_DIR)

    # Model B (ablation — evaluated on ablation val set)
    probs_b = model_b.predict_proba(data_abla["X_val"])[:, 1]
    val_results["B: XGBoost (ablation)"] = compute_metrics(
        data_abla["y_val"].values, probs_b, threshold=0.50, label="Model B"
    )
    plot_pr_roc(data_abla["y_val"].values, probs_b, "Model_B_XGBoost_Ablation", PLOTS_DIR)

    # Model C (RF on imputed val)
    probs_c = model_c.predict_proba(X_val_rf)[:, 1]
    val_results["C: Random Forest"] = compute_metrics(
        y_val.values, probs_c, threshold=0.50, label="Model C"
    )
    plot_pr_roc(y_val.values, probs_c, "Model_C_RandomForest", PLOTS_DIR)

    # Model D (LightGBM) — must cast LandCover to category before inference
    probs_d = model_d.predict_proba(prepare_lgbm_features(X_val))[:, 1]
    val_results["D: LightGBM"] = compute_metrics(
        y_val.values, probs_d, threshold=0.50, label="Model D"
    )
    plot_pr_roc(y_val.values, probs_d, "Model_D_LightGBM", PLOTS_DIR)

    # Save all val metrics
    val_metrics_path = OUTPUTS_DIR / "val_metrics.json"
    with open(val_metrics_path, "w") as f:
        json.dump(val_results, f, indent=2)
    logger.info("Validation metrics saved: %s", val_metrics_path)

    # Print comparison table
    print_comparison_table(val_results)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5 — Model E: Best Model + Threshold Tuning + SHAP
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n[STEP 5/6] Model E: threshold tuning + SHAP on best model ...")

    # Identify best model by AUC-ROC on validation set
    # (only compare A, C, D — B uses different features so different val set)
    comparable = {
        "A: XGBoost (full)":   (model_a, probs_a, X_train,                      X_val,                      y_val),
        "C: Random Forest":    (model_c, probs_c, X_train_rf,                    X_val_rf,                   y_val),
        "D: LightGBM":         (model_d, probs_d, prepare_lgbm_features(X_train), prepare_lgbm_features(X_val), y_val),
    }
    best_name = max(
        comparable,
        key=lambda n: val_results[n]["auc_roc"]
    )
    best_model, best_probs, best_X_train, best_X_val, best_y_val = comparable[best_name]
    logger.info("Best model: %s  (AUC-ROC=%.4f)", best_name, val_results[best_name]["auc_roc"])

    # Save best model reference
    best_pkl = MODELS_DIR / "best_model.pkl"
    with open(best_pkl, "wb") as f:
        pickle.dump(best_model, f)

    # Threshold tuning: find threshold where Recall(Fire=1) >= 0.85
    logger.info("\n  --- Threshold Tuning (target Recall >= 0.85) ---")
    optimal_threshold, tuned_val_metrics = tune_threshold(
        best_y_val.values, best_probs,
        target_recall=0.85,
        step=0.01,
        label=f"Model E ({best_name})",
    )

    threshold_path = MODELS_DIR / "optimal_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({
            "model":     best_name,
            "threshold": optimal_threshold,
            "metrics":   tuned_val_metrics
        }, f, indent=2)
    logger.info("Optimal threshold saved: %s", threshold_path)

    # Add Model E to comparison table
    val_results[f"E: Best ({best_name.split(':')[0]}) Tuned"] = tuned_val_metrics
    print_comparison_table(val_results)

    # Confusion matrix for Model E on val
    y_pred_e = (best_probs >= optimal_threshold).astype(int)
    plot_confusion_matrix(best_y_val.values, y_pred_e, "Model_E_Best_Tuned", PLOTS_DIR)

    # SHAP analysis for Model E
    logger.info("\n  --- SHAP Analysis ---")
    expected_top = [
        "NDVI", "EVI", "LST", "Temperature", "Wind",
        "Rainfall", "sin_month", "cos_month", "sin_doy", "cos_doy",
        "DEM", "Slope", "month", "is_peak_fire_season",
    ]
    shap_importance = run_shap_analysis(
        model       = best_model,
        X_train     = best_X_train,
        X_val       = best_X_val,
        model_name  = f"Model E ({best_name})",
        out_dir     = OUTPUTS_DIR,
        background_samples = 500,
        val_samples        = 500,
        expected_top       = expected_top,
    )

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6 — Final Test Set Evaluation
    # ══════════════════════════════════════════════════════════════════════
    logger.info("\n[STEP 6/6] Final test set evaluation ...")

    # Use the same best model and optimal threshold from val set
    if "C: Random Forest" in best_name:
        test_probs = best_model.predict_proba(X_test_rf)[:, 1]
    elif "LightGBM" in best_name:
        test_probs = best_model.predict_proba(prepare_lgbm_features(X_test))[:, 1]
    else:
        test_probs = best_model.predict_proba(X_test)[:, 1]

    test_metrics = compute_metrics(
        y_test.values, test_probs,
        threshold=optimal_threshold,
        label=f"TEST — Model E ({best_name})",
    )

    test_metrics_path = OUTPUTS_DIR / "test_metrics.json"
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Plot test set curves
    plot_pr_roc(y_test.values, test_probs, "Model_E_Test", PLOTS_DIR, threshold=optimal_threshold)
    y_pred_test = (test_probs >= optimal_threshold).astype(int)
    plot_confusion_matrix(y_test.values, y_pred_test, "Model_E_Test", PLOTS_DIR)

    # Print final report
    final_test_report(
        model_name   = f"Model E — {best_name} (threshold={optimal_threshold:.2f})",
        val_metrics  = tuned_val_metrics,
        test_metrics = test_metrics,
    )

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 70)
    logger.info("  TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Models saved to    : %s", MODELS_DIR)
    logger.info("Plots saved to     : %s", PLOTS_DIR)
    logger.info("Metrics saved to   : %s", OUTPUTS_DIR)
    logger.info("")
    logger.info("Best model         : %s", best_name)
    logger.info("Optimal threshold  : %.2f  (Recall >= 0.85)", optimal_threshold)
    logger.info("Test AUC-ROC       : %.4f", test_metrics["auc_roc"])
    logger.info("Test Recall (Fire) : %.4f", test_metrics["recall"])
    logger.info("Test F1            : %.4f", test_metrics["f1"])
    logger.info("Fires CAUGHT       : %d / %d  in test set", test_metrics["tp"],
                test_metrics["tp"] + test_metrics["fn"])
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
