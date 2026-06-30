"""
trainer.py
==========
Model training functions for the Texas Wildfire ML pipeline (V1).

Models trained
--------------
  A — XGBoost + scale_pos_weight=3   (full feature set: 18 cols)
  B — XGBoost + scale_pos_weight=3   (ablation: 17 cols, no is_peak_fire_season)
  C — Random Forest + class_weight='balanced'
  D — LightGBM + is_unbalance=True
  E — Best of A/C/D refit + threshold tuning + SHAP  (see run_training.py)

All models use chronological (temporal) cross-validation on the training set.
Early stopping (XGBoost, LightGBM) uses the validation set directly.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
    models_dir: Path,
    model_tag: str = "xgb_a",
    n_estimators: int = 600,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 5,
    gamma: float = 0.1,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
) -> Any:
    """
    Train an XGBClassifier with early stopping on val set.

    Automatically uses GPU (cuda) if available, falls back to CPU.
    NaN values in X are handled natively by XGBoost — no imputation needed.

    Parameters
    ----------
    X_train, y_train : training features and labels
    X_val, y_val     : validation set (used for early stopping only)
    scale_pos_weight : n_neg / n_pos (handles class imbalance)
    models_dir       : directory to save model artifacts
    model_tag        : filename prefix for saved artifacts
    """
    import xgboost as xgb

    # Detect GPU availability
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        device = "cuda" if result.returncode == 0 else "cpu"
    except Exception:
        device = "cpu"

    logger.info(
        "Training XGBoost [%s] | device=%s | scale_pos_weight=%.2f | "
        "n_est=%d | lr=%.3f | depth=%d",
        model_tag, device, scale_pos_weight, n_estimators, learning_rate, max_depth
    )

    model = xgb.XGBClassifier(
        n_estimators         = n_estimators,
        max_depth            = max_depth,
        learning_rate        = learning_rate,
        subsample            = subsample,
        colsample_bytree     = colsample_bytree,
        min_child_weight     = min_child_weight,
        gamma                = gamma,
        reg_alpha            = reg_alpha,
        reg_lambda           = reg_lambda,
        scale_pos_weight     = scale_pos_weight,
        random_state         = random_state,
        eval_metric          = "aucpr",      # PR-AUC (better for imbalanced)
        early_stopping_rounds = early_stopping_rounds,
        tree_method          = "hist",
        device               = device,
        verbosity            = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    best_iter = model.best_iteration
    logger.info("XGBoost [%s] best iteration: %d", model_tag, best_iter)

    _save_model(model, models_dir, model_tag)
    return model


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models_dir: Path,
    model_tag: str = "rf",
    n_estimators: int = 500,
    max_depth: int = 20,
    min_samples_leaf: int = 5,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Any:
    """
    Train a Random Forest with class_weight='balanced'.

    IMPORTANT: X_train must be median-imputed before passing to this function
    (use DataPreparer.X_train_rf). RF cannot handle NaN natively.

    class_weight='balanced' automatically adjusts weights inversely
    proportional to class frequencies — equivalent to scale_pos_weight for trees.
    """
    from sklearn.ensemble import RandomForestClassifier

    logger.info(
        "Training Random Forest [%s] | n_est=%d | depth=%d | class_weight=balanced",
        model_tag, n_estimators, max_depth
    )

    model = RandomForestClassifier(
        n_estimators    = n_estimators,
        max_depth       = max_depth,
        min_samples_leaf = min_samples_leaf,
        max_features    = max_features,
        class_weight    = "balanced",
        random_state    = random_state,
        n_jobs          = n_jobs,
    )
    model.fit(X_train, y_train)
    logger.info("Random Forest [%s] trained. OOB score not enabled (use val metrics).", model_tag)

    _save_model(model, models_dir, model_tag)
    return model


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    models_dir: Path,
    model_tag: str = "lgbm",
    n_estimators: int = 600,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_samples: int = 20,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    landcover_col: str = "LandCover",
) -> Any:
    """
    Train a LightGBM classifier with is_unbalance=True.

    LightGBM handles NaN natively (similar to XGBoost).
    LandCover is registered as a categorical feature for optimal splits.

    is_unbalance=True automatically balances class weights using the
    ratio of negative to positive samples in the training set.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("LightGBM not installed. Run: pip install lightgbm")
        raise

    # Register LandCover as categorical
    categorical_features = []
    if landcover_col in X_train.columns:
        X_train = X_train.copy()
        X_val   = X_val.copy()
        X_train[landcover_col] = X_train[landcover_col].astype("category")
        X_val[landcover_col]   = X_val[landcover_col].astype("category")
        categorical_features   = [landcover_col]
        logger.info("LightGBM: '%s' registered as categorical feature.", landcover_col)

    logger.info(
        "Training LightGBM [%s] | n_est=%d | lr=%.3f | depth=%d | is_unbalance=True",
        model_tag, n_estimators, learning_rate, max_depth
    )

    model = lgb.LGBMClassifier(
        n_estimators        = n_estimators,
        max_depth           = max_depth,
        learning_rate       = learning_rate,
        subsample           = subsample,
        colsample_bytree    = colsample_bytree,
        min_child_samples   = min_child_samples,
        reg_alpha           = reg_alpha,
        reg_lambda          = reg_lambda,
        is_unbalance        = True,
        random_state        = random_state,
        verbose             = -1,
        n_jobs              = -1,
    )

    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=True),
                 lgb.log_evaluation(100)]

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        categorical_feature=categorical_features,
        callbacks=callbacks,
    )

    logger.info(
        "LightGBM [%s] best iteration: %d",
        model_tag, model.best_iteration_
    )

    _save_model(model, models_dir, model_tag)
    return model


# ---------------------------------------------------------------------------
# LightGBM inference helper
# ---------------------------------------------------------------------------

def prepare_lgbm_features(
    X: pd.DataFrame,
    landcover_col: str = "LandCover",
) -> pd.DataFrame:
    """
    Cast LandCover to the ``category`` dtype that LightGBM expects.

    Must be called on every DataFrame passed to ``model_d.predict_proba()``
    (or any LightGBM inference call) whenever LandCover was registered as a
    categorical feature during training.

    Parameters
    ----------
    X            : feature DataFrame (will NOT be modified in-place)
    landcover_col: column name that was registered as categorical

    Returns
    -------
    X_out : copy of X with landcover_col cast to ``category``
    """
    X_out = X.copy()
    if landcover_col in X_out.columns:
        X_out[landcover_col] = X_out[landcover_col].astype("category")
    return X_out


# ---------------------------------------------------------------------------
# Temporal cross-validation (diagnostic check on train set)
# ---------------------------------------------------------------------------

def run_temporal_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    n_splits: int = 3,
) -> List[float]:
    """
    Chronological rolling cross-validation on the training set.

    Uses TimeSeriesSplit (no data leakage — future folds never train on future data).
    Reports PR-AUC on each fold to diagnose model stability.

    This is a DIAGNOSTIC step only — it uses fast settings (100 trees).
    Final model is always retrained on the full training set.
    """
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import average_precision_score

    logger.info("Temporal CV (%d folds) on train set...", n_splits)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

        fold_spw = float((y_tr == 0).sum()) / max(1, float(y_tr.sum()))
        m = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            scale_pos_weight=fold_spw, eval_metric="aucpr",
            early_stopping_rounds=20, verbosity=0,
            tree_method="hist", random_state=42,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        probs = m.predict_proba(X_va)[:, 1]
        score = average_precision_score(y_va, probs)
        scores.append(score)
        logger.info("  Fold %d: PR-AUC = %.4f", fold + 1, score)

    mean_s, std_s = np.mean(scores), np.std(scores)
    logger.info(
        "CV complete: mean PR-AUC = %.4f (+/- %.4f)  "
        "[good if mean > 0.35 for 1:3 imbalanced data]",
        mean_s, std_s
    )
    return scores


# ---------------------------------------------------------------------------
# Artifact save/load helpers
# ---------------------------------------------------------------------------

def _save_model(model: Any, models_dir: Path, tag: str) -> None:
    """Save model as both .pkl and (for XGBoost) native .json."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = models_dir / f"{tag}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved: %s", pkl_path)

    # XGBoost native JSON (portable across versions)
    try:
        json_path = models_dir / f"{tag}.json"
        model.save_model(json_path)
        logger.info("XGBoost JSON saved: %s", json_path)
    except AttributeError:
        pass   # Not XGBoost


def load_model(models_dir: Path, tag: str) -> Any:
    """Load a pickled model by tag."""
    pkl_path = Path(models_dir) / f"{tag}.pkl"
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Loaded model: %s", pkl_path)
    return model
