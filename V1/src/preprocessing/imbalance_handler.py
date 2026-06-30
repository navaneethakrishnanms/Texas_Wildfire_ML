import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score
from loguru import logger
from typing import Dict, Any, Tuple

"""
JUSTIFICATION OF THE IMBALANCE STRATEGY:
For Short-Term Wildfire Ignition prediction, the class imbalance is extreme (typically < 0.1% positive cells).
Our chosen strategy is a hybrid of **Class Weighting** inside XGBoost (`scale_pos_weight`) and **Precision-Recall Threshold Optimization** on validation predictions.

Here is the justification for this choice over alternatives:
1. **Why not SMOTE?** SMOTE generates synthetic minority samples by interpolating between nearest neighbors. In geospatial datasets, interpolating between fire ignitions ignores physical spatial boundaries (e.g. creating synthetic ignitions in waterbodies, urban centers, or areas with incompatible elevation profiles). 
2. **Why Class Weighting?** Class weighting modifies the loss function directly, multiplying the gradient of the minority class. This forces XGBoost to penalize false negatives severely, making it focus on predicting rare fire ignitions without altering the actual physical spatial-temporal structures of the dataset.
3. **Why Threshold Optimization?** Default classification thresholds assume a balanced 0.5 decision boundary. In extreme imbalance, this leads to 0 predicted fires. By evaluating the Precision-Recall curve on validation data, we identify the exact threshold (often between 0.01 and 0.08) that maximizes the F1-Score, balancing false alerts and missed ignitions.
"""

def get_class_weights(y: pd.Series) -> float:
    """
    Computes the scale_pos_weight for XGBoost.
    Formula: count(negative) / count(positive)
    """
    num_pos = int(y.sum())
    num_neg = len(y) - num_pos
    if num_pos == 0:
        logger.warning("No positive samples in target labels. Setting weight multiplier to 1.0.")
        return 1.0
    weight = float(num_neg) / num_pos
    logger.info(f"Imbalance stats: Negatives={num_neg}, Positives={num_pos}. Class weight factor (scale_pos_weight) = {weight:.4f}")
    return weight

def optimize_decision_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    """
    Finds the optimal decision threshold that maximizes the F1-Score on validation/test predictions.
    
    Returns:
        A tuple of (optimal_threshold, max_f1_score)
    """
    # Use scikit-learn precision_recall_curve to evaluate thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Calculate F1 for each threshold
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    # Avoid division by zero
    numerator = 2 * precisions * recalls
    denominator = precisions + recalls
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    
    # Identify index of maximum F1 (excluding the last element of precision/recall which corresponds to threshold=1)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    
    logger.info(f"Optimal threshold found at {best_threshold:.4f} with F1-Score = {best_f1:.4f}")
    logger.info(f"  Corresponding Precision: {precisions[best_idx]:.4f}, Recall: {recalls[best_idx]:.4f}")
    
    return best_threshold, best_f1

def apply_smote(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Attempts to apply SMOTE over-sampling.
    Falls back gracefully if imbalanced-learn is not installed in the environment.
    """
    try:
        from imblearn.over_sampling import SMOTE
        logger.info("Applying SMOTE oversampling to training features...")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(f"Oversampling complete. Resampled shapes: X={X_res.shape}, y={y_res.shape}")
        return X_res, y_res
    except ImportError:
        logger.warning("imblearn library is not installed. Skipping SMOTE oversampling and using raw features.")
        return X, y

def apply_hard_negative_sampling(X: pd.DataFrame, y: pd.Series, ratio: float = 5.0) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Subsamples negative cases (non-fire pixels) to reduce the imbalance ratio.
    Retains all positives, and randomly draws negative cases up to `ratio` times the number of positives.
    """
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    
    X_pos = X[pos_mask]
    y_pos = y[pos_mask]
    
    X_neg = X[neg_mask]
    y_neg = y[neg_mask]
    
    num_pos = len(X_pos)
    num_neg_to_sample = int(num_pos * ratio)
    
    if num_neg_to_sample >= len(X_neg):
        logger.info("Negative sample limit exceeds available negatives. Using all negative samples.")
        return X, y
        
    logger.info(f"Applying Hard Negative Sampling: Sampling {num_neg_to_sample} negative samples (Ratio 1:{ratio}).")
    X_neg_sampled = X_neg.sample(n=num_neg_to_sample, random_state=42)
    y_neg_sampled = y_neg.loc[X_neg_sampled.index]
    
    X_res = pd.concat([X_pos, X_neg_sampled]).sample(frac=1.0, random_state=42)
    y_res = y.loc[X_res.index]
    
    logger.info(f"Sampling complete. Balanced shapes: X={X_res.shape}, y={y_res.shape}")
    return X_res, y_res
