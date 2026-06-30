"""
predict.py
==========
Inference script for the trained XGBoost wildfire risk model.

Usage (predict on a CSV of locations):
    python src/inference/predict.py --input my_locations.csv

Usage (predict on a single coordinate):
    python src/inference/predict.py --lat 30.27 --lon -99.84

Usage (predict on the test set):
    python src/inference/predict.py --test-set

Input CSV must have at minimum: latitude, longitude
  (other feature columns are sampled from rasters automatically)

Output: outputs/predictions/predictions.csv
  columns: latitude, longitude, fire_risk_score, fire_predicted
"""

from __future__ import annotations
import sys
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset_builder.sample_rasters import RasterSampler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")
RAW_DIR    = Path("data/raw")
PROC_DIR   = Path("data/processed")
OUT_DIR    = Path("outputs/predictions")

MODEL_PKL     = MODELS_DIR / "xgb_model.pkl"
SCALER_PKL    = MODELS_DIR / "scaler.pkl"
IMPUTER_PKL   = MODELS_DIR / "imputer.pkl"
FEATURES_JSON = MODELS_DIR / "features.json"


# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
def load_artifacts():
    """Load model, scaler, imputer, and feature list."""
    if not MODEL_PKL.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PKL}. "
            "Run training first: python main.py --phase train"
        )

    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)

    with open(IMPUTER_PKL, "rb") as f:
        imputer = pickle.load(f)

    with open(FEATURES_JSON) as f:
        feature_cols = json.load(f)

    logger.info(f"Model loaded from {MODEL_PKL}")
    logger.info(f"Feature list ({len(feature_cols)} features): {feature_cols[:5]} ...")
    return model, scaler, imputer, feature_cols


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """
    Select and order feature columns. Missing columns are filled with NaN.
    Returns a numpy array ready for imputer → scaler → model.
    """
    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in df.columns:
            X[col] = df[col]
        else:
            logger.warning(f"Feature '{col}' not found in input — filling with NaN")
            X[col] = np.nan
    return X.values


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------
def predict(
    input_df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Predict wildfire ignition risk for each row in input_df.

    input_df must contain: latitude, longitude
    (Features are sampled from rasters if not already present)

    Returns DataFrame with predictions appended.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model, scaler, imputer, feature_cols = load_artifacts()

    # Check if features already present, otherwise sample from rasters
    raster_features = [
        "ndvi", "evi", "lst", "temperature", "wind",
        "rainfall", "dem", "slope", "aspect", "landcover"
    ]
    needs_sampling = not any(f in input_df.columns for f in raster_features)

    if needs_sampling:
        logger.info("Sampling raster features for input coordinates ...")
        with RasterSampler(RAW_DIR) as sampler:
            sampled_records = sampler.sample_batch(
                list(zip(input_df["latitude"], input_df["longitude"]))
            )
        feat_df = pd.DataFrame(sampled_records)
        # Drop duplicate lat/lon from sampler output
        feat_df = feat_df.drop(columns=["latitude", "longitude"], errors="ignore")
        input_df = pd.concat([input_df.reset_index(drop=True), feat_df], axis=1)

    # Prepare feature matrix
    X_raw = prepare_features(input_df, feature_cols)

    # Impute then scale
    X_imp    = imputer.transform(X_raw)
    X_scaled = scaler.transform(X_imp)

    # Predict
    proba = model.predict_proba(X_scaled)[:, 1]
    pred  = (proba >= threshold).astype(int)

    result = input_df.copy()
    result["fire_risk_score"] = np.round(proba, 4)
    result["fire_predicted"]  = pred

    # Save
    out_path = OUT_DIR / "predictions.csv"
    result.to_csv(out_path, index=False)
    logger.info(f"Predictions saved → {out_path}  ({len(result)} rows)")
    logger.info(f"  Predicted fires: {pred.sum()} / {len(pred)}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Wildfire risk prediction inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",    type=str, help="CSV file with latitude, longitude columns")
    group.add_argument("--lat",      type=float, help="Single point latitude")
    group.add_argument("--test-set", action="store_true", help="Run on data/processed/test.csv")
    parser.add_argument("--lon",       type=float, help="Single point longitude (required with --lat)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default 0.5)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    if args.lat is not None:
        if args.lon is None:
            parser.error("--lon is required when using --lat")
        df = pd.DataFrame({"latitude": [args.lat], "longitude": [args.lon]})
    elif args.test_set:
        test_path = PROC_DIR / "test.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Test set not found: {test_path}")
        df = pd.read_csv(test_path)
    else:
        df = pd.read_csv(args.input)

    result = predict(df, threshold=args.threshold)

    # Print summary to console
    print("\n--- PREDICTION RESULTS ---")
    display_cols = ["latitude", "longitude", "fire_risk_score", "fire_predicted"]
    display_cols = [c for c in display_cols if c in result.columns]
    print(result[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
