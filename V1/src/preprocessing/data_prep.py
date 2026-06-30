"""
data_prep.py
============
Feature preprocessing for Texas Wildfire ML pipeline (V1).

Reads from the chronologically pre-split CSVs produced by build_dataset.py:
  data/processed/train.csv      (Jan–Aug 2024)
  data/processed/val.csv        (Sep 2024)
  data/processed/test.csv       (Oct–Dec 2024)

Steps
-----
1. Load train/val/test CSVs.
2. Drop non-feature columns: latitude, longitude, acq_date.
3. Cast LandCover to int (MODIS integer category, 0–17).
4. Separate features (X) and target (y = Fire).
5. For Random Forest only: fit median imputer on X_train, transform val/test.
   XGBoost and LightGBM handle NaN natively — no imputation needed.
6. Save X/y arrays as parquet for fast loading in trainer.
7. Save imputer artifact (for RF, also used in inference).

Design Notes
------------
- latitude and longitude are DROPPED intentionally. They are spatial identifiers
  that would let the model memorize specific fire locations instead of learning
  generalizable environmental fire risk patterns.

- acq_date is DROPPED. All temporal signal is already captured by the 7 encoded
  features: month, day_of_year, season_code, sin_month, cos_month, sin_doy, cos_doy.

- LandCover is an INTEGER category (MODIS MCD12Q1 LC_Type1 scheme):
    0=Water, 1=ENF, 2=EBF, 3=DNF, 4=DBF, 5=Mixed Forest,
    6=Closed Shrubland, 7=Open Shrubland, 8=Woody Savanna,
    9=Savanna, 10=Grassland, 11=Wetland, 12=Cropland, 13=Urban,
    14=Mosaic, 15=Snow/Ice, 16=Barren, 17=Unclassified
  Tree models (XGBoost, LightGBM, RF) handle integer categoricals natively.
  No one-hot encoding needed.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column constants (must match build_dataset.py schema)
# ---------------------------------------------------------------------------

DROP_COLS = ["latitude", "longitude", "acq_date"]
TARGET_COL = "Fire"

RASTER_FEATURES = [
    "NDVI", "EVI", "LST", "Temperature", "Wind",
    "Rainfall", "DEM", "Slope", "Aspect", "LandCover",
]
TEMPORAL_FEATURES = [
    "month", "day_of_year", "season_code",
    "sin_month", "cos_month", "sin_doy", "cos_doy",
]
PEAK_FEATURE = ["is_peak_fire_season"]

# Full feature list for Model A (with peak feature)
FEATURES_FULL = RASTER_FEATURES + TEMPORAL_FEATURES + PEAK_FEATURE
# Ablation feature list for Model B (without peak feature)
FEATURES_BASE = RASTER_FEATURES + TEMPORAL_FEATURES


# ---------------------------------------------------------------------------
# DataPreparer
# ---------------------------------------------------------------------------

class DataPreparer:
    """
    Loads and prepares train/val/test splits from pre-split CSVs.

    Parameters
    ----------
    proc_dir : Path
        Directory containing train.csv, val.csv, test.csv.
        For ablation: pass data/processed_ablation/.
    models_dir : Path
        Directory where artifacts (imputer, feature list) are saved.
    use_peak_feature : bool
        If True, include is_peak_fire_season. Default True (Model A).
        Set False for Model B (ablation).
    """

    def __init__(
        self,
        proc_dir: Path,
        models_dir: Path,
        use_peak_feature: bool = True,
    ) -> None:
        self.proc_dir        = Path(proc_dir)
        self.models_dir      = Path(models_dir)
        self.use_peak_feature = use_peak_feature

        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.feature_cols = FEATURES_FULL if use_peak_feature else FEATURES_BASE

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load pre-split train/val/test CSVs."""
        logger.info("Loading splits from: %s", self.proc_dir)
        splits = {}
        for name in ("train", "val", "test"):
            path = self.proc_dir / f"{name}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"{name}.csv not found at {path}.\n"
                    f"Run build_dataset.py first."
                )
            splits[name] = pd.read_csv(path)
            logger.info("  %s: %d rows", name, len(splits[name]))

        return splits["train"], splits["val"], splits["test"]

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features (X) and target (y=Fire).

        Drops: latitude, longitude, acq_date.
        Casts: LandCover to int.
        Returns: (X DataFrame, y Series).
        """
        # Verify target exists
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found.")

        # Keep only known feature columns that exist in the dataframe
        available = [c for c in self.feature_cols if c in df.columns]
        missing_feats = [c for c in self.feature_cols if c not in df.columns]
        if missing_feats:
            logger.warning("Features not in CSV (will be skipped): %s", missing_feats)

        X = df[available].copy()
        y = df[TARGET_COL].astype(int)

        # Cast LandCover to int (remove decimal artifacts from CSV)
        if "LandCover" in X.columns:
            X["LandCover"] = X["LandCover"].fillna(-1).astype(int)

        logger.info(
            "Features extracted: %d cols  |  Target: Fire  |  "
            "Pos=%d  Neg=%d",
            len(X.columns), int(y.sum()), int((y == 0).sum())
        )
        return X, y

    # ------------------------------------------------------------------
    # Imputer (for Random Forest only)
    # ------------------------------------------------------------------

    def fit_rf_imputer(self, X_train: pd.DataFrame) -> SimpleImputer:
        """
        Fit a median imputer on training features.
        ONLY used for Random Forest — XGBoost and LightGBM handle NaN natively.

        Rule: fit on train ONLY. Transform val/test with the same fitted imputer.
        """
        logger.info("Fitting median imputer on X_train (for Random Forest)...")
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_train)

        imputer_path = self.models_dir / "rf_imputer.pkl"
        with open(imputer_path, "wb") as f:
            pickle.dump(imputer, f)
        logger.info("RF imputer saved to %s", imputer_path)
        return imputer

    def apply_rf_imputer(
        self,
        imputer: SimpleImputer,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply a pre-fitted imputer to a feature DataFrame."""
        arr = imputer.transform(X)
        return pd.DataFrame(arr, columns=X.columns, index=X.index)

    # ------------------------------------------------------------------
    # Main prepare pipeline
    # ------------------------------------------------------------------

    def prepare(self) -> Dict[str, pd.DataFrame | pd.Series]:
        """
        Full preparation pipeline. Returns dict with keys:
          X_train, y_train, X_val, y_val, X_test, y_test,
          X_train_rf, X_val_rf, X_test_rf (median-imputed, for RF),
          feature_cols, scale_pos_weight
        """
        train_df, val_df, test_df = self.load_splits()

        X_train, y_train = self.extract_features(train_df)
        X_val,   y_val   = self.extract_features(val_df)
        X_test,  y_test  = self.extract_features(test_df)

        # scale_pos_weight for XGBoost/LightGBM
        n_pos = int(y_train.sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = round(n_neg / max(1, n_pos), 4)
        logger.info(
            "Train set: Fire=1=%d, Fire=0=%d -> scale_pos_weight=%.4f",
            n_pos, n_neg, scale_pos_weight
        )

        # Fit RF imputer on train, transform val/test
        imputer     = self.fit_rf_imputer(X_train)
        X_train_rf  = self.apply_rf_imputer(imputer, X_train)
        X_val_rf    = self.apply_rf_imputer(imputer, X_val)
        X_test_rf   = self.apply_rf_imputer(imputer, X_test)

        # Save feature list
        feat_path = self.models_dir / "feature_cols.json"
        with open(feat_path, "w") as f:
            json.dump(list(X_train.columns), f, indent=2)
        logger.info("Feature list saved: %d features -> %s", len(X_train.columns), feat_path)

        logger.info(
            "Data preparation complete.\n"
            "  Train : %d rows  Val : %d rows  Test : %d rows\n"
            "  Features : %d",
            len(X_train), len(X_val), len(X_test), len(X_train.columns)
        )

        return {
            "X_train":          X_train,
            "y_train":          y_train,
            "X_val":            X_val,
            "y_val":            y_val,
            "X_test":           X_test,
            "y_test":           y_test,
            "X_train_rf":       X_train_rf,
            "X_val_rf":         X_val_rf,
            "X_test_rf":        X_test_rf,
            "feature_cols":     list(X_train.columns),
            "scale_pos_weight": scale_pos_weight,
        }
