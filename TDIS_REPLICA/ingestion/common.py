"""
common.py
---------
Shared config + model-loading helpers for the TDIS Portal Replica ingestion
scripts. All paths are relative to the main "Texas ML Wildfire" project so
this replica reuses the exact model already trained in TEXAS/, no retraining.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import h3

# ── Paths back into the main project ─────────────────────────────────────────
REPLICA_ROOT = Path(__file__).resolve().parent.parent      # TDIS_REPLICA/
PROJECT_ROOT = REPLICA_ROOT.parent                          # Texas ML Wildfire/
TEXAS_DIR = PROJECT_ROOT / "TEXAS"

MODEL_PATH = TEXAS_DIR / "model_32feat_tuned.json"
CALIBRATOR_PATH = TEXAS_DIR / "calibrator_32feat_tuned.joblib"
DATASET_PATH = TEXAS_DIR / "tdis_train_daily_imputed_v2.parquet"

DATA_OUT = REPLICA_ROOT / "data"
DATA_OUT.mkdir(exist_ok=True)

# ── The 32 features, in the exact order the model expects ───────────────────
FEATURES = [
    "elevation_m", "slope_deg", "aspect_deg", "road_dist_km", "ecoregion_id",
    "avg_burn_prob", "whp", "flep4", "cfl", "cbd", "cbh",
    "powerline_dist_km",
    "hrrr_tmp", "hrrr_vpd", "hrrr_wind", "hrrr_mstav",
    "erc", "fm100", "vpd", "vs", "rmax", "rmin", "tmmx", "pr",
    "sin_month", "cos_month", "sin_dow", "cos_dow", "is_weekend", "is_holiday",
    "lat", "lon",
]

# Cutover date: last date with real ground-truth-supported data.
# Everything on/before this is "historical". Everything after is the
# SIMULATED live daily feed (see build_live_august_feed.py for how it's built
# and why it has to be simulated rather than real).
HISTORICAL_CUTOVER = pd.Timestamp("2026-07-31")
LIVE_START = pd.Timestamp("2026-08-01")
LIVE_END = pd.Timestamp("2026-08-31")

RISK_BINS = [0.0, 0.25, 0.5, 0.75, 1.01]
RISK_LABELS = ["low", "moderate", "high", "extreme"]


def load_model_and_calibrator():
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    calibrator = joblib.load(str(CALIBRATOR_PATH))
    return model, calibrator


def score(model, calibrator, df: pd.DataFrame) -> np.ndarray:
    """Run the 32-feature model + isotonic calibrator on a feature frame."""
    dmat = xgb.DMatrix(df[FEATURES].values, feature_names=FEATURES)
    raw = model.predict(dmat)
    calibrated = calibrator.predict(raw)
    return calibrated


def risk_class(scores: np.ndarray) -> np.ndarray:
    return pd.cut(scores, bins=RISK_BINS, labels=RISK_LABELS, include_lowest=True).astype(str)


def hex_boundary_geojson(h3_cell: str):
    """Return a GeoJSON Polygon ring (lon, lat order) for one H3 cell."""
    boundary = h3.cell_to_boundary(h3_cell)  # tuple of (lat, lng)
    ring = [[lng, lat] for lat, lng in boundary]
    ring.append(ring[0])  # close the ring
    return [ring]
