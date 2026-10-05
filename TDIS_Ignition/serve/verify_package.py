#!/usr/bin/env python
"""
TDIS 24-48h Ignition Forecast — package self-check.

Run this to independently verify the package is production-ready, without
trusting any prior claim about it. Checks structure, model/calibrator
integrity, data integrity (nulls, flare exclusion, row counts), and does a
live end-to-end offline scoring pass. Exits non-zero if anything fails.

Usage:
  conda activate UAI2526   (or: pip install -r requirements.txt)
  python serve/verify_package.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PASS, FAIL, WARN = [], [], []


def ok(msg): PASS.append(msg); print(f"  [PASS] {msg}")
def bad(msg): FAIL.append(msg); print(f"  [FAIL] {msg}")
def warn(msg): WARN.append(msg); print(f"  [WARN] {msg}")


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
section("1. Structure — required files present")
required = [
    "model/tdis_forecast_hrrr_filtered.json",
    "model/operational_isotonic_calibrator.joblib",
    "model/tdis_forecast_hrrr_filtered_meta.json",
    "model/operational_realpop_validation.json",
    "data/static/tx_static_master.parquet",
    "data/static/powerline_dist_km.parquet",
    "data/flares/flare_cells_tx_v2.parquet",
    "data/labels/ignitions_daily_tx.parquet",
    "data/scored/peak_weather_features.parquet",
    "docs/TRD_wildfire_ignition_v1.md",
    "docs/PRD_daily_ignition_section3_datareqs.md",
    "requirements.txt",
    "serve/score_day.py",
]
for rel in required:
    (ok if (ROOT / rel).exists() else bad)(rel)

# ---------------------------------------------------------------------------
section("2. Model + calibrator integrity")
try:
    import xgboost as xgb, joblib, json

    m = xgb.XGBClassifier()
    m.load_model(str(ROOT / "model" / "tdis_forecast_hrrr_filtered.json"))
    feats = m.get_booster().feature_names
    if len(feats) == 23:
        ok(f"model loads, {len(feats)} features")
    else:
        bad(f"model has {len(feats)} features, expected 23")

    meta = json.load(open(ROOT / "model" / "tdis_forecast_hrrr_filtered_meta.json"))
    if meta.get("features") == feats:
        ok("meta.json feature list matches model file exactly")
    else:
        bad("meta.json feature list does NOT match model's actual feature_names "
            "(scoring with the wrong order silently corrupts predictions)")

    cal = joblib.load(ROOT / "model" / "operational_isotonic_calibrator.joblib")
    import numpy as np
    test_in = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    test_out = cal.predict(test_in)
    if np.all(np.diff(test_out) >= 0):
        ok(f"calibrator loads and is monotone (0.1->{test_out[0]:.4f}, 0.9->{test_out[-1]:.4f})")
    else:
        bad("calibrator output is NOT monotone in raw score -- broken calibrator")
    if 0 <= test_out.min() and test_out.max() <= 1:
        ok("calibrated outputs stay in [0,1]")
    else:
        bad(f"calibrated outputs out of range: [{test_out.min()},{test_out.max()}]")

except Exception as e:
    bad(f"model/calibrator load raised: {e}")

# ---------------------------------------------------------------------------
section("3. Data integrity")
try:
    import pandas as pd

    static = pd.read_parquet(ROOT / "data" / "static" / "tx_static_master.parquet")
    n_expect = 1_708_940
    if abs(len(static) - n_expect) < 1000:
        ok(f"tx_static_master row count sane: {len(static):,} (expect ~{n_expect:,})")
    else:
        warn(f"tx_static_master row count {len(static):,}, expected ~{n_expect:,} -- confirm this is intentional")

    key_cols = ["avg_burn_prob", "whp", "elevation_m", "ecoregion_id", "road_dist_km"]
    null_report = {c: int(static[c].isna().sum()) for c in key_cols if c in static.columns}
    if all(v == 0 for v in null_report.values()):
        ok(f"zero nulls in key static features {key_cols}")
    else:
        bad(f"nulls found in static features: {null_report} "
            f"(this exact bug -- broken static source -- happened before; see README)")

    dup = static["h3_cell"].duplicated().sum()
    (ok if dup == 0 else bad)(f"duplicate h3_cell in static master: {dup}")

    flare = pd.read_parquet(ROOT / "data" / "flares" / "flare_cells_tx_v2.parquet")
    if len(flare) > 0 and "h3_cell" in flare.columns:
        ok(f"flare list present: {len(flare):,} cells")
    else:
        bad("flare list empty or malformed")

    labels = pd.read_parquet(ROOT / "data" / "labels" / "ignitions_daily_tx.parquet")
    flare_set = set(flare["h3_cell"])
    if "h3_cell" in labels.columns:
        contam = labels["h3_cell"].isin(flare_set).mean()
        if contam < 0.001:
            ok(f"labels are flare-clean ({contam*100:.3f}% in flare cells)")
        else:
            warn(f"{contam*100:.1f}% of label rows sit in flare cells -- EXPECTED: "
                 f"ignitions_daily_tx.parquet is the RAW fused table, flare filtering is a "
                 f"downstream join (code/scripts/18), NOT baked in. The served model was "
                 f"trained flare-filtered; only this raw label file is not. See README §4.")

except Exception as e:
    bad(f"data integrity checks raised: {e}")

# ---------------------------------------------------------------------------
section("4. End-to-end offline scoring (no network)")
try:
    import subprocess, tempfile

    pw = pd.read_parquet(ROOT / "data" / "scored" / "peak_weather_features.parquet", columns=["date"])
    best_date = pw["date"].value_counts().idxmax()
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(
            [sys.executable, str(ROOT / "serve" / "score_day.py"),
             str(best_date.date()), "--offline", "--out", tmp],
            capture_output=True, text=True, timeout=180,
        )
        out_parquet = list(Path(tmp).glob("forecast_*_res8.parquet"))
        if r.returncode == 0 and out_parquet:
            scored = pd.read_parquet(out_parquet[0])
            cal_mean = scored["ign_cal"].mean()
            ok(f"serve/score_day.py ran end-to-end, scored {len(scored):,} cells, "
               f"cal mean={cal_mean:.4f}")
            if 0.005 < cal_mean < 0.05:
                ok(f"calibrated mean {cal_mean:.4f} is in the expected real-population "
                   f"band (~0.015-0.02, base rate 1.9%)")
            else:
                warn(f"calibrated mean {cal_mean:.4f} is outside the usual ~0.01-0.03 band "
                     f"-- check which date/inputs were used")
        else:
            bad(f"serve/score_day.py failed (exit {r.returncode}): {r.stderr[-500:]}")
except Exception as e:
    bad(f"end-to-end scoring test raised: {e}")

# ---------------------------------------------------------------------------
section("5. Reported vs actual real-population metrics")
try:
    import json
    rp = json.load(open(ROOT / "model" / "operational_realpop_validation.json"))
    ok(f"real-population validation on file: AUC-PR={rp['aucpr']:.4f}, "
       f"AUROC={rp['auroc']:.4f}, lift={rp['lift']:.2f}x, "
       f"ECE={rp['ece']:.4f} (base rate {rp['real_positive_rate']*100:.2f}%)")
    warn("this file's numbers are a CLAIM, not re-derived here -- cross-check "
         "against README.md section 1's table by eye")
except Exception as e:
    bad(f"could not read operational_realpop_validation.json: {e}")

# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"RESULT: {len(PASS)} passed, {len(WARN)} warnings, {len(FAIL)} failed")
if FAIL:
    print("FAILED CHECKS:")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
elif WARN:
    print("All hard checks passed; review warnings above before sign-off.")
    sys.exit(0)
else:
    print("All checks passed clean.")
    sys.exit(0)
