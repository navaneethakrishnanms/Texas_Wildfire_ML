"""
BIT-methodology retrain on TDIS's newest (TDIS_Ignition) dataset.

Methodology (same recipe that took our TX model 0.8637 -> 0.8687 AUROC, and
our TEXAS model 0.8249 -> 0.8309 AUROC): a genuine two-stage hyperparameter
search (tree structure, then learning-rate x subsample) on a held-out
validation fold, never touching test during search, followed by final
training with early stopping and isotonic probability calibration.

Data: TDIS_Ignition_BIT/data/bit_train_tx.parquet (built by build_dataset.py
from TDIS_Ignition/model_tristate/data/train_tristate_v2.parquet, TX rows
only, their own embedded split: train 2018-2020 / val 2021 / test 2022-2026).

GPU: device='cuda' (RTX, torch_gpu conda env).
"""
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score, brier_score_loss
)
from sklearn.isotonic import IsotonicRegression
import joblib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "bit_train_tx.parquet"
OUT = ROOT / "model"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    # Static — terrain / access / fuel hazard (12, TDIS's own served set)
    "road_dist_km", "ecoregion_id", "elevation_m", "slope_deg", "aspect_deg",
    "avg_burn_prob", "whp", "flep4", "cfl", "cbd", "cbh", "powerline_dist_km",
    # Calendar (6)
    "sin_month", "cos_month", "sin_dow", "cos_dow", "is_weekend", "is_holiday",
    # HRRR forecast weather (4) + drought (1) — TDIS's 23-feature served set ends here
    "hrrr_tmp", "hrrr_vpd", "hrrr_wind", "hrrr_mstav", "drought_score",
    # Tri-state interaction terms + human/fuel features (5, from TDIS's rev5/28-feature set)
    "bp_x_vpd", "whp_x_vpd", "bp_x_drought", "pop_density", "fm100_5d_min",
    # Our additions, proven valuable in TEXAS 32-feat (lat/lon landed #3/#5 there)
    "lat", "lon",
    # Our additions: legacy gridMET daily weather, seasonal-imputed (rmin was
    # TEXAS 32-feat's #4 feature by gain)
    "erc", "fm100", "vpd", "vs", "rmax", "rmin", "tmmx", "pr",
]

def metrics(y_true, p):
    aucpr = average_precision_score(y_true, p)
    auroc = roc_auc_score(y_true, p)
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = (2 * prec * rec / (prec + rec + 1e-12))
    best_i = np.nanargmax(f1[:-1]) if len(f1) > 1 else 0
    best_thr = thr[best_i] if len(thr) > best_i else 0.5
    best_f1, best_p, best_r = f1[best_i], prec[best_i], rec[best_i]
    base_rate = y_true.mean()
    return {
        "aucpr": float(aucpr), "auroc": float(auroc),
        "f1": float(best_f1), "precision": float(best_p), "recall": float(best_r),
        "best_threshold": float(best_thr), "base_rate": float(base_rate),
        "lift": float(aucpr / base_rate),
    }

def main():
    t0 = time.time()
    print(f"Loading {DATA} ...")
    df = pd.read_parquet(DATA)
    print(f"Shape: {df.shape}")

    train = df[df.split == "train"]
    val = df[df.split == "val"]
    test = df[df.split == "test"]
    print(f"Train {len(train):,} ({train.label.mean():.4f} pos) | "
          f"Val {len(val):,} ({val.label.mean():.4f} pos) | "
          f"Test {len(test):,} ({test.label.mean():.4f} pos)")

    X_train, y_train = train[FEATURES], train["label"].values
    X_val, y_val = val[FEATURES], val["label"].values
    X_test, y_test = test[FEATURES], test["label"].values

    spw = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {spw:.4f}")

    # ---------------- Stage 1: tree structure search ----------------
    print("\n=== Stage 1: max_depth x min_child_weight (12 trials) ===")
    stage1_results = []
    for depth in [6, 7, 8, 9]:
        for mcw in [10, 20, 30]:
            m = xgb.XGBClassifier(
                n_estimators=400, max_depth=depth, min_child_weight=mcw,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=spw, eval_metric="aucpr",
                early_stopping_rounds=30, tree_method="hist", device="cuda",
                random_state=42, n_jobs=-1,
            )
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            p = m.predict_proba(X_val)[:, 1]
            r = {"depth": depth, "mcw": mcw,
                 "val_auroc": roc_auc_score(y_val, p),
                 "val_aucpr": average_precision_score(y_val, p),
                 "best_iter": m.best_iteration}
            stage1_results.append(r)
            print(f"  depth={depth} mcw={mcw}: val_auroc={r['val_auroc']:.4f} "
                  f"val_aucpr={r['val_aucpr']:.4f} (best_iter={r['best_iter']})")

    best1 = max(stage1_results, key=lambda r: r["val_aucpr"])
    BEST_DEPTH, BEST_MCW = best1["depth"], best1["mcw"]
    print(f"Stage 1 winner: depth={BEST_DEPTH}, min_child_weight={BEST_MCW}")

    # ---------------- Stage 2: learning_rate x subsample search ----------------
    print("\n=== Stage 2: learning_rate x subsample (9 trials) ===")
    stage2_results = []
    for lr in [0.01, 0.02, 0.05]:
        for sub in [0.7, 0.8, 0.9]:
            m = xgb.XGBClassifier(
                n_estimators=1200, max_depth=BEST_DEPTH, min_child_weight=BEST_MCW,
                learning_rate=lr, subsample=sub, colsample_bytree=0.8,
                scale_pos_weight=spw, eval_metric="aucpr",
                early_stopping_rounds=50, tree_method="hist", device="cuda",
                random_state=42, n_jobs=-1,
            )
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            p = m.predict_proba(X_val)[:, 1]
            r = {"lr": lr, "subsample": sub,
                 "val_auroc": roc_auc_score(y_val, p),
                 "val_aucpr": average_precision_score(y_val, p),
                 "best_iter": m.best_iteration}
            stage2_results.append(r)
            print(f"  lr={lr} sub={sub}: val_auroc={r['val_auroc']:.4f} "
                  f"val_aucpr={r['val_aucpr']:.4f} (best_iter={r['best_iter']})")

    best2 = max(stage2_results, key=lambda r: r["val_aucpr"])
    BEST_LR, BEST_SUB = best2["lr"], best2["subsample"]
    print(f"Stage 2 winner: lr={BEST_LR}, subsample={BEST_SUB}")

    pd.DataFrame(stage1_results).to_csv(OUT / "tuning_stage1.csv", index=False)
    pd.DataFrame(stage2_results).to_csv(OUT / "tuning_stage2.csv", index=False)

    # ---------------- Final model ----------------
    print("\n=== Final training with winning hyperparameters ===")
    final_params = dict(
        n_estimators=3000, max_depth=BEST_DEPTH, min_child_weight=BEST_MCW,
        learning_rate=BEST_LR, subsample=BEST_SUB, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="aucpr",
        early_stopping_rounds=80, tree_method="hist", device="cuda",
        random_state=42, n_jobs=-1,
    )
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"Best iteration: {final_model.best_iteration}")

    # ---------------- Raw evaluation ----------------
    p_train = final_model.predict_proba(X_train)[:, 1]
    p_val = final_model.predict_proba(X_val)[:, 1]
    p_test = final_model.predict_proba(X_test)[:, 1]

    m_train = metrics(y_train, p_train)
    m_val = metrics(y_val, p_val)
    m_test = metrics(y_test, p_test)
    print(f"\nRAW  Train AUROC={m_train['auroc']:.4f} AUPR={m_train['aucpr']:.4f}")
    print(f"RAW  Val   AUROC={m_val['auroc']:.4f} AUPR={m_val['aucpr']:.4f}")
    print(f"RAW  Test  AUROC={m_test['auroc']:.4f} AUPR={m_test['aucpr']:.4f} "
          f"F1={m_test['f1']:.4f} Lift={m_test['lift']:.2f}x")

    # ---------------- Isotonic calibration (fit on val, never on test) ----------------
    print("\n=== Isotonic calibration (fit on val) ===")
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    calibrator.fit(p_val, y_val)
    p_val_cal = calibrator.predict(p_val)
    p_test_cal = calibrator.predict(p_test)

    m_val_cal = metrics(y_val, p_val_cal)
    m_test_cal = metrics(y_test, p_test_cal)
    brier_raw = brier_score_loss(y_test, p_test)
    brier_cal = brier_score_loss(y_test, p_test_cal)
    print(f"CAL  Val   AUROC={m_val_cal['auroc']:.4f} AUPR={m_val_cal['aucpr']:.4f}")
    print(f"CAL  Test  AUROC={m_test_cal['auroc']:.4f} AUPR={m_test_cal['aucpr']:.4f}")
    print(f"Test Brier raw={brier_raw:.4f} -> calibrated={brier_cal:.4f}")

    # ---------------- Feature importance ----------------
    importances = final_model.get_booster().get_score(importance_type="gain")
    total_gain = sum(importances.values())
    fi = sorted(
        [{"feature": f, "gain": g, "pct": g / total_gain} for f, g in importances.items()],
        key=lambda r: -r["gain"]
    )
    # Include zero-importance features (not split on at all)
    used = {r["feature"] for r in fi}
    for f in FEATURES:
        if f not in used:
            fi.append({"feature": f, "gain": 0.0, "pct": 0.0})

    print("\nTop 15 features by gain:")
    for r in fi[:15]:
        print(f"  {r['feature']:20s} gain={r['gain']:10.2f}  {r['pct']:.2%}")

    # ---------------- Save everything ----------------
    final_model.save_model(str(OUT / "model_bit_38feat.json"))
    joblib.dump(calibrator, OUT / "calibrator_bit_38feat.joblib")

    results = {
        "built": pd.Timestamp.now().isoformat(),
        "runtime_seconds": time.time() - t0,
        "dataset": str(DATA),
        "n_features": len(FEATURES),
        "features": FEATURES,
        "rows": {"train": len(train), "val": len(val), "test": len(test), "total": len(df)},
        "positive_rate": {"train": float(y_train.mean()), "val": float(y_val.mean()), "test": float(y_test.mean())},
        "split_date_ranges": {
            "train": [str(train.date.min().date()), str(train.date.max().date())],
            "val": [str(val.date.min().date()), str(val.date.max().date())],
            "test": [str(test.date.min().date()), str(test.date.max().date())],
        },
        "hyperparameter_search": {
            "stage1_winner": best1, "stage2_winner": best2,
            "stage1_all": stage1_results, "stage2_all": stage2_results,
        },
        "final_hyperparameters": {k: v for k, v in final_params.items() if k != "scale_pos_weight"} | {
            "scale_pos_weight": spw, "best_iteration": int(final_model.best_iteration)
        },
        "metrics_raw": {"train": m_train, "val": m_val, "test": m_test},
        "metrics_calibrated": {"val": m_val_cal, "test": m_test_cal},
        "calibration": {
            "brier_test_raw": float(brier_raw), "brier_test_calibrated": float(brier_cal),
        },
        "feature_importance_gain": fi,
        "comparison_vs_tdis_served_model": {
            "tdis_served_test_aucpr": 0.4941, "tdis_served_test_auroc": 0.7402,
            "tdis_served_test_f1": 0.4958,
            "tdis_served_realpop_aucpr": 0.0878, "tdis_served_realpop_auroc": 0.7968,
            "tdis_served_realpop_lift": 4.58,
            "note": ("TDIS served-model numbers are on THEIR balanced/matched test "
                     "set (23.4% positive rate, 2023-2026) and separately on their "
                     "real deployment population (1.92% base rate). Our numbers above "
                     "are on a different test population (30.3% positive rate, "
                     "2022-2026, TDIS's own tri-state-table split) -- NOT directly "
                     "comparable to either of TDIS's two reported numbers. See the "
                     "MD report for the lift-based fair comparison and the explicit "
                     "caveat that we have not run real-population validation."),
        },
    }
    with open(OUT / "training_results_bit_38feat.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved model, calibrator, and results to {OUT}")
    print(f"Total runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
