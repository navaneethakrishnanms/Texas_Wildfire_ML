"""
TDIS_Ignition_BIT_V2 — 36-feature retrain with improved hyperparameter search.

Improvements over the BIT v1 search (which picked min_child_weight=10 and
showed a larger train->test gap than our other models, 0.143 AUROC):

  1. A 3rd tuning stage (regularization: gamma x colsample_bytree) that v1
     did not have.
  2. A wider min_child_weight grid (10/20/30/50) in stage 1.
  3. Selection rule changed from "pick the single highest val AUC-PR" to
     "pick the highest val AUC-PR within 0.003 of the best, then prefer the
     *more regularized* (higher min_child_weight) option among those" --
     this directly targets the overfitting concern raised after v1,
     trading a small amount of validation score for a smaller train/test
     gap and a more stable model.

Data: TDIS_Ignition_BIT_V2/data/processed/bit_v2_train_tx.parquet
      (36 features -- the BIT v1 38-feature set minus lat/lon)
GPU: device='cuda' (torch_gpu conda env)
"""
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    brier_score_loss, confusion_matrix
)
from sklearn.isotonic import IsotonicRegression
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "bit_v2_train_tx.parquet"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "road_dist_km", "ecoregion_id", "elevation_m", "slope_deg", "aspect_deg",
    "avg_burn_prob", "whp", "flep4", "cfl", "cbd", "cbh", "powerline_dist_km",
    "sin_month", "cos_month", "sin_dow", "cos_dow", "is_weekend", "is_holiday",
    "hrrr_tmp", "hrrr_vpd", "hrrr_wind", "hrrr_mstav", "drought_score",
    "bp_x_vpd", "whp_x_vpd", "bp_x_drought", "pop_density", "fm100_5d_min",
    "erc", "fm100", "vpd", "vs", "rmax", "rmin", "tmmx", "pr",
]
assert len(FEATURES) == 36, f"Expected 36 features, got {len(FEATURES)}"

def metrics(y_true, p):
    aucpr = average_precision_score(y_true, p)
    auroc = roc_auc_score(y_true, p)
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = (2 * prec * rec / (prec + rec + 1e-12))
    best_i = np.nanargmax(f1[:-1]) if len(f1) > 1 else 0
    best_thr = thr[best_i] if len(thr) > best_i else 0.5
    base_rate = y_true.mean()
    return {
        "aucpr": float(aucpr), "auroc": float(auroc),
        "f1": float(f1[best_i]), "precision": float(prec[best_i]), "recall": float(rec[best_i]),
        "best_threshold": float(best_thr), "base_rate": float(base_rate),
        "lift": float(aucpr / base_rate),
    }

def pick_best_regularized(results, score_key="val_aucpr", reg_key="mcw", margin=0.003):
    """Pick the trial within `margin` of the best score, preferring higher reg_key."""
    best_score = max(r[score_key] for r in results)
    candidates = [r for r in results if best_score - r[score_key] <= margin]
    return max(candidates, key=lambda r: r[reg_key])

def main():
    t0 = time.time()
    print(f"Loading {DATA} ...")
    df = pd.read_parquet(DATA)
    print(f"Shape: {df.shape}, features: {len(FEATURES)}")

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

    # ---------------- Stage 1: tree structure (16 trials, wider mcw grid) ----------------
    print("\n=== Stage 1: max_depth x min_child_weight (16 trials) ===")
    stage1 = []
    for depth in [6, 7, 8, 9]:
        for mcw in [10, 20, 30, 50]:
            m = xgb.XGBClassifier(
                n_estimators=400, max_depth=depth, min_child_weight=mcw,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=spw, eval_metric="aucpr",
                early_stopping_rounds=30, tree_method="hist", device="cuda",
                random_state=42, n_jobs=-1,
            )
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            p = m.predict_proba(X_val)[:, 1]
            p_tr = m.predict_proba(X_train)[:, 1]
            r = {"depth": depth, "mcw": mcw,
                 "val_auroc": roc_auc_score(y_val, p),
                 "val_aucpr": average_precision_score(y_val, p),
                 "train_val_gap_auroc": roc_auc_score(y_train, p_tr) - roc_auc_score(y_val, p),
                 "best_iter": m.best_iteration}
            stage1.append(r)
            print(f"  depth={depth} mcw={mcw}: val_auroc={r['val_auroc']:.4f} "
                  f"val_aucpr={r['val_aucpr']:.4f} gap={r['train_val_gap_auroc']:.4f} "
                  f"(best_iter={r['best_iter']})")

    best1 = pick_best_regularized(stage1, "val_aucpr", "mcw", margin=0.003)
    BEST_DEPTH, BEST_MCW = best1["depth"], best1["mcw"]
    print(f"Stage 1 winner (best-within-0.003-AUPR, most regularized): "
          f"depth={BEST_DEPTH}, min_child_weight={BEST_MCW}")

    # ---------------- Stage 2: learning_rate x subsample (9 trials) ----------------
    print("\n=== Stage 2: learning_rate x subsample (9 trials) ===")
    stage2 = []
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
            stage2.append(r)
            print(f"  lr={lr} sub={sub}: val_auroc={r['val_auroc']:.4f} "
                  f"val_aucpr={r['val_aucpr']:.4f} (best_iter={r['best_iter']})")

    best2 = max(stage2, key=lambda r: r["val_aucpr"])
    BEST_LR, BEST_SUB = best2["lr"], best2["subsample"]
    print(f"Stage 2 winner: lr={BEST_LR}, subsample={BEST_SUB}")

    # ---------------- Stage 3: regularization (gamma x colsample_bytree, 9 trials) ----------------
    print("\n=== Stage 3: gamma x colsample_bytree (9 trials, new in V2) ===")
    stage3 = []
    for gamma in [0.0, 0.1, 0.3]:
        for cs in [0.7, 0.8, 0.9]:
            m = xgb.XGBClassifier(
                n_estimators=1500, max_depth=BEST_DEPTH, min_child_weight=BEST_MCW,
                learning_rate=BEST_LR, subsample=BEST_SUB, colsample_bytree=cs,
                gamma=gamma, scale_pos_weight=spw, eval_metric="aucpr",
                early_stopping_rounds=60, tree_method="hist", device="cuda",
                random_state=42, n_jobs=-1,
            )
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            p = m.predict_proba(X_val)[:, 1]
            r = {"gamma": gamma, "colsample_bytree": cs,
                 "val_auroc": roc_auc_score(y_val, p),
                 "val_aucpr": average_precision_score(y_val, p),
                 "best_iter": m.best_iteration}
            stage3.append(r)
            print(f"  gamma={gamma} colsample={cs}: val_auroc={r['val_auroc']:.4f} "
                  f"val_aucpr={r['val_aucpr']:.4f} (best_iter={r['best_iter']})")

    best3 = max(stage3, key=lambda r: r["val_aucpr"])
    BEST_GAMMA, BEST_CS = best3["gamma"], best3["colsample_bytree"]
    print(f"Stage 3 winner: gamma={BEST_GAMMA}, colsample_bytree={BEST_CS}")

    pd.DataFrame(stage1).to_csv(RESULTS_DIR / "tuning_stage1.csv", index=False)
    pd.DataFrame(stage2).to_csv(RESULTS_DIR / "tuning_stage2.csv", index=False)
    pd.DataFrame(stage3).to_csv(RESULTS_DIR / "tuning_stage3.csv", index=False)

    # ---------------- Final model ----------------
    print("\n=== Final training with winning hyperparameters ===")
    final_params = dict(
        n_estimators=3000, max_depth=BEST_DEPTH, min_child_weight=BEST_MCW,
        learning_rate=BEST_LR, subsample=BEST_SUB, colsample_bytree=BEST_CS,
        gamma=BEST_GAMMA, scale_pos_weight=spw, eval_metric="aucpr",
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

    m_train, m_val, m_test = metrics(y_train, p_train), metrics(y_val, p_val), metrics(y_test, p_test)
    gap_auroc = m_train["auroc"] - m_test["auroc"]
    print(f"\nRAW  Train AUROC={m_train['auroc']:.4f} AUPR={m_train['aucpr']:.4f}")
    print(f"RAW  Val   AUROC={m_val['auroc']:.4f} AUPR={m_val['aucpr']:.4f}")
    print(f"RAW  Test  AUROC={m_test['auroc']:.4f} AUPR={m_test['aucpr']:.4f} "
          f"F1={m_test['f1']:.4f} Lift={m_test['lift']:.2f}x")
    print(f"Train->Test AUROC gap: {gap_auroc:.4f}")

    # ---------------- Isotonic calibration ----------------
    print("\n=== Isotonic calibration (fit on val) ===")
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    calibrator.fit(p_val, y_val)
    p_test_cal = calibrator.predict(p_test)
    m_test_cal = metrics(y_test, p_test_cal)
    brier_raw = brier_score_loss(y_test, p_test)
    brier_cal = brier_score_loss(y_test, p_test_cal)
    print(f"CAL  Test  AUROC={m_test_cal['auroc']:.4f} AUPR={m_test_cal['aucpr']:.4f}")
    print(f"Test Brier raw={brier_raw:.4f} -> calibrated={brier_cal:.4f}")

    # ---------------- Confusion matrices (exact) ----------------
    def cm_at(p, thr):
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        return {"threshold": float(thr), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
                "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
                "accuracy": float((tp + tn) / len(y_test))}

    cm_raw_bestf1 = cm_at(p_test, m_test["best_threshold"])
    cm_cal_bestf1 = cm_at(p_test_cal, m_test_cal["best_threshold"])
    cm_raw_05 = cm_at(p_test, 0.5)
    print("\nConfusion matrix (raw, best-F1 thr):", cm_raw_bestf1)
    print("Confusion matrix (calibrated, best-F1 thr):", cm_cal_bestf1)
    print("Confusion matrix (raw, thr=0.5):", cm_raw_05)

    # ---------------- Feature importance ----------------
    importances = final_model.get_booster().get_score(importance_type="gain")
    total_gain = sum(importances.values())
    fi = sorted(
        [{"feature": f, "gain": g, "pct": g / total_gain} for f, g in importances.items()],
        key=lambda r: -r["gain"]
    )
    used = {r["feature"] for r in fi}
    for f in FEATURES:
        if f not in used:
            fi.append({"feature": f, "gain": 0.0, "pct": 0.0})

    print("\nTop 15 features by gain:")
    for r in fi[:15]:
        print(f"  {r['feature']:20s} gain={r['gain']:10.2f}  {r['pct']:.2%}")

    # ---------------- Save everything ----------------
    final_model.save_model(str(MODEL_DIR / "model_bit_v2_36feat.json"))
    joblib.dump(calibrator, MODEL_DIR / "calibrator_bit_v2_36feat.joblib")
    with open(MODEL_DIR / "feature_list.json", "w") as f:
        json.dump({"n_features": len(FEATURES), "features": FEATURES}, f, indent=2)

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
            "stage1_winner": best1, "stage2_winner": best2, "stage3_winner": best3,
            "stage1_all": stage1, "stage2_all": stage2, "stage3_all": stage3,
            "selection_rule": "stage1 picked via best-val-AUPR-within-0.003-margin, "
                              "preferring higher min_child_weight (more regularized)",
        },
        "final_hyperparameters": {k: v for k, v in final_params.items() if k != "scale_pos_weight"} | {
            "scale_pos_weight": spw, "best_iteration": int(final_model.best_iteration)
        },
        "metrics_raw": {"train": m_train, "val": m_val, "test": m_test},
        "metrics_calibrated": {"test": m_test_cal},
        "calibration": {"brier_test_raw": float(brier_raw), "brier_test_calibrated": float(brier_cal)},
        "train_test_auroc_gap": float(gap_auroc),
        "confusion_matrices": {
            "raw_best_f1_threshold": cm_raw_bestf1,
            "calibrated_best_f1_threshold": cm_cal_bestf1,
            "raw_threshold_0.5": cm_raw_05,
        },
        "feature_importance_gain": fi,
        "comparison_vs_bit_v1_38feat": {
            "note": "BIT v1 (38 features, incl. lat/lon) test AUROC=0.8064 AUPR=0.7237 F1=0.6284 "
                    "on the same split/population. Compare directly -- same data, same split, "
                    "only lat/lon removed and tuning improved.",
        },
    }
    with open(RESULTS_DIR / "training_results_bit_v2_36feat.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved model, calibrator, feature list, and results under {ROOT}")
    print(f"Total runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
