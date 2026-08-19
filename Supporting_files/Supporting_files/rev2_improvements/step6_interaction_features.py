"""
rev2_improvements — Step 6: EXPLICIT wind-dryness interaction features.

Motivated directly by the Step 5 SHAP finding: the tree ensemble does NOT
discover the multiplicative wind x dryness interaction on its own (interaction
SHAP ~0.013, unchanged whether it gets mean wind or gust), even though both
real-event validation (wind-driven fires under-scored) and the NOAA HWP formula
say that interaction is where the danger lives. Trees only find interactions if
depth + data density in that region support it -- nothing forces them to. So
this experiment hands the interaction to the model as explicit input features
instead of hoping it gets discovered:

  1. gust_x_vpd  = hrrr_gust * hrrr_vpd            (plain product)
  2. hwp_core    = 0.213 * max(gust,3)^1.5 * (vpd*10)^0.73
                   (NOAA HWP Eq.3 minus the soil/snow terms -- gust in m/s
                    floored at 3 per the paper, vpd kPa -> hPa)
  3. gust_x_dry5 = hrrr_gust * erc_5D_mean          (gust x persistent dryness)

All three are forecast-knowable (HRRR forecast gust/vpd + trailing stats of
PAST days), so no observed-weather leakage. Same table/filters/split/
hyperparameters as Step 5 and script 19 -- directly comparable to:
  baseline: AUC-PR=0.4825  AUROC=0.7333  lift=2.06x
  step 5:   AUC-PR=0.4823  AUROC=0.7348  lift=2.06x

Also caches the merged gust+trailing table to combined_table.parquet on first
build so later experiments (and diagnose_model.py) stop re-deriving it from
scratch (~15 min each time).

Output: rev2_improvements/step6_results.json, step6_model.json
"""
import json, warnings
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')

TF = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).parent
CACHE = OUT_DIR / "combined_table.parquet"
LAST_LABEL = pd.Timestamp('2026-07-29')

STATIC = ['road_dist_km', 'ecoregion_id', 'elevation_m', 'slope_deg', 'aspect_deg',
          'avg_burn_prob', 'whp', 'flep4', 'cfl', 'cbd', 'cbh']
TEMPORAL = ['sin_month', 'cos_month', 'sin_dow', 'cos_dow', 'is_weekend', 'is_holiday']
HRRR_FEATS = ['hrrr_tmp', 'hrrr_vpd', 'hrrr_wind', 'hrrr_gust']
TRAILING_FEATS = ['erc_5D_mean', 'erc_5D_max', 'vpd_5D_mean', 'vpd_5D_max', 'vs_5D_mean', 'vs_5D_max']
INTERACTION_FEATS = ['gust_x_vpd', 'hwp_core', 'gust_x_dry5']
FEATS = STATIC + TEMPORAL + HRRR_FEATS + TRAILING_FEATS + INTERACTION_FEATS


def log(m): print(m, flush=True)


def get_combined_table():
    if CACHE.exists():
        log(f"Loading cached combined table ({CACHE.name})...")
        df = pd.read_parquet(CACHE)
        df['date'] = pd.to_datetime(df['date'])
        return df
    log("No cache -- rebuilding gust + trailing features via step5 module (slow path)...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("step5mod", OUT_DIR / "step5_combined_gust_trailing.py")
    step5mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step5mod)
    gust = step5mod.build_gust()
    trailing = step5mod.build_trailing_stats()
    base = pd.read_parquet(TF / "tdis_train_daily_hrrr.parquet")
    base['date'] = pd.to_datetime(base['date'])
    df = base.merge(gust, on=['h3_cell', 'date'], how='left').merge(trailing, on=['h3_cell', 'date'], how='left')
    df.to_parquet(CACHE, index=False)
    log(f"Cached combined table -> {CACHE.name} ({CACHE.stat().st_size/1e6:.0f} MB)")
    return df


def run():
    df = get_combined_table()
    log(f"Combined table: {len(df):,} rows")

    log("Engineering interaction features (all forecast-knowable)...")
    g = df['hrrr_gust']
    v = df['hrrr_vpd']
    df['gust_x_vpd'] = g * v
    df['hwp_core'] = 0.213 * np.maximum(g, 3.0) ** 1.5 * np.clip(v * 10.0, 0.01, None) ** 0.73
    df['gust_x_dry5'] = g * df['erc_5D_mean']
    for f in INTERACTION_FEATS:
        log(f"  {f}: non-null {df[f].notna().mean()*100:.1f}%, mean={df[f].mean():.3f}")

    log("Applying flare-cell + phantom-date + coverage filters (matches script 19)...")
    flare_set = set(pd.read_parquet(TF / "data/labels_fused/flare_cells.parquet")['h3_cell'])
    df['year'] = df['date'].dt.year
    n0 = len(df)
    df = df[~df.h3_cell.isin(flare_set)]
    df = df[df.date <= LAST_LABEL]
    df = df.dropna(subset=STATIC).dropna(subset=['hrrr_vpd'])
    log(f"  rows: {n0:,} -> {len(df):,}")
    df['split'] = np.where(df.year <= 2021, 'train', np.where(df.year == 2022, 'val', 'test'))
    tr, va, te = df[df.split == 'train'], df[df.split == 'val'], df[df.split == 'test']
    log(f"  train {len(tr):,} / val {len(va):,} / test {len(te):,}  test pos rate {te.label.mean():.4f}")

    log("Training (same hyperparameters as script 19 / step 5)...")
    spw = (tr.label == 0).sum() / max((tr.label == 1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, min_child_weight=20, scale_pos_weight=spw,
                           eval_metric='aucpr', tree_method='hist', device='cuda',
                           n_jobs=-1, random_state=42)
    m.fit(tr[FEATS], tr.label.astype(int), eval_set=[(va[FEATS], va.label.astype(int))], verbose=False)

    y = te.label.astype(int).values
    p = m.predict_proba(te[FEATS])[:, 1]
    aucpr, auroc = average_precision_score(y, p), roc_auc_score(y, p)
    lift = aucpr / y.mean()
    log(f"\nSTEP 6 (explicit interactions, default hp): AUC-PR={aucpr:.4f} AUROC={auroc:.4f} "
        f"lift={lift:.2f}x  n={len(te):,}")
    log("  vs baseline: AUC-PR=0.4825 AUROC=0.7333 | vs step5: AUC-PR=0.4823 AUROC=0.7348")
    sweep = {}
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred = (p > t).astype(int)
        sweep[str(t)] = dict(precision=round(float(precision_score(y, pred, zero_division=0)), 3),
                              recall=round(float(recall_score(y, pred, zero_division=0)), 3),
                              f1=round(float(f1_score(y, pred, zero_division=0)), 3))
        log(f"  t={t}  P={sweep[str(t)]['precision']}  R={sweep[str(t)]['recall']}  F1={sweep[str(t)]['f1']}")

    imp = dict(sorted(zip(FEATS, [round(float(x), 4) for x in m.feature_importances_]), key=lambda x: -x[1]))
    log("\nFull importance ranking of the 3 interaction features:")
    ranked = list(imp.items())
    for i, (k, val) in enumerate(ranked):
        if k in INTERACTION_FEATS:
            log(f"  rank {i+1}/{len(ranked)}: {k} = {val}")
    log("\nTop 10 overall:")
    for k, val in ranked[:10]:
        log(f"  {k}: {val}")

    m.save_model(str(OUT_DIR / "step6_model.json"))
    json.dump(dict(n_test=len(te), test_pos_rate=round(float(y.mean()), 4),
                   aucpr=round(float(aucpr), 4), auroc=round(float(auroc), 4),
                   lift=round(float(lift), 2), threshold_sweep=sweep,
                   feature_importance=imp, features_used=FEATS,
                   comparisons=dict(baseline=dict(aucpr=0.4825, auroc=0.7333),
                                    step5=dict(aucpr=0.4823, auroc=0.7348))),
              open(OUT_DIR / "step6_results.json", 'w'), indent=2)
    log(f"\nSaved -> step6_results.json, step6_model.json")


if __name__ == '__main__':
    run()
