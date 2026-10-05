"""
New_Training817_moredata — Step 32: retrain with the power-line feature +
the type-flag-cleaned flare list (v2).

Two label/feature changes, tested TOGETHER first (rev2 lesson: cheap
combined gate before decomposing):
  - new static feature: powerline_dist_km (HIFLD TX transmission lines,
    distance from each res-8 cell to nearest line, densified to ~1km)
  - flare_cells_v2 (531 + 7 cells found by the VIIRS type-flag audit)

Config = the PROMOTED depth-9 config. Same table, same split. Gate 1 is
test AUC-PR vs the promoted model's 0.4875; if it clears, gates 2 (seeds)
and 3 (real population) follow before any promotion.

Output: step32_results.json (+ step32_model.json)
"""
import json, time, warnings
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
TF = HERE.parent
LAST_LABEL = pd.Timestamp('2026-07-29')

STATIC = ['road_dist_km', 'ecoregion_id', 'elevation_m', 'slope_deg', 'aspect_deg',
          'avg_burn_prob', 'whp', 'flep4', 'cfl', 'cbd', 'cbh', 'powerline_dist_km']
TEMPORAL = ['sin_month', 'cos_month', 'sin_dow', 'cos_dow', 'is_weekend', 'is_holiday']
HRRR = ['hrrr_tmp', 'hrrr_vpd', 'hrrr_wind']
FEATS = STATIC + TEMPORAL + HRRR
PROMOTED = dict(aucpr=0.4875, auroc=0.7331)


def log(m): print(m, flush=True)


def run():
    flare_v2 = set(pd.read_parquet(HERE / "flare_cells_v2.parquet")['h3_cell'])
    pl = pd.read_parquet(HERE / "powerline_dist_km.parquet")
    df = pd.read_parquet(TF / "tdis_train_daily_hrrr.parquet")
    df['date'] = pd.to_datetime(df['date']); df['year'] = df['date'].dt.year
    df = df[~df.h3_cell.isin(flare_v2)]
    df = df[df.date <= LAST_LABEL]
    df = df.merge(pl, on='h3_cell', how='left')
    df = df.dropna(subset=STATIC).dropna(subset=['hrrr_vpd'])
    tr = df[df.year <= 2021]; te = df[df.year >= 2023]
    log(f"train {len(tr):,} / test {len(te):,} (flare v2: {len(flare_v2)} cells excluded)")

    spw = (tr.label == 0).sum() / max((tr.label == 1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=1000, max_depth=9, min_child_weight=30,
                           learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
                           scale_pos_weight=spw, eval_metric='aucpr',
                           tree_method='hist', device='cuda', n_jobs=-1, random_state=42)
    t0 = time.time()
    m.fit(tr[FEATS], tr.label.astype(int), verbose=False)
    yt = te.label.astype(int).values
    pt = m.predict_proba(te[FEATS])[:, 1]
    aucpr = float(average_precision_score(yt, pt))
    auroc = float(roc_auc_score(yt, pt))
    log(f"powerline+flare_v2: test AUC-PR={aucpr:.4f} ({aucpr-PROMOTED['aucpr']:+.4f}) "
        f"AUROC={auroc:.4f} ({auroc-PROMOTED['auroc']:+.4f}) [{time.time()-t0:.0f}s]")
    imp = dict(zip(FEATS, m.feature_importances_.round(4).tolist()))
    log(f"powerline_dist_km importance: {imp['powerline_dist_km']:.4f} "
        f"(rank {sorted(imp.values(), reverse=True).index(imp['powerline_dist_km'])+1}/{len(FEATS)})")
    m.save_model(str(HERE / "step32_model.json"))
    json.dump(dict(promoted_baseline=PROMOTED,
                   result=dict(aucpr=round(aucpr, 4), auroc=round(auroc, 4),
                               delta_aucpr=round(aucpr - PROMOTED['aucpr'], 4)),
                   importances=imp),
              open(HERE / "step32_results.json", 'w'), indent=2)
    log("Saved -> step32_results.json, step32_model.json")


if __name__ == '__main__':
    run()
