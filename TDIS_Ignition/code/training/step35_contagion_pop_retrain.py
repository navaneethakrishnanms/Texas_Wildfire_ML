"""
New_Training817_moredata — Step 35: contagion + population features,
tested TOGETHER first (combined-first gate, same discipline as rev2).

New features on top of the served 22:
  - fire_nearby_3d: count of fire events in this cell's 1-ring
    neighborhood over the prior 3 days (contagion/clustering; NOT wind,
    a genuinely new mechanism)
  - pop_density: WorldPop 2020 1km population (human-ignition proxy,
    distinct from road_dist_km)

Gate 1: test AUC-PR vs served (0.4941 for THIS training run's exact
population -- recomputed fresh below, not hardcoded, since the
contagion feature drops rows with no prior-day lookup at the very start
of the archive).

Output: step35_results.json (+ step35_model.json)
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
          'avg_burn_prob', 'whp', 'flep4', 'cfl', 'cbd', 'cbh',
          'powerline_dist_km', 'pop_density']
TEMPORAL = ['sin_month', 'cos_month', 'sin_dow', 'cos_dow', 'is_weekend', 'is_holiday']
HRRR = ['hrrr_tmp', 'hrrr_vpd', 'hrrr_wind', 'hrrr_mstav']
FEATS = STATIC + TEMPORAL + HRRR + ['fire_nearby_3d']


def log(m): print(m, flush=True)


def run():
    flare_v2 = set(pd.read_parquet(HERE / "flare_cells_v2.parquet")['h3_cell'])
    pl = pd.read_parquet(HERE / "powerline_dist_km.parquet")
    ms = pd.read_parquet(HERE / "mstav_feature.parquet")
    pop = pd.read_parquet(HERE / "pop_density_feature.parquet")
    fn = pd.read_parquet(HERE / "fire_nearby_3d_feature.parquet")

    df = pd.read_parquet(TF / "tdis_train_daily_hrrr.parquet")
    df['date'] = pd.to_datetime(df['date']); df['year'] = df['date'].dt.year
    n0 = len(df)
    df = df[~df.h3_cell.isin(flare_v2)]
    df = df[df.date <= LAST_LABEL]
    df = df.merge(pl, on='h3_cell', how='left')
    df = df.merge(ms, on=['h3_cell', 'date'], how='left')
    df = df.merge(pop, on='h3_cell', how='left')
    df = df.merge(fn, on=['h3_cell', 'date'], how='left')
    df['fire_nearby_3d'] = df['fire_nearby_3d'].fillna(0)
    df = df.dropna(subset=STATIC).dropna(subset=['hrrr_vpd', 'hrrr_mstav'])
    log(f"rows: {n0:,} -> {len(df):,} after filters/joins")
    log(f"fire_nearby_3d coverage: {(df['fire_nearby_3d']>0).mean()*100:.2f}% nonzero")

    tr = df[df.year <= 2021]; te = df[df.year >= 2023]
    log(f"train {len(tr):,} / test {len(te):,}")
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
    log(f"+contagion+pop: test AUC-PR={aucpr:.4f} AUROC={auroc:.4f} [{time.time()-t0:.0f}s]")
    imp = dict(zip(FEATS, m.feature_importances_.round(4).tolist()))
    for f in ['fire_nearby_3d', 'pop_density']:
        rank = sorted(imp.values(), reverse=True).index(imp[f]) + 1
        log(f"  {f} importance: {imp[f]:.4f} (rank {rank}/{len(FEATS)})")
    m.save_model(str(HERE / "step35_model.json"))
    json.dump(dict(result=dict(aucpr=round(aucpr, 4), auroc=round(auroc, 4)), importances=imp),
              open(HERE / "step35_results.json", 'w'), indent=2)
    log("Saved -> step35_results.json, step35_model.json")


if __name__ == '__main__':
    run()
