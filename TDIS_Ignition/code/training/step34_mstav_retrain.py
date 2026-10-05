"""
New_Training817_moredata — Step 34: add HRRR soil moisture (MSTAV) as a
model feature, on top of the step32 power-line model.

MSTAV = soil-moisture availability (%), the M term of NOAA HWP. Freshly
backfilled for every date in the hrrr_24h archive (download_mstav.py,
2,932 dates, 0 errors). Joined to training rows the same way script 06
joins the other HRRR fields: nearest forecast-grid-point per cell per day.

Gate 1: test AUC-PR vs step32's 0.4925 (powerline+flare_v2, d9 config).
Feature table cached to mstav_feature.parquet so re-runs are instant.

Output: step34_results.json (+ step34_model.json)
"""
import json, time, warnings
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
TF = HERE.parent
MST = TF / "data" / "weather_hrrr_forecast" / "hrrr_24h_mstav"
LAST_LABEL = pd.Timestamp('2026-07-29')

STATIC = ['road_dist_km', 'ecoregion_id', 'elevation_m', 'slope_deg', 'aspect_deg',
          'avg_burn_prob', 'whp', 'flep4', 'cfl', 'cbd', 'cbh', 'powerline_dist_km']
TEMPORAL = ['sin_month', 'cos_month', 'sin_dow', 'cos_dow', 'is_weekend', 'is_holiday']
FEATS = STATIC + TEMPORAL + ['hrrr_tmp', 'hrrr_vpd', 'hrrr_wind', 'hrrr_mstav']
STEP32 = dict(aucpr=0.4925, auroc=0.7377)


def log(m): print(m, flush=True)


def build_mstav_feature(cells_dates):
    """cells_dates: DataFrame[h3_cell, date] unique. Returns + mstav column."""
    cache = HERE / "mstav_feature.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    st = pd.read_parquet(TF / "data/static_features/tx_static_master.parquet",
                          columns=['h3_cell', 'lat', 'lon'])
    st = st[st.h3_cell.isin(set(cells_dates['h3_cell']))]
    sample = pd.read_parquet(sorted(MST.glob('*.parquet'))[0])
    tree = cKDTree(sample[['lat', 'lon']].values)
    _, idx = tree.query(st[['lat', 'lon']].values)
    st['gidx'] = idx                      # grid index is stable across days (same HRRR grid)
    cd = cells_dates.merge(st[['h3_cell', 'gidx']], on='h3_cell', how='left')
    out = []
    t0 = time.time()
    for i, (d, grp) in enumerate(cd.groupby('date')):
        f = MST / f"{pd.Timestamp(d).date()}.parquet"
        if not f.exists():
            continue
        m = pd.read_parquet(f, columns=['mstav_pct'])
        vals = m['mstav_pct'].values
        g = grp.copy()
        g['hrrr_mstav'] = np.clip(vals[g['gidx'].values.astype(int)] / 100.0, 0, 1)
        out.append(g[['h3_cell', 'date', 'hrrr_mstav']])
        if (i + 1) % 500 == 0:
            log(f"  mstav join {i+1} dates ({(time.time()-t0)/60:.1f} min)")
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(cache, index=False)
    log(f"  cached {len(res):,} rows -> mstav_feature.parquet")
    return res


def run():
    flare_v2 = set(pd.read_parquet(HERE / "flare_cells_v2.parquet")['h3_cell'])
    pl = pd.read_parquet(HERE / "powerline_dist_km.parquet")
    df = pd.read_parquet(TF / "tdis_train_daily_hrrr.parquet")
    df['date'] = pd.to_datetime(df['date']); df['year'] = df['date'].dt.year
    df = df[~df.h3_cell.isin(flare_v2)]
    df = df[df.date <= LAST_LABEL]
    df = df.merge(pl, on='h3_cell', how='left')
    df = df.dropna(subset=STATIC).dropna(subset=['hrrr_vpd'])

    log("Building/loading MSTAV feature...")
    ms = build_mstav_feature(df[['h3_cell', 'date']].drop_duplicates())
    df = df.merge(ms, on=['h3_cell', 'date'], how='left')
    cov = df['hrrr_mstav'].notna().mean()
    log(f"MSTAV coverage on training rows: {cov*100:.1f}%")
    df = df.dropna(subset=['hrrr_mstav'])

    tr, te = df[df.year <= 2021], df[df.year >= 2023]
    log(f"train {len(tr):,} / test {len(te):,}")
    spw = (tr.label == 0).sum() / max((tr.label == 1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=1000, max_depth=9, min_child_weight=30,
                           learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
                           scale_pos_weight=spw, eval_metric='aucpr',
                           tree_method='hist', device='cuda', n_jobs=-1, random_state=42)
    m.fit(tr[FEATS], tr.label.astype(int), verbose=False)
    yt = te.label.astype(int).values
    pt = m.predict_proba(te[FEATS])[:, 1]
    aucpr = float(average_precision_score(yt, pt))
    auroc = float(roc_auc_score(yt, pt))
    log(f"+mstav: test AUC-PR={aucpr:.4f} ({aucpr-STEP32['aucpr']:+.4f} vs step32) "
        f"AUROC={auroc:.4f} ({auroc-STEP32['auroc']:+.4f})")
    imp = dict(zip(FEATS, m.feature_importances_.round(4).tolist()))
    log(f"hrrr_mstav importance: {imp['hrrr_mstav']:.4f} "
        f"(rank {sorted(imp.values(), reverse=True).index(imp['hrrr_mstav'])+1}/{len(FEATS)})")
    m.save_model(str(HERE / "step34_model.json"))
    json.dump(dict(step32_baseline=STEP32, mstav_coverage=round(float(cov), 4),
                   result=dict(aucpr=round(aucpr, 4), auroc=round(auroc, 4),
                               delta_vs_step32=round(aucpr - STEP32['aucpr'], 4)),
                   importances=imp),
              open(HERE / "step34_results.json", 'w'), indent=2)
    log("Saved -> step34_results.json, step34_model.json")


if __name__ == '__main__':
    run()
