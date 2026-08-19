"""
rev2_improvements — Step 5: COMBINED experiment (gust + 5-day trailing weather
stats), default (unchanged) hyperparameters, before doing individual ablations.
Goal: quick "is this worth pursuing at all" check before spending time isolating
which of the two additions actually drives any gain.

Adds two things to the operational-HRRR feature set (STATIC + TEMPORAL + hrrr_tmp/
hrrr_vpd/hrrr_wind):
  1. hrrr_gust — already being pulled in the cached HRRR archive
     (data/weather_hrrr_forecast/hrrr_24h/*.parquet has a gust_ms column), just
     never carried into the training table by script 06. Re-derived here via the
     same nearest-grid-point method, not a new download.
  2. 5-day TRAILING gridMET stats (erc/vpd/vs, mean+max) computed from the dense
     raw gridmet_tx archive (317,142 res-8 cells x 365 days/year, confirmed dense,
     2014-2026 coverage). Window is STRICTLY the 5 days before the target date
     (closed='left' after a 1-day shift) -- never includes the target day itself,
     to avoid leakage.

Same training setup as script 19 (400 trees, no early stopping, same flare-cell +
phantom-date filters, same chronological split) so results are directly comparable
to the existing operational_hrrr_filtered baseline (AUC-PR=0.4825, AUROC=0.7333,
lift=2.06x on its own clean test rows).

Output: rev2_improvements/step5_results.json
"""
import json, warnings, time
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')

TF = Path(__file__).resolve().parent.parent
GM = TF.parent / "gridmet_tx"
HRRR_DIR = TF / "data" / "weather_hrrr_forecast" / "hrrr_24h"
OUT_DIR = Path(__file__).parent
LAST_LABEL = pd.Timestamp('2026-07-29')
HRRR_START = pd.Timestamp('2018-07-16')

STATIC = ['road_dist_km', 'ecoregion_id', 'elevation_m', 'slope_deg', 'aspect_deg',
          'avg_burn_prob', 'whp', 'flep4', 'cfl', 'cbd', 'cbh']
TEMPORAL = ['sin_month', 'cos_month', 'sin_dow', 'cos_dow', 'is_weekend', 'is_holiday']
HRRR_FEATS = ['hrrr_tmp', 'hrrr_vpd', 'hrrr_wind', 'hrrr_gust']
TRAILING_FEATS = ['erc_5D_mean', 'erc_5D_max', 'vpd_5D_mean', 'vpd_5D_max', 'vs_5D_mean', 'vs_5D_max']
FEATS = STATIC + TEMPORAL + HRRR_FEATS + TRAILING_FEATS


def log(m): print(m, flush=True)


def build_gust():
    log("=== Re-deriving hrrr_gust (cheap: reuses cached HRRR archive) ===")
    tr = pd.read_parquet(TF / "tdis_train_daily_tx.parquet", columns=['h3_cell', 'date'])
    tr['date'] = pd.to_datetime(tr['date']).dt.normalize()
    tr = tr.drop_duplicates()
    log(f"  {len(tr):,} unique (cell,date) rows to attach gust to")

    master = pd.read_parquet(TF / "data" / "static_features" / "tx_static_master.parquet",
                              columns=['h3_cell', 'lat', 'lon'])
    cll = master.drop_duplicates('h3_cell').set_index('h3_cell')
    sample = pd.read_parquet(sorted(HRRR_DIR.glob('*.parquet'))[0], columns=['lat', 'lon'])
    tree = cKDTree(sample[['lat', 'lon']].values)
    ucells = pd.Index(tr['h3_cell'].unique())
    uc = cll.reindex(ucells).dropna()
    _, idx = tree.query(uc[['lat', 'lon']].values)
    uc = uc.copy()
    uc['glat'] = sample['lat'].values[idx].round(3)
    uc['glon'] = sample['lon'].values[idx].round(3)
    tr = tr.merge(uc[['glat', 'glon']], left_on='h3_cell', right_index=True, how='left')

    out_parts = []
    dates = sorted(tr['date'].unique())
    t0 = time.time()
    for i, d in enumerate(dates):
        g = tr[tr['date'] == d]
        f = HRRR_DIR / f"{pd.Timestamp(d).date()}.parquet"
        if d < HRRR_START or not f.exists():
            out_parts.append(g); continue
        hf = pd.read_parquet(f)
        if 'gust_ms' not in hf.columns:
            out_parts.append(g); continue
        hf['glat'] = hf['lat'].round(3); hf['glon'] = hf['lon'].round(3)
        lut = hf.groupby(['glat', 'glon']).agg(hrrr_gust=('gust_ms', 'mean')).reset_index()
        out_parts.append(g.merge(lut, on=['glat', 'glon'], how='left'))
        if (i + 1) % 500 == 0:
            log(f"  {i+1}/{len(dates)} dates ({(time.time()-t0)/60:.1f} min)")
    gust = pd.concat(out_parts, ignore_index=True).drop(columns=['glat', 'glon'], errors='ignore')
    cov = gust['hrrr_gust'].notna().mean() * 100
    log(f"  gust attached. coverage: {cov:.1f}%")
    return gust[['h3_cell', 'date', 'hrrr_gust']]


def build_trailing_stats():
    log("=== Building 5-day trailing gridMET stats (erc/vpd/vs, mean+max) ===")
    tr = pd.read_parquet(TF / "tdis_train_daily_tx.parquet", columns=['h3_cell', 'date'])
    tr['date'] = pd.to_datetime(tr['date']).dt.normalize()
    need = tr.drop_duplicates().copy()
    need_years = sorted(need['date'].dt.year.unique())
    log(f"  years needed: {need_years[0]}-{need_years[-1]}")

    results = []
    for yr in need_years:
        parts = []
        for var in ['erc', 'vpd', 'vs']:
            fp = GM / f"{var}_{yr}_tx_cells.parquet"
            if not fp.exists():
                continue
            d = pd.read_parquet(fp)
            d['date'] = pd.to_datetime(d['date_utc']).dt.normalize()
            d = d[['h3_cell', 'date', var]].sort_values(['h3_cell', 'date'])
            g = d.groupby('h3_cell')[var]
            # STRICTLY the 5 days before the target date -- shift(1) excludes today,
            # then a 5-row rolling window covers days t-5..t-1.
            d[f'{var}_5D_mean'] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
            d[f'{var}_5D_max'] = g.transform(lambda s: s.shift(1).rolling(5, min_periods=3).max())
            need_yr = need[need['date'].dt.year == yr]
            d = d.merge(need_yr, on=['h3_cell', 'date'], how='inner')
            parts.append(d[['h3_cell', 'date', f'{var}_5D_mean', f'{var}_5D_max']])
            del g
        if not parts:
            continue
        merged = parts[0]
        for p in parts[1:]:
            merged = merged.merge(p, on=['h3_cell', 'date'], how='outer')
        results.append(merged)
        log(f"  {yr}: {len(merged):,} rows with trailing stats")
        del parts

    out = pd.concat(results, ignore_index=True)
    log(f"  total trailing-stat rows: {len(out):,}")
    return out


def run():
    gust = build_gust()
    trailing = build_trailing_stats()

    log("=== Assembling combined training table ===")
    base = pd.read_parquet(TF / "tdis_train_daily_hrrr.parquet")
    base['date'] = pd.to_datetime(base['date'])
    df = base.merge(gust, on=['h3_cell', 'date'], how='left')
    df = df.merge(trailing, on=['h3_cell', 'date'], how='left')
    log(f"  combined table: {len(df):,} rows, {len(df.columns)} cols")
    for f in FEATS:
        if f not in df.columns:
            raise SystemExit(f"missing expected feature after merge: {f}")

    log("=== Applying flare-cell + phantom-date + coverage filters (matches script 19) ===")
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

    log("=== Training (same hyperparameters as script 19) ===")
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
    log(f"\nSTEP 5 (gust + 5D trailing, default hp): AUC-PR={aucpr:.4f} AUROC={auroc:.4f} "
        f"lift={lift:.2f}x  n={len(te):,}")
    sweep = {}
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred = (p > t).astype(int)
        sweep[str(t)] = dict(precision=round(float(precision_score(y, pred, zero_division=0)), 3),
                              recall=round(float(recall_score(y, pred, zero_division=0)), 3),
                              f1=round(float(f1_score(y, pred, zero_division=0)), 3))
        log(f"  t={t}  P={sweep[str(t)]['precision']}  R={sweep[str(t)]['recall']}  F1={sweep[str(t)]['f1']}")

    imp = dict(sorted(zip(FEATS, [round(float(x), 4) for x in m.feature_importances_]), key=lambda x: -x[1]))
    log("\nTop 10 feature importances:")
    for k, v in list(imp.items())[:10]:
        log(f"  {k}: {v}")

    m.save_model(str(OUT_DIR / "step5_model.json"))
    json.dump(dict(n_test=len(te), test_pos_rate=round(float(y.mean()), 4),
                   aucpr=round(float(aucpr), 4), auroc=round(float(auroc), 4),
                   lift=round(float(lift), 2), threshold_sweep=sweep,
                   feature_importance=imp, features_used=FEATS,
                   baseline_comparison=dict(aucpr=0.4825, auroc=0.7333, lift=2.06)),
              open(OUT_DIR / "step5_results.json", 'w'), indent=2)
    log(f"\nSaved -> {OUT_DIR/'step5_results.json'}, step5_model.json")


if __name__ == '__main__':
    run()
