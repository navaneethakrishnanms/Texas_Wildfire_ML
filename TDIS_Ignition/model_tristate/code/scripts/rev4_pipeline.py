"""
REV4 PIPELINE — daily-PEAK weather features from the 12Z forecast cycle.

The model-level fix for the Smokehouse blindness. Weather features keep
their NAMES (schema compatibility with all infra) but change MEANING:
  hrrr_tmp   = max 2m temperature across target-day 9AM/noon/3PM/6PM local
  hrrr_vpd   = max VPD across those hours
  hrrr_wind  = max GUST across those hours   (gust, not sustained wind --
               the fire-relevant quantity, now from a cycle that sees storms)
  hrrr_mstav = MIN soil-moisture availability (driest hour)
Interactions (bp_x_vpd etc.) computed on the peak values. Same fix1
config (monotone constraints incl.), same split, same gates.

Stages (cached, restart-safe):
  1 build peak-weather table for all training (cell,date) pairs
  2 train 3 seeds -> gate 1a on the same test rows as always
  3 2026 res-8 replay -> missed@tau + THE NAMED-FIRE BATTERY
    (pre-registered pass/fail: Smokehouse cells must leave the bottom)
  4 res-5 real-population replay -> gate 3 lift vs served 4.62
"""
import json, time, warnings
import numpy as np, pandas as pd, xgboost as xgb, h3
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
TF = HERE.parent
NT = TF / "New_Training817_moredata"
DR = TF / "rev2_improvements/dataAdditions/idea5_drought_monitor"
LEADS = [27, 30, 33, 36]
DIRS = {l: TF / f"data/weather_hrrr_forecast/hrrr_12z_f{l}" for l in LEADS}
LAST = pd.Timestamp('2026-07-29')
TAU = 0.4623

STATIC = ['road_dist_km', 'ecoregion_id', 'elevation_m', 'slope_deg', 'aspect_deg',
          'avg_burn_prob', 'whp', 'flep4', 'cfl', 'cbd', 'cbh', 'powerline_dist_km']
BASE = STATIC + ['sin_month', 'cos_month', 'sin_dow', 'cos_dow', 'is_weekend', 'is_holiday',
                  'hrrr_tmp', 'hrrr_vpd', 'hrrr_wind', 'hrrr_mstav', 'drought_score']
FEATS = BASE + ['bp_x_vpd', 'whp_x_vpd', 'bp_x_drought']
# hrrr_wind/hrrr_vpd added after SHAP diagnosis: unconstrained, the model learned a
# non-monotone gust response (22 m/s -> -0.42 log-odds, 27 m/s -> +0.34) that sank
# Smokehouse. Physically, ignition risk cannot fall as gusts or dryness rise.
MONO_UP = {'avg_burn_prob', 'whp', 'drought_score', 'bp_x_vpd', 'whp_x_vpd', 'bp_x_drought',
           'hrrr_wind', 'hrrr_vpd'}
HOL = {(1, 1), (7, 4), (6, 19), (11, 11), (12, 25), (10, 31)}
SERVED = dict(test_aucpr=0.4941, lift=4.62, g4_rank=0.612, missed_res8=8.53)
EVENTS = [
 ("Smokehouse Creek", '2024-02-26', 35.5, 36.3, -101.6, -100.2),
 ("Windy Deuce", '2024-02-27', 35.55, 35.95, -102.0, -101.3),
 ("Crabapple", '2025-03-15', 30.15, 30.75, -99.15, -98.55),
 ("Lavender", '2026-02-17', 35.1, 35.8, -102.9, -101.4),
 ("Hunggate", '2026-05-14', 34.55, 35.25, -102.7, -101.4),
]


def log(m): print(m, flush=True)


def read_day_peaks(d, gidx_cols=('tmp_c', 'vpd_kpa', 'gust_ms', 'mstav_pct')):
    """Read all 4 lead files for date d; return dict of per-gridpoint peak arrays
    (max for tmp/vpd/gust, min for mstav). None if any lead file missing."""
    frames = []
    for l in LEADS:
        f = DIRS[l] / f"{d.date()}.parquet"
        if not f.exists():
            return None
        frames.append(pd.read_parquet(f, columns=list(gidx_cols)))
    n = min(len(x) for x in frames)
    out = {}
    for c in gidx_cols:
        stack = np.vstack([x[c].values[:n] for x in frames])
        out[c] = stack.min(axis=0) if c == 'mstav_pct' else stack.max(axis=0)
    return out


def stage1_features():
    outp = HERE / "peak_weather_features.parquet"
    if outp.exists():
        log("stage1: cached"); return pd.read_parquet(outp)
    tr = pd.read_parquet(TF / "tdis_train_daily_hrrr.parquet", columns=['h3_cell', 'date'])
    tr['date'] = pd.to_datetime(tr['date'])
    tr = tr.drop_duplicates()
    st = pd.read_parquet(TF / "data/static_features/tx_static_master.parquet",
                          columns=['h3_cell', 'lat', 'lon'])
    st = st[st.h3_cell.isin(set(tr.h3_cell))]
    sample = pd.read_parquet(sorted(DIRS[33].glob('*.parquet'))[0], columns=['lat', 'lon'])
    _, gi = cKDTree(sample[['lat', 'lon']].values).query(st[['lat', 'lon']].values)
    gmap = dict(zip(st.h3_cell, gi))
    tr['gidx'] = tr.h3_cell.map(gmap)
    tr = tr.dropna(subset=['gidx'])

    rows, t0 = [], time.time()
    for i, (d, grp) in enumerate(tr.groupby('date')):
        pk = read_day_peaks(d)
        if pk is None:
            continue
        g = grp.copy()
        idx = g.gidx.values.astype(int)
        g['hrrr_tmp'] = pk['tmp_c'][idx]
        g['hrrr_vpd'] = pk['vpd_kpa'][idx]
        g['hrrr_wind'] = pk['gust_ms'][idx]
        g['hrrr_mstav'] = np.clip(pk['mstav_pct'][idx] / 100.0, 0, 1)
        rows.append(g[['h3_cell', 'date', 'hrrr_tmp', 'hrrr_vpd', 'hrrr_wind', 'hrrr_mstav']])
        if (i + 1) % 500 == 0:
            log(f"  {i+1} dates ({(time.time()-t0)/60:.1f} min)")
    feat = pd.concat(rows, ignore_index=True)
    feat.to_parquet(outp, index=False)
    log(f"stage1 done: {len(feat):,} rows, coverage vs pairs "
        f"{len(feat)/len(tr)*100:.1f}%")
    return feat


def add_ix(df):
    df['bp_x_vpd'] = df['avg_burn_prob'] * df['hrrr_vpd']
    df['whp_x_vpd'] = df['whp'] * df['hrrr_vpd']
    df['bp_x_drought'] = df['avg_burn_prob'] * df['drought_score']
    return df


def stage2_train(peaks):
    flare = set(pd.read_parquet(NT / "flare_cells_v2.parquet")['h3_cell'])
    df = pd.read_parquet(TF / "tdis_train_daily_hrrr.parquet")
    df = df.drop(columns=['hrrr_tmp', 'hrrr_vpd', 'hrrr_wind'], errors='ignore')
    df['date'] = pd.to_datetime(df['date']); df['year'] = df.date.dt.year
    df = df[(~df.h3_cell.isin(flare)) & (df.date <= LAST)]
    df = df.merge(pd.read_parquet(NT / "powerline_dist_km.parquet"), on='h3_cell', how='left')
    df = df.merge(pd.read_parquet(DR / "drought_feature.parquet"), on=['h3_cell', 'date'], how='left')
    df = df.merge(peaks, on=['h3_cell', 'date'], how='inner')   # peak weather REPLACES old
    df = df.dropna(subset=STATIC).dropna(subset=['hrrr_vpd', 'hrrr_mstav', 'drought_score'])
    df = add_ix(df)
    tr, te = df[df.year <= 2021], df[df.year >= 2023]
    log(f"stage2: train {len(tr):,} / test {len(te):,} "
        f"(note: train starts 2018-07 -- 12Z archive floor; fewer rows than fix1 by design)")

    spw = (tr.label == 0).sum() / max((tr.label == 1).sum(), 1)
    mono = '(' + ','.join('1' if f in MONO_UP else '0' for f in FEATS) + ')'
    yt = te.label.astype(int).values
    model, seeds = None, []
    for seed in [42, 7, 2026]:
        m = xgb.XGBClassifier(n_estimators=1000, max_depth=9, min_child_weight=30,
                               learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=spw, eval_metric='aucpr',
                               monotone_constraints=mono,
                               tree_method='hist', device='cuda', n_jobs=-1, random_state=seed)
        m.fit(tr[FEATS], tr.label.astype(int), verbose=False)
        a = float(average_precision_score(yt, m.predict_proba(te[FEATS])[:, 1]))
        seeds.append(round(a, 4))
        log(f"  seed {seed}: test AUC-PR={a:.4f} (fix1 was {SERVED['test_aucpr']})")
        if seed == 42:
            model = m; m.save_model(str(HERE / "rev4_model.json"))
            imp = dict(zip(FEATS, m.feature_importances_.round(4).tolist()))
            log(f"  gust(peak) importance: {imp['hrrr_wind']:.4f} "
                f"(fix1's sustained-wind was 0.021); vpd(peak): {imp['hrrr_vpd']:.4f}")
    return model, seeds


def build_static_res8(flare):
    st = pd.read_parquet(TF / "data/static_features/tx_static_master.parquet")
    st = st[~st.h3_cell.isin(flare)]
    st = st.merge(pd.read_parquet(NT / "powerline_dist_km.parquet"), on='h3_cell', how='left')
    st = st.merge(pd.read_parquet(TF / "data/static_features/tx_county_fips.parquet"), on='h3_cell', how='left')
    st = st.dropna(subset=STATIC)
    sample = pd.read_parquet(sorted(DIRS[33].glob('*.parquet'))[0], columns=['lat', 'lon'])
    _, gi = cKDTree(sample[['lat', 'lon']].values).query(st[['lat', 'lon']].values)
    st['gidx'] = gi
    return st


def score_day_res8(model, st, weekly, d):
    pk = read_day_peaks(d)
    if pk is None:
        return None
    day = st[['h3_cell', 'lat', 'lon', 'gidx', 'fips'] + STATIC].copy()
    idx = day.gidx.values.astype(int)
    day['hrrr_tmp'] = pk['tmp_c'][idx]; day['hrrr_vpd'] = pk['vpd_kpa'][idx]
    day['hrrr_wind'] = pk['gust_ms'][idx]
    day['hrrr_mstav'] = np.clip(pk['mstav_pct'][idx] / 100.0, 0, 1)
    mo, dow = d.month, d.dayofweek
    day['sin_month'] = np.sin(2*np.pi*mo/12); day['cos_month'] = np.cos(2*np.pi*mo/12)
    day['sin_dow'] = np.sin(2*np.pi*dow/7); day['cos_dow'] = np.cos(2*np.pi*dow/7)
    day['is_weekend'] = int(dow >= 5); day['is_holiday'] = int((mo, d.day) in HOL)
    wk = weekly[(weekly.ValidStart <= d) & (weekly.ValidEnd >= d)][['fips', 'drought_score']]
    day = day.merge(wk, on='fips', how='left')
    day['drought_score'] = day['drought_score'].fillna(day['drought_score'].mean())
    day = add_ix(day).dropna(subset=['hrrr_vpd'])
    day['p'] = model.predict_proba(day[FEATS])[:, 1]
    day['rank'] = day['p'].rank(pct=True)
    return day


def stage3_res8_and_battery(model):
    flare = set(pd.read_parquet(NT / "flare_cells_v2.parquet")['h3_cell'])
    st = build_static_res8(flare)
    weekly = pd.read_parquet(TF / "data/static_features/drought_weekly.parquet")
    weekly['ValidStart'] = pd.to_datetime(weekly['ValidStart'])
    weekly['ValidEnd'] = pd.to_datetime(weekly['ValidEnd'])
    lab = pd.read_parquet(TF / "data/labels_fused/ignitions_daily_tx.parquet",
                           columns=['h3_cell', 'date', 'label'])
    lab['date'] = pd.to_datetime(lab['date']).dt.normalize()
    lab = lab[(~lab.h3_cell.isin(flare)) & (lab.label == 1)]

    log("\n=== NAMED-FIRE BATTERY (rev4 model, model's own ranks -- no floor) ===")
    battery = {}
    for name, ds, la0, la1, lo0, lo1 in EVENTS:
        d = pd.Timestamp(ds)
        day = score_day_res8(model, st, weekly, d)
        if day is None:
            log(f"{name}: missing archive day"); continue
        box = day[(day.lat.between(la0, la1)) & (day.lon.between(lo0, lo1))]
        fcells = set(lab[lab.date == d].h3_cell) & set(box.h3_cell)
        fc = box[box.h3_cell.isin(fcells)]
        res = dict(n_fire_cells=len(fc),
                   mean_rank=round(float(fc['rank'].mean()), 3) if len(fc) else None,
                   below_tau_pct=round(float((fc['rank'] < TAU).mean())*100, 1) if len(fc) else None,
                   box_top25_pct=round(float((box['rank'] >= 0.75).mean())*100, 1))
        battery[name] = res
        log(f"  {name} ({ds}): fire cells {res['n_fire_cells']}, mean model rank "
            f"{res['mean_rank']}, below-tau (would show Low) {res['below_tau_pct']}%  "
            f"| box >=top-25%: {res['box_top25_pct']}%")

    log("\n=== 2026 res-8 missed@tau (held-out) ===")
    fire_by_date = lab[(lab.date >= '2026-01-01') & (lab.date <= LAST)].groupby('date')['h3_cell'].apply(set).to_dict()
    ranks, t0, nd = [], time.time(), 0
    for d in pd.date_range('2026-01-01', str(LAST.date()), freq='D'):
        day = score_day_res8(model, st, weekly, d)
        if day is None: continue
        cr = dict(zip(day.h3_cell.values, day['rank'].values))
        for c in fire_by_date.get(d, set()):
            r = cr.get(c)
            if r is not None: ranks.append(r)
        nd += 1
        if nd % 50 == 0: log(f"  {nd} days ({(time.time()-t0)/60:.1f} min)")
    fr = np.array(ranks)
    missed = float((fr < TAU).mean())
    log(f"missed@tau: {missed*100:.2f}% (fix1: {SERVED['missed_res8']}%)  "
        f"median fire rank: {np.median(fr):.3f}  n={len(fr):,}")
    pd.DataFrame(dict(rank=fr)).to_parquet(HERE / "rev4_fire_ranks_2026.parquet", index=False)
    return battery, missed, fr


def stage4_realpop_res5(model):
    flare = set(pd.read_parquet(NT / "flare_cells_v2.parquet")['h3_cell'])
    st = build_static_res8(flare)
    st['h3_5'] = [h3.cell_to_parent(c, 5) for c in st.h3_cell.values]
    weekly = pd.read_parquet(TF / "data/static_features/drought_weekly.parquet")
    weekly['ValidStart'] = pd.to_datetime(weekly['ValidStart'])
    weekly['ValidEnd'] = pd.to_datetime(weekly['ValidEnd'])
    lab = pd.read_parquet(TF / "data/labels_fused/ignitions_daily_tx.parquet",
                           columns=['h3_cell', 'date', 'label'])
    lab['date'] = pd.to_datetime(lab['date']).dt.normalize()
    lab = lab[(~lab.h3_cell.isin(flare)) & (lab.label == 1)]
    lab['h3_5'] = [h3.cell_to_parent(c, 5) for c in lab.h3_cell.values]

    # res-5 static means once
    scols = {f: 'mean' for f in STATIC}
    s5_static = st.groupby('h3_5').agg({**scols, 'lat': 'mean', 'lon': 'mean', 'fips': 'first'}).reset_index()
    sample = pd.read_parquet(sorted(DIRS[33].glob('*.parquet'))[0], columns=['lat', 'lon'])
    _, gi = cKDTree(sample[['lat', 'lon']].values).query(s5_static[['lat', 'lon']].values)
    s5_static['gidx'] = gi

    rows, t0 = [], time.time()
    for d in pd.date_range('2024-01-01', str(LAST.date()), freq='D'):
        pk = read_day_peaks(d)
        if pk is None: continue
        day = s5_static.copy()
        idx = day.gidx.values.astype(int)
        day['hrrr_tmp'] = pk['tmp_c'][idx]; day['hrrr_vpd'] = pk['vpd_kpa'][idx]
        day['hrrr_wind'] = pk['gust_ms'][idx]
        day['hrrr_mstav'] = np.clip(pk['mstav_pct'][idx] / 100.0, 0, 1)
        mo, dow = d.month, d.dayofweek
        day['sin_month'] = np.sin(2*np.pi*mo/12); day['cos_month'] = np.cos(2*np.pi*mo/12)
        day['sin_dow'] = np.sin(2*np.pi*dow/7); day['cos_dow'] = np.cos(2*np.pi*dow/7)
        day['is_weekend'] = int(dow >= 5); day['is_holiday'] = int((mo, d.day) in HOL)
        wk = weekly[(weekly.ValidStart <= d) & (weekly.ValidEnd >= d)][['fips', 'drought_score']]
        day = day.merge(wk, on='fips', how='left')
        day['drought_score'] = day['drought_score'].fillna(day['drought_score'].mean())
        day = add_ix(day).dropna(subset=['hrrr_vpd'])
        day['p'] = model.predict_proba(day[FEATS])[:, 1]
        day['date'] = d
        rows.append(day[['h3_5', 'date', 'p']])
    sc = pd.concat(rows, ignore_index=True)
    sc.to_parquet(HERE / "rev4_realpop_scores.parquet", index=False)
    pos = set(zip(lab[(lab.date >= sc.date.min()) & (lab.date <= sc.date.max())].h3_5,
                  lab[(lab.date >= sc.date.min()) & (lab.date <= sc.date.max())].date))
    sc['y'] = [1 if k in pos else 0 for k in zip(sc.h3_5, sc.date)]
    y, p = sc.y.values, sc.p.values
    aucpr = float(average_precision_score(y, p)); auroc = float(roc_auc_score(y, p))
    lift = aucpr / y.mean()
    log(f"\nstage4 gate3: AUC-PR={aucpr:.4f} AUROC={auroc:.4f} lift={lift:.2f} "
        f"(served {SERVED['lift']})")
    # gate 4
    ex = pd.read_parquet(TF / "Training_rev3/gate4_ensemble/extreme_keys.parquet")
    sc['rank'] = sc.groupby('date')['p'].rank(pct=True)
    exset = set(zip(ex.h3_5, ex.date))
    m = np.array([k in exset for k in zip(sc.h3_5, sc.date)])
    g4 = float(sc[m]['rank'].mean())
    log(f"stage4 gate4 danger rank: {g4:.4f} (fix1 {SERVED['g4_rank']})")
    return dict(aucpr=round(aucpr, 4), auroc=round(auroc, 4), lift=round(lift, 2),
                gate4_rank=round(g4, 4))


if __name__ == '__main__':
    peaks = stage1_features()
    model, seeds = stage2_train(peaks)
    battery, missed, fr = stage3_res8_and_battery(model)
    g34 = stage4_realpop_res5(model)
    json.dump(dict(test_seeds=seeds, battery=battery,
                   missed_res8_2026=round(missed*100, 2),
                   median_fire_rank=round(float(np.median(fr)), 4),
                   gate3_gate4=g34, fix1_baseline=SERVED),
              open(HERE / "rev4_results.json", 'w'), indent=2)
    log("\nDONE -> rev4_results.json")
