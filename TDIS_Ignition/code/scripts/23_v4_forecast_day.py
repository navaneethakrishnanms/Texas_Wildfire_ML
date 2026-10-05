"""
TDIS Forecast v4 -- live daily forecast with the fix1 (false-negative-
hardened) model: hazard x weather interactions + monotone hazard, tier
scheme with the verified <=10%-missed Low boundary, and the danger floor.

Differences from script 13 (which stays untouched, serving v3):
  - model: Training_rev3/fix1_hazard_weather/fix1_model.json + its
    calibrator (2026-holdout ECE 0.0012)
  - computes the 3 interaction features + drought at res-8
  - outputs TIERS from the day's statewide res-8 percentile rank:
      Low < 0.4623 (verified: <=10% of fires occur there)
      Moderate < 0.75 | High < 0.90 | Very High >= 0.90
  - DANGER FLOOR: cells in the top-25%-hazard x top-10%-fire-weather
    (today's HWP) combo can never display below High
  - exports BOTH: res-5 JSON for the base map AND a res-8 detail JSON
    of every cell above the Low boundary (for true native-resolution
    zoom -- the Low majority renders from res-5, detail appears where
    it matters)

Run: python 23_v4_forecast_day.py 2026-08-21 24
"""
import sys, json, warnings
import numpy as np, pandas as pd, h3, xgboost as xgb, joblib
from pathlib import Path
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')

TF = Path(__file__).resolve().parent.parent
R3 = TF / "Training_rev3"
DASH = TF / "dashboard"
TX = dict(lat_min=25.75, lat_max=36.65, lon_min=-106.70, lon_max=-93.40)
TAU = 0.4623            # verified 90%-recall Low boundary (95% option: 0.311)
HAZ_Q, FWI_Q = 0.75, 0.90

import fwi_config
HOL = {(1, 1), (7, 4), (6, 19), (11, 11), (12, 25), (10, 31)}
STATIC = ['road_dist_km', 'ecoregion_id', 'elevation_m', 'slope_deg', 'aspect_deg',
          'avg_burn_prob', 'whp', 'flep4', 'cfl', 'cbd', 'cbh', 'powerline_dist_km']


def log(m): print(m, flush=True)


def vpd_from(t, td):
    es = 0.6108*np.exp(17.27*t/(t+237.3)); ea = 0.6108*np.exp(17.27*td/(td+237.3))
    return np.clip(es-ea, 0, None)


def main():
    TARGET = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-08-21")
    LEAD = int(sys.argv[2]) if len(sys.argv) > 2 else 24

    from herbie import Herbie
    init = TARGET - pd.Timedelta(hours=LEAD)
    model_nm, prod = ("hrrr", "sfc") if LEAD <= 48 else ("gfs", "pgrb2.0p25")
    log(f"Pulling {model_nm.upper()} F{LEAD}: target {TARGET.date()}")
    hh = Herbie(init.strftime("%Y-%m-%d %H:%M"), model=model_nm, product=prod, fxx=LEAD, verbose=False)
    if hh.grib is None:
        log("forecast not found"); sys.exit(1)
    dsl = hh.xarray(":(TMP|DPT):2 m above ground|:WIND:10 m above ground|:GUST:surface|:MSTAV:", remove_grib=True)
    if not isinstance(dsl, list): dsl = [dsl]
    mg = {}
    for ds in dsl:
        la = ds.latitude.values.ravel(); lo = ds.longitude.values.ravel()
        lo = np.where(lo > 180, lo-360, lo)
        msk = (la >= TX['lat_min']) & (la <= TX['lat_max']) & (lo >= TX['lon_min']) & (lo <= TX['lon_max'])
        mg.setdefault('lat', la[msk]); mg.setdefault('lon', lo[msk])
        for nm, da in ds.data_vars.items():
            mg[nm] = da.values.ravel()[msk]
    g = pd.DataFrame(mg).rename(columns={'t2m': 'tmp', 'd2m': 'dpt', 'max_10si': 'wind',
                                          'si10': 'wind', 'gust': 'gust', 'mstav': 'soilm'})
    g['tmp'] -= 273.15; g['dpt'] -= 273.15
    g['vpd'] = vpd_from(g['tmp'].values, g['dpt'].values)
    soilmv = np.clip(g['soilm'].values/100.0, 0, 1) if 'soilm' in g.columns else None
    gustv = g['gust'].values if 'gust' in g.columns else None
    G_eff = gustv if gustv is not None else 1.5*g['wind'].values
    g['fwiN'] = fwi_config.hwp_hrrr(g['vpd'].values, G_eff, soilmv, variant='noaa') \
        if soilmv is not None else fwi_config.fwi_composite_hrrr(g['vpd'].values, g['wind'].values, gust=gustv)
    log(f"  {len(g):,} grid points")

    log("Loading fix1 model + static features...")
    mdl = xgb.XGBClassifier()
    mdl.load_model(str(R3 / "fix1_hazard_weather/fix1_model.json"))
    FEATS = mdl.get_booster().feature_names
    iso = joblib.load(R3 / "fix1_hazard_weather/fix1_isotonic_calibrator.joblib")

    st = pd.read_parquet(TF/"data"/"static_features"/"tx_static_master.parquet")
    st = st.merge(pd.read_parquet(TF/"New_Training817_moredata"/"powerline_dist_km.parquet"),
                  on='h3_cell', how='left')
    st = st.merge(pd.read_parquet(TF/"data"/"static_features"/"tx_county_fips.parquet"),
                  on='h3_cell', how='left')
    st = st.dropna(subset=STATIC)

    tree = cKDTree(np.c_[g['lat'].values, g['lon'].values])
    _, idx = tree.query(np.c_[st['lat'].values, st['lon'].values])
    st['hrrr_tmp'] = g['tmp'].values[idx]; st['hrrr_vpd'] = g['vpd'].values[idx]
    st['hrrr_wind'] = g['wind'].values[idx]
    st['hrrr_mstav'] = np.clip(soilmv[idx], 0, 1) if soilmv is not None else 0.5
    st['fwiN_today'] = g['fwiN'].values[idx]

    import sys as _s; _s.path.insert(0, str(TF/"scripts"))
    from fetch_live_drought import get_drought_score
    dr = get_drought_score()
    st = st.merge(dr, on='fips', how='left')
    st['drought_score'] = st['drought_score'].fillna(st['drought_score'].mean())

    mo, dow = TARGET.month, TARGET.dayofweek
    st['sin_month'] = np.sin(2*np.pi*mo/12); st['cos_month'] = np.cos(2*np.pi*mo/12)
    st['sin_dow'] = np.sin(2*np.pi*dow/7); st['cos_dow'] = np.cos(2*np.pi*dow/7)
    st['is_weekend'] = int(dow >= 5); st['is_holiday'] = int((mo, TARGET.day) in HOL)
    st['bp_x_vpd'] = st['avg_burn_prob']*st['hrrr_vpd']
    st['whp_x_vpd'] = st['whp']*st['hrrr_vpd']
    st['bp_x_drought'] = st['avg_burn_prob']*st['drought_score']

    log("Scoring fix1 on 1.7M res-8 cells...")
    st['p'] = mdl.predict_proba(st[FEATS])[:, 1]
    st['cal'] = iso.predict(st['p'].values)
    st['rank'] = st['p'].rank(pct=True)

    # danger floor: extreme-condition cells never below High (rank 0.75)
    haz_thr = st['avg_burn_prob'].quantile(HAZ_Q)
    fwi_thr = st['fwiN_today'].quantile(FWI_Q)
    extreme = (st['avg_burn_prob'] >= haz_thr) & (st['fwiN_today'] >= fwi_thr)
    st['disp_rank'] = np.where(extreme, np.maximum(st['rank'], 0.75), st['rank'])
    n_floor = int((extreme & (st['rank'] < 0.75)).sum())
    log(f"  danger floor raised {n_floor:,} cells ({n_floor/len(st)*100:.2f}%)")

    def tier(r):
        return np.select([r < TAU, r < 0.75, r < 0.90], [0, 1, 2], 3)  # 0=Low..3=VeryHigh
    st['tier'] = tier(st['disp_rank'].values)
    log(f"  tiers: Low {(st.tier==0).mean()*100:.1f}% | Mod {(st.tier==1).mean()*100:.1f}% | "
        f"High {(st.tier==2).mean()*100:.1f}% | VHigh {(st.tier==3).mean()*100:.1f}%")

    # res-5 base layer (all cells)
    st['h3_5'] = [h3.cell_to_parent(c, 5) for c in st['h3_cell'].values]
    agg = st.groupby('h3_5').agg(rank=('disp_rank', 'mean'), cal=('cal', 'mean'),
                                  fwiN=('fwiN_today', 'mean'),
                                  tier_max=('tier', 'max')).reset_index()
    agg['tier'] = tier(agg['rank'].values)
    # a res-5 cell containing ANY floored/VH res-8 pocket shows a "peak" marker tier
    out5 = {'target': str(TARGET.date()), 'lead_h': LEAD, 'tau': TAU,
            'source': f'fix1 v4 {model_nm.upper()} F{LEAD}',
            'tier': {r.h3_5: int(r.tier) for r in agg.itertuples()},
            'peak': {r.h3_5: int(r.tier_max) for r in agg.itertuples()},
            'cal': {r.h3_5: round(float(r.cal), 4) for r in agg.itertuples()},
            'fwiN': {r.h3_5: round(float(r.fwiN), 4) for r in agg.itertuples()}}
    p5 = DASH / f"v4_forecast_{TARGET.date()}.json"
    json.dump(out5, open(p5, 'w'), separators=(',', ':'))
    log(f"Saved {p5.name} ({len(agg):,} res-5 cells)")

    # res-8 DETAIL layer for zoom: High + Very High tiers only (the pockets
    # that matter at street level; Low/Moderate render fine from res-5).
    # Compact format: one array of cell ids per tier.
    detail = st[st['tier'] >= 2]
    out8 = {'target': str(TARGET.date()), 'lead_h': LEAD,
            't2': detail.loc[detail['tier'] == 2, 'h3_cell'].tolist(),
            't3': detail.loc[detail['tier'] == 3, 'h3_cell'].tolist()}
    p8 = DASH / f"v4_detail8_{TARGET.date()}_lead{LEAD}h.json"
    json.dump(out8, open(p8, 'w'), separators=(',', ':'))
    log(f"Saved {p8.name} ({len(detail):,} res-8 High/VHigh cells, "
        f"{p8.stat().st_size/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
