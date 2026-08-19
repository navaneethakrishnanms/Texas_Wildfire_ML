"""
TDIS Forecast — Step 23: score the CEILING model (same-day OBSERVED weather) day-by-day
across the dashboard's historical range (2024-2026, res-5), so "Ignition" can show a real
model output for history instead of the current susceptibility x FWI heuristic proxy.

This is genuinely new inference, not a rebuild of anything else -- the ceiling model
(models/tdis_forecast_baseline_ceiling_filtered.json) was already trained on same-day
observed gridMET weather (script 22) but was never scored day-by-day across history; it
was built purely as a research upper-bound control ("never served, diagnostic only").

Exactly replicates the training feature pipeline (03_build_dataset.py, 22) so the model
sees the same feature distribution it was trained on:
  STATIC (11)  : mean-aggregated res-8 -> res-5 (same method as 08_build_dashboard_data.py)
  TEMPORAL (6) : sin/cos month + dow, is_weekend, is_holiday, computed from date
  WEATHER (8)  : erc/fm100/vpd/vs already cached (fwi_components_res5.parquet);
                 rmax/rmin/tmmx/pr pulled fresh from gridmet_tx and res-5 aggregated
                 the same way script 09 does for the other 4 variables.

Output: models/ceiling_historical_res5.parquet (h3_5, date, p_ceiling)
        + merges a 'ignC' per-day array into dashboard/tdis_dashboard_data.json
        (kept SEPARATE from 'ign' -- does not overwrite the existing heuristic layer).
"""
import json, warnings
import numpy as np, pandas as pd, xgboost as xgb, h3
from pathlib import Path
warnings.filterwarnings('ignore')

TF = Path(__file__).resolve().parent.parent
GRIDMET = TF.parent / "gridmet_tx"
YEARS = [2024, 2025, 2026]

STATIC = ['road_dist_km', 'ecoregion_id', 'elevation_m', 'slope_deg', 'aspect_deg',
          'avg_burn_prob', 'whp', 'flep4', 'cfl', 'cbd', 'cbh']
TEMPORAL = ['sin_month', 'cos_month', 'sin_dow', 'cos_dow', 'is_weekend', 'is_holiday']
WEATHER = ['erc', 'fm100', 'vpd', 'vs', 'rmax', 'rmin', 'tmmx', 'pr']
FEATS = STATIC + TEMPORAL + WEATHER
HOLIDAYS = {(1, 1), (7, 4), (6, 19), (11, 11), (12, 25), (10, 31)}


def log(m):
    print(m, flush=True)


def build_static_res5():
    log("Aggregating static features res-8 -> res-5 (mean, matching script 08)...")
    m = pd.read_parquet(TF / "data" / "static_features" / "tx_static_master.parquet",
                         columns=['h3_cell'] + STATIC)
    m['h3_5'] = [h3.cell_to_parent(c, 5) for c in m['h3_cell'].values]
    s5 = m.groupby('h3_5')[STATIC].mean().reset_index()
    log(f"  static res-5 cells: {len(s5):,}")
    return s5


def build_weather_res5():
    log("Loading cached erc/vpd/vs/fm100 (fwi_components_res5.parquet)...")
    comp = pd.read_parquet(TF / "data" / "fwi_components_res5.parquet")
    comp['date'] = pd.to_datetime(comp['date']).dt.normalize()

    log("Pulling rmax/rmin/tmmx/pr from gridmet_tx, aggregating res-8 -> res-5 (mean)...")
    extra = []
    for yr in YEARS:
        parts = []
        for var in ['rmax', 'rmin', 'tmmx', 'pr']:
            d = pd.read_parquet(GRIDMET / f"{var}_{yr}_tx_cells.parquet")
            d['date'] = pd.to_datetime(d['date_utc']).dt.normalize()
            parts.append(d.set_index(['h3_cell', 'date'])[var])
        wx = pd.concat(parts, axis=1).reset_index()
        wx['h3_5'] = [h3.cell_to_parent(c, 5) for c in wx['h3_cell'].values]
        c5 = wx.groupby(['h3_5', 'date']).agg(
            rmax=('rmax', 'mean'), rmin=('rmin', 'mean'),
            tmmx=('tmmx', 'mean'), pr=('pr', 'mean')).reset_index()
        # SANITY_CHECK F6: gridMET tmmx fill-value artifact (~-53C, impossible in TX)
        c5.loc[c5['tmmx'] < -30, 'tmmx'] = np.nan
        extra.append(c5)
        log(f"  {yr}: {c5['date'].nunique()} days, {c5['h3_5'].nunique():,} res-5 cells")

    extra = pd.concat(extra, ignore_index=True)
    wx5 = comp.merge(extra, on=['h3_5', 'date'], how='inner')
    log(f"  merged weather (8 vars): {len(wx5):,} cell-days")
    return wx5


def add_temporal(df):
    dow = df['date'].dt.dayofweek
    df['sin_dow'] = np.sin(2 * np.pi * dow / 7)
    df['cos_dow'] = np.cos(2 * np.pi * dow / 7)
    df['is_weekend'] = (dow >= 5).astype(int)
    mo = df['date'].dt.month
    df['sin_month'] = np.sin(2 * np.pi * mo / 12)
    df['cos_month'] = np.cos(2 * np.pi * mo / 12)
    df['is_holiday'] = [(d.month, d.day) in HOLIDAYS for d in df['date']]
    return df


def run():
    static5 = build_static_res5()
    wx5 = build_weather_res5()

    log("Assembling full feature table...")
    df = wx5.merge(static5, on='h3_5', how='inner')
    df = add_temporal(df)
    n_before = len(df)
    df = df.dropna(subset=STATIC + WEATHER).reset_index(drop=True)
    log(f"  {len(df):,} cell-days scoreable ({n_before - len(df):,} dropped for missing static/weather)")

    log("Loading ceiling model and scoring...")
    model = xgb.XGBClassifier()
    model.load_model(str(TF / "models" / "tdis_forecast_baseline_ceiling_filtered.json"))
    df['p_ceiling'] = model.predict_proba(df[FEATS])[:, 1]
    log(f"  scored. p_ceiling: mean={df['p_ceiling'].mean():.4f} "
        f"p95={df['p_ceiling'].quantile(0.95):.4f} max={df['p_ceiling'].max():.4f}")

    out_path = TF / "models" / "ceiling_historical_res5.parquet"
    df[['h3_5', 'date', 'p_ceiling']].to_parquet(out_path, index=False)
    log(f"Saved -> {out_path}")

    # Merge into dashboard JSON as a NEW 'ignC' array, alongside the existing 'ign'/'fwi'
    log("Merging 'ignC' (ceiling ignition) into dashboard JSON...")
    dash_path = TF / "dashboard" / "tdis_dashboard_data.json"
    dash = json.load(open(dash_path))
    dates = [pd.Timestamp(d) for d in dash['dates']] if 'dates' in dash else sorted(df['date'].unique())
    date_idx = {d: i for i, d in enumerate(dates)}
    df['di'] = df['date'].map(date_idx)
    by_cell = {}
    for cid, g in df.dropna(subset=['di']).groupby('h3_5'):
        arr = [None] * len(dates)
        for di, val in zip(g['di'].values, g['p_ceiling'].values):
            arr[int(di)] = round(float(val), 4)
        by_cell[cid] = arr
    n = 0
    for c in dash['cells']:
        if c['id'] in by_cell:
            c['ignC'] = by_cell[c['id']]
            n += 1
    dash['meta']['ceiling_ignition_source'] = ('ceiling model (same-day observed gridMET weather), '
                                                 'scored day-by-day 2024-2026, NOT the served operational model')
    json.dump(dash, open(dash_path, 'w'), separators=(',', ':'))
    sz = dash_path.stat().st_size / 1e6
    log(f"Saved {dash_path.name} ({sz:.1f} MB). ignC on {n:,} cells x {len(dates)} days.")


if __name__ == '__main__':
    run()
