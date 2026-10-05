"""
New_Training817_moredata — PROMOTION of the tuned depth-9 model.

1. Re-score the tuned model on the real full population (same replay as
   scripts 26/31), this time SAVING the per-cell-day scores.
2. Fit its isotonic calibrator with the same discipline as script 27:
   fit on 2024-25, evaluate on a 2026-only holdout the calibrator never saw.
3. Report holdout ECE/Brier so the promotion is evidence-backed.

The actual file swap (models/ + dashboard) is done by the caller after
this script succeeds -- kept separate so a failure here leaves the served
pair untouched.

Output: models/tuned_d9_historical_res5.parquet
        models/tuned_d9_isotonic_calibrator.joblib
        New_Training817_moredata/promote_d9_report.json
"""
import json, time, warnings
import numpy as np, pandas as pd, xgboost as xgb, h3, joblib
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
TF = HERE.parent
HRRR_DIR = TF / "data" / "weather_hrrr_forecast" / "hrrr_24h"
START, END = pd.Timestamp('2024-01-01'), pd.Timestamp('2026-07-29')
FIT_END = pd.Timestamp('2025-12-31')          # calibrator fit window; 2026 = holdout

STATIC = ['road_dist_km', 'ecoregion_id', 'elevation_m', 'slope_deg', 'aspect_deg',
          'avg_burn_prob', 'whp', 'flep4', 'cfl', 'cbd', 'cbh']
HOLIDAYS = {(1, 1), (7, 4), (6, 19), (11, 11), (12, 25), (10, 31)}


def log(m): print(m, flush=True)


def ece_brier(y, p):
    brier = float(np.mean((p - y) ** 2))
    bidx = np.clip(np.digitize(p, np.linspace(0, 1, 11)) - 1, 0, 9)
    ece = 0.0
    for b in range(10):
        mask = bidx == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(p)) * abs(p[mask].mean() - y[mask].mean())
    return float(ece), brier


def run():
    m8 = pd.read_parquet(TF / "data/static_features/tx_static_master.parquet",
                          columns=['h3_cell', 'lat', 'lon'] + STATIC)
    m8['h3_5'] = [h3.cell_to_parent(c, 5) for c in m8['h3_cell'].values]
    s5 = m8.groupby('h3_5').agg({**{f: 'mean' for f in STATIC}, 'lat': 'mean', 'lon': 'mean'}).reset_index()
    sample = pd.read_parquet(sorted(HRRR_DIR.glob('2024-*.parquet'))[0], columns=['lat', 'lon'])
    tree = cKDTree(sample[['lat', 'lon']].values)
    _, idx = tree.query(s5[['lat', 'lon']].values)
    s5['glat'] = sample['lat'].values[idx].round(3)
    s5['glon'] = sample['lon'].values[idx].round(3)

    model = xgb.XGBClassifier()
    model.load_model(str(HERE / "tdis_forecast_hrrr_tuned_d9.json"))
    FEATS = model.get_booster().feature_names

    log("Scoring tuned d9 on real population (saving scores)...")
    rows = []
    t0 = time.time()
    for d in pd.date_range(START, END, freq='D'):
        f = HRRR_DIR / f"{d.date()}.parquet"
        if not f.exists():
            continue
        hf = pd.read_parquet(f)
        if not {'tmp_c', 'vpd_kpa', 'wind_ms'} <= set(hf.columns):
            continue
        hf['glat'] = hf['lat'].round(3); hf['glon'] = hf['lon'].round(3)
        lut = hf.groupby(['glat', 'glon']).agg(
            hrrr_tmp=('tmp_c', 'mean'), hrrr_vpd=('vpd_kpa', 'mean'), hrrr_wind=('wind_ms', 'mean')).reset_index()
        day = s5.merge(lut, on=['glat', 'glon'], how='left')
        mo, dow = d.month, d.dayofweek
        day['sin_month'] = np.sin(2 * np.pi * mo / 12); day['cos_month'] = np.cos(2 * np.pi * mo / 12)
        day['sin_dow'] = np.sin(2 * np.pi * dow / 7); day['cos_dow'] = np.cos(2 * np.pi * dow / 7)
        day['is_weekend'] = int(dow >= 5); day['is_holiday'] = int((mo, d.day) in HOLIDAYS)
        day = day.dropna(subset=['hrrr_vpd'])
        day['p'] = model.predict_proba(day[FEATS])[:, 1]
        day['date'] = d
        rows.append(day[['h3_5', 'date', 'p']])
    scores = pd.concat(rows, ignore_index=True)
    log(f"  {len(scores):,} cell-days ({(time.time()-t0)/60:.1f} min)")
    scores.to_parquet(TF / "models" / "tuned_d9_historical_res5.parquet", index=False)

    log("Ground truth (fused, flare-excluded)...")
    flare = pd.read_parquet(TF / "data/labels_fused/flare_cells.parquet")
    fc = set(flare['h3_cell'] if 'h3_cell' in flare.columns else flare.iloc[:, 0])
    lab = pd.read_parquet(TF / "data/labels_fused/ignitions_daily_tx.parquet",
                           columns=['h3_cell', 'date', 'label'])
    lab['date'] = pd.to_datetime(lab['date']).dt.normalize()
    lab = lab[(lab['date'] >= scores['date'].min()) & (lab['date'] <= scores['date'].max())]
    lab = lab[(~lab['h3_cell'].isin(fc)) & (lab['label'] == 1)]
    lab['h3_5'] = [h3.cell_to_parent(c, 5) for c in lab['h3_cell'].values]
    pos = set(zip(lab['h3_5'], lab['date']))
    scores['y'] = [1 if (h, d) in pos else 0 for h, d in zip(scores['h3_5'], scores['date'])]

    fit = scores[scores['date'] <= FIT_END]
    hold = scores[scores['date'] > FIT_END]
    log(f"Calibrator: fit on {len(fit):,} rows (2024-25), holdout {len(hold):,} rows (2026)")
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(fit['p'].values, fit['y'].values)
    joblib.dump(iso, TF / "models" / "tuned_d9_isotonic_calibrator.joblib")

    yh, ph = hold['y'].values, hold['p'].values
    pch = iso.predict(ph)
    e0, b0 = ece_brier(yh, ph)
    e1, b1 = ece_brier(yh, pch)
    log(f"2026 holdout: raw ECE={e0:.4f} Brier={b0:.5f}  ->  calibrated ECE={e1:.4f} Brier={b1:.5f}")
    log(f"holdout base rate={yh.mean():.4f}  mean raw={ph.mean():.4f}  mean cal={pch.mean():.4f}")

    json.dump(dict(model='tdis_forecast_hrrr_tuned_d9', n_scored=len(scores),
                   fit_rows=len(fit), holdout_rows=len(hold),
                   holdout=dict(base_rate=float(yh.mean()),
                                raw=dict(ece=round(e0, 4), brier=round(b0, 5), mean_pred=float(ph.mean())),
                                calibrated=dict(ece=round(e1, 4), brier=round(b1, 5), mean_pred=float(pch.mean())))),
              open(HERE / "promote_d9_report.json", 'w'), indent=2)
    log("Saved -> models/tuned_d9_isotonic_calibrator.joblib, promote_d9_report.json")


if __name__ == '__main__':
    run()
