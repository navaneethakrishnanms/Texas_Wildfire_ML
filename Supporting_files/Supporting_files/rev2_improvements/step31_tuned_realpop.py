"""
rev2_improvements — Step 31: score the TUNED model (step30_tuned_seed42,
depth9/mcw30/lr.02/1000 trees — seed-robust +0.0049 test AUC-PR) on the
REAL full population, same replay as script 26, and compare to the served
model's real-population result (AUC-PR=0.0851, AUROC=0.7838, lift=4.43x).

This answers the promotion question: the tuned config wins on the balanced
test; does it also win where it matters — every cell x every day?

Output: rev2_improvements/step31_tuned_realpop.json
"""
import json, time, warnings
import numpy as np, pandas as pd, xgboost as xgb, h3
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')

TF = Path(__file__).resolve().parent.parent
HRRR_DIR = TF / "data" / "weather_hrrr_forecast" / "hrrr_24h"
START, END = pd.Timestamp('2024-01-01'), pd.Timestamp('2026-07-29')

STATIC = ['road_dist_km', 'ecoregion_id', 'elevation_m', 'slope_deg', 'aspect_deg',
          'avg_burn_prob', 'whp', 'flep4', 'cfl', 'cbd', 'cbh']
HOLIDAYS = {(1, 1), (7, 4), (6, 19), (11, 11), (12, 25), (10, 31)}
SERVED_REALPOP = dict(aucpr=0.0851, auroc=0.7838, lift=4.43)


def log(m): print(m, flush=True)


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
    model.load_model(str(Path(__file__).resolve().parent / "step30_tuned_seed42.json"))
    FEATS = model.get_booster().feature_names

    log("Scoring tuned model on real population...")
    dates = pd.date_range(START, END, freq='D')
    rows, skipped = [], 0
    t0 = time.time()
    for i, d in enumerate(dates):
        f = HRRR_DIR / f"{d.date()}.parquet"
        if not f.exists():
            skipped += 1; continue
        hf = pd.read_parquet(f)
        if not {'tmp_c', 'vpd_kpa', 'wind_ms'} <= set(hf.columns):
            skipped += 1; continue
        hf['glat'] = hf['lat'].round(3); hf['glon'] = hf['lon'].round(3)
        lut = hf.groupby(['glat', 'glon']).agg(
            hrrr_tmp=('tmp_c', 'mean'), hrrr_vpd=('vpd_kpa', 'mean'), hrrr_wind=('wind_ms', 'mean')).reset_index()
        day = s5.merge(lut, on=['glat', 'glon'], how='left')
        mo, dow = d.month, d.dayofweek
        day['sin_month'] = np.sin(2 * np.pi * mo / 12); day['cos_month'] = np.cos(2 * np.pi * mo / 12)
        day['sin_dow'] = np.sin(2 * np.pi * dow / 7); day['cos_dow'] = np.cos(2 * np.pi * dow / 7)
        day['is_weekend'] = int(dow >= 5)
        day['is_holiday'] = int((mo, d.day) in HOLIDAYS)
        day = day.dropna(subset=['hrrr_vpd'])
        day['p'] = model.predict_proba(day[FEATS])[:, 1]
        day['date'] = d
        rows.append(day[['h3_5', 'date', 'p']])
        if (i + 1) % 200 == 0:
            log(f"  {i+1}/{len(dates)} days ({(time.time()-t0)/60:.1f} min)")
    scores = pd.concat(rows, ignore_index=True)
    log(f"scored {len(scores):,} cell-days ({skipped} skipped)")

    flare = pd.read_parquet(TF / "data/labels_fused/flare_cells.parquet")
    flare_cells = set(flare['h3_cell'] if 'h3_cell' in flare.columns else flare.iloc[:, 0])
    lab = pd.read_parquet(TF / "data/labels_fused/ignitions_daily_tx.parquet",
                           columns=['h3_cell', 'date', 'label'])
    lab['date'] = pd.to_datetime(lab['date']).dt.normalize()
    lab = lab[(lab['date'] >= scores['date'].min()) & (lab['date'] <= scores['date'].max())]
    lab = lab[(~lab['h3_cell'].isin(flare_cells)) & (lab['label'] == 1)]
    lab['h3_5'] = [h3.cell_to_parent(c, 5) for c in lab['h3_cell'].values]
    pos = set(zip(lab['h3_5'], lab['date']))
    scores['y'] = [1 if (h, d) in pos else 0 for h, d in zip(scores['h3_5'], scores['date'])]
    y, p = scores['y'].values, scores['p'].values
    base = y.mean()
    aucpr = float(average_precision_score(y, p))
    auroc = float(roc_auc_score(y, p))
    lift = aucpr / base
    log(f"\nTUNED, real full population: AUC-PR={aucpr:.4f} AUROC={auroc:.4f} lift={lift:.2f}x")
    log(f"SERVED (script 26)         : AUC-PR={SERVED_REALPOP['aucpr']} AUROC={SERVED_REALPOP['auroc']} lift={SERVED_REALPOP['lift']}x")
    log(f"delta: AUC-PR {aucpr-SERVED_REALPOP['aucpr']:+.4f}  AUROC {auroc-SERVED_REALPOP['auroc']:+.4f}")

    json.dump(dict(model='step30_tuned_seed42 (depth9/mcw30/lr.02/1000t)',
                   n_scored=len(scores), base_rate=float(base),
                   aucpr=round(aucpr, 4), auroc=round(auroc, 4), lift=round(lift, 2),
                   served=SERVED_REALPOP,
                   delta=dict(aucpr=round(aucpr - SERVED_REALPOP['aucpr'], 4),
                              auroc=round(auroc - SERVED_REALPOP['auroc'], 4))),
              open(Path(__file__).resolve().parent / "step31_tuned_realpop.json", 'w'), indent=2)
    log("Saved -> step31_tuned_realpop.json")


if __name__ == '__main__':
    run()
