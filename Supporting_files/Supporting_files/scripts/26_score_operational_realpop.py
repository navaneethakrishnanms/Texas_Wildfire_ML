"""
TDIS Forecast — Step 26: score the OPERATIONAL model (tdis_forecast_hrrr_filtered,
the one actually served) day-by-day on the REAL full population (every res-5 cell x
every day, 2024-2026) and validate against real fire outcomes.

Closes the gap noted for the paper: the real-population evaluation existed only for
the CEILING model (script 23/24: AUC-PR=0.094, lift=3.88x, severe miscalibration).
The deployment-population row should describe the served model, not the diagnostic
control. Uses HRRR *forecast* weather (the operational model's actual inputs --
data/weather_hrrr_forecast/hrrr_24h has full 2024-2026 coverage, 943 daily files),
so this is a genuine "what would the served model have said each day" replay.

Same population and ground truth as script 24 for apples-to-apples comparison:
res-5 cells, fused ignitions labels, flare cells excluded.

Output: models/operational_realpop_validation.json
        models/calibration_reliability_operational.png
        models/operational_historical_res5.parquet (scores, reusable)
"""
import json, time, warnings
import numpy as np, pandas as pd, xgboost as xgb, h3
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

TF = Path(__file__).resolve().parent.parent
HRRR_DIR = TF / "data" / "weather_hrrr_forecast" / "hrrr_24h"
START, END = pd.Timestamp('2024-01-01'), pd.Timestamp('2026-07-29')  # matches label coverage

STATIC = ['road_dist_km', 'ecoregion_id', 'elevation_m', 'slope_deg', 'aspect_deg',
          'avg_burn_prob', 'whp', 'flep4', 'cfl', 'cbd', 'cbh']
HOLIDAYS = {(1, 1), (7, 4), (6, 19), (11, 11), (12, 25), (10, 31)}


def log(m): print(m, flush=True)


def run():
    log("Building res-5 static features (mean-aggregated, matches scripts 08/23)...")
    m8 = pd.read_parquet(TF / "data/static_features/tx_static_master.parquet",
                          columns=['h3_cell', 'lat', 'lon'] + STATIC)
    m8['h3_5'] = [h3.cell_to_parent(c, 5) for c in m8['h3_cell'].values]
    s5 = m8.groupby('h3_5').agg({**{f: 'mean' for f in STATIC}, 'lat': 'mean', 'lon': 'mean'}).reset_index()
    log(f"  {len(s5):,} res-5 cells")

    log("Mapping res-5 centroids -> HRRR grid points (once)...")
    sample = pd.read_parquet(sorted(HRRR_DIR.glob('2024-*.parquet'))[0], columns=['lat', 'lon'])
    tree = cKDTree(sample[['lat', 'lon']].values)
    _, idx = tree.query(s5[['lat', 'lon']].values)
    s5['glat'] = sample['lat'].values[idx].round(3)
    s5['glon'] = sample['lon'].values[idx].round(3)

    model = xgb.XGBClassifier()
    model.load_model(str(TF / "models" / "tdis_forecast_hrrr_filtered.json"))
    FEATS = model.get_booster().feature_names
    log(f"Model features: {FEATS}")

    log("Scoring every res-5 cell x day with HRRR forecast weather...")
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
    log(f"  scored {len(scores):,} cell-days across {scores['date'].nunique()} days ({skipped} days skipped)")
    log(f"  p: mean={scores['p'].mean():.4f}  p95={scores['p'].quantile(0.95):.4f}  max={scores['p'].max():.4f}")
    scores.to_parquet(TF / "models" / "operational_historical_res5.parquet", index=False)

    log("\nBuilding ground truth (fused ignitions, flare-excluded — same as script 24)...")
    flare = pd.read_parquet(TF / "data/labels_fused/flare_cells.parquet")
    flare_cells = set(flare['h3_cell'] if 'h3_cell' in flare.columns else flare.iloc[:, 0])
    lab = pd.read_parquet(TF / "data/labels_fused/ignitions_daily_tx.parquet",
                           columns=['h3_cell', 'date', 'label'])
    lab['date'] = pd.to_datetime(lab['date']).dt.normalize()
    lab = lab[(lab['date'] >= scores['date'].min()) & (lab['date'] <= scores['date'].max())]
    lab = lab[(~lab['h3_cell'].isin(flare_cells)) & (lab['label'] == 1)]
    lab['h3_5'] = [h3.cell_to_parent(c, 5) for c in lab['h3_cell'].values]
    pos = set(zip(lab['h3_5'], lab['date']))
    log(f"  {len(pos):,} unique positive (h3_5, date) pairs")

    scores['y'] = [1 if (h, d) in pos else 0 for h, d in zip(scores['h3_5'], scores['date'])]
    y, p = scores['y'].values, scores['p'].values
    base = y.mean()
    aucpr, auroc = average_precision_score(y, p), roc_auc_score(y, p)
    brier = float(np.mean((p - y) ** 2))
    lift = aucpr / base
    log(f"\nOPERATIONAL, real full population 2024-2026:")
    log(f"  base rate={base:.4f}  mean_pred={p.mean():.4f}")
    log(f"  AUC-PR={aucpr:.4f}  AUROC={auroc:.4f}  Brier={brier:.5f}  lift={lift:.2f}x")
    log(f"  (ceiling on same population: AUC-PR=0.0940 AUROC=0.7774 lift=3.88x)")

    bin_edges = np.linspace(0, 1, 11)
    bidx = np.clip(np.digitize(p, bin_edges) - 1, 0, 9)
    ece, mp_l, fr_l, ct_l = 0.0, [], [], []
    for b in range(10):
        mask = bidx == b
        cnt = int(mask.sum()); ct_l.append(cnt)
        if cnt == 0:
            mp_l.append(np.nan); fr_l.append(np.nan); continue
        mp, fr = float(p[mask].mean()), float(y[mask].mean())
        mp_l.append(mp); fr_l.append(fr)
        ece += (cnt / len(p)) * abs(mp - fr)
    log(f"  ECE={ece:.4f}")
    for mp, fr, cnt in zip(mp_l, fr_l, ct_l):
        if cnt:
            log(f"    pred~{mp:.3f} -> actual {fr:.3f}  (n={cnt:,})")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Perfect calibration')
    ax.plot(mp_l, fr_l, 'o-', color='#C8481E', linewidth=1.6, markersize=6)
    ax.set_xlabel('Mean predicted probability (bin)')
    ax.set_ylabel('Real empirical fire frequency (bin)')
    ax.set_title(f'Operational model, real population 2024-2026\n'
                 f'AUC-PR={aucpr:.4f}  AUROC={auroc:.4f}  lift={lift:.2f}x  ECE={ece:.4f}', fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(TF / "models" / "calibration_reliability_operational.png", dpi=150, bbox_inches='tight')

    json.dump(dict(model='tdis_forecast_hrrr_filtered', n_scored=len(scores),
                   real_positive_rate=float(base), mean_pred=float(p.mean()),
                   aucpr=float(aucpr), auroc=float(auroc), brier=brier, ece=float(ece),
                   lift=float(lift), bins=dict(mean_pred=mp_l, freq=fr_l, count=ct_l),
                   ceiling_comparison=dict(aucpr=0.0940, auroc=0.7774, lift=3.88)),
              open(TF / "models" / "operational_realpop_validation.json", 'w'), indent=2)
    log("\nSaved -> operational_realpop_validation.json, calibration_reliability_operational.png")


if __name__ == '__main__':
    run()
