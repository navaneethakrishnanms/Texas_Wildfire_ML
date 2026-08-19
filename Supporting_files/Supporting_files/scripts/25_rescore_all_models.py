"""
TDIS Forecast — Step 25: re-score honest/operational-HRRR/ceiling (all _filtered) fresh,
directly from the saved model files -- resolves a discrepancy found between
flare_filter_retrain.log (AUC-PR=0.413) and retrain20.log (AUC-PR=0.450) for what's
supposedly the same honest_filtered model. Ground truth from the files currently on
disk, not a possibly-stale log line.

Two populations, by data-engineering necessity (matches how 18/19/22 actually built
their own test sets, not an artificial unification):
  - honest_filtered + ceiling_filtered: same table (tdis_train_daily_tx_flarefiltered.parquet)
    -> mutually comparable, identical test rows.
  - operational_hrrr_filtered: its own table (tdis_train_daily_hrrr.parquet, HRRR archive
    only goes back to 2018-07 and is a separate join) -> its own clean test rows, same
    flare+phantom+coverage filters script 19 applies. Overlapping population, not identical.

Also runs the same threshold sweep (0.2-0.8) reported in VALIDATION.md for all three.
"""
import json
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

TF = Path(__file__).resolve().parent.parent

STATIC = ['road_dist_km', 'ecoregion_id', 'elevation_m', 'slope_deg', 'aspect_deg',
          'avg_burn_prob', 'whp', 'flep4', 'cfl', 'cbd', 'cbh']
TEMPORAL = ['sin_month', 'cos_month', 'sin_dow', 'cos_dow', 'is_weekend', 'is_holiday']
WEATHER_OBS = ['erc', 'fm100', 'vpd', 'vs', 'rmax', 'rmin', 'tmmx', 'pr']
HRRR_FEATS = ['hrrr_tmp', 'hrrr_vpd', 'hrrr_wind']
LAST_LABEL = pd.Timestamp('2026-07-29')

FEATS_HONEST = STATIC + TEMPORAL
FEATS_CEILING = STATIC + TEMPORAL + WEATHER_OBS
FEATS_HRRR = STATIC + TEMPORAL + HRRR_FEATS


def log(m): print(m, flush=True)


def score_and_sweep(m, te, feats):
    y = te['label'].astype(int).values
    p = m.predict_proba(te[feats])[:, 1]
    aucpr = average_precision_score(y, p)
    auroc = roc_auc_score(y, p)
    lift = aucpr / y.mean()
    sweep = {}
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred = (p > t).astype(int)
        sweep[str(t)] = dict(
            precision=round(float(precision_score(y, pred, zero_division=0)), 3),
            recall=round(float(recall_score(y, pred, zero_division=0)), 3),
            f1=round(float(f1_score(y, pred, zero_division=0)), 3),
        )
    return dict(n_test=len(te), test_pos_rate=round(float(y.mean()), 4),
                aucpr=round(float(aucpr), 4), auroc=round(float(auroc), 4),
                lift=round(float(lift), 2), threshold_sweep=sweep, features_used=feats)


def run():
    results = {}

    log("=== honest_filtered + ceiling_filtered (shared table) ===")
    df = pd.read_parquet(TF / "tdis_train_daily_tx_flarefiltered.parquet")
    clean = df.dropna(subset=FEATS_CEILING).reset_index(drop=True)
    te = clean[clean['split'] == 'test'].reset_index(drop=True)
    log(f"Clean rows: {len(clean):,} total, {len(te):,} in test split, "
        f"test pos rate {te['label'].mean():.4f}")

    for tag, model_file, feats in [
        ('honest_filtered', 'tdis_forecast_baseline_honest_filtered.json', FEATS_HONEST),
        ('ceiling_filtered', 'tdis_forecast_baseline_ceiling_filtered.json', FEATS_CEILING),
    ]:
        m = xgb.XGBClassifier()
        m.load_model(str(TF / "models" / model_file))
        results[tag] = score_and_sweep(m, te, feats)
        r = results[tag]
        log(f"\n{tag}: AUC-PR={r['aucpr']} AUROC={r['auroc']} lift={r['lift']}x (n={r['n_test']:,})")
        for t, v in r['threshold_sweep'].items():
            log(f"  t={t}  P={v['precision']}  R={v['recall']}  F1={v['f1']}")

    log("\n=== operational_hrrr_filtered (own table, own filters, matches script 19) ===")
    flare_set = set(pd.read_parquet(TF / "data/labels_fused/flare_cells.parquet")['h3_cell'])
    dfh = pd.read_parquet(TF / "tdis_train_daily_hrrr.parquet")
    dfh['date'] = pd.to_datetime(dfh['date']); dfh['year'] = dfh['date'].dt.year
    n0 = len(dfh)
    dfh = dfh[~dfh.h3_cell.isin(flare_set)]
    dfh = dfh[dfh.date <= LAST_LABEL]
    dfh = dfh.dropna(subset=STATIC).dropna(subset=['hrrr_vpd'])
    log(f"HRRR rows: {n0:,} -> {len(dfh):,} after flare+phantom+coverage filters")
    dfh['split'] = np.where(dfh.year <= 2021, 'train', np.where(dfh.year == 2022, 'val', 'test'))
    teh = dfh[dfh.split == 'test'].reset_index(drop=True)
    log(f"HRRR test rows: {len(teh):,}, test pos rate {teh['label'].mean():.4f}")

    m = xgb.XGBClassifier()
    m.load_model(str(TF / "models" / "tdis_forecast_hrrr_filtered.json"))
    results['operational_hrrr_filtered'] = score_and_sweep(m, teh, FEATS_HRRR)
    r = results['operational_hrrr_filtered']
    log(f"\noperational_hrrr_filtered: AUC-PR={r['aucpr']} AUROC={r['auroc']} lift={r['lift']}x (n={r['n_test']:,})")
    for t, v in r['threshold_sweep'].items():
        log(f"  t={t}  P={v['precision']}  R={v['recall']}  F1={v['f1']}")

    json.dump(results, open(TF / "models" / "rescore_all_models.json", 'w'), indent=2)
    log("\nSaved -> models/rescore_all_models.json")


if __name__ == '__main__':
    run()
