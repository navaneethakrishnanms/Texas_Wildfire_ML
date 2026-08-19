"""
rev2_improvements — Step 30: two targeted improvement tests for the
OPERATIONAL model, aimed at the real-population gap.

A) MONOTONE CONSTRAINTS -- the SHAP diagnostic showed the model learned
   wind with a BACKWARDS direction (more wind -> lower risk), and three
   feature-engineering attempts failed to fix it. Instead of adding
   features, constrain the trees: hrrr_wind (and optionally hrrr_vpd)
   may only ever INCREASE the score. This directly targets the diagnosed
   defect. Variants:
     mono_wind      : wind +1
     mono_wind_vpd  : wind +1, vpd +1

B) SEED ROBUSTNESS of the step4 tuned config (depth 9, mcw 30, lr 0.02,
   sub 0.8, 1000 trees) -- won by +0.0050 test AUC-PR on a single seed.
   Retrain with seeds {42, 7, 2026}; promote only if the gain holds
   across all three (i.e. min gain > 0).

All variants: same table/filters/split/features as script 19 / step4, so
directly comparable to the served model (test AUC-PR=0.4825, AUROC=0.7333).
Selection stays honest: test is scored for every variant here because these
are CONFIRMATORY runs of pre-registered hypotheses, not a search.

Output: rev2_improvements/step30_results.json (+ step30_<name>.json models)
"""
import json, time, warnings
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')

TF = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
LAST_LABEL = pd.Timestamp('2026-07-29')

STATIC = ['road_dist_km', 'ecoregion_id', 'elevation_m', 'slope_deg', 'aspect_deg',
          'avg_burn_prob', 'whp', 'flep4', 'cfl', 'cbd', 'cbh']
TEMPORAL = ['sin_month', 'cos_month', 'sin_dow', 'cos_dow', 'is_weekend', 'is_holiday']
HRRR = ['hrrr_tmp', 'hrrr_vpd', 'hrrr_wind']
FEATS = STATIC + TEMPORAL + HRRR


def log(m): print(m, flush=True)


def load_splits():
    flare_set = set(pd.read_parquet(TF / "data/labels_fused/flare_cells.parquet")['h3_cell'])
    df = pd.read_parquet(TF / "tdis_train_daily_hrrr.parquet")
    df['date'] = pd.to_datetime(df['date']); df['year'] = df['date'].dt.year
    df = df[~df.h3_cell.isin(flare_set)]
    df = df[df.date <= LAST_LABEL]
    df = df.dropna(subset=STATIC).dropna(subset=['hrrr_vpd'])
    df['split'] = np.where(df.year <= 2021, 'train', np.where(df.year == 2022, 'val', 'test'))
    return (df[df.split == 'train'], df[df.split == 'val'], df[df.split == 'test'])


def mono_str(constrained):
    """XGBoost monotone_constraints tuple: +1 for constrained feats, 0 otherwise."""
    return '(' + ','.join('1' if f in constrained else '0' for f in FEATS) + ')'


def train_score(tr, te, params, n_est, seed, mono=None, save_as=None):
    spw = (tr.label == 0).sum() / max((tr.label == 1).sum(), 1)
    kw = dict(n_estimators=n_est, scale_pos_weight=spw, eval_metric='aucpr',
              tree_method='hist', device='cuda', n_jobs=-1, random_state=seed,
              colsample_bytree=0.8, **params)
    if mono:
        kw['monotone_constraints'] = mono_str(mono)
    m = xgb.XGBClassifier(**kw)
    t0 = time.time()
    m.fit(tr[FEATS], tr.label.astype(int), verbose=False)
    yt = te.label.astype(int).values
    pt = m.predict_proba(te[FEATS])[:, 1]
    aucpr = float(average_precision_score(yt, pt))
    auroc = float(roc_auc_score(yt, pt))
    if save_as:
        m.save_model(str(OUT_DIR / save_as))
    return aucpr, auroc, time.time() - t0


def run():
    tr, va, te = load_splits()
    log(f"train {len(tr):,} / val {len(va):,} / test {len(te):,}")
    served = dict(aucpr=0.4825, auroc=0.7333)
    results = {'served_baseline': served}

    SERVED_CFG = dict(max_depth=6, min_child_weight=20, learning_rate=0.05, subsample=0.8)
    TUNED_CFG = dict(max_depth=9, min_child_weight=30, learning_rate=0.02, subsample=0.8)

    log("\n=== A) Monotone constraints (served config, seed 42) ===")
    for name, mono in [('mono_wind', ['hrrr_wind']),
                       ('mono_wind_vpd', ['hrrr_wind', 'hrrr_vpd'])]:
        aucpr, auroc, dt = train_score(tr, te, SERVED_CFG, 400, 42, mono=mono,
                                        save_as=f"step30_{name}.json")
        results[name] = dict(aucpr=round(aucpr, 4), auroc=round(auroc, 4),
                             delta_aucpr=round(aucpr - served['aucpr'], 4))
        log(f"  {name}: AUC-PR={aucpr:.4f} ({aucpr-served['aucpr']:+.4f}) "
            f"AUROC={auroc:.4f} ({auroc-served['auroc']:+.4f})  [{dt:.0f}s]")

    log("\n=== B) Tuned config seed robustness (depth9/mcw30/lr.02/1000t) ===")
    seed_res = []
    for seed in [42, 7, 2026]:
        aucpr, auroc, dt = train_score(tr, te, TUNED_CFG, 1000, seed,
                                        save_as=f"step30_tuned_seed{seed}.json" if seed == 42 else None)
        seed_res.append(dict(seed=seed, aucpr=round(aucpr, 4), auroc=round(auroc, 4),
                             delta_aucpr=round(aucpr - served['aucpr'], 4)))
        log(f"  seed {seed}: AUC-PR={aucpr:.4f} ({aucpr-served['aucpr']:+.4f}) "
            f"AUROC={auroc:.4f}  [{dt:.0f}s]")
    gains = [r['delta_aucpr'] for r in seed_res]
    results['tuned_seeds'] = dict(runs=seed_res, min_gain=min(gains), max_gain=max(gains),
                                   mean_gain=round(float(np.mean(gains)), 4),
                                   robust=bool(min(gains) > 0))
    log(f"  gain across seeds: min {min(gains):+.4f} / mean {np.mean(gains):+.4f} / "
        f"max {max(gains):+.4f}  -> robust: {min(gains) > 0}")

    json.dump(results, open(OUT_DIR / "step30_results.json", 'w'), indent=2)
    log("\nSaved -> step30_results.json")


if __name__ == '__main__':
    run()
