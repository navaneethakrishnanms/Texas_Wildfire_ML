"""
rev2_improvements — Step 4: hyperparameter sweep for the OPERATIONAL model.
First tuning ever run on any TDIS model (all inherited a fixed config: 400
trees, depth 6, lr 0.05, mcw 20).

Design mirrors IgnitionNet's two-stage search for comparability:
  Stage 1: depth {6,7,8,9} x min_child_weight {10,20,30}   (12 trials, lr=0.05)
  Stage 2: winner's depth/mcw, lr {0.05,0.02,0.01} x subsample {0.8,0.9}
           (6 trials; n_estimators scaled inversely with lr so low-lr runs
            are not undertrained: 400 / 1000 / 2000)
Selection on VAL (2022) AUC-PR only; TEST scored exactly once, for the
winner. Same table/filters/split/features as script 19 -- directly
comparable to the served model (test AUC-PR=0.4825, AUROC=0.7333).

Specific question this answers: IgnitionNet's tuned optimum was depth 9 vs
our 6, and insufficient depth was the hypothesis for why the wind x dryness
interaction never got learned. If depth 9 wins AND meaningfully beats depth
6, the interaction story deserves a second look.

Output: rev2_improvements/step4_sweep_results.json (+ step4_best_model.json)
"""
import json, time, warnings
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')

TF = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).parent
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


def train_eval(tr, va, params, n_est):
    spw = (tr.label == 0).sum() / max((tr.label == 1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=n_est, scale_pos_weight=spw, eval_metric='aucpr',
                           tree_method='hist', device='cuda', n_jobs=-1, random_state=42,
                           colsample_bytree=0.8, **params)
    m.fit(tr[FEATS], tr.label.astype(int), verbose=False)
    yv = va.label.astype(int).values
    pv = m.predict_proba(va[FEATS])[:, 1]
    return m, float(average_precision_score(yv, pv)), float(roc_auc_score(yv, pv))


def run():
    tr, va, te = load_splits()
    log(f"train {len(tr):,} / val {len(va):,} / test {len(te):,}")
    trials = []

    log("\n=== Stage 1: depth x min_child_weight (lr=0.05, sub=0.8, 400 trees) ===")
    for depth in [6, 7, 8, 9]:
        for mcw in [10, 20, 30]:
            t0 = time.time()
            _, aucpr, auroc = train_eval(tr, va, dict(max_depth=depth, min_child_weight=mcw,
                                                       learning_rate=0.05, subsample=0.8), 400)
            trials.append(dict(stage=1, depth=depth, mcw=mcw, lr=0.05, sub=0.8,
                               n_est=400, val_aucpr=round(aucpr, 4), val_auroc=round(auroc, 4)))
            log(f"  d{depth}_mcw{mcw}: val AUC-PR={aucpr:.4f} AUROC={auroc:.4f} ({time.time()-t0:.0f}s)")

    s1 = max([t for t in trials if t['stage'] == 1], key=lambda t: t['val_aucpr'])
    log(f"\nStage 1 winner: depth={s1['depth']} mcw={s1['mcw']} (val AUC-PR={s1['val_aucpr']})")

    log("\n=== Stage 2: lr x subsample (winner's depth/mcw) ===")
    for lr, n_est in [(0.05, 400), (0.02, 1000), (0.01, 2000)]:
        for sub in [0.8, 0.9]:
            t0 = time.time()
            _, aucpr, auroc = train_eval(tr, va, dict(max_depth=s1['depth'], min_child_weight=s1['mcw'],
                                                       learning_rate=lr, subsample=sub), n_est)
            trials.append(dict(stage=2, depth=s1['depth'], mcw=s1['mcw'], lr=lr, sub=sub,
                               n_est=n_est, val_aucpr=round(aucpr, 4), val_auroc=round(auroc, 4)))
            log(f"  lr{lr}_s{sub} ({n_est} trees): val AUC-PR={aucpr:.4f} AUROC={auroc:.4f} ({time.time()-t0:.0f}s)")

    best = max(trials, key=lambda t: t['val_aucpr'])
    log(f"\nOverall winner: depth={best['depth']} mcw={best['mcw']} lr={best['lr']} "
        f"sub={best['sub']} n_est={best['n_est']} (val AUC-PR={best['val_aucpr']})")

    log("\n=== Scoring winner on TEST (once) ===")
    m, _, _ = train_eval(tr, va, dict(max_depth=best['depth'], min_child_weight=best['mcw'],
                                       learning_rate=best['lr'], subsample=best['sub']), best['n_est'])
    yt = te.label.astype(int).values
    pt = m.predict_proba(te[FEATS])[:, 1]
    t_aucpr = float(average_precision_score(yt, pt))
    t_auroc = float(roc_auc_score(yt, pt))
    log(f"TUNED  : test AUC-PR={t_aucpr:.4f} AUROC={t_auroc:.4f} lift={t_aucpr/yt.mean():.2f}x")
    log(f"CURRENT: test AUC-PR=0.4825 AUROC=0.7333 lift=2.06x (served model, fixed config)")
    log(f"delta  : AUC-PR {t_aucpr-0.4825:+.4f}  AUROC {t_auroc-0.7333:+.4f}")

    m.save_model(str(OUT_DIR / "step4_best_model.json"))
    json.dump(dict(trials=trials, winner=best,
                   winner_test=dict(aucpr=round(t_aucpr, 4), auroc=round(t_auroc, 4),
                                    lift=round(t_aucpr / float(yt.mean()), 2)),
                   served_baseline=dict(aucpr=0.4825, auroc=0.7333, lift=2.06)),
              open(OUT_DIR / "step4_sweep_results.json", 'w'), indent=2)
    log("\nSaved -> step4_sweep_results.json, step4_best_model.json")


if __name__ == '__main__':
    run()
