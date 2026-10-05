"""
NM_OK_TexRev — Step 15: THE CONTROL. rev4's exact config (1000 trees,
depth 9, mcw 30, lr 0.02, monotone set) trained on TX-ONLY rows of
train_tristate.parquet, with the documented split (train 2018-07..2022,
val 2023 excluded, test 2024-2026). This isolates: new negative-sampling
draw + new statics scale + the two new candidate features, WITHOUT
NM/OK. Every rev5 (tri-state) claim compares against THIS, never rev4
directly (rev4 trained on a different table entirely -- different
negative sample, no pop_density/fm100_5d_min).

FEATS = rev4's 26 + pop_density (unconstrained) + fm100_5d_min
(monotone DOWN -- low moisture = high risk, opposite sign from
hrrr_mstav's convention). 3 seeds [42,7,2026].

Output: models/control_tx_only_seed{42,7,2026}.json, results/control_results.json
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / "Training_rev4_dailymax"))
import json
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
import rev4_pipeline as R

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)

FEATS = R.FEATS + ['pop_density', 'fm100_5d_min']
MONO_MAP = {f: 1 for f in R.MONO_UP}
MONO_MAP['fm100_5d_min'] = -1
MONO = '(' + ','.join(str(MONO_MAP.get(f, 0)) for f in FEATS) + ')'
SEEDS = [42, 7, 2026]


def log(m): print(m, flush=True)


def load_split(train_states):
    # caught 2026-09-16 (before it ran): test set must ALWAYS be TX-only, 2024-2026 --
    # "NM/OK never appear in any test set or metric" is a hard PRD rule, not just a
    # training-data choice. Filtering train/test by the SAME states list would silently
    # let NM/OK 2024+ rows into rev5's test set.
    df = pd.read_parquet(ROOT / "data" / "train_tristate.parquet")
    df = df.dropna(subset=FEATS)
    tr = df[df.state.isin(train_states) & (df.year <= 2022)]
    te = df[(df.state == 'tx') & (df.year >= 2024)]
    log(f"train_states={train_states}: train {len(tr):,} (pos {int(tr.label.sum()):,}) | "
        f"test (TX-ONLY, always) {len(te):,} (pos {int(te.label.sum()):,})")
    return tr, te


def train_and_eval(tr, te, label):
    spw = (tr.label == 0).sum() / max((tr.label == 1).sum(), 1)
    yt = te.label.astype(int).values
    results = []
    for seed in SEEDS:
        m = xgb.XGBClassifier(n_estimators=1000, max_depth=9, min_child_weight=30,
                               learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=spw, eval_metric='aucpr',
                               monotone_constraints=MONO,
                               tree_method='hist', device='cuda', n_jobs=-1, random_state=seed)
        m.fit(tr[FEATS], tr.label.astype(int), verbose=False)
        p = m.predict_proba(te[FEATS])[:, 1]
        aucpr = float(average_precision_score(yt, p))
        auroc = float(roc_auc_score(yt, p))
        imp = dict(zip(FEATS, m.feature_importances_.round(4).tolist()))
        log(f"  [{label}] seed {seed}: test AUC-PR={aucpr:.4f}  AUROC={auroc:.4f}")
        m.save_model(str(ROOT / "models" / f"{label}_seed{seed}.json"))
        results.append(dict(seed=seed, aucpr=aucpr, auroc=auroc, importances=imp))
    mean_aucpr = float(np.mean([r['aucpr'] for r in results]))
    std_aucpr = float(np.std([r['aucpr'] for r in results]))
    mean_auroc = float(np.mean([r['auroc'] for r in results]))
    log(f"  [{label}] MEAN: AUC-PR={mean_aucpr:.4f} (std {std_aucpr:.4f})  AUROC={mean_auroc:.4f}")
    return dict(label=label, n_train=len(tr), n_test=len(te), seeds=results,
                mean_aucpr=mean_aucpr, std_aucpr=std_aucpr, mean_auroc=mean_auroc)


if __name__ == '__main__':
    log("=== CONTROL: TX-only, new table, new split, 28 features ===")
    tr, te = load_split(['tx'])
    result = train_and_eval(tr, te, 'control_tx_only')
    with open(ROOT / "results" / "control_results.json", 'w') as f:
        json.dump(result, f, indent=2)
    log(f"\nSaved results/control_results.json -- MEAN AUC-PR={result['mean_aucpr']:.4f}")
    log(f"(rev4's own served sample AUC-PR reference: {R.SERVED['test_aucpr']:.4f} -- "
        f"NOT directly comparable, different table/split/features; this control IS "
        f"the correct baseline for rev5 comparison)")
