"""
NM_OK_TexRev — Step 19: model-BEHAVIOR validation battery for rev5
(pre-handoff). Performance was validated in 15-18; this validates the model
acts the way it claims (model-autopsy discipline):

A. Monotonicity probes: sweep each constrained feature over its 1-99th pct
   range on a 20k reference sample; fitted mean response must be monotone
   (up for the 8 MONO_UP features, DOWN for fm100_5d_min). Finding-9 test.
B. Permutation importance (TX test, 200k subsample, 3 repeats) -- second
   lens on the gain-based ranking; report top-10 agreement.
C. Calibration: raw reliability + ECE on TX test; isotonic fit on held-out
   2023 rows, verified on 2024+ (rev4 convention).
D. Stability from saved fire ranks (results/fire_ranks_rev5.parquet):
   top-half capture by month (seasonal holes?), repeat-miss cell clustering.
E. Cross-seed agreement: corr of predicted probs across the 3 seeds (50k
   sample) -- prediction-level stability, not just metric-level.

Output: results/extra_validation_rev5.json + printed report.
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / "Training_rev4_dailymax"))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import json
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
import rev4_pipeline as R

ROOT = Path(__file__).resolve().parent.parent
FEATS = R.FEATS + ['pop_density', 'fm100_5d_min']
MONO_UP = sorted(R.MONO_UP)
MONO_DOWN = ['fm100_5d_min']
rng = np.random.RandomState(42)
out = {}


def log(m): print(m, flush=True)


def load():
    df = pd.read_parquet(ROOT / "data/train_tristate.parquet")
    df = df.dropna(subset=FEATS)
    te = df[(df.state == 'tx') & (df.year >= 2024)]
    val23 = df[(df.state == 'tx') & (df.year == 2023)]
    m = xgb.XGBClassifier(); m.load_model(str(ROOT / "models/rev5_tristate_seed42.json"))
    return df, te, val23, m


if __name__ == '__main__':
    df, te, val23, model = load()
    log(f"test rows {len(te):,} | 2023 val rows {len(val23):,}")

    # ---- A. monotonicity probes ----
    log("\n=== A. MONOTONICITY PROBES ===")
    ref = te.sample(20000, random_state=42)
    mono_results = {}
    for feat, direction in [(f, 'up') for f in MONO_UP] + [(f, 'down') for f in MONO_DOWN]:
        lo, hi = np.nanpercentile(df[feat], [1, 99])
        grid = np.linspace(lo, hi, 25)
        means = []
        X = ref[FEATS].copy()
        for v in grid:
            X[feat] = v
            means.append(float(model.predict_proba(X)[:, 1].mean()))
        d = np.diff(means)
        tol = 1e-6
        ok = bool(np.all(d >= -tol)) if direction == 'up' else bool(np.all(d <= tol))
        total_change = means[-1] - means[0]
        mono_results[feat] = dict(direction=direction, monotone=ok,
                                   response_range=round(float(total_change), 4))
        log(f"  {feat} ({direction}): monotone={'PASS' if ok else 'FAIL'} "
            f"response {means[0]:.3f}->{means[-1]:.3f}")
    out['monotonicity'] = mono_results

    # ---- B. permutation importance ----
    log("\n=== B. PERMUTATION IMPORTANCE (3 repeats, 200k subsample) ===")
    sub = te.sample(min(200000, len(te)), random_state=42)
    ys = sub.label.astype(int).values
    from sklearn.metrics import average_precision_score
    base_ap = average_precision_score(ys, model.predict_proba(sub[FEATS])[:, 1])
    perm = {}
    for feat in FEATS:
        drops = []
        for rep in range(3):
            X = sub[FEATS].copy()
            X[feat] = X[feat].sample(frac=1, random_state=rep).values
            drops.append(base_ap - average_precision_score(ys, model.predict_proba(X)[:, 1]))
        perm[feat] = round(float(np.mean(drops)), 5)
    perm_rank = sorted(perm, key=perm.get, reverse=True)
    gain = dict(zip(FEATS, model.feature_importances_))
    gain_rank = sorted(gain, key=gain.get, reverse=True)
    top10_overlap = len(set(perm_rank[:10]) & set(gain_rank[:10]))
    log(f"  base AUC-PR {base_ap:.4f}; top-10 permutation: {perm_rank[:10]}")
    log(f"  gain-vs-permutation top-10 overlap: {top10_overlap}/10")
    out['permutation'] = dict(base_aucpr=round(float(base_ap), 4), drops=perm,
                               top10_overlap_with_gain=top10_overlap,
                               perm_top10=perm_rank[:10], gain_top10=gain_rank[:10])

    # ---- C. calibration ----
    log("\n=== C. CALIBRATION ===")
    from sklearn.isotonic import IsotonicRegression
    p_te = model.predict_proba(te[FEATS])[:, 1]
    y_te = te.label.astype(int).values
    def ece(p, y, bins=10):
        edges = np.linspace(0, 1, bins + 1)
        e, tot = 0.0, len(p)
        for i in range(bins):
            m_ = (p >= edges[i]) & (p < edges[i+1])
            if m_.sum() == 0: continue
            e += m_.sum() / tot * abs(p[m_].mean() - y[m_].mean())
        return e
    raw_ece = ece(p_te, y_te)
    p_val = model.predict_proba(val23[FEATS])[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip').fit(p_val, val23.label.astype(int).values)
    cal_ece = ece(iso.predict(p_te), y_te)
    log(f"  raw ECE (TX test, sample universe): {raw_ece:.4f}")
    log(f"  isotonic (fit 2023, applied to 2024+): ECE {cal_ece:.4f}")
    out['calibration'] = dict(raw_ece=round(float(raw_ece), 4),
                               isotonic_ece=round(float(cal_ece), 4))
    import joblib
    joblib.dump(iso, ROOT / "models/rev5_isotonic_calibrator.joblib")
    log("  calibrator saved -> models/rev5_isotonic_calibrator.joblib")

    # ---- D. stability from fire ranks ----
    log("\n=== D. REPLAY STABILITY (population fire ranks) ===")
    fr = pd.read_parquet(ROOT / "results/fire_ranks_rev5.parquet")
    fr['month'] = pd.to_datetime(fr.date).dt.month
    bym = fr.groupby('month')['rank'].agg(n='count', top50=lambda r: 100*(r >= 0.5).mean())
    worst_m = bym.top50.idxmin()
    log(f"  top-half capture by month: min {bym.top50.min():.1f}% (month {worst_m}), "
        f"max {bym.top50.max():.1f}%")
    misses = fr[fr['rank'] < 0.5]
    rep = misses.groupby('h3_cell').size()
    log(f"  missed fire-cell-days: {len(misses):,} in {rep.size:,} cells; "
        f"cells missed >=5 times: {(rep >= 5).sum():,}")
    out['stability'] = dict(top50_by_month={int(k): round(float(v), 1) for k, v in bym.top50.items()},
                             n_missed=len(misses), repeat_miss_cells_ge5=int((rep >= 5).sum()))

    # ---- E. cross-seed agreement ----
    log("\n=== E. CROSS-SEED PREDICTION AGREEMENT ===")
    s = te.sample(50000, random_state=42)
    preds = {}
    for seed in [42, 7, 2026]:
        mm = xgb.XGBClassifier(); mm.load_model(str(ROOT / f"models/rev5_tristate_seed{seed}.json"))
        preds[seed] = mm.predict_proba(s[FEATS])[:, 1]
    cors = {}
    for a, b in [(42, 7), (42, 2026), (7, 2026)]:
        cors[f"{a}v{b}"] = round(float(np.corrcoef(preds[a], preds[b])[0, 1]), 4)
    log(f"  pairwise prediction correlations: {cors}")
    out['cross_seed_corr'] = cors

    with open(ROOT / "results/extra_validation_rev5.json", 'w') as f:
        json.dump(out, f, indent=2)
    n_mono_fail = sum(1 for v in mono_results.values() if not v['monotone'])
    log(f"\n=== SUMMARY: monotonicity {9-n_mono_fail}/9 PASS | "
        f"gain-perm top10 overlap {top10_overlap}/10 | ECE raw {raw_ece:.4f} -> iso {cal_ece:.4f} | "
        f"worst month top50 {bym.top50.min():.1f}% | min seed corr {min(cors.values()):.4f} ===")
