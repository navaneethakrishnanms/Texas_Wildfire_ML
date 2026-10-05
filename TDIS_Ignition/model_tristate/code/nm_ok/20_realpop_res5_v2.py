"""
NM_OK_TexRev — Step 20 (v2, 2026-09-23): FULL real-population all-of-Texas
evaluation for rev5_tristate_v2 (and control_tx_only_v2 for comparison).

This is the missing piece that gives rev5 the same standard of proof as the
served TX-only model and rev4's `tbl_results_multithresh_data.json`
("All of Texas" column): score EVERY TX res-5 cell x EVERY day 2024-01-01 ->
archive end, against real flare-excluded fire outcomes, at the true ~2-3%
base rate. Matched-sample numbers (31.5% base rate) are NOT comparable to
this — that's the whole point of running it.

Method (same convention as rev4's all-of-Texas table):
  - score all res-8 cells per day with the v2 models (17_population_replay_v2
    machinery: corrected national ecoregion, flare-excluded universe)
  - aggregate res-8 scores -> res-5 by mean
  - res-5 cell-day is POSITIVE if any flare-excluded res-8 fire fell in it
  - report: base rate, AUC-PR, AUROC, lift, P/R/F1 at raw thresholds
    [0.4, 0.5, 0.6] (identical grid to rev4's table for comparability)
  - fit isotonic calibrator on 2024-25 rows, verify ECE on held-out 2026 ->
    this REPLACES the PRELIM matched-sample calibrator in the handoff package

Output:
  results/realpop_res5_v2_panel.parquet    (res5_cell, date, label, p_control, p_rev5)
  results/realpop_res5_v2_results.json     (the screenshot-format table, both models)
  models/rev5_tristate_v2_isotonic_calibrator_REALPOP.joblib (+ report json)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Training_rev4_dailymax"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib, json, time
import numpy as np, pandas as pd, xgboost as xgb, joblib, h3
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression
import rev4_pipeline as R

replay = importlib.import_module('17_population_replay_v2')

ROOT = Path(__file__).resolve().parent.parent
TF = ROOT.parent
FEATS = R.FEATS + ['pop_density', 'fm100_5d_min']
START, END = pd.Timestamp('2024-01-01'), R.LAST


def log(m): print(m, flush=True)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    tot = 0.0
    for b in range(bins):
        m_ = idx == b
        if m_.sum() == 0:
            continue
        tot += m_.sum() * abs(p[m_].mean() - y[m_].mean())
    return tot / len(y)


def sweep(y, p, thresholds=(0.4, 0.5, 0.6)):
    out = []
    for t in thresholds:
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append(dict(threshold=t, precision=round(prec, 4), recall=round(rec, 4),
                        f1=round(f1, 4), flagged=int(pred.sum())))
    return out


if __name__ == '__main__':
    st, flare = replay.build_static()
    st['res5'] = [h3.cell_to_parent(c, 5) for c in st['h3_cell'].values]
    weekly = pd.read_parquet(TF / "data/static_features/drought_weekly.parquet")
    weekly['ValidStart'] = pd.to_datetime(weekly['ValidStart'])
    weekly['ValidEnd'] = pd.to_datetime(weekly['ValidEnd'])
    lab = pd.read_parquet(TF / "data/labels_fused/ignitions_daily_tx.parquet",
                          columns=['h3_cell', 'date', 'label'])
    lab['date'] = pd.to_datetime(lab['date']).dt.normalize()
    lab = lab[(~lab.h3_cell.isin(flare)) & (lab.label == 1) &
              (lab.date >= START) & (lab.date <= END)]
    lab['res5'] = [h3.cell_to_parent(c, 5) for c in lab['h3_cell'].values]
    fires_by_date = lab.groupby('date')['res5'].apply(set).to_dict()
    log(f"label window {START.date()}..{END.date()}: {len(lab):,} res-8 fire cell-days")

    fm_mat, fm_cell_idx, fm_day_idx = replay.build_fm100_lookup(set(st.h3_cell))
    fm_cell_rows = st.h3_cell.map(fm_cell_idx).values.astype(int)

    models = {}
    for name, path in [('control', ROOT / "models" / "control_tx_only_v2_seed42.json"),
                       ('rev5', ROOT / "models" / "rev5_tristate_v2_seed42.json")]:
        m = xgb.XGBClassifier(); m.load_model(str(path)); models[name] = m

    rows, t0, nd = [], time.time(), 0
    for d in pd.date_range(str(START.date()), str(END.date()), freq='D'):
        day = replay.score_day(models['control'], st, weekly, fm_mat, fm_cell_rows, fm_day_idx, d)
        if day is None:
            continue
        # score_day computed control's p; rescore same feature frame with rev5 (no reassembly)
        day = day.rename(columns={'p': 'p_control'})
        day['p_rev5'] = models['rev5'].predict_proba(day[FEATS])[:, 1]
        day = day.merge(st[['h3_cell', 'res5']], on='h3_cell', how='left')
        agg = day.groupby('res5').agg(p_control=('p_control', 'mean'),
                                      p_rev5=('p_rev5', 'mean')).reset_index()
        fset = fires_by_date.get(d, set())
        agg['label'] = agg['res5'].isin(fset).astype(np.int8)
        agg['date'] = d
        rows.append(agg)
        nd += 1
        if nd % 100 == 0:
            log(f"  {nd} days scored ({(time.time()-t0)/60:.1f} min)")

    panel = pd.concat(rows, ignore_index=True)
    panel.to_parquet(ROOT / "results" / "realpop_res5_v2_panel.parquet", index=False)
    y = panel['label'].values.astype(int)
    base = float(y.mean())
    log(f"\nPANEL: {len(panel):,} res-5 cell-days over {nd} days | base rate {base:.4f}")

    results = dict(n=len(panel), n_days=nd, base_rate=round(base, 5),
                   window=f"{START.date()}..{END.date()}",
                   note=("All-of-Texas res-5 real population, flare-excluded, corrected "
                         "national ecoregion. Same convention/threshold grid as rev4's "
                         "tbl_results_multithresh_data.json 'alltx' column."))
    for name in ['control', 'rev5']:
        p = panel[f'p_{name}'].values
        aucpr = float(average_precision_score(y, p)); auroc = float(roc_auc_score(y, p))
        results[name] = dict(aucpr=round(aucpr, 4), auroc=round(auroc, 4),
                             lift=round(aucpr / base, 2), sweep=sweep(y, p))
        log(f"[{name}] AUC-PR={aucpr:.4f}  AUROC={auroc:.4f}  lift={aucpr/base:.2f}x")
        for s in results[name]['sweep']:
            log(f"    thr {s['threshold']}: P={s['precision']:.3f} R={s['recall']:.3f} "
                f"F1={s['f1']:.3f} flagged={s['flagged']:,}")

    # real-population isotonic calibrator for rev5 (fit 2024-25, verify 2026)
    fit_m = panel['date'] < '2026-01-01'
    p_fit, y_fit = panel.loc[fit_m, 'p_rev5'].values, y[fit_m.values]
    p_hold, y_hold = panel.loc[~fit_m, 'p_rev5'].values, y[~fit_m.values]
    iso = IsotonicRegression(out_of_bounds='clip'); iso.fit(p_fit, y_fit)
    cal_hold = iso.predict(p_hold)
    rep = dict(fit_n=int(fit_m.sum()), holdout_n=int((~fit_m).sum()),
               holdout_base_rate=round(float(y_hold.mean()), 5),
               ece_before=round(float(ece(y_hold, p_hold)), 5),
               ece_after=round(float(ece(y_hold, cal_hold)), 5),
               mean_pred_before=round(float(p_hold.mean()), 5),
               mean_pred_after=round(float(cal_hold.mean()), 5),
               note="REAL-POPULATION calibrator (res-5, fit 2024-25, verified held-out 2026). "
                    "Supersedes the PRELIM matched-sample calibrator.")
    joblib.dump(iso, ROOT / "models" / "rev5_tristate_v2_isotonic_calibrator_REALPOP.joblib")
    results['rev5_calibration'] = rep
    log(f"\n[calibrator] holdout ECE {rep['ece_before']} -> {rep['ece_after']} "
        f"(mean pred {rep['mean_pred_before']} -> {rep['mean_pred_after']}, "
        f"base {rep['holdout_base_rate']})")

    with open(ROOT / "results" / "realpop_res5_v2_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    log("\nSaved results/realpop_res5_v2_results.json + panel + REALPOP calibrator")
