"""
TDIS Forecast — Step 29: does ignition_score x fire-weather-index beat the
ignition model alone, on the REAL full population?

Motivating question (explicit user ask): the ML model is wind-blind (three
independent feature attempts all flat -- see rev2_improvements/), while
NOAA HWP is built specifically around wind*dryness. Multiplying the two
is an appealing heuristic fix. This script tests it directly rather than
arguing about it.

Reuses two already-computed artifacts, no new data pull:
  - models/operational_historical_res5.parquet  (script 26: real full
    population, 2024-01-01 to 2026-07-29, model score 'p' per h3_5/date)
  - dashboard/tdis_dashboard_data.json           (script 09/15: historical
    fwi/fwiN/fwiT per res-5 cell per date, 2024-01-01 to 2026-07-21,
    gridMET-based -- fwiN is the NOAA-HWP-form proxy: G=1.5x*wind,
    M=fm100/30)

Ground truth: same fused-ignition labels, flare-excluded, as scripts 24/26.

Evaluates AUC-PR/AUROC/lift for FOUR scores on the IDENTICAL population
(intersection of both artifacts' dates/cells, so it's a fair comparison,
not different samples):
  1. model alone (p)                    <- current served behavior
  2. model x fwiN  (NOAA-HWP-form)
  3. model x fwiT  (TX-fit HWP-form)
  4. model x fwi   (composite, wind-forward)

AUC-PR/AUROC are rank-based and need no calibration to compare fairly --
multiplying by another variable is a genuinely different score, not a
monotonic transform of p, so this is a legitimate independent test.

Output: rev2_improvements/step29_ignition_x_fwi.json
"""
import json, warnings
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')

TF = Path(__file__).resolve().parent.parent
OUT_DIR = TF / "rev2_improvements"
def log(m): print(m, flush=True)


def run():
    log("Loading model scores (real full population, script 26 output)...")
    scores = pd.read_parquet(TF / "models" / "operational_historical_res5.parquet")
    scores['date'] = pd.to_datetime(scores['date'])
    log(f"  {len(scores):,} cell-days, {scores.date.nunique()} days, {scores.h3_5.nunique()} cells")

    log("Loading historical FWI variants (dashboard data)...")
    d = json.load(open(TF / "dashboard" / "tdis_dashboard_data.json"))
    dates = d['dates']
    didx = {t: i for i, t in enumerate(dates)}
    fwi_rows = []
    for c in d['cells']:
        h5 = c['id']
        for var in ('fwi', 'fwiN', 'fwiT'):
            if var not in c:
                continue
        for i, dt in enumerate(dates):
            fwi_rows.append((h5, dt,
                              c.get('fwi', [None]*len(dates))[i],
                              c.get('fwiN', [None]*len(dates))[i],
                              c.get('fwiT', [None]*len(dates))[i]))
    fwidf = pd.DataFrame(fwi_rows, columns=['h3_5', 'date', 'fwi', 'fwiN', 'fwiT'])
    fwidf['date'] = pd.to_datetime(fwidf['date'])
    log(f"  {len(fwidf):,} cell-days of historical FWI, {fwidf.h3_5.nunique()} cells")

    log("Joining on (h3_5, date) -- intersection only, for a fair identical population...")
    m = scores.merge(fwidf, on=['h3_5', 'date'], how='inner')
    m = m.dropna(subset=['p', 'fwiN'])
    log(f"  {len(m):,} cell-days in the joined population "
        f"({m.date.min().date()} to {m.date.max().date()})")

    log("\nBuilding ground truth (fused ignitions, flare-excluded -- same as scripts 24/26)...")
    flare = pd.read_parquet(TF / "data/labels_fused/flare_cells.parquet")
    flare_cells = set(flare['h3_cell'] if 'h3_cell' in flare.columns else flare.iloc[:, 0])
    lab = pd.read_parquet(TF / "data/labels_fused/ignitions_daily_tx.parquet",
                           columns=['h3_cell', 'date', 'label'])
    lab['date'] = pd.to_datetime(lab['date']).dt.normalize()
    lab = lab[(lab['date'] >= m['date'].min()) & (lab['date'] <= m['date'].max())]
    lab = lab[(~lab['h3_cell'].isin(flare_cells)) & (lab['label'] == 1)]
    import h3
    lab['h3_5'] = [h3.cell_to_parent(c, 5) for c in lab['h3_cell'].values]
    pos = set(zip(lab['h3_5'], lab['date']))
    m['y'] = [1 if (h, dt) in pos else 0 for h, dt in zip(m['h3_5'], m['date'])]
    base = m['y'].mean()
    log(f"  base rate = {base:.4f} ({m['y'].sum():,} positives / {len(m):,})")

    y = m['y'].values
    variants = {
        'model_alone':       m['p'].values,
        'model_x_fwiN_noaa': m['p'].values * m['fwiN'].values,
        'model_x_fwiT_tx':   m['p'].values * m['fwiT'].values,
        'model_x_fwi_comp':  m['p'].values * m['fwi'].values,
    }
    results = {}
    log(f"\n{'variant':<20} {'AUC-PR':>8} {'AUROC':>8} {'lift':>7}  vs model_alone")
    base_aucpr = None
    for name, score in variants.items():
        aucpr = float(average_precision_score(y, score))
        auroc = float(roc_auc_score(y, score))
        lift = aucpr / base
        if base_aucpr is None:
            base_aucpr = aucpr
        delta = aucpr - base_aucpr
        results[name] = dict(aucpr=round(aucpr, 4), auroc=round(auroc, 4), lift=round(lift, 3))
        log(f"{name:<20} {aucpr:>8.4f} {auroc:>8.4f} {lift:>6.2f}x  {delta:+.4f}")

    json.dump(dict(n_scored=len(m), base_rate=float(base),
                   date_range=[str(m['date'].min().date()), str(m['date'].max().date())],
                   results=results),
              open(OUT_DIR / "step29_ignition_x_fwi.json", 'w'), indent=2)
    log("\nSaved -> rev2_improvements/step29_ignition_x_fwi.json")


if __name__ == '__main__':
    run()
