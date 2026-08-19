"""
TDIS Forecast — Step 28: flare-threshold ROBUSTNESS BAND (SANITY_CHECK F2 fix).

The 3%-persistence flare cutoff is load-bearing (1% removes 36.9% of positive
labels, 10% removes 18.0%) and unvalidated against an independent source. Until
the VIIRS `type`-flag cross-check is done, this quantifies how much the served
model's REAL-POPULATION metrics move across cutoffs 1% / 3% / 10% — if the band
is narrow, the headline conclusions don't hinge on the arbitrary choice.

Uses the already-computed operational real-population scores (script 26) and
rebuilds the ground truth under each flare definition. Scores never change --
only which cells count as 'industrial hotspot' when building truth.

Output: models/flare_robustness_band.json
"""
import json, warnings
import numpy as np, pandas as pd, h3
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')

TF = Path(__file__).resolve().parent.parent


def log(m): print(m, flush=True)


def run():
    log("Loading operational real-population scores...")
    s = pd.read_parquet(TF / "models" / "operational_historical_res5.parquet")
    s['date'] = pd.to_datetime(s['date']).dt.normalize()

    log("Loading raw fused labels and computing per-cell persistence...")
    lab = pd.read_parquet(TF / "data/labels_fused/ignitions_daily_tx.parquet",
                           columns=['h3_cell', 'date', 'label'])
    lab['date'] = pd.to_datetime(lab['date']).dt.normalize()
    lab = lab[lab['label'] == 1]
    total_days = (lab['date'].max() - lab['date'].min()).days + 1
    fire_days = lab.groupby('h3_cell')['date'].nunique()
    log(f"  record spans {total_days} days; {len(fire_days):,} cells ever had fire")

    win = lab[(lab['date'] >= s['date'].min()) & (lab['date'] <= s['date'].max())].copy()

    results = {}
    for frac in [0.01, 0.03, 0.10]:
        flare_cells = set(fire_days[fire_days > frac * total_days].index)
        w = win[~win['h3_cell'].isin(flare_cells)]
        w5 = set(zip([h3.cell_to_parent(c, 5) for c in w['h3_cell'].values], w['date']))
        y = np.array([1 if (h, d) in w5 else 0 for h, d in zip(s['h3_5'], s['date'])])
        p = s['p'].values
        base = y.mean()
        aucpr = average_precision_score(y, p)
        auroc = roc_auc_score(y, p)
        lift = aucpr / base
        results[f'{int(frac*100)}pct'] = dict(
            n_flare_cells=len(flare_cells), base_rate=round(float(base), 4),
            aucpr=round(float(aucpr), 4), auroc=round(float(auroc), 4),
            lift=round(float(lift), 2))
        log(f"flare cutoff >{int(frac*100)}%: {len(flare_cells):,} flare cells | "
            f"base={base:.4f}  AUC-PR={aucpr:.4f}  AUROC={auroc:.4f}  lift={lift:.2f}x")

    lifts = [v['lift'] for v in results.values()]
    aurocs = [v['auroc'] for v in results.values()]
    log(f"\nBAND: lift {min(lifts):.2f}-{max(lifts):.2f}x | AUROC {min(aurocs):.4f}-{max(aurocs):.4f}")
    results['verdict'] = (f"lift band {min(lifts):.2f}-{max(lifts):.2f}x, AUROC band "
                           f"{min(aurocs):.4f}-{max(aurocs):.4f} across a 10x change in the "
                           f"flare cutoff -- conclusions {'DO NOT' if max(lifts)-min(lifts) < 0.5 else 'MAY'} "
                           f"hinge on the 3% choice")
    json.dump(results, open(TF / "models" / "flare_robustness_band.json", 'w'), indent=2)
    log("Saved -> models/flare_robustness_band.json")


if __name__ == '__main__':
    run()
