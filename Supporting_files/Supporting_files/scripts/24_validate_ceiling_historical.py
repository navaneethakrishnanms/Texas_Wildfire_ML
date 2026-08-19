"""
TDIS Forecast — Step 24: validate the day-by-day ceiling-model historical scores
(models/ceiling_historical_res5.parquet, from script 23) against REAL fire outcomes,
in response to the mean-probability sanity check flagging p_ceiling mean (0.374) far
above the model's own matched-negative test positive rate (0.209).

Builds a dense (h3_5, date) x {real fire occurred?} ground truth from the fused
ignitions label file (flare cells excluded, same convention as script 18) for the
exact 2024-2026 population/period that was scored, then reports:
  - the REAL positive rate in this population (resolves whether 0.374 is explainable
    by a base-rate/population difference vs the model's own matched-negative test set)
  - AUC-PR / AUROC (does the model rank real fire-days above non-fire-days at all?)
  - Brier score, ECE, and a 10-bin reliability diagram (is it a scale problem,
    fixable by calibration, or a ranking problem?)

Output: models/ceiling_historical_validation.json + calibration_reliability_ceiling.png
"""
import json
import numpy as np, pandas as pd, h3
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TF = Path(__file__).resolve().parent.parent


def log(m): print(m, flush=True)


def run():
    log("Loading ceiling historical scores...")
    scores = pd.read_parquet(TF / "models" / "ceiling_historical_res5.parquet")
    scores['date'] = pd.to_datetime(scores['date']).dt.normalize()
    log(f"  {len(scores):,} scored (h3_5, date) rows, "
        f"{scores['h3_5'].nunique():,} cells x {scores['date'].nunique()} days")

    log("Loading flare cell exclusion list...")
    flare = pd.read_parquet(TF / "data" / "labels_fused" / "flare_cells.parquet")
    flare_cells = set(flare['h3_cell'] if 'h3_cell' in flare.columns else flare.iloc[:, 0])
    log(f"  {len(flare_cells):,} flare cells to exclude")

    log("Loading real fused ignitions, building ground truth for the scored period...")
    lab = pd.read_parquet(TF / "data" / "labels_fused" / "ignitions_daily_tx.parquet",
                           columns=['h3_cell', 'date', 'label'])
    lab['date'] = pd.to_datetime(lab['date']).dt.normalize()
    lab = lab[(lab['date'] >= scores['date'].min()) & (lab['date'] <= scores['date'].max())]
    lab = lab[~lab['h3_cell'].isin(flare_cells)]
    lab = lab[lab['label'] == 1]
    lab['h3_5'] = [h3.cell_to_parent(c, 5) for c in lab['h3_cell'].values]
    real_positive_pairs = set(zip(lab['h3_5'], lab['date']))
    log(f"  {len(lab):,} real (post-flare-filter) ignition rows -> "
        f"{len(real_positive_pairs):,} unique (h3_5, date) positive pairs")

    scores['y_true'] = [
        1 if (h5, d) in real_positive_pairs else 0
        for h5, d in zip(scores['h3_5'], scores['date'])
    ]
    real_pos_rate = scores['y_true'].mean()
    log(f"\nREAL positive rate in this exact scored population: {real_pos_rate:.4f}")
    log(f"(vs. model's own matched-negative test positive rate: 0.2092)")
    log(f"(vs. mean predicted p_ceiling: {scores['p_ceiling'].mean():.4f})")

    y = scores['y_true'].values
    p = scores['p_ceiling'].values
    aucpr = average_precision_score(y, p)
    auroc = roc_auc_score(y, p)
    brier = float(np.mean((p - y) ** 2))
    lift = aucpr / max(real_pos_rate, 1e-9)
    log(f"\nAUC-PR={aucpr:.4f}  AUROC={auroc:.4f}  Brier={brier:.5f}  lift={lift:.2f}x "
        f"(vs. real base rate {real_pos_rate:.4f})")

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, bin_edges) - 1, 0, n_bins - 1)
    ece = 0.0
    mean_pred, freq, count = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        count.append(cnt)
        if cnt == 0:
            mean_pred.append(np.nan); freq.append(np.nan); continue
        mp, fr = float(p[mask].mean()), float(y[mask].mean())
        mean_pred.append(mp); freq.append(fr)
        ece += (cnt / len(p)) * abs(mp - fr)
    log(f"ECE={ece:.4f}")
    log("\nReliability (mean predicted -> real freq, count):")
    for mp, fr, cnt in zip(mean_pred, freq, count):
        if cnt:
            log(f"  pred~{mp:.3f} -> actual {fr:.3f}  (n={cnt:,})")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Perfect calibration')
    ax.plot(mean_pred, freq, 'o-', color='#C8481E', linewidth=1.6, markersize=6)
    ax.set_xlabel('Mean predicted p_ceiling (bin)')
    ax.set_ylabel('Real empirical fire frequency (bin)')
    ax.set_title(f'Ceiling model, historical scoring, 2024-2026\n'
                 f'AUC-PR={aucpr:.4f}  AUROC={auroc:.4f}  Brier={brier:.5f}  ECE={ece:.4f}',
                 fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(TF / "models" / "calibration_reliability_ceiling.png", dpi=150, bbox_inches='tight')

    json.dump(dict(n_scored=len(scores), real_positive_rate=real_pos_rate,
                   model_test_pos_rate=0.2092, mean_p_ceiling=float(p.mean()),
                   aucpr=aucpr, auroc=auroc, brier=brier, ece=ece, lift=lift,
                   bins=dict(mean_pred=mean_pred, freq=freq, count=count)),
              open(TF / "models" / "ceiling_historical_validation.json", 'w'), indent=2)
    log("\nSaved -> ceiling_historical_validation.json, calibration_reliability_ceiling.png")


if __name__ == '__main__':
    run()
