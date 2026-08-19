"""
TDIS Forecast — Step 27: isotonic calibration for the OPERATIONAL model (and the
ceiling model), fixing the deployment-population miscalibration found by scripts
24/26 (mean predicted ~0.34-0.37 vs real base rate ~0.02, ECE ~0.32-0.35).

Method: Zadrozny & Elkan 2002 isotonic regression, the right tool here because
script 26 showed the miscalibration is perfectly MONOTONIC across all 10 bins --
isotonic fixes that shape essentially exactly.

Discipline: TEMPORAL HOLDOUT, never fit-and-evaluate on the same rows. The
calibrator is fit on 2024-2025 predictions vs real outcomes, then evaluated on
held-out 2026 (never seen by the calibrator). Reported before/after numbers are
holdout-only.

Outputs:
  models/operational_isotonic_calibrator.joblib   (for script 13 / live serving)
  models/ceiling_isotonic_calibrator.joblib       (for the ignC layer, if displayed)
  models/isotonic_calibration_report.json
"""
import json, warnings
import numpy as np, pandas as pd, joblib, h3
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
warnings.filterwarnings('ignore')

TF = Path(__file__).resolve().parent.parent
FIT_END = pd.Timestamp('2025-12-31')   # fit on 2024-2025; evaluate on 2026 holdout


def log(m): print(m, flush=True)


def load_truth(scores):
    flare = pd.read_parquet(TF / "data/labels_fused/flare_cells.parquet")
    flare_cells = set(flare['h3_cell'] if 'h3_cell' in flare.columns else flare.iloc[:, 0])
    lab = pd.read_parquet(TF / "data/labels_fused/ignitions_daily_tx.parquet",
                           columns=['h3_cell', 'date', 'label'])
    lab['date'] = pd.to_datetime(lab['date']).dt.normalize()
    lab = lab[(lab['date'] >= scores['date'].min()) & (lab['date'] <= scores['date'].max())]
    lab = lab[(~lab['h3_cell'].isin(flare_cells)) & (lab['label'] == 1)]
    lab['h3_5'] = [h3.cell_to_parent(c, 5) for c in lab['h3_cell'].values]
    pos = set(zip(lab['h3_5'], lab['date']))
    return np.array([1 if (h, d) in pos else 0 for h, d in zip(scores['h3_5'], scores['date'])])


def ece_brier(y, p, n_bins=10):
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, n_bins + 1)
    bidx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece, bins = 0.0, []
    for b in range(n_bins):
        mask = bidx == b
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append(None); continue
        mp, fr = float(p[mask].mean()), float(y[mask].mean())
        bins.append(dict(mean_pred=round(mp, 4), freq=round(fr, 4), n=cnt))
        ece += (cnt / len(p)) * abs(mp - fr)
    return brier, float(ece), bins


def calibrate(tag, scores_path, pred_col):
    log(f"\n=== {tag} ===")
    s = pd.read_parquet(scores_path)
    s['date'] = pd.to_datetime(s['date']).dt.normalize()
    s['y'] = load_truth(s)
    fit = s[s['date'] <= FIT_END]
    hold = s[s['date'] > FIT_END]
    log(f"fit rows (2024-25): {len(fit):,}  |  holdout rows (2026): {len(hold):,}  "
        f"holdout base rate: {hold['y'].mean():.4f}")

    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(fit[pred_col].values, fit['y'].values)
    joblib.dump(iso, TF / "models" / f"{tag}_isotonic_calibrator.joblib")

    yh = hold['y'].values
    p_raw = hold[pred_col].values
    p_cal = iso.predict(p_raw)
    b0, e0, bins0 = ece_brier(yh, p_raw)
    b1, e1, bins1 = ece_brier(yh, p_cal)
    log(f"HOLDOUT (2026) before: Brier={b0:.5f}  ECE={e0:.4f}  mean_pred={p_raw.mean():.4f}")
    log(f"HOLDOUT (2026) after : Brier={b1:.5f}  ECE={e1:.4f}  mean_pred={p_cal.mean():.4f}")
    log(f"  Brier improvement: {100*(b0-b1)/b0:.1f}%   ECE improvement: {100*(e0-e1)/e0:.1f}%")
    log("  calibrated reliability (holdout):")
    for b in bins1:
        if b:
            log(f"    pred~{b['mean_pred']:.3f} -> actual {b['freq']:.3f}  (n={b['n']:,})")
    return dict(fit_rows=len(fit), holdout_rows=len(hold),
                holdout_base_rate=float(hold['y'].mean()),
                before=dict(brier=b0, ece=e0, mean_pred=float(p_raw.mean()), bins=bins0),
                after=dict(brier=b1, ece=e1, mean_pred=float(p_cal.mean()), bins=bins1))


def run():
    report = {}
    report['operational'] = calibrate('operational', TF / "models" / "operational_historical_res5.parquet", 'p')
    report['ceiling'] = calibrate('ceiling', TF / "models" / "ceiling_historical_res5.parquet", 'p_ceiling')
    report['method'] = ('isotonic regression fit on 2024-2025 real-population predictions vs real '
                        'fused-label outcomes (flare-excluded); all reported numbers are on the '
                        '2026 temporal holdout never seen by the calibrator')
    json.dump(report, open(TF / "models" / "isotonic_calibration_report.json", 'w'), indent=2)
    log("\nSaved -> operational/ceiling _isotonic_calibrator.joblib, isotonic_calibration_report.json")


if __name__ == '__main__':
    run()
