"""
v2 (2026-09-23): scores with control_tx_only_v2/rev5_tristate_v2 via
17_population_replay_v2's build_static() (corrected national ecoregion_id).

NM_OK_TexRev — Step 18: named-fire battery + tau menu, control vs rev5.

Battery: the 5 pre-registered catastrophic events (rev4_pipeline.EVENTS —
Smokehouse Creek, Windy Deuce, Crabapple, Lavender, Hunggate), scored with
both models on the actual event day: where did the real fire cells rank.

Tau menu: fit tau on 2024-25 fire ranks (from script 17's saved
fire_ranks_{model}_v2.parquet), verify missed% on held-out 2026.

Output: results/battery_tau_control_vs_rev5_v2.json + printed tables.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Training_rev4_dailymax"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json, importlib
import numpy as np, pandas as pd, xgboost as xgb
import rev4_pipeline as R

replay = importlib.import_module('17_population_replay_v2')

ROOT = Path(__file__).resolve().parent.parent
TF = ROOT.parent


def log(m): print(m, flush=True)


if __name__ == '__main__':
    st, flare = replay.build_static()
    weekly = pd.read_parquet(TF / "data/static_features/drought_weekly.parquet")
    weekly['ValidStart'] = pd.to_datetime(weekly['ValidStart'])
    weekly['ValidEnd'] = pd.to_datetime(weekly['ValidEnd'])
    lab = pd.read_parquet(TF / "data/labels_fused/ignitions_daily_tx.parquet",
                           columns=['h3_cell', 'date', 'label'])
    lab['date'] = pd.to_datetime(lab['date']).dt.normalize()
    lab = lab[(~lab.h3_cell.isin(flare)) & (lab.label == 1)]

    fm_mat, fm_cell_idx, fm_day_idx = replay.build_fm100_lookup(set(st.h3_cell))
    fm_cell_rows = st.h3_cell.map(fm_cell_idx).values.astype(int)

    out = {}
    log("\n=== NAMED-FIRE BATTERY (both models, model's own ranks, no floor) ===")
    for label, path in [('control', ROOT / "models" / "control_tx_only_v2_seed42.json"),
                         ('rev5', ROOT / "models" / "rev5_tristate_v2_seed42.json")]:
        model = xgb.XGBClassifier(); model.load_model(str(path))
        battery = {}
        for name, ds, la0, la1, lo0, lo1 in R.EVENTS:
            d = pd.Timestamp(ds)
            day = replay.score_day(model, st, weekly, fm_mat, fm_cell_rows, fm_day_idx, d)
            if day is None:
                log(f"  [{label}] {name}: missing archive day"); continue
            box = day[(day.lat.between(la0, la1)) & (day.lon.between(lo0, lo1))]
            fcells = set(lab[lab.date == d].h3_cell) & set(box.h3_cell)
            fc = box[box.h3_cell.isin(fcells)]
            res = dict(n_fire_cells=len(fc),
                       mean_rank=round(float(fc['rank'].mean()), 3) if len(fc) else None,
                       min_rank=round(float(fc['rank'].min()), 3) if len(fc) else None,
                       box_top25_pct=round(float((box['rank'] >= 0.75).mean()) * 100, 1))
            battery[name] = res
            log(f"  [{label}] {name} ({ds}): fire cells {res['n_fire_cells']}, "
                f"mean rank {res['mean_rank']}, min rank {res['min_rank']}, "
                f"box>=top-25%: {res['box_top25_pct']}%")
        out[label] = dict(battery=battery)

    log("\n=== TAU MENU (fit 2024-25 -> verify held-out 2026) ===")
    for label in ['control', 'rev5']:
        fr = pd.read_parquet(ROOT / "results" / f"fire_ranks_{label}_v2.parquet")
        fit = fr[fr.date < '2026-01-01']; hold = fr[fr.date >= '2026-01-01']
        menu = {}
        log(f"  [{label}] fit n={len(fit):,} | 2026 holdout n={len(hold):,}")
        for pct, nm in [(10, '90pct_recall'), (5, '95pct_recall'), (1, '99pct_recall')]:
            t = float(np.percentile(fit['rank'], pct))
            missed = float((hold['rank'] < t).mean() * 100)
            menu[nm] = dict(tau=round(t, 4), state_share_above_pct=round((1 - t) * 100, 1),
                             missed_2026_pct=round(missed, 2))
            log(f"    {nm}: tau={t:.4f}  state-share flagged={100*(1-t):.1f}%  "
                f"2026 missed={missed:.2f}%")
        out[label]['tau_menu'] = menu

    with open(ROOT / "results" / "battery_tau_control_vs_rev5_v2.json", 'w') as f:
        json.dump(out, f, indent=2)
    log("\nSaved results/battery_tau_control_vs_rev5_v2.json")
