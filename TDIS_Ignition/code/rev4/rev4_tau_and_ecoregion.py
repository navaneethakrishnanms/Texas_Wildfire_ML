"""Replay 2024-01-01..2026-07-29 at res-8 with the rev4 model; save every real
fire cell's daily statewide rank + ecoregion. Then: tau menu refit on 2024-25,
held-out 2026 verification, and the C3 ecoregion miss table."""
import sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np, xgboost as xgb, time
from rev4_pipeline import *

flare = set(pd.read_parquet(NT / "flare_cells_v2.parquet")['h3_cell'])
st = build_static_res8(flare)
eco = dict(zip(st.h3_cell, st.ecoregion_id))
weekly = pd.read_parquet(TF / "data/static_features/drought_weekly.parquet")
weekly['ValidStart'] = pd.to_datetime(weekly['ValidStart'])
weekly['ValidEnd'] = pd.to_datetime(weekly['ValidEnd'])
model = xgb.XGBClassifier(); model.load_model('rev4_model.json')
lab = pd.read_parquet(TF / "data/labels_fused/ignitions_daily_tx.parquet",
                       columns=['h3_cell', 'date', 'label'])
lab['date'] = pd.to_datetime(lab['date']).dt.normalize()
lab = lab[(~lab.h3_cell.isin(flare)) & (lab.label == 1)]
fbd = lab[(lab.date >= '2024-01-01') & (lab.date <= LAST)].groupby('date')['h3_cell'].apply(set).to_dict()

rows, t0, nd = [], time.time(), 0
for d in pd.date_range('2024-01-01', str(LAST.date()), freq='D'):
    fires = fbd.get(d, set())
    if not fires: continue
    day = score_day_res8(model, st, weekly, d)
    if day is None: continue
    cr = dict(zip(day.h3_cell.values, day['rank'].values))
    for c in fires:
        r = cr.get(c)
        if r is not None:
            rows.append((d, c, r, eco.get(c)))
    nd += 1
    if nd % 100 == 0: print(f"{nd} fire-days ({(time.time()-t0)/60:.1f} min)", flush=True)

fr = pd.DataFrame(rows, columns=['date', 'h3_cell', 'rank', 'ecoregion_id'])
fr.to_parquet('rev4_fire_ranks_2024_2026.parquet', index=False)
fit = fr[fr.date < '2026-01-01']; hold = fr[fr.date >= '2026-01-01']
print(f"\nfit (2024-25) n={len(fit):,}  holdout (2026) n={len(hold):,}")
print("\nTAU MENU (fit on 2024-25 fire ranks -> verified on held-out 2026):")
for pct, nm in [(10, '90% recall'), (5, '95% recall'), (1, '99% recall')]:
    t = float(np.percentile(fit['rank'], pct))
    print(f"  {nm}: tau={t:.4f}  Low share of state={t*100:.1f}%  "
          f"2026 missed={100*(hold['rank']<t).mean():.2f}%")
print(f"  current tau 0.4623: 2026 missed={100*(hold['rank']<0.4623).mean():.2f}%")
print("\nC3 ECOREGION TABLE (2026 held-out, missed at tau=0.4623; fix-target: SW Tablelands was 45.7%):")
g = hold.groupby('ecoregion_id')['rank'].agg(n='count', missed=lambda r: 100*(r < 0.4623).mean())
print(g.round(1).sort_values('missed', ascending=False).to_string())
