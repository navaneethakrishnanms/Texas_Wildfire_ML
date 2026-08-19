"""
TDIS Forecast — Step 20: multi-resolution static layers for the ZOOM-ADAPTIVE dashboard (v2).

Pre-aggregates the two STATIC per-cell layers at H3 res 4..8 so the v2 dashboard can
re-tessellate the viewport at the resolution matching the zoom level (h3geo.org-style):

  haz  = TxWRAP hazard composite (same formula as 08: 0.45*WHP/9 + 0.40*bp/11 + 0.15*CBD/0.20)
  ign  = flare-filtered honest-model ignition susceptibility (mean clean-test score per cell;
         only exists for label-universe cells — null elsewhere)

Dynamic layers stay res-5 in the dashboard; the client computes fine-res risk as
haz(res N cell) x FWI(res-5 ancestor, day) — weather is smooth, statics carry the detail.

Size control: res-8 rows with haz==0 AND no ign are omitted (rendered as neutral grid);
values quantized to 3dp. Output: dashboard/explorer_static_multires.json
  { "4": {h3: [haz, ign|null], ...}, ..., "8": {...}, "meta": {...} }
"""
import json, warnings
import numpy as np, pandas as pd, h3
from pathlib import Path
warnings.filterwarnings('ignore')
TF = Path(__file__).resolve().parent.parent
def log(m): print(m, flush=True)

log("Loading static master (1.7M cells)...")
st = pd.read_parquet(TF/"data/static_features/tx_static_master.parquet",
                     columns=['h3_cell','whp','avg_burn_prob','cbd'])
st['haz'] = np.clip(0.45*st['whp']/9 + 0.40*st['avg_burn_prob']/11 + 0.15*np.clip(st['cbd']/0.20,0,1), 0, 1)

log("Loading filtered honest-model susceptibility (per res-8 cell)...")
pr = pd.read_parquet(TF/"models/baseline_forecast_predictions_test_filtered.parquet",
                     columns=['h3_cell','p'])
ign8 = pr.groupby('h3_cell')['p'].mean().rename('ign')
st = st.merge(ign8, on='h3_cell', how='left')
log(f"  cells: {len(st):,}; with ign: {st['ign'].notna().sum():,}")

out = {}
cur = st[['h3_cell','haz','ign']].copy()
for res in [8,7,6,5,4]:
    if res < 8:
        cur['h3_cell'] = [h3.cell_to_parent(c, res) for c in cur['h3_cell'].values]
        cur = cur.groupby('h3_cell').agg(haz=('haz','mean'), ign=('ign','mean')).reset_index()
    d = cur if res < 8 else cur[(cur.haz > 0) | cur.ign.notna()]   # res-8: drop empty cells
    layer = {}
    for r in d.itertuples():
        ign = None if pd.isna(r.ign) else round(float(r.ign), 3)
        layer[r.h3_cell] = [round(float(r.haz), 3), ign]
    out[str(res)] = layer
    log(f"  res {res}: {len(layer):,} cells")

out['meta'] = {'layers': 'haz (TxWRAP composite) + ign (flare-filtered honest susceptibility)',
               'res8_note': 'zero-hazard/no-ign cells omitted at res 8 (render as neutral grid)',
               'built_from': 'tx_static_master.parquet + baseline_forecast_predictions_test_filtered.parquet'}
p = TF/"dashboard/explorer_static_multires.json"
json.dump(out, open(p,'w'), separators=(',',':'))
log(f"Saved {p.name} ({p.stat().st_size/1e6:.1f} MB)")
