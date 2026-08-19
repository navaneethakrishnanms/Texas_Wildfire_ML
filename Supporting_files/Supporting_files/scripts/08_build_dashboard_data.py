"""
TDIS Forecast — Step 8: build compact JSON for the two-layer dashboard.

Layer 1 — WILDFIRE RISK (static hazard, EMPHASIZED, full 1.7M-cell TX coverage):
  composite of WHP (Wildfire Hazard Potential, TxWRAP 0-9) + burn probability (0-11)
  + canopy bulk density. WHP/burn_prob are the canonical TxWRAP hazard metrics.
  (flep4/cfl flame-length are NaN in the full-grid source -> excluded from the full map.)

Layer 2 — IGNITION FORECAST (dynamic model output):
  actual honest-model predictions on the 2023-2026 FUTURE test set (real outputs on
  valid features), aggregated to res-5 by calendar month -> a seasonal ignition-risk
  surface. Distinct from hazard: "will a fire START" vs "how bad if it does".

Aggregated to H3 res-5 (~3.6k cells) for a light, shareable payload; the dashboard
draws hexes client-side via h3-js.

Output: TDIS_Forecast/dashboard/tdis_dashboard_data.json
"""
import json, warnings
import numpy as np, pandas as pd, h3
from pathlib import Path
warnings.filterwarnings('ignore')
TF = Path(__file__).resolve().parent.parent
OUT = TF/"dashboard"; OUT.mkdir(exist_ok=True)
def log(m): print(m, flush=True)

# ── Layer 1: wildfire hazard (full grid) ──
log("Loading static master (1.7M cells) for wildfire hazard...")
m = pd.read_parquet(TF/"data"/"static_features"/"tx_static_master.parquet",
                    columns=['h3_cell','lat','lon','ecoregion_name','whp','avg_burn_prob','cbd'])
whp_n = (m['whp']/9).clip(0,1)
bp_n  = (m['avg_burn_prob']/11).clip(0,1)
cbd_n = (m['cbd']/0.20).clip(0,1)
m['hazard'] = (0.45*whp_n + 0.40*bp_n + 0.15*cbd_n)          # 0-1, emphasizes WHP + burn prob
log("  computing res-5 parents...")
m['h3_5'] = [h3.cell_to_parent(c,5) for c in m['h3_cell'].values]
haz5 = m.groupby('h3_5').agg(
    lat=('lat','mean'), lon=('lon','mean'),
    eco=('ecoregion_name','first'),
    hazard=('hazard','mean'), whp=('whp','mean'),
    burn_prob=('avg_burn_prob','mean'), cbd=('cbd','mean'),
).reset_index()
log(f"  wildfire-hazard res-5 cells: {len(haz5):,}")

# ── Layer 2: ignition forecast (test-set model predictions) ──
log("Loading honest-model future-test predictions for ignition layer...")
pred = pd.read_parquet(TF/"models"/"baseline_forecast_predictions_test.parquet")
pred['month'] = pd.to_datetime(pred['date']).dt.month
pred['h3_5'] = [h3.cell_to_parent(c,5) for c in pred['h3_cell'].values]
ign_month = pred.groupby(['h3_5','month'])['p'].mean().unstack().reindex(columns=range(1,13))
ign_overall = pred.groupby('h3_5')['p'].mean()
log(f"  ignition res-5 cells (test coverage): {len(ign_overall):,}")

# ── merge into per-cell records (hazard is the full universe; ignition where available) ──
cells = []
ign_month = ign_month.round(4)
for _, r in haz5.iterrows():
    cid = r['h3_5']
    rec = {'id': cid, 'lat': round(float(r['lat']),4), 'lon': round(float(r['lon']),4),
           'eco': str(r['eco']) if pd.notna(r['eco']) else 'Unknown',
           'haz': round(float(r['hazard']),4),
           'whp': round(float(r['whp']),2), 'bp': round(float(r['burn_prob']),2),
           'cbd': round(float(r['cbd']),4)}
    if cid in ign_overall.index:
        rec['ign'] = round(float(ign_overall[cid]),4)
        mrow = ign_month.loc[cid] if cid in ign_month.index else None
        rec['ignm'] = [None if (mrow is None or pd.isna(mrow[mo])) else round(float(mrow[mo]),4)
                       for mo in range(1,13)]
    cells.append(rec)

out = {'cells': cells,
       'meta': {'n_cells': len(cells),
                'n_ignition_cells': int(ign_overall.shape[0]),
                'hazard_formula': '0.45*WHP/9 + 0.40*burn_prob/11 + 0.15*CBD/0.20',
                'ignition_source': 'honest forecast model, 2023-2026 future test, monthly mean',
                'note': 'Wildfire hazard = static TxWRAP-based, full TX. Ignition = dynamic forecast, test-cell coverage.'}}
json.dump(out, open(OUT/"tdis_dashboard_data.json",'w'), separators=(',',':'))
sz = (OUT/"tdis_dashboard_data.json").stat().st_size/1e6
log(f"\nSaved dashboard data: {OUT/'tdis_dashboard_data.json'} ({sz:.1f} MB, {len(cells):,} cells)")
log(f"  cells with ignition forecast: {sum('ign' in c for c in cells):,}")
