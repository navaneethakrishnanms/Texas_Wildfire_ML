"""
TDIS Forecast — Step 15: INSTANT rebuild of the fire-weather arrays (all 3 variants).

From the cached components (data/fwi_components_res5.parquet, built by 09) this
computes THREE per-day fire-weather arrays per res-5 cell — no gridMET reload:

  fwi   composite  (additive, FWI_WEIGHTS knob)
  fwiN  NOAA HWP   (James et al. 2025 Eq. 3, gridMET proxies: G=1.5*wind, M=fm100/30)
  fwiT  TX HWP     (same form, coefficients fit to TX VIIRS FRP by 16_fit_hwp_tx.py)

The dashboard's "Fire-weather" toggle switches between them client-side.
Edit knobs in scripts/fwi_config.py (or re-run 16 to re-fit), then run this.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path
import fwi_config
TF=Path(__file__).resolve().parent.parent
def log(m):print(m,flush=True)

comp=pd.read_parquet(TF/"data"/"fwi_components_res5.parquet")
comp['date']=pd.to_datetime(comp['date'])
has_fm=('fm100' in comp.columns)
if not has_fm: log("WARNING: no fm100 in cache (re-run 09) — HWP variants skipped")

variants={'fwi': fwi_config.fwi_composite_gridmet(comp['erc'].values,comp['vpd'].values,comp['vs'].values)}
if has_fm:
    variants['fwiN']=fwi_config.hwp_gridmet(comp['vpd'].values,comp['vs'].values,comp['fm100'].values,variant='noaa')
    variants['fwiT']=fwi_config.hwp_gridmet(comp['vpd'].values,comp['vs'].values,comp['fm100'].values,variant='tx')

dates=sorted(comp['date'].unique()); didx={d:i for i,d in enumerate(dates)}
comp['di']=comp['date'].map(didx)
by={k:{} for k in variants}
gidx=comp.groupby('h3_5').indices
for k,vals in variants.items():
    for cid,idx in gidx.items():
        a=[None]*len(dates)
        for di,v in zip(comp['di'].values[idx], vals[idx]): a[int(di)]=round(float(v),3)
        by[k][cid]=a

dash=json.load(open(TF/"dashboard"/"tdis_dashboard_data.json"))
n=0
for c in dash['cells']:
    hit=False
    for k in by:
        if c['id'] in by[k]: c[k]=by[k][c['id']]; hit=True
        elif k in c: del c[k]
    n+=hit
dash['dates']=[d.strftime('%Y-%m-%d') for d in dates]
m=dash['meta']
m['fwi_weights']=fwi_config.FWI_WEIGHTS
m['fwi_formula']=f"composite = {fwi_config.FWI_WEIGHTS['erc']}*ERC/100 + {fwi_config.FWI_WEIGHTS['vpd']}*VPD/5 + {fwi_config.FWI_WEIGHTS['wind']}*wind/12 (gridMET daily)"
m['fwi_variants']={
  'fwi':'TDIS composite (additive)',
  'fwiN':'NOAA HWP — James et al. 2025 Eq.3 (0.213*G^1.50*VPD^0.73*(1-M)^5.10), gridMET proxies, p99.5-normalized',
  'fwiT':'TX-calibrated HWP — same form, coefficients fit to TX VIIRS FRP (16_fit_hwp_tx.py)'}
m['hwp_params']=fwi_config.HWP_PARAMS
json.dump(dash,open(TF/"dashboard"/"tdis_dashboard_data.json",'w'),separators=(',',':'))
sz=(TF/"dashboard"/"tdis_dashboard_data.json").stat().st_size/1e6
log(f"Rebuilt {list(by)} on {n:,} cells x {len(dates)} days ({sz:.1f} MB). "
    f"weights={fwi_config.FWI_WEIGHTS} hwp_tx={ {k:round(v,3) for k,v in fwi_config.HWP_PARAMS['tx'].items() if isinstance(v,float)} }")
