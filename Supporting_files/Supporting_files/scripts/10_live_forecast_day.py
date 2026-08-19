"""
TDIS Forecast — Step 10: a REAL live forecast for a target day.

Pulls the actual HRRR forecast (init N hours before the target, F=lead) valid for the
TARGET day, computes the daily fire-weather index on HRRR's native grid, nearest-joins
to the res-5 dashboard cells, and writes a per-cell FWI for that day. Combined with the
static hazard / ignition-susceptibility already in the dashboard, this is a genuine
24h (or 48h) forward wildfire-risk + ignition forecast -- not observed weather.

Default target: 2026-07-31 via HRRR F24 from a 2026-07-30 00Z init.

Output: dashboard/forecast_<target>.json  ->  {target, lead_h, source, fwi:{res5_id:val}}
The dashboard loads this as a special "Live forecast" day.

Run: /home/mte1224/mambaforge/envs/UAI2526/bin/python 10_live_forecast_day.py 2026-07-31 24
"""
import sys, json, warnings
import numpy as np, pandas as pd, h3
from pathlib import Path
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')
TF = Path(__file__).resolve().parent.parent
DASH = TF/"dashboard"
def log(m): print(m, flush=True)

TARGET = pd.Timestamp(sys.argv[1] if len(sys.argv)>1 else "2026-07-31")
LEAD   = int(sys.argv[2]) if len(sys.argv)>2 else 24     # HRRR forecast hour (<=48)
TX = dict(lat_min=25.75, lat_max=36.65, lon_min=-106.70, lon_max=-93.40)

def fwi(erc, vpd, vs):
    # HRRR has no ERC; approximate fire-weather from temp/RH/wind (Hot-Dry-Windy style).
    # Here erc arg unused for HRRR; we build FWI from VPD + wind + (1-RH proxy via VPD).
    v = np.clip(vpd/5.0, 0, 1); w = np.clip(vs/12.0, 0, 1)
    return np.clip(0.6*v + 0.4*w, 0, 1)   # dryness + wind (no ERC available in forecast)

def vpd_from(t_c, td_c):
    es=0.6108*np.exp(17.27*t_c/(t_c+237.3)); ea=0.6108*np.exp(17.27*td_c/(td_c+237.3))
    return np.clip(es-ea,0,None)

log(f"Pulling HRRR forecast: target {TARGET.date()}, lead F{LEAD}")
from herbie import Herbie
init = TARGET - pd.Timedelta(hours=LEAD)
h = Herbie(init.strftime("%Y-%m-%d %H:%M"), model='hrrr', product='sfc', fxx=LEAD, verbose=False)
if h.grib is None:
    log(f"  HRRR not found for init {init}. Try a different lead."); sys.exit(1)
ds_list = h.xarray(":(TMP|DPT):2 m above ground|:WIND:10 m above ground", remove_grib=True)
if not isinstance(ds_list, list): ds_list=[ds_list]
merged={}
for ds in ds_list:
    lat=ds.latitude.values; lon=ds.longitude.values; lon=np.where(lon>180,lon-360,lon)
    mask=(lat>=TX['lat_min'])&(lat<=TX['lat_max'])&(lon>=TX['lon_min'])&(lon<=TX['lon_max'])
    for nm,da in ds.data_vars.items():
        merged.setdefault('lat',lat[mask]); merged.setdefault('lon',lon[mask]); merged[nm]=da.values[mask]
g = pd.DataFrame(merged).rename(columns={'t2m':'tmp','d2m':'dpt','max_10si':'wind','si10':'wind'})
g['tmp']-=273.15; g['dpt']-=273.15
g['vpd']=vpd_from(g['tmp'].values,g['dpt'].values)
g['fwi']=fwi(None,g['vpd'].values,g['wind'].values)
log(f"  HRRR grid points in TX: {len(g):,}  fwi range {g['fwi'].min():.3f}-{g['fwi'].max():.3f}")

# nearest-join HRRR grid -> res-5 dashboard cells (use cell centroids from dashboard json)
dash=json.load(open(DASH/"tdis_dashboard_data.json"))
cells=[c for c in dash['cells']]
clat=np.array([c['lat'] for c in cells]); clon=np.array([c['lon'] for c in cells])
tree=cKDTree(np.c_[g['lat'].values, g['lon'].values])
_,idx=tree.query(np.c_[clat,clon])
fwi_vals=g['fwi'].values[idx]
out={'target':str(TARGET.date()),'lead_h':LEAD,'init':str(init),
     'source':f'HRRR F{LEAD} (live forecast)',
     'fwi':{cells[i]['id']:round(float(fwi_vals[i]),4) for i in range(len(cells))}}
p=DASH/f"forecast_{TARGET.date()}.json"
json.dump(out,open(p,'w'),separators=(',',':'))
log(f"Saved {p.name}: real HRRR forecast FWI for {len(cells):,} cells, target {TARGET.date()}")
