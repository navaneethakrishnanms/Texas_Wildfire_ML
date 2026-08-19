"""
TDIS Forecast — Step 6: attach HRRR *forecast* weather to the training table.

For each training cell-day, look up the HRRR F24 forecast (what the forecast SAID 24h
ahead) valid that day, at the nearest HRRR grid point. This is the "forecast-realistic"
weather that makes a true forecast model (no observed→forecast train/serve mismatch).

Method (memory-safe, single pass over dates):
  - map each training cell centroid -> nearest HRRR grid point ONCE (KDTree)
  - for each date with an HRRR file, merge that day's forecast onto its rows
HRRR F24 exists only from 2018-07-16 -> earlier dates get NaN (XGBoost handles natively).

Output: tdis_train_daily_hrrr.parquet  (training table + hrrr_tmp/hrrr_vpd/hrrr_wind)
"""
import warnings, time
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')
TF = Path(__file__).resolve().parent.parent
HRRR_DIR = TF/"data"/"weather_hrrr_forecast"/"hrrr_24h"
def log(m): print(m, flush=True)

log("Loading training table...")
tr = pd.read_parquet(TF/"tdis_train_daily_tx.parquet")
tr['date'] = pd.to_datetime(tr['date']).dt.normalize()
log(f"  {len(tr):,} rows")

log("Loading cell centroids (static master)...")
master = pd.read_parquet(TF/"data"/"static_features"/"tx_static_master.parquet",
                         columns=['h3_cell','lat','lon'])
cll = master.drop_duplicates('h3_cell').set_index('h3_cell')

log("Mapping each cell -> nearest HRRR grid point (once)...")
sample = pd.read_parquet(sorted(HRRR_DIR.glob('*.parquet'))[0], columns=['lat','lon'])
tree = cKDTree(sample[['lat','lon']].values)
ucells = pd.Index(tr['h3_cell'].unique())
uc = cll.reindex(ucells).dropna()
_, idx = tree.query(uc[['lat','lon']].values)
uc = uc.copy()
uc['glat'] = sample['lat'].values[idx].round(3)
uc['glon'] = sample['lon'].values[idx].round(3)
cell2grid = uc[['glat','glon']]
tr = tr.merge(cell2grid, left_on='h3_cell', right_index=True, how='left')
log(f"  mapped {cell2grid.shape[0]:,} unique cells to HRRR grid")

log("Attaching HRRR forecast per date (2018-07-16 onward)...")
HRRR_START = pd.Timestamp('2018-07-16')
out_parts = []
dates = sorted(tr['date'].unique())
t0 = time.time(); nfiles = 0
for i, d in enumerate(dates):
    g = tr[tr['date']==d]
    f = HRRR_DIR/f"{pd.Timestamp(d).date()}.parquet"
    if d < HRRR_START or not f.exists():
        out_parts.append(g); continue
    hf = pd.read_parquet(f)
    if not {'tmp_c','vpd_kpa','wind_ms'} <= set(hf.columns):   # incomplete file -> skip (NaN this date)
        log(f"  [skip] {f.name} missing cols {sorted(set(hf.columns))} — leaving NaN")
        out_parts.append(g); continue
    hf['glat'] = hf['lat'].round(3); hf['glon'] = hf['lon'].round(3)
    lut = hf.groupby(['glat','glon']).agg(
        hrrr_tmp=('tmp_c','mean'), hrrr_vpd=('vpd_kpa','mean'), hrrr_wind=('wind_ms','mean')).reset_index()
    out_parts.append(g.merge(lut, on=['glat','glon'], how='left'))
    nfiles += 1
    if nfiles % 300 == 0:
        log(f"  {nfiles} HRRR dates merged ({(time.time()-t0)/60:.1f} min)")

tr2 = pd.concat(out_parts, ignore_index=True).drop(columns=['glat','glon'], errors='ignore')
for c in ['hrrr_tmp','hrrr_vpd','hrrr_wind']:
    if c not in tr2.columns: tr2[c]=np.nan
cov = tr2['hrrr_vpd'].notna().mean()*100
log(f"  HRRR forecast attached. coverage: {cov:.1f}% of rows have HRRR (rest predate 2018-07 or missing)")
out = TF/"tdis_train_daily_hrrr.parquet"
tr2.to_parquet(out, index=False)
log(f"Saved {out.name}: {len(tr2):,} rows, {len(tr2.columns)} cols")
