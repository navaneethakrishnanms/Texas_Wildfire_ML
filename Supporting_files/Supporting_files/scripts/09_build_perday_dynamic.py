"""
TDIS Forecast — Step 9: PER-DAY dynamic fire-weather layers, MULTI-YEAR.

Builds a daily FIRE-WEATHER index (FWI) per res-5 cell across YEARS (default 2024-2026)
from gridMET (ERC + VPD + wind). The dashboard computes, per selected day:
  Wildfire risk(day) = static hazard(cell)          x FWI(cell, day)
  Ignition     (day) = ML ignition-susceptibility(cell) x FWI(cell, day)

Continuous multi-year timeline -> the scrubber spans all days of all YEARS (2026 ends
whenever gridMET observed data ends, ~mid-year). Reuses the hazard + ignition-baseline
already in the dashboard JSON and (re)writes the 'fwi' arrays + 'dates' axis.

Output: overwrites dashboard/tdis_dashboard_data.json
"""
import json, warnings
import numpy as np, pandas as pd, h3
from pathlib import Path
warnings.filterwarnings('ignore')
TF = Path(__file__).resolve().parent.parent
GRIDMET = Path("/net/cven-mosta-nas.engr.tamu.edu/volume2/NASdata2/miguel_shared/alphaearth_nds/gridmet_tx")  # external dep, see REPRODUCE.md
DASH = TF/"dashboard"/"tdis_dashboard_data.json"
def log(m): print(m, flush=True)

import fwi_config
YEARS = [2024, 2025, 2026]

all_comp = []   # per-year res-5 component means: h3_5, date, erc, vpd, vs, fm100
for yr in YEARS:
    log(f"Loading gridMET {yr} (erc, vpd, vs, fm100)...")
    parts=[]
    for var in ['erc','vpd','vs','fm100']:
        d = pd.read_parquet(GRIDMET/f"{var}_{yr}_tx_cells.parquet")
        d['date'] = pd.to_datetime(d['date_utc']).dt.normalize()
        parts.append(d.set_index(['h3_cell','date'])[var])
    wx = pd.concat(parts, axis=1).reset_index()
    wx['h3_5'] = [h3.cell_to_parent(c,5) for c in wx['h3_cell'].values]
    c5 = wx.groupby(['h3_5','date']).agg(erc=('erc','mean'),vpd=('vpd','mean'),vs=('vs','mean'),fm100=('fm100','mean')).reset_index()
    all_comp.append(c5)
    log(f"  {yr}: {c5['date'].dt.normalize().nunique()} days, {c5['h3_5'].nunique():,} res-5 cells")
    del wx, parts

comp = pd.concat(all_comp, ignore_index=True)
# store res-5 weather COMPONENTS so the wind knob can be re-tuned instantly (no gridMET reload)
comp.to_parquet(TF/"data"/"fwi_components_res5.parquet", index=False)
log(f"Saved fwi_components_res5.parquet ({len(comp):,} cell-days) for instant re-tuning")
# FWI from the tunable knob
comp['fwi'] = fwi_config.fwi_gridmet(comp['erc'].values, comp['vpd'].values, comp['vs'].values)
fwi5 = comp[['h3_5','date','fwi']].copy()
dates = sorted(fwi5['date'].unique())
date_idx = {d:i for i,d in enumerate(dates)}
log(f"Combined timeline: {len(dates)} days ({dates[0].date()} -> {dates[-1].date()})")

fwi5['di'] = fwi5['date'].map(date_idx)
fwi_by_cell = {}
for cid, g in fwi5.groupby('h3_5'):
    arr=[None]*len(dates)
    for di,val in zip(g['di'].values, g['fwi'].values): arr[int(di)]=round(float(val),4)
    fwi_by_cell[cid]=arr

log("Merging into dashboard JSON...")
dash = json.load(open(DASH))
# drop any prior single-year fwi, attach fresh multi-year
n=0
for c in dash['cells']:
    if c['id'] in fwi_by_cell:
        c['fwi']=fwi_by_cell[c['id']]; n+=1
    elif 'fwi' in c:
        del c['fwi']
dash['dates']=[d.strftime('%Y-%m-%d') for d in dates]
dash['meta']['perday_years']=YEARS
dash['meta']['perday_range']=f"{dates[0].date()} to {dates[-1].date()}"
json.dump(dash, open(DASH,'w'), separators=(',',':'))
sz=DASH.stat().st_size/1e6
log(f"Saved {DASH.name} ({sz:.1f} MB). fwi on {n:,} cells x {len(dates)} days.")
