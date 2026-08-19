import pandas as pd, numpy as np, json
from pathlib import Path
TF=str(Path(__file__).resolve().parent.parent)  # package-relative (was hardcoded to source project)
GM='/net/cven-mosta-nas.engr.tamu.edu/volume2/NASdata2/miguel_shared/alphaearth_nds/gridmet_tx'  # external dep, see REPRODUCE.md
EVENTS=[
 ("Smokehouse Creek",'2024-02-26',35.5,36.3,-101.6,-100.2,"Panhandle, power line/wind, 1.06M ac"),
 ("Crabapple",       '2025-03-15',30.15,30.75,-99.15,-98.55,"Hill Country/Gillespie, wind+low RH, 9.9K ac"),
 ("Lavender",        '2026-02-17',35.1,35.8,-102.9,-101.4,"Panhandle Oldham/Potter, 18.4K ac"),
 ("Hunggate",        '2026-05-14',34.55,35.25,-102.7,-101.4,"Panhandle Randall/Deaf Smith, LIGHTNING, 34K ac"),
]
v=pd.read_parquet(f'{TF}/data/labels_viirs/viirs_tx_h3.parquet'); v['date']=pd.to_datetime(v['date'])
d=json.load(open(f'{TF}/dashboard/tdis_dashboard_data.json')); dates=d['dates']; didx={t:i for i,t in enumerate(dates)}
st=pd.read_parquet(f'{TF}/data/static_features/tx_static_master.parquet',columns=['h3_cell','lat','lon'])
allh=[c['haz'] for c in d['cells']]; statemean=np.mean(allh)
gmc={}
for var in ['erc','vpd','vs']:
    for yr in [2024,2025,2026]:
        gmc[(var,yr)]=None
def gload(var,yr):
    if gmc[(var,yr)] is None:
        g=pd.read_parquet(f'{GM}/{var}_{yr}_tx_cells.parquet'); g['date']=pd.to_datetime(g['date_utc']); gmc[(var,yr)]=g
    return gmc[(var,yr)]

for name,ds,la0,la1,lo0,lo1,desc in EVENTS:
    D=pd.Timestamp(ds); yr=D.year
    print(f"\n{'='*70}\n{name}  ({ds})  — {desc}\n{'='*70}")
    # 1) VIIRS capture
    box=v[(v.latitude.between(la0,la1))&(v.longitude.between(lo0,lo1))]
    win=box[(box.date>=D-pd.Timedelta(days=3))&(box.date<=D+pd.Timedelta(days=12))]
    byday=win.groupby(win.date.dt.date).size()
    print(f"1) VIIRS detections {D.date()}±: total {len(win)}, peak day {byday.idxmax() if len(byday) else 'none'} ({byday.max() if len(byday) else 0})")
    # 2) static hazard WHERE
    boxcells=[c for c in d['cells'] if la0<=c['lat']<=la1 and lo0<=c['lon']<=lo1]
    if boxcells:
        hz=np.mean([c['haz'] for c in boxcells]); pct=np.mean([statemean< c['haz'] for c in boxcells])*100
        print(f"2) Static hazard (WHERE): {hz:.3f} = {hz/statemean:.1f}x state mean, ~{pct:.0f}th pctile")
    # 3) FWI WHEN
    fcells=[c for c in boxcells if 'fwi' in c]
    i=didx.get(ds)
    if fcells and i is not None:
        fwid=np.nanmean([c['fwi'][i] for c in fcells if c['fwi'][i] is not None])
        ann=np.nanmean([np.nanmedian([x for x in c['fwi'] if x is not None]) for c in fcells])
        # percentile rank of that day within the year for these cells
        ranks=[]
        for c in fcells:
            s=np.array([x if x is not None else np.nan for x in c['fwi']])
            if not np.isnan(s[i]): ranks.append(np.nansum(s<s[i])/np.sum(~np.isnan(s))*100)
        print(f"3) Fire-weather index (WHEN): day {fwid:.3f} vs annual median {ann:.3f}  | ~{np.mean(ranks):.0f}th pctile of the year")
    else:
        print("3) Fire-weather index (WHEN): no gridMET coverage for these cells/date")
    # 4) raw weather to explain
    pcells=set(st[(st.lat.between(la0,la1))&(st.lon.between(lo0,lo1))]['h3_cell'])
    for var in ['erc','vpd','vs']:
        g=gload(var,yr); gp=g[g.h3_cell.isin(pcells)]
        if len(gp):
            dm=gp[gp.date==D][var].mean(); annm=gp[var].median(); mx=gp.groupby('date')[var].mean().max()
            print(f"   {var}: day {dm:.1f} | annual median {annm:.1f} | year daily-max {mx:.1f}")
print("\nDONE")
