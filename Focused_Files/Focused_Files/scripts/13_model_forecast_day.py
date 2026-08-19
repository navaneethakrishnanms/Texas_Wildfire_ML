"""
TDIS Forecast — Step 13: a real MODEL-BASED forecast for a target day.

Runs the trained HRRR ignition model (models/tdis_forecast_hrrr.json) on every TX cell
using that day's live forecast weather -> a genuine operational ignition forecast (not the
hazard×FWI heuristic). Also outputs the fire-weather index for the wildfire-risk layer.

HRRR for lead<=48h; GFS for lead>48h (e.g., 72h). Aggregated res-8 -> res-5 for the dashboard.

Output: dashboard/forecast_<target>.json = {target, lead_h, source, ign:{res5:v}, fwi:{res5:v}}

Run: python 13_model_forecast_day.py 2026-08-01 24
"""
import sys, json, warnings
import numpy as np, pandas as pd, h3, xgboost as xgb
from pathlib import Path
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')
TF=Path(__file__).resolve().parent.parent
DASH=TF/"dashboard"
def log(m):print(m,flush=True)

TARGET=pd.Timestamp(sys.argv[1] if len(sys.argv)>1 else "2026-08-01")
LEAD=int(sys.argv[2]) if len(sys.argv)>2 else 24
TX=dict(lat_min=25.75,lat_max=36.65,lon_min=-106.70,lon_max=-93.40)
import fwi_config
def vpd_from(t,td):
    es=0.6108*np.exp(17.27*t/(t+237.3));ea=0.6108*np.exp(17.27*td/(td+237.3));return np.clip(es-ea,0,None)

# ── pull forecast grid ──
from herbie import Herbie
init=TARGET-pd.Timedelta(hours=LEAD)
model,prod=("hrrr","sfc") if LEAD<=48 else ("gfs","pgrb2.0p25")
log(f"Pulling {model.upper()} F{LEAD}: target {TARGET.date()} init {init.date()}")
h=Herbie(init.strftime("%Y-%m-%d %H:%M"),model=model,product=prod,fxx=LEAD,verbose=False)
if h.grib is None: log("forecast not found"); sys.exit(1)
if model=="hrrr":
    dsl=h.xarray(":(TMP|DPT):2 m above ground|:WIND:10 m above ground|:GUST:surface|:MSTAV:",remove_grib=True)
else:
    dsl=h.xarray(":(TMP|DPT):2 m above ground|:(UGRD|VGRD):10 m above ground|:SOILW:0-0.1 m below ground",remove_grib=True)
if not isinstance(dsl,list): dsl=[dsl]
mg={}
for ds in dsl:
    la=ds.latitude.values; lo=ds.longitude.values; lo=np.where(lo>180,lo-360,lo)
    # HRRR: 2D lat/lon (y,x). GFS: 1D axes -> meshgrid to match the 2D data arrays.
    if la.ndim==1 and lo.ndim==1:
        LO,LA=np.meshgrid(lo,la); la=LA.ravel(); lo=LO.ravel()
    else:
        la=la.ravel(); lo=lo.ravel()
    msk=(la>=TX['lat_min'])&(la<=TX['lat_max'])&(lo>=TX['lon_min'])&(lo<=TX['lon_max'])
    mg.setdefault('lat',la[msk]); mg.setdefault('lon',lo[msk])
    for nm,da in ds.data_vars.items():
        v=da.values.ravel()[msk]; mg[nm]=v
g=pd.DataFrame(mg).rename(columns={'t2m':'tmp','d2m':'dpt','max_10si':'wind','si10':'wind','gust':'gust',
                                   'mstav':'soilm','avail_smois':'soilm','soilw':'soilw'})
g['tmp']-=273.15; g['dpt']-=273.15
if 'wind' not in g.columns and {'u10','v10'}<=set(g.columns):
    g['wind']=np.sqrt(g['u10']**2+g['v10']**2)
g['vpd']=vpd_from(g['tmp'].values,g['dpt'].values)
gustv = g['gust'].values if 'gust' in g.columns else None   # HRRR has gust; GFS doesn't
# soil-moisture availability M (0-1): HRRR MSTAV is %, GFS SOILW is volumetric (/0.45 porosity)
if 'soilm' in g.columns:   soilmv=np.clip(g['soilm'].values/100.0,0,1)
elif 'soilw' in g.columns: soilmv=np.clip(g['soilw'].values/0.45,0,1)
else:                      soilmv=None
G_eff = gustv if gustv is not None else fwi_config.GUST_FACTOR*g['wind'].values
g['fwi']=fwi_config.fwi_composite_hrrr(g['vpd'].values, g['wind'].values, gust=gustv)
if soilmv is not None:
    g['fwiN']=fwi_config.hwp_hrrr(g['vpd'].values, G_eff, soilmv, variant='noaa')
    g['fwiT']=fwi_config.hwp_hrrr(g['vpd'].values, G_eff, soilmv, variant='tx')
else:
    g['fwiN']=g['fwi']; g['fwiT']=g['fwi']   # no soil field -> fall back to composite
    log("  (no soil-moisture field — HWP variants fall back to composite)")
log(f"  {len(g):,} grid points; vpd {g.vpd.min():.2f}-{g.vpd.max():.2f}; soilm={'yes' if soilmv is not None else 'no'}")

# ── static master (res-8) + model ──
log("Loading static master + HRRR model...")
_mp = TF/"models"/"tdis_forecast_hrrr_filtered.json"          # flare-filtered (preferred)
if not _mp.exists(): _mp = TF/"models"/"tdis_forecast_hrrr.json"
mdl=xgb.XGBClassifier(); mdl.load_model(str(_mp))
log(f"model: {_mp.name}")
FEATS=mdl.get_booster().feature_names
st=pd.read_parquet(TF/"data"/"static_features"/"tx_static_master.parquet")
# powerline_dist_km: static feature added 2026-08-17 (HIFLD TX transmission lines)
_pl_p = TF/"data"/"static_features"/"powerline_dist_km.parquet"
if 'powerline_dist_km' in FEATS:
    st = st.merge(pd.read_parquet(_pl_p), on='h3_cell', how='left')
# nearest forecast grid point per cell
tree=cKDTree(np.c_[g['lat'].values,g['lon'].values])
_,idx=tree.query(np.c_[st['lat'].values,st['lon'].values])
st['hrrr_tmp']=g['tmp'].values[idx]; st['hrrr_vpd']=g['vpd'].values[idx]; st['hrrr_wind']=g['wind'].values[idx]
# hrrr_mstav: soil-moisture feature added 2026-08-17 (same MSTAV field HWP uses;
# GFS fallback path (lead>48h) uses SOILW-derived soilmv; if neither, fill 0.5)
if 'hrrr_mstav' in FEATS:
    st['hrrr_mstav'] = np.clip(soilmv[idx],0,1) if soilmv is not None else 0.5
st['_fwi']=g['fwi'].values[idx]; st['_fwiN']=g['fwiN'].values[idx]; st['_fwiT']=g['fwiT'].values[idx]
# temporal features for target date
mo=TARGET.month; dow=TARGET.dayofweek
st['sin_month']=np.sin(2*np.pi*mo/12); st['cos_month']=np.cos(2*np.pi*mo/12)
st['sin_dow']=np.sin(2*np.pi*dow/7); st['cos_dow']=np.cos(2*np.pi*dow/7)
st['is_weekend']=int(dow>=5); st['is_holiday']=int((mo,TARGET.day) in {(1,1),(7,4),(6,19),(11,11),(12,25),(10,31)})
# predict ignition per res-8 cell
log("Running model on 1.7M cells...")
st['ign']=mdl.predict_proba(st[FEATS])[:,1]

# calibrated probability (isotonic, fit on 2024-25 real outcomes, validated on 2026
# holdout: ECE 0.327 -> 0.0018 -- scripts 26/27). The raw score is a RANK, not a
# probability (trained on a rebalanced sample, ~17x overstated); ign_cal is the
# value safe to read as "chance of a fire-day here". Calibrator was fit on res-5
# scores; applying to res-8 scores is an approximation (same model, same scale).
_cal_p = TF/"models"/"operational_isotonic_calibrator.joblib"
if _cal_p.exists():
    import joblib
    _iso = joblib.load(_cal_p)
    st['ign_cal'] = _iso.predict(st['ign'].values)
    log(f"calibrated: mean raw={st['ign'].mean():.3f} -> mean cal={st['ign_cal'].mean():.4f}")
else:
    st['ign_cal'] = np.nan
    log("WARNING: no isotonic calibrator found -- ign_cal omitted (run scripts 26+27)")

# save native res-8 output (parquet, not JSON -- 1.7M cells as JSON would be
# ~137MB per snapshot, bigger than the entire multi-year res-5 history file;
# parquet is ~48MB and is what the v2 explorer's tiling pipeline should consume,
# not a single browser-loaded blob)
res8_path = DASH/f"forecast_{TARGET.date()}_lead{LEAD}h_res8.parquet"
st[['h3_cell','lat','lon','ign','ign_cal','_fwi','_fwiN','_fwiT']].rename(
    columns={'_fwi':'fwi','_fwiN':'fwiN','_fwiT':'fwiT'}
).to_parquet(res8_path, index=False)
log(f"Saved native res-8 forecast -> {res8_path.name} "
    f"({res8_path.stat().st_size/1e6:.1f} MB, {len(st):,} cells)")

# aggregate res-8 -> res-5 (kept for the existing dashboards' lightweight JSON path)
st['h3_5']=[h3.cell_to_parent(c,5) for c in st['h3_cell'].values]
agg=st.groupby('h3_5').agg(ign=('ign','mean'), ign_cal=('ign_cal','mean'), fwi=('_fwi','mean'),
                           fwiN=('_fwiN','mean'), fwiT=('_fwiT','mean')).reset_index()
out={'target':str(TARGET.date()),'lead_h':LEAD,'init':str(init.date()),
     'source':f'{model.upper()} F{LEAD} (live model forecast)',
     'ign':{r.h3_5:round(float(r.ign),4) for r in agg.itertuples()},
     'ignCal':{r.h3_5:round(float(r.ign_cal),4) for r in agg.itertuples()},
     'fwi':{r.h3_5:round(float(r.fwi),4) for r in agg.itertuples()},
     'fwiN':{r.h3_5:round(float(r.fwiN),4) for r in agg.itertuples()},
     'fwiT':{r.h3_5:round(float(r.fwiT),4) for r in agg.itertuples()}}
p=DASH/f"forecast_{TARGET.date()}.json"; json.dump(out,open(p,'w'),separators=(',',':'))
log(f"Saved {p.name}: model ign + fwi for {len(agg):,} res-5 cells. ign {agg.ign.min():.3f}-{agg.ign.max():.3f}")
