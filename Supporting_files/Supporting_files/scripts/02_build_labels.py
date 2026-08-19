"""
TDIS Forecast — Step 2: fuse FPA-FOD (2014-2020, complete inventory, imputed times)
with VIIRS (2014-2024, real times, extends past 2020) into a DAILY ignition inventory.

Output: data/labels_fused/ignitions_daily_tx.parquet
  one row per (h3_cell, date) that had >=1 ignition; columns:
  h3_cell, date, label=1, time_source, cause, max_size_acres, n_sources
This is the positive set. Negatives are sampled in Step 3.
"""
import pandas as pd, numpy as np
from pathlib import Path
TF  = Path(__file__).resolve().parent.parent
PAR = Path("/net/cven-mosta-nas.engr.tamu.edu/volume2/NASdata2/miguel_shared/alphaearth_nds")  # external dep, see REPRODUCE.md
OUT = TF/"data"/"labels_fused"; OUT.mkdir(parents=True, exist_ok=True)
def log(m): print(m, flush=True)

# ── FPA-FOD (already H3-gridded, 2014-2020) ──
log("Loading FPA-FOD TX...")
fpa = pd.read_parquet(PAR/'90%_ig_dec/Transferibility_OutOfState/data/TX/ignition/fpa_fod_tx_h3.parquet')
fpa = fpa[fpa['label']==1].copy()
fpa['date'] = pd.to_datetime(fpa['window_6h_utc']).dt.normalize()
fpa_daily = fpa.groupby(['h3_cell','date']).agg(
    cause=('cause_class','first'),
    max_size_acres=('max_size_acres','max'),
).reset_index()
fpa_daily['src_fpa']=1
log(f"  FPA-FOD daily cell-days: {len(fpa_daily):,}  ({fpa_daily.date.dt.year.min().astype(int)}-{fpa_daily.date.dt.year.max().astype(int)})")

# ── VIIRS (real times, 2014-2024) ──
vp = TF/"data"/"labels_viirs"/"viirs_tx_h3.parquet"
if vp.exists():
    log("Loading VIIRS detections...")
    v = pd.read_parquet(vp)
    v['date']=pd.to_datetime(v['date']).dt.normalize()
    viirs_daily = v.groupby(['h3_cell','date']).agg(
        first_hour=('hour','min'), n_det=('hour','size'),
        frp_max=('frp','max') if 'frp' in v.columns else ('hour','size'),
    ).reset_index()
    viirs_daily['src_viirs']=1
    log(f"  VIIRS daily cell-days: {len(viirs_daily):,}  ({viirs_daily.date.dt.year.min().astype(int)}-{viirs_daily.date.dt.year.max().astype(int)})")
else:
    log("  VIIRS parquet missing — labels will be FPA-FOD only (run step 1 first)")
    viirs_daily = pd.DataFrame(columns=['h3_cell','date','first_hour','n_det','src_viirs'])

# ── FUSE at daily resolution ──
log("Fusing...")
m = fpa_daily.merge(viirs_daily, on=['h3_cell','date'], how='outer')
m['src_fpa']=m['src_fpa'].fillna(0); m['src_viirs']=m.get('src_viirs',0)
m['src_viirs']=m['src_viirs'].fillna(0)
m['n_sources']=(m['src_fpa']+m['src_viirs']).astype(int)
# time_source: viirs = real time available; else imputed (FPA only)
m['time_source']=np.where(m['src_viirs']==1,'viirs','imputed')
m['cause']=m['cause'].fillna('satellite_only')
m['label']=1
out = m[['h3_cell','date','label','time_source','cause','max_size_acres','n_sources']].copy()
# SANITY_CHECK F1 fix: the VIIRS download bbox is a rectangle around TX and captures
# NM/OK spillover (~6% of positives, incl. Ruidoso South Fork). Keep only cells inside
# the Texas static master (built from the TX polygon).
_tx = set(pd.read_parquet(TF/"data"/"static_features"/"tx_static_master.parquet",
                          columns=['h3_cell'])['h3_cell'])
n0 = len(out); out = out[out.h3_cell.isin(_tx)]
log(f"  TX-polygon filter: {n0:,} -> {len(out):,} (dropped {n0-len(out):,} out-of-state)")
out.to_parquet(OUT/"ignitions_daily_tx.parquet", index=False)
log(f"  fused ignition cell-days: {len(out):,}")
log(f"  by year:\n{out.groupby(out.date.dt.year).size().to_string()}")
log(f"  time_source: {out.time_source.value_counts().to_dict()}")
log(f"  >>> post-2020 labels (from VIIRS, enables forward validation): "
    f"{(out.date.dt.year>=2021).sum():,}")
log("STEP 2 complete.")
