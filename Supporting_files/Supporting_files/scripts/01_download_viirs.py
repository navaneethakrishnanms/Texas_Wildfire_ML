"""
TDIS Forecast — Step 1 (driver): download VIIRS active fire for Texas, NEWEST YEAR FIRST,
2014-2026, real timestamps. Uses SP (archive) + NRT (recent ~2 months) so 2025-2026 are
covered. After the FIRST year finishes, auto-builds a STARTER dataset so the model +
dashboard can be tried while the rest downloads. Resume-safe, throttled, rate-limit aware.

Forecast-oriented: recent years land first. Path to future years = the NRT sources here +
a daily cron (see HANDOFF.md).
"""
import os, time, sys, subprocess
import numpy as np, pandas as pd, requests, h3
from pathlib import Path

TF   = Path(__file__).resolve().parent.parent
RAW  = TF/"data"/"labels_viirs"/"raw"; RAW.mkdir(parents=True, exist_ok=True)
OUT  = TF/"data"/"labels_viirs"
SCR  = TF/"scripts"
KEY  = (TF/"firms_map_key.txt").read_text().strip()
PY   = "/home/mte1224/mambaforge/envs/UAI2526/bin/python"
TX   = "-106.70,25.75,-93.40,36.65"
STATUS_URL = f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={KEY}"
H3_RES, MAX_DAYS = 8, 5

YEARS = list(range(2026, 2013, -1))          # NEWEST FIRST
SP_START  = {"VIIRS_SNPP_SP":2014, "VIIRS_NOAA20_SP":2018}
NRT_SRC   = ["VIIRS_SNPP_NRT","VIIRS_NOAA20_NRT"]   # recent ~2 months only
NRT_MIN_YEAR = 2025
REBUILD_AFTER_ITER = {1, len(YEARS)}         # starter after 1st year; full at end
TODAY = pd.Timestamp.today().normalize()

def log(m): print(m, flush=True)

def sources_for(y):
    s=[k for k,y0 in SP_START.items() if y>=y0]
    if y>=NRT_MIN_YEAR: s+=NRT_SRC
    return s

def check_budget():
    try:
        st=requests.get(STATUS_URL,timeout=30).json()
        cur,lim=st.get("current_transactions",0),st.get("transaction_limit",5000)
        if cur>lim*0.8:
            log(f"  [budget] {cur}/{lim} — sleeping 10 min"); time.sleep(610)
        return cur
    except Exception: return None

def windows(year):
    d=pd.Timestamp(f"{year}-01-01")
    end=min(pd.Timestamp(f"{year}-12-31"), TODAY)
    while d<=end:
        n=min(MAX_DAYS,(end-d).days+1)
        yield d.strftime("%Y-%m-%d"), n
        d+=pd.Timedelta(days=n)

def fetch(source,start,ndays):
    out=RAW/f"{source}_{start}_{ndays}d.csv"
    if out.exists(): return
    url=f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/{source}/{TX}/{ndays}/{start}"
    for att in range(5):
        try:
            r=requests.get(url,timeout=120); t=r.text; tl=t.lower()
            if tl.lstrip().startswith("latitude") or t.strip()=="":
                out.write_text(t); return
            if "invalid" in tl[:60]:
                log(f"  [{source} {start}] permanent err: {t[:70]!r} skip"); return
            if any(w in tl[:120] for w in ["rate","limit","exceed"]):
                log(f"  [{source} {start}] throttled backoff {att+1}"); time.sleep(90*(att+1)); continue
            out.write_text(t); return
        except Exception as e:
            log(f"  [{source} {start}] {e} retry {att+1}"); time.sleep(15*(att+1))
    log(f"  [{source} {start}] FAILED (left for re-run)")

def download_year(y):
    call=0
    for src in sources_for(y):
        for start,n in windows(y):
            if (RAW/f"{src}_{start}_{n}d.csv").exists(): continue
            fetch(src,start,n); call+=1; time.sleep(0.8)
            if call%150==0: check_budget()
    return call

def assign_all():
    """Re-assign ALL downloaded raw CSVs -> cumulative viirs_tx_h3.parquet."""
    frames=[]
    for f in sorted(RAW.glob("*.csv")):
        try: df=pd.read_csv(f)
        except Exception: continue
        if len(df)==0 or 'latitude' not in df.columns: continue
        frames.append(df)
    if not frames: return 0
    v=pd.concat(frames,ignore_index=True)
    v['acq_time']=v['acq_time'].astype(str).str.zfill(4)
    v['hour']=v['acq_time'].str[:2].astype(int).clip(0,23)
    v['date']=pd.to_datetime(v['acq_date'])
    v['dt_utc']=v['date']+pd.to_timedelta(v['hour'],unit='h')
    v['window_hour']=(v['hour']//6)*6
    v['h3_cell']=[h3.latlng_to_cell(la,lo,H3_RES) for la,lo in zip(v['latitude'],v['longitude'])]
    cols=[c for c in ['h3_cell','date','dt_utc','hour','window_hour','latitude','longitude','confidence','frp'] if c in v.columns]
    v=v[cols].drop_duplicates()
    v.to_parquet(OUT/"viirs_tx_h3.parquet",index=False)
    return len(v)

def rebuild_dataset(tag):
    log(f"  [rebuild:{tag}] assigning VIIRS + fusing labels + building dataset...")
    n=assign_all(); log(f"  [rebuild:{tag}] cumulative VIIRS detections: {n:,}")
    subprocess.run([PY,"-u",str(SCR/"02_build_labels.py")],check=False)
    subprocess.run([PY,"-u",str(SCR/"03_build_dataset.py")],check=False)
    (TF/f"DATASET_READY_{tag}").write_text(str(pd.Timestamp.now()))
    log(f"  [rebuild:{tag}] dataset saved -> tdis_train_daily_tx.parquet")

if __name__=="__main__":
    check_budget()
    log(f"STEP 1 — VIIRS download, NEWEST FIRST {YEARS[0]}..{YEARS[-1]} (through {TODAY.date()})")
    for i,y in enumerate(YEARS,1):
        log(f">> year {y}  (iter {i}/{len(YEARS)}, sources: {sources_for(y)})")
        c=download_year(y)
        (TF/f"VIIRS_YEAR_{y}_DONE").write_text(str(pd.Timestamp.now()))
        log(f"   year {y} done ({c} new calls)")
        if i in REBUILD_AFTER_ITER:
            tag="starter" if i==1 else "full"
            rebuild_dataset(tag)
            if i==1:
                log("   >>> STARTER DATASET READY — you can train + map now while the rest downloads.")
    log("STEP 1 complete (all years).")
