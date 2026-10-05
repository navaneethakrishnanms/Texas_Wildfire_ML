"""
NM_OK_TexRev — Step 5: US Drought Monitor weekly county drought for NM + OK.
Same source/method as TX's rev2_improvements/dataAdditions/idea5_drought_
monitor/download_and_build.py -- county FIPS already mapped in
static_master_{nm,ok}.parquet (33 NM counties, 77 OK counties), no need to
rebuild the point-in-polygon step. Small pull (110 counties total vs TX's 254).

Output: NM_OK_TexRev/data/drought_feature_{nm,ok}.parquet [h3_cell, date, drought_score]
"""
import io, time, warnings
import numpy as np, pandas as pd
import urllib.request
from pathlib import Path
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
START, END = pd.Timestamp('2014-01-01'), pd.Timestamp('2026-09-16')


def log(m): print(m, flush=True)


def download_drought(state, fips_list):
    out = ROOT / "data" / f"drought_weekly_raw_{state}.parquet"
    if out.exists():
        log(f"[{state}] drought raw: cached"); return pd.read_parquet(out)
    frames = []
    for i, fips in enumerate(fips_list):
        url = (f"https://usdmdataservices.unl.edu/api/CountyStatistics/"
               f"GetDroughtSeverityStatisticsByAreaPercent?aoi={fips}&"
               f"startdate={START.month}/{START.day}/{START.year}&"
               f"enddate={END.month}/{END.day}/{END.year}&statisticsType=1")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    txt = r.read().decode()
                if txt.startswith('MapDate'):
                    frames.append(pd.read_csv(io.StringIO(txt)))
                break
            except Exception as e:
                log(f"  retry {fips}: {str(e)[:60]}"); time.sleep(5)
        if (i + 1) % 20 == 0:
            log(f"  [{state}] {i+1}/{len(fips_list)} counties")
        time.sleep(0.3)
    d = pd.concat(frames, ignore_index=True)
    d.to_parquet(out, index=False)
    log(f"[{state}] downloaded {len(d):,} county-weeks")
    return d


def build_score(state, raw, cmap):
    raw['ValidStart'] = pd.to_datetime(raw['ValidStart'])
    raw['ValidEnd'] = pd.to_datetime(raw['ValidEnd'])
    raw['FIPS'] = raw['FIPS'].astype(str).str.zfill(5)
    for c in ['D0', 'D1', 'D2', 'D3', 'D4']:
        raw[c] = pd.to_numeric(raw[c], errors='coerce').fillna(0)
    # exclusive-band weighting, same formula as TX's idea5 script
    raw['drought_score'] = (raw.D0 - raw.D1) * 1 + (raw.D1 - raw.D2) * 2 + \
                            (raw.D2 - raw.D3) * 3 + (raw.D3 - raw.D4) * 4 + raw.D4 * 5
    raw['drought_score'] /= 100.0
    weekly = raw[['FIPS', 'ValidStart', 'ValidEnd', 'drought_score']].rename(columns={'FIPS': 'fips'})
    weekly.to_parquet(ROOT / "data" / f"drought_weekly_{state}.parquet", index=False)
    log(f"[{state}] weekly drought_score saved ({len(weekly):,} county-weeks)")
    return weekly


if __name__ == '__main__':
    for state in ['nm', 'ok']:
        cmap = pd.read_parquet(ROOT / "data" / f"static_master_{state}.parquet", columns=['h3_cell', 'fips'])
        fips_list = sorted(cmap.fips.dropna().unique())
        log(f"=== {state.upper()}: {len(fips_list)} counties ===")
        raw = download_drought(state, fips_list)
        build_score(state, raw, cmap)
    log("STEP 5 complete (both states).")
