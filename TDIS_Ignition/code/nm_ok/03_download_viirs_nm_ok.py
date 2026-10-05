"""
NM_OK_TexRev — Step 3: full-state VIIRS active-fire re-pull for NM + OK,
2014-2026. The existing viirs_tx_h3.parquet only has NM/OK fires that
happened to fall inside the TX download bbox (spillover, ~6% of TX
positives per SANITY_CHECK finding F1) -- this pulls each state's own
full bbox so labels aren't missing detections outside that rectangle.

Same FIRMS API pattern as scripts/01_download_viirs.py (SP archive +
NRT recent), same throttle/budget/resume logic. Runs NM then OK,
newest-year-first per state.

Output: NM_OK_TexRev/data/labels_viirs/viirs_{nm,ok}_h3.parquet
Restart-safe: skip-if-exists raw CSVs, DONE-file markers per state/year.
"""
import time
import numpy as np, pandas as pd, requests, h3
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TF = ROOT.parent
RAW = ROOT / "data" / "labels_viirs" / "raw"; RAW.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "data" / "labels_viirs"
KEY = (TF / "firms_map_key.txt").read_text().strip()
STATUS_URL = f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={KEY}"
H3_RES, MAX_DAYS = 8, 5

BBOX = {
    'nm': "-109.10,31.00,-102.90,37.10",   # lon_min,lat_min,lon_max,lat_max, padded
    'ok': "-103.10,33.50,-94.30,37.10",
}
YEARS = list(range(2026, 2013, -1))
SP_START = {"VIIRS_SNPP_SP": 2014, "VIIRS_NOAA20_SP": 2018}
NRT_SRC = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"]
NRT_MIN_YEAR = 2025
TODAY = pd.Timestamp.today().normalize()


def log(m): print(m, flush=True)


def sources_for(y):
    s = [k for k, y0 in SP_START.items() if y >= y0]
    if y >= NRT_MIN_YEAR:
        s += NRT_SRC
    return s


def check_budget():
    try:
        st = requests.get(STATUS_URL, timeout=30).json()
        cur, lim = st.get("current_transactions", 0), st.get("transaction_limit", 5000)
        if cur > lim * 0.8:
            log(f"  [budget] {cur}/{lim} -- sleeping 10 min"); time.sleep(610)
        return cur
    except Exception:
        return None


def windows(year):
    d = pd.Timestamp(f"{year}-01-01")
    end = min(pd.Timestamp(f"{year}-12-31"), TODAY)
    while d <= end:
        n = min(MAX_DAYS, (end - d).days + 1)
        yield d.strftime("%Y-%m-%d"), n
        d += pd.Timedelta(days=n)


def fetch(state, source, start, ndays):
    out = RAW / f"{state}_{source}_{start}_{ndays}d.csv"
    if out.exists():
        return
    bbox = BBOX[state]
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/{source}/{bbox}/{ndays}/{start}"
    for att in range(5):
        try:
            r = requests.get(url, timeout=120); t = r.text; tl = t.lower()
            if tl.lstrip().startswith("latitude") or t.strip() == "":
                out.write_text(t); return
            if "invalid" in tl[:60]:
                log(f"  [{state} {source} {start}] permanent err: {t[:70]!r} skip"); return
            if any(w in tl[:120] for w in ["rate", "limit", "exceed"]):
                log(f"  [{state} {source} {start}] throttled backoff {att+1}"); time.sleep(90 * (att + 1)); continue
            out.write_text(t); return
        except Exception as e:
            log(f"  [{state} {source} {start}] {e} retry {att+1}"); time.sleep(15 * (att + 1))
    log(f"  [{state} {source} {start}] FAILED (left for re-run)")


def download_year(state, y):
    call = 0
    for src in sources_for(y):
        for start, n in windows(y):
            if (RAW / f"{state}_{src}_{start}_{n}d.csv").exists():
                continue
            fetch(state, src, start, n); call += 1; time.sleep(0.8)
            if call % 150 == 0:
                check_budget()
    return call


def assign_all(state):
    frames = []
    for f in sorted(RAW.glob(f"{state}_*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df) == 0 or 'latitude' not in df.columns:
            continue
        frames.append(df)
    if not frames:
        return 0
    v = pd.concat(frames, ignore_index=True)
    v['acq_time'] = v['acq_time'].astype(str).str.zfill(4)
    v['hour'] = v['acq_time'].str[:2].astype(int).clip(0, 23)
    v['date'] = pd.to_datetime(v['acq_date'])
    v['dt_utc'] = v['date'] + pd.to_timedelta(v['hour'], unit='h')
    v['window_hour'] = (v['hour'] // 6) * 6
    v['h3_cell'] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(v['latitude'], v['longitude'])]
    cols = [c for c in ['h3_cell', 'date', 'dt_utc', 'hour', 'window_hour', 'latitude', 'longitude',
                         'confidence', 'frp'] if c in v.columns]
    v = v[cols].drop_duplicates()
    v.to_parquet(OUT / f"viirs_{state}_h3.parquet", index=False)
    return len(v)


if __name__ == '__main__':
    check_budget()
    for state in ['nm', 'ok']:
        log(f"=== {state.upper()}: VIIRS download, {YEARS[0]}..{YEARS[-1]} (through {TODAY.date()}) ===")
        for i, y in enumerate(YEARS, 1):
            log(f">> {state} year {y} (iter {i}/{len(YEARS)}, sources: {sources_for(y)})")
            c = download_year(state, y)
            (ROOT / f"VIIRS_{state.upper()}_YEAR_{y}_DONE").write_text(str(pd.Timestamp.now()))
            log(f"   {state} year {y} done ({c} new calls)")
        n = assign_all(state)
        log(f"=== {state.upper()} complete: {n:,} detections -> viirs_{state}_h3.parquet ===")
    log("STEP 3 complete (both states).")
