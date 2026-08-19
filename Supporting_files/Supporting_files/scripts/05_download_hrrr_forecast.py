"""
TDIS Forecast — Step 5: HISTORICAL FORECAST weather (not observed) for training.

Why this exists: training on OBSERVED gridMET weather and deploying with FORECAST
weather (HRRR/GFS) is a train/serve mismatch -- the model never sees forecast error.
This pulls what the forecast ACTUALLY said, at each lead time, for our label dates --
so the model learns on forecast-realistic inputs from day one.

Sources (via herbie, AWS Open Data, no auth):
  HRRR  sfc product,  F24 / F48   (3 km, CONUS)
  GFS   pgrb2.0p25,   F72         (25 km -- HRRR has no F72; GFS is the standard
                                    extended-range source)

IMPORTANT (found by empirical testing this session): HRRR's 24h+ forecast fields
(F24/F48) did NOT EXIST before 2018-07-12 (confirmed: F24 fails through 2018-07-01,
works from 2018-07-15 -- matches NOAA's HRRRv3 upgrade, which first added extended
forecast hours). This is a hard product-availability ceiling, not a flaky download --
retrying pre-cutoff dates will NEVER succeed. ~35% of our label dates (1,656/4,748)
fall before this cutoff.

Fix applied: dates before HRRR_F24_MIN_DATE are skipped WITHOUT a network call (fast)
and logged to unavailable_pre2018.txt. Those rows get their weather feature BACKFILLED
from observed gridMET in step 06, flagged with weather_source='observed_backfill' vs
'forecast' for real HRRR/GFS rows -- so the model/eval can distinguish them (this is
option C from the pipeline review: same data volume, no hidden train/serve mismatch).

Variables pulled per date+lead: t2m, d2m (dewpoint), r2 (RH), wind speed, gust.
VPD (kPa) is derived from t2m/d2m (Magnus formula) to match gridMET's vpd units.

Output: one flat-grid parquet per (date, horizon) under
  data/weather_hrrr_forecast/hrrr_{horizon}h/{date}.parquet
  columns: lat, lon, tmp_c, dpt_c, rh, wind_ms, gust_ms, vpd_kpa
Nearest-join to H3 cells happens in a later assemble step (mirrors the gridMET pattern
-- keeps this raw-grid layer reusable across dataset revisions).

KNOB: HORIZONS below. Default = [24] only (~13 hrs for 4,018 dates). Add 48/72 later
by re-running with HORIZONS=[24,48,72] -- already-done dates/horizons are skipped.

Run: nohup /home/mte1224/mambaforge/envs/UAI2526/bin/python -u 05_download_hrrr_forecast.py > ../hrrr_forecast.log 2>&1 &
"""
import warnings, time
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

TF = Path(__file__).resolve().parent.parent
OUT = TF/"data"/"weather_hrrr_forecast"; OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [24]              # <-- KNOB. add 48, 72 later; already-done work is skipped.
HRRR_MIN_YEAR = 2016         # loose pre-filter; the real gate is HRRR_F24_MIN_DATE below
HRRR_F24_MIN_DATE = pd.Timestamp('2018-07-15')  # empirically confirmed cutoff (see header)
GFS_MIN_DATE = pd.Timestamp('2015-01-15')       # GFS archive on AWS is reliable well before this
TX = dict(lat_min=25.75, lat_max=36.65, lon_min=-106.70, lon_max=-93.40)
INIT_HOUR = 0                 # 00Z run -> F24 valid = next day 00Z, F48 = day+2 00Z, F72 = day+3 00Z
UNAVAIL_LOG = TF/"data"/"weather_hrrr_forecast"/"unavailable_pre2018.txt"

def log(m): print(m, flush=True)

def get_target_dates():
    p = TF/"tdis_train_daily_tx.parquet"
    if not p.exists():
        log("No dataset yet -- run 03_build_dataset.py first."); return []
    df = pd.read_parquet(p, columns=['date'])
    df['date'] = pd.to_datetime(df['date'])
    d = df.loc[df['date'].dt.year >= HRRR_MIN_YEAR, 'date'].dt.normalize().unique()
    return sorted(pd.to_datetime(d))

def vpd_from_temp_dewpoint(t_c, td_c):
    es = 0.6108*np.exp(17.27*t_c/(t_c+237.3))
    ea = 0.6108*np.exp(17.27*td_c/(td_c+237.3))
    return np.clip(es-ea, 0, None)

# ── HARD TIMEOUT WATCHDOG ──────────────────────────────────────────────────
# Root cause of the earlier multi-hour stall: a single network read hung in
# CLOSE_WAIT with no application-level timeout, silently blocking the whole
# job. Herbie/requests don't reliably enforce one on their own here, so wrap
# every fetch attempt in a SIGALRM watchdog -- no single date can ever again
# block the loop for more than FETCH_TIMEOUT_S seconds.
import signal
FETCH_TIMEOUT_S = 90

class FetchTimeout(Exception): pass
def _alarm_handler(signum, frame): raise FetchTimeout()
signal.signal(signal.SIGALRM, _alarm_handler)

def fetch_one(valid_date, horizon):
    """valid_date = the date the forecast is FOR. Init = valid_date - horizon hours, 00Z run."""
    from herbie import Herbie
    init_time = valid_date - pd.Timedelta(hours=horizon) + pd.Timedelta(hours=INIT_HOUR)
    out_dir = OUT/f"hrrr_{horizon}h"; out_dir.mkdir(exist_ok=True)
    out_p = out_dir/f"{valid_date.date()}.parquet"
    if out_p.exists(): return "skip"

    model, product = ("hrrr","sfc") if horizon <= 48 else ("gfs","pgrb2.0p25")

    # FAST SKIP -- no network call -- for dates the product structurally can't cover.
    if model == "hrrr" and init_time < HRRR_F24_MIN_DATE:
        with open(UNAVAIL_LOG, "a") as f:
            f.write(f"{valid_date.date()},{horizon},pre_hrrrv3\n")
        return "unavailable_pre2018"
    if model == "gfs" and init_time < GFS_MIN_DATE:
        with open(UNAVAIL_LOG, "a") as f:
            f.write(f"{valid_date.date()},{horizon},pre_gfs_archive\n")
        return "unavailable_pre2018"

    try:
        signal.alarm(FETCH_TIMEOUT_S)
        h = Herbie(init_time.strftime("%Y-%m-%d %H:%M"), model=model, product=product, fxx=horizon, verbose=False)
        if h.grib is None:
            signal.alarm(0)
            return "missing"
        ds_list = h.xarray(":(TMP|RH|DPT):2 m above ground|:WIND:10 m above ground|:GUST:surface",
                            remove_grib=True)
        signal.alarm(0)
        if not isinstance(ds_list, list): ds_list = [ds_list]
        merged = {}
        for ds in ds_list:
            lat = ds.latitude.values; lon = ds.longitude.values
            lon = np.where(lon > 180, lon-360, lon)
            mask = (lat>=TX['lat_min'])&(lat<=TX['lat_max'])&(lon>=TX['lon_min'])&(lon<=TX['lon_max'])
            for name, da in ds.data_vars.items():
                merged.setdefault('lat', lat[mask])
                merged.setdefault('lon', lon[mask])
                merged[name] = da.values[mask]
        df = pd.DataFrame(merged)
        rename = {'t2m':'tmp_c','d2m':'dpt_c','r2':'rh','max_10si':'wind_ms','si10':'wind_ms','gust':'gust_ms'}
        df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
        if 'tmp_c' in df: df['tmp_c'] = df['tmp_c'] - 273.15
        if 'dpt_c' in df: df['dpt_c'] = df['dpt_c'] - 273.15
        if 'tmp_c' in df and 'dpt_c' in df:
            df['vpd_kpa'] = vpd_from_temp_dewpoint(df['tmp_c'].values, df['dpt_c'].values)
        df.to_parquet(out_p, index=False)
        return "ok"
    except FetchTimeout:
        log(f"    [{valid_date.date()} F{horizon}] TIMEOUT after {FETCH_TIMEOUT_S}s -- skipping, will retry on next run")
        return "error"
    except Exception as e:
        log(f"    [{valid_date.date()} F{horizon}] ERROR: {str(e)[:100]}")
        return "error"
    finally:
        signal.alarm(0)   # belt-and-suspenders: never let an alarm leak into the next date

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    dates = get_target_dates()
    n_pre = sum(1 for d in dates if (d - pd.Timedelta(hours=min(HORIZONS))) < HRRR_F24_MIN_DATE and min(HORIZONS) <= 48)
    log(f"STEP 5 — HRRR/GFS forecast weather. {len(dates)} target dates, horizons={HORIZONS}")
    log(f"  ~{n_pre} dates predate HRRR F24 availability ({HRRR_F24_MIN_DATE.date()}) -- "
        f"will fast-skip (no network call) and flag for gridMET backfill in step 06")
    counts = {"ok":0,"skip":0,"missing":0,"error":0,"unavailable_pre2018":0}
    t0 = time.time()
    for i, d in enumerate(dates, 1):
        for hz in HORIZONS:
            r = fetch_one(d, hz)
            counts[r] += 1
        if i % 200 == 0:
            el = time.time()-t0
            attempted = counts['ok']+counts['missing']+counts['error']  # real network attempts only
            rate = attempted/el if el>0 else 0
            n_left_real = len(dates) - i  # rough; good enough for an ETA
            eta_h = (n_left_real*rate and (n_left_real/max(rate,1e-6)))/3600
            log(f"  {i}/{len(dates)} dates | {counts} | {el/60:.1f} min elapsed | rough ETA {eta_h:.1f} hr")
    log(f"DONE. {counts}")
    log("Next: 06_assemble_forecast_weather.py to nearest-join these onto H3 cells "
        "(real forecast rows + gridMET backfill for pre-2018-07 rows, flagged via weather_source).")
