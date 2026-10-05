"""
New_Training817_moredata — backfill HRRR soil-moisture (MSTAV) for the
historical forecast archive, as a future model feature.

The existing archive (data/weather_hrrr_forecast/hrrr_24h/) has
tmp/dpt/rh/wind/gust/vpd but NOT soil moisture. MSTAV (soil-moisture
availability, %) is the M term of the NOAA HWP formula and the one
untested feature family with a real physical argument (fuel dryness).
This pulls ONLY :MSTAV: for every date that already exists in the
hrrr_24h archive (same init/lead convention: 00Z run, F24), so the new
column joins 1:1 onto the existing training weather.

Mirrors scripts/05_download_hrrr_forecast.py exactly (watchdog timeout,
skip-if-done, same TX bbox). Restart-safe: re-run to resume.

Output: data/weather_hrrr_forecast/hrrr_24h_mstav/{date}.parquet
        columns: lat, lon, mstav_pct
Run: nohup python -u download_mstav.py > mstav_download.log 2>&1 &
"""
import warnings, time, signal
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
TF = HERE.parent
SRC = TF / "data" / "weather_hrrr_forecast" / "hrrr_24h"
OUT = TF / "data" / "weather_hrrr_forecast" / "hrrr_24h_mstav"
OUT.mkdir(parents=True, exist_ok=True)
TX = dict(lat_min=25.75, lat_max=36.65, lon_min=-106.70, lon_max=-93.40)
HORIZON = 24
FETCH_TIMEOUT_S = 90


def log(m): print(m, flush=True)


class FetchTimeout(Exception): pass
def _alarm(s, f): raise FetchTimeout()
signal.signal(signal.SIGALRM, _alarm)


def fetch_one(valid_date):
    from herbie import Herbie
    out_p = OUT / f"{valid_date.date()}.parquet"
    if out_p.exists():
        return "skip"
    init_time = valid_date - pd.Timedelta(hours=HORIZON)
    try:
        signal.alarm(FETCH_TIMEOUT_S)
        h = Herbie(init_time.strftime("%Y-%m-%d %H:%M"), model="hrrr", product="sfc",
                   fxx=HORIZON, verbose=False)
        if h.grib is None:
            signal.alarm(0); return "missing"
        ds = h.xarray(":MSTAV:", remove_grib=True)
        signal.alarm(0)
        if isinstance(ds, list): ds = ds[0]
        lat = ds.latitude.values; lon = ds.longitude.values
        lon = np.where(lon > 180, lon - 360, lon)
        mask = (lat >= TX['lat_min']) & (lat <= TX['lat_max']) & \
               (lon >= TX['lon_min']) & (lon <= TX['lon_max'])
        var = list(ds.data_vars)[0]
        df = pd.DataFrame(dict(lat=lat[mask], lon=lon[mask],
                                mstav_pct=ds[var].values[mask]))
        df.to_parquet(out_p, index=False)
        return "ok"
    except FetchTimeout:
        log(f"    [{valid_date.date()}] TIMEOUT"); return "error"
    except Exception as e:
        log(f"    [{valid_date.date()}] ERROR: {str(e)[:90]}"); return "error"
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    dates = sorted(pd.Timestamp(p.stem) for p in SRC.glob("*.parquet"))
    log(f"MSTAV backfill: {len(dates)} archive dates ({dates[0].date()} .. {dates[-1].date()})")
    counts = {"ok": 0, "skip": 0, "missing": 0, "error": 0}
    t0 = time.time()
    for i, d in enumerate(dates, 1):
        counts[fetch_one(d)] += 1
        if i % 100 == 0:
            el = time.time() - t0
            att = counts['ok'] + counts['missing'] + counts['error']
            eta = (len(dates) - i) / max(att / el, 1e-6) / 3600 if att else 0
            log(f"  {i}/{len(dates)} | {counts} | {el/60:.1f} min | ETA ~{eta:.1f} h")
    log(f"DONE. {counts}")
