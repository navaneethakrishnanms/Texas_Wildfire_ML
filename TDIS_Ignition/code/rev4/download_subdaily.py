"""
Rev4 archive build: pull HRRR F21 / F27 / F30 (same 00Z init as the
existing F24 archive) for every archived date -- the sub-daily hours
needed for daily-PEAK weather features.

Ordered EVAL-YEARS-FIRST (2024->2026 desc priority, then 2018-2023) so
the Smokehouse re-test can run before the full training backfill lands.
Restart-safe (skips existing). Watchdog per fetch (pattern from
scripts/05). Output: data/weather_hrrr_forecast/hrrr_f{21,27,30}/{date}.parquet
columns: lat, lon, tmp_c, dpt_c, wind_ms, gust_ms, vpd_kpa, mstav_pct
"""
import warnings, time, signal
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

TF = Path(__file__).resolve().parent.parent
SRC = TF / "data/weather_hrrr_forecast/hrrr_24h"
TX = dict(lat_min=25.75, lat_max=36.65, lon_min=-106.70, lon_max=-93.40)
LEADS = [27, 30, 33, 36]   # from 12Z init: 9AM, noon, 3PM, 6PM local
FETCH_TIMEOUT_S = 90


def log(m): print(m, flush=True)


class FetchTimeout(Exception): pass
def _alarm(s, f): raise FetchTimeout()
signal.signal(signal.SIGALRM, _alarm)


def vpd_from(t, td):
    es = 0.6108*np.exp(17.27*t/(t+237.3)); ea = 0.6108*np.exp(17.27*td/(td+237.3))
    return np.clip(es-ea, 0, None)


def fetch_one(valid_date, lead):
    from herbie import Herbie
    outdir = TF / f"data/weather_hrrr_forecast/hrrr_12z_f{lead}"
    outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / f"{valid_date.date()}.parquet"
    if outp.exists():
        return "skip"
    # REV4 PIVOT (2026-08-20, see PLAN.md): init = 12Z the day BEFORE the
    # target (leads 27-36h = a proper previous-afternoon forecast). The 00Z
    # init missed Smokehouse's windstorm 5x (3.9 vs 20.6 m/s box gusts); the
    # 12Z init nailed it. F27/30/33/36 = target-day 9AM/noon/3PM/6PM local.
    init = valid_date - pd.Timedelta(hours=36) + pd.Timedelta(hours=24)
    try:
        signal.alarm(FETCH_TIMEOUT_S)
        h = Herbie(init.strftime("%Y-%m-%d %H:%M"), model="hrrr", product="sfc",
                   fxx=lead, verbose=False)
        if h.grib is None:
            signal.alarm(0); return "missing"
        dsl = h.xarray(":(TMP|DPT):2 m above ground|:WIND:10 m above ground|:GUST:surface|:MSTAV:",
                        remove_grib=True)
        signal.alarm(0)
        if not isinstance(dsl, list): dsl = [dsl]
        mg = {}
        for ds in dsl:
            la = ds.latitude.values.ravel(); lo = ds.longitude.values.ravel()
            lo = np.where(lo > 180, lo-360, lo)
            msk = (la >= TX['lat_min']) & (la <= TX['lat_max']) & \
                  (lo >= TX['lon_min']) & (lo <= TX['lon_max'])
            mg.setdefault('lat', la[msk]); mg.setdefault('lon', lo[msk])
            for nm, da in ds.data_vars.items():
                mg[nm] = da.values.ravel()[msk]
        df = pd.DataFrame(mg).rename(columns={'t2m': 'tmp_c', 'd2m': 'dpt_c',
                                               'max_10si': 'wind_ms', 'si10': 'wind_ms',
                                               'gust': 'gust_ms', 'mstav': 'mstav_pct'})
        if 'tmp_c' in df: df['tmp_c'] -= 273.15
        if 'dpt_c' in df: df['dpt_c'] -= 273.15
        if 'tmp_c' in df and 'dpt_c' in df:
            df['vpd_kpa'] = vpd_from(df['tmp_c'].values, df['dpt_c'].values)
        df.to_parquet(outp, index=False)
        return "ok"
    except FetchTimeout:
        return "timeout"
    except Exception as e:
        log(f"    [{valid_date.date()} F{lead}] ERR {str(e)[:80]}")
        return "error"
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    dates = sorted(pd.Timestamp(p.stem) for p in SRC.glob('*.parquet'))
    # eval years first (newest first), then training years
    eval_d = [d for d in dates if d.year >= 2024]
    train_d = [d for d in dates if d.year < 2024]
    ordered = sorted(eval_d, reverse=True) + sorted(train_d, reverse=True)
    total = len(ordered) * len(LEADS)
    log(f"rev4 sub-daily archive: {len(ordered)} dates x {LEADS} = {total} pulls "
        f"(eval years first)")
    counts = {"ok": 0, "skip": 0, "missing": 0, "error": 0, "timeout": 0}
    t0 = time.time(); i = 0
    for d in ordered:
        for lead in LEADS:
            counts[fetch_one(d, lead)] += 1
            i += 1
            if i % 200 == 0:
                att = counts['ok']+counts['missing']+counts['error']+counts['timeout']
                rate = att/max(time.time()-t0, 1)
                eta_h = (total-i)/max(rate, 1e-6)/3600
                log(f"  {i}/{total} | {counts} | ETA ~{eta_h:.1f} h")
    log(f"DONE. {counts}")
