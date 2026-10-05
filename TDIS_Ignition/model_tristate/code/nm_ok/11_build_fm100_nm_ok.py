"""
NM_OK_TexRev — dead-fuel-moisture (fm100) for NM + OK (stakeholder feedback,
2026-09-11). Same gridMET source/method as TX's gridmet_tx/fm100_{year}_
tx_cells.parquet (see alphaearth_nds/download_gridmet_tx_hist.py), generalized
to NM/OK cell centroids from static_master_{nm,ok}.parquet, 2014-2026 (rev4's
training window). fm100 only -- not the full 8-var gridMET set -- since that's
the one feature actually being tested; extend to the other vars only if fm100
proves out.

Output: NM_OK_TexRev/data/gridmet_nm_ok/fm100_{year}_{state}_cells.parquet
        [date_utc, h3_cell, fm100]   (same schema as TX's fm100_*_tx_cells.parquet)
Restart-safe: skips existing (year, state) files.
"""
import numpy as np, pandas as pd, xarray as xr
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = ROOT / "data" / "gridmet_nm_ok"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2014, 2027))
STATES = ['nm', 'ok']

OPENDAP_URL = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_fm100_1979_CurrentYear_CONUS.nc"
HTTPS_TMPL = "https://www.northwestknowledge.net/metdata/data/fm100_{year}.nc"
VAR_LONG = "dead_fuel_moisture_100hr"
NC_CACHE = OUT_DIR / "nc_cache"
NC_CACHE.mkdir(exist_ok=True)


def log(m): print(m, flush=True)


def load_cells(state):
    st = pd.read_parquet(ROOT / "data" / f"static_master_{state}.parquet",
                          columns=['h3_cell', 'lat', 'lon'])
    return st.h3_cell.values, st.lat.values, st.lon.values


def try_opendap(year, lat_min, lat_max, lon_min, lon_max):
    ds = xr.open_dataset(OPENDAP_URL, engine='netcdf4')
    da = ds[VAR_LONG]
    lat_n = 'lat' if 'lat' in da.dims else 'latitude'
    lon_n = 'lon' if 'lon' in da.dims else 'longitude'
    time_n = 'day' if 'day' in da.dims else 'time'
    sub = da.sel({lat_n: slice(lat_max, lat_min), lon_n: slice(lon_min, lon_max),
                  time_n: slice(f"{year}-01-01", f"{year}-12-31")})
    data, g_lats, g_lons = sub.values, sub[lat_n].values, sub[lon_n].values
    times = pd.to_datetime(sub[time_n].values)
    ds.close()
    return data, g_lats, g_lons, times


def try_https(year, lat_min, lat_max, lon_min, lon_max):
    import requests, shutil
    nc_path = NC_CACHE / f"fm100_{year}.nc"
    if not nc_path.exists():
        log(f"    downloading {HTTPS_TMPL.format(year=year)} ...")
        r = requests.get(HTTPS_TMPL.format(year=year), stream=True, timeout=120)
        r.raise_for_status()
        with open(nc_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    ds = xr.open_dataset(nc_path)
    da = ds[VAR_LONG]
    lat_n = 'lat' if 'lat' in da.dims else 'latitude'
    lon_n = 'lon' if 'lon' in da.dims else 'longitude'
    time_n = 'day' if 'day' in da.dims else 'time'
    sub = da.sel({lat_n: slice(lat_max, lat_min), lon_n: slice(lon_min, lon_max),
                  time_n: slice(f"{year}-01-01", f"{year}-12-31")})
    data, g_lats, g_lons = sub.values, sub[lat_n].values, sub[lon_n].values
    times = pd.to_datetime(sub[time_n].values)
    ds.close()
    return data, g_lats, g_lons, times


def extract_year(state, year, h3s, lats, lons):
    out_path = OUT_DIR / f"fm100_{year}_{state}_cells.parquet"
    if out_path.exists():
        log(f"  [skip] {out_path.name}"); return True
    pad = 0.3
    lat_min, lat_max = lats.min() - pad, lats.max() + pad
    lon_min, lon_max = lons.min() - pad, lons.max() + pad
    log(f"  fetching fm100 {state} {year}...")
    try:
        try:
            data, g_lats, g_lons, times = try_opendap(year, lat_min, lat_max, lon_min, lon_max)
        except Exception as e:
            log(f"    OPeNDAP failed ({e}), falling back to HTTPS...")
            data, g_lats, g_lons, times = try_https(year, lat_min, lat_max, lon_min, lon_max)

        lat_idx = np.argmin(np.abs(g_lats[:, None] - lats[None, :]), axis=0)
        lon_idx = np.argmin(np.abs(g_lons[:, None] - lons[None, :]), axis=0)
        vals = data[:, lat_idx, lon_idx]

        df = pd.DataFrame(vals, index=times, columns=h3s)
        df.index.name = 'date_utc'
        df = df.reset_index().melt(id_vars='date_utc', var_name='h3_cell', value_name='fm100')
        df['date_utc'] = df['date_utc'].dt.date
        df.to_parquet(out_path, index=False)
        log(f"    saved {out_path.name} ({len(df):,} rows, {out_path.stat().st_size/1e6:.0f} MB)")
        return True
    except Exception as e:
        log(f"    ERROR: {e}")
        if out_path.exists(): out_path.unlink()
        return False


if __name__ == '__main__':
    failed = []
    for state in STATES:
        h3s, lats, lons = load_cells(state)
        log(f"=== {state.upper()}: {len(h3s):,} cells, {YEARS[0]}-{YEARS[-1]} ===")
        for year in YEARS:
            if not extract_year(state, year, h3s, lats, lons):
                failed.append(f"{state}_{year}")
    if failed:
        log(f"\nFAILED ({len(failed)}): {', '.join(failed)} -- re-run to retry (skip-if-exists)")
    else:
        log("\nAll fm100 files fetched for NM + OK.")
