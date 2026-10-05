"""
Full-coverage TX dead-fuel-moisture (fm100), 2014-2026 (stakeholder feedback,
2026-09-11 -- fuel moisture ablation showed fm100 ranks #3 in feature
importance, but the LEGACY archive (alphaearth_nds/gridmet_tx/fm100_*_tx_
cells.parquet) only covers 23.2% of TX's current training-cell universe --
it was built years ago against an old, smaller candidate-cell list, not
tx_static_master.parquet's full 1,708,940 cells.

This rebuilds fm100 against the FULL static-master cell list, same gridMET
source/method as the legacy archive and as NM_OK_TexRev/scripts/
11_build_fm100_nm_ok.py (kept identical on purpose so all three states are
one consistent source). Written to a NEW directory -- does not touch or
overwrite the legacy partial archive.

Output: TDIS_Forecast/data/gridmet_tx_full/fm100_{year}_tx_cells.parquet
        [date_utc, h3_cell, fm100]
Restart-safe: skips existing year files. ~4.6x NM's cell count -> expect
~4-5x NM's per-year runtime.
"""
import numpy as np, pandas as pd, xarray as xr
from pathlib import Path

TF = Path("/net/cven-mosta-nas.engr.tamu.edu/volume2/NASdata2/miguel_shared/alphaearth_nds/TDIS_Forecast")
OUT_DIR = TF / "data" / "gridmet_tx_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NC_CACHE = OUT_DIR / "nc_cache"
NC_CACHE.mkdir(exist_ok=True)

YEARS = list(range(2014, 2027))
OPENDAP_URL = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_fm100_1979_CurrentYear_CONUS.nc"
HTTPS_TMPL = "https://www.northwestknowledge.net/metdata/data/fm100_{year}.nc"
VAR_LONG = "dead_fuel_moisture_100hr"


def log(m): print(m, flush=True)


def load_cells():
    st = pd.read_parquet(TF / "data/static_features/tx_static_master.parquet",
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


def extract_year(year, h3s, lats, lons):
    out_path = OUT_DIR / f"fm100_{year}_tx_cells.parquet"
    if out_path.exists():
        log(f"  [skip] {out_path.name}"); return True
    pad = 0.3
    lat_min, lat_max = lats.min() - pad, lats.max() + pad
    lon_min, lon_max = lons.min() - pad, lons.max() + pad
    log(f"  fetching fm100 tx-full {year}...")
    try:
        try:
            data, g_lats, g_lons, times = try_opendap(year, lat_min, lat_max, lon_min, lon_max)
        except Exception as e:
            log(f"    OPeNDAP failed ({e}), falling back to HTTPS...")
            data, g_lats, g_lons, times = try_https(year, lat_min, lat_max, lon_min, lon_max)

        # chunk the nearest-index match to keep peak memory sane at 1.71M cells
        CH = 200_000
        lat_idx = np.empty(len(lats), dtype=np.int32)
        lon_idx = np.empty(len(lons), dtype=np.int32)
        for s in range(0, len(lats), CH):
            e = s + CH
            lat_idx[s:e] = np.argmin(np.abs(g_lats[:, None] - lats[None, s:e]), axis=0)
            lon_idx[s:e] = np.argmin(np.abs(g_lons[:, None] - lons[None, s:e]), axis=0)
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
    h3s, lats, lons = load_cells()
    log(f"=== TX FULL: {len(h3s):,} cells, {YEARS[0]}-{YEARS[-1]} ===")
    failed = []
    for year in YEARS:
        if not extract_year(year, h3s, lats, lons):
            failed.append(str(year))
    if failed:
        log(f"\nFAILED ({len(failed)}): {', '.join(failed)} -- re-run to retry (skip-if-exists)")
    else:
        log("\nAll fm100 files fetched for full TX.")
