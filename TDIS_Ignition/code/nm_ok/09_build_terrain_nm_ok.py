"""
NM_OK_TexRev — terrain (elevation/slope/aspect) for NM + OK, filling the
pass-2 placeholder from 02_build_static_masters.py. Copernicus GLO-30 DEM
is global (direct S3 vsicurl tile access) -- same method as TX's
scripts/04_build_full_tx_static.py, just pointed at NM/OK cell lat/lons.
No new download/URL needed.

Output: NM_OK_TexRev/data/terrain_{nm,ok}.parquet [h3_cell, elevation_m, slope_deg, aspect_deg]
"""
import numpy as np, pandas as pd, rasterio
from rasterio.transform import rowcol
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

COPERNICUS_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def log(m): print(m, flush=True)


def cop_url(lat_tile, lon_tile):
    nm = f"Copernicus_DSM_COG_10_N{lat_tile:02d}_00_W{lon_tile:03d}_00_DEM"
    return f"/vsicurl/{COPERNICUS_BASE}/{nm}/{nm}.tif"


def build_for_state(state):
    out = ROOT / "data" / f"terrain_{state}.parquet"
    if out.exists():
        log(f"[skip] {out.name} exists"); return

    st = pd.read_parquet(ROOT / "data" / f"static_master_{state}.parquet",
                          columns=['h3_cell', 'lat', 'lon'])
    lats, lons, h3s = st.lat.values, st.lon.values, st.h3_cell.values
    n = len(h3s)
    lat_tiles = np.floor(lats).astype(int)
    lon_tiles = np.ceil(np.abs(lons)).astype(int)
    unique_tiles = sorted(set(zip(lat_tiles, lon_tiles)))
    elevations = np.full(n, np.nan); slopes = np.full(n, np.nan); aspects = np.full(n, np.nan)
    log(f"[{state}] {n:,} cells, {len(unique_tiles)} unique 1x1deg DEM tiles")

    t0 = time.time()
    for ti, (lt, lnt) in enumerate(unique_tiles):
        url = cop_url(lt, lnt)
        mask = (lat_tiles == lt) & (lon_tiles == lnt)
        idxs = np.where(mask)[0]
        try:
            with rasterio.open(url) as ds:
                data = ds.read(1).astype(np.float32)
                data[data <= -9999] = np.nan
                px_deg = ds.transform.a; py_deg = -ds.transform.e
                px_m = px_deg * 111320 * np.cos(np.radians(lt + 0.5))
                py_m = py_deg * 111320
                gy, gx = np.gradient(data, py_m, px_m)
                slope = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))
                aspect = (np.degrees(np.arctan2(-gx, gy)) + 360) % 360
                coords_lon = [float(lons[i]) for i in idxs]
                coords_lat = [float(lats[i]) for i in idxs]
                rows, cols = rowcol(ds.transform, coords_lon, coords_lat)
                rows = np.clip(np.asarray(rows, dtype=int), 0, data.shape[0] - 1)
                cols = np.clip(np.asarray(cols, dtype=int), 0, data.shape[1] - 1)
                elevations[idxs] = data[rows, cols]
                slopes[idxs] = slope[rows, cols]
                aspects[idxs] = aspect[rows, cols]
        except Exception as e:
            log(f"  tile N{lt:02d}_W{lnt:03d} failed: {str(e)[:80]}")
        if (ti + 1) % 20 == 0:
            done = (~np.isnan(elevations)).sum()
            log(f"  [{state}] tile {ti+1}/{len(unique_tiles)} | {done:,}/{n:,} elevations "
                f"({(time.time()-t0)/60:.1f} min)")

    med_e, med_s, med_a = np.nanmedian(elevations), np.nanmedian(slopes), np.nanmedian(aspects)
    df = pd.DataFrame({'h3_cell': h3s, 'elevation_m': elevations,
                        'slope_deg': slopes, 'aspect_deg': aspects})
    miss = df.elevation_m.isna().sum()
    df['elevation_m'] = df['elevation_m'].fillna(med_e)
    df['slope_deg'] = df['slope_deg'].fillna(med_s)
    df['aspect_deg'] = df['aspect_deg'].fillna(med_a)
    df.to_parquet(out, index=False)
    log(f"[{state}] done: mean elev={df.elevation_m.mean():.0f}m, mean slope={df.slope_deg.mean():.1f}deg "
        f"({miss:,} cells filled with median -- tile fetch gaps)")


if __name__ == '__main__':
    for state in ['nm', 'ok']:
        build_for_state(state)
