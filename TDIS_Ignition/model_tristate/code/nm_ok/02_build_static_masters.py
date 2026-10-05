"""02 — Static masters for TX/NM/OK on ONE national scale (Option A).

This pass covers the layers we have on disk today:
  FSim CONUS 270m:  avg_burn_prob = CONUS_BP
                    flep4 = FLP3+FLP4+FLP5+FLP6      (P(flame length >= 4 ft | fire);
                            FSim classes: 1:0-2ft 2:2-4 3:4-6 4:6-8 5:8-12 6:12+)
                    cfl   = sum(mid_c * FLP_c)/sum(FLP_c), mids = 1,3,5,7,10,14 ft
  EPA L3 ecoregions: ecoregion_id (US_L3CODE, national coding — NOT the old TX 0-19 coding)
  TIGER prisec roads: road_dist_km (cKDTree to densified line vertices)
  Census counties:    fips (for USDM drought join)
Deferred to pass 2 (URLs TBD): whp, cbd, cbh, powerline_dist_km, terrain
(elevation/slope/aspect) — columns created as NaN placeholders.

Output: data/static_master_{tx,nm,ok}.parquet
Ends with the SCALE-TRAP GUARD: identical value-range assertions per state.
"""
import os, sys
_proj = os.path.join(sys.executable.split('/bin/python')[0], 'share', 'proj')
os.environ['PROJ_DATA'] = _proj
import pyproj; pyproj.datadir.set_data_dir(_proj)
import numpy as np, pandas as pd, geopandas as gpd, rasterio, h3
from pathlib import Path
from scipy.spatial import cKDTree
from shapely.geometry import Point

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data/raw"
FSIM = RAW / "Data/I_FSim_CONUS_LF2020_270m"
STATES = {'tx': '48', 'nm': '35', 'ok': '40'}
MIDS = np.array([1, 3, 5, 7, 10, 14], dtype=np.float64)


def log(m): print(m, flush=True)


def cells_for_state(stusps_fips):
    states = gpd.read_file(RAW / "states_us")
    poly = states[states.STATEFP == stusps_fips].geometry.iloc[0]
    # H3 polyfill via geo interface (handles multipolygon)
    cells = set()
    geoms = poly.geoms if poly.geom_type == 'MultiPolygon' else [poly]
    for g in geoms:
        cells |= set(h3.geo_to_cells(g.__geo_interface__, 8))
    return sorted(cells)


def sample_raster(path, lats, lons, band=1):
    """Sample a raster at lat/lon points (vectorized affine indexing)."""
    with rasterio.open(path) as src:
        tf = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = tf.transform(lons, lats)
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows, cols = np.asarray(rows), np.asarray(cols)
        arr = src.read(band)
        nod = src.nodata
        ok = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
        out = np.full(len(lats), np.nan, dtype=np.float64)
        v = arr[rows[ok], cols[ok]].astype(np.float64)
        if nod is not None:
            v[v == nod] = np.nan
        out[ok] = v
        return out


def road_dist_km(lats, lons, roads_dir):
    rd = gpd.read_file(roads_dir)
    pts = []
    for geom in rd.geometry:
        lines = geom.geoms if geom.geom_type == 'MultiLineString' else [geom]
        for ln in lines:
            xy = np.asarray(ln.coords)
            pts.append(xy[::3])          # every 3rd vertex is plenty at prisec density
    P = np.vstack(pts)
    tree = cKDTree(np.c_[P[:, 1], P[:, 0]])          # (lat, lon)
    d, _ = tree.query(np.c_[lats, lons])
    return d * 111.0                                  # deg -> km (approx, fine for a distance feature)


def build_state(st):
    log(f"=== {st.upper()} ===")
    cells = cells_for_state(STATES[st])
    ll = np.array([h3.cell_to_latlng(c) for c in cells])
    df = pd.DataFrame({'h3_cell': cells, 'lat': ll[:, 0], 'lon': ll[:, 1]})
    log(f"{len(df):,} res-8 cells")

    df['avg_burn_prob'] = sample_raster(FSIM / "CONUS_BP.tif", df.lat.values, df.lon.values)
    flp = np.vstack([sample_raster(FSIM / f"CONUS_FLP{i}.tif", df.lat.values, df.lon.values)
                     for i in range(1, 7)])          # (6, n)
    tot = np.nansum(flp, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        df['flep4'] = np.where(tot > 0, np.nansum(flp[2:], axis=0) / tot, 0.0)
        df['cfl'] = np.where(tot > 0, np.nansum(flp * MIDS[:, None], axis=0) / tot, np.nan)
    df.loc[df.avg_burn_prob.isna(), ['flep4', 'cfl']] = np.nan

    eco = gpd.read_file(RAW / "ecoregions_l3").to_crs("EPSG:4326")
    gpts = gpd.GeoDataFrame(geometry=[Point(xy) for xy in zip(df.lon, df.lat)], crs="EPSG:4326")
    j = gpd.sjoin(gpts, eco[['US_L3CODE', 'geometry']], how='left', predicate='within')
    j = j[~j.index.duplicated(keep='first')]
    df['ecoregion_id'] = pd.to_numeric(j['US_L3CODE'], errors='coerce').reindex(df.index).values

    df['road_dist_km'] = road_dist_km(df.lat.values, df.lon.values, RAW / f"roads_{st}")

    cty = gpd.read_file(RAW / "counties_us").to_crs("EPSG:4326")
    jc = gpd.sjoin(gpts, cty[['GEOID', 'geometry']], how='left', predicate='within')
    jc = jc[~jc.index.duplicated(keep='first')]
    df['fips'] = jc['GEOID'].reindex(df.index).values

    for col in ['whp', 'cbd', 'cbh', 'powerline_dist_km',
                'elevation_m', 'slope_deg', 'aspect_deg']:
        df[col] = np.nan                              # pass-2 placeholders

    # ---- SCALE-TRAP GUARD: identical assertions for every state ----
    bp = df.avg_burn_prob.dropna()
    assert 0 <= bp.min() and bp.max() <= 1.0001, f"{st}: burn_prob out of [0,1]!"
    f4 = df.flep4.dropna()
    assert 0 <= f4.min() and f4.max() <= 1.0001, f"{st}: flep4 out of [0,1]!"
    log(f"  bp mean {bp.mean():.4f} | flep4 mean {f4.mean():.3f} | cfl mean {df.cfl.mean():.2f} | "
        f"eco coverage {df.ecoregion_id.notna().mean()*100:.1f}% | road km p50 {df.road_dist_km.median():.1f}")
    df.to_parquet(HERE / f"data/static_master_{st}.parquet", index=False)
    log(f"  saved static_master_{st}.parquet")


if __name__ == '__main__':
    for st in ['nm', 'ok', 'tx']:
        build_state(st)
    log("DONE pass 1 (hazard/eco/roads/fips). Pass 2 pending: whp, canopy, powerlines, terrain.")
