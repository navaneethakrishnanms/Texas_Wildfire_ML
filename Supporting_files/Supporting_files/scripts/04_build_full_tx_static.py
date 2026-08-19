"""
TDIS Forecast — Step 4: MASTER static-feature layer for the FULL Texas H3-8 grid.

Fixes the coverage gap found in the starter dataset: the old tx_geo_features.parquet
only covered 317,142 curated cells (~1/3 of Texas). VIIRS fire detections land on real
GPS coordinates and 55% fell outside that set -> missing geo/landscape features for
over half the fire-positive rows.

Target cell universe: the 1,708,940 unique H3-8 cells in landscape_h3_tx.parquet, which
already spans the FULL Texas lat/lon range (confirmed) and already carries avg_burn_prob/
whp/flep4/cfl for all of them. This script adds the still-missing pieces for ALL of those
cells:
  - ecoregion_id      (EPA Level-3 Ecoregion, vectorized STRtree spatial join)
  - elevation_m       (Copernicus GLO-30 DEM, same tiled-raster approach as before)
  - slope_deg/aspect_deg  (derived from the elevation raster, computed properly this time
                           -- the old file had these as NaN placeholders)
  - road_dist_km      (Census TIGER/Line 2022 primary/secondary roads, same STRtree approach)
  - cbd/cbh           (LANDFIRE LF2022, direct raster sample -- already cached as GeoTIFFs)

Output: TDIS_Forecast/data/static_features/tx_static_master.parquet
        one row per H3-8 cell, ALL of Texas, ALL static features, no coverage gap.

Run (can take a few hours for 1.7M cells):
  nohup /home/mte1224/mambaforge/envs/UAI2526/bin/python -u 04_build_full_tx_static.py > ../full_static.log 2>&1 &
"""
import os, warnings, json, time
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from shapely.ops import transform as shp_transform
import pyproj
from pyproj import Transformer
import pyogrio.raw
from shapely import from_wkb
import shapely
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform

warnings.filterwarnings('ignore')

BASE = Path("/net/cven-mosta-nas.engr.tamu.edu/volume2/NASdata2/miguel_shared/alphaearth_nds")
TF   = Path(__file__).resolve().parent.parent  # package-relative; BASE above stays absolute (external cross-project deps below)
CACHE = BASE/"geo_cache"                       # reuse existing cached downloads (roads, ecoregion geojson)
OUT_DIR = TF/"data"/"static_features"; OUT_DIR.mkdir(parents=True, exist_ok=True)
LANDSCAPE_PQ = BASE/"90%_ig_dec/Transferibility_OutOfState/data/TX/landscape/landscape_h3_tx.parquet"
NAT_DIR = BASE/"90%_ig_dec/Transferibility_OutOfState/data/national_rasters"
CBD_TIF, CBH_TIF = NAT_DIR/"CBD_TX300m.tif", NAT_DIR/"CBH_TX300m.tif"

def log(m): print(m, flush=True)

# ── 1. TARGET CELL UNIVERSE — full TX from landscape_h3_tx.parquet ──
log("Loading target cell universe (full TX)...")
land = pd.read_parquet(LANDSCAPE_PQ)  # h3_cell, lat, lon, avg_burn_prob, whp, flep4, cfl, ghm
land = land.drop_duplicates('h3_cell').reset_index(drop=True)
lats, lons, h3s = land['lat'].values, land['lon'].values, land['h3_cell'].values
n = len(h3s)
log(f"  {n:,} unique H3-8 cells (full TX coverage)")

df = land[['h3_cell','lat','lon','avg_burn_prob','whp','flep4','cfl']].copy()
for c in ['avg_burn_prob','whp','flep4','cfl']:
    df[c] = df[c].fillna(0)

p_wgs = pyproj.Proj(proj='latlong', ellps='WGS84')
p_utm = pyproj.Proj(proj='utm', zone=14, ellps='WGS84')
to_utm = Transformer.from_proj(p_wgs, p_utm, always_xy=True)

# ── 2. ECOREGION (EPA L3, reuse cached geojson, VECTORIZED via STRtree) ──
eco_cache = CACHE/"tx_eco_l3.geojson"
if not eco_cache.exists():
    log("Fetching EPA Level-3 Ecoregions (not cached -- unexpected, downloading)...")
    url = ("https://geodata.epa.gov/arcgis/rest/services/ORD/"
           "USEPA_Ecoregions_Level_III_and_IV/MapServer/10/query"
           "?where=1%3D1&outFields=US_L3CODE%2CUS_L3NAME"
           "&geometryType=esriGeometryEnvelope&geometry=-107%2C25%2C-93%2C37"
           "&inSR=4326&spatialRel=esriSpatialRelIntersects&returnGeometry=true&f=geojson")
    r = requests.get(url, timeout=60); r.raise_for_status()
    eco_cache.write_bytes(r.content)

eco_full_cache = OUT_DIR/"ecoregion_full.parquet"
if eco_full_cache.exists():
    log("[skip] ecoregion already computed for full grid")
    eco_df = pd.read_parquet(eco_full_cache)
else:
    log("Ecoregion spatial join (full TX, STRtree)...")
    with open(eco_cache) as f: eco_data = json.load(f)
    eco_polys = [(f['properties']['US_L3CODE'], f['properties']['US_L3NAME'], shape(f['geometry']))
                 for f in eco_data['features'] if f.get('geometry') is not None]
    eco_shapes = [p for _,_,p in eco_polys]
    tree = STRtree(eco_shapes)
    codes, names = [], []
    t0=time.time()
    for i,(lat,lon) in enumerate(zip(lats,lons)):
        pt = Point(lon,lat)
        cand = tree.query(pt, predicate='covered_by')
        if len(cand)>0:
            idx=cand[0]; codes.append(eco_polys[idx][0]); names.append(eco_polys[idx][1])
        else:
            nidx=tree.nearest(pt); codes.append(eco_polys[nidx][0]); names.append(eco_polys[nidx][1])
        if (i+1)%200000==0:
            log(f"  ecoregion: {i+1:,}/{n:,}  ({time.time()-t0:.0f}s elapsed)")
    eco_df = pd.DataFrame({'h3_cell':h3s,'ecoregion_code':codes,'ecoregion_name':names})
    eco_df.to_parquet(eco_full_cache, index=False)
log(f"  ecoregion done: {eco_df['ecoregion_code'].nunique()} unique regions")

# ── 3. ELEVATION + SLOPE/ASPECT (Copernicus GLO-30, tiled raster + gradient) ──
elev_full_cache = OUT_DIR/"elevation_slope_aspect_full.parquet"
if elev_full_cache.exists():
    log("[skip] elevation/slope/aspect already computed for full grid")
    esa_df = pd.read_parquet(elev_full_cache)
else:
    log("Elevation + slope/aspect (Copernicus GLO-30, tiled)...")
    COPERNICUS_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
    def cop_url(lat_tile, lon_tile):
        nm = f"Copernicus_DSM_COG_10_N{lat_tile:02d}_00_W{lon_tile:03d}_00_DEM"
        return f"/vsicurl/{COPERNICUS_BASE}/{nm}/{nm}.tif"
    lat_tiles = np.floor(lats).astype(int)
    lon_tiles = np.ceil(np.abs(lons)).astype(int)
    unique_tiles = sorted(set(zip(lat_tiles, lon_tiles)))
    elevations = np.full(n, np.nan); slopes = np.full(n, np.nan); aspects = np.full(n, np.nan)
    log(f"  {len(unique_tiles)} unique 1x1 deg tiles")
    for ti,(lt,lnt) in enumerate(unique_tiles):
        url = cop_url(lt, lnt)
        mask = (lat_tiles==lt) & (lon_tiles==lnt)
        idxs = np.where(mask)[0]
        try:
            with rasterio.open(url) as ds:
                data = ds.read(1).astype(np.float32)
                data[data <= -9999] = np.nan
                # pixel size in meters (approx, GLO-30 ~30m)
                px_deg = ds.transform.a; py_deg = -ds.transform.e
                px_m = px_deg * 111320 * np.cos(np.radians(lt+0.5))
                py_m = py_deg * 111320
                gy, gx = np.gradient(data, py_m, px_m)
                slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
                aspect = (np.degrees(np.arctan2(-gx, gy)) + 360) % 360
                coords = [(float(lons[i]), float(lats[i])) for i in idxs]
                rows, cols = rowcol(ds.transform, [c[0] for c in coords], [c[1] for c in coords])
                rows = np.clip(np.asarray(rows,dtype=int), 0, data.shape[0]-1)
                cols = np.clip(np.asarray(cols,dtype=int), 0, data.shape[1]-1)
                elevations[idxs] = data[rows, cols]
                slopes[idxs] = slope[rows, cols]
                aspects[idxs] = aspect[rows, cols]
        except Exception:
            pass
        if (ti+1) % 20 == 0:
            done=(~np.isnan(elevations)).sum()
            log(f"  tile {ti+1}/{len(unique_tiles)} | {done:,}/{n:,} elevations")
    esa_df = pd.DataFrame({'h3_cell':h3s,'elevation_m':elevations,'slope_deg':slopes,'aspect_deg':aspects})
    med_e, med_s, med_a = np.nanmedian(elevations), np.nanmedian(slopes), np.nanmedian(aspects)
    esa_df['elevation_m']=esa_df['elevation_m'].fillna(med_e)
    esa_df['slope_deg']=esa_df['slope_deg'].fillna(med_s)
    esa_df['aspect_deg']=esa_df['aspect_deg'].fillna(med_a)
    esa_df.to_parquet(elev_full_cache, index=False)
log(f"  elevation: mean={esa_df.elevation_m.mean():.0f}m  slope: mean={esa_df.slope_deg.mean():.1f}deg")

# ── 4. ROAD DISTANCE (Census TIGER, reuse cached shapefile, vectorized STRtree) ──
road_full_cache = OUT_DIR/"road_dist_full.parquet"
if road_full_cache.exists():
    log("[skip] road distance already computed for full grid")
    road_df = pd.read_parquet(road_full_cache)
else:
    log("Road distance (TIGER 2022, vectorized)...")
    tiger_cache = CACHE/"tl_2022_48_prisecroads.zip"
    meta, _, geom_wkb, fields = pyogrio.raw.read(f"zip://{tiger_cache}")
    road_geoms_wgs = from_wkb(geom_wkb)
    def proj_geom(g): return shp_transform(lambda x,y: to_utm.transform(x,y), g)
    road_geoms_utm = [proj_geom(g) for g in road_geoms_wgs if g is not None]
    road_tree = STRtree(road_geoms_utm)
    xs_utm, ys_utm = to_utm.transform(lons, lats)
    pts_utm = shapely.points(xs_utm, ys_utm)
    nearest_idxs = road_tree.nearest(pts_utm)
    nearest_roads = np.array(road_geoms_utm, dtype=object)[nearest_idxs]
    road_dists_m = shapely.distance(pts_utm, nearest_roads)
    road_df = pd.DataFrame({'h3_cell':h3s, 'road_dist_km': (road_dists_m/1000).round(3)})
    road_df.to_parquet(road_full_cache, index=False)
log(f"  road_dist_km: mean={road_df.road_dist_km.mean():.2f}km")

# ── 5. CBD / CBH (LANDFIRE rasters, direct sample -- works anywhere) ──
cbdcbh_cache = OUT_DIR/"cbd_cbh_full.parquet"
if cbdcbh_cache.exists():
    log("[skip] CBD/CBH already computed for full grid")
    cbdcbh_df = pd.read_parquet(cbdcbh_cache)
else:
    log("CBD/CBH (LANDFIRE raster sample, full grid)...")
    def sample_raster(tif_path, lats, lons, nodata_fill=0.0):
        with rasterio.open(tif_path) as src:
            xs, ys = warp_transform(src_crs='EPSG:4326', dst_crs=src.crs, xs=lons.tolist(), ys=lats.tolist())
            rows, cols = rowcol(src.transform, xs, ys)
            rows = np.asarray(rows, dtype=int); cols = np.asarray(cols, dtype=int)
            valid = (rows>=0)&(rows<src.height)&(cols>=0)&(cols<src.width)
            nodata = src.nodata
            values = np.full(len(lats), nodata_fill, dtype=np.float32)
            valid_idx = np.where(valid)[0]
            coords = [(float(xs[i]), float(ys[i])) for i in valid_idx]
            sampled = np.empty(len(valid_idx), dtype=np.float32)
            for s in range(0, len(coords), 50_000):
                sampled[s:s+50_000] = [v[0] for v in src.sample(coords[s:s+50_000])]
            values[valid_idx] = sampled
            if nodata is not None: values[values==nodata]=nodata_fill
            values[~valid]=nodata_fill
            values = np.where(np.isnan(values), nodata_fill, values)
        return values
    cbd = sample_raster(CBD_TIF, lats, lons) / 100.0   # LANDFIRE CBD units: kg/m3 * 100
    cbh = sample_raster(CBH_TIF, lats, lons) / 10.0    # LANDFIRE CBH units: m * 10
    cbdcbh_df = pd.DataFrame({'h3_cell':h3s, 'cbd':cbd, 'cbh':cbh})
    cbdcbh_df.to_parquet(cbdcbh_cache, index=False)
log(f"  cbd: mean={cbdcbh_df.cbd.mean():.4f}  cbh: mean={cbdcbh_df.cbh.mean():.2f}")

# ── 6. ASSEMBLE MASTER ──
log("Assembling master static-features file...")
eco_map = {v:i for i,v in enumerate(sorted(eco_df['ecoregion_code'].fillna('Unknown').unique()))}
eco_df['ecoregion_id'] = eco_df['ecoregion_code'].fillna('Unknown').map(eco_map).astype(int)

master = df.merge(eco_df[['h3_cell','ecoregion_id','ecoregion_name']], on='h3_cell', how='left')
master = master.merge(esa_df, on='h3_cell', how='left')
master = master.merge(road_df, on='h3_cell', how='left')
master = master.merge(cbdcbh_df, on='h3_cell', how='left')

out = OUT_DIR/"tx_static_master.parquet"
master.to_parquet(out, index=False)
log(f"\nSaved MASTER: {out}")
log(f"  {len(master):,} cells x {len(master.columns)} cols (FULL Texas, no coverage gap)")
log(f"  columns: {list(master.columns)}")
log(f"  NaN check:\n{master.isna().sum()[master.isna().sum()>0]}")
log("STEP 4 complete.")
