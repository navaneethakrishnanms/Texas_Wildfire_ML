"""
NM_OK_TexRev — powerline_dist_km for NM + OK, filling the pass-2
placeholder from 02_build_static_masters.py. HIFLD Electric Power
Transmission Lines is published as a public ArcGIS FeatureServer
(services1.arcgis.com/Hp6G80Pky0om7QvQ) -- paginated query (2000
records/page, confirmed via resultOffset) within the NM+OK bbox, then
same STRtree nearest-distance approach as TX's road_dist_km computation
(scripts/04_build_full_tx_static.py).

Output: NM_OK_TexRev/data/hifld_transmission_nmok.geojson (raw pull, shared)
        NM_OK_TexRev/data/powerline_dist_{nm,ok}.parquet [h3_cell, powerline_dist_km]
"""
import time, json
import numpy as np, pandas as pd, requests
from pathlib import Path
import shapely
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree
import pyproj
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
URL = ("https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/"
       "Electric_Power_Transmission_Lines/FeatureServer/0/query")
BBOX = "-109.10,31.00,-94.30,37.10"
PAGE = 2000


def log(m): print(m, flush=True)


def pull_lines():
    out = ROOT / "data" / "hifld_transmission_nmok.geojson"
    if out.exists():
        log("[skip] hifld_transmission_nmok.geojson exists"); return json.load(open(out))
    features = []
    offset = 0
    while True:
        params = dict(where="1=1", outFields="OBJECTID", geometryType="esriGeometryEnvelope",
                      geometry=BBOX, inSR=4326, spatialRel="esriSpatialRelIntersects",
                      returnGeometry="true", outSR=4326, resultOffset=offset,
                      resultRecordCount=PAGE, f="geojson")
        for attempt in range(4):
            try:
                r = requests.get(URL, params=params, timeout=60)
                d = r.json()
                break
            except Exception as e:
                log(f"  retry offset={offset}: {str(e)[:60]}"); time.sleep(5 * (attempt + 1))
        else:
            log(f"  FAILED offset={offset}, stopping"); break
        feats = d.get('features', [])
        features.extend(feats)
        log(f"  pulled {len(features):,} lines so far (offset={offset})")
        if len(feats) < PAGE:
            break
        offset += PAGE
        time.sleep(0.3)
    geo = {'type': 'FeatureCollection', 'features': features}
    json.dump(geo, open(out, 'w'))
    log(f"saved {len(features):,} transmission line segments -> {out.name}")
    return geo


def build_for_state(state, geo, to_utm):
    out = ROOT / "data" / f"powerline_dist_{state}.parquet"
    if out.exists():
        log(f"[skip] {out.name} exists"); return

    st = pd.read_parquet(ROOT / "data" / f"static_master_{state}.parquet",
                          columns=['h3_cell', 'lat', 'lon'])
    log(f"[{state}] {len(st):,} cells")

    def proj_geom(g): return shp_transform(lambda x, y: to_utm.transform(x, y), g)
    geoms_utm = []
    for f in geo['features']:
        if f.get('geometry') is None:
            continue
        g = shape(f['geometry'])
        try:
            geoms_utm.append(proj_geom(g))
        except Exception:
            continue
    tree = STRtree(geoms_utm)
    xs_utm, ys_utm = to_utm.transform(st.lon.values, st.lat.values)
    pts_utm = shapely.points(xs_utm, ys_utm)
    nearest_idxs = tree.nearest(pts_utm)
    nearest_lines = np.array(geoms_utm, dtype=object)[nearest_idxs]
    dists_m = shapely.distance(pts_utm, nearest_lines)
    df = pd.DataFrame({'h3_cell': st.h3_cell.values, 'powerline_dist_km': (dists_m / 1000).round(3)})
    df.to_parquet(out, index=False)
    log(f"[{state}] powerline_dist_km: mean={df.powerline_dist_km.mean():.2f}km")


if __name__ == '__main__':
    geo = pull_lines()
    p_wgs = pyproj.Proj(proj='latlong', ellps='WGS84')
    p_utm = pyproj.Proj(proj='utm', zone=13, ellps='WGS84')  # central UTM zone for NM/OK
    to_utm = Transformer.from_proj(p_wgs, p_utm, always_xy=True)
    for state in ['nm', 'ok']:
        build_for_state(state, geo, to_utm)
