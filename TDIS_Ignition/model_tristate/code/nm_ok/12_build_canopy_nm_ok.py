"""
NM_OK_TexRev — canopy (cbd/cbh) for NM + OK, filling the pass-2 placeholder
from 02_build_static_masters.py. LANDFIRE LF2022 CBD/CBH are published as
CONUS-wide ArcGIS ImageServer layers (edcintl.cr.usgs.gov) with a
getSamples point-query endpoint -- no full raster download needed, just
batched point requests (500/batch, tested working via POST).

Units match TX's convention (scripts/04_build_full_tx_static.py):
CBD raw units /100 = kg/m^3, CBH raw units /10 = m.

Output: NM_OK_TexRev/data/canopy_{nm,ok}.parquet [h3_cell, cbd, cbh]
"""
import time
import numpy as np, pandas as pd, requests, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = "https://edcintl.cr.usgs.gov/arcgis/rest/services/Landfire_LF2022"
BATCH = 500


def log(m): print(m, flush=True)


def sample_layer(layer, lons, lats):
    vals = np.full(len(lons), np.nan)
    url = f"{BASE}/{layer}/ImageServer/getSamples"
    for s in range(0, len(lons), BATCH):
        pts = list(zip(lons[s:s+BATCH].tolist(), lats[s:s+BATCH].tolist()))
        geom = {'points': pts, 'spatialReference': {'wkid': 4326}}
        for attempt in range(4):
            try:
                r = requests.post(url, data={'geometry': json.dumps(geom),
                                              'geometryType': 'esriGeometryMultipoint',
                                              'returnFirstValueOnly': 'true', 'f': 'json'},
                                   timeout=60)
                d = r.json()
                if 'samples' in d:
                    for samp in d['samples']:
                        i = s + samp['locationId']
                        try:
                            vals[i] = float(samp['value'])
                        except (ValueError, TypeError):
                            pass
                    break
                else:
                    time.sleep(5 * (attempt + 1))
            except Exception as e:
                time.sleep(5 * (attempt + 1))
        if (s // BATCH + 1) % 100 == 0:
            log(f"    {s+BATCH:,}/{len(lons):,} points ({layer})")
    return vals


def build_for_state(state):
    out = ROOT / "data" / f"canopy_{state}.parquet"
    if out.exists():
        log(f"[skip] {out.name} exists"); return

    st = pd.read_parquet(ROOT / "data" / f"static_master_{state}.parquet",
                          columns=['h3_cell', 'lat', 'lon'])
    log(f"[{state}] {len(st):,} cells")

    cbd_raw = sample_layer('LF2022_CBD_CONUS', st.lon.values, st.lat.values)
    cbh_raw = sample_layer('LF2022_CBH_CONUS', st.lon.values, st.lat.values)

    df = pd.DataFrame({'h3_cell': st.h3_cell.values,
                        'cbd': np.nan_to_num(cbd_raw, nan=0.0) / 100.0,
                        'cbh': np.nan_to_num(cbh_raw, nan=0.0) / 10.0})
    df.to_parquet(out, index=False)
    log(f"[{state}] cbd mean={df.cbd.mean():.4f} cbh mean={df.cbh.mean():.2f} "
        f"(nan->0 for {np.isnan(cbd_raw).sum():,} cbd / {np.isnan(cbh_raw).sum():,} cbh cells)")


if __name__ == '__main__':
    for state in ['nm', 'ok']:
        build_for_state(state)
