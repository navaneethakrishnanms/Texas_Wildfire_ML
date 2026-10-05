"""
NM_OK_TexRev — Wildfire Hazard Potential (whp) for NM + OK, filling the
pass-2 placeholder from 02_build_static_masters.py. No WHP raster exists
anywhere on disk for ANY state (checked) -- TX's own whp values came from
an even earlier pre-baked source with no local raster to resample. Found
a live source instead: USFS's classified WHP is published as a public
ArcGIS ImageServer (migrated to imagery.geoplatform.gov/iipp), 0-9
integer classes -- confirmed matching TX's existing whp value range/
distribution (0-9, mode at 0, second peak in 4-6) via getSamples point
query, same method as canopy (script 12).

Output: NM_OK_TexRev/data/whp_{nm,ok}.parquet [h3_cell, whp]
"""
import time, json
import numpy as np, pandas as pd, requests
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
URL = ("https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/"
       "USFS_EDW_RMRS_WildfireHazardPotentialClassified/ImageServer/getSamples")
BATCH = 500


def log(m): print(m, flush=True)


def sample_whp(lons, lats):
    vals = np.full(len(lons), np.nan)
    for s in range(0, len(lons), BATCH):
        pts = list(zip(lons[s:s+BATCH].tolist(), lats[s:s+BATCH].tolist()))
        geom = {'points': pts, 'spatialReference': {'wkid': 4326}}
        for attempt in range(4):
            try:
                r = requests.post(URL, data={'geometry': json.dumps(geom),
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
            except Exception:
                time.sleep(5 * (attempt + 1))
        if (s // BATCH + 1) % 100 == 0:
            log(f"    {s+BATCH:,}/{len(lons):,} points")
    return vals


def build_for_state(state):
    out = ROOT / "data" / f"whp_{state}.parquet"
    if out.exists():
        log(f"[skip] {out.name} exists"); return

    st = pd.read_parquet(ROOT / "data" / f"static_master_{state}.parquet",
                          columns=['h3_cell', 'lat', 'lon'])
    log(f"[{state}] {len(st):,} cells")

    whp_raw = sample_whp(st.lon.values, st.lat.values)
    df = pd.DataFrame({'h3_cell': st.h3_cell.values, 'whp': np.nan_to_num(whp_raw, nan=0.0)})
    df.to_parquet(out, index=False)
    log(f"[{state}] whp value counts:\n{df.whp.value_counts().sort_index().to_string()}")


if __name__ == '__main__':
    for state in ['nm', 'ok']:
        build_for_state(state)
