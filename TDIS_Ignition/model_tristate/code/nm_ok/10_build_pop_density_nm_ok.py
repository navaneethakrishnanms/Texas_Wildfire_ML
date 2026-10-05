"""
NM_OK_TexRev — population density for NM + OK (Phase: stakeholder feedback,
2026-09-11). Same recipe as TX's New_Training817_moredata/build_contagion_
population.py::build_population(), generalized: the source raster
(usa_pop_2020_1km.tif) is already national extent, so this is just a
different bounding window + a different static-master source per state.

Output: NM_OK_TexRev/data/pop_density_{nm,ok}.parquet [h3_cell, pop_density]
"""
import numpy as np, pandas as pd, rasterio
from pathlib import Path
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TF = ROOT.parent
POP_TIF = TF / "New_Training817_moredata" / "usa_pop_2020_1km.tif"

STATES = {
    'nm': dict(lat_min=31.0, lat_max=37.1, lon_min=-109.1, lon_max=-102.9),
    'ok': dict(lat_min=33.5, lat_max=37.1, lon_min=-103.1, lon_max=-94.3),
}


def log(m): print(m, flush=True)


def build_for_state(state):
    bbox = STATES[state]
    out = ROOT / "data" / f"pop_density_{state}.parquet"
    if out.exists():
        log(f"[skip] {out.name} exists"); return

    st = pd.read_parquet(ROOT / "data" / f"static_master_{state}.parquet",
                          columns=['h3_cell', 'lat', 'lon'])
    log(f"{state}: {len(st):,} cells")

    with rasterio.open(POP_TIF) as src:
        win = from_bounds(bbox['lon_min'], bbox['lat_min'], bbox['lon_max'], bbox['lat_max'],
                           src.transform)
        arr = src.read(1, window=win)
        wt = src.window_transform(win)
        arr = np.where(arr == src.nodata, 0, arr)
        arr = np.clip(arr, 0, None)
    rows, cols = np.indices(arr.shape)
    xs, ys = rasterio.transform.xy(wt, rows.ravel(), cols.ravel())
    pop_pts = pd.DataFrame(dict(lon=xs, lat=ys, pop=arr.ravel()))

    tree = cKDTree(pop_pts[['lon', 'lat']].values)
    _, idx = tree.query(st[['lon', 'lat']].values)
    st['pop_density'] = pop_pts['pop'].values[idx]
    st[['h3_cell', 'pop_density']].to_parquet(out, index=False)
    log(f"{state}: pop_density mean={st.pop_density.mean():.2f} "
        f"p95={st.pop_density.quantile(.95):.2f} max={st.pop_density.max():.1f}")
    log(f"saved {out}")


if __name__ == '__main__':
    for s in ['nm', 'ok']:
        build_for_state(s)
