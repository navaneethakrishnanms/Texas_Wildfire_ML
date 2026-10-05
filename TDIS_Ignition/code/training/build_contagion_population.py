"""
New_Training817_moredata — build TWO new candidate features:

1. fire_nearby_3d: count of fire events within the cell's 1-ring
   neighborhood (7 res-8 cells, ~1.4 km radius) over the PRIOR 3 days
   (d-1, d-2, d-3 — strictly before the target day, so no leakage and it
   IS forecastable operationally: you'd know yesterday's/last week's real
   VIIRS detections when producing tomorrow's forecast). Targets fire
   clustering/reburn/multi-day complexes -- nothing in the current 22
   features captures this; it's the first genuinely new MECHANISM tried
   since the original feature set was locked.

2. pop_density: WorldPop 2020 1km population raster (public, no auth),
   nearest-pixel join to each res-8 cell. Targets human-caused ignitions
   (most small fires) -- distinct signal from road_dist_km (a road can be
   remote-but-well-built OR run through a dense suburb).

Output: fire_nearby_3d_feature.parquet [h3_cell, date, fire_nearby_3d]
        pop_density_feature.parquet    [h3_cell, pop_density]
"""
import time, warnings
from collections import defaultdict
import numpy as np, pandas as pd, h3
from pathlib import Path
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
TF = HERE.parent


def log(m): print(m, flush=True)


def build_contagion():
    lab = pd.read_parquet(TF / "data/labels_fused/ignitions_daily_tx.parquet",
                           columns=['h3_cell', 'date'])
    lab['date'] = pd.to_datetime(lab['date']).dt.normalize()
    log(f"positive fire events: {len(lab):,}")

    # neighbor cache: cell -> its 1-ring (7 cells incl. self)
    uniq_cells = lab['h3_cell'].unique()
    nbrs = {c: h3.grid_disk(c, 1) for c in uniq_cells}

    # danger[date] = Counter of cell -> how many nearby fires "reach" that date
    danger = defaultdict(lambda: defaultdict(int))
    t0 = time.time()
    for i, (c, d) in enumerate(zip(lab['h3_cell'].values, lab['date'].values)):
        d = pd.Timestamp(d)
        for offset in (1, 2, 3):
            dd = d + pd.Timedelta(days=offset)
            for n in nbrs[c]:
                danger[dd][n] += 1
        if (i + 1) % 200000 == 0:
            log(f"  {i+1:,}/{len(lab):,} events ({(time.time()-t0)/60:.1f} min)")
    log(f"built danger index for {len(danger):,} dates ({(time.time()-t0)/60:.1f} min)")

    # training table's actual (cell, date) pairs -- only compute what's needed
    tr = pd.read_parquet(TF / "tdis_train_daily_hrrr.parquet", columns=['h3_cell', 'date'])
    tr['date'] = pd.to_datetime(tr['date']).dt.normalize()
    tr = tr.drop_duplicates()
    log(f"training (cell,date) pairs needing the feature: {len(tr):,}")

    vals = np.zeros(len(tr), dtype=np.int16)
    for i, (c, d) in enumerate(zip(tr['h3_cell'].values, tr['date'].values)):
        dmap = danger.get(pd.Timestamp(d))
        if dmap:
            vals[i] = dmap.get(c, 0)
        if (i + 1) % 500000 == 0:
            log(f"  lookup {i+1:,}/{len(tr):,}")
    tr['fire_nearby_3d'] = vals
    tr.to_parquet(HERE / "fire_nearby_3d_feature.parquet", index=False)
    log(f"nonzero rows: {(vals>0).sum():,} / {len(tr):,} "
        f"({(vals>0).mean()*100:.2f}%)")
    log("saved fire_nearby_3d_feature.parquet")


def build_population():
    import rasterio
    from rasterio.windows import from_bounds
    TX = dict(lat_min=25.75, lat_max=36.65, lon_min=-106.70, lon_max=-93.40)
    with rasterio.open(HERE / "usa_pop_2020_1km.tif") as src:
        win = from_bounds(TX['lon_min'], TX['lat_min'], TX['lon_max'], TX['lat_max'], src.transform)
        arr = src.read(1, window=win)
        wt = src.window_transform(win)
        arr = np.where(arr == src.nodata, 0, arr)
        arr = np.clip(arr, 0, None)
    rows, cols = np.indices(arr.shape)
    xs, ys = rasterio.transform.xy(wt, rows.ravel(), cols.ravel())
    pop_pts = pd.DataFrame(dict(lon=xs, lat=ys, pop=arr.ravel()))

    st = pd.read_parquet(TF / "data/static_features/tx_static_master.parquet",
                          columns=['h3_cell', 'lat', 'lon'])
    tree = cKDTree(pop_pts[['lon', 'lat']].values)
    _, idx = tree.query(st[['lon', 'lat']].values)
    st['pop_density'] = pop_pts['pop'].values[idx]
    st[['h3_cell', 'pop_density']].to_parquet(HERE / "pop_density_feature.parquet", index=False)
    log(f"pop_density: mean={st['pop_density'].mean():.2f} p95={st['pop_density'].quantile(.95):.2f} "
        f"max={st['pop_density'].max():.1f}")
    log("saved pop_density_feature.parquet")


if __name__ == '__main__':
    log("=== Building population density feature ===")
    build_population()
    log("\n=== Building fire contagion feature ===")
    build_contagion()
