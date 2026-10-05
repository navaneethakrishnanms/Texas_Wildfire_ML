"""
Build the corrected national-US_L3CODE ecoregion_id lookup for ALL TX H3 res-8
cells (not just the ~482K appearing in train_tristate.parquet) -- needed so
that population-replay/battery/tau scoring uses the SAME ecoregion scale the
v2 models were trained on. Persisted so training and scoring share one source.

Output: NM_OK_TexRev/data/tx_ecoregion_national.parquet [h3_cell, ecoregion_id]
"""
import pyproj
pyproj.datadir.set_data_dir("/home/mte1224/mambaforge/envs/UAI2526/share/proj")
import geopandas as gpd
import h3
import pandas as pd
from pathlib import Path
from shapely.geometry import Point

AE = Path("/net/cven-mosta-nas.engr.tamu.edu/volume2/NASdata2/miguel_shared/alphaearth_nds/TDIS_Forecast")
RAW_ECO = AE / "NM_OK_TexRev" / "data" / "raw" / "ecoregions_l3"
OUT = AE / "NM_OK_TexRev" / "data" / "tx_ecoregion_national.parquet"


def log(m): print(m, flush=True)


log("Loading full TX static master (1.7M cells)...")
st = pd.read_parquet(AE / "data" / "static_features" / "tx_static_master.parquet", columns=["h3_cell"])
log(f"  {len(st):,} TX cells")

latlon = st["h3_cell"].apply(lambda c: h3.cell_to_latlng(c))
st["lat"] = [p[0] for p in latlon]
st["lon"] = [p[1] for p in latlon]

log("Loading EPA Level-3 ecoregions shapefile...")
eco = gpd.read_file(RAW_ECO).to_crs("EPSG:4326")

gpts = gpd.GeoDataFrame(st, geometry=[Point(xy) for xy in zip(st.lon, st.lat)], crs="EPSG:4326")
log("Spatial join (within)...")
j = gpd.sjoin(gpts, eco[["US_L3CODE", "geometry"]], how="left", predicate="within")
j = j[~j.index.duplicated(keep="first")]
st["ecoregion_id"] = pd.to_numeric(j["US_L3CODE"], errors="coerce").reindex(st.index).values

n_null = st["ecoregion_id"].isna().sum()
log(f"  join coverage: {100*(1-n_null/len(st)):.2f}% ({len(st)-n_null:,}/{len(st):,} matched, "
    f"{n_null:,} unmatched -- Mexico/Gulf-of-Mexico cells outside US soil, expected)")

out = st[["h3_cell", "ecoregion_id"]]
out.to_parquet(OUT, index=False)
log(f"\nSaved {OUT} ({len(out):,} rows)")
