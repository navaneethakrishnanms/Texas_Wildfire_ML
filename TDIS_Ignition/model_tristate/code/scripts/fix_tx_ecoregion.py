"""
Fix: TX ecoregion_id in train_tristate.parquet uses a LOCAL 0-19 re-index
(alphabetized TX ecoregion names), while NM/OK use the genuine national
US_L3CODE (via gpd.sjoin against us_eco_l3.shp, same method 02_build_static_masters.py
uses for NM/OK). This makes ecoregion_id incommensurate across states in the
tri-state model -- code 8 means "Cross Timbers" in TX but an unrelated US_L3CODE
region in NM/OK. Recompute TX's ecoregion_id on the SAME national scale.

Output: train_tristate_v2.parquet (identical to v1 except ecoregion_id for TX rows
is replaced by the national US_L3CODE; also fills the 202,078 previously-null
TX ecoregion_id rows where possible).
"""
import pyproj
pyproj.datadir.set_data_dir("/home/mte1224/mambaforge/envs/UAI2526/share/proj")  # env's PROJ_DATA not auto-detected
import geopandas as gpd
import h3
import pandas as pd
from pathlib import Path
from shapely.geometry import Point

AE = Path("/net/cven-mosta-nas.engr.tamu.edu/volume2/NASdata2/miguel_shared/alphaearth_nds/TDIS_Forecast")
REV5 = AE / "rev5_ignition_fireweather" / "ignition"
RAW_ECO = AE / "NM_OK_TexRev" / "data" / "raw" / "ecoregions_l3"

def log(m): print(m, flush=True)

log("Loading tri-state table...")
tri = pd.read_parquet(REV5 / "data" / "train_tristate.parquet")
log(f"  {len(tri):,} rows, states: {tri.state.value_counts().to_dict()}")

tx_mask = tri.state == "tx"
log(f"TX rows: {tx_mask.sum():,} | current ecoregion_id range "
    f"{tri.loc[tx_mask,'ecoregion_id'].min()}-{tri.loc[tx_mask,'ecoregion_id'].max()} "
    f"(LOCAL scale, to be replaced)")

# unique TX cells (join is per-cell, not per-row -- cheaper)
# NOTE: train_tristate.parquet's lat/lon columns are 100% NULL for TX rows (a
# separate, minor bug -- never attached during table build). Derive lat/lon
# directly from h3_cell via the H3 library instead -- more reliable anyway.
tx_cells = tri.loc[tx_mask, ["h3_cell"]].drop_duplicates("h3_cell").reset_index(drop=True)
latlon = tx_cells["h3_cell"].apply(lambda c: h3.cell_to_latlng(c))
tx_cells["lat"] = [p[0] for p in latlon]
tx_cells["lon"] = [p[1] for p in latlon]
log(f"Unique TX cells needing ecoregion lookup: {len(tx_cells):,} "
    f"(lat/lon derived from h3_cell, not the table's null columns)")

log("Loading EPA Level-3 ecoregions shapefile...")
eco = gpd.read_file(RAW_ECO).to_crs("EPSG:4326")
log(f"  {len(eco)} ecoregion polygons, US_L3CODE range "
    f"{pd.to_numeric(eco['US_L3CODE'], errors='coerce').min()}-"
    f"{pd.to_numeric(eco['US_L3CODE'], errors='coerce').max()}")

gpts = gpd.GeoDataFrame(
    tx_cells.reset_index(drop=True),
    geometry=[Point(xy) for xy in zip(tx_cells.lon, tx_cells.lat)],
    crs="EPSG:4326",
)
log("Spatial join (within) -- this is the same method used for NM/OK...")
j = gpd.sjoin(gpts, eco[["US_L3CODE", "geometry"]], how="left", predicate="within")
j = j[~j.index.duplicated(keep="first")]
tx_cells = tx_cells.reset_index(drop=True)
tx_cells["ecoregion_id_national"] = pd.to_numeric(j["US_L3CODE"], errors="coerce").reindex(tx_cells.index).values

n_null = tx_cells["ecoregion_id_national"].isna().sum()
log(f"  join coverage: {100*(1-n_null/len(tx_cells)):.2f}% "
    f"({len(tx_cells)-n_null:,}/{len(tx_cells):,} cells matched; "
    f"{n_null:,} unmatched -- likely coastal/border edge cases)")

# NM/OK codes span 20-40 (confirmed earlier); report TX's new range + any overlap
tx_codes = set(tx_cells["ecoregion_id_national"].dropna().unique())
nm_codes = set(tri.loc[tri.state == "nm", "ecoregion_id"].unique())
ok_codes = set(tri.loc[tri.state == "ok", "ecoregion_id"].unique())
log(f"  TX national codes: {sorted(tx_codes)}")
log(f"  TX∩NM overlap (expected now, real shared biomes): {sorted(tx_codes & nm_codes)}")
log(f"  TX∩OK overlap (expected now, real shared biomes): {sorted(tx_codes & ok_codes)}")

log("Merging corrected ecoregion_id back into the tri-state table...")
tri = tri.merge(tx_cells[["h3_cell", "ecoregion_id_national"]], on="h3_cell", how="left")
# for TX rows, replace ecoregion_id with the national code; leave NM/OK untouched
tri.loc[tx_mask, "ecoregion_id"] = tri.loc[tx_mask, "ecoregion_id_national"]
tri = tri.drop(columns=["ecoregion_id_national"])

n_null_after = tri.loc[tx_mask, "ecoregion_id"].isna().sum()
log(f"TX ecoregion_id nulls AFTER fix: {n_null_after:,} / {tx_mask.sum():,} "
    f"(base table pre-fix had ~202,078 null on the old local-scale column)")

out = REV5 / "data" / "train_tristate_v2.parquet"
tri.to_parquet(out, index=False)
log(f"\nSaved {out} ({len(tri):,} rows)")
log("ecoregion_id is now on ONE unified national US_L3CODE scale across TX/NM/OK.")
