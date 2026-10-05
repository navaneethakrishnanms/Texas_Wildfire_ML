# PRD — TDIS Daily Ignition Model · Section 3: Input Data Requirements
*(Official TDIS PRD requirement-table style: ID · Requirement · Source/Access ·
Priority · Acceptance Criteria. Priority P0 = blocks training; P1 = full feature
set; P2 = enhancement. Status in Acceptance Criteria: ✅ done · 🔶 in progress /
to build · ⬇ downloading. Static/label sources give the on-server path (+
origin catalog); dynamic sources give the live endpoint + access method so a
data-engineering team can build the pull — full pull specs in §3.4.)*

## 3.1 Fire Labels and Training Target

| ID | Requirement | Source / Access | Priority | Acceptance Criteria |
|---|---|---|---|---|
| DR-LBL-01 | Ingest FPA-FOD incidents (cause, size, discovery date), TX/NM/OK 2014–2020. | On server: `30%_Rev2/data/ignition/20*_FPA_FOD_cons.csv`. Origin: USFS RDS-2013-0009.6 + Zenodo 8381129. | P0 | Keyed to H3 res-8 cell+day; cause+size populated; counts reconcile 54,911 TX / 7,790 NM / 8,894 OK. TX times 86.9% imputed (daily-only). **✅ on disk** |
| DR-LBL-02 | Ingest VIIRS S-NPP/N20/N21 375 m active-fire detections 2018–2026, full three-state footprint. | On server (TX): `TDIS_Forecast/data/labels_viirs/viirs_tx_h3.parquet`. Origin: NASA FIRMS (see §3.4-A). | P0 | Full-state NM/OK (not bbox spillover); res-8 keyed; TX 1.62M. **✅ TX · 🔶 NM/OK re-pull** |
| DR-LBL-03 | Apply per-state flare exclusion (episode-based ≥6 episodes/365 d, gap≤2 d) before fusion. | Derived. On server: `New_Training817_moredata/flare_cells_v2.parquet` (TX); `NM_OK_TexRev/data/flare_cells_{nm,ok}.parquet`. Builder: `NM_OK_TexRev/scripts/06_build_flare_lists_nm_ok.py`. | P0 | TX v2 538 / NM 1,956 / OK 260 cells; NM Permian subset manually reviewed. **✅ TX · ✅ NM/OK built · ⚠ NM Permian review pending** |
| DR-LBL-04 | Produce fused (cell,day) VIIRS+FPA flare-filtered ignition labels — training target. | Derived (project output). On server (TX ref): `TDIS_Forecast/data/labels_fused/ignitions_daily_tx.parquet`. | P0 | Every fire cell-day labeled; `time_source` per row (TX 96% VIIRS / 4% imputed-FPA); zero flare cells. **✅ TX · 🔶 NM/OK (§5 output)** |

## 3.2 Static Features (all states re-extracted from one CONUS source)

| ID | Requirement | Source / Access | Priority | Acceptance Criteria |
|---|---|---|---|---|
| DR-STA-01 | FSim 270 m → `avg_burn_prob`, `cfl`, `flep4`. | USFS RDS-2016-0034-3 (1.64 GB zip; catalog `fs.usda.gov/rds`). One-time download → CONUS raster on server. | P0 | All cells; per-state `avg_burn_prob ∈ [0,1]` assertion passes (not legacy TX 0–11). **⬇** |
| DR-STA-02 | WHP 2023 → `whp`. | USFS RDS-2015-0047 (catalog / firelab.org WHP-2023). One-time download. | P0 | All cells; integer-class range validated per state. **🔶 URL to resolve** |
| DR-STA-03 | LANDFIRE LF2022 canopy → `cbd`, `cbh`. | landfire.gov data distribution. One-time download. | P1 | Populated; null rate logged, in tolerance. **🔶 URL to resolve** |
| DR-STA-04 | 3DEP 1 arc-sec → `elevation_m`, `slope_deg`, `aspect_deg`. | `prd-tnm.s3.amazonaws.com` (scripted tile fetch; public S3). | P0 | All 3 fields per cell; slope/aspect derived from DEM; ranges valid. **🔶 scripted** |
| DR-STA-05 | EPA Level-3 ecoregions → `ecoregion_id`. | `dmap-prod-oms-edc.s3.us-east-1.amazonaws.com/ORD/Ecoregions/us/us_eco_l3.zip` (public S3). | P0 | Every cell coded (national codes); cross-state seam test passes. **⬇** |
| DR-STA-06 | TIGER 2023 primary/secondary roads → `road_dist_km`. | `www2.census.gov/geo/tiger/TIGER2023/PRISECROADS/` (states NM 35 / OK 40 / TX 48). | P0 | Distance-to-road for all cells, all states. **⬇** |
| DR-STA-07 | HIFLD transmission lines → `powerline_dist_km`. | `hifld-geoplatform.opendata.arcgis.com` (national transmission lines). | P1 | Distance-to-powerline for all cells. **🔶 URL to resolve** |
| DR-STA-08 | Census 2023 counties + states → FIPS join, state clip. | `www2.census.gov/geo/tiger/GENZ2023/shp/`. | P1 | Every cell FIPS-tagged; output clipped to 3-state boundary. **⬇** |
| DR-STA-09 | Per-state unit-range assertions on all static features (DoD gate). | Derived check (single extraction script). | P0 | `avg_burn_prob ∈ [0,1]`, WHP integer classes, terrain ranges, etc. pass per state; out-of-range blocks release (§9). |

## 3.3 Dynamic Features (live pull — see §3.4 for full pipeline specs)

| ID | Requirement | Source / Access | Priority | Acceptance Criteria |
|---|---|---|---|---|
| DR-DYN-01 | 12Z day-before HRRR daily peaks (`gust_max`, `vpd_max`, `tmp_max`, `mstav_min`; leads F27/30/33/36). | AWS S3 `noaa-hrrr-bdp-pds` (public, no auth) via Herbie; product `sfc`. On server: `data/weather_hrrr_forecast/hrrr_12z_f{24..45}/`. **§3.4-B** | P0 | Peaks per cell-day; all TX / ~60% NM / ~90% OK; out-of-bbox flagged. **✅ on disk · v1 bbox-limited** |
| DR-DYN-02 | USDM weekly `drought_score` (0–4+) by county. | USDM Data Services REST API `usdmdataservices.unl.edu/api/` (JSON, weekly Thu, no auth). Cache: `data/static_features/drought_weekly.parquet`. **§3.4-C** | P1 | Weekly by FIPS, all 3 states. **🔶 extend pull to NM/OK** |
| DR-DYN-03 | gridMET NFDRS `fm100` (100-h dead-fuel moisture) 2014–2026, full cell coverage. | gridMET (climatologylab) OPeNDAP + HTTPS fallback (no auth). Builder: `scripts/30_build_fm100_tx_full.py`. **§3.4-D** | P1 | Full coverage all cells; TX rebuild vs 23.2% legacy. **✅ NM/OK · 🔶 TX running** |
| DR-DYN-04 | WorldPop 2020 1 km `pop_density`. | worldpop.org (`data.worldpop.org` GIS). On server: `usa_pop_2020_1km.tif`. Annual vintage → effectively static. | P0 | All cells, all states; P0 = top ignition driver. **✅ done** |

---

## 3.4 Dynamic-data pull specifications (for the pipeline engineering team)

These are the live/updating sources. Static features (§3.2) are one-time
downloads; the labels (§3.1) update only when FPA/VIIRS re-pulled. The four
below are what a production pipeline must fetch on a schedule.

### 3.4-A · VIIRS active fire (FIRMS)
- **Endpoint:** NASA FIRMS Area API — `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{bbox}/{days}/{start}`; sources `VIIRS_SNPP_SP`/`VIIRS_NOAA20_SP` (archive) + `..._NRT` (recent ~2 months). Map key: `TDIS_Forecast/firms_map_key.txt`.
- **Auth:** free FIRMS map key (rate-limited; the lab script honors a transaction budget).
- **Cadence:** archive is static; NRT updates within hours of overpass (~2/day per satellite). Daily pipeline pulls NRT; monthly reconcile against SP archive.
- **Params:** full three-state bbox (not TX-only); H3 res-8 assignment via `h3.latlng_to_cell`; keep `frp`, `confidence`, `acq_date/time`.
- **Reference script:** `TDIS_Forecast/scripts/01_download_viirs.py`.

### 3.4-B · HRRR 12Z forecast (the model's live weather)
- **Endpoint:** AWS S3 `s3://noaa-hrrr-bdp-pds/hrrr.{YYYYMMDD}/conus/hrrr.t12z.wrfsfcf{FF}.grib2` (public, no auth). Pulled via **Herbie** (`Herbie(init, model="hrrr", product="sfc", fxx=lead)`).
- **Init / leads:** 12Z run of the day *before* the target day; leads **F24, F27, F30, F33, F36** (F39–F45 exist only Dec 2020+, not needed for daily peaks).
- **GRIB subset (byte-range):** `:(TMP|DPT):2 m above ground|:WIND:10 m above ground|:GUST:surface|:MSTAV:` — pull only these messages. Derive **VPD** from TMP + DPT (Tetens); **gust_max/vpd_max/tmp_max** = max over the 4 leads, **mstav_min** = min.
- **Cadence / latency:** HRRR runs hourly; the 12Z cycle is available on S3 ~90 min after cycle time. Pipeline pulls once daily after the 12Z run posts.
- **Regrid:** HRRR native → nearest gridpoint to each H3 cell centroid (cKDTree; never float-equality join).
- **Reference script:** `90%_ig_dec/CA_rev5/scripts/01_download_hrrr_12z_ca.py`.

### 3.4-C · U.S. Drought Monitor (weekly)
- **Endpoint:** USDM Data Services REST API — base `https://usdmdataservices.unl.edu/api/`; county series e.g. `CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent?aoi={FIPS}&startdate={d}&enddate={d}&statisticsType=1` → JSON. (Alternatively bulk GIS shapefiles from `droughtmonitor.unl.edu`.)
- **Auth:** none.
- **Cadence:** released **weekly, Thursday ~08:30 ET**. Pipeline polls weekly; forward-fill the current week's value onto each day until the next release.
- **Params:** all NM/OK/TX county FIPS; map `drought_score` (0=None … 4=D4) to the H3 cell via its county FIPS.
- **Note:** no lab pull script exists yet — this is a net-new pipeline step (TX cache was built ad hoc).

### 3.4-D · gridMET fm100 (dead-fuel moisture)
- **Endpoint (primary):** OPeNDAP — `http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_fm100_1979_CurrentYear_CONUS.nc` (open with xarray/netCDF4, subset by bbox + time).
- **Endpoint (fallback):** HTTPS per-year NetCDF — `https://www.northwestknowledge.net/metdata/data/fm100_{year}.nc`.
- **Auth:** none.
- **Cadence / latency:** daily 4 km CONUS, 1979–present; gridMET publishes with a **~1–4 day lag** — so `fm100` is a *lag-safe* feature (use the most recent available day ≤ forecast issue date; the sub-daily models use `fm100_5d_min` over D-6…D-2).
- **Params:** subset to the three-state bbox; extract variable `dead_fuel_moisture_100hr`; assign to H3 cells; output `[date_utc, h3_cell, fm100]`.
- **Reference scripts:** `TDIS_Forecast/scripts/30_build_fm100_tx_full.py`, `NM_OK_TexRev/scripts/11_build_fm100_nm_ok.py`.

*Pipeline scheduling summary:* HRRR 12Z = **daily** (post-12Z-cycle); USDM =
**weekly** (Thu, forward-filled); fm100 = **daily with ~2-day lag** (lag-safe);
VIIRS NRT = **daily** (monthly SP reconcile). Static features + WorldPop =
one-time / annual re-extraction. All join to the H3 res-8 grid keyed `h3_cell`.
