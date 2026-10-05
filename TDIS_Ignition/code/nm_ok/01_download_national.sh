#!/bin/bash
# Stage 1 — national source downloads for NM_OK_TexRev (option A rebuild).
# All URLs verified with curl -I on 2026-08-28 unless marked TBD.
cd "$(dirname "$0")/../data/raw"
dl () { [ -f "$2" ] && { echo "skip $2"; return; }; echo "GET $2"; curl -sL --retry 3 -o "$2" "$1" && echo "ok $2 ($(du -h "$2" | cut -f1))"; }

# FSim national wildfire risk components, 270m (BP, CFL, FLEP4, FLEP8, RPS) — Short et al.
dl "https://www.fs.usda.gov/rds/archive/products/RDS-2016-0034-3/RDS-2016-0034-3.zip" fsim_national_270m.zip
# TIGER primary+secondary roads, NM (35) + OK (40) + TX (48; for the option-A TX re-extract consistency check)
dl "https://www2.census.gov/geo/tiger/TIGER2023/PRISECROADS/tl_2023_35_prisecroads.zip" roads_nm.zip
dl "https://www2.census.gov/geo/tiger/TIGER2023/PRISECROADS/tl_2023_40_prisecroads.zip" roads_ok.zip
dl "https://www2.census.gov/geo/tiger/TIGER2023/PRISECROADS/tl_2023_48_prisecroads.zip" roads_tx.zip
# County boundaries (FIPS for USDM drought join)
dl "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_20m.zip" counties_us.zip
# State boundaries (population-universe clipping)
dl "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip" states_us.zip
# EPA Level-3 ecoregions (CONUS)
dl "https://dmap-prod-oms-edc.s3.us-east-1.amazonaws.com/ORD/Ecoregions/us/us_eco_l3.zip" ecoregions_l3.zip
# TBD (resolve direct links, see PRD open items): WHP 2023 (RDS-2015-0047 catalog),
# LANDFIRE LF2022 CBD/CBH CONUS, HIFLD transmission lines, 3DEP terrain tiles NM/OK.
echo DONE
