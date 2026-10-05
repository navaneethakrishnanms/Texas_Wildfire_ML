# TDIS Portal TRD — Wildfire Ignition & Intensity Forecast (TIF / LTS Fire) Data Flow
*(Copy-paste sections, styled to match the TDIS Portal TRD v5.1 hazard data
flows — e.g. §2.3.9 TIF/LTS Flash Flood and §2.3.10 Wind HSI. Insert as the next
`2.3.x` data flow. This documents the daily wildfire ignition + conditional-
intensity model as a Bronze→Silver→Gold medallion pipeline with PostgreSQL sync
and presentation. Companion data-requirement tables: `PRD_daily_ignition_
section3_datareqs.md`.)*

---

## Provider row (append to §2.2.1 "External Data Providers" table)

| Event | Source | API / Product | Filter Criteria |
|---|---|---|---|
| Wildfire Ignition Likelihood | VIIRS (NASA FIRMS) + FPA-FOD | VIIRS S-NPP/N20/N21 375 m active fire; FPA-FOD incident records (labels). HRRR 12Z forecast (features) | Texas; industrial gas-flare cells excluded (episode-based persistence rule); daily fire cell-days |
| Wildfire Conditional Intensity | VIIRS (NASA FIRMS) | VIIRS peak Fire Radiative Power (FRP, MW) | Texas; flare-excluded; conditional on ignition (fire cell-days only) |

---

## 2.3.11 Wildfire Ignition & Intensity Forecast (TIF / LTS Fire) Data Flow

### 2.3.11.1 Flow Summary
The Wildfire Ignition & Intensity Forecast (surfaced as the **LTS Fire** layer)
supports anticipatory pre-positioning of suppression resources using a 24-hour
forecast outlook. It expresses two independent, machine-learned surfaces on the
H3 hex grid: **ignition likelihood** (the probability that a fire is discovered
in a cell during the forecast day — "will a fire start") and **conditional
intensity** (the expected fire radiative power *given* a fire occurs — "how
intense if it does"). A scheduled Databricks pipeline ingests VIIRS active-fire
and FPA-FOD records (labels) and forecast-realistic HRRR weather plus static
fuel/terrain rasters (features), lands raw payloads in Bronze, builds
flare-filtered labels and the H3-indexed feature table in Silver, scores two
gradient-boosted-tree models in Gold to produce calibrated ignition probability
and a conditional-intensity ranking, and publishes hex tiles, a combined
consequence-weighted priority surface, and per-county rollups to PostgreSQL for
the HMED/TIF tab. LTS Fire is anticipatory decision support and **does not
replace** NWS Red Flag Warnings or Texas A&M Forest Service operational
guidance.

### 2.3.11.2 Requirements and Design Decisions
- **Two orthogonal axes.** Ignition likelihood and conditional intensity are
  modeled separately and are near-independent (Spearman ρ ≈ 0.1 statewide), so
  both are published. Their product with an exposure term forms the combined
  LTS-Fire priority. A single ignition map cannot express consequence; a single
  intensity map cannot express timing/probability.
- **Resolution H3 Level-8 (~860 m).** The VIIRS 375 m active-fire pixel is the
  labeling floor; L8 is the finest resolution the label supports (L10, used by
  flood/wind, is rejected for fire). Parent hexes L6–L7 are computed for
  multi-scale visualization and county rollups.
- **Forecast-realistic weather (deployable, not hindcast).** All weather
  features come from the **previous-afternoon 12Z HRRR run** (leads F27/F30/F33/
  F36 → target-day 9 am–6 pm CST peaks), so the daily surface is issuable ~2 pm
  the day before.
- **Industrial-flare exclusion is mandatory.** ~28% of raw TX VIIRS detections
  are persistent gas-flare hotspots; they are excluded via an episode-based
  persistence rule before label fusion (else the models learn always-on
  industrial cells, not wildfire).
- **Calibrated probability.** Ignition output is isotonically calibrated so a
  "0.3" means ~30% observed frequency (interpretable for tiering/alerting).
- **Cause scope.** Human- and equipment-caused fires (the large majority of TX
  ignitions) are the predictable target; lightning ignition is near-chance to
  localize at hex scale and is documented as an accepted limitation.
- **Intensity is a ranker, not an absolute-MW forecast.** The intensity surface
  ranks cells by relative fire-radiative-power potential (validated), but
  under-predicts absolute megawatts of extreme fires; the UI presents it as a
  relative severity indicator, not a MW value.
- **Temporal packaging.** v1 is a daily (0–24 h) outlook. A sub-daily (four
  6-hour window) extension is a research capability; if enabled, per-hex
  next-window and peak-window risk are reported (one layer, not four).

### 2.3.11.3 Fetch Model Inputs (Bronze)
A scheduled Databricks job fetches labels and features and lands them in Unity
Catalog Bronze, preserving provider fidelity for replay.
- **VIIRS active fire** — NASA FIRMS Area API (`firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{VIIRS_*}/…`); SP archive + NRT; keep `frp`, `confidence`, `acq_date/time`, lat/lon.
- **FPA-FOD incidents** — USFS RDS-2013-0009.6 + Zenodo augmented CSVs; cause, size, discovery date/time (Texas 2014–2020; labels only).
- **HRRR 12Z forecast** — AWS S3 `noaa-hrrr-bdp-pds` via Herbie, product `sfc`, leads F27/30/33/36; GRIB subset `:(TMP|DPT):2 m above ground|:WIND:10 m|:GUST:surface|:MSTAV:`.
- **Static rasters (one-time / annual)** — FSim (RDS-2016-0034-3: `avg_burn_prob`, `cfl`, `flep4`), WHP 2023, LANDFIRE LF2022 canopy (`cbd`, `cbh`), 3DEP terrain (`elevation_m`, `slope_deg`, `aspect_deg`), EPA Level-3 ecoregion, TIGER 2023 roads (`road_dist_km`), HIFLD transmission (`powerline_dist_km`).
- **gridMET fm100** — climatologylab OPeNDAP/HTTPS (`fm100`, ~2-day lag, lag-safe).
- **USDM drought** — `usdmdataservices.unl.edu` weekly by county FIPS.
- **WorldPop 2020** — 1 km population density (`pop_density`).

Bronze tables (append-only): `tdis_data_catalog.hazard_forecast_bronze.wildfire_viirs_raw`, `…wildfire_fpa_raw`, `…wildfire_hrrr12z_raw`, `…wildfire_static_raw`. Each fetch is logged to `wildfire_api_run_logs` with `record_count` and downstream-processed flags for run-log gating (per §2.3.8.7 pattern).

### 2.3.11.4 Spatial Indexing (H3 L8)
- Assign an H3 Level-8 index to every VIIRS detection, FPA incident, HRRR grid
  point (nearest-cell via cKDTree — never float-equality), and static-raster
  cell.
- Persist L8 index plus parent L6/L7 IDs and boundary geometry per row so all
  downstream joins use H3 rather than geometry operations.
- Reuse a static Texas H3 L8 cell master (`static_master`) with a pre-computed
  nearest-HRRR-grid-point mapping per hex.

### 2.3.11.5 Label Construction & Flare Filtering (Silver)
- **Flare exclusion:** flag industrial-hotspot cells (episode-based: ≥6 distinct
  fire episodes in any rolling 365 d, gap ≤ 2 d) per state; TX v2 = 538 cells,
  NM = 1,956, OK = 260. Exclude before fusion.
- **Fused labels:** for each (cell, day), set `label = 1` if a flare-filtered
  VIIRS detection or FPA incident occurred; record `time_source` (VIIRS-observed
  vs imputed-FPA) and `cause`.
- **Negatives:** day-matched non-fire cells (10:1) sampled from the same
  operational day for the classification training table.
- Unity Catalog: `tdis_data_catalog.hazard_forecast_silver.wildfire_labels_daily`.

### 2.3.11.6 Feature Assembly (Silver)
- Join per (cell, day): 12 static fuel/terrain features, 12Z-HRRR daily peaks
  (`gust_max`, `vpd_max`, `tmp_max`, `mstav_min`), `drought_score`,
  `fm100_5d_min` (lag-safe: min over D-6…D-2), `pop_density`, calendar encodings
  (`sin/cos_month`, `sin/cos_dow`, `is_weekend`, `is_holiday`), and physics
  interactions (`bp_x_vpd`, `whp_x_vpd`, `bp_x_drought`).
- Per-state unit-range assertions (e.g. `avg_burn_prob ∈ [0,1]`) run as a data
  gate; out-of-range values block the run.
- Unity Catalog: `tdis_data_catalog.hazard_forecast_silver.wildfire_features_daily`.

### 2.3.11.7 Model Scoring — Ignition Probability + Conditional Intensity (Gold)
Two independent gradient-boosted-tree (XGBoost) models score every Texas L8 hex
for the forecast day.
- **Ignition likelihood (classifier).** Inputs = the full feature table;
  monotone constraints on physically-directional features (burn prob, WHP, VPD,
  wind, drought ↑; fm100 ↓). Output `ignition_prob` ∈ [0,1], **isotonically
  calibrated**. Classify into tiers: Low (<0.1), Elevated (0.1–0.3), High
  (0.3–0.6), Extreme (≥0.6) — tier cutpoints set on validation.
- **Conditional intensity (regressor).** Inputs = static fuel/terrain features
  only (weather adds no skill for peak intensity). Target = log(1+peak VIIRS
  FRP). Output `intensity_rank` = statewide percentile (a relative severity
  indicator, not absolute MW).
- **Combined LTS-Fire priority.** `lts_fire = f(ignition_prob_pct ×
  intensity_rank × exposure)` (geometric), plus the two axes retained
  separately so operators can see high-consequence "sleeper" cells (low
  ignition, high intensity) the product can dilute.
- Aggregate to parent hexes L6–L7 (means) for multi-scale visualization.
- Unity Catalog: `tdis_data_catalog.hazard_forecast_gold.wildfire_ignition` (per-hex `ignition_prob`, `ignition_tier`, `intensity_rank`, `lts_fire`, `lts_fire_label`, parent hex IDs, run timestamp).
- Notebooks/Jobs: `run_wildfire_ignition_gold.py` (scoring); models `ignition_xgb_calibrated`, `intensity_xgb`.

**LTS-Fire severity interpretation**

| Numeric | Name | Actionable description |
|---|---|---|
| 0 | Negligible | Ignition unlikely and/or low intensity potential. |
| 1 | Minor | Elevated ignition likelihood; limited intensity potential. |
| 2 | Moderate | Meaningful ignition likelihood in fire-capable terrain; monitor. |
| 3 | High | High ignition likelihood in high-intensity fuel/terrain; pre-position. |
| 4 | Critical | High ignition likelihood + top-tier intensity potential; stage suppression, prioritize. |

### 2.3.11.8 PostgreSQL Sync (Serving Layer)
- Follow the HSI Gold sync template (§2.2.6–2.2.7): **full refresh (truncate →
  bulk insert)**, since each daily run is a fresh recomputed surface (single
  snapshot in time), with run-log gating (§2.3.8.7) so an empty/failed run
  leaves tables empty rather than stale.
- JDBC write (partitioned); cast hex boundary WKT to PostGIS `geometry(Geometry,
  4326)` with GIST indexes; auto-add new columns from the Gold schema.
- Serving tables (public schema): `public.wildfire_ignition` (per-hex L8 with
  `ignition_prob`, `ignition_tier`, `intensity_rank`, `lts_fire`), MVT views
  `public.mv_wildfire_ignition_*` per resolution band (L6–L8), and
  `public.wildfire_ignition_counties` (per-county rollup: max tier, mean
  ignition prob, exposure counts).
- Exposure join (optional, mirrors LTS Flood): population / structures /
  critical facilities within High and Critical LTS-Fire cells, from TDDL Silver.

### 2.3.11.9 Presentation and Interpretation (React / Mapbox)
- Rendered in the HMED/TIF tab as layer `lts_fire`. Default toggle shows the
  combined LTS-Fire priority (L8, aggregating to L6–L7 on zoom-out); optional
  overlays expose the **ignition-likelihood** and **conditional-intensity** axes
  separately (so high-consequence "sleeper" cells are visible).
- Severity ramp: Negligible `#22c55e`, Minor `#eab308`, Moderate `#f97316`,
  High `#ea580c`, Critical `#dc2626` (colorblind-checked; matches the HSI ramp
  family).
- Hex click → detail panel: ignition probability + tier, intensity percentile,
  combined LTS-Fire level, dominant drivers, and exposure counts.
- Left panel / mobile "Counties" tab: per-county LTS-Fire level with search and
  refresh; refresh timestamp shown (last 12Z run).
- **Disclaimer / attribution:** anticipatory decision support issued from the
  previous-afternoon 12Z HRRR forecast; **not** a replacement for NWS Red Flag
  Warnings or Texas A&M Forest Service directives. Intensity is a relative
  severity indicator (fire radiative power), not observed damage, final fire
  size, or ecological burn severity. Lightning-caused ignitions are not reliably
  localized at hex scale.

---

## Output Data Products (append to §4.3 style table)

| Output | Field name | Type | Update frequency |
|---|---|---|---|
| Ignition probability | `wildfire_ignition_prob_l8` | Float [0,1] (calibrated) | Daily (per 12Z run) |
| Ignition tier | `wildfire_ignition_tier_l8` | Integer (0–4) | Daily |
| Conditional intensity rank | `wildfire_intensity_rank_l8` | Float [0,1] percentile | Static-driven; refreshed w/ run |
| LTS-Fire priority | `lts_fire_l8` | Integer (0–4) | Daily |
| Per-county rollup | `wildfire_ignition_counties` | table | Daily |

*Coordinate system EPSG:4326; spatial index H3 Level-8 (parents L6–L7);
output TDDL Gold (Parquet/Delta), keyed `h3_l8`.*
