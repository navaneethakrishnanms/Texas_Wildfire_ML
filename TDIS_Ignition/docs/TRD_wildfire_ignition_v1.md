# Technical Requirements Document — TDIS Wildfire Ignition Forecast (TIF / LTS Fire)

| | |
|---|---|
| Document Type | Technical Requirements Document |
| Product | LTS Fire — Wildfire Ignition Forecast Data Layer |
| Application Family | TIF – Threat & Impact Forecasting |
| Platform | TDIS |
| Version | 1.0 (draft) |
| Date | 21 September 2026 |
| Status | Draft |
| Author | Urban Resilience.AI Lab, Texas A&M University |
| Affiliations | IDRT = Institute for a Disaster-Resilient Texas; TDIS = Texas Disaster Information System |

*Styled to match the TDIS Portal TRD v5.1. Scope is the wildfire **ignition
likelihood** forecast at two horizons (0–24h, 24–48h). Conditional fire
intensity is a separate TIF product (see §1.3.1). Companion data-requirement
tables: `PRD_daily_ignition_section3_datareqs.md`.*

---

## Table of Contents
1. Introduction — 1.1 Purpose · 1.2 Intended Audience · 1.3 Definitions/Acronyms · 1.3.1 Scope
2. High-Level System Overview — 2.1 System Context · 2.2 Technical Architecture (2.2.1 Data Sources · 2.2.2 Bronze Ingestion · 2.2.3 Silver + PostgreSQL Sync · 2.2.4 Ignition Forecast Sync Job · 2.2.5 Shared Sync Behavior · 2.2.6 Serving Layer · 2.2.7 Presentation Layer · 2.2.8 CI/CD) · 2.3 Ignition Data Flow (0–24h / 24–48h) · 2.4 Database · 2.5 Notification Backend
3. Security
4. Scalability
5. Monitoring and Logging
6. Deployment
7. Backup and Disaster Recovery
8. Testing — 8.1 Testing Strategy · 8.2 Acceptance Criteria

---

# 1 Introduction

## 1.1 Purpose
The Wildfire Ignition Forecast (surfaced as the **LTS Fire** layer within the
TDIS Threat & Impact Forecasting family) is an anticipatory decision-support
data layer that estimates, on the H3 hex grid, the probability that a wildfire
will be discovered in each cell over the next 24 and 24–48 hours. It translates
static landscape susceptibility and forecast fire-weather into a calibrated,
map-based ignition-likelihood surface, issued the afternoon before, so
emergency managers and Texas A&M Forest Service coordinators can pre-position
suppression resources ahead of dangerous days.

This TRD defines the reference architecture, integration patterns, data flows,
and non-functional requirements for the ignition forecast implementation within
the TDIS/TDDL medallion stack. It is the contract between product, data
engineering, and operations for what is built, how it behaves, and how it is
operated in production.

## 1.2 Intended Audience
- **Data Engineering** — own and build the Databricks ingestion + scoring
  pipelines; schedule/orchestrate/monitor; normalize feeds and model outputs
  into curated PostgreSQL tables.
- **Backend Engineers** — expose the Flask REST + tile endpoints for the
  ignition layer; enforce contracts, pagination, geospatial queries.
- **Frontend Engineers** — render the H3 hex layer and detail panels in
  React/Mapbox; time-horizon toggle.
- **DevOps / SRE** — operate Azure App Service, Databricks jobs, PostgreSQL;
  observability and CI/CD.
- **Product & Technical Leads** — validate PRD alignment; approve scope.
- **Security & Compliance Reviewers** — review data flows, API surface,
  network boundaries.
- **Model owner (Urban Resilience.AI Lab)** — maintain the trained ignition
  model, calibrator, and flare-exclusion lists; approve retrains.

## 1.3 Definitions, Acronyms, and Abbreviations
- **TDIS / TDDL** — Texas Disaster Information System; TDIS Data & Delivery
  Layer (medallion Bronze/Silver/Gold on Databricks + ADLS Gen2 + Unity
  Catalog).
- **TIF** — Threat & Impact Forecasting application family.
- **LTS Fire** — Localized Threat Severity for wildfire; the presentation name
  of this ignition layer.
- **Ignition likelihood** — calibrated probability a fire is *discovered* in a
  cell in the forecast window (occurrence risk, not a warning).
- **H3 Level-8** — Uber H3 hexagonal grid at resolution 8 (~0.74 km², ~860 m
  edge); the model's native resolution (parents L6–L7 for aggregation).
- **HRRR** — NOAA High-Resolution Rapid Refresh NWP (3 km, hourly; 12Z run to
  F48). **GFS** — Global Forecast System (0.25°, to F384) used for the 24–48h
  horizon beyond HRRR's reach.
- **FPA-FOD** — Fire Program Analysis Fire-Occurrence Database (agency incident
  labels, incl. cause).
- **VIIRS FRP** — Visible Infrared Imaging Radiometer Suite Fire Radiative
  Power (375 m active-fire detections; labels).
- **Flare cell** — a persistent industrial hotspot (gas flare) that VIIRS reads
  as "fire"; excluded from labels.
- **Provider Feed** — external authoritative source (NASA FIRMS, USFS, NOAA
  HRRR/GFS, gridMET, USDM, WorldPop).

### 1.3.1 Scope
**In Scope**
- Daily ignition-likelihood computation for Texas at H3 Level-8, two horizons:
  0–24h and 24–48h.
- Consumption of VIIRS + FPA-FOD (labels) and forecast-realistic HRRR 12Z (0–24h)
  and GFS 12Z (24–48h) weather + static fuel/terrain rasters (features).
- Industrial-flare exclusion prior to label fusion.
- Calibrated probability, tiering (0–4), per-county rollups.
- PostgreSQL serving tables + MVT tiles; interactive hex map with time-horizon
  toggle and detail panels.

**Out of Scope**
- **Conditional fire-intensity model** (VIIRS-FRP "how bad if it starts") — a
  separate TIF Fire product; this TRD is ignition likelihood only.
- Sub-daily (6-hour window) ignition — a research extension, not v1.
- Fire spread / perimeter simulation, containment modeling.
- Official Red Flag Warning or alert issuance (NWS/TFS authority).
- Model *training* infrastructure (offline; the trained model + calibrator +
  flare lists are delivered as versioned artifacts and consumed here).

---

# 2 High-Level System Overview
A modular, cloud-native layer on the existing TDIS/Azure stack: Databricks
ingestion + scoring pipelines write curated ignition surfaces into PostgreSQL;
a Flask API + tile server serve them; a React/Mapbox frontend renders them.
Designed for deterministic daily runs, low-latency retrieval, and
fault-tolerant operation.

## 2.1 System Context
External Providers → Databricks Pipelines (Bronze→Silver→Gold + scoring) →
PostgreSQL → Flask API → React/Mapbox UI.
- Providers supply labels (VIIRS, FPA-FOD) and forecast features (HRRR, GFS,
  gridMET, USDM, WorldPop, static rasters).
- Databricks jobs extract, H3-index, flare-filter, assemble features, score the
  ignition model for both horizons, and load curated surfaces to PostgreSQL.
- Flask reads PostgreSQL and exposes REST + tile endpoints.
- React/Mapbox renders the hex layer with a 0–24h / 24–48h toggle.

## 2.2 Technical Architecture
Deployed via Terraform/Terragrunt into the TDIS Azure environment; reuses the
TDDL medallion (ADLS Gen2 + Unity Catalog), Azure Database for PostgreSQL,
Azure App Service (Flask), and Azure Static Web App (React).

### 2.2.1 Data Sources and Ingestion Entry Point (TDDL / Databricks)
| Input | Source / API | Role | Filter |
|---|---|---|---|
| VIIRS active fire | NASA FIRMS Area API (S-NPP/N20/N21 375 m) | Label | Texas; flare cells excluded |
| FPA-FOD incidents | USFS RDS-2013-0009.6 + Zenodo 8381129 | Label (cause) | Texas 2014–2020 |
| HRRR 12Z forecast | AWS S3 `noaa-hrrr-bdp-pds` via Herbie (sfc, F24–F48) | Feature (0–24h) | TX bbox |
| GFS 12Z forecast | NOMADS / AWS `noaa-gfs-bdp-pds` (0.25°, F24–F48) | Feature (24–48h) | TX bbox |
| Static fuel/terrain | FSim RDS-2016-0034-3, WHP 2023, LANDFIRE LF2022, 3DEP, EPA L3 ecoregion, TIGER 2023, HIFLD | Feature | CONUS → TX clip |
| gridMET fm100 | climatologylab OPeNDAP/HTTPS | Feature (lag-safe) | TX |
| USDM drought | `usdmdataservices.unl.edu` (weekly) | Feature | TX counties |
| WorldPop 2020 | WorldPop 1 km | Feature | TX |

All providers expose HTTP/HTTPS APIs or file endpoints, pulled never pushed.
Full pull specifications (endpoints, auth, cadence, GRIB subsets) are in
`PRD_daily_ignition_section3_datareqs.md §3.4`.

### 2.2.2 Bronze Ingestion Component
Per the TDDL Bronze strategy: each provider job (a) calls the API on schedule,
(b) writes raw payloads to ADLS Gen2 `/bronze/wildfire/{provider}/ingestion_date=YYYY-MM-DD/…`,
(c) registers/refreshes Unity Catalog bronze Delta tables, (d) logs to
`wildfire_api_run_logs` with `record_count` + processed flags for run-log
gating. Bronze is append-only and preserves full provider fidelity for replay.

### 2.2.3 Transformation Layer (Silver) and PostgreSQL Sync
Silver Databricks jobs consume bronze and apply H3-8 indexing, flare exclusion,
label fusion, feature assembly, schema normalization, and per-state
unit-range validation, writing Silver Delta tables partitioned by
`horizon`/`run_date`. A PostgreSQL sync job then projects the Gold scored
surface into relational serving schemas via idempotent, transactionally-safe
loads. ADLS + Unity Catalog remain system-of-record; PostgreSQL is the serving
copy.

### 2.2.4 Ignition Forecast Sync Job
**Purpose:** publish the Gold ignition surface (per-hex probability, tier, and
combined LTS-Fire level) for both horizons.
**Coverage:** `lts_fire_0_24h_l8`, `lts_fire_24_48h_l8`, and county rollups.
**Load pattern:** **full refresh (truncate → bulk insert)** — each daily run is
a fresh recomputed surface (single snapshot in time), not incremental —
gated by run-logs so an empty/failed run yields empty tables, never stale data.

### 2.2.5 Shared Sync Behavior
Jobs run as Databricks notebook tasks / Python modules on ephemeral job
clusters; credentials from Key Vault secret scopes; JDBC load with tuned batch
size and partitioning; WKT hex boundary cast to PostGIS `geometry(Geometry,4326)`
with GIST indexes; schema-evolution auto-adds new columns (never drops); each
job logs `row_count`, `run_status`, timestamps, and errors.

### 2.2.6 Serving Layer: Flask API and Tile Server
Flask on Azure App Service (stateless, containerized):
- **JSON endpoints** — `GET /fire/ignition?bbox=…&horizon=0_24|24_48`,
  `GET /fire/ignition/{h3}`, `GET /fire/ignition/counties?horizon=…`,
  `GET /fire/ignition/metadata` (run timestamp, model version).
- **Tile server** — MVT vector tiles (XYZ / Web Mercator) for the dense L8 hex
  layer, backed by PostGIS or pre-rendered tiles. Stateless, horizontally
  scalable behind the App Service load balancer / Application Gateway.

### 2.2.7 Presentation Layer: React + Mapbox
Single-page React app (Azure Static Web App). Mapbox GL renders the L8 hex
polygon layer colored by LTS-Fire tier, aggregating to L6–L7 on zoom-out; a
prominent **0–24h / 24–48h toggle**; legend; hover tooltip (tier); click detail
panel (ignition probability, tier, dominant drivers, county, refresh time). No
heavy client transformation — payloads/tiles are presentation-ready.

### 2.2.8 CI/CD, Container Registry, Operational Flow
All code (Databricks jobs, Flask, React, IaC) in GitHub. Azure DevOps pipelines:
React → Static Web App; Flask → container → Azure Container Registry → App
Service; Databricks jobs deployed as code via REST/CLI; Terraform/Terragrunt for
infra. Runtime: Users → App Gateway → Frontend → Flask → PostgreSQL; Databricks
(ingest + score + sync) → ADLS Gen2 + Unity Catalog + PostgreSQL.

## 2.3 Ignition Data Flow (0–24h / 24–48h)

### 2.3.1 Flow Summary
A scheduled Databricks pipeline fetches labels and forecast features, lands them
in Bronze, builds flare-filtered (cell, day) labels and the H3-8 feature table
in Silver, scores the trained ignition classifier **twice** — once with 0–24h
HRRR-derived weather and once with 24–48h GFS-derived weather (static/calendar
features identical; only the weather horizon differs) — calibrates the output,
assigns tiers, and publishes two hex surfaces + county rollups to PostgreSQL.
Issued from the previous-afternoon 12Z model runs.

### 2.3.2 Requirements and Design Decisions
- **Two horizons, one model.** The same trained, calibrated classifier is scored
  with horizon-specific weather features: 0–24h from HRRR 12Z (F24–F48), 24–48h
  from GFS 12Z (F24–F48 of the following day, since HRRR ends at F48). Static +
  calendar features are shared. This keeps one risk logic and avoids
  divergent models.
- **Forecast-realistic.** Weather is the previous-afternoon 12Z *forecast*, so
  both surfaces are deployable the day before — not a hindcast.
- **Resolution H3 L8 (~860 m).** VIIRS 375 m pixel is the labeling floor; L8 is
  the finest supportable. Parents L6–L7 for aggregation/rollup.
- **Flare exclusion mandatory** before label fusion (~28% of TX raw detections).
- **Calibrated probability** (isotonic) so tier cutpoints and the "0.3 ≈ 30%"
  reading are valid.
- **Cause scope:** human/equipment fires (majority of TX ignitions) are the
  predictable target; lightning is near-chance at hex scale (documented).
- **Higher uncertainty at 24–48h:** GFS is coarser (0.25°) and longer-lead; the
  UI states the 24–48h surface is less certain and should be re-checked as the
  day approaches.

### 2.3.3 Fetch Model Inputs (Bronze)
Fetch VIIRS + FPA (labels); HRRR 12Z (F24–F48) and GFS 12Z (F24–F48) weather;
static rasters; gridMET fm100; USDM drought; WorldPop. Land raw payloads in
ADLS Gen2 bronze; register Unity Catalog bronze Delta tables; log runs.
(Endpoints/auth/cadence: companion §3.4.)

### 2.3.4 Spatial Indexing (H3 L8)
Assign H3 L8 to VIIRS, FPA, HRRR grid points, GFS grid points, and static
cells (nearest-cell via cKDTree; never float-equality). Persist L8 + parent
L6/L7 IDs and boundary geometry. Reuse the static Texas L8 cell master with a
pre-computed nearest-HRRR and nearest-GFS grid-point mapping per hex.

### 2.3.5 Label Construction & Flare Filtering (Silver)
Flag industrial flare cells (episode-based: ≥6 distinct fire episodes / rolling
365 d, gap ≤ 2 d; TX 538 / NM 1,956 / OK 260) and exclude. Fuse VIIRS + FPA
into `(cell, day, label)` with `time_source` and `cause`; day-matched 10:1
negatives for training. Unity Catalog:
`tdis_data_catalog.hazard_forecast_silver.wildfire_labels_daily`.

### 2.3.6 Feature Assembly (Silver)
Per (cell, day, horizon): static fuel/terrain (burn prob, WHP, FLEP4, canopy,
elevation, slope, aspect, roads, powerlines, ecoregion); horizon weather peaks
(gust/vpd/tmp max, mstav min — HRRR for 0–24h, GFS for 24–48h); drought;
lag-safe fm100; pop density; calendar encodings; physics interactions. Per-state
range assertions gate the run. Unity Catalog:
`…hazard_forecast_silver.wildfire_features_{0_24,24_48}`.

### 2.3.7 Scoring & Calibration (Gold)
Score the trained XGBoost ignition classifier (monotone constraints on
physically-directional features) on each horizon's feature table → raw
probability → **isotonic calibrator** → `ignition_prob` ∈ [0,1]. Tier: Low
(<0.1), Elevated (0.1–0.3), High (0.3–0.6), Extreme (≥0.6). Aggregate to L6–L7.
Unity Catalog: `…hazard_forecast_gold.wildfire_ignition_{0_24,24_48}` (fields
`ignition_prob`, `ignition_tier`, parent IDs, model_version, run timestamp).
Notebook/Job: `run_wildfire_ignition_gold.py`; model artifact
`ignition_xgb_calibrated` + `ignition_isotonic.joblib` (versioned).

**LTS-Fire ignition interpretation**

| Tier | Name | Actionable description |
|---|---|---|
| 0 | Negligible | Ignition very unlikely in the window. |
| 1 | Minor | Slightly elevated; routine awareness. |
| 2 | Moderate | Meaningful likelihood; monitor fire-capable areas. |
| 3 | High | High likelihood; pre-position initial-attack resources. |
| 4 | Extreme | Very high likelihood; stage suppression, prioritize. |

### 2.3.8 PostgreSQL Sync (Serving Layer)
Full-refresh (truncate → bulk insert) per horizon, run-log gated. Serving tables
(public): `wildfire_ignition_0_24h_l8`, `wildfire_ignition_24_48h_l8`, MVT views
`mv_wildfire_ignition_*` (L6–L8), `wildfire_ignition_counties`. Idempotent,
atomic per batch.

### 2.3.9 Presentation and Interpretation (React/Mapbox)
Rendered in the HMED/TIF tab as `lts_fire`, with a **0–24h / 24–48h toggle**
that swaps the map + widgets within 2 s. Severity ramp: Negligible `#22c55e`,
Minor `#eab308`, Moderate `#f97316`, High `#ea580c`, Extreme `#dc2626`
(colorblind-checked). Hex click → detail panel (ignition probability + tier,
county, dominant drivers, run time). County panel/tab lists per-county tier with
search. **Disclaimer:** anticipatory decision support issued from the
previous-afternoon 12Z runs; the 24–48h surface carries higher uncertainty (GFS,
longer lead); **not** a replacement for NWS Red Flag Warnings or Texas A&M
Forest Service directives; lightning ignitions are not reliably localized at hex
scale.

## 2.4 Database
PostgreSQL (Azure Database for PostgreSQL) holds the serving schema: per-hex
ignition tables (two horizons), county rollups, and a `model_runs` metadata
table (run timestamp, model_version, horizon, row_count, status). GIST spatial
indexes on hex geometry; B-tree on `h3_l8`, `ignition_tier`, `run_date`. DB
migration scripts in GitHub.

## 2.5 Notification Backend
No public authentication (public layer). Optional integration with the common
TDIS notification service to email a daily "High/Extreme tier present"
summary to subscribed IDRT/TFS operational inboxes (reuses the portal's
notification backend; no new email stack).

---

# 3 Security
## 3.1 Authentication and Authorization
The layer is publicly accessible; no end-user authentication. Databricks→
PostgreSQL and API→PostgreSQL credentials are stored only in Azure Key Vault
secret scopes, never in code.
## 3.2 Data Security
No confidential/PII data (public hazard forecast); data at rest is not
encrypted beyond Azure storage defaults. Network security via Azure network
controls (App Service access restrictions, private endpoints where configured).
Model artifacts and flare lists are versioned, read-only in production.

# 4 Scalability
- **Horizontal scaling:** Flask App Service and the tile server scale
  horizontally; PostgreSQL read replicas for tile/query load.
- **Auto-scaling:** App Service auto-scales on request load; Databricks scoring
  runs on ephemeral job clusters sized to the ~0.8 M-cell × 2-horizon daily
  scoring (minutes).
- **Data volume:** two L8 surfaces (~0.8 M hexes each) are bounded and refreshed
  daily; MVT tiling keeps map payloads small.

# 5 Monitoring and Logging
## 5.1 Monitoring
- **Azure Monitor / Application Insights** — Flask API latency, errors,
  dependency health; SWA.
- **Databricks job telemetry** — run status, duration, row counts for
  ingestion, scoring, and sync jobs; alert on failure or zero-row output.
- **Data freshness** — a staleness alert fires if the latest run timestamp
  exceeds the daily SLA (e.g. no successful 12Z-cycle scoring by target time);
  UI shows last-run timestamp.
## 5.2 Logging
Azure Log Analytics centralizes ingestion, scoring, sync, and API logs. Each
Databricks job logs `row_count`, `run_status`, timestamps, model_version, and
errors; run-log tables track processed/synced state per horizon.

# 6 Deployment
## 6.1 CI/CD Pipeline
Azure DevOps: React → Static Web App; Flask → container → ACR → App Service;
Databricks jobs deployed as code (REST/CLI); Terraform/Terragrunt (or ARM) for
infrastructure. Model artifacts (classifier, isotonic calibrator, flare lists)
are versioned and promoted through a controlled release with a model-version
tag recorded in `model_runs`.
## 6.2 Deployment Strategy
Blue-green deployment for the API/frontend (zero downtime); Databricks jobs
deployed to a staging workspace and validated on a shadow run before production
promotion. Model promotion requires the §8 acceptance gates to pass.

# 7 Backup and Disaster Recovery
## 7.1 Backup Strategy
Automated PostgreSQL backups with retention per organizational policy; ADLS +
Unity Catalog (Bronze/Silver/Gold Delta) are the system-of-record and are the
authoritative source for rebuilding PostgreSQL. Model artifacts and flare lists
are version-controlled and reproducible.
## 7.2 Disaster Recovery Plan
Geo-redundant ADLS and PostgreSQL for regional-outage resilience; automated
failover. Because PostgreSQL is a full-refresh serving copy, it can be
rebuilt from the latest Gold Delta at any time; a missed daily run degrades
gracefully (UI shows staleness, prior surface retained only if within SLA else
tables emptied by run-log gating).

# 8 Testing
## 8.1 Testing Strategy
- **Unit tests** — H3 indexing, flare-episode rule, label fusion, feature-range
  assertions, tier cutpoints, calibrator application.
- **Model-behavior tests** (offline, pre-promotion) — monotonicity of
  constrained features, permutation-importance sanity, isotonic calibration
  (ECE), cross-seed prediction stability, label-shuffle control (skill → ~0).
- **Integration tests** — end-to-end Bronze→Silver→Gold→PostgreSQL for both
  horizons; run-log gating (empty API → empty tables, not stale).
- **Performance/load tests** — tile + JSON endpoints under expected map load;
  daily scoring within SLA.
- **UI tests** — horizon toggle, detail panel population, WCAG 2.1 AA,
  colorblind-safe ramp.

## 8.2 Acceptance Criteria
- All P0 functional requirements implemented and passing automated suites.
- **Model quality gates (held-out forward test years):** forward-year AUROC ≥
  0.80; real-population lift ≥ 4× over base rate; isotonic ECE ≤ 0.01;
  3-seed prediction correlation ≥ 0.98; label-shuffle AUC-PR ≈ base rate.
- **Flare integrity:** zero known industrial-flare cells present in published
  labels/positives; NM Permian subset reviewed.
- **Data gates:** per-state unit-range assertions pass; no null/unkeyed hexes;
  both horizons generated and stored.
- **Freshness:** end-to-end latency from 12Z availability to published surface
  ≤ target SLA across 5 consecutive runs; staleness alerting verified.
- **UI:** horizon toggle updates map + widgets ≤ 2 s; detail panel ≤ 200 ms;
  WCAG 2.1 AA; disclaimer present.
- **Docs:** technical doc + user guide published; model card (version, features,
  training window, known limitations — lightning, 24–48h uncertainty, intensity
  out of scope) published.

---
## Open Questions
| # | Question | Owner |
|---|---|---|
| OQ-1 | 24–48h weather: GFS only, or blend HRRR-F48 boundary + GFS? | Data Science / Data Eng |
| OQ-2 | Tier cutpoints — statewide fixed, or seasonally/regionally calibrated? | TIF Product / Data Science |
| OQ-3 | Extend beyond Texas (NM/OK) in v1 serving, or TX-only first? | TIF Product |
| OQ-4 | Surface the separate conditional-intensity product alongside, or ship ignition first? | TIF Product |
| OQ-5 | Daily run trigger — automatic on 12Z availability, or fixed schedule? | TDIS Platform |
