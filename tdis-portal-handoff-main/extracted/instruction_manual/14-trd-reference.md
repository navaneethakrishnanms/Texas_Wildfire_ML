# 14 — TRD Reference Summary

Plain-English summary of [TDIS_Portal_TRD-v3.pdf](reference/TDIS_Portal_TRD-v3.pdf). The PDF is the authoritative contract.

## What the portal does

Public-facing Texas disaster information portal focused on **Active Weather Events** — map-based visualization of hazards with data from multiple external providers.

## Hazard types and filters (TRD §2.2.1)

| Hazard | Provider | Filter (summary) |
|--------|----------|------------------|
| High Wind | Baron | Exclude &lt; 30 |
| Hail | Baron | Exclude &lt; 70% |
| Rainfall | Baron 1hr accum | Exclude &lt; 0.1 inch |
| Wildfire | NIFC | Texas only; exclude stale fires |
| Earthquake | USGS | Texas boundary |
| Hurricane | NOAA | Gulf / Texas relevance |
| Power outage | PowerOutage.us | Texas only |

## What BIT builds — three layers + sync

1. **Bronze** — raw provider payloads → Azure Data Lake (ADLS Gen2) + Unity Catalog  
2. **Silver** — normalize to unified hazard schema, Texas filter, H3 indexing  
3. **Gold** — HSI (Hazard Severity Index) rainfall surfaces  
4. **PostgreSQL sync jobs** — three jobs write serving tables:
   - **Events sync** — 6 hazard layers → PostgreSQL  
   - **HSI Gold sync** — full refresh of precipitation HSI  
   - **Power outage sync** — current outages + aggregations  

Detail for Bronze/Silver/Gold pipelines: **TDDL TRD** (ask TDIS separately if needed).

## TRD vs this code snapshot

| TRD says | This handoff |
|----------|--------------|
| Python **Flask** API | **FastAPI** (`tdis-portal-api`) — same role |
| Cube.js replaced by API | **Done** — ignore any Cube.js frontend code |
| No user authentication (§3.1) | Public dashboards work without login; Keycloak code exists but **skip it** |
| Email via backend API | `tdis-portal-backend` (Node, SMTP) |
| Terragrunt / IaC | **Not included** — you deploy by hand per ch. 08 |
| AI assistant | **Excluded** from this handoff |

## Data flow (one paragraph per provider)

**Baron (wind, hail, rainfall):** Grid cells over Texas → bronze raw → filter thresholds → silver unified events → PostgreSQL sync.

**NIFC (wildfire):** Incident/perimeter API → bronze → Texas + status filter → silver polygons → sync.

**USGS (earthquake):** Recent events feed → bronze → point-in-Texas filter → silver points → sync.

**NOAA (hurricane):** Advisories/tracks → bronze → Gulf/Texas filter → silver tracks/cones → sync.

**Power outage:** GeoJSON feed → bronze → Texas clip → silver events + county/city/provider aggregations → sync.

**HSI (precipitation severity):** Baron + Atlas 14 → H3 L10 indexing → rolling windows → gold HSI → full-refresh PostgreSQL sync.

## Acceptance (TRD §8)

- Automated tests pass in CI before production deploy  
- Performance/load testing for expected public traffic  
- All components meet regulatory/compliance standards per TDIS process  

## Wiki (internal TDIS)

TRD references: `https://wikijs.dev.cloud.tdis.io/en/tdis/portal-application/PortalAppDeploymentArchitecture` — you may not have access; this manual replaces that for BIT.
