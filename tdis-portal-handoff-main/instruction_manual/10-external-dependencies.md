# 10 — External Dependencies

| Dependency | Required? | Notes |
|------------|-----------|-------|
| **PostgreSQL** | Yes | `tdis_portal` / `public` |
| **Mapbox** | Yes | BIT creates account + token — [03-prerequisites.md](03-prerequisites.md) |
| **FastAPI** | Yes | Included in handoff |
| **pg_tileserv** | Recommended | Docker `pramsey/pg_tileserv` — vector tiles |
| **Email backend** | Optional | Contact Us / Subscribe — needs SMTP |
| **Databricks + ADLS** | Yes (BIT builds) | Populates Source B tables — not portal code |
| ~~**Cube.js**~~ | **No** | Replaced by FastAPI — ignore `cubeClient.js` |
| ~~**TDIS AI**~~ | **No** | Excluded |
| ~~**Keycloak**~~ | **No** | Public portal — no login |
| ~~**Helpdesk API**~~ | **No** | App intake only — not in handoff |

## What breaks without each service

| Missing | Symptom |
|---------|---------|
| Mapbox token | Blank map, console errors |
| API down | All data widgets fail |
| PostgreSQL down | API health check fails |
| Source B tables empty | Widgets show no data (expected until sync runs) |
| pg_tileserv down | Vector tile layers missing; other layers may work |
| Email backend down | Contact Us / Subscribe buttons fail only |
| SMTP misconfigured | Email API returns 500 |

## Provider APIs (BIT ingestion — not portal runtime)

These are **inputs to your Databricks pipelines**, not runtime deps for the portal UI:

- Baron Weather  
- NIFC (wildfire)  
- USGS (earthquake)  
- NOAA (hurricane)  
- PowerOutage.us  

See TRD §2.3 for data flows.
