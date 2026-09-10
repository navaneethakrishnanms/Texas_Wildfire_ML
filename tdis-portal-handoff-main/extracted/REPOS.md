# Repositories — Sent Separately from This Manual

TDIS sends **four code repositories** in a **separate delivery** (archive or git). This instruction package does not include source code.

## Repos and what they deploy to

| # | Repository | Branch | Runtime |
|---|------------|--------|---------|
| 1 | **tdis-portal-frontend** | `development` | Azure **Static Web App** (or `npm run dev` locally) |
| 2 | **tdis-portal-api** | `develop` | Azure **App Service** — Linux Docker container (FastAPI) |
| 3 | **tdis-portal-backend** | `development` | Azure **App Service** — Node.js (Contact Us / Subscribe email) |
| 4 | **tdis-portal-db** | `development` | **Not deployed** — PostgreSQL migration scripts only |

## Snapshot commits (handoff date)

See [CODE_SNAPSHOTS.md](CODE_SNAPSHOTS.md).

## External runtime (not a TDIS repo)

| Component | Image / service |
|-----------|-----------------|
| **pg_tileserv** | Docker Hub `pramsey/pg_tileserv` — vector map tiles |
| **PostgreSQL** | Azure Database for PostgreSQL Flexible Server (you create) |
| **Mapbox** | Your own account + token at mapbox.com |

## Not included in code delivery

- `tdis-azure-infra` (TDIS internal — deployment described in manual instead)  
- Production database data  
- TDIS AI, Cube.js, Keycloak  

## Folder layout after you have manual + code

```
workspace/
├── instruction_manual/
├── CODE_SNAPSHOTS.md
├── tdis-portal-api/
├── tdis-portal-frontend/
├── tdis-portal-db/
└── tdis-portal-backend/
```
