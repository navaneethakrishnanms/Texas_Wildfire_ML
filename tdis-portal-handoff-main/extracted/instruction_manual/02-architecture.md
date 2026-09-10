# 02 — Deployment Architecture

This chapter shows **how TDIS deploys the portal today** (reference) and **how you deploy the same logical stack on your Azure subscription**.

Code repos are **sent separately** — see [REPOS.md](../REPOS.md).

---

## Repos you will receive (code — separate delivery)

| Repository | Branch | Deployed as |
|------------|--------|-------------|
| **tdis-portal-frontend** | `development` | Static Web App (React build) |
| **tdis-portal-api** | `develop` | App Service — Linux container (FastAPI) |
| **tdis-portal-backend** | `development` | App Service — Node (Contact Us email) |
| **tdis-portal-db** | `development` | Not a runtime — SQL migrations / schema reference |

Pinned commits: [CODE_SNAPSHOTS.md](../CODE_SNAPSHOTS.md)

**Not sent:** TDIS Azure infra repos, Keycloak, TDIS AI, production data.

---

## Data flow (both environments)

```mermaid
flowchart LR
  subgraph ingest [BIT Builds — Databricks]
    Providers[External providers] --> Bronze --> Silver --> Gold --> Sync[PG sync]
  end
  Sync --> PG[(PostgreSQL tdis_portal)]
  PG --> API[FastAPI]
  PG --> Tiles[pg_tileserv Docker]
  API --> FE[React + Mapbox]
  Tiles --> FE
  Email[Email API] --> FE
```

Your Databricks jobs populate PostgreSQL. Portal code only **reads** the database and serves the UI.

---

## How TDIS deploys it today (reference)

TDIS runs this in its own Azure subscription. You do **not** get access to this environment — use this as a **blueprint**.

```mermaid
flowchart TB
  Users[Users browser]

  subgraph edge [Edge]
    AGW[Application Gateway]
  end

  subgraph apps [App Services — Linux]
    SWA[Static Web App\nportal.dev.cloud.tdis.io]
    APINew[App Service container\napi-new-portal FastAPI]
    APIEmail[App Service container\nportal-api email legacy]
    PGTILE[App Service container\npramsey/pg_tileserv]
  end

  subgraph data [Data]
    PG[(Azure PostgreSQL Flexible Server\ndatabase tdis_portal)]
    KV[Key Vault\nDB creds secrets]
  end

  subgraph ingest [TDIS Databricks — separate]
    DBX[Databricks sync jobs]
  end

  Users --> AGW
  AGW --> SWA
  AGW --> APINew
  AGW --> PGTILE
  SWA -->|VITE_API_URL| APINew
  SWA -->|VITE_EMAIL_API_URL| APIEmail
  SWA -->|VITE_TILE_SERVER| PGTILE
  APINew --> PG
  PGTILE --> PG
  KV -.-> APINew
  KV -.-> PGTILE
  DBX --> PG
```

| TDIS component | Azure resource | Hostname (dev example) |
|----------------|----------------|------------------------|
| Frontend | Static Web App | `portal.dev.cloud.tdis.io` |
| Portal API (FastAPI) | App Service + ACR | `api-new-portal.dev.cloud.tdis.io` |
| Email API | App Service + ACR | `api-portal.dev.cloud.tdis.io` |
| Map tiles | App Service + Docker Hub image | `portal-pgtile.dev.cloud.tdis.io` |
| Database | PostgreSQL Flexible Server | Key Vault secret — not public |
| Secrets | Key Vault | `tdis-dev-key-vault` |
| Ingress | Application Gateway | TLS + routing |
| Ingestion | Databricks → PostgreSQL | TDIS internal |

---

## How you deploy it (your Azure)

Same **logical** stack. You create resources manually (Portal or `az` CLI) — **no Terragrunt**.

```mermaid
flowchart TB
  Users[Users browser]

  subgraph yourAzure [Your Azure Subscription]
    SWA[Static Web App\nyour-portal URL]
    API[App Service + ACR\ntdis-portal-api container]
    EMAIL[App Service Node\ntdis-portal-backend]
    PGTILE[App Service\npg_tileserv Docker]
    PG[(PostgreSQL Flexible Server\ntdis_portal)]
    KV[Key Vault]
    ACR[Container Registry]
  end

  subgraph yourIngest [Your Databricks]
    DBX[Your sync jobs]
  end

  Users --> SWA
  SWA -->|HTTPS| API
  SWA -->|HTTPS| EMAIL
  SWA -->|HTTPS| PGTILE
  API --> PG
  PGTILE --> PG
  ACR --> API
  KV -.-> API
  KV -.-> PGTILE
  DBX --> PG
```

| Your component | Azure resource | See chapter |
|----------------|----------------|-------------|
| Frontend | Static Web App | [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md) §6 |
| FastAPI | App Service (Linux container) + ACR | §4–5 |
| Email | App Service (Node) | §8 |
| pg_tileserv | App Service — image `pramsey/pg_tileserv` | §7 |
| Database | PostgreSQL Flexible Server | §2 |
| Secrets | Key Vault | §3 |
| Ingestion | Your Databricks workspace + ADLS | [14-trd-reference.md](14-trd-reference.md) |

**Optional later:** Application Gateway for custom domain — not required to start.

---

## Local deployment (your laptop)

Same components, all on localhost — for development before Azure.

```mermaid
flowchart LR
  Browser --> FE[Vite :5173]
  FE --> API[FastAPI :8000]
  FE --> TILES[pg_tileserv :7800]
  FE --> EMAIL[Node email :8080]
  API --> PG[(Docker Postgres :5432)]
  TILES --> PG
```

Step-by-step: [06-local-development.md](06-local-development.md)

---

## Local ↔ Azure ↔ TDIS mapping

| Role | Local | Your Azure | TDIS (reference) |
|------|-------|------------|------------------|
| Frontend | `:5173` | Static Web App | SWA + App Gateway |
| API | `:8000` | App Service + ACR | App Service `portal-api-new` |
| Email | `:8080` | App Service Node | App Service `portal-api` |
| Tiles | Docker `:7800` | App Service Docker | App Service `pg_tileserv` |
| Database | Docker Postgres | Flexible Server | Flexible Server + KV |
| Secrets | `.env` | Key Vault | Key Vault |

---

## API surface (FastAPI)

Base path: `/api/v1` — full list in TRD and OpenAPI at `/docs` when API runs locally.

Data-heavy endpoints need **Source B** PostgreSQL objects from your Databricks sync — see [07-postgresql-schema.md](07-postgresql-schema.md).
