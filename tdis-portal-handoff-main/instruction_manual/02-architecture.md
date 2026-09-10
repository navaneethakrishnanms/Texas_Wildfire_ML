# 02 — Deployment Architecture

Three deployment models: **how TDIS runs production**, **how you deploy Tier 1 (first)**, and **local laptop**.

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

---

## Data flow (all environments)

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

---

## Diagram A — TDIS production (reference only)

TDIS uses **VNet, subnets, and Application Gateway**. You do **not** build this on day one.

```mermaid
flowchart TB
  Users[Users browser]

  subgraph vnet [VNet tdis-dev-vnet]
    subgraph edge [app-gateway-subnet]
      AGW[Application Gateway]
    end
    subgraph apps [app-services-subnet]
      APINew[portal-api-new FastAPI]
      PGTILE[pg_tileserv]
    end
    subgraph data [postgres-subnet or private link]
      PG[(PostgreSQL Flexible Server)]
    end
  end

  SWA[Static Web App]
  KV[Key Vault]
  DBX[Databricks sync]

  Users --> AGW
  AGW --> SWA
  AGW --> APINew
  AGW --> PGTILE
  APINew --> PG
  PGTILE --> PG
  KV -.-> APINew
  DBX --> PG
```

| TDIS component | Azure resource | Hostname (dev example) |
|----------------|----------------|------------------------|
| Frontend | Static Web App | `portal.dev.cloud.tdis.io` |
| Portal API | App Service + ACR, VNet integrated | `api-new-portal.dev.cloud.tdis.io` |
| Map tiles | App Service Docker, VNet integrated | `portal-pgtile.dev.cloud.tdis.io` |
| Database | PostgreSQL (private) | Key Vault — not public |
| Ingress | Application Gateway | TLS + routing |
| Secrets | Key Vault + managed identity | `tdis-dev-key-vault` |

Full networking detail: [08c-azure-tier2-reference.md](08c-azure-tier2-reference.md)

---

## Diagram B — BIT Tier 1 (follow ch. 08 first)

**No VNet.** Public App Service and Static Web App URLs. PostgreSQL public hostname + firewall rules.

```mermaid
flowchart TB
  Users[Users browser]

  subgraph tier1 [Your Azure — Tier 1]
    SWA[Static Web App\nswa-bit-portal.azurestaticapps.net]
    API[App Service\napp-bit-portal-api.azurewebsites.net]
    EMAIL[App Service email]
    PGTILE[App Service pg_tileserv]
    PG[(PostgreSQL public + firewall)]
    KV[Key Vault optional]
    ACR[Container Registry]
  end

  DBX[Your Databricks sync]

  Users --> SWA
  SWA -->|HTTPS| API
  SWA -->|HTTPS| EMAIL
  SWA -->|HTTPS| PGTILE
  API --> PG
  PGTILE --> PG
  ACR --> API
  DBX --> PG
```

Walkthrough: [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md) — **Easy** (Portal) or **Detailed** (CLI + optional Key Vault).

---

## Diagram C — Local laptop

```mermaid
flowchart LR
  Browser --> FE[Vite :5173]
  FE --> API[FastAPI :8000]
  FE --> TILES[pg_tileserv :7800]
  FE --> EMAIL[Node email :8080]
  API --> PG[(Docker Postgres :5432)]
  TILES --> PG
```

Walkthrough: [06-local-development.md](06-local-development.md)

---

## Side-by-side mapping

| Role | Local | BIT Tier 1 (ch. 08) | TDIS production |
|------|-------|---------------------|-----------------|
| Frontend | `:5173` | Static Web App URL | SWA + App Gateway |
| API | `:8000` | `*.azurewebsites.net` | VNet + App Gateway |
| Email | `:8080` | App Service Node | App Service + AGW |
| Tiles | Docker `:7800` | App Service Docker | VNet + App Gateway |
| Database | Docker Postgres | Flexible Server + firewall | Private PG in VNet |
| Secrets | `.env` files | App settings or Key Vault | Key Vault + MI |
| **Networking** | None | Public endpoints | VNet + subnets + AGW |
| Ingestion | (empty DB) | Your Databricks | TDIS Databricks |

---

## Which diagram should I follow?

| Your situation | Read |
|----------------|------|
| Never used Azure | [00-start-here.md](00-start-here.md) → local → **Diagram B / ch. 08 Easy path** |
| Comfortable with CLI | ch. 08 **Detailed path** |
| Security requires private DB / custom domain | [08c-azure-tier2-reference.md](08c-azure-tier2-reference.md) → **Diagram A** |
| First meeting with TDIS | This chapter + [14-trd-reference.md](14-trd-reference.md) |

---

## API surface (FastAPI)

Base path: `/api/v1` — OpenAPI at `/docs` when API runs.

Data-heavy endpoints need **Source B** PostgreSQL objects from Databricks sync — see [07-postgresql-schema.md](07-postgresql-schema.md).
