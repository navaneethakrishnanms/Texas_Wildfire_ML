# 08b — Azure Resource Checklist

You do **not** need TDIS infrastructure repos. Create these in **your** Azure subscription.

## Required resources

| # | Azure resource | Purpose | Local equivalent |
|---|----------------|---------|------------------|
| 1 | **Resource group** | Container for all portal resources | — |
| 2 | **PostgreSQL Flexible Server** | Database `tdis_portal` | Docker Postgres |
| 3 | **Key Vault** | DB password, API secrets | `.env` files |
| 4 | **Container Registry (ACR)** | Store API Docker image | local Docker build |
| 5 | **App Service Plan** | Hosts web apps | — |
| 6 | **App Service (Linux)** ×2–3 | API container, pg_tileserv, email | `:8000`, `:7800`, `:8080` |
| 7 | **Static Web App** | React frontend build | `npm run dev` |

## Optional (later)

| Resource | Purpose |
|----------|---------|
| Application Gateway | Custom domain + TLS |
| Azure Front Door | CDN / global entry |
| Databricks workspace | Your ingestion pipelines (TRD) |
| ADLS Gen2 | Bronze/Silver/Gold storage (TRD) |

## Suggested naming (example)

Replace `bit` with your prefix:

| Resource | Example name |
|----------|--------------|
| Resource group | `rg-bit-portal-dev` |
| PostgreSQL | `pg-bit-portal-dev` |
| Key Vault | `kv-bit-portal-dev` |
| ACR | `acrbitportaldev` |
| App Service Plan | `plan-bit-portal-dev` |
| API App Service | `app-bit-portal-api` |
| SWA | `swa-bit-portal` |
| pg_tileserv | `app-bit-pgtile` |
| Email backend | `app-bit-portal-email` |

## Region

Pick one region and use it for all resources (e.g. `southcentralus` near Texas).

## Secrets to store in Key Vault

| Secret name (suggested) | Used by |
|-------------------------|---------|
| `database-host` | API, pg_tileserv |
| `database-user` | API, pg_tileserv |
| `database-password` | API, pg_tileserv |
| `database-name` | API (`tdis_portal`) |
| `api-secret-key` | API (`SECRET_KEY`) |
| `mail-password` | Email backend |

## Estimated monthly cost

See [03-prerequisites.md](03-prerequisites.md) (~$45–60 minimal dev).

## Next step

[08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md) — create each resource with commands.
