# 08b — Azure Resource Checklist

You do **not** need TDIS infrastructure repos. Create these in **your** Azure subscription.

## Tier 1 — required for first deploy (follow ch. 08)

These are enough to run the portal. **No VNet or Application Gateway needed.**

| # | Azure resource | Purpose | Local equivalent |
|---|----------------|---------|------------------|
| 1 | **Resource group** | Container for all portal resources | — |
| 2 | **PostgreSQL Flexible Server** | Database `tdis_portal` | Docker Postgres |
| 3 | **Key Vault** | DB password, API secrets (Detailed path only) | `.env` files |
| 4 | **Container Registry (ACR)** | Store API Docker image | local Docker build |
| 5 | **App Service Plan** | Hosts web apps | — |
| 6 | **App Service (Linux)** ×2–3 | API container, pg_tileserv, email | `:8000`, `:7800`, `:8080` |
| 7 | **Static Web App** | React frontend build | `npm run dev` |

**Walkthrough:** [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md)

---

## Tier 2 — TDIS production parity (reference only)

For VNet, subnets, Application Gateway, and private PostgreSQL — **do not build on day one.**

| Resource | Purpose |
|----------|---------|
| Virtual Network + subnets | Private networking |
| Application Gateway | Custom domains, TLS ingress |
| Private PostgreSQL | No public DB endpoint |
| NSG rules | Outbound lockdown |

**Reference:** [08c-azure-tier2-reference.md](08c-azure-tier2-reference.md)

---

## Optional (later, either tier)

| Resource | Purpose |
|----------|---------|
| Azure Front Door | CDN / global entry |
| Databricks workspace | Your ingestion pipelines (TRD) |
| ADLS Gen2 | Bronze/Silver/Gold storage (TRD) |

---

## Suggested naming (example)

Replace `bit` with your prefix:

| Resource | Example name |
|----------|--------------|
| Resource group | `rg-bit-portal-dev` |
| PostgreSQL | `pg-bit-portal-dev` |
| Key Vault | `kv-bit-portal-dev` |
| ACR | `acrbitportaldev` (globally unique, alphanumeric only) |
| App Service Plan | `plan-bit-portal-dev` |
| API App Service | `app-bit-portal-api` |
| SWA | `swa-bit-portal` |
| pg_tileserv | `app-bit-pgtile` |
| Email backend | `app-bit-portal-email` |

---

## Region

Pick one region and use it for all resources (e.g. `southcentralus` near Texas).

Static Web App must be created in a [supported region](https://learn.microsoft.com/en-us/azure/static-web-apps/overview) — e.g. `centralus`.

---

## Secrets to store in Key Vault

| Secret name (suggested) | Used by |
|-------------------------|---------|
| `database-host` | API, pg_tileserv |
| `database-user` | API, pg_tileserv |
| `database-password` | API, pg_tileserv |
| `database-name` | API (`tdis_portal`) |
| `api-secret-key` | API (`SECRET_KEY`) |
| `mail-password` | Email backend |

---

## Estimated monthly cost

| Tier | Approx. cost |
|------|--------------|
| Tier 1 only | ~$45–60 — see [03-prerequisites.md](03-prerequisites.md) |
| Tier 2 (+ App Gateway) | + ~$150/mo |

---

## Next step

[08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md) — Tier 1 step-by-step deploy
