# 03 — Prerequisites

## Track A — Local development

### Required software

| Tool | Version | Install |
|------|---------|---------|
| **Python** | 3.11+ | https://www.python.org/downloads/ |
| **Node.js** | 20+ | https://nodejs.org/ |
| **Docker** | Latest | https://docs.docker.com/get-docker/ |
| **Docker Compose** | v2 | Included with Docker Desktop |
| **Git** | Any | Optional (code is pre-unzipped) |
| **psql** | PostgreSQL client | `sudo apt install postgresql-client` or use Docker |

### Mapbox account (required for maps)

The map **will not render** without a Mapbox token.

1. Go to https://account.mapbox.com/auth/signup/  
2. Create a free account  
3. Open **Account → Access tokens**  
4. Copy the **Default public token** (starts with `pk.`)  
5. You will paste it into `tdis-portal-frontend/.env` as `VITE_MAPBOX_TOKEN`

> Mapbox free tier is sufficient for development. BIT owns this account — TDIS does not share a token.

### SMTP (optional — for Contact Us emails)

Only needed if you test `tdis-portal-backend`. Use your org's mail relay or skip email testing locally.

---

## Track B — Azure deployment

Everything in Track A, plus:

| Tool | Install |
|------|---------|
| **Azure account** | https://azure.microsoft.com/free/ |
| **Azure CLI** | https://learn.microsoft.com/en-us/cli/azure/install-azure-cli |

After install:

```bash
az login
az account set --subscription "<your-subscription-id>"
az account show   # verify
```

### Rough Azure cost (dev-sized)

| Resource | Approx. monthly |
|----------|-----------------|
| PostgreSQL Flexible Server (B1ms) | $25–40 |
| App Service Plan (B1) | $13 |
| Static Web App (Free) | $0 |
| Key Vault | ~$1 |
| Container Registry (Basic) | ~$5 |
| **Total (minimal)** | **~$45–60** |

Costs vary by region and usage. Shut down resources when not testing.

---

## Knowledge assumptions

This manual explains Azure from scratch in [04-azure-for-beginners.md](04-azure-for-beginners.md). No prior cloud experience required.

## Next step

- **Local:** [05-get-the-code.md](05-get-the-code.md)  
- **Azure:** [04-azure-for-beginners.md](04-azure-for-beginners.md)
