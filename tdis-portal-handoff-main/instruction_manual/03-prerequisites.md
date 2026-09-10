# 03 — Prerequisites

Run this **before** local or Azure setup. Every check below should succeed.

## Pre-flight checklist (copy-paste)

```bash
# 1. Docker installed and running
docker --version
docker run --rm hello-world

# 2. Python 3.11+
python3 --version

# 3. Node.js 20+
node --version

# 4. Git (optional — code may arrive as zip)
git --version

# 5. PostgreSQL client (psql) — needed to load schema
psql --version
```

### If something fails

| Check failed | Fix |
|--------------|-----|
| `docker` | Install [Docker Desktop](https://docs.docker.com/get-docker/) (Windows/Mac) or `sudo apt install docker.io` (Linux). Start Docker Desktop / `sudo systemctl start docker`. |
| `python3` not 3.11+ | Install Python 3.11 from python.org or `sudo apt install python3.11 python3.11-venv` |
| `node` not v20+ | Install from https://nodejs.org/ (LTS 20.x) |
| `psql` missing | `sudo apt install postgresql-client` (Linux) or install full PostgreSQL locally (includes psql) |

### Optional tools

| Tool | Needed for |
|------|------------|
| `make` | API quick start (`make dev`) — or follow manual steps without make |
| `curl` | Health checks |
| Azure CLI (`az`) | Azure deploy only — [04-azure-for-beginners.md](04-azure-for-beginners.md) |

---

## Mapbox account (required for maps)

The map **will not render** without a token.

1. Go to https://account.mapbox.com/auth/signup/  
2. Create a free account  
3. Open **Account → Access tokens**  
4. Copy the **Default public token** (starts with `pk.`)  
5. Save it — you will paste into `tdis-portal-frontend/.env` as `VITE_MAPBOX_TOKEN`

---

## SMTP (optional — Contact Us emails only)

Only if you test `tdis-portal-backend`. Use your org mail relay or skip email testing.

---

## Azure (Track B only)

| Requirement | Link |
|-------------|------|
| Azure account | https://azure.microsoft.com/free/ |
| Azure CLI | [04-azure-for-beginners.md](04-azure-for-beginners.md) |

Budget: ~$45–60/month for **Tier 1** minimal dev stack (see ch. 04). Tier 2 (VNet + App Gateway) costs significantly more — see [08c-azure-tier2-reference.md](08c-azure-tier2-reference.md).

---

## How many terminal windows? (local)

You will use **up to 4 terminals** at once:

| Terminal | Service | Port |
|----------|---------|------|
| 1 | Docker (Postgres + pg_tileserv) | 5432, 7800 |
| 2 | FastAPI (`make dev`) | 8000 |
| 3 | Frontend (`npm run dev`) | 5173 |
| 4 | Email backend (`npm start`) — optional | 8080 |

Keep terminals open while testing. `Ctrl+C` stops a service.

---

## Next step

- **Local:** [07-postgresql-schema.md](07-postgresql-schema.md) → [06-local-development.md](06-local-development.md)  
- **Azure:** [04-azure-for-beginners.md](04-azure-for-beginners.md) → [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md) (Tier 1)
