# 06 — Local Development (Track A)

Run the full portal stack on your laptop. Complete [07-postgresql-schema.md](07-postgresql-schema.md) first.

## Overview — services and ports

| Service | Port | Folder |
|---------|------|--------|
| PostgreSQL | 5432 | Docker |
| FastAPI | 8000 | `tdis-portal-api` |
| React (Vite) | 5173 | `tdis-portal-frontend` |
| pg_tileserv | 7800 | Docker (`pramsey/pg_tileserv`) |
| Email backend | 8080 | `tdis-portal-backend` |

## Step 1 — PostgreSQL

```bash
docker run -d --name tdis-postgres \
  -e POSTGRES_DB=tdis_portal \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  postgres:15-alpine
```

Load schema:

```bash
psql -h localhost -U postgres -d tdis_portal \
  -f instruction_manual/schema/portal_schema.sql
# Password: postgres
```

## Step 2 — FastAPI backend

```bash
cd tdis-portal-api
python3.11 -m venv venv
source venv/bin/activate
make setup    # creates .env
make install
make dev
```

Verify:

```bash
curl -s http://localhost:8000/api/v1/health
curl -s -H "X-API-Key: tdis_dev_master_key_2024_secure_token_123456789" \
  http://localhost:8000/api/v1/health/detailed
```

API docs: http://localhost:8000/docs

**Default dev API key** (printed on startup / in `scripts/setup-env.sh`):

```
tdis_dev_master_key_2024_secure_token_123456789
```

### Alternative: Docker Compose (API + Postgres + Redis)

```bash
cd tdis-portal-api/docker
docker compose up -d
```

## Step 3 — pg_tileserv (vector map tiles)

External Docker container — not TDIS code.

```bash
docker run -d --name pg_tileserv -p 7800:7800 \
  -e DATABASE_URL="postgresql://postgres:postgres@host.docker.internal:5432/tdis_portal?sslmode=disable" \
  pramsey/pg_tileserv:20240614
```

On Linux if `host.docker.internal` fails, use your machine IP or `--network host`.

Health check: http://localhost:7800/health

## Step 4 — Email backend (optional)

```bash
cd tdis-portal-backend
npm install
```

Create `.env`:

```env
NODE_ENV=development
CONTACT_US_FORM_EMAIL_ADDRESS=you@example.com
MAIL_SERVER=smtp.example.com
MAIL_PORT=587
MAIL_USERNAME=
MAIL_PASSWORD=
MAIL_DISPLAY_NAME=TDIS Portal Contact
MAIL_REPLY_TO=you@example.com
SUBSCRIBE_USER_REQUEST_EMAIL_ADDRESS=subscribe@example.com
```

```bash
npm start
# http://localhost:8080/health
```

Without valid SMTP, Contact Us forms will fail — that is OK for map/API testing.

## Step 5 — Frontend

```bash
cd tdis-portal-frontend
cp .env.example .env
```

Edit `.env` — **minimum required**:

```env
VITE_MAPBOX_TOKEN=pk.YOUR_TOKEN_FROM_MAPBOX.com
VITE_API_URL=http://127.0.0.1:8000/api/v1
VITE_EVENTS_API_KEY=tdis_dev_master_key_2024_secure_token_123456789
VITE_TILE_SERVER_BASE_URL=http://127.0.0.1:7800
VITE_EMAIL_API_URL=http://127.0.0.1:8080
```

**Do not set:** `VITE_CUBE_*`, `VITE_AI_*`, `VITE_KEYCLOAK_*`

```bash
npm install --legacy-peer-deps
npm run dev
```

Open http://localhost:5173

## Success checklist

- [ ] Map renders (Mapbox token works)  
- [ ] No CORS errors in browser console  
- [ ] `curl` health check returns 200  
- [ ] API docs load at `/docs`  
- [ ] Dashboard widgets may be **empty** until Source B tables exist — expected  

## Local ↔ Azure

When this works locally, use [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md) to deploy the same components to Azure.

| Local | Azure |
|-------|-------|
| Docker Postgres | PostgreSQL Flexible Server |
| `make dev` | App Service + ACR |
| `npm run dev` | Static Web App |
| pg_tileserv Docker | App Service (same image) |
| email `:8080` | App Service (Node) |

## Troubleshooting

See [12-troubleshooting-faq.md](12-troubleshooting-faq.md).
