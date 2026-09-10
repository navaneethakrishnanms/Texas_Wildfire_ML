# 06 — Local Development (Track A)

Run the portal on your laptop. Do [03-prerequisites.md](03-prerequisites.md) and [07-postgresql-schema.md](07-postgresql-schema.md) first.

> **Open a new terminal window** for each step below (Step 1, 2, 3…). In VS Code: Terminal → New Terminal.

### Windows notes

| Task | Windows command |
|------|-----------------|
| Activate Python venv | `venv\Scripts\activate` (not `source venv/bin/activate`) |
| Docker compose | Use Docker Desktop terminal or WSL2 |
| Paste in terminal | Ctrl+Shift+V in some terminals, or right-click → Paste |

> **You need the code repos** (sent separately). Paths below assume:
> ```
> workspace/
> ├── instruction_manual/    ← this manual
> ├── tdis-portal-api/
> ├── tdis-portal-frontend/
> └── tdis-portal-backend/
> ```

## What runs where

| Service | Port | How |
|---------|------|-----|
| PostgreSQL | 5432 | Docker |
| pg_tileserv | 7800 | Docker |
| FastAPI | 8000 | Terminal 2 |
| React (Vite) | 5173 | Terminal 3 |
| Email API | 8080 | Terminal 4 (optional) |

---

## Step 1 — Start database + tiles (Terminal 1)

### Option A — docker compose (recommended for beginners)

```bash
cd instruction_manual
docker compose -f docker-compose.local.yml up -d
```

Wait 10–30 seconds for Postgres to become healthy:

```bash
docker compose -f docker-compose.local.yml ps
# postgres should show "healthy"
```

**Expected:** Two containers running. Check:

```bash
docker ps
# Should list tdis-postgres and tdis-pg-tileserv
```

### Load schema (once)

From your **workspace** folder (parent of `instruction_manual/`):

```bash
psql -h localhost -U postgres -d tdis_portal \
  -f instruction_manual/schema/portal_schema.sql
```

Password when prompted: `postgres` (characters may not appear as you type — that is normal)

**Expected:** Many `CREATE TABLE` / `CREATE VIEW` messages, no `ERROR` at the end.

**If you see `extension "postgis" is not available`:** use [docker-compose.local.yml](docker-compose.local.yml) as shipped (PostGIS image). Do not use plain `postgres:15-alpine`.

**Verify:**

```bash
psql -h localhost -U postgres -d tdis_portal -c "\dt"
# Should list tables like event_types, events_store, batch_jobs
```

**Verify pg_tileserv:**

Open http://localhost:7800/health in a browser — should return OK.

> **Optional:** For Texas county boundaries on the map, run `008_counties_table_seed_data.sql` from `tdis-portal-db` separately (contains data; not in bundled schema).

---

## Step 2 — Start API (Terminal 2 — new window)

**Mac / Linux / WSL:**

```bash
cd tdis-portal-api
python3 -m venv venv
source venv/bin/activate
make setup
make dev
```

**Windows (no WSL)** — `make` requires bash; use:

```powershell
cd tdis-portal-api
python -m venv venv
venv\Scripts\activate
copy env.example .env
pip install -r requirements/dev.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected:** Server starts on port 8000. Look for a line mentioning the dev API key.

**Verify** — browser is fine if `curl` is missing:

- http://localhost:8000/api/v1/health  
- http://localhost:8000/docs (Swagger UI)

**If API fails to connect to DB:** check `.env` has `DATABASE_HOST=localhost`, `DATABASE_NAME=tdis_portal`, password `postgres`.

---

## Step 3 — Start frontend (Terminal 3 — new window)

```bash
cd tdis-portal-frontend
cp .env.example .env
```

1. Open `tdis-portal-frontend/.env` in VS Code or Notepad.  
2. Set **only** these lines (delete or comment out other `VITE_*` lines, especially `VITE_CUBE_*` and `VITE_KEYCLOAK_*`):

```env
VITE_MAPBOX_TOKEN=pk.PASTE_YOUR_MAPBOX_TOKEN_HERE
VITE_API_URL=http://127.0.0.1:8000/api/v1
VITE_EVENTS_API_KEY=tdis_dev_master_key_2024_secure_token_123456789
VITE_TILE_SERVER_BASE_URL=http://127.0.0.1:7800
VITE_EMAIL_API_URL=http://127.0.0.1:8080
```

3. Save the file.  
4. If dev server was already running, stop it (`Ctrl+C`) before continuing.

**Do not set:** `VITE_CUBE_*`, `VITE_AI_*`, `VITE_KEYCLOAK_*`

```bash
npm install --legacy-peer-deps
npm run dev
```

**Expected:** `Local: http://localhost:5173/`

Open that URL. **You should see the portal UI and a map** (Mapbox token required).

**Common issues:**
- Blank map → wrong or missing `VITE_MAPBOX_TOKEN`
- Network errors in browser console → API not running (Step 2) or wrong `VITE_API_URL`
- CORS errors → add `http://localhost:5173` to API `BACKEND_CORS_ORIGINS` in `.env`, restart API

---

## Step 4 — Email backend (Terminal 4 — optional)

Skip if you only need maps and data widgets.

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
```

**Verify:** http://localhost:8080/health

Without real SMTP, Contact Us will error — that is OK.

---

## Success checklist

- [ ] http://localhost:5173 loads  
- [ ] Map visible (not gray blank)  
- [ ] http://localhost:8000/docs loads  
- [ ] http://localhost:7800/health OK  
- [ ] No red CORS errors in browser DevTools (F12 → Console)  
- [ ] Data widgets may be **empty** until Databricks sync tables exist — **expected**

---

## Stop everything

```bash
# Stop API / frontend / email: Ctrl+C in each terminal

# Stop Docker:
cd instruction_manual
docker compose -f docker-compose.local.yml down
```

---

## Next: Azure

When local works, see [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md) (**Tier 1** — no VNet required).

| Local | Azure Tier 1 |
|-------|--------------|
| Docker Postgres | PostgreSQL Flexible Server |
| `make dev` | App Service + ACR |
| `npm run dev` | Static Web App |
| pg_tileserv Docker | App Service (same image) |
| email `:8080` | App Service (Node) |

For TDIS production networking (VNet, App Gateway), see [08c-azure-tier2-reference.md](08c-azure-tier2-reference.md).

Troubleshooting: [12-troubleshooting-faq.md](12-troubleshooting-faq.md)

---

## Advanced — separate docker commands (skip unless compose fails)

Only use if `docker compose` does not work:

```bash
docker run -d --name tdis-postgres \
  -e POSTGRES_DB=tdis_portal \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  postgis/postgis:15-3.4-alpine

docker run -d --name tdis-pg-tileserv -p 7800:7800 \
  --link tdis-postgres:postgres \
  -e DATABASE_URL="postgresql://postgres:postgres@postgres:5432/tdis_portal?sslmode=disable" \
  pramsey/pg_tileserv:20240614
```

On Linux, if tiles cannot reach Postgres, use [docker-compose.local.yml](docker-compose.local.yml) instead.
