# 08 — Deploy to Azure by Hand (Track B — Tier 1)

Deploy the same stack as [06-local-development.md](06-local-development.md) to **your Azure subscription**.

**Prerequisites:** [04-azure-for-beginners.md](04-azure-for-beginners.md), [08b-azure-resource-checklist.md](08b-azure-resource-checklist.md), local deploy working (recommended).

> **Two paths in every phase:** **Easy** = Azure Portal clicks. **Detailed** = `az` CLI + Key Vault (how TDIS does production). Beginners: use **Easy** first.

---

## Before you start — name your resources

Copy this list. Change **only** names and the password. Write them down — you will use them in every phase.

| Variable | Example value | Your value |
|----------|---------------|------------|
| Resource group | `rg-bit-portal-dev` | __________ |
| Region | `southcentralus` | __________ |
| PostgreSQL server name | `pg-bit-portal-dev` | __________ |
| Database name | `tdis_portal` | (keep this) |
| DB admin user | `portaladmin` | __________ |
| DB password | strong password | __________ |
| Key Vault name | `kv-bit-portal-dev` | __________ |
| ACR name (letters/numbers only, globally unique) | `acrbitportaldev001` | __________ |
| App Service Plan | `plan-bit-portal-dev` | __________ |
| API App Service | `app-bit-portal-api` | __________ |
| Static Web App | `swa-bit-portal` | __________ |
| pg_tileserv App Service | `app-bit-pgtile` | __________ |
| Email App Service | `app-bit-portal-email` | __________ |

> **Tier 1 API auth note:** The code snapshot only auto-loads the dev API key when `ENVIRONMENT=development`. For Tier 1 Azure, use **`ENVIRONMENT=development`** in App Service settings (not `production`) until TDIS provides a production key mechanism. See Phase 5.

**Detailed path only** — paste into terminal once:

```bash
RG=rg-bit-portal-dev
LOC=southcentralus
PG_SERVER=pg-bit-portal-dev
KV_NAME=kv-bit-portal-dev
ACR_NAME=acrbitportaldev001
PLAN=plan-bit-portal-dev
API_APP=app-bit-portal-api
SWA_NAME=swa-bit-portal
TILE_APP=app-bit-pgtile
EMAIL_APP=app-bit-portal-email
PG_ADMIN=portaladmin
PG_PASS='ChangeMe-StrongPass123!'
```

---

## Phase 1 — Resource group

> **What this is:** A folder in Azure that holds all your portal resources.

### Easy — Azure Portal

1. Open https://portal.azure.com  
2. Search **Resource groups** → **Create**  
3. Subscription: yours | Resource group name: `rg-bit-portal-dev` | Region: `South Central US`  
4. Click **Review + create** → **Create**  

[Screenshot: Create resource group form]

### Detailed — CLI

```bash
az group create --name $RG --location $LOC
```

### STOP — verify

- [ ] Portal → Resource groups → your group appears  
- **Good:** Status shows the group in the list  
- **Bad:** "Authorization failed" → ask your Azure admin for Contributor access on the subscription  

---

## Phase 2 — PostgreSQL database

> **What this is:** The database where portal data lives (empty until your Databricks sync fills it).

### Easy — Azure Portal

1. Search **Azure Database for PostgreSQL flexible server** → **Create**  
2. Resource group: `rg-bit-portal-dev`  
3. Server name: `pg-bit-portal-dev` (must be unique worldwide)  
4. Region: same as resource group  
5. PostgreSQL version: **15**  
6. Workload: **Development**  
7. Compute + storage: **Burstable B1ms** (cheapest dev size)  
8. Admin username: `portaladmin` | Password: your strong password  
9. Networking: **Public access** — allow public access to this resource  
10. Create  

After server is ready:

11. Open the server → **Databases** → **Add** → name: `tdis_portal`  
12. Open **Networking** → **Add current client IP address** → Save  
13. Enable **Allow public access from any Azure service within Azure** (checkbox) → Save  

[Screenshot: PostgreSQL networking blade with firewall rules]

### Detailed — CLI

```bash
az postgres flexible-server create \
  --resource-group $RG \
  --name $PG_SERVER \
  --location $LOC \
  --admin-user $PG_ADMIN \
  --admin-password "$PG_PASS" \
  --sku-name Standard_B1ms \
  --tier Burstable \
  --version 15 \
  --storage-size 32 \
  --public-access 0.0.0.0

az postgres flexible-server db create \
  --resource-group $RG \
  --server-name $PG_SERVER \
  --database-name tdis_portal

# Allow YOUR laptop IP (for loading schema)
MYIP=$(curl -s ifconfig.me)
az postgres flexible-server firewall-rule create \
  --resource-group $RG --name $PG_SERVER \
  --rule-name AllowMyIP --start-ip-address $MYIP --end-ip-address $MYIP

# Allow Azure App Services to connect (0.0.0.0 = special "Azure services" rule)
az postgres flexible-server firewall-rule create \
  --resource-group $RG --name $PG_SERVER \
  --rule-name AllowAzureServices --start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0
```

### Load schema (both paths)

From your workspace folder (where `instruction_manual/` lives):

```bash
psql "host=pg-bit-portal-dev.postgres.database.azure.com port=5432 dbname=tdis_portal user=portaladmin sslmode=require" \
  -f instruction_manual/schema/portal_schema.sql
```

Replace `pg-bit-portal-dev` and `portaladmin` with your names. Enter password when prompted.

### STOP — verify

```bash
psql "host=YOUR_SERVER.postgres.database.azure.com port=5432 dbname=tdis_portal user=YOUR_ADMIN sslmode=require" -c "\dt"
```

- [ ] Lists tables: `event_types`, `events_store`, `batch_jobs`, etc.  
- **Good:** Table list appears, no `ERROR`  
- **Bad:** `connection timed out` → add your IP in Networking (step 12 above)  
- **Bad:** `password authentication failed` → wrong password  

---

## Phase 3 — Key Vault (Detailed path — optional for Easy path)

> **What this is:** A secure vault for passwords. TDIS production uses this. **Easy path skips to Phase 4** and pastes secrets directly into App Service settings.

### Easy — skip this phase

Write your DB password on a sticky note (dev only). You will paste it into App Service in Phase 5.

### Detailed — CLI + Portal

```bash
az keyvault create --name $KV_NAME --resource-group $RG --location $LOC

az keyvault secret set --vault-name $KV_NAME --name database-host \
  --value "${PG_SERVER}.postgres.database.azure.com"
az keyvault secret set --vault-name $KV_NAME --name database-user --value "$PG_ADMIN"
az keyvault secret set --vault-name $KV_NAME --name database-password --value "$PG_PASS"
az keyvault secret set --vault-name $KV_NAME --name database-name --value "tdis_portal"
az keyvault secret set --vault-name $KV_NAME --name api-secret-key --value "$(openssl rand -hex 32)"
```

### STOP — verify (Detailed only)

- [ ] Portal → Key vaults → your vault → **Secrets** → five secrets listed  

---

## Phase 4 — Container Registry + API Docker image

> **What this is:** Stores the FastAPI backend as a Docker image Azure can run.

### Easy — Azure Portal

1. Search **Container registry** → **Create**  
2. Resource group: yours | Name: `acrbitportaldev001` (must be globally unique) | SKU: **Basic**  
3. Create  
4. Open registry → **Services** → **Repositories** (empty for now)  

Build the image from your laptop (requires Azure CLI — one-time):

```bash
cd tdis-portal-api
az acr build --registry acrbitportaldev001 --image portal-api:latest -f docker/Dockerfile .
```

Wait 5–10 minutes. Refresh **Repositories** — you should see `portal-api`.

[Screenshot: ACR Repositories showing portal-api]

### Detailed — CLI

```bash
az acr create --resource-group $RG --name $ACR_NAME --sku Basic --admin-enabled true
cd tdis-portal-api
az acr build --registry $ACR_NAME --image portal-api:latest -f docker/Dockerfile .
```

### STOP — verify

- [ ] ACR → Repositories → `portal-api` → tag `latest` exists  
- **Bad:** `ACR name already exists` → pick a new unique `ACR_NAME`  

---

## Phase 5 — App Service Plan + API

> **What this is:** Runs the FastAPI backend 24/7 in Azure.

Host name will be: `https://app-bit-portal-api.azurewebsites.net`

### Easy — Azure Portal

1. Search **App Service Plan** → **Create**  
   - OS: **Linux** | Region: yours | SKU: **Basic B1**  
2. Search **App Service** → **Create**  
   - Publish: **Docker Container**  
   - Image source: **Azure Container Registry** → pick your registry → image `portal-api:latest`  
   - Name: `app-bit-portal-api`  
3. After created, open App Service → **Settings** → **Environment variables** (or **Configuration** → **Application settings**)  
4. Add these settings (click **+ Add** for each):

| Name | Value |
|------|-------|
| `ENVIRONMENT` | `development` |
| `DEBUG` | `false` |
| `DATABASE_HOST` | `pg-bit-portal-dev.postgres.database.azure.com` |
| `DATABASE_PORT` | `5432` |
| `DATABASE_USER` | `portaladmin` |
| `DATABASE_PASSWORD` | your DB password |
| `DATABASE_NAME` | `tdis_portal` |
| `SECRET_KEY` | any long random string (e.g. 64 hex chars) |
| `BACKEND_CORS_ORIGINS` | `http://localhost:5173` (add SWA URL in Phase 7 after SWA is created) |

5. **Save** → click **Continue** when warned about restart  
6. **Settings** → **Health check** → Path: `/api/v1/health/liveness` → Save  

Enable ACR pull (Portal):

7. **Settings** → **Identity** → **System assigned** → **On** → Save  
8. Note the **Object (principal) ID**  
9. Go to your **Container registry** → **Access control (IAM)** → **Add role assignment**  
   - Role: **AcrPull** | Members: your App Service name → Save  

> **IAM note:** Assigning roles requires **Owner** or **User Access Administrator** on the resource group. If this fails, ask your Azure admin to grant AcrPull to the App Service identity.

10. **Settings** → **Configuration** → **General settings** → ensure container pulls via managed identity (or run Detailed command `acrUseManagedIdentityCreds` below).

[Screenshot: App Service application settings with DATABASE_* values]

### Detailed — CLI (plain settings OR Key Vault refs)

**Option B1 — plain settings (same as Easy):**

```bash
az appservice plan create --name $PLAN --resource-group $RG --sku B1 --is-linux

az webapp create --resource-group $RG --plan $PLAN --name $API_APP \
  --deployment-container-image-name ${ACR_NAME}.azurecr.io/portal-api:latest

ACR_ID=$(az acr show --name $ACR_NAME --query id -o tsv)
PRINCIPAL=$(az webapp identity assign --resource-group $RG --name $API_APP --query principalId -o tsv)
az role assignment create --assignee $PRINCIPAL --role AcrPull --scope $ACR_ID

az webapp config set --resource-group $RG --name $API_APP \
  --generic-configurations '{"acrUseManagedIdentityCreds": true}'

az webapp config appsettings set --resource-group $RG --name $API_APP --settings \
  ENVIRONMENT=development DEBUG=false \
  DATABASE_HOST="${PG_SERVER}.postgres.database.azure.com" \
  DATABASE_PORT=5432 \
  DATABASE_USER="$PG_ADMIN" \
  DATABASE_PASSWORD="$PG_PASS" \
  DATABASE_NAME=tdis_portal \
  SECRET_KEY="$(openssl rand -hex 32)" \
  BACKEND_CORS_ORIGINS="http://localhost:5173"

az webapp config set --resource-group $RG --name $API_APP \
  --generic-configurations '{"healthCheckPath": "/api/v1/health/liveness"}'
```

**Option B2 — Key Vault references (TDIS production style):**

After Phase 3 secrets exist and App Service identity is on:

```bash
# Grant Key Vault read to App Service (RBAC — default on new vaults)
API_PRINCIPAL=$(az webapp identity show -g $RG -n $API_APP --query principalId -o tsv)
KV_ID=$(az keyvault show -n $KV_NAME -g $RG --query id -o tsv)
az role assignment create --assignee $API_PRINCIPAL --role "Key Vault Secrets User" --scope $KV_ID

# Legacy access-policy vaults only (if RBAC disabled):
# az keyvault set-policy --name $KV_NAME --object-id $API_PRINCIPAL --secret-permissions get list

az webapp config appsettings set --resource-group $RG --name $API_APP --settings \
  ENVIRONMENT=development DEBUG=false \
  DATABASE_HOST="@Microsoft.KeyVault(VaultName=${KV_NAME};SecretName=database-host)" \
  DATABASE_USER="@Microsoft.KeyVault(VaultName=${KV_NAME};SecretName=database-user)" \
  DATABASE_PASSWORD="@Microsoft.KeyVault(VaultName=${KV_NAME};SecretName=database-password)" \
  DATABASE_NAME="@Microsoft.KeyVault(VaultName=${KV_NAME};SecretName=database-name)" \
  SECRET_KEY="@Microsoft.KeyVault(VaultName=${KV_NAME};SecretName=api-secret-key)" \
  BACKEND_CORS_ORIGINS="http://localhost:5173"
```

### STOP — verify

Wait 2–3 minutes after save, then:

```bash
curl -s "https://app-bit-portal-api.azurewebsites.net/api/v1/health/liveness"
```

- [ ] Returns JSON (not 503, not HTML error page)  
- **Good:** `{"status":"alive"}` or similar  
- **Bad:** 503 for 10+ min → App Service → **Log stream** — look for `ImagePullBackOff` (fix ACR role + `acrUseManagedIdentityCreds`, wait 2–5 min) or DB connection errors (fix firewall)  
- **Bad:** API works but widgets 401 → confirm `ENVIRONMENT=development` and frontend uses dev master key  
- **Bad:** Key Vault value shows literal `@Microsoft.KeyVault(...)` in Portal → run RBAC role assignment (Option B2)  

API base URL for frontend: `https://app-bit-portal-api.azurewebsites.net/api/v1`

---

## Phase 6 — pg_tileserv (map tiles)

> **What this is:** Serves map vector tiles from PostgreSQL. Uses public Docker image `pramsey/pg_tileserv`.

### Easy — Azure Portal

1. **App Service** → **Create** (same plan as API)  
2. Publish: **Docker Container** | Image: Docker Hub  
   - Access type: Public | Image and tag: `pramsey/pg_tileserv:20240614`  
3. Name: `app-bit-pgtile`  
4. **Environment variables**:

| Name | Value |
|------|-------|
| `WEBSITES_PORT` | `7800` |
| `PORT` | `7800` |
| `DATABASE_URL` | `postgresql://portaladmin:YOUR_PASSWORD@pg-bit-portal-dev.postgres.database.azure.com:5432/tdis_portal?sslmode=require` |

Replace user, password, and host with yours. **Save.**

### Detailed — CLI

```bash
az webapp create --resource-group $RG --plan $PLAN --name $TILE_APP \
  --deployment-container-image-name pramsey/pg_tileserv:20240614

az webapp config appsettings set --resource-group $RG --name $TILE_APP --settings \
  WEBSITES_PORT=7800 PORT=7800 \
  DATABASE_URL="postgresql://${PG_ADMIN}:${PG_PASS}@${PG_SERVER}.postgres.database.azure.com:5432/tdis_portal?sslmode=require"
```

For Key Vault: store `DATABASE_URL` as a secret and use `@Microsoft.KeyVault(...)` — grant tile app identity `get` on Key Vault same as Phase 5.

### STOP — verify

Open: `https://app-bit-pgtile.azurewebsites.net/health`

- [ ] Returns OK / healthy  
- **Bad:** 502 → wait 2 min; check `WEBSITES_PORT=7800` is set  

---

## Phase 7 — Static Web App (frontend)

> **What this is:** Hosts the React website. **Important:** Mapbox token and API URLs must be set **before** you run `npm run build`.

### Easy + Detailed — build locally, deploy with SWA CLI

1. Create Static Web App in Portal: Search **Static Web App** → Create → Region: **Central US** (required for SWA) → SKU: Free  

2. **Copy your SWA hostname** — Portal → Static Web App → **URL** looks like `https://nice-field-0123456789.2.azurestaticapps.net` (NOT the resource name `swa-bit-portal`).

3. **Update API CORS** — App Service → Environment variables → edit `BACKEND_CORS_ORIGINS` to include your SWA URL:
   ```
   https://nice-field-0123456789.2.azurestaticapps.net,http://localhost:5173
   ```
   Save and wait for API restart.

4. On your laptop:

```bash
cd tdis-portal-frontend

# Set these BEFORE build — replace with YOUR values
export VITE_MAPBOX_TOKEN=pk.your-mapbox-token-here
export VITE_API_URL=https://app-bit-portal-api.azurewebsites.net/api/v1
export VITE_EVENTS_API_KEY=tdis_dev_master_key_2024_secure_token_123456789
export VITE_TILE_SERVER_BASE_URL=https://app-bit-pgtile.azurewebsites.net
export VITE_EMAIL_API_URL=https://app-bit-portal-email.azurewebsites.net

npm ci --legacy-peer-deps
npm run build
```

3. Install deploy tool (once): `npm i -g @azure/static-web-apps-cli`

4. Get deploy token: Portal → Static Web App → **Manage deployment token** → copy  

5. Deploy:

```bash
swa deploy ./dist --deployment-token PASTE_YOUR_TOKEN_HERE
```

**Windows PowerShell** — set env vars differently:

```powershell
$env:VITE_MAPBOX_TOKEN="pk.your-token"
$env:VITE_API_URL="https://app-bit-portal-api.azurewebsites.net/api/v1"
# ... etc
npm run build
```

### STOP — verify

- [ ] Portal → Static Web App → URL opens in browser  
- [ ] Map visible (not gray)  
- [ ] Browser F12 → Console — no red CORS errors  
- **Bad:** Gray map → `VITE_MAPBOX_TOKEN` wrong or missing at build time — rebuild  
- **Bad:** CORS errors → add SWA URL to API `BACKEND_CORS_ORIGINS`, restart API  

---

## Phase 8 — Email backend (optional)

Skip if you only need maps and API.

### Easy — Azure Portal

1. **App Service** → Create → Publish: **Code** | Runtime: **Node 20 LTS**  
2. Name: `app-bit-portal-email`  
3. Deploy code: zip `tdis-portal-backend` (exclude `node_modules`), use **Advanced Tools (Kudu)** or CLI zip deploy below  
4. Set environment variables: `MAIL_SERVER`, `MAIL_PORT`, `CONTACT_US_FORM_EMAIL_ADDRESS`, etc. — see [06-local-development.md](06-local-development.md) Step 4  

### Detailed — CLI

```bash
cd tdis-portal-backend
npm install --production
zip -r deploy.zip . -x "node_modules/*" ".git/*"
az webapp create --resource-group $RG --plan $PLAN --name $EMAIL_APP --runtime "NODE:22-lts"
az webapp config set --resource-group $RG --name $EMAIL_APP --startup-file "node server.js"
az webapp deployment source config-zip --resource-group $RG --name $EMAIL_APP --src deploy.zip
```

**Windows PowerShell zip:** `Compress-Archive -Path * -DestinationPath deploy.zip` (exclude node_modules manually).

### STOP — verify

`https://app-bit-portal-email.azurewebsites.net/health` → OK

---

## Phase 9 — Final smoke test

| Check | URL / command | Pass? |
|-------|---------------|-------|
| API alive | `curl https://YOUR-API.azurewebsites.net/api/v1/health/liveness` | [ ] |
| Tiles alive | `https://YOUR-TILE.azurewebsites.net/health` | [ ] |
| Frontend loads | Static Web App URL | [ ] |
| Map renders | Visual check | [ ] |
| Widgets empty | Expected until Databricks sync | [ ] |

---

## Databricks (your work — not this chapter)

Bronze/Silver/Gold pipelines populate PostgreSQL. See [14-trd-reference.md](14-trd-reference.md).

---

## Production networking (later)

VNet, subnets, Application Gateway — [08c-azure-tier2-reference.md](08c-azure-tier2-reference.md)

---

## Troubleshooting

[12-troubleshooting-faq.md](12-troubleshooting-faq.md)
