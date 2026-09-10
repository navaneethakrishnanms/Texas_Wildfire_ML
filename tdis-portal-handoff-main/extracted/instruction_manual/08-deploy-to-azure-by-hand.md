# 08 — Deploy to Azure by Hand (Track B)

Deploy the same stack as [06-local-development.md](06-local-development.md) to **your Azure subscription**. Requires [04-azure-for-beginners.md](04-azure-for-beginners.md) and [08b-azure-resource-checklist.md](08b-azure-resource-checklist.md).

Set variables once (edit for your environment):

```bash
RG=rg-bit-portal-dev
LOC=southcentralus
PG_SERVER=pg-bit-portal-dev
KV_NAME=kv-bit-portal-dev
ACR_NAME=acrbitportaldev   # must be globally unique, alphanumeric only
PLAN=plan-bit-portal-dev
API_APP=app-bit-portal-api
SWA_NAME=swa-bit-portal
TILE_APP=app-bit-pgtile
EMAIL_APP=app-bit-portal-email
PG_ADMIN=portaladmin
PG_PASS='ChangeMe-StrongPass123!'
```

## 1. Resource group

```bash
az group create --name $RG --location $LOC
```

## 2. PostgreSQL Flexible Server

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
```

Load schema from your machine:

```bash
psql "host=${PG_SERVER}.postgres.database.azure.com port=5432 dbname=tdis_portal user=${PG_ADMIN} sslmode=require" \
  -f instruction_manual/schema/portal_schema.sql
```

Allow your IP if connection fails:

```bash
MYIP=$(curl -s ifconfig.me)
az postgres flexible-server firewall-rule create \
  --resource-group $RG --name $PG_SERVER \
  --rule-name AllowMyIP --start-ip-address $MYIP --end-ip-address $MYIP
```

## 3. Key Vault

```bash
az keyvault create --name $KV_NAME --resource-group $RG --location $LOC

az keyvault secret set --vault-name $KV_NAME --name database-host \
  --value "${PG_SERVER}.postgres.database.azure.com"
az keyvault secret set --vault-name $KV_NAME --name database-user --value "$PG_ADMIN"
az keyvault secret set --vault-name $KV_NAME --name database-password --value "$PG_PASS"
az keyvault secret set --vault-name $KV_NAME --name database-name --value "tdis_portal"
az keyvault secret set --vault-name $KV_NAME --name api-secret-key --value "$(openssl rand -hex 32)"
```

## 4. Container Registry + API image

```bash
az acr create --resource-group $RG --name $ACR_NAME --sku Basic --admin-enabled true

cd tdis-portal-api
az acr build --registry $ACR_NAME --image portal-api:latest -f docker/Dockerfile .
```

## 5. App Service Plan + API

```bash
az appservice plan create --name $PLAN --resource-group $RG --sku B1 --is-linux

az webapp create --resource-group $RG --plan $PLAN --name $API_APP \
  --deployment-container-image-name ${ACR_NAME}.azurecr.io/portal-api:latest

ACR_ID=$(az acr show --name $ACR_NAME --query id -o tsv)
az role assignment create --assignee $(az webapp identity assign --resource-group $RG --name $API_APP --query principalId -o tsv) \
  --role AcrPull --scope $ACR_ID

# App settings — use Key Vault references in production
az webapp config appsettings set --resource-group $RG --name $API_APP --settings \
  ENVIRONMENT=production \
  DEBUG=false \
  DATABASE_HOST="@Microsoft.KeyVault(VaultName=${KV_NAME};SecretName=database-host)" \
  DATABASE_USER="@Microsoft.KeyVault(VaultName=${KV_NAME};SecretName=database-user)" \
  DATABASE_PASSWORD="@Microsoft.KeyVault(VaultName=${KV_NAME};SecretName=database-password)" \
  DATABASE_NAME="@Microsoft.KeyVault(VaultName=${KV_NAME};SecretName=database-name)" \
  SECRET_KEY="@Microsoft.KeyVault(VaultName=${KV_NAME};SecretName=api-secret-key)" \
  BACKEND_CORS_ORIGINS="https://${SWA_NAME}.azurestaticapps.net,http://localhost:5173"

az webapp config set --resource-group $RG --name $API_APP \
  --generic-configurations '{"healthCheckPath": "/api/v1/health/liveness"}'
```

Grant the App Service managed identity **Get** permission on Key Vault secrets.

API URL: `https://${API_APP}.azurewebsites.net/api/v1`

## 6. Static Web App (frontend)

```bash
az staticwebapp create --name $SWA_NAME --resource-group $RG --location centralus
```

Build frontend locally with **production** env vars:

```bash
cd tdis-portal-frontend
export VITE_MAPBOX_TOKEN=pk.your-mapbox-token
export VITE_API_URL=https://${API_APP}.azurewebsites.net/api/v1
export VITE_EVENTS_API_KEY=your-production-api-key
export VITE_TILE_SERVER_BASE_URL=https://${TILE_APP}.azurewebsites.net
export VITE_EMAIL_API_URL=https://${EMAIL_APP}.azurewebsites.net

npm ci --legacy-peer-deps
npm run build
```

Deploy (install SWA CLI if needed: `npm i -g @azure/static-web-apps-cli`):

```bash
DEPLOY_TOKEN=$(az staticwebapp secrets list --name $SWA_NAME --resource-group $RG --query properties.apiKey -o tsv)
swa deploy ./dist --deployment-token $DEPLOY_TOKEN
```

## 7. pg_tileserv (Docker on App Service)

```bash
az webapp create --resource-group $RG --plan $PLAN --name $TILE_APP \
  --deployment-container-image-name pramsey/pg_tileserv:20240614

az webapp config appsettings set --resource-group $RG --name $TILE_APP --settings \
  WEBSITES_PORT=7800 \
  PORT=7800 \
  DATABASE_URL="postgresql://${PG_ADMIN}:${PG_PASS}@${PG_SERVER}.postgres.database.azure.com:5432/tdis_portal?sslmode=require"
```

Prefer storing `DATABASE_URL` in Key Vault for production.

## 8. Email backend (Node App Service)

```bash
cd tdis-portal-backend
az webapp create --resource-group $RG --plan $PLAN --name $EMAIL_APP --runtime "NODE:20-lts"

# Deploy code (zip deploy example)
zip -r deploy.zip . -x "node_modules/*" ".git/*"
az webapp deployment source config-zip --resource-group $RG --name $EMAIL_APP --src deploy.zip

az webapp config appsettings set --resource-group $RG --name $EMAIL_APP --settings \
  NODE_ENV=production \
  CONTACT_US_FORM_EMAIL_ADDRESS=you@example.com \
  MAIL_SERVER=smtp.your-org.com \
  MAIL_PORT=587 \
  MAIL_USERNAME=... \
  MAIL_PASSWORD=... \
  MAIL_DISPLAY_NAME="TDIS Portal" \
  MAIL_REPLY_TO=you@example.com \
  SUBSCRIBE_USER_REQUEST_EMAIL_ADDRESS=subscribe@example.com
```

## 9. Smoke test

```bash
curl -s "https://${API_APP}.azurewebsites.net/api/v1/health/liveness"
curl -s -H "X-API-Key: YOUR_KEY" "https://${API_APP}.azurewebsites.net/api/v1/health/detailed"
```

Open Static Web App URL in browser — map should load.

## Databricks (BIT)

Your Bronze/Silver/Gold pipelines and PostgreSQL sync jobs run in **your** Databricks + ADLS — not deployed by this portal code. See [14-trd-reference.md](14-trd-reference.md).

## Troubleshooting

[12-troubleshooting-faq.md](12-troubleshooting-faq.md)
