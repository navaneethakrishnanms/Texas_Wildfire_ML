# 12 — Troubleshooting FAQ

Plain fixes for common problems. Start with the **Symptom** that matches yours.

---

## Local development

### Map is blank / gray

- Check `VITE_MAPBOX_TOKEN` in `tdis-portal-frontend/.env` — must start with `pk.`  
- Restart frontend after changing `.env`: `Ctrl+C`, then `npm run dev` again  
- Open browser DevTools (F12) → Console — look for Mapbox errors  

### CORS errors in browser (red text in Console)

- API must allow your frontend origin in `BACKEND_CORS_ORIGINS`  
- Local: add `http://localhost:5173,http://127.0.0.1:5173` to API `.env`  
- Restart API after changing `.env`  

### API returns 401 / Invalid API key (Azure)

- Tier 1 uses `ENVIRONMENT=development` so the dev master key works — see ch. 08 Phase 5  
- If you set `ENVIRONMENT=production`, **no API keys are loaded** in the current code snapshot — widgets will 401  
- Frontend `VITE_EVENTS_API_KEY` must match: `tdis_dev_master_key_2024_secure_token_123456789` for Tier 1  

### API returns 401 / Invalid API key (local)

- Frontend must send `X-API-Key` header — set `VITE_EVENTS_API_KEY` in frontend `.env`  
- Local dev key: `tdis_dev_master_key_2024_secure_token_123456789`  

### `extension "postgis" is not available` (schema load)

- Use PostGIS Docker image: [docker-compose.local.yml](docker-compose.local.yml) uses `postgis/postgis:15-3.4-alpine`  
- Do not use plain `postgres:15-alpine`  

### Database connection failed (API local)

- Is Docker Postgres running? `docker ps` should show `tdis-postgres`  
- Did you load schema? See [06-local-development.md](06-local-development.md) Step 1  
- Check API `.env`: `DATABASE_HOST=localhost`, password `postgres`  

### `psql: command not found`

- Install client: `sudo apt install postgresql-client` (Linux) or use full PostgreSQL installer  
- Or run psql inside Docker:  
  `docker exec -it tdis-postgres psql -U postgres -d tdis_portal`

### pg_tileserv cannot connect to Postgres

- Use docker-compose (recommended) — [docker-compose.local.yml](docker-compose.local.yml) links containers automatically  
- If using separate `docker run`, see Advanced section in ch. 06  

### Port already in use

```bash
# Linux/Mac — find what is using the port
lsof -i :8000   # or :5173, :5432, :8080, :7800
```

Stop the other program or restart Docker.

### `docker: command not found`

- Install Docker Desktop — [03-prerequisites.md](03-prerequisites.md)  

---

## Azure deployment

### `az: command not found`

- Install Azure CLI — [04-azure-for-beginners.md](04-azure-for-beginners.md)  

### PostgreSQL connection timeout from laptop

- Portal → PostgreSQL server → **Networking** → **Add current client IP** → Save  
- Retry `psql` with `sslmode=require`  

### PostgreSQL connection failed from App Service

- Enable **Allow Azure services** firewall rule (see ch. 08 Phase 2)  
- Check `DATABASE_HOST` ends with `.postgres.database.azure.com`  
- Password must match exactly — no extra spaces  

### Key Vault value shows `@Microsoft.KeyVault(...)` literally in app settings

- New vaults use **RBAC** — use role `Key Vault Secrets User`, not `set-policy` only  
- See ch. 08 Phase 5 Option B2  
- Or use **Easy path**: paste plain values in App Service settings  

### CORS errors after Azure frontend deploy

- SWA hostname is **auto-generated** (e.g. `nice-field-0123456789.2.azurestaticapps.net`) — NOT the resource name  
- Copy URL from Portal → Static Web App → add to API `BACKEND_CORS_ORIGINS`, restart API  

### App Service shows 503 / "Application Error"

- Wait 2–3 minutes after first deploy  
- App Service → **Log stream** — read the error  
- **ImagePullBackOff** → AcrPull role + `acrUseManagedIdentityCreds: true` (ch. 08 Phase 5), wait 2–5 min for IAM  
- DB connection error → PostgreSQL firewall (Allow Azure services)  

### Frontend works locally but broken in Azure

- All `VITE_*` variables must be set **before** `npm run build` — they are baked into the build  
- Rebuild and redeploy after changing any `VITE_*` value  
- Windows: use `$env:VITE_MAPBOX_TOKEN="..."` in PowerShell before build  

### Map works in Azure but API calls fail (CORS)

- SWA URL must match **defaultHostname** from Portal (not the resource name)  
- Add exact URL to API `BACKEND_CORS_ORIGINS`, save, wait for restart  

### ACR name rejected / already exists

- ACR names are **globally unique** — pick a new name like `acrbitportaldev002`  

### Static Web App deploy fails

- Install SWA CLI: `npm i -g @azure/static-web-apps-cli`  
- Copy fresh deployment token from Portal → Static Web App → Manage deployment token  

### Empty dashboards but health check OK

- **Expected** until Source B tables exist (Databricks sync) — [07-postgresql-schema.md](07-postgresql-schema.md)  

### `view_events_store_v2` does not exist

- Created by your Databricks sync — not in bundled schema  

---

## Email backend

### Contact Us form fails

- Email backend running? Check `https://YOUR-EMAIL-APP.azurewebsites.net/health`  
- `VITE_EMAIL_API_URL` in frontend points to correct URL (set before build)  
- Valid SMTP in backend app settings — or skip email testing  

---

## Still stuck?

1. Re-read the **STOP — verify** section for the phase you are on — [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md)  
2. Ask your Azure admin if you lack **Contributor** access on the subscription  
3. Contact **your TDIS point of contact** with: phase number, exact error text, screenshot of Log stream  

> TDIS will provide a named contact at handoff — replace this line with their email when known.
