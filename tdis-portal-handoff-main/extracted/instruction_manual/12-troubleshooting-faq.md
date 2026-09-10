# 12 — Troubleshooting FAQ

## Map is blank

- Check `VITE_MAPBOX_TOKEN` is set and starts with `pk.`  
- Rebuild frontend after changing Vite env vars (`npm run dev` picks up `.env` on restart)  
- Browser console for Mapbox errors  

## CORS errors in browser

- Add your frontend origin to API `BACKEND_CORS_ORIGINS`  
- Local: `http://localhost:5173,http://127.0.0.1:5173`  
- Azure: your Static Web App URL  

## API returns 401 / Invalid API key

- Send header `X-API-Key: <key>`  
- Match `VITE_EVENTS_API_KEY` in frontend to API's accepted key  
- Local dev key: `tdis_dev_master_key_2024_secure_token_123456789`  

## Database connection failed (API)

**Local:** Postgres running? Schema loaded? Credentials match `.env`?

**Azure:** SSL required — API uses `ssl=require`. Firewall allows App Service outbound or use VNet integration.

Common error: connecting to Azure PG without `sslmode=require`.

## Empty dashboards but health check OK

**Expected** until Source B tables exist (Databricks sync). See [07-postgresql-schema.md](07-postgresql-schema.md).

## pg_tileserv cannot connect to Postgres

**Linux Docker:** replace `host.docker.internal` with host IP or run Postgres on same Docker network:

```bash
docker network create tdis
docker network connect tdis tdis-postgres
docker run ... --network tdis -e DATABASE_URL="postgresql://postgres:postgres@tdis-postgres:5432/tdis_portal" ...
```

## Email form fails

- Email backend running on `:8080`?  
- `VITE_EMAIL_API_URL` points to it?  
- Valid SMTP credentials in backend `.env`?  

## Frontend build fails on Azure

- Node 20+ required  
- Use `npm ci --legacy-peer-deps`  
- All `VITE_*` vars must be set **before** `npm run build`  

## `view_events_store_v2` does not exist

Created by Databricks sync (Source B), not bundled schema. API events endpoints need it — your pipeline must create it.

## Port already in use

```bash
lsof -i :8000   # or :5173, :5432, :8080, :7800
```

Stop conflicting process or change ports in config.
