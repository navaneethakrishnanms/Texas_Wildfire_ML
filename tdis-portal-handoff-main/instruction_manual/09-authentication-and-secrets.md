# 09 — Authentication and Secrets

## Public portal — no user login

TRD §3.1: the portal is **publicly accessible**. You do **not** need Keycloak or user accounts for the handoff.

Ignore frontend Keycloak code and `VITE_KEYCLOAK_*` env vars.

## API key authentication

The FastAPI backend uses **`X-API-Key`** header authentication.

| Setting | Local default | Azure |
|---------|---------------|-------|
| Header name | `X-API-Key` | Same |
| Dev master key | `tdis_dev_master_key_2024_secure_token_123456789` | Same key when `ENVIRONMENT=development` |
| Backend env | `SECRET_KEY`, `API_KEY_HEADER` | Key Vault or plain settings |

> **Azure Tier 1:** Set `ENVIRONMENT=development` on App Service so the dev master key works. The current code snapshot does **not** load API keys when `ENVIRONMENT=production`. TDIS will address production key config in a future snapshot.

Frontend sends the key as `VITE_EVENTS_API_KEY` (or `VITE_PORTAL_API_KEY`).

Example:

```bash
curl -H "X-API-Key: tdis_dev_master_key_2024_secure_token_123456789" \
  http://localhost:8000/api/v1/events/
```

## Environment variable matrix

### FastAPI (`tdis-portal-api`)

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_HOST` | Yes | PostgreSQL host |
| `DATABASE_PORT` | Yes | Usually `5432` |
| `DATABASE_USER` | Yes | DB username |
| `DATABASE_PASSWORD` | Yes | DB password |
| `DATABASE_NAME` | Yes | `tdis_portal` |
| `SECRET_KEY` | Yes | App signing secret |
| `API_KEY_HEADER` | No | Default `X-API-Key` |
| `BACKEND_CORS_ORIGINS` | Yes (Azure) | Frontend URL(s), comma-separated |
| `ENVIRONMENT` | Yes | `development` / `production` |
| `DEBUG` | No | `true` locally |
| ~~`DATABRICKS_*`~~ | **No** | AI assistant — excluded |

### Frontend (`tdis-portal-frontend`) — build time

Vite bakes these at **build** time. Rebuild after changing.

| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_MAPBOX_TOKEN` | Yes | Your Mapbox public token (`pk.…`) |
| `VITE_API_URL` | Yes | e.g. `http://127.0.0.1:8000/api/v1` |
| `VITE_EVENTS_API_KEY` | Yes | Same as API key |
| `VITE_TILE_SERVER_BASE_URL` | Recommended | pg_tileserv URL |
| `VITE_EMAIL_API_URL` | Optional | Email backend base URL |
| ~~`VITE_CUBE_*`~~ | **No** | Deprecated |
| ~~`VITE_AI_*`~~ | **No** | Excluded |
| ~~`VITE_KEYCLOAK_*`~~ | **No** | Skip |

### Email backend (`tdis-portal-backend`)

| Variable | Required |
|----------|----------|
| `NODE_ENV` | Yes |
| `MAIL_SERVER`, `MAIL_PORT`, `MAIL_USERNAME`, `MAIL_PASSWORD` | Yes for sending |
| `CONTACT_US_FORM_EMAIL_ADDRESS` | Yes |
| `SUBSCRIBE_USER_REQUEST_EMAIL_ADDRESS` | Yes |
| `MAIL_DISPLAY_NAME`, `MAIL_REPLY_TO` | Recommended |

### pg_tileserv (Docker)

| Variable | Required |
|----------|----------|
| `DATABASE_URL` | Yes — PostgreSQL connection string |

## Azure secrets — two ways (ch. 08)

### Easy path (dev / beginners)

Paste values **directly** in App Service → **Environment variables** / **Application settings**:

- `DATABASE_HOST`, `DATABASE_PASSWORD`, `SECRET_KEY`, etc.

No Key Vault required. Fine for dev. **Do not commit these values to git.**

### Detailed path (TDIS production style)

Store secrets in **Key Vault**. Reference in App Service:

```
@Microsoft.KeyVault(VaultName=my-vault;SecretName=database-password)
```

Enable **managed identity** on App Service and grant Key Vault **Get** on secrets:

```bash
az keyvault set-policy --name YOUR_KV --object-id YOUR_APP_PRINCIPAL_ID --secret-permissions get list
```

Full steps: [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md) Phase 3 and Phase 5 Option B2.

## Security reminders

- Never commit `.env` to git  
- Never put secrets in frontend public repos — API key in Vite is visible in browser (acceptable for this portal's public API model; use rotation in production)  
- Use HTTPS on Azure (App Service / SWA default)
