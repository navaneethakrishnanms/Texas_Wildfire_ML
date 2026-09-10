# 07 — PostgreSQL Schema

Database: **`tdis_portal`** | Schema: **`public`**

## Two sources of tables

### Source A — bundled with handoff (you load this)

Core reference model from `tdis-portal-db` migrations. File: [schema/portal_schema.sql](schema/portal_schema.sql)

| Object | Type |
|--------|------|
| `event_types` | table |
| `data_sources` | table |
| `counties` | table |
| `cities` | table |
| `batch_jobs` | table |
| `events_store` | table |
| `view_events_store` | view (v1 — bundled) |
| `view_hurricanes` | view |
| `view_batch_jobs_latest` | view |

Extensions: **PostGIS**, **pgcrypto** (requires PostGIS Docker image — see [docker-compose.local.yml](docker-compose.local.yml))

### Source B — your Databricks sync (you create and populate)

These objects are **not** in the handoff SQL. Your ingestion pipelines must create and fill them per TRD §2.2.3.

| Object | Type | API endpoints |
|--------|------|---------------|
| `view_events_store_v2` | view | `/events`, `/events/analytics` |
| `power_outage_county` | table | `/power-outage/*` |
| `power_outage_city` | table | `/power-outage/*` |
| `power_outage_utility` | table | `/power-outage/*` |
| `power_outage_by_zip` | table | `/power-outage/*` |
| `lts_county` | table | `/lts-flash-flood/*` |
| `crowdsource_waze_road_status_county` | table | `/weather-hazard/*` |
| `google_flood_hub_gauges` | table | `/flood-hub/gauges` |
| `baron_nationwide_rainfall_accumulation_1hr_tile` | table | `/baron/*` |

Until Source B exists, those endpoints return **empty results**. Health checks still pass.

> **Important:** The FastAPI layer reads **`view_events_store_v2`**, not `view_events_store`. Source B — your Databricks sync must **create this view** (DDL not in handoff). Typical pattern: sync writes **`events_store`**, view v2 reads from it. TDIS will provide sync DDL in a follow-up package or TDDL TRD — ask at kickoff.

### Source B scope vs TRD (first meeting)

| TRD sync job (ch. 14) | PostgreSQL objects |
|------------------------|-------------------|
| Events Sync | `view_events_store_v2`, related event tables |
| HSI Gold Sync | `baron_nationwide_rainfall_accumulation_1hr_tile`, etc. |
| Power Outage Sync | `power_outage_*` tables |

Additional objects in API (confirm in-scope with TDIS): `lts_county`, `crowdsource_waze_road_status_county`, `google_flood_hub_gauges`.

## Load Source A — when to run SQL

**Local:** load schema once in [06-local-development.md](06-local-development.md) Step 1 (after Docker is healthy).

**Azure:** load in [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md) Phase 2.

This chapter explains **what** is in the schema; those chapters explain **when** to run it.

### Option 1: Pre-built SQL file (recommended)

```bash
# From workspace root, after Postgres is running:
psql -h localhost -U postgres -d tdis_portal -f instruction_manual/schema/portal_schema.sql
```

If you see `extension "postgis" is not available`, your Postgres image is wrong — use `docker-compose.local.yml` (PostGIS image), not plain `postgres:15-alpine`.

### Option 2: Run migrations manually (advanced — maintainers only)

From `tdis-portal-db/migrations/`, run files in numeric order. **Skip** seed/test/truncate files. Requires bash and experience — **not recommended for first-time setup.**

## Load Source A — Azure

After creating PostgreSQL Flexible Server and database `tdis_portal`:

```bash
psql "host=myserver.postgres.database.azure.com port=5432 dbname=tdis_portal user=myadmin sslmode=require" \
  -f instruction_manual/schema/portal_schema.sql
```

Azure PostgreSQL requires **SSL** — the API uses `ssl=require` in production.

## Verify

```bash
psql -h localhost -U postgres -d tdis_portal -c "\dt public.*"
psql -h localhost -U postgres -d tdis_portal -c "\dv public.*"
```

You should see Source A tables/views. Source B objects will appear after your Databricks sync runs.

## Reference data (required for sync — optional for empty portal UI)

Source A creates **empty** lookup tables. Your Databricks sync may need rows in:

| Table | Purpose | Seed files in `tdis-portal-db` (optional, contain data) |
|-------|---------|-----------------------------------------------------------|
| `event_types` | Hazard type FK on `events_store` | `031_reset_and_update_event_types.sql` |
| `data_sources` | Provider FK | check migrations for `data_sources` seeds |
| `counties` | Texas county boundaries / FK | `008_counties_table_seed_data.sql` |
| `batch_jobs` | Job tracking | may need stub rows for sync |

For **local map testing** only, county seed is often enough. For **sync development**, ask TDIS which seed files to run.

County/city seed files (optional): `008_counties_table_seed_data.sql`, `028_cities_table_seed_data*.sql`
