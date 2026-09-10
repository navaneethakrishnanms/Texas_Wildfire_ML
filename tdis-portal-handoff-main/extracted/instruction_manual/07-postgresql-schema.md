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
| `view_events_store` | view (v1) |
| `view_hurricanes` | view |

Extensions: **PostGIS**, **pgcrypto**

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

> **Important:** The FastAPI layer reads `view_events_store_v2`, not `view_events_store`. Your sync job must provide v2 (or an equivalent view the API expects).

## Load Source A — local

### Option 1: Pre-built SQL file

```bash
# Start Postgres (see ch. 06) then:
psql -h localhost -U postgres -d tdis_portal -f instruction_manual/schema/portal_schema.sql
```

### Option 2: Run migrations manually

From `tdis-portal-db/migrations/`, run files in numeric order. **Skip** any file matching:

- `*seed_data*`
- `*test_data*`
- `truncate_*`
- `sample_*`
- `cleanup_sample_*`

```bash
for f in $(ls migrations/*.sql migrations/041_modify_event_date 2>/dev/null | sort -V); do
  case "$(basename $f)" in *seed_data*|*test_data*|truncate_*|sample_*|cleanup_sample_*) continue ;; esac
  psql -h localhost -U postgres -d tdis_portal -f "$f"
done
```

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

## Reference data (counties)

Source A migrations do **not** include county seed data in the bundled schema file. For local map testing you may need to run county seed migrations from `tdis-portal-db` separately, or load boundaries via your own pipeline.

County seed files (optional, contains data): `008_counties_table_seed_data.sql`, `028_cities_table_seed_data*.sql`
