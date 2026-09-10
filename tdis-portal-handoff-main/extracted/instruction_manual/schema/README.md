# PostgreSQL Schema — Source A

## Database and schema

- **Database name:** `tdis_portal` (create this empty database first)
- **PostgreSQL schema:** `public`

## File: `portal_schema.sql`

Consolidated from `tdis-portal-db` migrations on branch `development`.

**Included:** table DDL, views, indexes, extensions (PostGIS, pgcrypto)  
**Excluded:** all `*seed_data*`, `*test_data*`, `truncate_*`, and `sample_*` migration files (no reference data)

## Load locally

```bash
createdb tdis_portal   # or use Docker — see ch. 06
psql -h localhost -U postgres -d tdis_portal -f portal_schema.sql
```

## Load on Azure PostgreSQL

```bash
psql "host=<your-server>.postgres.database.azure.com port=5432 dbname=tdis_portal user=<admin> sslmode=require" \
  -f portal_schema.sql
```

## Source B tables (not in this file)

Objects populated by **your Databricks sync jobs** are listed in [07-postgresql-schema.md](../07-postgresql-schema.md). Until those pipelines run, related API endpoints return empty results — that is expected.
