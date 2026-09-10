# 05 — Get the Code

## Code is sent separately

This instruction package does **not** include source code. TDIS provides these repos **in a separate delivery** (archive or git access):

| Repository | Branch |
|------------|--------|
| `tdis-portal-api` | `develop` |
| `tdis-portal-frontend` | `development` |
| `tdis-portal-db` | `development` |
| `tdis-portal-backend` | `development` |

Pinned commit SHAs: see `CODE_SNAPSHOTS.md` in the manual package root.

## Recommended folder layout

Place all repos in one workspace next to this manual:

```
your-workspace/
├── instruction_manual/          # this documentation
├── CODE_SNAPSHOTS.md            # at package root (sibling to instruction_manual)
├── tdis-portal-api/
├── tdis-portal-frontend/
├── tdis-portal-db/
└── tdis-portal-backend/
```

Paths in this manual assume you `cd` into each repo from that workspace.

## tdis-portal-db note

Use branch **`development`** — `main` is empty. Migrations are also consolidated in [schema/portal_schema.sql](schema/portal_schema.sql).

## What you do not have

- TDIS Azure subscription or Key Vault  
- Production data  
- TDIS infrastructure (Terragrunt) repos  

## Next step

Load schema: [07-postgresql-schema.md](07-postgresql-schema.md), then [06-local-development.md](06-local-development.md).
