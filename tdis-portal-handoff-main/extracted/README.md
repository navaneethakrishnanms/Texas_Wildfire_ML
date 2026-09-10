# TDIS Portal — BIT Instruction Manual

Documentation only. **Code repos are sent separately.**

## Start here

1. **[02-architecture.md](instruction_manual/02-architecture.md)** — deployment diagrams (how TDIS runs it + how you deploy it)  
2. **[06-local-development.md](instruction_manual/06-local-development.md)** — run on your laptop  
3. **[08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md)** — deploy to your Azure  

Full table of contents: [instruction_manual/README.md](instruction_manual/README.md)

---

## Repos TDIS will send (code — separate package)

| Repository | Branch | Purpose |
|------------|--------|---------|
| `tdis-portal-frontend` | `development` | React UI → **Static Web App** |
| `tdis-portal-api` | `develop` | FastAPI → **App Service container** |
| `tdis-portal-backend` | `development` | Email API → **App Service Node** |
| `tdis-portal-db` | `development` | SQL migrations (schema also in this manual) |

Commit SHAs: [CODE_SNAPSHOTS.md](CODE_SNAPSHOTS.md)

---

## This package contains

| Item | Path |
|------|------|
| Instruction manual | `instruction_manual/` |
| TRD PDF | `instruction_manual/reference/TDIS_Portal_TRD-v3.pdf` |
| PostgreSQL schema (empty) | `instruction_manual/schema/portal_schema.sql` |

---

## Archive

Extract: `tar -xzf tdis-portal-instructions.tar.gz`
