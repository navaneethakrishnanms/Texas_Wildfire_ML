# TDIS Portal — BIT Instruction Manual

Documentation only. **Code repos are sent separately.** Written for **beginners**.

## Start here

**[instruction_manual/00-start-here.md](instruction_manual/00-start-here.md)** — terminal basics, local vs Azure paths

Then:

1. **[02-architecture.md](instruction_manual/02-architecture.md)** — TDIS vs Tier 1 vs local diagrams  
2. **[06-local-development.md](instruction_manual/06-local-development.md)** — run on your laptop  
3. **[08-deploy-to-azure-by-hand.md](instruction_manual/08-deploy-to-azure-by-hand.md)** — Azure (Easy + Detailed paths)  

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
| Local docker-compose | `instruction_manual/docker-compose.local.yml` |

---

## Azure tiers

| Tier | Chapter | Who |
|------|---------|-----|
| **Tier 1** (start here) | ch. 08 Easy or Detailed | Everyone — get portal working |
| **Tier 2** (reference) | ch. 08c | Network team — VNet, App Gateway |
