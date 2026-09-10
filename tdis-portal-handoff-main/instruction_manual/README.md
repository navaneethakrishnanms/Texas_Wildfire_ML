# TDIS Portal — Instruction Manual

Welcome. This manual is written for **beginners** — including people who have never used Azure.

**Start here:** [00-start-here.md](00-start-here.md)

Code repos are **sent separately** — [REPOS.md](../REPOS.md).

---

## Read first (first meeting)

1. [00-start-here.md](00-start-here.md) — terminal basics, two Azure paths  
2. [02-architecture.md](02-architecture.md) — TDIS vs your Tier 1 vs local diagrams  
3. [14-trd-reference.md](14-trd-reference.md) — plain-English TRD summary  

---

## Track A — Run locally (do this first)

1. [03-prerequisites.md](03-prerequisites.md) — install tools, Mapbox account  
2. [05-get-the-code.md](05-get-the-code.md) — folder layout  
3. [07-postgresql-schema.md](07-postgresql-schema.md) — load empty schema  
4. [06-local-development.md](06-local-development.md) — docker-compose + services  

**Success:** http://localhost:5173 — map loads.

Uses [docker-compose.local.yml](docker-compose.local.yml) for Postgres + pg_tileserv.

---

## Track B — Deploy to Azure (Tier 1)

After Track A works:

1. [04-azure-for-beginners.md](04-azure-for-beginners.md) — Azure account, CLI, costs  
2. [08b-azure-resource-checklist.md](08b-azure-resource-checklist.md) — what to create  
3. [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md) — **Easy** (Portal) + **Detailed** (CLI / Key Vault)  

**Success:** Static Web App URL — map loads.

**Production networking (VNet, App Gateway):** reference only — [08c-azure-tier2-reference.md](08c-azure-tier2-reference.md). For your network team, not day one.

---

## Package contents

| Item | Location |
|------|----------|
| This manual | `instruction_manual/` |
| TRD PDF | `reference/TDIS_Portal_TRD-v3.pdf` |
| PostgreSQL schema (Source A) | `schema/portal_schema.sql` |
| Code snapshot SHAs | `../CODE_SNAPSHOTS.md` |

---

## Table of contents

| # | Chapter | Track |
|---|---------|-------|
| 00 | [Start here](00-start-here.md) | Both |
| 01 | [What is this?](01-what-is-this.md) | Both |
| 02 | [Architecture](02-architecture.md) | Both |
| 03 | [Prerequisites](03-prerequisites.md) | A |
| 04 | [Azure for beginners](04-azure-for-beginners.md) | B |
| 05 | [Get the code](05-get-the-code.md) | A |
| 06 | [Local development](06-local-development.md) | A |
| 07 | [PostgreSQL schema](07-postgresql-schema.md) | Both |
| 08b | [Azure resource checklist](08b-azure-resource-checklist.md) | B |
| 08 | [Deploy to Azure (Tier 1)](08-deploy-to-azure-by-hand.md) | B |
| 08c | [Azure Tier 2 reference](08c-azure-tier2-reference.md) | B (later) |
| 09 | [Authentication and secrets](09-authentication-and-secrets.md) | Both |
| 10 | [External dependencies](10-external-dependencies.md) | Both |
| 11 | [CI/CD overview (TDIS reference)](11-cicd-overview.md) | Info |
| 12 | [Troubleshooting FAQ](12-troubleshooting-faq.md) | Both |
| 13 | [Glossary](13-glossary.md) | Both |
| 14 | [TRD reference summary](14-trd-reference.md) | Both |

---

## Explicitly excluded (ignore)

- TDIS AI, Databricks assistant, MCP  
- Cube.js (`VITE_CUBE_*`)  
- Keycloak / user login  
- Terragrunt / TDIS infrastructure repos  

---

## Local ↔ Azure quick map

| Component | Local | Azure Tier 1 |
|-----------|-------|--------------|
| PostgreSQL | Docker `:5432` | Flexible Server + firewall |
| API | `:8000` | App Service (container) |
| Frontend | `:5173` | Static Web App |
| pg_tileserv | Docker `:7800` | App Service (same image) |
| Email | `:8080` | App Service (Node) |
| Secrets | `.env` files | App settings or Key Vault |

Need help? [12-troubleshooting-faq.md](12-troubleshooting-faq.md)
