# TDIS Portal — Instruction Manual

Welcome. This manual explains **how TDIS deploys the portal** and **how you deploy the same stack** on your laptop and on **your Azure subscription**.

**Code repos are sent separately** — [REPOS.md](../REPOS.md).

## Read first

**[02-architecture.md](02-architecture.md)** — deployment diagrams (TDIS reference + your Azure + local)

| Item | Location |
|------|----------|
| This manual | `instruction_manual/` |
| TRD PDF | `reference/TDIS_Portal_TRD-v3.pdf` |
| PostgreSQL schema (Source A) | `schema/portal_schema.sql` |
| Code snapshot SHAs | `../CODE_SNAPSHOTS.md` (package root) |

**Source code is sent separately** by TDIS — see [05-get-the-code.md](05-get-the-code.md).

## Two tracks — pick your path

### Track A — Run locally (recommended first)

Prove the stack works on your laptop before spending money on Azure.

1. [03-prerequisites.md](03-prerequisites.md) — install tools, create Mapbox account  
2. [05-get-the-code.md](05-get-the-code.md) — unzip and folder layout  
3. [07-postgresql-schema.md](07-postgresql-schema.md) — load empty schema  
4. [06-local-development.md](06-local-development.md) — start all services  

**Success check:** open http://localhost:5173 — map loads (needs Mapbox token).

### Track B — Deploy to your Azure subscription

After Track A works (or in parallel if you prefer).

1. [04-azure-for-beginners.md](04-azure-for-beginners.md) — Azure account, CLI, portal tour  
2. [08b-azure-resource-checklist.md](08b-azure-resource-checklist.md) — what to create  
3. [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md) — step-by-step deploy  

### First meeting prep

1. [01-what-is-this.md](01-what-is-this.md) — scope  
2. [02-architecture.md](02-architecture.md) — diagrams  
3. [14-trd-reference.md](14-trd-reference.md) — plain-English TRD summary  

## Table of contents

| # | Chapter | Track |
|---|---------|-------|
| 01 | [What is this?](01-what-is-this.md) | Both |
| 02 | [Architecture](02-architecture.md) | Both |
| 03 | [Prerequisites](03-prerequisites.md) | A (+ Mapbox) |
| 04 | [Azure for beginners](04-azure-for-beginners.md) | B |
| 05 | [Get the code](05-get-the-code.md) | A |
| 06 | [Local development](06-local-development.md) | A |
| 07 | [PostgreSQL schema](07-postgresql-schema.md) | Both |
| 08b | [Azure resource checklist](08b-azure-resource-checklist.md) | B |
| 08 | [Deploy to Azure by hand](08-deploy-to-azure-by-hand.md) | B |
| 09 | [Authentication and secrets](09-authentication-and-secrets.md) | Both |
| 10 | [External dependencies](10-external-dependencies.md) | Both |
| 11 | [CI/CD overview (TDIS reference)](11-cicd-overview.md) | Info |
| 12 | [Troubleshooting FAQ](12-troubleshooting-faq.md) | Both |
| 13 | [Glossary](13-glossary.md) | Both |
| 14 | [TRD reference summary](14-trd-reference.md) | Both |

## Explicitly excluded (ignore)

- TDIS AI tab, Databricks AI assistant, MCP  
- Cube.js (`VITE_CUBE_*`) — replaced by FastAPI  
- Keycloak / user login — public portal needs no login (TRD §3.1)  
- Terragrunt / TDIS infrastructure repos  

## Local ↔ Azure quick map

| Component | Local | Azure |
|-----------|-------|-------|
| PostgreSQL | Docker `:5432` | Flexible Server |
| API | `:8000` | App Service (container) |
| Frontend | `:5173` | Static Web App |
| pg_tileserv | Docker `:7800` | App Service (same Docker image) |
| Email backend | `:8080` | App Service (Node) |
| Secrets | `.env` files | Key Vault |

## Need help?

See [12-troubleshooting-faq.md](12-troubleshooting-faq.md).
