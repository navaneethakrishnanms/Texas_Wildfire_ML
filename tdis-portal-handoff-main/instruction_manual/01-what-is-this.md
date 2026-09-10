# 01 — What Is This?

## Purpose

The **TDIS Portal** is a public web application that shows **Active Weather Events in Texas** — wildfires, hurricanes, earthquakes, power outages, rainfall, and related hazard data on a Mapbox map.

This handoff gives you **documentation and schema SQL**. TDIS sends **portal code separately** (see `CODE_SNAPSHOTS.md`).

> **New to terminals or Azure?** Start with [00-start-here.md](00-start-here.md).

1. **Portal code** (separate delivery) — frontend + API + email backend
2. **This manual** — architecture + local + Azure instructions (Easy + Detailed paths)
3. **Core PostgreSQL schema** (empty tables — no production data)
4. **Greenfield Azure setup** on **your own subscription** — **Tier 1 first** (no VNet). Full TDIS networking: [08c-azure-tier2-reference.md](08c-azure-tier2-reference.md)

## What BIT builds (your work)

Per the [TRD](reference/TDIS_Portal_TRD-v3.pdf), you are building the **Databricks ingestion layers**:

```
External providers → Bronze → Silver → Gold → PostgreSQL sync
```

Your pipelines populate PostgreSQL. The portal code **reads** that database and serves the UI.

## What TDIS gives you (this package)

| You receive | You do not receive |
|-------------|-------------------|
| React frontend | TDIS Azure infrastructure repos |
| FastAPI backend | Production data |
| Core DB schema (Source A) | TDIS AI / Cube.js |
| Email backend (optional SMTP) | Keycloak setup |
| TRD + this manual | TDIS secrets or hostnames |

## After you finish

TDIS will send an **updated code snapshot** so you can retrofit your ingestion work into the latest portal codebase.

## Who should read what

| Role | Start with |
|------|------------|
| Beginner (any) | [00-start-here.md](00-start-here.md) |
| Developer (local) | 03 → 05 → 07 → 06 |
| DevOps (Azure) | 04 → 08b → 08 (Easy path first) |
| Network / security lead | 08c (Tier 2 reference) |
| Architect / lead | 02 → 14 → TRD PDF |
| Data engineer | 14 → 07 (Source B tables) |
