# 13 — Glossary

| Term | Meaning |
|------|---------|
| **ADLS Gen2** | Azure Data Lake Storage — where Databricks stores bronze/silver/gold files |
| **App Service** | Azure service that runs web apps and Docker containers |
| **ACR** | Azure Container Registry — stores Docker images |
| **API key** | Shared secret sent as `X-API-Key` header to the FastAPI backend |
| **Bronze layer** | Raw provider data landing zone (BIT builds) |
| **Databricks** | Platform where BIT runs ingestion/sync jobs |
| **FastAPI** | Python web framework used by `tdis-portal-api` |
| **Flexible Server** | Azure managed PostgreSQL product |
| **Gold layer** | Business-ready datasets like HSI (BIT builds) |
| **HSI** | Hazard Severity Index — rainfall severity by H3 hex |
| **H3** | Uber's hexagonal geospatial index |
| **Key Vault** | Azure secret storage |
| **Mapbox** | Map rendering service — requires public token |
| **Medallion architecture** | Bronze → Silver → Gold data pattern |
| **pg_tileserv** | Open-source server that serves map vector tiles from PostgreSQL |
| **Silver layer** | Normalized, Texas-filtered hazard events (BIT builds) |
| **Source A / B** | Source A = bundled schema; Source B = Databricks-sync tables |
| **Static Web App** | Azure hosting for static React builds |
| **SWA** | Abbreviation for Static Web App |
| **Sync job** | Databricks job that writes Delta tables into PostgreSQL |
| **TDIS** | Texas Disaster Information System |
| **TRD** | Technical Requirements Document — [reference/TDIS_Portal_TRD-v3.pdf](reference/TDIS_Portal_TRD-v3.pdf) |
| **Unity Catalog** | Databricks governance layer for tables |
| **Vite** | Frontend build tool — env vars must start with `VITE_` |
