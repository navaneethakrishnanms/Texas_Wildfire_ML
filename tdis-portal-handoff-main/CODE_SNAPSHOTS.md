# Code Snapshots — TDIS Portal Handoff

Frozen commit SHAs for the separate code delivery (**August 2026** handoff).

| Folder (use this name) | GitHub repository | Branch | Commit SHA |
|------------------------|-------------------|--------|------------|
| `tdis-portal-api/` | TexasDIS/tdis-portal-api | `develop` | `905a35cbd7e820519653de532e949dbe6bd4806d` |
| `tdis-portal-frontend/` | TexasDIS/event-detection-frontend | `development` | `f897dde34a028110648850cf7d4b28417e3e4eb3` |
| `tdis-portal-db/` | TexasDIS/tdis-portal-db | `development` | `5b85aa9a41e484a1ed4d411835ae343cd778d233` |
| `tdis-portal-backend/` | TexasDIS/tdis-portal-backend | `development` | `2fc412b98dbe3d03523abb5a98610593699876a0` |

> **Folder vs GitHub name:** Clone `event-detection-frontend` but keep local folder name `tdis-portal-frontend/` so paths in this manual work.

Schema SQL in this package: generated **2026-08-28** from `tdis-portal-db` `development` migrations (Source A only).

After BIT completes Databricks ingestion layers, TDIS will send an updated code snapshot for retrofit.

**Code delivery:** Ask your TDIS contact for git access or zip archives matching the SHAs above.
