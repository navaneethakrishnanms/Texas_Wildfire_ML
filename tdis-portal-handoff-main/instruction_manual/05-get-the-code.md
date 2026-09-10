# 05 — Get the Code

## Code is sent separately

This instruction package does **not** include source code. TDIS provides repos in a **separate delivery** (zip archive or git access). Ask your TDIS contact for delivery method and timeline.

Pinned commit SHAs: [CODE_SNAPSHOTS.md](../CODE_SNAPSHOTS.md)

| Repository | Branch | Folder name (must match) |
|------------|--------|--------------------------|
| `tdis-portal-api` | `develop` | `tdis-portal-api/` |
| `tdis-portal-frontend` | `development` | `tdis-portal-frontend/` |
| `tdis-portal-db` | `development` | `tdis-portal-db/` |
| `tdis-portal-backend` | `development` | `tdis-portal-backend/` |

> GitHub may use different repo names (see CODE_SNAPSHOTS.md). **Folder names above** are what this manual expects.

---

## If you received zip archives

1. Create one workspace folder, for example:
   - Windows: `C:\Users\You\tdis-workspace`
   - Mac/Linux: `~/tdis-workspace`
2. Unzip this **instruction manual** package so `instruction_manual/` and `CODE_SNAPSHOTS.md` are inside the workspace.
3. Unzip each code repo so folder names match **exactly** (see table above).
4. **Verify** — open a terminal, go to the workspace, and list folders:

```bash
cd ~/tdis-workspace          # or cd C:\Users\You\tdis-workspace
ls                           # Mac/Linux
dir                          # Windows CMD
```

You should see:

```
instruction_manual/
CODE_SNAPSHOTS.md
tdis-portal-api/
tdis-portal-frontend/
tdis-portal-db/
tdis-portal-backend/
```

If any folder is missing or nested wrong (e.g. `tdis-portal-api/tdis-portal-api/`), fix before continuing.

---

## If you received git access

From your workspace folder:

```bash
git clone <TDIS-provided-url>/tdis-portal-api.git
cd tdis-portal-api && git checkout develop && cd ..

git clone <TDIS-provided-url>/tdis-portal-frontend.git
cd tdis-portal-frontend && git checkout development && cd ..

git clone <TDIS-provided-url>/tdis-portal-db.git
cd tdis-portal-db && git checkout development && cd ..

# main branch is empty — always use development

git clone <TDIS-provided-url>/tdis-portal-backend.git
cd tdis-portal-backend && git checkout development && cd ..
```

Verify each repo matches the SHA in `CODE_SNAPSHOTS.md`:

```bash
cd tdis-portal-api && git rev-parse HEAD && cd ..
```

---

## Where to run commands

Most commands assume your terminal is **inside a repo folder** (`cd tdis-portal-api`) or **workspace root** (for schema load). Each chapter says which.

| Command starts with | You should be in |
|---------------------|------------------|
| `cd tdis-portal-api` | workspace root |
| `psql ... -f instruction_manual/schema/...` | workspace root |
| `cd instruction_manual` + docker compose | workspace, then `instruction_manual/` |

Use `pwd` (Mac/Linux) or `cd` (Windows) to check current folder if lost.

---

## tdis-portal-db note

Branch **`development`** only — `main` is empty. Schema is also bundled in [schema/portal_schema.sql](schema/portal_schema.sql) (Source A).

---

## What you do not have

- TDIS Azure subscription or Key Vault secrets  
- Production data  
- TDIS infrastructure (Terragrunt) repos  

---

## Next step

[03-prerequisites.md](03-prerequisites.md) → [06-local-development.md](06-local-development.md)

Read [07-postgresql-schema.md](07-postgresql-schema.md) to understand Source A vs Source B — schema **load** happens in ch. 06 Step 1.
