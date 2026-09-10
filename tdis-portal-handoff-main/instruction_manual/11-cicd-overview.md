# 11 — CI/CD Overview (TDIS Reference)

**You do not need to replicate this.** TDIS internal deployment only — for context.

## TDIS pipeline summary

| Repo | CI | Deploy target |
|------|-----|---------------|
| `tdis-portal-api` | Azure DevOps — pytest, Docker build | Azure Container Registry → App Service |
| `tdis-portal-frontend` | ESLint, Vitest, SonarQube, `npm run build` | Azure Static Web Apps |
| `tdis-portal-backend` | Tests, Docker | App Service |

Secrets (Mapbox, API URLs, keys) come from Azure DevOps **variable groups** linked to Key Vault at build time.

## What BIT does instead

- **Local:** manual `make dev` / `npm run dev` — [06-local-development.md](06-local-development.md)  
- **Azure:** manual `az` CLI — [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md)  
- **Optional:** set up your own GitHub Actions or Azure DevOps later  

## Infrastructure as Code

TDIS uses Terragrunt/Terraform internally. **Not included in handoff.**
