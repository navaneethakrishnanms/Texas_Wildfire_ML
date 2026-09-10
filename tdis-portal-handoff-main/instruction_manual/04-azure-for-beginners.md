# 04 — Azure for Beginners

For BIT deploying to **your own Azure subscription**. Skip this chapter if you only run locally (Track A).

## Core concepts

| Term | Plain English |
|------|---------------|
| **Subscription** | Your billing account with Microsoft. One company may have several. |
| **Resource group** | A folder that holds related Azure resources (database, web app, etc.). |
| **Region** | Physical datacenter location (e.g. `southcentralus`). Pick one close to Texas users. |
| **Resource** | Any single thing you create (database, app, vault). |
| **VNet** | Virtual network — private network inside Azure. TDIS uses one; **you skip it in Tier 1**. See [08c-azure-tier2-reference.md](08c-azure-tier2-reference.md). |
| **Subnet** | A slice of a VNet assigned to one type of resource (e.g. App Gateway needs its own). |

## Two Azure deployment tiers

| Tier | Who | Networking | Chapter |
|------|-----|------------|---------|
| **Tier 1** | Junior dev — **start here** | Public App Service / SWA URLs, PostgreSQL firewall rules | [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md) |
| **Tier 2** | Network / security team | VNet, subnets, Application Gateway, private PostgreSQL | [08c-azure-tier2-reference.md](08c-azure-tier2-reference.md) |

> TDIS production runs **Tier 2**. You do **not** need Tier 2 to prove the portal works.

---

## Create an Azure account

1. Visit https://azure.microsoft.com/free/  
2. Sign up with work email  
3. Complete verification (credit card may be required for identity; free tier available)

---

## Install Azure CLI

### Linux

```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az --version
```

### Windows

```powershell
winget install Microsoft.AzureCLI
```

Or download from https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows

### macOS

```bash
brew install azure-cli
```

---

## Sign in and verify

```bash
az login
# Browser opens — sign in with your Azure account

az account list --output table
az account set --subscription "Your Subscription Name"

az account show
```

**Expected:** JSON showing your subscription name and `"state": "Enabled"`.

---

## Azure Portal tour

Open https://portal.azure.com

1. **Search bar (top)** — type any resource name (e.g. "App Services", "PostgreSQL")  
2. **Resource groups** — left menu → Resource groups → Create  
3. **All resources** — see everything in your subscription  

Common tasks:

| Task | Portal path |
|------|-------------|
| Create resource group | Resource groups → Create |
| Create database | Search "Azure Database for PostgreSQL flexible server" |
| Create web app | Search "App Service" |
| Create Static Web App | Search "Static Web App" |
| Store secrets | Search "Key Vault" |
| Container images | Search "Container registry" |

[Screenshot: Azure Portal home — search bar and Resource groups menu]

---

## Rough Azure cost

### Tier 1 (minimal dev — follow ch. 08)

| Resource | Approx. monthly |
|----------|-----------------|
| PostgreSQL Flexible Server (B1ms) | $25–40 |
| App Service Plan (B1) | $13 |
| Static Web App (Free) | $0 |
| Key Vault | ~$1 |
| Container Registry (Basic) | ~$5 |
| **Total** | **~$45–60** |

### Tier 2 additions (see 08c)

| Resource | Approx. monthly |
|----------|-----------------|
| Application Gateway | ~$150+ |
| VNet | $0 (resources inside cost money) |

Shut down resources when not testing.

---

## Why we do not use Terragrunt

TDIS manages its own infrastructure with Terragrunt/Terraform internally. **You do not need those tools.** This manual uses the Portal and `az` CLI only.

---

## Next steps

1. [08b-azure-resource-checklist.md](08b-azure-resource-checklist.md) — Tier 1 shopping list  
2. [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md) — Tier 1 step-by-step  
3. [08c-azure-tier2-reference.md](08c-azure-tier2-reference.md) — TDIS production networking reference (later)
