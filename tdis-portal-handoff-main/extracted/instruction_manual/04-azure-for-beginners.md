# 04 — Azure for Beginners

For BIT deploying to **your own Azure subscription**. Skip this chapter if you only run locally (Track A).

## Core concepts

| Term | Plain English |
|------|---------------|
| **Subscription** | Your billing account with Microsoft. One company may have several. |
| **Resource group** | A folder that holds related Azure resources (database, web app, etc.). |
| **Region** | Physical datacenter location (e.g. `southcentralus`). Pick one close to Texas users. |
| **Resource** | Any single thing you create (database, app, vault). |

## Create an Azure account

1. Visit https://azure.microsoft.com/free/  
2. Sign up with work email  
3. Complete verification (credit card may be required for identity; free tier available)

## Install and sign in to Azure CLI

```bash
# Linux (example)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

az login
# Browser opens — sign in

az account list --output table
az account set --subscription "Your Subscription Name"
```

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

## Why we do not use Terragrunt

TDIS manages its own infrastructure with Terragrunt/Terraform internally. **You do not need those tools.** This manual uses the Portal and `az` CLI only — see [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md).

## Next steps

1. [08b-azure-resource-checklist.md](08b-azure-resource-checklist.md) — what to create  
2. [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md) — how to create it  
