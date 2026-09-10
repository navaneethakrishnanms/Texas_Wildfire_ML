# 00 — Start Here

Read this **once** before anything else. This manual assumes you may have **never used Azure or the command line**.

## What you are building

The **TDIS Portal** — a web map showing Texas weather events. You will:

1. **Run it on your laptop** (free, recommended first) — [06-local-development.md](06-local-development.md)  
2. **Deploy to your Azure account** (costs ~$45–60/month) — [08-deploy-to-azure-by-hand.md](08-deploy-to-azure-by-hand.md)  
3. **Build Databricks pipelines** (your main job per TRD) — [14-trd-reference.md](14-trd-reference.md)

Code repos arrive **separately** from TDIS — see [05-get-the-code.md](05-get-the-code.md).

---

## What is a terminal?

A **terminal** (also called command line or shell) is a text window where you type commands.

| OS | How to open |
|----|-------------|
| **Windows** | Search "Terminal" or "PowerShell" in Start menu. Docker Desktop users: use the terminal inside Docker Desktop or install [Windows Terminal](https://aka.ms/terminal). |
| **Mac** | Applications → Utilities → **Terminal** |
| **Linux** | Ctrl+Alt+T or search "Terminal" |

You will use **several terminals at once** for local dev — see [03-prerequisites.md](03-prerequisites.md).

---

## How to copy-paste commands

1. Highlight the command text in this manual  
2. Copy: **Ctrl+C** (Windows/Linux) or **Cmd+C** (Mac)  
3. Click inside the terminal window  
4. Paste: **Ctrl+Shift+V** (Linux terminal) or **Ctrl+V** / **Cmd+V**  
5. Press **Enter** to run  

Paste **one block at a time**. Wait for each command to finish before running the next.

---

## Pick your path

### Path 1 — Local first (recommended)

Best if you have never deployed to the cloud.

```
00-start-here  →  03-prerequisites  →  05-get-the-code
    →  07-postgresql-schema  →  06-local-development
```

**Success:** http://localhost:5173 shows a map.

### Path 2 — Azure deploy

Do Path 1 first if possible. Then:

```
04-azure-for-beginners  →  08b-azure-resource-checklist
    →  08-deploy-to-azure-by-hand
```

**Success:** Your Static Web App URL shows a map.

### Path 3 — First meeting prep

```
00-start-here  →  01-what-is-this  →  02-architecture  →  07-postgresql-schema  →  14-trd-reference
```

---

## Two ways to do Azure (both in ch. 08)

Every Azure phase offers **two paths**:

| Path | For who | What it means |
|------|---------|---------------|
| **Easy** | Beginners | Azure Portal — click buttons, paste values into forms |
| **Detailed** | Comfortable with CLI / prod setup | `az` commands + Key Vault like TDIS production |

Use **Easy** until the portal works. Switch to **Detailed** when your security team requires Key Vault.

Tier 2 (VNet, Application Gateway) is **reference only** — [08c-azure-tier2-reference.md](08c-azure-tier2-reference.md). Do not start there.

---

## Normal things that look like bugs

| What you see | Is it a bug? |
|--------------|--------------|
| Dashboard widgets empty, no errors | **No** — data comes from Databricks sync (you build this later) |
| Contact Us email fails locally | **No** — unless you configured SMTP |
| Map is gray/blank | **Yes** — usually missing Mapbox token — see ch. 12 |
| API health check returns 200 but no events | **No** — expected until Source B tables exist |

---

## Next step

Run the pre-flight checklist: [03-prerequisites.md](03-prerequisites.md)
