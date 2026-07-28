# Meeting Explanation — What Was Said & What It Means

## 1. Project Overview (What This Is)

This is the **IgnitionNet** project — a machine learning system that predicts WHERE and WHEN wildfires will be discovered, using:
- **FPA-FOD** (US Forest Service fire database) → Labels only
- **gridMET** (daily weather data) → Weather features
- **LANDFIRE** (landscape rasters) → Static terrain/fuel features
- **H3 hexagonal grid cells** → Spatial framework

The project is split into two states: **Texas first**, then **California**.

---

## 2. What the Texas Official Said (Interpreting the Meeting)

### 🔵 Statement 1: "Complete the process for Texas"

**What it means:**

Texas is currently **Phase 3 complete** — the baseline XGBoost model has been trained and evaluated:
- AUROC = **0.857**, AUPR = **0.398** (4.4× better than random)
- But **this is NOT fully complete** — the model was run WITHOUT the 4 LANDFIRE raster files

**The 4 missing files** (the "missing four files they will share") are:

| File | What it is | Saves as |
|------|-----------|----------|
| `BP_national.tif` | Burn Probability (from FSim) | `avg_burn_prob` feature |
| `WHP_2023.tif` | Wildfire Hazard Potential | `whp` feature |
| `FLEP4_national.tif` | Flame Length Exceedance Prob | `flep4` feature |
| `CFL_national.tif` | Canopy Fuel Load | `cfl` feature |

Right now ALL FOUR of these features are **zero-filled placeholders** in the Texas model. The official said their team will **share these 4 raster files** (they have them and will provide them).

**What "complete the process for Texas" means step-by-step:**
1. Receive the 4 LANDFIRE `.tif` files from the Texas officials
2. Place them in `V2/data/rasters/`
3. Re-run `run_phase2e_static.py --state TX` → extracts real landscape features
4. Re-run `run_phase2g_assemble.py --state TX` → assembles final training dataset
5. Re-run `run_phase3_train.py --state TX` → train model with full feature set
6. Expected AUROC jumps from **0.857 → ~0.90–0.93**
7. Run `run_phase3_visualize.py --state TX` → generate final evaluation plots

Also fix two known bugs before re-running:
- **Bug 1 (gridMET fill value):** Values of 32767 leaking through as valid data
- **Bug 2 (5-day rolling stats):** 84.4% are NaN because rolling is computed on sparse rows instead of fetching 5 actual preceding days from NetCDF files

---

### 🔵 Statement 2: "Go for California" / "Work for California like that"

**What it means:**

After Texas is fully complete, do the **same exact pipeline for California**.

California is currently at a much earlier stage — only Phase 2A and 2B have been done (the `outputs/california/` folder only has the H3 grid and schema files — no training data, no model).

**"Like that" means:** Mirror the exact same pipeline structure that was built for Texas, but applied to California:
- Same phases: 2A → 2B → 2C → 2D → 2E → 2F → 2G → Phase 3
- Same feature set (LANDFIRE + gridMET + temporal + location)
- Same negative sampling method (DAY-MATCHED, 1:10 ratio)
- But California uses **H3 Resolution-8** (smaller cells, ~860m) vs Texas's **Resolution-7** (~1.9km)
- California has **15,639 fire events** (vs Texas's 34,203)
- Expected training rows: **~172,073** (15,639 fires × 11 including negatives)

The 4 LANDFIRE rasters the officials share are **national rasters** — they cover the entire CONUS, so **the same 4 files work for both Texas and California**. No separate download needed for California.

---

### 🔵 Statement 3: "Change the Texas into index file"

**What it means:**

Currently all the pipeline scripts (`run_phase2a.py`, `run_phase2b.py`, etc.) are individual Python scripts. The instruction is to **create a master `index.py` or `main.py`** entry-point file for Texas — essentially a single file that:
- Runs all the Texas phases in the correct order
- Accepts `--state TX` as an argument
- Orchestrates the whole pipeline from start to finish

Think of it like converting individual chapter scripts into a single book with a table of contents. Instead of manually running each `run_phase2a.py`, `run_phase2b.py`, etc., you run ONE file that handles everything.

This is consistent with how the V1 version had a `main.py` as the orchestrator.

**Then "work for California like that"** = build the same California pipeline organized the same way (with its own California-specific index/main file).

---

## 3. Summary — What Needs to Happen (In Order)

```
STEP 1 — Receive the 4 LANDFIRE rasters from Texas officials
         (they said they will share: BP, WHP, FLEP4, CFL)

STEP 2 — Fix known bugs in Texas pipeline
         Bug 1: gridMET fill-value contamination (32767)
         Bug 2: 5-day rolling stats computed wrong

STEP 3 — Complete Texas with real LANDFIRE data
         → Place .tif files in V2/data/rasters/
         → Re-run Phase 2E, 2G, Phase 3
         → Expected AUROC improvement: 0.857 → ~0.90-0.93

STEP 4 — Create Texas "index file" (master orchestrator script)
         → Single entry point to run all Texas phases

STEP 5 — Replicate the same pipeline for California
         → Same phases (2A-2G, Phase 3)
         → Same 4 raster files work (national rasters cover all CONUS)
         → California index file organized same way as Texas
         → H3 Resolution 8 (not 7 like Texas)
         → ~172,073 training rows expected
```

---

## 4. What Is Already Done vs. What Is Pending

| Component | Texas Status | California Status |
|-----------|-------------|-------------------|
| Phase 1 (EDA) | ✅ Complete | ✅ Complete |
| Phase 2A (Schema) | ✅ Complete | ✅ Complete |
| Phase 2B (H3 Grid) | ✅ Complete | ✅ Complete |
| Phase 2C (Fire labels) | ✅ Complete | ❌ Not started |
| Phase 2D (Negative sampling) | ✅ Complete | ❌ Not started |
| Phase 2E (LANDFIRE extraction) | ⚠️ Done but with ZERO data (rasters missing) | ❌ Not started |
| Phase 2F (gridMET extraction) | ⚠️ Done but has bugs | ❌ Not started |
| Phase 2G (Dataset assembly) | ⚠️ Done but incomplete features | ❌ Not started |
| Phase 3 (Model training) | ⚠️ Baseline only (AUROC 0.857) | ❌ Not started |
| Index/Main orchestrator file | ❌ Not created | ❌ Not created |

---

## 5. The "Missing Four Files" — Confirmed

From `missing_data_diagnosis.md` and `PLACE_RASTERS_HERE.txt`:

The 4 LANDFIRE rasters are expected in `V2/data/rasters/` and are currently absent.
The Texas officials said **they will provide these files**.

Once received, place them in `V2/data/rasters/` with these filenames (any will be auto-detected):
- `BP_national.tif` or `BP_CONUS.tif`
- `WHP_2023.tif` or `WHP_CONUS.tif`
- `FLEP4_national.tif` or `LC22_FLEP4_220.tif`
- `CFL_national.tif` or `LC22_CFL_220.tif`
