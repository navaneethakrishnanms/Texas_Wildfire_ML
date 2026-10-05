# Demo Script — Smokehouse Creek Fire (Feb 26, 2024)

The centerpiece case study: the **largest wildfire in Texas history**, replayed on our
dashboard with real stored data. Everything below is either documented public record or
computed from our actual datasets (`VALIDATION.md §3`, `HANDBOOK.md §7B`).

---

## 1. The fire — documented facts (set the scene, ~2 min)

- **Ignited:** February 26, 2024, near Stinnett, **Hutchinson County, Texas Panhandle**
- **Size:** ~**1.06 million acres** (with the merged Windy Deuce complex) — the **largest
  wildfire in Texas history**, burning into Oklahoma
- **Cause:** downed **power line** (utility later acknowledged involvement) — a human-
  infrastructure ignition, not lightning
- **Conditions:** a classic **cool-season wind-driven Panhandle grass fire** — dormant cured
  grass, strong winds with gusts reported **60+ mph**, low humidity; *not* a hot day
- **Impact:** fatalities, thousands of cattle lost, widespread ranch/structure damage;
  weeks to full containment
- **Why it's the right demo:** everyone in a Texas fire audience knows this fire — and it's
  the *hard* case for weather indices (wind-driven, not heat-driven)

## 2. What our system shows — real numbers (the evidence)

### Did the satellite labels capture it? ✅ Perfectly
| Date | VIIRS detections in the fire box |
|---|---|
| Feb 25 | 3 |
| **Feb 26 (ignition day)** | **184** |
| Feb 27 (peak spread) | **3,529** |

4,730 detections total — the label data tracks the documented timeline exactly.

### Did the model flag the PLACE? ✅ 98th percentile
- Static hazard at the ignition area: **0.568 → 98th percentile statewide, 2.9× the state
  mean.** The Panhandle was flagged as one of the most dangerous places in Texas *before
  any weather is considered* — cured grass fuels, high burn probability.

### Did the fire-weather index flag the DAY? ⚠️ Partially — and the story is the point
Real gridMET values at the fire location on Feb 26:

```
ERC 51   ·   VPD 0.78 kPa (LOW — it was a COLD day)   ·   wind 7.3 m/s (near year-max 9.9)

FWI = 0.30·(51/100) + 0.30·(0.78/5) + 0.40·(7.3/12)
    = 0.153         + 0.047        + 0.244          =  0.443
Wildfire risk = hazard 0.568 × FWI 0.443 = 0.252
```

- **Wind contributes over half the index (0.244)** — dryness terms are small because this
  was a *cold* windy day, not a hot one.
- Day percentile: **61st with the original (dryness-heavy) weights → 75th after we raised
  the wind weight** (0.20 → 0.40). Improved, honestly not nailed.
- **Why not higher? A data limit, not a model limit:** the historical archive (gridMET) has
  only **daily-mean wind** (7.3 m/s). The danger that day was **gusts of 60+ mph** — the
  daily mean literally cannot see them. **The live forecast now uses HRRR gust**, so a
  Smokehouse-type setup *forecast for tomorrow* scores far better than this replay implies.

## 2B. Homework-style proof: all three fire-weather equations, computed by hand

Everything below is computed from the stored data (`data/fwi_components_res5.parquet`,
box mean over the 40 res-5 cells in 35.5–36.3°N, −101.6–−100.2°W on **2024-02-26**) and
then **checked against what the dashboard actually contains** — so every number on screen
is reproducible by hand.

**Descriptive statistics — how unusual was Feb 26 in that box?**

| Input | Feb 26 (box mean) | Same cells, all 933 days (median) | Reading |
|---|---|---|---|
| ERC | 50.8 | 46.0 | slightly above normal drought |
| VPD | **0.78 kPa** | 1.26 | **BELOW normal — a cold day** |
| wind | **7.32 m/s** | 4.38 | **~year-max territory** |
| fm100 | **9.0 %** | 11.8 | fuels drier than usual |

Statewide context (3,381,192 cell-days, 2024–2026): ERC mean 44.2 (p95 = 78.6),
VPD mean 1.33 kPa (p95 = 2.77), wind mean 4.22 m/s (p95 = 7.03 → the box's 7.32 beats
the statewide 95th percentile), fm100 mean 12.8 %.

**Equation 1 — Composite (additive, the default):**
```
FWI = 0.30·ERC/100 + 0.30·VPD/5 + 0.40·wind/12
    = 0.30·50.8/100 + 0.30·0.78/5 + 0.40·7.32/12
    = 0.153 + 0.047 + 0.244            = 0.443
```
Wind carries over half the score; the dryness terms are small because it was cold.

**Equation 2 — NOAA HWP (James et al. 2025, Eq. 3 — multiplicative):**
```
inputs: G = 1.5·7.32 = 11.0 m/s (gust proxy) · VPD = 7.8 hPa · M = fm100/30 = 0.300 → dry = 0.700
raw HWP = 0.213 · 11.0^1.50 · 7.8^0.73 · 0.700^5.10 = 5.63
normalized = 5.63 / 26.25 (TX archive p99.5)        = 0.214
```
Absolute value looks low — but the NOAA index is extremely right-skewed (statewide
median is just 0.040), so 0.214 is still the **80th percentile of the year** at that
location. Multiplicative indices are read by *rank*, not magnitude.

**Equation 3 — TX HWP (our re-fit of the same form):**
```
raw = 17.60 · 11.0^0.05 · 7.8^0.05 · 0.700^0.92 = 15.84
normalized = 15.84 / 20.10                      = 0.788
```
Nearly all of it comes from the dryness term (exponent 0.92) — the wind exponent (0.05)
contributes almost nothing, which is the fit's core finding: at daily resolution TX fire
activity tracks fuel dryness.

**Verification against the dashboard (the "check your answer" step):**

| | Hand calc | Stored in dashboard | Wildfire risk (× hazard 0.568) |
|---|---|---|---|
| Composite | 0.443 | **0.444** ✓ | 0.252 |
| NOAA HWP | 0.214 | **0.217** ✓ | 0.123 |
| TX HWP | 0.788 | **0.788** ✓ | 0.447 |

(Tiny differences = mean-of-index vs index-of-mean across the 40 cells; per-cell they
match exactly.) Reproduce any of this via the components parquet + `scripts/fwi_config.py`.

### Ignition sample calc (same day, same box)

The dashboard's historical **Ignition** layer is the transparent two-stage heuristic:

```
Ignition(day) = ML susceptibility(cell) × FWI(cell, day)
              = 0.357 × 0.443 = 0.158
```

where **0.357** is the trained model's mean ignition susceptibility over the 40 box cells
(FLARE-FILTERED honest model, clean 2023–2026 test means; per-cell range 0.205–0.495).
*History note:* before the flare filter this box averaged 0.316 with one cell at 0.716 —
that cell is a gas flare (see below), and its collapse to ≤0.495 after label cleaning is
itself a validation that the filter worked.

**Honest note:** the *live* 24/48 h ignition forecast is different — there the actual
XGBoost model (400 trees, 20 features, flare-filtered) runs directly on HRRR forecast
weather for all 1.7M cells. That output is not hand-computable; its evidence is
statistical (clean-label temporal hold-out: honest AUC-PR ≈ 0.45 / lift 1.9×,
operational-HRRR ≈ 0.48 / lift 2.1×) — see `VALIDATION.md §9`.

### ⚠️ Score ≠ probability — the most important reading rule

![Score vs probability](demo_pngs/12_learning_score_vs_probability.png)
*(Figure shows the PRE-filter model — the 0.716 flare cell it exposes is what motivated the
label cleaning; post-filter that cell scores ≤0.495.)*

**0.316 susceptibility does NOT mean a 32% chance of fire** (nor does 0.140 mean 14%).
The empirical truth for these same 40 cells: a VIIRS fire on a **median of 0.3% of days**
(mean 2.0%), even though each hexagon is ~250 km². The model's scores live on a
compressed internal scale because it was trained on a **matched-negative sample**
(synthetic ~27% base rate) — its job is **ranking** (this cell above that cell, today
above yesterday), which validation shows it does well (2–3× lift, 4/4 events in top
percentiles). Converting scores to honest percentages is the **calibration pass** in the
roadmap (README §12, Tier 1.2). Until then: read colors and percentiles, never literal %.

**Bonus finding from this figure (data-quality lesson):** the dark-red hexagon "burned"
on 58.6% of days — that's a **gas flare**, not wildfire (VIIRS flags any persistent heat
source; the Panhandle is oil/gas country). It's also the box's highest-susceptibility
cell (0.716) — flare detections leaked into training labels and taught the model that
cell is fire-prone. Filtering persistent-anomaly detections (NASA FIRMS publishes a
static flare mask) is a roadmap label-cleaning item.

### Why these inputs are credible — the literature behind the demo

Every input driving this replay is an established predictor in peer-reviewed fire science:

- **ERC** — core NFDRS index (Bradshaw et al. 1983; Cohen & Deeming 1985); associated
  with large-fire occurrence in the western US (Riley et al. 2013, *IJWF*). Integrates
  weeks of drying — the "how primed is the season" signal.
- **VPD** — among the strongest atmospheric correlates of fire activity: Williams et al.
  (2014/2015); Higuera & Abatzoglou (2021, *PNAS*); the dryness axis of the Hot-Dry-Windy
  Index (Srock et al. 2018, *Atmosphere*).
- **Wind / gust** — the accelerant in foundational spread physics (Rothermel 1972) and
  the other HDWI axis (Srock et al. 2018). Wind-driven grass fire outbreaks are the
  hallmark southern-Great-Plains hazard — exactly the Smokehouse regime.
- **FM100** — NFDRS dead-fuel-moisture theory (Fosberg & Deeming 1971); the drought-lag
  fuel signal.
- **Soil moisture** (live HWP input) — linked to southern-plains wildfire occurrence/size
  (Krueger et al. 2015/2016, *IJWF*); the strongest term (exp 5.1) in NOAA's operational
  HWP (James et al. 2025, *Wea. Forecasting*).
- **Burn probability & WHP** (the WHERE layer) — USFS FSim framework (Finney et al. 2011)
  and WHP mapping (Dillon et al. 2015); TxWRAP is the Texas implementation.
- **Road distance & calendar features** — 84% of US wildfires are human-started, coupled
  to roads/WUI and weekly/holiday cycles (Balch et al. 2017, *PNAS*; Syphard et al. 2007).
- **Terrain & ecoregion** — slope enters spread physically (Rothermel 1972); environmental
  controls on fire regimes (Parisien & Moritz 2009).
- **VIIRS labels** — 375 m product validated in Schroeder et al. (2014, *RSE*); FPA-FOD is
  the authoritative US fire record (Short 2014).

In short: these are the same quantities operational systems (NFDRS, HDWI, NOAA HWP,
FSim/WHP) already stake decisions on — our contribution is wiring them into a
forward-validated, hexagon-level Texas forecast.

---

## 3. Dashboard walkthrough (click-by-click, ~5 min)

> Ready-made slide images for every step (and the other 3 validation fires) are in
> **`demo_pngs/`** — use these as backup if live demo/internet fails.

1. Open `dashboard/tdis_fire_dashboard_standalone.html` · mode = **🔥 Wildfire Risk**
2. Scrub to **2024-02-19** (one week before): Panhandle shows its usual elevated base — *point
   out it is NOT extreme yet*
3. Scrub forward day by day to **2024-02-26**: the Panhandle **lights up on ignition day**
4. **Hover the Stinnett-area hex** and read the tooltip decomposition out loud:
   hazard 0.57 × fire-weather 0.44 → the number is explainable, not a black box
5. Switch to **▦ Static Hazard** mode: "this is what a *static* map shows — the same picture
   every day of the year. The scrubber is what forecasting adds."
6. (Optional) Scrub past **Feb 27–28** and mention VIIRS recorded 3,500+ detections as the
   fire ran — our labels see the spread, not just the start

## 4. Talking points — what this case proves and doesn't

**Proves:**
- The **place** was flagged at the 98th percentile — the WHERE layer works
- The map **moves with the weather** — day-specific, not a hazard poster
- The validation process **found a real weakness** (wind-driven cool-season fires
  under-scored), we diagnosed the cause, re-tuned wind 0.20→0.40, verified improvement
  with **no regression** on heat-driven fires — the system is correctable and honest

**Doesn't prove / say plainly if asked:**
- It did not "predict" Smokehouse Creek — no system predicts a specific power-line failure.
  It identifies **elevated danger**: the right framing is SPC-style outlooks at hex scale
- The 75th-percentile day score is honest evidence of the daily-mean-wind data limit;
  the fix (HRRR gust) applies to the *forward* product, which is the one that matters
- Colors are **relative danger tiers**, not calibrated probabilities

## 5. Likely expert questions on this fire

| Question | Answer |
|---|---|
| "A red-flag warning was up that day — what do you add?" | Yes — and we agree with it. We add per-hex resolution, an explainable hazard×weather split, and one statewide picture that also runs on quiet days when no warning exists. |
| "Would your live forecast have caught it?" | The historical replay (daily-mean wind) scores 75th pctile. The live product uses HRRR **gust** — the 60+ mph gusts that drove this fire are exactly what it ingests. We can't re-run the past with HRRR gust archive-wide yet; that's on the roadmap. |
| "Was this fire in your training data?" | No — training ends 2021; Feb 2024 is fully out-of-sample. |
| "Power-line failures — do you model infrastructure?" | Not directly; road distance is our human-access proxy. Utility-line layers are a natural feature to add — good suggestion (invite them in). |

---
*Numbers: `VALIDATION.md §3/§6` (percentiles, before/after wind fix), `HANDBOOK.md §7B`
(worked FWI calculation), `data/fwi_components_res5.parquet` (raw weather components).*
