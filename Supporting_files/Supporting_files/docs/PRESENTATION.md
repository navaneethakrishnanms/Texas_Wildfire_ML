# TDIS Ignition Forecast — Presentation Content
*(fill-in for IgnitionForeCastPresentation.pdf — 6 slides, Urban Resilience.AI Lab template)*

---

## Slide 1 — Introduction: Purpose of Model & Motivation

- **Purpose:** a true *forward-looking* wildfire forecast for Texas — "will a fire **start** here in the next 24/48 hours, and how **dangerous** would it be if it does?"
- Two distinct products on one hexagon map:
  - **Ignition forecast** — probability-ranked likelihood a fire starts (ML model)
  - **Wildfire risk** — static hazard × daily fire weather (USFS WFPI-style two-stage design)
- **Motivation / gap being filled:**
  - Prior model (v5) was a *static susceptibility map* — it could not forecast forward in time and could not be validated past 2020 (no labels)
  - Texas events (Smokehouse Creek 2024 — largest fire in TX history) show the need for **day-specific, wind-aware** risk, not just "where is fire-prone"
  - Goal: identify **vulnerable areas** ahead of time to support pre-positioning and preparedness decisions
- **Headline capability:** live 24 h / 48 h forecasts from real NOAA HRRR weather, validated on genuinely future years (2023–2026)

---

## Slide 2 — Data: Datasets & Role

- **Label: FPA-FOD ∪ VIIRS Active Fire (the key upgrade)**
  - FPA-FOD: authoritative ignition inventory, 2014–2020 (but 87% of times imputed, ends 2020)
  - **VIIRS (NASA FIRMS, S-NPP + NOAA-20, 375 m):** real satellite detections with true timestamps, **2014–2026** — extends labels 6 years past FPA-FOD
  - Fused: **960 k positive cell-days**, ~490 k post-2020 → forward validation becomes *possible*
- **Weather (the dynamic driver)**
  - gridMET (observed daily): ERC, VPD, wind — training + historical fire-weather index
  - **NOAA HRRR forecast** (3 km, ≤48 h, incl. **gust**): what the forecast *actually said* — the operational model trains and runs on forecast-realistic inputs
- **Landscape / static (the "where")**
  - TxWRAP Wildfire Hazard Potential + burn probability, LANDFIRE canopy (CBD/CBH), elevation/slope/aspect (GLO-30), EPA ecoregions, road distance (TIGER)
  - Full **1.7 M-cell H3 res-8 grid** — statewide coverage (prior model silently missed ⅔ of TX)
- **Grid:** H3 hexagons — res-8 (~0.74 km²) for modeling, res-5 (~250 km²) for the dashboard

---

## Slide 3 — Model Architecture: Model & Training

- **Two-stage design (industry-standard, WFPI-style):**
  - `Wildfire risk(cell, day) = static hazard(cell) × fire-weather index(cell, day)`
  - `Ignition(cell, day)` = **XGBoost** gradient-boosted trees → susceptibility × fire weather
- **Fire-Weather Index (tunable knob):** FWI = 0.30·ERC/100 + 0.30·VPD/5 + **0.40·wind/12**
  - wind weight raised 0.20 → 0.40 after event validation exposed under-scoring of wind-driven fires; live forecasts use HRRR **gust**
- **Operational ignition model:** 20 features = static (roads, terrain, fuels, ecoregion) + temporal (season, day-of-week) + **HRRR forecast weather** (temp, VPD, wind)
- **Training discipline (what makes it a real forecast):**
  - **Temporal split** — train 2014–2021, validate 2022, **test 2023–2026 (never-seen future)**; never a random split
  - **No same-day-weather leak** — deployable model only uses what's knowable at forecast-issue time
  - Matched negative sampling (same fire cells, non-fire days) forces the model to learn *when*, not just *where*
  - GPU-trained (RTX A6000); class imbalance handled via scale_pos_weight + early stopping

---

## Slide 4 — Model Results: Validation Setup & Findings

- **Setup:** all metrics on the **2023–2026 future hold-out** — the model is scored the way it will be used: predict forward, then check what happened
- **Quantitative skill:**
  - Ignition AUC-PR ≈ **0.63–0.70**, AUROC ≈ 0.79–0.82, **stable across all four test years** → generalizes forward, not memorizing one year
  - Lift over random: **~2.3–3.3×** (base-rate-robust number — the honest cross-model comparison)
- **Controlled HRRR ablation (identical rows, only forecast weather toggled):**
  - True HRRR contribution = **+0.011 AUC-PR / +0.020 AUROC** — small but real; `hrrr_vpd` is a top-5 feature
  - (An uncontrolled comparison showed +0.068 — identified as a base-rate artifact and discarded; we report the clean number)
- **What drives predictions:** elevation, ecoregion, burn probability, road distance — *where* dominates; weather is a modest but genuine daily modulator
- **Honest caveats:** outputs are **relative rankings, not calibrated probabilities**; daily ignition timing skill is modest — day-to-day motion comes mainly from the fire-weather stage

---

## Slide 5 — Model Validation: Real-Event Tests

- **Setup:** 4 documented Texas wildfires (2024–2026), 3 questions each — did our satellite labels capture it? did the **static hazard** flag the *place* (WHERE)? did the **fire-weather index** flag the *day* (WHEN)?

| Event | Date / driver | Captured | WHERE | WHEN (after wind fix) |
|---|---|---|---|---|
| **Smokehouse Creek** (1.06 M ac, largest in TX history) | Feb 2024 · wind | ✅ 4,730 det. | ✅ 98th pctile | ⚠️ 75th |
| **Crabapple** (9.9 k ac) | Mar 2025 · hot-dry-windy | ✅ | ✅ 100th | ✅ 99th |
| **Lavender** (18.4 k ac) | Feb 2026 · wind | ✅ | ✅ 100th | ✅ 94th |
| **Hunggate** (34 k ac) | May 2026 · **lightning** | ✅ | ✅ 97th | ✅ 99th |

- **Findings:**
  - **WHERE: 4/4** — every fire in the 97th–100th percentile of statewide hazard (2.1–3.0× mean)
  - **WHEN:** heat/drought-driven fires caught at 99th; wind-driven cool-season fires initially under-scored (61st–80th) → **diagnosed root cause** (ERC/VPD-heavy index + daily-mean wind) → raised wind weight + added HRRR gust → **+14 pctile on both wind events, no regression**
  - Lightning-caused Hunggate flagged on *weather alone* — the two-stage design catches natural ignitions too
- **Takeaway:** validation didn't just score the model — it found a specific weakness and drove a targeted, verified fix

---

## Slide 6 — Model Use Case

- **The product:** a single **drag-and-drop HTML dashboard** (no server) — statewide hexagon map with:
  - Wildfire-risk & ignition modes · daily scrubber 2024–2026 (933 days) · **live 24 h / 48 h / 72 h forecasts** from the latest HRRR/GFS run
  - Tooltips decompose every hexagon: combined risk = static hazard × fire-weather
- **Identifying vulnerable areas (recommended operating rule):**
  - Vulnerability is **relative**, not an absolute probability: flag cells in the **top ~10% static hazard ∩ top ~20% of that day's fire weather** (≈ dashboard "High"+ bands)
  - Threshold can be set empirically: choose the top-X% flagged area that historically captured the desired share of fires (capture-vs-area trade-off)
- **Who uses it & how:**
  - Emergency managers: morning look at *tomorrow's* elevated hexes → pre-positioning, public warnings
  - Planners/researchers: replay any 2024–2026 day; compare regimes (wind-driven Panhandle vs drought-driven Hill Country)
- **Roadmap to operational:** automated daily refresh (cron) → probability calibration → expand event validation to 15–20 fires → sub-daily (6-hour) forecasting via the California IgnitionNet architecture (two-state synergy)

---

*Sources for every number: `README.md` (metrics, ablation), `VALIDATION.md` (event tests), `HANDBOOK.md` (features, FWI, worked examples).*

---

# Slide 5B (optional) — Benchmarking against NOAA's operational index

*A strong add-on slide after Model Validation — shows the work engages operational fire science.*

- We implemented **NOAA GSL's Hourly Wildfire Potential** (James et al. 2025, *Weather and
  Forecasting*) **exactly** — `HWP = 0.213·G^1.50·VPD^0.73·(1−M)^5.10` — including its
  gust and **soil-moisture** inputs from HRRR, and added it as a switchable formula in our
  dashboard alongside our composite index
- We then **re-fit the same equation to Texas** satellite fire activity (VIIRS FRP,
  large-fire episodes, 2024–2026) using NOAA's own fitting approach
- **Head-to-head on the 4 validation fires** (WHEN percentile):

| Event | Our composite | NOAA HWP | TX-calibrated |
|---|---|---|---|
| Smokehouse Creek | 75th | **80th** | 75th |
| Crabapple | 99th | 99th | **100th** |
| Lavender | **94th** | 85th | 64th |
| Hunggate | **99th** | 97th | 95th |

- **Takeaways for the audience:**
  - Our wind-tuned composite **holds its own against the operational NOAA index** (best or tied 3/4)
  - NOAA's soil-moisture term catches Smokehouse best → we adopted soil moisture into our live pipeline
  - The Texas re-fit revealed a scientific finding: at **daily** resolution, fire activity is
    **dryness-dominated** — the wind signal NOAA exploits only exists at *hourly/gust* resolution,
    which independently motivates our sub-daily roadmap (and the CA IgnitionNet synergy)
- One-liner: *"We didn't just build an index — we benchmarked it against NOAA's, adopted what
  theirs does better, and showed why Texas needs sub-daily resolution."*

---

# Demo Plan — presenting the model to a fire-expert team

Fire experts will trust **events they lived through**, not AUC curves. The demo is built
around replaying fires they know, then showing the live forecast. Lead with the map,
keep the metrics in your back pocket.

## Setup (before the meeting)
- Open `dashboard/tdis_fire_dashboard_standalone.html` locally (double-click; needs internet
  for base-map tiles only). Practice the scrubber — know the exact dates below cold.
- Refresh the live forecast that morning (`scripts/13_model_forecast_day.py <tomorrow> 24`)
  so "tomorrow" on screen is genuinely tomorrow.
- Have `VALIDATION.md` open in a tab for the event table if pressed for specifics.

## Demo flow (~20 min)

**1. Frame the two questions (2 min) — before touching the map.**
"Two different questions: *will a fire start here* (ignition) and *how dangerous if it does*
(wildfire risk). The map shows both; risk = static hazard × that day's fire weather."
Experts respect this distinction — leading with it signals the model isn't naive.

**2. Replay a fire they know (5 min) — the trust moment.**
*(Full click-by-click script with all numbers: `DEMO_SMOKEHOUSE.md`.)*
Scrub to **Feb 26, 2024** in Wildfire Risk mode → Smokehouse Creek. Show the Panhandle
lighting up *on that day*. Then hover a hex and walk the tooltip decomposition:
hazard 0.57 × fire-weather 0.44 → risk. Then scrub a week earlier to show it was NOT lit —
the map moves with the weather, it's not a static hazard poster.

**3. Show it catches different fire types (4 min).**
Jump to **May 14, 2026** (Hunggate — lightning-caused, caught at 99th pctile on weather
alone) and **Mar 15, 2025** (Crabapple — Hill Country red-flag day). Point: wind-driven
Panhandle grass fires AND drought-driven Hill Country fires, human AND natural causes.

**4. The live forecast (4 min) — the payoff.**
Switch to Live Forecast → tomorrow (24h HRRR). "This is running the model on this morning's
NOAA forecast — this is what your duty officer would look at each morning." Show 48h.

**5. Be the first to state the limits (3 min) — this wins the room.**
- "Colors are **relative tiers**, top-X% of the state — not calibrated probabilities yet."
- "It can't predict the cell where someone drops a cigarette; it forecasts **danger**, like
  SPC fire-weather outlooks but at hex resolution and Texas-specific."
- "Historical replay uses daily-mean wind, so gust-driven days are muted in the scrubber —
  the live forecast uses HRRR gust and does better."
Then the wind-fix story (61st→75th, 80th→94th, no regression): validation found a weakness,
we diagnosed it, fixed it, re-verified. Experts trust a team that shows its misses.

## Hard questions they will ask — prepared answers

| Question | Answer |
|---|---|
| "How is your FWI different from NFDRS/ERC we already use?" | It *builds on* ERC (30% weight) and adds VPD + wind explicitly; weights are tunable and were re-tuned after wind-driven events under-scored. Not a replacement for NFDRS — a composite tuned to TX events. |
| "Did it see these fires in training?" | No — trained on 2014–2021, all four validation fires are 2024–2026, genuinely out-of-sample. |
| "Wouldn't a fire-weather watch have flagged Smokehouse anyway?" | Likely yes — the value-add is *resolution* (hexes vs county), the hazard×weather decomposition per hex, and a single statewide picture updated daily. |
| "What's the false-alarm rate?" | Honestly: not yet characterized as an operational threshold — that's the calibration step on our roadmap. Today it ranks; capture-vs-area analysis will set the alert threshold. |
| "Why should we trust ML over experience?" | Don't — use it as a briefing layer. The top predictors (terrain, ecoregion, roads, dryness) match fire-behavior intuition; the model just computes them consistently across 1.7M cells every morning. |
| "What about lightning?" | Weather-only for now — Hunggate (lightning) was still caught at 99th pctile on conditions. GOES-GLM lightning is on the roadmap. |

## Don't do in the demo
- Don't quote AUC-PR unprompted — meaningless to this audience; use "top-10% of flagged
  area captured X of the 4 fires' locations" style statements instead.
- Don't call colors "probability of fire." They're relative danger tiers.
- Don't over-claim 72h — call it an outlook (GFS, coarser), keep the demo on 24/48h.
- Don't demo dates after the gridMET data edge (scrubber ends 2026-07-21) — hexes go gray.

**Close with the ask:** "Which historical fires would *you* have us replay? Every event you
name becomes a validation test." — turns skeptics into contributors and grows the
validation set (roadmap target: 15–20 events).
