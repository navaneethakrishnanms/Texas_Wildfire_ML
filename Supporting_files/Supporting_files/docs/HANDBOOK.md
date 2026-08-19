# TDIS Wildfire Forecasting — Data & Modeling Handbook (Zero → Hero)

A beginner-friendly, in-depth guide to **every dataset, every feature, how they combine,
how the models learn, how forecasting works, and how it all ends up on the dashboard.**
Read top-to-bottom the first time; use the glossary (§10) as a cheat-sheet after.

> Companion docs: `README.md` (project summary), `PLAN.md` (design decisions),
> `../DATA_AND_FORECAST_ROADMAP.md` (data-source roadmap).

---

## 0. The two questions we answer (read this first)

Wildfire modeling conflates two genuinely different questions. Keep them separate in your head:

| | **Ignition** | **Wildfire (hazard) risk** |
|---|---|---|
| Question | Will a fire **start** here? | If a fire occurs, how **likely + severe** is it? |
| About | the spark / point of origin | the burn and its consequences |
| Nature | **dynamic** (changes hour to hour) | mostly **static** (fuels/terrain change slowly) |
| Analogy | "Will someone drop a match here today?" | "If a match is dropped, will it become a monster?" |

A useful mental model used operationally (e.g., the US Forest Service **Wildland Fire
Potential Index**, Dillon et al.):

```
Forecast fire danger  =  STATIC susceptibility (where fuels/terrain/people allow fire)
                       ×  DYNAMIC fire-weather (how dry/hot/windy it is that day)
```

Everything below builds toward being able to compute both sides of that equation.

---

## 1. The unit of analysis: H3 hexagons and "cell-days"

Before features, understand the grid. We chop Texas into **H3 hexagons** (Uber's H3
geospatial indexing system). We mostly use **resolution 8** (~0.74 km² per hex, ~460 m
across). Texas is ~940,000 hexes at this size.

- A **cell** = one hexagon (a fixed patch of ground).
- A **cell-day** = one hexagon on one calendar day. This is the fundamental **row** in
  our training data: "hexagon X on date D."
- Each cell-day gets a **label** (did a fire start there that day? 1/0) and a set of
  **features** (numbers describing that place and that day).

Why hexagons instead of a lat/lon grid? Hexagons have uniform neighbor distances (no
corner/edge distortion), tile the globe cleanly, and give every location a stable ID
you can aggregate up (res-8 → res-5 "parent" for coarser maps). The dashboard displays
**res-5** (~250 km² hexes, ~5,400 covering TX) so it's not 940k tiny polygons.

---

## 2. The features — what each one means, why it matters, where it comes from

Features fall into four families. For each: **plain meaning → why it matters for fire →
units/range → source (with literature)** and whether it's **static** (fixed per cell) or
**dynamic** (changes by day).

### 2A. FIRE LABELS — the "answer key" the model learns from
These aren't model inputs; they're the **targets** (what we're trying to predict).

**FPA-FOD ignitions** *(static inventory, 2014–2020)*
- **Meaning:** the official record of where and when wildfires were reported in the US.
- **Why:** it tells the model where fires *actually* started historically — the ground truth.
- **Detail:** ~36,000 Texas fire events; includes cause (human/equipment/lightning/unknown)
  and size. **Caveat:** 87% of the *time-of-day* is imputed (guessed), so it's reliable at
  the **day** level, not sub-daily.
- **Source:** Short, K.C. (2014, updated). *Spatial wildfire occurrence data for the US
  (FPA_FOD)*, USDA Forest Service Research Data Archive, RDS-2013-0009.

**VIIRS active-fire detections** *(dynamic, 2014–2026)*
- **Meaning:** satellites (S-NPP, NOAA-20) detect actively burning fires by their heat signature.
- **Why:** gives **real timestamps** (fixing FPA-FOD's imputed-time problem) and **extends
  labels past 2020** (FPA-FOD stops there), which is what lets us validate forecasts on
  recent years.
- **Detail:** 375 m resolution, ~2–4 satellite passes/day. **Caveat:** detections ≠
  ignitions (a fire must grow hot/big enough to be seen), so small/short fires are missed.
- **Source:** Schroeder, W. et al. (2014). *The New VIIRS 375 m active fire detection data
  product*, Remote Sensing of Environment. Distributed via NASA FIRMS.

We **fuse** these two: FPA-FOD gives the complete "where/what-day" inventory; VIIRS adds
real times and post-2020 coverage. A cell-day is labeled **1** if either source recorded
fire there. **Strictly, this label means "fire ACTIVE here this day" (fire occurrence),
not "fire ignited here today"** — a multi-day fire contributes several positive cell-days.
Why that fusion is valid, what it implies, and the reviewer-proof defense: **§19**.

---

### 2B. WEATHER — the dynamic driver (changes every day)
Weather is why fire danger differs between Tuesday and Wednesday. Source for the historical
daily values: **gridMET** (Abatzoglou, J.T. 2013, *Development of gridded surface
meteorological data...*, Int. J. Climatology) — a 4 km daily US weather dataset.

**ERC — Energy Release Component** *(dynamic; the single most important fire-weather index)*
- **Meaning:** a fire-danger index representing the **potential energy release** at the
  flaming front — essentially "how much dry fuel energy is available to burn." Integrates
  temperature, humidity, and especially **cumulative dryness** of dead fuels over weeks.
- **Why:** high ERC = fuels are cured and primed; strongly associated with large fires. It's
  slow-moving (reflects drought), so it captures "how bad is the season," not just today.
- **Range:** 0 to ~110+ (unitless index; higher = more dangerous).
- **Source:** part of the US **National Fire Danger Rating System** (Bradshaw et al. 1983;
  Cohen & Deeming 1985, *The National Fire-Danger Rating System*).

**FM100 — 100-hour dead fuel moisture** *(dynamic)*
- **Meaning:** moisture content of dead woody fuels 1–3 inches thick (they respond to
  weather over ~100 hours). **Lower = drier = more flammable.**
- **Why:** dry larger fuels sustain and intensify fires; a key drought-lag signal.
- **Range:** ~2–30 (percent moisture). *Note it's inverse* — low FM100 is dangerous.
- **Source:** NFDRS; Fosberg & Deeming (1971).

**VPD — Vapor Pressure Deficit** *(dynamic; rising star in fire science)*
- **Meaning:** how "thirsty" the air is — the gap between how much moisture the air *could*
  hold and how much it *does*. High VPD pulls moisture out of fuels and plants.
- **Why:** strongly linked to fire activity in recent research; captures atmospheric dryness
  better than humidity alone.
- **Range:** 0 to ~6 kPa (higher = drier air).
- **Source:** Williams, A.P. et al. (2015), *Correlations between components of the water
  balance and burned area...*; Seager et al. on VPD trends.

**vs — wind speed** *(dynamic)*
- **Meaning:** daily mean 10 m wind speed.
- **Why:** wind supplies oxygen, tilts flames into unburned fuel, and drives spread — the
  core accelerant in fire-behavior physics.
- **Range:** ~0–18 m/s.
- **Source:** gridMET; fire-spread role from Rothermel (1972), *A mathematical model for
  predicting fire spread in wildland fuels*.

**rmax / rmin — max/min relative humidity** *(dynamic)*
- **Meaning:** daily humidity extremes. **rmin** (afternoon low) is the fire-relevant one.
- **Why:** low humidity dries fine fuels within hours (fuels equilibrate to the air).
- **Range:** 0–100 %.

**tmmx — maximum temperature** *(dynamic)*
- **Meaning:** daily high temperature (°C).
- **Why:** heat dries fuels and preheats them toward ignition.

**pr — precipitation** *(dynamic)*
- **Meaning:** daily rainfall (mm).
- **Why:** rain wets fuels and suppresses fire; *absence* over time = drought = danger.

**HRRR forecast weather** *(dynamic, forward-looking)*
- **Meaning:** same kinds of variables (temp, dewpoint→VPD, wind) but **as predicted** for a
  future day, not observed. This is what turns a hindcast into a real **forecast**.
- **Why:** at forecast time you don't know tomorrow's *observed* weather — only the forecast.
  Training/serving on forecast weather avoids a "train/serve mismatch."
- **Detail:** 3 km, forecasts to 48 h; **only exists from 2018-07-12 onward** (HRRRv3). 72 h
  needs GFS (25 km).
- **Source:** Dowell, D.C. et al. (2022), *The High-Resolution Rapid Refresh (HRRR)*,
  Weather and Forecasting.

### The daily Fire-Weather Index (FWI) — full specification

> **Naming caveat:** this is a **custom, pragmatic index built for this project** — NOT the
> official Canadian "Fire Weather Index" System, nor the US NFDRS. It's a simple, transparent,
> *tunable* composite inspired by the **Hot-Dry-Windy Index** (Srock et al. 2018), whose thesis
> is that fire weather is fundamentally *dryness × wind*.

**What it is:** a single 0–1 number per cell per day = "how dangerous is the weather today,"
formed by normalizing three ingredients to ~0–1 and taking a weighted sum:

```
FWI = w_erc · clip(ERC/100)  +  w_vpd · clip(VPD/5)  +  w_wind · clip(wind/12)      (clipped to 0–1)

current weights (scripts/fwi_config.py):   w_erc = 0.30,  w_vpd = 0.30,  w_wind = 0.40
```

| Ingredient | Measures | Normalized by | Role |
|---|---|---|---|
| **ERC** (Energy Release Component) | cumulative fuel dryness (slow drought signal) | ÷ 100 | big-fire potential |
| **VPD** (Vapor Pressure Deficit) | atmospheric "thirst" (acute dryness) | ÷ 5 kPa | pulls moisture from fuels |
| **wind** (10 m speed) | the spread accelerant | ÷ 12 m/s | **heaviest weight** (raised 0.20→0.40 after validation) |

The divisors (100 / 5 / 12) are each variable's approximate high-end, so every term lands ~0–1
before weighting.

**The weights are the tunable knob** (`FWI_WEIGHTS` in `scripts/fwi_config.py`). They began at
`{0.50, 0.30, 0.20}`; the event validation (VALIDATION.md) showed that under-scored wind-driven
fires, so wind was raised to **0.40**. Re-tune in ~8 s via `scripts/15_reweight_fwi.py` (recomputes
from cached components — no gridMET reload).

**Two variants:**
- **Historical** (gridMET, 2024–2026 scrubber): the formula above with daily-**mean** wind.
- **Live forecast** (HRRR, next 24–48 h): HRRR has no ERC, so that weight folds into VPD+wind, and
  it uses **gust** (peak wind ÷ 22) instead of daily-mean wind — afternoon gusts, not averages,
  drive real events (e.g., Smokehouse Creek).

**Where it's used:** FWI is the dynamic "when" multiplier in the **two-stage** design —
`Wildfire risk = static hazard × FWI` and the dashboard's heuristic `Ignition = susceptibility × FWI`.
*(The ML ignition model does NOT use this composite — it ingests the raw HRRR weather variables and
learns its own weighting. FWI is the transparent index behind the wildfire-risk layer and the
validation "WHEN" score.)*

**Honest limitation:** being a linear blend of *daily* variables, FWI is transparent and tunable but
coarser than operational systems (NFDRS / Canadian FWI) that model fuel-moisture dynamics over time.
Right trade-off for a prioritization tool; worth revisiting if this moves toward fire-service use.

---

### 2C. LANDSCAPE / FUELS — the static "how bad if it burns" layer
These change slowly (years), so they're **static** per cell. Together they define wildfire
*hazard*. Primary source: **Texas Wildfire Risk Assessment Portal (TxWRAP)** /
Pyrologix WildEST simulations, and **LANDFIRE**.

**avg_burn_prob — burn probability** *(static)*
- **Meaning:** modeled annual probability that a given cell burns, from thousands of
  simulated fire seasons.
- **Why:** a direct "how fire-prone is this spot" summary.
- **Range:** 0–11 scale (TxWRAP's binned scale, *not* a 0–1 probability).
- **Source:** wildfire simulation à la Finney, M.A. et al. (2011), *A simulation of
  probabilistic wildfire risk components (FSim)*.

**whp — Wildfire Hazard Potential** *(static; the canonical hazard metric)*
- **Meaning:** an index combining burn probability and expected intensity into overall
  hazard. This is *the* standard US hazard layer.
- **Why:** it's the most direct single number for "wildfire risk" in the hazard sense.
- **Range:** 0–9 class scale.
- **Source:** Dillon, G.K. et al. (2015), *Wildfire Hazard Potential for the US*, USFS.

**flep4 — probability flame length > 4 ft** *(static)*
- **Meaning:** chance that, if it burns, flames exceed 4 feet (the threshold where fire
  becomes dangerous to direct suppression).
- **Why:** captures fire *intensity*, not just occurrence. *(Note: only available for the
  317k training cells, not the full 1.7M grid.)*
- **Range:** 0–1. **Source:** TxWRAP WildEST.

**cfl — characteristic flame length** *(static)*
- **Meaning:** typical flame length (feet) expected in that fuel type. Same coverage caveat.
- **Source:** TxWRAP WildEST; fuel models from Scott & Burgan (2005), *Standard fire behavior
  fuel models*.

**cbd — Canopy Bulk Density** *(static)* & **cbh — Canopy Base Height** *(static)*
- **Meaning:** how dense the tree-canopy fuel is (kg/m³) and how high off the ground it
  starts (m). Together they govern **crown fire** potential (fire climbing into treetops).
- **Why:** low CBH + high CBD = surface fire can jump into the canopy = extreme fire.
  (In Texas grassland/shrubland these are mostly ~0 — real, not missing.)
- **Range:** cbd 0–0.45 kg/m³; cbh 0–10 m.
- **Source:** LANDFIRE (LF2022); crown-fire theory from Scott & Reinhardt (2001),
  *Assessing crown fire potential...*.

---

### 2D. GEOGRAPHY — static physical setting
**ecoregion_id — EPA Level-III ecoregion** *(static, categorical)*
- **Meaning:** which named ecological region the cell sits in (e.g., Chihuahuan Deserts,
  Southwestern Tablelands). A proxy for the whole fuel/climate/vegetation regime.
- **Why:** fire behaves very differently across ecoregions; a compact "what kind of place."
- **Source:** Omernik, J.M. (1987), *Ecoregions of the conterminous US*, EPA.

**elevation_m / slope_deg / aspect_deg — topography** *(static)*
- **Meaning:** height above sea level (m); steepness (degrees); compass direction the slope
  faces (0–360°).
- **Why:** fire spreads faster uphill (flames preheat fuel above); **aspect** controls sun
  exposure → south/west faces are drier; elevation sets vegetation/climate zones.
- **Source:** Copernicus GLO-30 Digital Elevation Model (30 m); slope/aspect derived from it.

**road_dist_km — distance to nearest road** *(static; the dominant human-ignition signal)*
- **Meaning:** how far the cell is from the nearest primary/secondary road (km).
- **Why:** most Texas ignitions are **human-caused**, and humans start fires near where they
  are — roads, which correlate with people, vehicles, power lines, activity. This is
  consistently the strongest single ignition predictor.
- **Caveat:** partly a *reporting-bias* signal too (near-road fires get spotted/logged more).
- **Source:** US Census TIGER/Line roads; human-ignition role from Syphard, A.D. et al.
  (2007), *Human influence on California fire regimes*; Radeloff et al. (WUI).

---

### 2E. TEMPORAL — when in the season / week (dynamic, known in advance)
These are **cyclical encodings** — we convert "month" and "day-of-week" into sine/cosine
pairs so the model sees that December is *next to* January (a circle, not 12→1 jump).

- **sin_month / cos_month:** position in the annual cycle → captures **fire season** (Texas
  peaks late winter–spring and summer).
- **sin_dow / cos_dow / is_weekend:** day-of-week → human ignitions rise on weekends
  (recreation) vs weekdays.
- **is_holiday:** July 4th, etc. — spikes in human-caused fire.
- **Why they're "fair" for forecasting:** you always know in advance what month/weekday a
  future date is, so using them doesn't cheat.

---

## 3. How the data comes together (the training table)

Think of building one giant spreadsheet where **each row is a cell-day** and columns are
all the features above plus the label.

**Step 1 — Labels (positives).** From the fused FPA-FOD∪VIIRS inventory: every (cell, day)
that had an ignition → **label = 1**. (~960k positive cell-days.)

**Step 2 — Negatives (the subtle, important part).** We can't use *all* non-fire cell-days
(that's ~1 billion rows, 99.99% negative). Instead we **sample** negatives two ways:
- **Matched/temporal negatives:** the *same fire-prone cells* on days they did *not* ignite.
  This forces the model to learn **"when"** (what's different about the fire day) instead of
  just **"where"** (which cell). This is the key design choice for a real forecast.
- **Spatial controls:** a sample of never-fire cells, so the model still learns the broad
  geography of where fire is possible.

**Step 3 — Attach features.** Join each cell-day to:
- its **static** features (same every day for that cell): geography + landscape/fuels.
- its **dynamic** features for that date: gridMET weather (and, for the operational model,
  HRRR *forecast* weather).
- its **temporal** encodings (month/weekday/holiday).

**Step 4 — Split by time (never randomly).** train = 2014–2021, validate = 2022,
test = **2023–2026** (held out). More on why in §5.

Result: `tdis_train_daily_tx.parquet` — the modeling table.

---

## 4. The two products — ONE is a trained model, the OTHER is an assembled formula

⚠️ **Read this first — they are NOT both machine-learning models.** The dashboard's two
products are built in fundamentally different ways:

| | **4A · Ignition** | **4B · Wildfire (hazard) risk** |
|---|---|---|
| Is it a trained ML model? | **YES — XGBoost, learned from fire labels** | **NO — assembled, nothing is trained** |
| Where the numbers come from | decision trees fit to data | a fire-*simulation* layer × an *equation* |
| Answers | "will a fire **start / be active** here?" | "if a fire occurs, how **dangerous**?" |

Only **4A is an XGBoost model.** 4B contains **no XGBoost, no learned weights, nothing
trained by us** — it is a simulation-derived hazard layer multiplied by a hand-written
fire-weather equation. Keep this distinction sharp; the rest of the handbook depends on it.

### 4A. Ignition — a trained XGBoost model (this is the only ML)

XGBoost = gradient-boosted decision trees (an ensemble of small "if feature > threshold"
trees whose votes sum to a score); chosen because it handles mixed feature types, missing
values, and non-linear interactions, and is the standard strong baseline for tabular risk.

We actually train **three variants of this one model** (same architecture, different inputs):

| Variant | Inputs | Role | Served? |
|---|---|---|---|
| **honest** | static + temporal (**no weather**) | stable susceptibility map → the scrubber's Ignition layer | ✅ (pre-computed) |
| **operational HRRR** | static + temporal + **HRRR forecast weather** | the live 24/48/72 h forecaster | ✅ (live inference) |
| **ceiling** | static + temporal + **observed** same-day weather | diagnostic only — the optimistic bound *if* forecasts were perfect | ❌ never served |

- **Target:** the fused label (fire active that cell-day, 1/0 — see §19 on what "active" means).
- **Output:** a 0–1 **relative score** per cell-day (a ranking, not a calibrated probability — §20A, DEMO §2B).

### 4B. Wildfire (hazard) risk — assembled, NOT trained (no XGBoost)

This product multiplies two non-ML pieces:
- **Static hazard** = a weighted **composite of TxWRAP layers** (WHP + burn probability +
  canopy). TxWRAP itself comes from **FSim — Monte-Carlo fire-*spread simulation*** (Forest
  Service / Texas A&M Forest Service), i.e. a physics/stochastic simulation, **not a learned
  model** and not built by us. It changes slowly (fuels/terrain), so there is nothing to
  "train" day to day.
- **× daily FWI** = the **fire-weather equation** (Composite / NOAA HWP / TX HWP — §16), a
  hand-written arithmetic formula of that day's weather. On a calm humid day a hexagon is
  low; on a hot/dry/windy day it lights up. This is the two-stage WFPI-style product.

`Wildfire risk = (simulation hazard) × (weather equation)` — every term is either an external
simulation or an equation. **No gradient boosting, no training, no labels anywhere in it.**
(The one asterisk: the **TX HWP** *variant* of the FWI had its coefficients fit to VIIRS fire
data — so if you select that formula, a small regression touches 4B; the default Composite
and the NOAA HWP variant are pure equations.)

**Bottom line:** **4A is the machine-learning model** (trained on labels, in honest /
operational / ceiling variants); **4B is a simulation × equation product** with no ML. They
share the same daily weather driver but are built by completely different machinery — which
is why a *label* problem (e.g. the flare fix, §VALIDATION 9) re-baselines 4A but leaves 4B
untouched.

---

## 5. How forecasting actually works (and why our evaluation is honest)

**The golden rule of forecasting evaluation: train on the past, test on the future.** If you
shuffle rows randomly, the model can "peek" at the future and you fool yourself. We instead:
- Train on **2014–2021**, tune on **2022**, and **never touch 2023–2026 until the final test.**
- That simulates deployment: "given only history, can it predict years it has never seen?"

**Two disciplines that make it a genuine forecast (not a leaky hindcast):**
1. **Temporal split** (above).
2. **No same-day-weather leak** in the deployable model — because at 6am when you issue a
   forecast for tomorrow, you don't yet know tomorrow's *observed* weather; you only have a
   *forecast* of it. So the honest model uses only what's knowable in advance (static +
   temporal), and the operational model uses **HRRR forecast** weather.

**Result on unseen years (2023–2026):** see `CLASSIFICATION_METRICS.md` for the current,
authoritative numbers (this paragraph previously carried early-run values). Headline for
the served operational model: test AUC-PR 0.4825 / best F1 0.488 on the balanced sample;
AUC-PR 0.0851 / best F1 0.165 / **lift 4.4×** on the real full population. (AUC-PR = area
under the precision–recall curve; for rare events, the honest headline number is the
**lift over random**, because AUC-PR itself scales with how common the positive class is.)

### Mechanics: what a daily forecast actually computes (inference, not retraining)

The model file on disk is ~400 decision trees whose split questions
("hrrr_vpd > 1.85?", "road_dist_km < 3.0?") and leaf values were chosen
ONCE, during training. A daily forecast never changes them. What happens
each day, concretely (`scripts/13_model_forecast_day.py`):

1. **Build the feature matrix** — one row per cell (1.7M rows), with the
   exact 20 columns the model was trained on, side by side (this is
   column concatenation, not any arithmetic):
   - 11 static columns: copied unchanged from `tx_static_master.parquet`
     — identical every day.
   - 6 calendar columns: computed from the target date (e.g. Aug 19 →
     sin_month = sin(2π·8/12), is_weekend = 0).
   - 3 weather columns: filled with the NEW HRRR forecast values for the
     target day — literally `st['hrrr_vpd'] = g['vpd'].values[idx]`,
     nearest forecast-grid-point per cell.
2. **Run inference** — `model.predict_proba(features)`. Each row walks
   down every one of the ~400 frozen trees, answering the stored yes/no
   questions with its own values; the ~400 leaf values it lands on are
   summed and squashed to a 0–1 score. Comparisons and additions of
   stored numbers — nothing is learned, fitted, or adjusted.
3. **Calibrate** — the isotonic calibrator maps the raw score (a relative
   rank, ~17× inflated by the rebalanced training sample) to a real
   probability (`ign_cal`).

Training chose the questions; inference just answers them with new data.
Tomorrow's weather changes which *paths* rows take through the trees —
that is the entire mechanism by which a new forecast produces a new map.
Nothing is ever multiplied onto the model's output (tested and rejected —
see `METHODOLOGY.md`, retired-approaches table).

### What features matter for forecasting? (measured, not guessed)
From the trained models' importance scores:

**Ignition (honest) — top drivers:**
| Feature | Importance | Interpretation |
|---|---|---|
| elevation_m | 0.16 | vegetation/climate zone |
| ecoregion_id | 0.12 | overall fire regime of the region |
| avg_burn_prob | 0.10 | modeled fire-proneness |
| road_dist_km | 0.08 | human-ignition proximity |
| cfl / whp | 0.08 / 0.06 | fuel intensity/hazard |
| slope/aspect | 0.07 / 0.04 | topography |
| sin/cos_month | ~0.09 combined | fire season |

**Ignition (ceiling, +weather):** `rmin` (afternoon humidity, 0.13) and `vpd` (0.07) jump to
the top — confirming weather *does* carry signal, but note the honest→ceiling accuracy gain
is only ~+0.03 AUC-PR. **Takeaway:** *where* (terrain, ecoregion, fuels, roads) dominates;
weather is a real but modest daily modulator. This matches the wildfire literature for
human-dominated fire regimes like Texas.

---

## 6. From model → dashboard (the last mile)

How a trained model becomes the colored hexagons you click:

1. **Score.** Run the model (or the hazard×FWI product) for every cell → a 0–1 risk number
   per cell (per day for dynamic layers).
2. **Aggregate to res-5.** Average res-8 scores up to their res-5 parent hexagons (~5,400
   cells) so the map is light enough for a browser. *(scripts 08 & 09.)*
3. **Package as compact JSON.** Per cell: id, centroid lat/lon, ecoregion, static hazard,
   ignition baseline, and a daily fire-weather array. Live forecasts (HRRR) are separate
   small JSONs. *(scripts 08/09/10.)*
4. **Render.** The HTML dashboard (`tdis_fire_dashboard_standalone.html`) uses **Leaflet**
   (the map) + **h3-js** (draws each hexagon's boundary from its ID) and colors each hex on a
   heat ramp by its risk value. A day-scrubber indexes into the daily arrays; the
   live-forecast toggle swaps in the HRRR forecast values.
5. **Interact.** Mode toggle (Wildfire Risk / Ignition / Static Hazard), horizon selector
   (24h / 48h live forecast), tooltips that decompose each hex into hazard × fire-weather.

Data flow in one line:
`gridMET/VIIRS/FPA-FOD/HRRR → feature table → XGBoost + hazard composite → per-cell scores → res-5 JSON → Leaflet/h3-js hex map.`

---

## 7. Worked example (tying it together)

Take one hexagon near Abilene on a hot, dry, windy day in March 2026:
- **Static:** grassland (ecoregion), flat (low slope), 6 km from a road, moderate WHP (5/9),
  burn prob 6/11 → decent baseline hazard.
- **Weather that day:** ERC 85 (very high), VPD 3.8 kPa (very dry air), wind 9 m/s (strong),
  FM100 6% (bone dry) → **FWI ≈ 0.8** (extreme fire weather).
- **Wildfire risk (forecast)** = hazard(≈0.5) × FWI(0.8) = **high** → "if a fire starts here
  today it could be severe."
- **Ignition risk** = ML susceptibility (elevated: near-ish a road, fire-prone ecoregion,
  fire season) × FWI(0.8) = **elevated** → "and a fire is relatively likely to start."
- On a **rainy day** the same hexagon: FWI ≈ 0.15 → both products drop to low. Same place,
  different day — that's forecasting.

---

## 7B. Real worked calculations — the four validation fires

The example above is illustrative; here is the **actual arithmetic on real stored data** for the
four documented wildfires we validate against (see **`VALIDATION.md`** for the full event write-up,
methodology, and the wind-knob before/after — some of it is repeated here so this section stands
alone). Formula: **FWI = 0.30·ERC/100 + 0.30·VPD/5 + 0.40·wind/12**, then
**Wildfire risk = static hazard × FWI**.

**Smokehouse Creek — 2024-02-26** (Panhandle; largest fire in TX history; wind-driven)
```
day weather: ERC 51, VPD 0.78 kPa, wind 7.3 m/s
FWI = 0.30·(51/100) + 0.30·(0.78/5) + 0.40·(7.3/12)
    = 0.153        + 0.047         + 0.244          = 0.443
Wildfire risk = hazard 0.568 × FWI 0.443 = 0.252
```
→ Wind term (0.244) dominates even though VPD was tiny (a *cold* dry-windy day) — the wind knob at work.

**Crabapple — 2025-03-15** (Hill Country; hot-dry-windy / red-flag)
```
day weather: ERC 71, VPD 1.33 kPa, wind 7.4 m/s
FWI = 0.214 + 0.080 + 0.246 = 0.540   →   risk = 0.477 × 0.540 = 0.258
```
→ Drought (high ERC) + wind → top-tier fire weather.

**Lavender — 2026-02-17** (Panhandle; wind-driven) — the highest-risk of the four
```
day weather: ERC 55, VPD 1.49 kPa, wind 8.9 m/s
FWI = 0.165 + 0.089 + 0.298 = 0.552   →   risk = 0.587 × 0.552 = 0.324
```
→ Strongest wind term (0.298) × highest static hazard (0.587). Jumped 80th→94th pctile with the wind knob.

**Hunggate — 2026-05-14** (Panhandle; lightning-caused)
```
day weather: ERC 79, VPD 3.16 kPa, wind 6.0 m/s
FWI = 0.237 + 0.190 + 0.202 = 0.628   →   risk = 0.400 × 0.628 = 0.251
```
→ Highest FWI of all (extreme drought + dry air); lower local hazard keeps overall risk mid. Caught on *weather* alone despite being a natural (lightning) ignition.

**What the four show:** every fire lands at an elevated FWI (0.44–0.63 vs a statewide-typical ~0.35–0.40),
and the two FWI ingredients trade off exactly as physics says — Smokehouse/Lavender carried by **wind**,
Crabapple/Hunggate by **dryness**. Full percentile scores and the wind-knob before/after are in `VALIDATION.md`.

---

## 7C. How a risk number becomes a map color (the legend)

The dashboard paints a continuous 0–1 heat ramp; the six labeled legend bands are its color stops:

| Band | Color-scale value |
|---|---|
| **Extreme** | ≥ 0.88 |
| **Very high** | 0.72–0.88 |
| **High** | 0.55–0.72 |
| **Elevated** | 0.40–0.55 |
| **Moderate** | 0.25–0.40 |
| **Low** | < 0.25 |

**Important — the value is *stretched* before coloring**, so the map isn't uniformly dark (raw risk
rarely exceeds ~0.5). The stretch depends on the mode:
- **Wildfire Risk:** color-value = risk ÷ 0.5
- **Ignition (live model):** ÷ 0.6
- **Ignition (heuristic):** ÷ 0.4

Example: Lavender's wildfire risk 0.324 → 0.324/0.5 = **0.65 → "High"**; Smokehouse's 0.252 → 0.50 →
top of **"Elevated."** So the bands are **relative tiers** — "Extreme" = among the most dangerous
cell-days on the current map, "Low" = benign — **not absolute probabilities.** (Consistent with the
whole system's "relative-risk ranking, not literal %" framing.)

---

## 8. Honest limitations (a hero knows the weak spots)
- **Ignition daily skill is modest** — the honest ML model has little day-to-day signal on
  its own; daily motion comes from the ×FWI step. Weather adds only ~+0.03 AUC-PR.
- **AUC-PR isn't comparable across models** with different positive rates — use **lift**.
- **Labels have reporting bias** (near-road fires over-represented); satellite labels miss
  small fires. Both inherited from the sources, not fixable in modeling.
- **Live forecast uses VPD+wind only** (HRRR carries no ERC) — a slightly different FWI recipe
  than the gridMET layers.
- **This is a preliminary baseline** — the full-coverage dataset and forecast-weather-trained
  model are still being built.

---

## 9. The processing pipeline (script by script)
| Script | Turns … into … |
|---|---|
| `01_download_viirs.py` | FIRMS API → VIIRS detections on H3 cells |
| `02_build_labels.py` | FPA-FOD + VIIRS → daily ignition inventory (positives) |
| `03_build_dataset.py` | labels + matched negatives + features → training table |
| `04_build_full_tx_static.py` | rasters/APIs → full-TX static feature master (1.7M cells) |
| `05_download_hrrr_forecast.py` | HRRR/GFS → historical *forecast* weather |
| `07_train_baseline_forecast.py` | training table → ignition models (honest + ceiling) |
| `08_build_dashboard_data.py` | model + hazard → dashboard JSON (hazard + ignition) |
| `09_build_perday_dynamic.py` | gridMET → daily fire-weather arrays (2024–2026 scrubber) |
| `10_live_forecast_day.py` | live HRRR → a real forward forecast for a target day |

---

## 10. Glossary / cheat-sheet
- **Cell-day** — one hexagon on one day (a data row).
- **Static feature** — fixed per cell (terrain, fuels, roads).
- **Dynamic feature** — changes by day (weather, season).
- **Ignition risk** — chance a fire *starts*. **Wildfire/hazard risk** — how bad if it burns.
- **ERC / FM100 / VPD** — fire-weather dryness indices (high ERC/VPD, low FM100 = dangerous).
- **WHP** — Wildfire Hazard Potential (the standard static hazard index).
- **FWI (ours)** — 0–1 daily fire-weather danger = 0.5·ERC + 0.3·VPD + 0.2·wind (normalized).
- **XGBoost** — the tree-ensemble ML model used.
- **AUC-PR** — accuracy metric for rare events; compare via **lift over random**, not raw value.
- **Temporal split** — train on past, test on future (honest forecast evaluation).
- **Two-stage** — forecast risk = static susceptibility × dynamic fire-weather.
- **H3** — the hexagon grid system; res-8 (~0.74 km²) for modeling, res-5 for display.
- **HRRR / GFS** — weather forecast models (HRRR 3 km ≤48 h; GFS 25 km, longer range).

---

## 11. Key references (for deeper study)
- Abatzoglou (2013) — gridMET dataset.
- Short (2014) — FPA-FOD fire occurrence database.
- Schroeder et al. (2014) — VIIRS 375 m active fire.
- Dowell et al. (2022) — HRRR model.
- Cohen & Deeming (1985) — National Fire Danger Rating System (ERC, FM100).
- Rothermel (1972) — wildland fire spread model (wind/slope).
- Williams et al. (2015) — VPD and burned area.
- Srock et al. (2018) — Hot-Dry-Windy Index.
- Finney et al. (2011) — FSim burn-probability simulation.
- Dillon et al. (2015) — Wildfire Hazard Potential.
- Scott & Burgan (2005) — standard fire behavior fuel models.
- Scott & Reinhardt (2001) — crown fire potential (CBD/CBH).
- Omernik (1987) — EPA ecoregions.
- Syphard et al. (2007) — human influence on fire regimes (roads/ignitions).

---

## 12. Why this system beats the previous (v5) model

**Be precise: it's better as a *capability*, not mainly as a raw predictor.**

| Dimension | Old v5_tuned | This TDIS system | Why it matters |
|---|---|---|---|
| What it does | static susceptibility *ranking* | **forward forecast (24/48/72h)** | v5 couldn't predict a specific future day |
| Labels | FPA-FOD only, ends 2020, 87% imputed times | **+ VIIRS satellite** (real times, through 2026) | we can finally *see* recent fires |
| Validation | held-out 2019–2020 | **tested on 2023–2026 it never saw** | proves forward generalization |
| Coverage | 317K cells (~⅓ of TX) | **full 1.7M-cell Texas** | v5 silently missed ⅔ of the state |
| Live forecast | impossible | **real HRRR/GFS pull for tomorrow** | actually operational |

**Honest caveat on the metric:** the new AUC-PR (0.63–0.70) looks far above v5's 0.37, but a
big chunk is a **base-rate artifact** (AUC-PR scales with the positive fraction, which differs
across datasets). By base-rate-robust **lift**, the two are comparable. Fair statement:
*"a far more capable, honestly-validated forecast **system**, built on comparable signal."*
The single biggest unlock was **VIIRS labels** — real timestamps + coverage past 2020, which
is what made forward forecasting and forward *validation* possible at all.

---

## 13. Why v5's HRRR attempt failed, but ours works

v5 *did* try HRRR — and it **hurt** performance (AUC-PR dropped ~0.32→0.31). Here's why the
same data source failed then and helps now. This is the most instructive lesson in the project:

1. **Wrong role.** v5 bolted HRRR on as just another *observed* weather feature inside a
   **static susceptibility ranker** — a model answering "*which cells* are fire-prone," not
   "*which day* will it ignite." Weather is a *temporal* signal; feeding it to a spatial-ranking
   task adds noise, not skill. (v5's QC showed 92% of its variance was spatial.)
2. **Coverage gaps → NaN noise.** HRRR only existed from ~2016 in v5's training window, so
   ~46% of rows had missing HRRR → the model learned to ignore it. We fixed this by
   **restricting to the HRRR-covered window (2018-07+)** so every training row has real
   forecast weather — no NaN-gap problem.
3. **No forward labels to even test it.** v5's labels ended 2020, so there was *no way to
   measure forecast skill* — you couldn't tell if HRRR helped forecasting because there was no
   forward test. **VIIRS** (2021–2026 labels) gave us a real future to validate on and to run a
   clean **ablation**.
4. **Sampling didn't force temporal learning.** v5's negatives were essentially spatial, so
   even with weather present the model had no reason to learn "when." Our **matched negatives**
   (same fire-prone cells on non-fire days) force a "when" contrast — giving HRRR something to
   actually contribute.

**In one line:** v5 used HRRR as an *observed feature in a static, gap-riddled, un-validatable
spatial model*, so it was noise. We use HRRR as *forecast weather in a temporal-forecast
framework with clean coverage, matched negatives, and real forward validation* — the role it
was built for. The payoff (a clean +0.011 AUC-PR / +0.020 AUROC) is modest but real and,
crucially, **measurable** for the first time.

---

## 14. Model architecture

**Core learner: XGBoost** (eXtreme Gradient Boosting) — an *ensemble of decision trees*:
- A single decision tree asks a chain of "is feature > threshold?" questions to reach a risk
  estimate. One tree is weak; XGBoost builds **hundreds of them sequentially**, each new tree
  correcting the errors the previous ones made (that's the "gradient boosting"). Their outputs
  sum, passed through a sigmoid → a 0–1 probability.
- **Why XGBoost here:** it handles mixed feature types, learns non-linear interactions (e.g.,
  "dry *and* windy *and* near a road"), **handles missing values natively** (important — our
  weather has gaps), is fast, and is the standard strong baseline for tabular risk problems.

**Key settings (operational model):** 600 trees, max_depth 6, learning_rate 0.05,
subsample 0.8, colsample_bytree 0.8, min_child_weight 20, `scale_pos_weight` (to counter class
imbalance — fires are rare), early-stopping on the 2022 validation set, `tree_method='hist'`,
`device='cuda'` (trained on the RTX A6000 GPU).

**The system is two-stage** (not one model):
```
                    ┌─ Ignition:  XGBoost(static + temporal + HRRR forecast weather)  → P(fire starts)
Forecasting layer ──┤
                    └─ Wildfire risk:  static hazard composite  ×  daily fire-weather index
```
- The **ignition** side is the learned XGBoost model (this is where HRRR forecast weather feeds in).
- The **wildfire-risk** side is a composite hazard layer (WHP + burn-prob + canopy) multiplied
  by the fire-weather index — the WFPI-style "potential × weather" product.
- Both share the same daily weather driver but answer different questions (start vs. severity).

**Output path:** model/composite → per-cell 0–1 score → aggregate H3 res-8 → res-5 → JSON →
Leaflet + h3-js hex map.

---

## 15. Feature list at a glance (quick reference)

*(Full detail with citations in §2. This is the cheat-sheet.)*

| Feature | What it is | What it does (general) | Role in model | Source |
|---|---|---|---|---|
| **label** | fire started that cell-day (1/0) | the target | what we predict | FPA-FOD + VIIRS |
| erc | Energy Release Component | cumulative fuel dryness / potential energy | dynamic weather | gridMET (NFDRS) |
| fm100 | 100-hr dead fuel moisture | larger-fuel dryness (low=dry) | dynamic weather | gridMET (NFDRS) |
| vpd / hrrr_vpd | vapor pressure deficit | atmospheric "thirst"; top weather signal | dynamic weather (obs / forecast) | gridMET / HRRR |
| vs / hrrr_wind | wind speed | oxygen + flame tilt → spread | dynamic weather | gridMET / HRRR |
| rmin/rmax | min/max humidity | fine-fuel drying | dynamic weather | gridMET |
| tmmx / hrrr_tmp | max temperature | dries & preheats fuels | dynamic weather | gridMET / HRRR |
| pr | precipitation | wets fuels; absence=drought | dynamic weather | gridMET |
| **avg_burn_prob** | modeled burn probability | how fire-prone the spot is | **static, top-4** | TxWRAP (FSim) |
| **whp** | Wildfire Hazard Potential | standard hazard index | **static, hazard layer** | USFS (Dillon) |
| flep4 | P(flame length > 4 ft) | fire intensity | static (317K cells only) | TxWRAP WildEST |
| cfl | characteristic flame length | fire intensity | static | TxWRAP WildEST |
| cbd / cbh | canopy bulk density / base height | crown-fire potential | static | LANDFIRE |
| **ecoregion_id** | EPA Level-III region | the area's fire regime | **static, top-2** | EPA (Omernik) |
| **elevation_m** | height (m) | vegetation/climate zone | **static, top feature** | Copernicus DEM |
| slope_deg / aspect_deg | steepness / facing | uphill spread; sun-dried aspects | static | Copernicus DEM |
| **road_dist_km** | distance to nearest road | human-ignition proximity | **static, top-4** | Census TIGER |
| sin/cos_month | annual cycle | fire season | dynamic temporal | derived |
| sin/cos_dow, is_weekend, is_holiday | week cycle / holidays | human-activity timing | dynamic temporal | derived |

**The pattern (bolded rows):** the model's heaviest hitters are the **static "where" features**
— elevation, ecoregion, burn probability, road distance. Weather (led by VPD) is a **real but
modest** "when" modulator. That's the empirical signature of a human-dominated fire regime.

---

## 16. The fire-weather formula knob — three switchable formulations

As of 2026-08-05 the fire-weather index that drives both dashboard products
(`risk = hazard × FWI`, `ignition = susceptibility × FWI`) is **selectable** — a dashboard
toggle ("Fire-weather formula") and a config knob (`FWI_MODE` in `scripts/fwi_config.py`).
The ML model is untouched by all of this; only the interpretable weather layer changes.

### The three variants

**1. Composite (default)** — our original additive index (§2), wind-tuned after validation:
```
FWI = 0.30·ERC/100 + 0.30·VPD/5 + 0.40·wind/12
```
Factors *add*: a windy day scores even if fuels are damp. Simple, interpretable, and the
best overall performer on the 4-event battery (below).

**2. NOAA HWP** — the **Hourly Wildfire Potential** from NOAA GSL, implemented EXACTLY
(James et al. 2025, *Weather and Forecasting*, doi:10.1175/WAF-D-24-0068.1, their Eq. 3):
```
HWP = 0.213 · G^1.50 · VPD^0.73 · (1−M)^5.10        [· snow term, =1 in TX]
  G = 10-m wind gust (m/s, floor 3) · VPD in hPa · M = soil-moisture availability (0-1)
```
Factors *multiply*: any wet factor suppresses the whole index (soaked soil kills fire
potential regardless of wind) — physically appealing, and the reason humid East Texas goes
dark under this variant while the Panhandle glows.
- **Live forecasts use it exactly**: real HRRR gust + real HRRR `MSTAV` soil moisture.
- **The historical scrubber needs proxies** (gridMET has neither gust nor soil moisture):
  G ≈ 1.5 × daily-mean wind, M ≈ fm100/30. Labeled as such; treat historical HWP as approximate.

**3. TX HWP** — the same power-law form, but coefficients **re-fit to Texas satellite fire
activity** (`scripts/16_fit_hwp_tx.py`): log-linear least squares of daily VIIRS FRP per
res-5 cell-day against (G, VPD, dryness), restricted to **large-fire episodes** (240 cells
whose peak day ≥ 500 MW, ±7-day windows; 50,893 cell-days) — mirroring the paper's choice
to train on large wildfires. Result:
```
TX HWP = 17.6 · G^0.05 · VPD^0.05 · dry^0.92        (r_log ≈ 0.10)
```
**The fit itself is a finding:** at *daily* resolution, Texas fire activity tracks **fuel
dryness almost exclusively** — the wind and VPD exponents collapse to ~0. NOAA's r = 0.44
was achieved at *hourly* resolution with real gusts at active fires; average weather to a
day and the wind–activity coupling washes out (the same daily-mean-wind blindness that
muted Smokehouse in §7B). So TX HWP is effectively a *fuel-cured-ness map* — useful as a
dryness lens, weak on wind-driven events. Parameters live in `data/hwp_params.json`.

### Head-to-head on the 4 validation fires (WHEN percentile, method of VALIDATION.md)

| Event (driver) | Composite | NOAA HWP | TX HWP |
|---|---|---|---|
| Smokehouse Creek (wind, cold) | 75th | **80th** | 75th |
| Crabapple (dry + wind) | 99th | 99th | **100th** |
| Lavender (wind) | **94th** | 85th | 64th |
| Hunggate (lightning, heat) | **99th** | 97th | 95th |

**Read:** the wind-tuned **composite remains the default** (best or tied on 3/4). NOAA HWP
is the strongest alternative — it wins Smokehouse (its soil-moisture term sees the drought)
and its live-path version runs on exact inputs, so expect it to do better operationally
than this proxy-based historical replay suggests. TX HWP confirms the dryness-dominance
finding but under-scores wind events.

### How to tune / re-fit
```bash
# switch the primary formulation for scripts:
#   edit FWI_MODE in scripts/fwi_config.py  ('composite' | 'hwp_noaa' | 'hwp_tx')
# re-fit TX coefficients (e.g., after new VIIRS years):
$PY scripts/16_fit_hwp_tx.py
# rebuild all 3 per-day arrays from the cached components (~30 s, no gridMET reload):
$PY scripts/15_reweight_fwi.py
# re-embed the drag-and-drop dashboard:
$PY scripts/17_embed_standalone.py
```
The dashboard toggle needs no rebuild at all — all three arrays ship in the file
(the standalone grew 25 → 61 MB to carry them).

---

## 17. Descriptive statistics of every dataset + source links

Computed directly from the stored files (2026-08-05). Reproduce any row by loading the
parquet and calling `.describe()`.

### 17A. Fire labels
| Dataset | Rows | Coverage | Key numbers |
|---|---|---|---|
| Fused ignitions (`data/labels_fused/ignitions_daily_tx.parquet`) | **960,054** positive cell-days | 281,646 res-8 cells, 2014-01-01 → 2026-07-29 | ~490k post-2020 (VIIRS-only era) |
| VIIRS detections (`data/labels_viirs/viirs_tx_h3.parquet`) | **1,619,215** detections | 2014–2026, 375 m, real timestamps | FRP 0–1,677 MW; ⚠️ persistent-hotspot (gas-flare) contamination exists — see DEMO_SMOKEHOUSE §2B |

### 17B. Training table (`tdis_train_daily_tx.parquet`)
- 3,595,513 rows (cell-days) × 30 cols · 619,218 cells · 2014→2026 · label rate 0.267 (matched-negative design — **synthetic**, not the real-world rate ~0.3%/day at res-5)
- Weather ranges: ERC 0–112 (mean 38) · VPD 0–5.6 kPa (mean 1.25) · wind 0–17.7 m/s (mean 4.2) · fm100 2–28.9% (mean 13.9) · tmmx −53*–47°C (*≈3% fill-value artifact rows, QC item)
- Known QC items: weather null on 77% of rows (gridMET universe ⊂ full grid); 86k phantom 2026 test rows after last-label date (fix queued)

### 17C. Per-day dynamic components (`data/fwi_components_res5.parquet`) — drives the dashboard
3,381,192 cell-days · 3,624 res-5 cells × 933 days (2024-01-01 → 2026-07-21):

| Var | mean | std | p5 | median | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| ERC | 44.2 | 18.5 | 17.3 | 42.0 | 78.6 | 91.8 | 113 |
| VPD (kPa) | 1.33 | 0.76 | 0.28 | 1.21 | 2.77 | 3.48 | 5.83 |
| wind (m/s) | 4.22 | 1.54 | 2.09 | 4.02 | 7.03 | 8.72 | 16.4 |
| fm100 (%) | 12.8 | 4.0 | 6.1 | 12.9 | 19.1 | 21.4 | 27.7 |

Resulting index distributions (why percentiles, not raw values, are the comparable unit):

| Index | mean | median | p95 | p99 |
|---|---|---|---|---|
| Composite | 0.353 | 0.344 | 0.536 | 0.616 |
| NOAA HWP | 0.102 | **0.040** | 0.415 | 0.815 |
| TX HWP | 0.650 | 0.646 | 0.903 | 0.979 |

### 17D. Static features (`data/static_features/tx_static_master.parquet`)
1,708,940 res-8 cells, 0 nulls. Zero-fraction caveats: flep4/cfl 100% zero (dead features);
cbd/cbh zero on 87% of cells; whp/burn-prob zero on 54% (non-burnable/urban/water).
Dashboard hazard (res-5): mean 0.194, max 0.739.

### 17E. Data source links (all free & public)
| Source | Role | Link |
|---|---|---|
| FPA-FOD (6th ed.) | ignition inventory 1992–2020 | https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6 |
| NASA FIRMS / VIIRS | real-timestamp fire labels 2014– | https://firms.modaps.eosdis.nasa.gov/ (bulk: /download/, key: /api/map_key/) |
| gridMET | observed daily weather 4 km | https://www.climatologylab.org/gridmet.html |
| NOAA HRRR | forecast weather 3 km ≤48 h (+ gust, MSTAV) | https://registry.opendata.aws/noaa-hrrr-pds/ |
| NOAA GFS | 72 h forecast fallback | https://registry.opendata.aws/noaa-gfs-bdp-pds/ |
| TxWRAP | TX wildfire hazard/burn probability | https://www.texaswildfirerisk.com/ |
| LANDFIRE | canopy CBD/CBH, fuels | https://landfire.gov/ |
| Copernicus GLO-30 | elevation/slope/aspect | https://registry.opendata.aws/copernicus-dem/ |
| Census TIGER 2022 | roads (human-access proxy) | https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html |
| EPA Level-III ecoregions | fire-regime zones | https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states |
| NOAA HWP paper | fire-weather benchmark | https://doi.org/10.1175/WAF-D-24-0068.1 |

---

## 18. How the training database was built — step by step, with code links

Every step is a numbered script in [`scripts/`](scripts/); rerun any step to reproduce its
output. (Statistics below computed 2026-08-05.)

**Step 1 — Fire labels** · [`scripts/01_download_viirs.py`](scripts/01_download_viirs.py) → [`scripts/02_build_labels.py`](scripts/02_build_labels.py)
- Download VIIRS active-fire detections (NASA FIRMS API, 5-day windows, per-year) →
  1,619,215 TX detections 2014–2026, assigned to H3 res-8 cells.
- Fuse with FPA-FOD (2014–2020 inventory): a cell-day is a **positive** if either source
  fired. Result: **960,054 positive cell-days** / 281,646 cells.
- **Label composition (know your label!):** 96.2% satellite-only · 3.8% FPA-FOD ·
  0.1% corroborated by both · known cause only 2.7% (human 16,683 / equipment 7,485 /
  lightning 1,772) · median known fire size **2 acres**.
- **Known defects & fixes:** ~18–27% of positives come from persistent industrial hotspots
  (gas flares — 531 cells fire >3% of all days; worst cell 3,176 days). Filtered variant +
  flare-cell list built by [`scripts/18_flare_filter_retrain.py`](scripts/18_flare_filter_retrain.py).
- Temporal reality checks that pass: March peak (122,988 rows) matches TX fire season;
  Sunday is the *lowest* day (fires follow the agricultural/industrial work week).

**Step 2 — Static features (the WHERE)** · [`scripts/04_build_full_tx_static.py`](scripts/04_build_full_tx_static.py)
- Rasterize/lookup per res-8 cell: TxWRAP WHP + burn probability, LANDFIRE CBD/CBH,
  GLO-30 elevation/slope/aspect, EPA ecoregion, TIGER road distance.
- Result: `data/static_features/tx_static_master.parquet` — **1,708,940 cells, 0 nulls**
  (see §17D for zero-fraction caveats: flep4/cfl dead, cbd/cbh 87% zero).

**Step 3 — Assemble the training table** · [`scripts/03_build_dataset.py`](scripts/03_build_dataset.py)
- Rows = positives + **matched negatives** (same fire cells on non-fire days — forces the
  model to learn WHEN, not just WHERE) + a spatial-control sample of never-fire cells.
- Attach gridMET daily weather (8 vars), static features, temporal encodings
  (sin/cos month & day-of-week, weekend, holiday).
- Result: `tdis_train_daily_tx.parquet` — **3,595,513 cell-days × 30 cols**, label rate
  0.267 (**synthetic** — real-world res-5 rate is ~0.3%/day; scores are ranks, not
  probabilities until the calibration pass).
- Filtered variant (flares + phantom-date rows removed):
  `tdis_train_daily_tx_flarefiltered.parquet` (step 18).

**Step 4 — Forecast-weather variant** · [`scripts/05_download_hrrr_forecast.py`](scripts/05_download_hrrr_forecast.py) → [`scripts/06_attach_hrrr_forecast.py`](scripts/06_attach_hrrr_forecast.py)
- Download what HRRR *actually forecast* 24 h ahead for every label date (F24 exists only
  from 2018-07-12, HRRRv3): ~2,935 daily files, **~14 GB** in `data/weather_hrrr_forecast/`.
- Attach to training rows → `tdis_train_daily_hrrr.parquet` (hrrr_tmp/vpd/wind; null on
  38.7% of rows = the pre-2018-07 era).

**Step 5 — Train / evaluate** · [`scripts/07_train_baseline_forecast.py`](scripts/07_train_baseline_forecast.py), [`scripts/11_train_hrrr_forecast.py`](scripts/11_train_hrrr_forecast.py), [`scripts/12_hrrr_ablation.py`](scripts/12_hrrr_ablation.py)
- Temporal split: train 2014–2021 · val 2022 · test **2023–2026 (unseen future)**.
- XGBoost (600 trees, depth 6, lr 0.05, scale_pos_weight, early stopping, GPU).
- Models in [`models/`](models/): honest / ceiling / operational-HRRR (+ flare-filtered).

**Step 6 — Serve** · [`scripts/08`](scripts/08_build_dashboard_data.py)–[`09`](scripts/09_build_perday_dynamic.py) (dashboard data), [`13`](scripts/13_model_forecast_day.py) (live forecast), [`15`](scripts/15_reweight_fwi.py)–[`17`](scripts/17_embed_standalone.py) (FWI knobs & embed)
- Dashboard: [`dashboard/tdis_fire_dashboard_standalone.html`](dashboard/tdis_fire_dashboard_standalone.html) (drag-and-drop).

**Full pipeline map:** label scripts (01–02) → statics (04) → table (03, 06) → models
(07, 11, 12, 18) → serving (08, 09, 13, 15–17) → validation (14) → docs
([`README.md`](README.md), [`VALIDATION.md`](VALIDATION.md), [`DEMO_SMOKEHOUSE.md`](DEMO_SMOKEHOUSE.md)).

---

## 19. Is the fused label valid? (satellite detections + agency records)

Our label fuses two sources that measure different things — VIIRS ("a satellite saw heat
here today," 96.2% of positives) and FPA-FOD ("an authority recorded a wildfire ignition,"
3.8%) — and they barely overlap (1,169 corroborated rows). Is that fusion defensible?

**Yes — the low overlap is the expected, literature-documented behavior, and fusion is the
standard remedy:**

- **Fusco et al. (2019, *Remote Sensing of Environment*)** compared MODIS satellite
  detections against agency fire records across the conterminous US: satellites detected
  **<25%** of agency-recorded fires; agencies recorded **<50%** of satellite-detected
  fires. Each source has systematic blind spots (satellites: small/short/cloud-obscured
  fires — 50% detection probability only at ~10–78 ha; agencies: jurisdiction/reporting
  gaps). The two databases capture **complementary populations of fires** — fusing them
  offsets biases neither can fix alone.
- **Satellite active fire as an ML label is field-standard practice:** Google's Next Day
  Wildfire Spread benchmark (Huot et al. 2022) uses VIIRS detections as its target, as do
  TS-SatFire (2025) and the California VIIRS fire-tracking dataset (*Scientific Data* 2022).
  Satellite-*derived ignition* products are themselves published datasets (Global Fire
  Atlas, Andela et al. 2019).
- **FPA-FOD-only is the other standard** (Short 2014) with its own known costs: ends 2020,
  87% of times imputed.

**The honest terminology this implies:** strictly, the fused label marks **"fire ACTIVE in
this cell on this day" (fire occurrence)**, not "fire ignited here today." At daily
resolution the two nearly coincide for the small fires that dominate Texas (median known
size 2 acres — ignite and die within a day), but a multi-day fire contributes several
positive cell-days from one ignition. So the "Ignition" product is precisely: **fire-activity
risk** — for identifying vulnerable areas this is arguably the *more* useful target (an
area with active fire tomorrow is dangerous regardless of which day it ignited). Docs and
dashboard use "ignition" as the product name with this §19 definition behind it.

**Validity preconditions (all implemented):**
1. **Persistent-anomaly (flare) filtering** — the one failure mode where "satellite heat"
   and "wildfire" fully diverge; 531 cells / 27% of raw positives removed (VALIDATION §9).
   (Next label refresh: retain VIIRS's own `type` flag — it marks "other static land
   source" — as an independent cross-check on our persistence filter.)
2. **Day-level granularity** — no sub-daily claims from imputed FPA-FOD times.
3. **Independent event validation** — the headline claims rest on documented real fires
   (Smokehouse etc., §VALIDATION), not on the label's construction alone.

**One-paragraph reviewer defense:** *Neither agency records nor satellite detections alone
capture fire occurrence without bias (Fusco et al. 2019: <25% / <50% mutual coverage); we
fuse them to offset complementary blind spots, following standard practice in ML wildfire
benchmarks that use VIIRS active fire as ground truth. We filter persistent thermal
anomalies via a persistence criterion, define the target at daily/cell granularity as fire
activity rather than strict ignition, and validate against independently documented fire
events rather than the label alone.*

---

## 20. How each dashboard tab computes (the wiring), and everything the trained model can do

### 20A. The wiring — what each dashboard state shows

The dashboard legend **names its own machinery** — the titles below are exactly what you see
on screen, so the two ways "Ignition" is computed are never conflated:

| Dashboard state (legend title) | What is computed | ML? | Which trained model | Weather | Formula toggle |
|---|---|---|---|---|---|
| **"Ignition · heuristic (2024–26)"** — scrubber | `susceptibility(cell) × FWI(cell,day)` — a **proxy, NOT the model's output**: susceptibility is the honest model's **pre-computed** average score per cell (monthly profile), then multiplied by the day's FWI | model (pre-computed, weather-free) × equation | **Ignition — honest model** (`..._baseline_honest_filtered.json`; static+calendar, no weather) | observed gridMET via FWI | **shown** (re-ranks map) |
| **"Ignition · live model — <date>"** — Live | the trained model **runs directly**: static + calendar + HRRR forecast weather → score. **No FWI multiply** (weather is learned inside the model) | **pure trained model** | **Ignition — operational HRRR model** (`..._hrrr_filtered.json`) | HRRR F24/F48 or GFS F72 | **hidden** (no effect) |
| **"Wildfire risk (2024–26)"** — scrubber | `TxWRAP hazard(cell) × FWI(cell,day)` | **no ML** | — (TxWRAP FSim simulation) | observed gridMET via FWI | shown |
| **"Wildfire risk · forecast — <date>"** — Live | `hazard(cell) × forecast-FWI(cell)` | **no ML** | — (same TxWRAP layer) | HRRR/GFS via FWI (gust + soil moisture) | shown |
| **"Static wildfire hazard"** | `haz` / `WHP` / `burn-prob` / `CBD` (metric picker) | **no ML** | — (raw TxWRAP layers, no time) | none | hidden |

**Punchline:** only the two **Ignition** rows touch a trained XGBoost model — and they use
*different* machinery: the **scrubber is a heuristic** (honest susceptibility × FWI — a proxy,
not the model replayed), the **live is the real operational model** run on forecast weather.
Every **Wildfire Risk / Static Hazard** row uses **no ML at all** (TxWRAP simulation × FWI
equation). That asymmetry is why: (a) the flare label fix re-baselined Ignition but left
Wildfire Risk untouched, and (b) the **Fire-weather formula toggle** (Composite / NOAA HWP /
TX HWP) appears only where an FWI term exists — Wildfire Risk (both) and scrubber-Ignition —
and is **hidden for live-Ignition and Static Hazard**, where no FWI is used.

**Honest caveat (the one real inconsistency):** scrubber-Ignition and live-Ignition are *not
the same computation* — the scrubber is a susceptibility×weather proxy because we don't have
historical HRRR *forecast* weather assembled for all 5,382 cells × 933 days. The legend now
says "heuristic" vs "live model" so this is disclosed, not hidden; making them identical is a
scoped future data task.

**Key mental model:** training happened once (offline, on 2018–2021 history of *what HRRR
had forecast* vs *what fires occurred*); each model is a frozen file of decision trees.
Every forecast is **inference** — a pure calculation on new inputs. Nothing retrains daily.
Retraining is a deliberate offline event (new label years, or a data fix like the flare filter).

### 20A-viz. What this looks like on the dashboard

![TDIS dashboard — Wildfire Risk mode, Feb 26 2024 (Smokehouse Creek day)](demo_pngs/13_dashboard_current_full.png)

Visual cues, keyed to the control panel (top-left) and legend (bottom-right):

- **Mode selector (the three stacked buttons):**
  🔥 *Wildfire Risk* (shown active/orange) → **no-ML** `hazard × FWI`;
  ✦ *Ignition* → **trained model** (a susceptibility×FWI *heuristic* for the scrubber, the
  real operational model when Live is on — the legend title says which);
  ▦ *Static Hazard* → **no-ML** raw TxWRAP.
- **📅 Live Forecast toggle:** OFF here (scrubbing history) → the pre-computed layers; the
  legend reads "· heuristic". Turn it ON and Ignition switches to the **operational HRRR model
  run live** (legend "· live model"), the fire-weather formula picker hides (no FWI applies),
  and a 24/48/72 h horizon picker appears.
- **Legend (bottom-right):** announces the exact machinery of the current view, shows the six
  band cutoffs on the 0–1 colour scale, and ends every note with "not a probability / relative
  ranking" — so the reading rule travels with whatever tab you're on.
- **Day scrubber** (`DAY 2024 · 2024-02-26`): the timeline slider (§the scrubber) indexing the
  933 daily arrays; ▶ animates it.
- **Fire-weather formula** (Composite / NOAA HWP / TX HWP): the FWI knob — re-ranks every
  FWI-based view (Wildfire Risk + scrubber-Ignition); leaves live-Ignition and Static Hazard
  unchanged.
- **Stats panel** (Hexagons / Mean / Top 5%): summary of the *currently displayed* value array
  — note these are the **raw** values (mean 0.121 here), i.e. what gets ranked, before the
  cosmetic colour-stretch.
- **Colour ramp + legend bands** (Extreme…Low): the fixed value→colour mapping (§7C). The
  Panhandle glowing red on this exact day (Feb 26 2024) is the Smokehouse Creek signal — see
  `DEMO_SMOKEHOUSE.md`.
- **Grey/black hexes** (deep South TX, dim NW corner): cells with no value for this
  mode/day (outside the dynamic universe or missing that day's weather) — ranked as null, not
  as "safe."

### 20B. Everything the trained model can do (it's an engine, not just a map)

The model is a function: `score = f(static features, calendar, weather)`. Anything you can
phrase as inputs to that function is a product:

1. **Daily operational forecast** (current use) — feed tomorrow's HRRR → tomorrow's map.
   Knobs: lead time (24/48 h HRRR, 72 h GFS), model file (filtered vs legacy), FWI variant
   for the risk layer.
2. **Hindcast any date** — feed any historical day's weather → "what would the model have
   said the morning of Smokehouse?" Case studies, retrospective skill audits.
3. **What-if scenarios** — feed *synthetic* weather: "score all of Texas under red-flag
   conditions (VPD 4 kPa, wind 15 m/s)" → planning products, worst-case maps, climate-
   scenario screening. No new code — just different input rows.
4. **Susceptibility mapping** — average scores over a period → the WHERE layer (this is
   exactly how the dashboard's scrubber ignition layer is made).
5. **Ranked triage lists** — sort cells by score within a county/district → "top 20 hexes
   to patrol tomorrow" CSV for operations. The score's ranking is its validated strength.
6. **Threshold alerting** — pick an operating point from the capture-vs-area curve
   ("top X% of cells captured Y% of historical fires") → binary alert product.
7. **Per-hex explanation** — SHAP values on any single prediction → "this cell is high
   today because forecast VPD + road density + ecoregion" (tooltip-ready explainability).
8. **Cross-region scoring** — the features are national datasets, so the frozen model can
   score any state for a transferability test (ties into the IgnitionNet CA↔TX work).
9. **Calibrated probabilities** (after roadmap 1.2) — same scores, mapped through a
   reliability curve → honest "% of days like this had fire."
10. **Ensembles / uncertainty** (future) — run the same frozen model over multiple weather-
    model members → per-hex risk spread.

The through-line: **one training investment, many products, all inference-only** — each
use is just a different way of assembling the input table.

---

## 21. Dashboard v2 — zoom-adaptive resolution (res 4→8) + "% of record" color scale

**`dashboard/tdis_fire_explorer_standalone.html`** (~102 MB, drag-and-drop; built by
[`scripts/20_build_explorer_data.py`](scripts/20_build_explorer_data.py) →
[`scripts/21_embed_explorer.py`](scripts/21_embed_explorer.py); template
`tdis_fire_dashboard_v2.html`). Everything v1 has, plus:

**Zoom-adaptive H3 grid (h3geo.org-style).** The map re-tessellates the *viewport* at the
H3 resolution matching the zoom (res 4 zoomed out → **res 8 = the model's native grid**
zoomed in), drawing ≤6,000 visible hexes at a time — never statewide res-8 at once.
- **Static layers carry the fine detail:** hazard + ML susceptibility pre-aggregated at
  res 4/5/6/7/8 (1.02M res-8 cells embedded; zero-hazard cells render as neutral grid).
- **Dynamic weather stays res-5:** fine-res risk = `haz(res-N hex) × FWI(res-5 ancestor)` —
  legitimate because weather fields are smooth at 250 km² while statics vary at 0.74 km².
- A res badge (top-right) always shows the current resolution and visible cell count.

**"% of record" color scale (the summer-saturation fix).** Dryness-driven formulas
(NOAA/TX HWP) max out statewide in dry seasons — absolute colors stop discriminating.
The new toggle colors each hex by **today's percentile within that cell's own 2024–26
record** ("is today unusual *for here*?"), restoring contrast exactly when absolute maps
saturate. Applies to FWI-based views only (hidden in live-forecast and static modes);
tooltips still show the absolute values.

**72-hour forecast retired (both dashboards).** Horizons are now **24 h / 48 h (HRRR)**
only: the 72 h GFS tab had the weakest forecast skill, a coarser grid, and a GFS
soil-moisture gap that broke the HWP variants on 475 cells. `13_model_forecast_day.py`
still *supports* lead 72 if ever needed; the chain and embeds no longer produce it.

---

## 22. Classification metrics: precision, recall, F1, and the threshold concept

AUC-PR/AUROC (used throughout this handbook) are **threshold-free** — they summarize
skill across all possible cutoffs at once. Precision/recall/F1 require picking ONE cutoff
on the model's 0–1 score, so they answer a more operational question: *"if we flag every
cell scoring above X, how good are those flags?"*

**Threshold, defined:** score ≥ threshold → flag as at-risk.
- Lower threshold → flag more cells → catch more real fires (**higher recall**) but more
  of the flags are wrong (**lower precision**).
- Higher threshold → flag fewer, more confident cells → fewer false alarms (**higher
  precision**) but more real fires slip through unflagged (**lower recall**).
- **F1** = harmonic mean of precision & recall — a common "balance point," but not
  automatically the right choice: for wildfire, a missed fire is usually costlier than a
  false alarm, which argues for a threshold *below* F1-optimal (accept more false alarms
  to catch more real fires).

### Threshold sweep — honest vs operational-HRRR vs ceiling (test 2023–2026)

| Threshold | Honest P/R/F1 | Operational-HRRR P/R/F1 | Ceiling* P/R/F1 |
|---|---|---|---|
| 0.2 | .250/.988/.399 | .261/.965/.411 | .240/.976/.386 |
| 0.3 | .267/.939/.416 | .289/.893/.437 | .262/.931/.408 |
| 0.4 | .305/.807/.443 | .336/.767/.468 | .289/.847/.431 |
| **0.5** | .383/.584/.463 | **.410/.602/.488** | .331/.717/.453 |
| 0.6 | .515/.380/.437 | .511/.438/.472 | .396/.550/.460 |
| 0.7 | .599/.282/.383 | .596/.313/.410 | .510/.337/.406 |
| 0.8 | .633/.171/.269 | .644/.219/.327 | .633/.131/.217 |

**\* Ceiling** (same-day **observed** gridMET weather — the "if forecasts were perfect"
diagnostic bound, never served) runs on a smaller, differently-based test set (203,699
rows @ 20.9% positive vs 892,372 @ 23.4% for the other two) — **not directly comparable**
to the honest/operational columns without a controlled ablation. Full detail + caveats:
`VALIDATION.md §10`.

**Key takeaway:** operational-HRRR beats honest at *every* threshold on identical rows —
the cleanest proof in the whole project that forecast weather adds real value. F1 peaks
near threshold 0.5, but that still misses ~40% of real fire-days (recall ≈ 60%) — a
recall-first operational policy would likely run at threshold 0.3–0.4 instead.

**Caveat that applies to every number above:** these are computed on the training
design's ~23% matched-sample positive rate, not the real-world ~0.3%/day base rate.
Precision should NOT be read as "X% of real-world alerts are correct" until the
calibration step (roadmap Tier 1.2) is done.
