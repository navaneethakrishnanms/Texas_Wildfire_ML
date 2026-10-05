# Project overview — plain-English guide to understand everything

**One place that explains the whole project in plain terms: what the models
are, what every term means, what we found, and how the paper hangs together.**
(2026-09-21. Points to the detailed docs for numbers.)

---

## 1. What this project is, in three sentences

We predict wildfire risk two ways that answer different questions: **(1) will a
fire start in this ~860 m cell during this 6-hour window** (ignition
probability), and **(2) if a fire started here, how intense would it be**
(conditional intensity). These two predictions turn out to be nearly
independent, so combining them shows high-consequence places an ordinary
"where will a fire start" map misses. The ignition model runs at sub-daily
(6-hour) resolution, which — although it barely changes accuracy — lets an
agency re-time a fixed suppression budget through the day and catch materially
more fires at no added cost.

## 2. The two models (plain)

| | Ignition model | Intensity model |
|---|---|---|
| Question | *Will* a fire start here? | *If* one starts, how intense? |
| Kind | classifier → probability (0–1) | regressor → a number (fire radiative power) |
| Learns from | recorded fire occurrences (FPA-FOD) | satellite heat measurements (VIIRS FRP) |
| Driven by | people, roads, dryness, time-of-day, weather | fuel, terrain, elevation, ecoregion |
| Runs at | sub-daily (4 windows/day) | static (one map, doesn't change hour to hour) |
| Both are | XGBoost (gradient-boosted trees) on an H3 hex grid | — |

They are **separate models** trained on **separate data**. They share the same
landscape *inputs* (elevation, fuel, etc.) but never share labels or train/test
rows — so combining them is legitimate, not double-counting.

## 3. Glossary — every term we've used

- **H3 cell / resolution-8** — the hexagonal map grid; res-8 ≈ 860 m across.
- **Window (6-hour / sub-daily)** — the day split into four 6-hour periods
  (12Z/18Z/00Z/06Z UTC). "Sub-daily" = we predict per window, not per day.
- **Forecast-realistic** — every window's weather comes from the *previous
  afternoon's* 12Z HRRR forecast, so the maps are issuable ~2 pm the day before
  (deployable, not a hindcast).
- **FPA-FOD** — the federal fire-occurrence database; source of ignition labels.
- **VIIRS FRP** — satellite-measured Fire Radiative Power (megawatts); how much
  heat a fire radiates. Our intensity *label*. A validated combustion-rate
  measure, not a raw pixel value (Wooster 2003/2005).
- **Conditional intensity** — intensity *given a fire occurs* ("conditional on
  ignition"). Parallels the USFS FSim "Conditional Flame Length" product, but
  learned from observations instead of simulated.
- **Orthogonal axes** — ignition and intensity are nearly uncorrelated
  (ρ ≈ 0.04–0.16). Knowing one tells you little about the other, so together
  they carry two independent pieces of information.
- **Sleeper cell** — low ignition probability but high intensity: a place that
  rarely ignites but would burn hot if it did (remote forest/canyon). 24% of TX.
  Invisible to an ignition-only map.
- **Spearman ρ** — a regression metric: correlation between the model's
  *predicted* values and the *actual* values, measuring whether the model
  *ranks* cells correctly. Right metric for a ranking/prioritization tool.
  (R² is its magnitude-based cousin; we demote it — see below.)
- **Calibration / ECE** — whether a "0.3" prediction really means 30% happen.
  We calibrate the ignition model (isotonic); Expected Calibration Error dropped
  0.19 → 0.07.
- **Budget** — how many cells an agency can afford to actively watch.
- **Cell-shift** — one cell watched for one 6-hour shift; the resource unit.
- **Reallocation** — daily forecast = same watch-list all day; sub-daily = re-aim
  the same budget each shift. The source of the operational gain.
- **Population recall** — flag the top X% of the whole state → catch Y% of the
  fires, at the true (tiny) base rate. The honest operational recall number.
- **Lift** — how many times better than random a model is at its base rate.
- **Flare contamination** — persistent industrial gas-flare hotspots that VIIRS
  sees as "fire"; ~28% of TX detections. We excluded them (they inflate/distort
  intensity skill). A real bug we caught and fixed.

## 4. What we found (plain bullets)

1. **Sub-daily doesn't beat daily on accuracy** — it's a tie (AUC-PR 0.305 vs
   0.300). We don't pretend otherwise.
2. **But sub-daily catches 25–47% more fires at equal crew budget** — by timing
   coverage to the diurnal fire cycle (right cell, right shift).
3. **Fires don't move in space through the day** (region overlap ~0.8); the
   diurnal signal is *timing* (≈4× afternoon peak), not location. This is why
   the sub-daily argument is *temporal*, not spatial.
4. **Intensity is nearly independent of ignition** (ρ ≈ 0.11) — a genuinely new
   second axis, not a re-weighting of the ignition map.
5. **Sleeper cells (24% of TX) are structural, not random** — 9.7× more
   lightning, 8.8× farther from roads, higher/steeper terrain. The ignition
   model correctly deprioritizes them (no people); the intensity model is the
   only thing that flags them.
6. **The intensity model is a validated ranking tool** — ρ 0.55 (TX), beats the
   physical Hot-Dry-Windy index (−0.20), identifies top-1% intensity fires at
   AUC 0.85. It ranks well but under-predicts absolute megawatts (a ranker, not
   a magnitude forecaster).
7. **Two correctness bugs found and fixed** — flare contamination and a broken
   TX static-feature file. Fixing them strengthened the results and the
   credibility of the methods section.

## 5. The paper's spine (how it hangs together)

- **Headline / novelty:** two orthogonal axes ("will it start" × "how bad")
  reveal high-consequence *sleeper cells* an ignition-only forecast misses.
  This is the unique, literature-grounded contribution.
- **Sub-daily's role:** the *delivery mechanism*, justified by **operational
  reallocation** (+25–47% fires at equal budget via temporal targeting) — NOT
  by classification accuracy (a tie) and NOT by spatial relocation (null).
- **Intensity's role:** the "how bad" axis; scientifically grounded (FRP =
  validated intensity; parallels FSim CFL but observation-based), honestly
  scoped as a ranker.
- **Honesty as a feature:** the nulls (lightning unhelped by sub-daily; no
  spatial relocation; classification tie) *sharpen* the claims rather than
  weaken them, and the bug fixes make the methods trustworthy.

## 6. Honest limitations (so nothing surprises you)

- Sub-daily accuracy gain is small; the case rests on operational framing.
- Intensity is modest in absolute skill (CA ρ 0.26; TX 0.56) and is a ranker,
  not a magnitude predictor (compresses the extremes).
- TX ignition labels are 87% imputed times → TX per-window claims are caveated;
  California (96.6% real) is the clean per-window venue.
- Lightning ignition is near-chance to localize at cell scale — sub-daily is a
  human/equipment-fire tool.
- Test years are 2020 (ignition) / 2025–26 (intensity); single-year caveats apply.

## 7. Map of the documents (where the numbers live)

> **Packaging note:** this overview was written in the research repo. Of the
> docs below, only `DATASET_BREAKDOWN.md` is included in this package (same
> `docs/` folder). The rest live in the source repo
> (`alphaearth_nds/.../OperationalLayer/`) — ask the research owner if needed;
> nothing in the served product depends on them.

| Doc | Contains |
|---|---|
| `RESULTS_PAPER.md` | the full results in paper order (models → metrics → orthogonality → sleepers → cases → operational) |
| `OPERATIONAL_VALUE.md` | operational metrics menu + plain-terms budget/reallocation + novelty assessment |
| `FORECAST_INTENSITY.md` | intensity model method, metrics, corrections log |
| `INTENSITY_METHODS_LITERATURE.md` | verified citations + novelty positioning for the intensity axis |
| `DATASET_BREAKDOWN.md` | full feature list, importances, descriptive stats for both datasets |
| `FINDING_ignition_x_intensity.md` | the orthogonality finding on its own |
| `CASE_STUDY_tx_2020-06-16.md` | the Big Bend sleeper-cell case study |
| `results/*.json` | machine-readable numbers behind every claim |
| `results/fig_*.png` | all figures |

If you read only two: **this file** for the concepts, **`RESULTS_PAPER.md`** for
the numbers.
