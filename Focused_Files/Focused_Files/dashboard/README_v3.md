# TDIS Wildfire Forecast Dashboard — v3

`tdis_fire_explorer_v3_standalone.html` — open directly in a browser, no
server needed. Needs internet only for the base-map tiles and the
Leaflet/h3-js libraries (loaded from a CDN).

v3 drops two things v2 had: the historical day-by-day scrubber (2024–2026
replay) and the static-hazard-only tab. **v3 shows only what's live and
forward-looking**: the current 24h and 48h HRRR forecast, three ways.

## The three tabs

### 🔥 Wildfire Risk
`Static hazard × live NOAA HWP fire-weather.` No machine learning. Hazard
is a fixed, place-based layer (TxWRAP-style — terrain, fuel, historical
burn probability; doesn't change day to day). Fire-weather is today's live
forecast. Multiplying the two answers "how dangerous is this place, given
today's actual weather" — a place with high hazard but calm, humid weather
today shows lower than its neighbor having a hot, dry, windy day.
Zoom in and hexagons refine down to the model's true resolution (H3 res-8,
~0.74 km² per cell); zoomed out, they coarsen for render speed only — the
underlying hazard data doesn't lose resolution.

### ✦ Ignition
The **trained ML model's own output**, run on the same live forecast
weather — not a heuristic, not hazard × weather. This is the model that
learned from 2+ years of actual VIIRS/FPA-FOD fire occurrence. Two numbers
show on hover:
- **Rank score** — what colors the map. Relative ranking only; a value of
  0.6 does *not* mean "60% chance of a fire." The model was trained on a
  rebalanced sample (~15–20% positive rate) to learn faster, so its raw
  output overstates real-world probability by roughly 17×.
- **Calibrated probability** — the number that's actually safe to read as
  "chance of fire-related activity here in this window." Fit with
  isotonic regression against real 2024–25 outcomes, validated on a 2026
  holdout the calibrator never saw (error dropped from ~35% miscalibration
  to under 0.2%). If you need to quote a number in a meeting, quote this
  one, not the map color.

Always shown at H3 res-5 (~21.5 km²) — the forecast's native output grid;
it does not refine on zoom the way Wildfire Risk does.

**Which model, and how good is it?** The model is
`models/tdis_forecast_hrrr_filtered.json` (XGBoost + live HRRR weather
features) — the best-validated ignition model in the project. Headline
results, both rows the same model:

| Evaluated on | AUC-PR | Best F1 |
|---|---|---|
| Test set (balanced sample, ~20% fire-days) | 0.4941 | 0.496 |
| Real full population (all TX, 2024–26, 1.9% fire-days) | 0.0878 | 0.169 — lift 4.58× over random |

The two rows differ only because fires are ~10× rarer in reality than in
the balanced test sample (AUC-PR and F1 both shrink with the base rate).
Use the first row to compare against other models; use the second to
describe real-world performance. Full detail:
`docs/CLASSIFICATION_METRICS.md`.

**Does it forecast without retraining?** Yes. The model is already
trained; each day's forecast is inference only — new HRRR weather in,
risk scores out (`scripts/13_model_forecast_day.py`). No training happens
in the daily loop; retraining is an annual maintenance task.

### 🌬️ Fire Weather Index
**Weather only** — no hazard layer, no ML. This is the NOAA GSL Hourly
Wildfire Potential (HWP), shown by itself so you can see what the weather
is doing independent of where fuel/terrain happen to be dangerous. See
below for the equation and how it's normalized. Also res-5, matching the
forecast grid.

## The NOAA HWP equation

```
HWP = 0.213 × G^1.50 × VPD^0.73 × (1 − M)^5.10
```
(James et al. 2025, *Weather and Forecasting*, Eq. 3 — the source paper
for NOAA's Global Systems Laboratory Hourly Wildfire Potential index.)

| Symbol | Meaning | Source in this pipeline |
|---|---|---|
| **G** | 10 m wind gust (m/s), floored at 3 | HRRR live forecast (`GUST:surface`) |
| **VPD** | Vapor pressure deficit (hPa) — how much moisture the air can still pull out of vegetation | Derived from HRRR temperature + dewpoint |
| **M** | Soil-moisture availability, 0–1 (1 = saturated, 0 = bone dry) | HRRR `MSTAV` field |
| `(1−M)` | Dryness | — |

**Why this shape, not a simple average**: the exponents are not arbitrary
scaling — they encode how fire risk actually compounds. Gust enters at the
1.5 power (double the wind, ~2.8× the term), and dryness at 5.1 — meaning
a landscape near saturation barely registers even under high wind, but as
it dries out the same wind produces a rapidly escalating number. That's
the intended behavior: HWP is built to spike sharply on genuine red-flag
days, not scale smoothly.

## How it's normalized (why the map doesn't show huge numbers)

The raw equation above is **unbounded** — plug in a hot, dry, gusty day
and the raw value can land anywhere from single digits to several hundred,
depending on how extreme the inputs are. Two things are done so the
dashboard shows a sane 0–1 (or 0–100%) scale, same as every other layer:

1. **Fixed reference divide**: the raw HWP is divided by its own
   99.5th-percentile value over the full 2024–2026 Texas archive (stored
   in `data/hwp_params.json`), clipped, and **displayed on a 0–2 intensity
   scale** (matching the TDIS portal's hazard-intensity convention, e.g.
   Precipitation HSI's 0.25–2.0+ bins). So "2.0" means "among the most
   extreme fire-weather days Texas has seen in this record," not an
   arbitrary ceiling.
2. **Same color ramp, same 6 severity bands, as Wildfire Risk and
   Ignition** — Low / Moderate / Elevated / High / Very High / Extreme,
   with the same color-to-band mapping across all three tabs. Switching
   tabs doesn't require re-learning what a color means.

This is also why the earlier composite and TX-fit HWP variants were
removed from this version — with only one formula on screen, there's one
number, one equation box, and one legend to read, instead of three
formulas each needing their own explanation.

## Forecast horizon

24h and 48h buttons switch which live HRRR run is shown — both are pulled
fresh via `scripts/13_model_forecast_day.py` (target date, lead hours) and
bundled into this file by `scripts/22_embed_v3.py`. There is no "yesterday"
or historical view in v3 by design; for that, see the v2 explorer
(`tdis_fire_explorer_standalone.html`), which keeps the full 2024–2026
day-by-day scrubber.

## Regenerating this file

```bash
python scripts/13_model_forecast_day.py <tomorrow's date> 24
python scripts/13_model_forecast_day.py <day after> 48
python scripts/22_embed_v3.py
```
Produces a fresh `tdis_fire_explorer_v3_standalone.html` from whatever the
two newest `forecast_*.json` files in this folder are.
