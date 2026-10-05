# TDIS Fire-Weather Product (HWP / FWI) — Team Handoff Package

> **Start here if you are building the pipeline:** `PRD_FireWeather_HWP.md`
> in this folder is the TRD-style spec (problem statement, data flow, handoff
> assets, acceptance criteria). This README is the working reference behind it.

**The dashboard's fire-weather layer: three formulations of daily fire-weather
intensity per H3 cell, driven by NOAA HRRR forecast fields.** Locked/served
since 2026-08. This is a SEPARATE product from the ignition model in `../model/`
— see the hard rule in §5. Packaged 2026-09-23; verified reproducing the
documented worked example from this location (TX 0.788, NOAA 0.215).

---

## 1. The three formulations (all computed side-by-side, dashboard-toggleable)

### `hwp_noaa` — NOAA GSL Hourly Wildfire Potential (the equation)
**James et al. 2025, *Weather and Forecasting*, doi:10.1175/WAF-D-24-0068.1 — Eq. 3, implemented EXACTLY:**

```
HWP = 0.213 · G^1.50 · VPD^0.73 · (1 − M)^5.10 · S
```

| Term | Meaning | Live (HRRR) source | Historical (gridMET) proxy |
|---|---|---|---|
| G | 10-m wind gust, m/s (floor 3.0) | `GUST` field — **exact** | `1.5 ×` daily-mean wind |
| VPD | vapor-pressure deficit, **hPa** *in the equation* | from TMP+DPT (Tetens) — exact | gridMET vpd |
| M | soil-moisture availability, 0–1 | `MSTAV`/100 — **exact** (GFS fallback: SOILW/0.45) | `fm100 / 30` (saturation proxy) |
| S | snow term | =1 for the TX archive | =1 |

### `hwp_tx` — same multiplicative form, coefficients fit to Texas fire activity
`17.6 · G^0.05 · VPD^0.05 · dry^0.92` — log-linear LSQ on large-fire episodes
(peak day ≥500 MW VIIRS FRP, ±7-day windows, 50,893 cell-days, TX 2024–26;
`scripts/16_fit_hwp_tx.py`). **Key scientific finding baked into these
coefficients: at daily resolution, TX fire activity is dryness-dominated —
the wind signal only emerges at hourly/gust resolution.** Fit r_log ≈ 0.10
(weak by design: daily aggregation washes out wind; this is a finding, not a
failure). Do not present TX-HWP as a better *predictor*; it is a
TX-calibrated *characterization*.

### `composite` — the original additive TDIS index (served default)
`0.30·n(erc) + 0.30·n(vpd) + 0.40·n(wind)`, each term normalized by fixed
scales (erc/100, vpd/5, wind/12, gust/22). On the live HRRR path (no ERC),
the erc weight folds into vpd+wind.

**Normalization:** HWP is unbounded, so each variant is divided by its p99.5
over the 2024–26 TX archive (refs stored in `data/hwp_params.json`: NOAA
26.25, TX 20.10) and clipped to 0–1 — a relative scale, same spirit as the
composite.

## 2. Validation status (from `hwp_validation.json` + the 2026-08-05 shoot-out)

- **Worked example reproduces**: TX variant 0.788 vs documented 0.8 ✅ (NOAA
  0.215 for the same inputs — different ref scaling, both correct).
- **Input QC clean**: zero negative wind/VPD, zero out-of-range vpd/fm100,
  zero date gaps (677K nulls = known coverage edges).
- **Named-event percentiles** (NOAA HWP on event day, statewide): Smokehouse
  70.1 · Windy Deuce 74.9 · Crabapple 86.6 · Lavender 90.9 · Hunggate 91.8.
- **4-event "WHEN" shoot-out** (percent-of-record on event day,
  Smokehouse/Crabapple/Lavender/Hunggate): composite 75/99/94/99 · NOAA
  80/99/85/97 · TX 75/100/64/95. **Composite stays the served default; NOAA
  wins Smokehouse** (the wind-driven event — consistent with the equation's
  G^1.50 term).
- **Monthly climatology sanity**: peaks Mar–May (mean 0.166–0.169), winter
  minimum — matches the TX fire calendar.

## 2b. Overall validation, 2026-09-23 (`overall_validation_2026-09-23.json`)

Fresh package-level checks answering "is it all of Texas, is it 0–1, does it
carry signal":

- **Coverage — live vs historical differ.** The **live product covers all of
  Texas** (every res-8 cell from each HRRR pull). The packaged **historical
  components archive covers 3,624 of 5,382 res-5 cells (67.3%)** — not border
  slivers; consistent with the fire-relevant dynamic-panel universe. If the
  pipeline needs full-state *history*, rebuild components statewide; live
  serving is unaffected.
- **Range — yes, 0–1 by construction.** Composite spans 0.047–0.855 (never
  clips); both HWP variants clip ~0.5% of cell-days at 1.0 (beyond their
  p99.5 refs). Dashboards display percent-of-record; TDIS portal maps to its
  0–2 intensity convention.
- **Pooled discrimination ≈ none — and that's expected.** Across all res-5
  cell-days 2024–26 (base rate 2.4%), all three variants score AUROC ≈ 0.50
  for "did a fire occur here today." Fire weather does not know WHERE fires
  start (TX ignitions are 84% human-caused; roads/people dominate). This is
  the quantitative backing for the §5 never-multiply rule — the ignition
  model owns WHERE, fire weather owns conditions.
- **Within-cell timing/severity — the real signal.** Judged against each
  cell's own record (the percent-of-record framing the dashboard uses):
  ordinary fire days sit just above the cell median (mean pct 0.53–0.55),
  but the **top-1% most intense fires (by VIIRS FRP) concentrate on high
  fire-weather days** — mean percentile 0.63–0.68, with 41–46% landing in
  the cell's top quartile. The TX variant is strongest on intensity days
  (0.680), consistent with its large-fire fit target.

**Recommended additions for full production sign-off (not yet done):**
(1) external cross-check of a few days' computed HWP against NOAA GSL's
published operational HWP maps (independent ground truth for the
implementation); (2) quantify gridMET-proxy vs exact-HRRR agreement (G=1.5·wind
vs real gust; M=fm100/30 vs MSTAV) on overlapping days; (3) Red Flag Warning
correspondence once RFW polygons are pulled (a long-standing "need to pull").

## 3. What's in this folder

| Path | Contents |
|---|---|
| `scripts/fwi_config.py` | **The single source of truth**: all three formulations, both input paths (HRRR live / gridMET historical), the knobs (`FWI_MODE`, `FWI_WEIGHTS`), normalization refs. Import this; do not re-derive the math. |
| `data/hwp_params.json` | Served parameters: TX-fit coefficients + both p99.5 refs. `fwi_config` loads it relative to this folder — keep them together. |
| `data/fwi_components_res5.parquet` | 3,381,192 res-5 cell-days (2024–26): raw `erc`, `vpd`, `vs` (wind), `fm100` components — recompute any variant historically without re-pulling weather. |
| `scripts/16_fit_hwp_tx.py` | The TX-coefficient fitting script (episode-windowed, log-space — raw-space curve_fit degenerates on many-zero daily data, documented in-script). |
| `scripts/15_reweight_fwi.py` | Rebuilds all three variant arrays over the historical archive. |
| `scripts/run_hwp_chain.sh` | The full historical chain (components → fit → arrays) as run in the source repo. |
| `31_hwp_validation.py`, `hwp_validation.json` | The validation harness + its results (§2). |
| `DEMO_SMOKEHOUSE.md` | Worked, literature-cited example calculation on the Smokehouse Creek day — the best onboarding doc for how the numbers happen. |
| `figures/` | Smokehouse-day maps for both HWP variants + summer absolute-vs-percentile comparisons (the summer-saturation motivation for percent-of-record display). |

## 4. Computing it live (for the pipeline team)

The live inputs are **the same HRRR fields the ignition entrypoint already
pulls** (`../serve/score_day.py`: TMP/DPT→VPD, GUST, MSTAV — one 12Z pull
serves both products):

```python
import sys; sys.path.insert(0, "fireweather/scripts")
import fwi_config as F
# UNITS: the API takes vpd in kPa and converts to the equation's hPa
# internally (x10 inside hwp_from_components) — do NOT pre-convert.
# vpd in kPa, gust in m/s, soilm = MSTAV/100 (0-1), per cell:
hwp_noaa = F.hwp_hrrr(vpd, gust, soilm, variant='noaa')   # 0-1
hwp_tx   = F.hwp_hrrr(vpd, gust, soilm, variant='tx')     # 0-1
composite = F.fwi_composite_hrrr(vpd, wind, gust=gust)    # 0-1
```

The source repo's `scripts/13_model_forecast_day.py` shows the full serving
pattern (emits `fwi`/`fwiN`/`fwiT` alongside ignition). GFS fallback for
lead >48h: SOILW/0.45 for M, `1.5×wind` for G (no gust field).

Display note: dashboards show HWP as **percent-of-record per cell** (each
cell against its own 2024–26 history) — raw 0–1 saturates in summer
(`figures/17` vs `18` shows why). TDIS portal convention displays HWP on a
0–2 intensity scale.

## 5. ⛔ The one hard rule

**Never multiply (or otherwise combine) HWP onto the ignition model's
output.** This was explicitly tested and degrades ignition AUC-PR by
**18–61%** (documented in the source repo's METHODOLOGY.md). They are two
separate products answering different questions — ignition = "will a fire
start here" (learned, calibrated probability); HWP = "how severe is the fire
weather" (physical index, relative scale). Serve them as separate layers,
exactly as the TDIS dashboard does.

## 6. Provenance

Canonical source: `alphaearth_nds/TDIS_Forecast/` (`scripts/fwi_config.py`,
`data/hwp_params.json`, fit/validation scripts) — this folder is a verified
copy of the curated handoff in
`alphaearth_nds/TDIS_Forecast/rev5_ignition_fireweather/fireweather/`.
History/decisions: source repo `HANDBOOK.md` §16, `VALIDATION.md` §8.
