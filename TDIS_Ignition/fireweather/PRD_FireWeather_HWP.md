# Product Requirements Document for TDIS Fire Weather (NOAA HWP)

**Author:** Miguel (IDRT research) — handoff to TDIS Portal engineering
**IDRT SME:** research owner of the ignition + fire-weather models
**Source package:** `miguel_shared/TDIS_Ignition/fireweather/` (verified 2026-09-23)
**Companion products:** TDIS 24–48h Ignition Model (`../model/`, `../model_tristate/`)
**Style note:** structured to match the TDIS Portal TRD (v5.1), esp. §2.3.5
(Precipitation HSI) and §2.3.10 (Wind HSI), which this layer parallels.

---

## 1 Introduction

### 1.0 Problem Statement

Fire-weather indices exist to help land managers, forecasters, and
firefighters anticipate conditions that could affect active or potential
wildfires in the coming days. Most of these indices, however, produce a single
value for an entire 24-hour period. Recent extreme wildfire activity has shown
that fire behavior is driven by weather that changes within the day, including
thunderstorm gust-front passages, radiative shading by smoke plumes, and
pyroconvective development. The arrival of rapidly updating,
convection-allowing forecast models made a higher-cadence index possible.
NOAA GSL answered this gap with the Hourly Wildfire Potential (HWP; James et
al. 2025, doi:10.1175/WAF-D-24-0068.1): an hourly index computed from
operational HRRR output and calibrated against satellite fire radiative power
from large wildfires. HWP tracks hourly fire intensity better than a
climatological approach and shows promise for separating flaming from
smoldering conditions.

The TDIS Portal currently has no fire-weather severity layer with this
pedigree. This product brings the published HWP to Texas: the exact NOAA
equation, computed from the same HRRR fields TDIS already pulls daily,
normalized onto the portal's common HSI interpretation scale, and served
beside — never blended with — the ignition-probability layer.

### 1.1 Purpose

The Fire Weather layer expresses **how severe the fire weather is** for every
Texas cell on the forecast day, as a normalized 0–1 index derived from NOAA
HRRR forecast fields. It is the fire analog of the Precipitation HSI and Wind
HSI layers: a physical severity indicator on a common interpretation scale,
computed on the Texas H3 grid, refreshed daily, and served as a map layer.

It answers "under these forecast conditions, how intense would a fire burn and
how fast would it spread" — it deliberately does **not** answer "will a fire
start here" (that is the separate ignition model). The two products share one
HRRR pull and are served as separate layers.

This document specifies: what the data is, exactly how the index is computed,
what the engineering team needs to build the pipeline, and the acceptance
checks that define a correct implementation.

### 1.2 Intended Audience

- **Data Engineering** — build the daily Databricks/ETL job: HRRR fetch,
  field derivation, index computation, PostgreSQL sync.
- **Backend Engineers** — serve the surface via the existing Flask API / tile
  patterns used for Precipitation HSI.
- **Frontend Engineers** — render the layer (React/Mapbox) using the
  percent-of-record display convention and the portal's 0–2 intensity scale.
- **Product/Technical Leads and reviewers** — verify the acceptance criteria
  in §4 and the hard rule in §5.

### 1.3 Definitions

- **HWP** — Hourly Wildfire Potential; NOAA GSL's operational fire-weather
  index. Reference: **James et al. 2025, *Weather and Forecasting*,
  doi:10.1175/WAF-D-24-0068.1**, Equation 3.
- **HRRR** — NOAA High-Resolution Rapid Refresh forecast model. The live
  input source. Fields used: `GUST`, `TMP`, `DPT`, `MSTAV`.
- **gridMET** — historical daily surface-weather dataset; used only for the
  historical archive via documented proxies (§2.4), never for live serving.
- **VPD** — vapor-pressure deficit, computed from TMP + DPT (Tetens).
  **Unit trap:** the published equation uses hPa, but the packaged API
  (`fwi_config.hwp_from_components` and callers) takes **kPa** and converts
  to hPa internally — pass kPa, never pre-convert. Unit handling is owned by
  `scripts/fwi_config.py`; import it, do not re-derive.
- **p99.5 reference** — fixed normalization constant per variant (the 99.5th
  percentile of the 2024–26 TX archive), stored in `data/hwp_params.json`.
- **Percent-of-record** — display convention: today's value ranked against
  the same cell's own history, rather than the raw 0–1 value.

### 1.4 Scope

In scope: the daily fire-weather surface (three formulations, §2.3), its
ingestion, computation, sync, and presentation. Out of scope: the ignition
probability model (separate package, separate TRD material in `../README.md`);
convective-scale phenomena; any blending of this index with the ignition
model's output (§5).

---

## 2 Fire Weather (HWP) Data Flow

### 2.3.11-style Flow Summary

The Fire Weather layer extends the HSI family to fire. A scheduled daily job
reuses the **same 12Z day-before HRRR pull that serves the ignition model**
(one pull, two products), derives per-cell gust, VPD, and soil-moisture
availability on the Texas H3 grid, computes three fire-weather formulations
side-by-side (NOAA HWP exact, a Texas-fit variant, and the served composite),
normalizes each by a fixed p99.5 reference to a 0–1 scale, and publishes the
surface with the standard HSI full-refresh sync. Presentation uses
percent-of-record per cell, mapped to the portal's 0–2 intensity convention
(same as Precipitation HSI).

### 2.1 Requirements and Design Decisions

- **Source:** NOAA HRRR forecast fields, pulled once daily at the 12Z
  day-before run, lead window F24–F45 (the 24–48h forecast day). This is the
  identical pull specification as `../serve/score_day.py` — do not schedule a
  second download.
- **Equation fidelity:** the NOAA variant implements James et al. 2025 Eq. 3
  **exactly** on HRRR fields (verified bit-exact:
  `spec_validation_2026-09-23.json`). The implementation source of truth is
  `scripts/fwi_config.py`; the pipeline must import it, not re-implement it.
- **Normalization:** each variant is divided by its fixed p99.5 reference
  (NOAA 26.25, TX 20.10; `data/hwp_params.json`) and clipped to 0–1, so all
  fire-weather variants read on one scale. ~0.5% of cell-days clip at 1.0 by
  construction.
- **Three formulations, one layer:** all three are computed side-by-side and
  are dashboard-toggleable; the **composite remains the served default**
  (per the 2026-08-05 four-event shoot-out).
- **Separation of concerns:** fire weather owns "conditions"; the ignition
  model owns "where." Pooled discrimination of fire occurrence by fire
  weather alone is AUROC ≈ 0.50 — expected, since 84% of TX ignitions are
  human-caused. This is a design fact, not a defect (§4, §5).
- **Cadence and volatility:** one surface per day (fresh recomputed snapshot,
  like the HSI layers). Full refresh; no incremental updates.

### 2.2 Data Source (TRD source-table row)

| Event | Source | API / Product | Filter Criteria |
|---|---|---|---|
| Fire Weather (HWP) | NOAA HRRR (via Herbie/NOMADS or archive) | 12Z run, leads F24–F45; fields `GUST`, `TMP`, `DPT`, `MSTAV` | Texas; H3-indexed; snow term S=1 for TX archive |
| Fire Weather (historical rebuild only) | gridMET | daily wind, vpd, fm100, erc | Texas; proxies per §2.4 |

### 2.3 The Three Formulations (Gold-layer math)

**`hwp_noaa` — NOAA GSL HWP, Eq. 3 implemented exactly:**

```
HWP = 0.213 · G^1.50 · VPD^0.73 · (1 − M)^5.10 · S
```

| Term | Meaning | Live (HRRR) source | Historical (gridMET) proxy |
|---|---|---|---|
| G | 10-m wind gust, m/s (floor 3.0) | `GUST` — exact | 1.5 × daily-mean wind |
| VPD | vapor-pressure deficit | from TMP+DPT (Tetens) — exact | gridMET vpd |
| M | soil-moisture availability, 0–1 | `MSTAV`/100 — exact | `fm100 / 30` |
| S | snow term | = 1 for Texas | = 1 |

**`hwp_tx` — same multiplicative form, coefficients fit to Texas fires:**
`17.6 · G^0.05 · VPD^0.05 · dry^0.92` (log-linear fit on large-fire episodes,
50,893 cell-days, TX 2024–26). The near-zero wind exponents encode a real
finding: at **daily** resolution TX fire activity is dryness-dominated; the
wind signal only emerges at hourly/gust resolution. Present it as a
TX-calibrated characterization, not a better predictor.

**`composite` — the served default:** `0.30·n(erc) + 0.30·n(vpd) +
0.40·n(wind)` with fixed normalization scales; on the live HRRR path the erc
weight folds into vpd + wind.

**GFS fallback (leads > 48 h, if ever extended):** M = `SOILW`/0.45,
G = 1.5 × wind (GFS has no gust field).

### 2.4 Fetch HRRR Forecast Fields (Bronze)

- Reuse the ignition entrypoint's daily 12Z HRRR pull (`../serve/score_day.py`
  pattern, Herbie). Fields: `GUST`, `TMP`, `DPT`, `MSTAV`, leads F24–F45.
- Land raw GRIB extracts in the Bronze layer with the standard
  provider-path convention, preserving original structure for replay.
- Historical backfill does not require re-pulling weather: the packaged
  `data/fwi_components_res5.parquet` (3,381,192 res-5 cell-days, 2024–26)
  carries the raw components (`erc`, `vpd`, `vs`, `fm100`) and can recompute
  any variant historically.

### 2.5 Spatial Indexing (H3)

- Live serving grid: **Texas H3 res-8** (full state, every cell of each pull —
  the same grid the ignition model scores).
- Historical archive: res-5 (3,624 of 5,382 res-5 cells, 67.3% — the
  fire-relevant dynamic-panel universe, not border slivers). If full-state
  *history* is required, rebuild components statewide; live serving is
  unaffected.
- Nearest-gridpoint mapping from HRRR points to hexes follows the same
  pre-computed static-mapping approach as the Precipitation HSI hex grid.

### 2.6 Field Derivation and Window Computation (Silver)

- Per cell and forecast day: derive VPD from TMP+DPT (Tetens), take the
  window-peak gust and the paired VPD/MSTAV values across the F24–F45 leads
  (the "forecast-day peak conditions" convention shared with the ignition
  features).
- All unit handling and floors (gust floor 3.0 m/s) live in
  `scripts/fwi_config.py` — the pipeline imports `fwi_config` and calls:

```python
import fwi_config as F
hwp_noaa  = F.hwp_hrrr(vpd, gust, soilm, variant='noaa')   # 0–1
hwp_tx    = F.hwp_hrrr(vpd, gust, soilm, variant='tx')     # 0–1
composite = F.fwi_composite_hrrr(vpd, wind, gust=gust)     # 0–1
```

- Persist per-cell daily components keyed by H3 cell and run timestamp.

### 2.7 Index Calculation and Normalization (Gold)

- Compute all three variants per cell per day (§2.3).
- Normalize `hwp_noaa` and `hwp_tx` by their fixed p99.5 references from
  `data/hwp_params.json`; clip to 0–1. The composite is 0–1 by construction
  (observed span 0.047–0.855, never clips).
- Additionally compute each cell's **percent-of-record** (today vs the cell's
  own 2024–26 history) — this is the display value (§2.9). Raw 0–1 saturates
  in summer; `figures/17` vs `figures/18` documents why.

### 2.8 PostgreSQL Sync (Serving Layer)

- Follow the HSI Gold sync template (TRD §2.2.6–2.2.7): **full refresh
  (truncate → bulk insert)** — the surface is a fresh recomputed snapshot,
  a single point in time, with no incremental semantics.
- One serving table for the fire-weather surface carrying: H3 cell id,
  parent hexes for multi-scale rendering, the three variant values, the
  percent-of-record value, and the run timestamp.
- Standard sync mechanics apply (Key Vault credentials, JDBC batch writes,
  row_count/run_status logging).

### 2.9 Presentation and Interpretation (React / Mapbox)

- One Fire Weather layer with a variant toggle (composite default; NOAA HWP
  and TX-HWP selectable).
- **Display percent-of-record per cell**, mapped to the TDIS portal's 0–2
  intensity convention (same convention as Precipitation HSI).
- The layer info panel must state: (1) this layer describes fire-weather
  *severity*, not ignition likelihood — for "where fires start," see the
  Ignition layer; (2) the NOAA variant is the published NOAA GSL HWP
  equation; (3) values are relative to each cell's own record.
- The Smokehouse Creek worked example (`DEMO_SMOKEHOUSE.md`) is the
  onboarding walkthrough for how a displayed number is produced.

---

## 3 What the Team Receives (handoff assets)

| Path | Contents |
|---|---|
| `scripts/fwi_config.py` | Single source of truth: all three formulations, both input paths, normalization refs. Import, don't re-derive. |
| `data/hwp_params.json` | Served parameters: TX-fit coefficients + p99.5 refs (NOAA 26.25, TX 20.10). Keep co-located with `fwi_config.py`. |
| `data/fwi_components_res5.parquet` | 3.38M res-5 cell-days of raw components (2024–26) — historical recompute without re-pulling weather. |
| `scripts/16_fit_hwp_tx.py`, `scripts/15_reweight_fwi.py`, `scripts/run_hwp_chain.sh` | Full historical chain (components → fit → arrays), reference lineage. |
| `31_hwp_validation.py`, `hwp_validation.json`, `overall_validation_2026-09-23.json`, `spec_validation_2026-09-23.json` | Validation harness + results (§4). |
| `DEMO_SMOKEHOUSE.md`, `figures/` | Worked example + event-day maps + the summer-saturation motivation for percent-of-record display. |
| `../serve/score_day.py` | The HRRR pull this product piggybacks on (one 12Z pull serves ignition + fire weather). |

Dependencies: Python with `herbie-data`, `xarray/cfgrib`, `h3`, `pandas`
(same environment as the ignition entrypoint; see `../requirements.txt`).

---

## 4 Validation and Acceptance Criteria

An implementation is accepted when it reproduces the packaged checks:

1. **Spec fidelity (bit-exact):** recomputed variants match
   `spec_validation_2026-09-23.json` on the packaged components.
2. **Worked example:** the Smokehouse Creek inputs reproduce TX 0.788 and
   NOAA 0.215 from the package location.
3. **Range:** all variants in [0, 1]; composite never clips; HWP variants
   clip ≈0.5% of cell-days.
4. **Coverage:** live surface covers every Texas cell of the pull.
5. **Input QC:** zero negative wind/VPD, zero out-of-range vpd/fm100, zero
   date gaps.
6. **Climatology sanity:** monthly means peak Mar–May, winter minimum.
7. **Named-event percentiles (NOAA variant, statewide, event day):**
   Smokehouse 70.1 · Windy Deuce 74.9 · Crabapple 86.6 · Lavender 90.9 ·
   Hunggate 91.8.
8. **Signal placement:** pooled fire-occurrence AUROC ≈ 0.50 (expected —
   see §2.1); within-cell, the top-1% most intense fires (VIIRS FRP) land at
   mean percentile 0.63–0.68 of their cell's record, 41–46% in the top
   quartile. Severity-timing, not occurrence, is the validated role.

**Open items for full production sign-off (recommended, not blocking):**
(1) cross-check a few days against NOAA GSL's published operational HWP maps;
(2) quantify gridMET-proxy vs exact-HRRR agreement on overlapping days;
(3) Red Flag Warning correspondence once RFW polygons are pulled.

---

## 5 ⛔ Hard Rule

**Never multiply or otherwise combine HWP with the ignition model's output.**
This was tested and degrades ignition AUC-PR by 18–61%. Ignition = "will a
fire start here" (learned, calibrated probability). Fire weather = "how severe
are the conditions" (physical index, relative scale). Serve them as two
separate layers, exactly as the TDIS dashboard does today.
