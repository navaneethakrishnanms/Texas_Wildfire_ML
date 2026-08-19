# TDIS Forecast — Dataset & Model Plan (Texas, 24-hour ignition forecast)

**Goal:** a daily-refreshed dashboard showing **forecast wildfire ignition risk for the next
24 h (primary), 48 h / 72 h (stretch)** across Texas, H3-8 hexagons.
**Status:** SCOPE CONFIRMED (2026-07-28). Ready to start Phase 0 probe.

## LOCKED DECISIONS (all parameterized — easy to change later)
| Knob | Value now | Change later? |
|---|---|---|
| Region | Texas | config bbox |
| Label span | FPA-FOD 2014–2020 (have it) + VIIRS 2014–2024 + GOES 2017–2024 | config years |
| Causes | ALL wildfire causes (human/equipment/unknown/lightning); no RX class present to exclude | config filter |
| Size threshold | none (keep `max_size_acres` as a field) | config threshold |
| **Forecast horizon** | **24 h primary; 48 h / 72 h as knobs** (`HORIZON_HOURS=[24,48,72]`) | config list |
| **Time resolution** | **per-cell per-DAY (dropped 6-h window)** | `WINDOW_HOURS=24`; set to 6 to restore sub-daily — designed as a knob from day 1 |
| Deployment | **offline build first**; operational host TBD (documented as open decision) | — |

FPA-FOD TX is already in hand: `fpa_fod_tx_h3.parquet`, 36,182 ignition events 2014–2020
(human 16,691 · unknown 10,225 · equipment 7,492 · lightning 1,774; sizes 1–318,156 ac).
Storage 42 TB free · compute 128 cores / 995 GB RAM · GEE authenticated. All green.

All work (data, models, scripts, dashboard) lives under this `TDIS_Forecast/` folder.

---

## 1. What makes this different from the current model

The current v5 model is a **static susceptibility map** (QC: 92% spatial). A 24-h *forecast*
is a **dynamic ignition-risk** problem. We keep v5 as the static base and add the two things
it never had: **real ignition timestamps** (for a real temporal target) and **forecast
weather** (for a real forecast).

```
Forecast ignition risk(cell, next 24h)
  = Stage-1 STATIC susceptibility   (v5_tuned: fuels, roads, terrain)     ← reuse
  × Stage-2 DYNAMIC ignition-danger (forecast weather, lightning,          ← build
                                     curing, human-activity timing)
```

---

## 2. Label strategy — fuse three sources (each fixes the others' gaps)

| Source | Gives | Gap it has | Role in fusion |
|---|---|---|---|
| **FPA-FOD** | complete inventory: where + **day**, cause, size (1992–2020) | times 87% imputed; ends 2020 | base inventory of ignition events |
| **VIIRS** (FIRMS) | real UTC time, 375 m, ~4×/day (2012→present) | misses small fires | attach real timestamps; extend past 2020 |
| **GOES-16/18** | real ~5-min time, 2 km (2017→present) | coarse; misses small fires | sub-hourly timing for matched fires; live feed |

**Fusion logic:** FPA-FOD event → search VIIRS/GOES detections in same H3 cell within ±1 day
→ if matched, assign the satellite's real detection time; else keep day-level (flag as
`time_source = imputed`). Post-2020, VIIRS/GOES detections themselves become new ignition
events (recovers the 2021+ label gap that blocked forward validation).

**Label = 1** if an ignition (real or satellite-detected) occurred in (cell, time-window).

---

## 3. Feature strategy

| Group | Features | Source | Static/Dynamic |
|---|---|---|---|
| **Susceptibility (Stage-1)** | avg_burn_prob, whp, cfl, flep4, cbd, cbh, road_dist, ecoregion, elevation, slope, aspect | TxWRAP/LANDFIRE + geography (reuse) | static |
| **Fire weather (train)** | erc, fm100, vpd, vs, rmax, rmin, tmmx, pr + rolling | gridMET (observed) | dynamic |
| **Fire weather (forecast)** | same variables, **predicted** | **HRRR / NBM** | dynamic — runtime |
| **Lightning** | strike density / convective threat (natural ignition) | **GOES-GLM** (free, AWS) | dynamic |
| **Fuel curing** | NDVI / greenness anomaly | MODIS via GEE | slow-dynamic |
| **Human-activity timing** | day-of-week, holiday, hour-of-day | derived | dynamic |
| **Dryness anomaly** | today's ERC/VPD vs cell climatology; days-since-rain | derived from gridMET | dynamic |

> Note on lightning: use **GOES-GLM** (free/public on AWS), NOT NLDN (commercial). Correction
> from earlier notes.

---

## 4. Training-table schema (one row per cell × time-window)

```
h3_cell, window_start_utc, window_hour,
label,                       # 1 = ignition in this cell-window
time_source,                 # 'viirs' | 'goes' | 'imputed'  (label quality flag)
<susceptibility features>,   # static
<fire-weather features>,     # observed gridMET for training
<lightning, ndvi, anomaly, human-timing features>,
split                        # train / val / test  (temporal)
```

**Negative sampling (critical fix):** draw negatives as the **same fire-prone cells on
non-fire windows** (case-control matched on location), so the model is forced to learn
*when*, not *where*. This is the fix for the "92% spatial" problem.

---

## 5. Model architecture

1. **Stage-1 (have it):** v5_tuned static susceptibility → per-cell base score.
2. **Stage-2 (build):** XGBoost (or similar) on the dynamic features + Stage-1 score as an
   input, trained on real-timestamp labels with matched negatives.
3. **Forecast product:** at runtime, feed **HRRR/NBM forecast** weather for the next 24/48/72 h
   → Stage-2 → per-cell forecast ignition-danger.

**Evaluation:** conditional / within-cell temporal AUC-PR (does it rank the right *windows*),
not just overall AUC-PR. Plus reliability by lead time (24 vs 48 vs 72 h). Hold out the most
recent year(s) temporally.

---

## 6. Data acquisition plan & FIRMS transaction budget

**Respect the 5,000 transactions / 10-minute limit.** Estimated needs:

| Dataset | Method | Volume / budget |
|---|---|---|
| VIIRS (TX, 2014–2024, SNPP+NOAA20) | **Bulk download tool** (no API burn) OR API in 10-day windows | ~800 API calls if API; **prefer bulk tool** to save the key for live use |
| VIIRS live feed (last 1–7 days) | API (`VIIRS_*_NRT`) | a few calls/day — this is what the key is really for |
| FPA-FOD | one-time manual download (USDA RDS) | single file (national geodatabase) |
| GOES-GLM + GOES-FDC | AWS S3 (`--no-sign-request`), filter to TX bbox | large; filter on download |
| HRRR forecast | AWS S3, TX subset, next 24–48 h | daily job, small per-run |
| NDVI | GEE (already authenticated) | batched, ~2–3 h one-time |
| gridMET (historical) | already downloaded (reuse) | 0 |

**Policy:** historical bulk pulls via the bulk tool / AWS (no API burn); reserve the FIRMS
**API key for the daily live feed**. Throttle any API loop to well under 5,000/10 min and
check the status page.

---

## 7. WHAT I NEED FROM YOU (decisions + access) — before any download

### Decisions (scope)
1. **Label span & sources:** OK to build labels as FPA-FOD (2014–2020) **fused with** VIIRS
   (2014–2024) + GOES (2017–2024)? (VIIRS/GOES also fill the 2021+ gap.)
2. **Ignition definition:** include **all wildfire causes** (human + natural), **exclude
   prescribed/RX burns**, **no minimum size** (but keep size as a field)? ← proposed default
3. **Forecast horizon:** 24 h primary, 48/72 h stretch — confirm.
4. **Target resolution:** keep H3-8 (~0.74 km²) + 6-hour windows? (Finer time needs GOES-heavy.)
5. **Live cadence & host:** the "forecast" only lives if a **daily job** runs somewhere that
   can pull HRRR each morning. Where will that run (this NAS? a server with internet + cron)?
   For now we build offline; but decide the operational home.

### Access / data you provide or confirm
6. **FIRMS map key** — ✅ received (stored, owner-only).
7. **FPA-FOD download:** it's a manual USDA RDS download (national geodatabase). Can you
   download it, or confirm I may fetch it programmatically if a direct URL exists?
8. **NASA Earthdata login:** the FIRMS *bulk* archive tool may require a free Earthdata
   account. Do you have one (username), or should we route everything through the API?
9. **GEE project:** you authenticated earlier — confirm it's still active for NDVI pulls.
10. **Storage OK?** GOES/HRRR are large even filtered; confirm we can use this NAS space.
11. **Compute:** any limit on cores/RAM for the daily forecast job?

### Verify
12. I'll respect the transaction limit and check the status page before/after pulls.

---

## 8. Proposed execution order (once §7 is confirmed)
1. **Phase 0 probe** (cheap): pull ~1 year of VIIRS for TX, match to FPA-FOD → confirm
   coverage + whether weather separates at real times. Go/no-go.
2. **Phase 1:** full VIIRS + GOES + FPA-FOD fusion → real-timestamp label set.
3. **Phase 2:** assemble training table (labels + static + dynamic features, matched negatives).
4. **Phase 3:** train Stage-2 dynamic model; evaluate conditionally by lead time.
5. **Phase 4:** wire HRRR forecast feed; generate 24/48/72 h forecast.
6. **Phase 5:** forecast dashboard (H3 hexbin, 24/48/72 h tabs, daily refresh).
