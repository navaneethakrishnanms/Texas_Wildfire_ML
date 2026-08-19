# TDIS Forecast — Sanity Check (Full Project Sweep)

**Date:** 2026-08-06 · **Scope:** database forensics, train/serve consistency, code
walkthrough, case-study re-derivation + one new event, credibility retrains
(label-shuffle + spatial holdout), honest forecast-value verdict.
**Written honestly and critically — this document exists to record what's wrong, what's
merely imperfect, and what genuinely holds.**

> Companions: `README.md` (summary) · `HANDBOOK.md` (how it works) · `VALIDATION.md`
> (event tests + metrics). Raw sweep logs: `sweep_phase1.log`, `sweep_phase6.log`.
> Sweep code: `scripts/sweep_phase1_forensics.py`, `scripts/sweep_phase6_credibility.py`.

---

## 0. Verdict in three sentences

The pipeline has **no hidden leakage** (label-shuffle control collapses to chance) and its
skill is **transferable geography, not cell memorization** (spatial holdout gap ≈ 0.005
AUC-PR) — the two strongest credibility results a reviewer can demand. The sweep found
**one new contained data defect** (6.1% of labels are out-of-state spillover — provably
never seen by the models), **one load-bearing analytic choice** (the 3% flare threshold
swings 18–37% of the labels), and **one documentation error** (Smokehouse hazard
percentile is 94th by the transparent method, not the documented 98th). Forecast weather
helps **modestly but unambiguously** (+0.011 AUC-PR controlled; better P/R/F1 at every
threshold) — worth serving, not worth overselling.

---

## 1. Findings, ranked by severity

### 🔴 F1 — 6.1% of "Texas" labels are New Mexico / Oklahoma (CONTAINED)
The VIIRS download used a rectangular bounding box (25.75–36.65°N, −106.7–−93.4°W) that
includes chunks of NM/OK — the famous **Ruidoso South Fork Fire (NM, June 2024)** and the
**March 2025 Oklahoma outbreak** are inside our label set: **42,368 clean-label fire-days
(6.1%)** fall outside the Texas static master.
**Why contained:** those rows have **100% null static features** (the TX-only master
doesn't cover them) and are **provably dropped** by `dropna(subset=STATIC)` before any
training (verified this sweep: all 185,336 such training rows null, 0 survive). The models
never saw them.
**What it does affect:** every label count quoted in docs (960k raw / 698k clean includes
~6% out-of-state), and any analysis run on raw labels without the TX-master join.
**Fix:** filter labels to TX-master membership at build time (`02_build_labels.py`), or
document the counts as "TX bounding box" not "Texas."

### 🟠 F2 — The flare threshold (3% persistence) is load-bearing
Sensitivity sweep of "cell is an industrial hotspot if detected >X% of all days":
| Threshold | Cells | Labels removed |
|---|---|---|
| >1% | 1,732 | **36.9%** of positives |
| >3% (current) | 531 | 27.3% |
| >10% | 145 | 18.0% |

A ±factor-3 change in the cutoff moves **~1/5 of the positive labels**. The 3% choice is
reasonable (>3% ≈ 138 fire-days in 12.5 yr — implausible for wildfire) but currently
**unvalidated against an independent source**. The per-cell positive-rate audit also shows
a residual tail (max 0.956 — cells burning "most days" that sit *below* 3%-of-record
because of short lifespans).
**Fix:** re-download VIIRS retaining the `type` flag (NASA's own "static land source"
classification) and cross-check our 531-cell list; report metrics at 1%/3%/10% as a
robustness band.

### 🟠 F3 — Class balance drifts across years, with a 2018 step-change
Positive rate: 0.145 (2014) → **0.177→0.239 jump at 2018** → 0.23–0.27 (2022–26).
The 2018 step coincides with **NOAA-20 coming online** (second VIIRS satellite ≈ doubles
detection opportunities) — a *sensor-era inhomogeneity in the label itself*, not a fire
trend. Consequences: (a) per-year AUC-PR comparisons are confounded (always quote lift);
(b) train (0.199) vs test (0.236) base rates differ, mildly flattering test AUC-PR;
(c) any "fires are increasing" reading of our label is partly instrument artifact.
**Fix (documentation-level):** note the satellite-era discontinuity wherever per-year
numbers appear; consider single-satellite (S-NPP-only) sensitivity check for trend claims.

### 🟠 F4 — Documented Smokehouse hazard percentile is wrong: 94th, not 98th
Independent re-derivation (fresh code, this sweep) reproduces VIIRS counts (3/184/3,529),
box hazard (0.568), ×state-mean (2.9×), and WHEN (75th) **exactly** — but the statewide
hazard percentile computes to **94th** (fraction of all 5,382 res-5 hazards below 0.568),
vs the **98th** in VALIDATION §3/DEMO. The discrepancy is percentile *methodology* in the
original event script, not data. Still top-decile; claims of "97th–100th WHERE" across
events need rechecking with one declared method.
**Fix:** standardize the percentile definition, recompute the event table, correct docs.

### 🟡 F5 — WHP and burn probability are near-duplicates (r = 0.94)
The hazard composite (0.45·WHP + 0.40·burn-prob) effectively puts **0.85 weight on one
signal measured twice**, and any feature-importance split between them in the ML model is
arbitrary. Not an error — both are TxWRAP/FSim outputs — but importance narratives should
treat them as one factor. Also notable: road_dist correlates −0.6 with both (fire-prone
land is remote land), which partially entangles the "human access" and "fuels" stories.

### 🟡 F6 — tmmx fill-value artifact: 25,500 rows at −53 °C (3.35% of non-null)
Physically impossible for Texas (record ≈ −23 °C). Trees split around it, so model impact
is minimal, but it should be nulled at build. All other variables pass literature bounds
**clean** (ERC ≤ 113 NFDRS max, VPD ≤ 9 kPa, wind ≤ 25 m/s daily-mean, RH 0–100, pr ≤ 650mm).

### 🟡 F7 — 1,183 cells have *partial* weather nulls
The 76.6% weather-null block is cell-structured (gridMET universe — expected), but 1,183
cells are partially null — likely gridMET tile-edge artifacts. Small (0.2% of cells);
worth a look before the "extend gridMET to all cells" job.

### 🟡 F8 — Matched-negative ratios are not uniform per cell
Per-fire-cell positive-rate: median 0.125 (~1:8 design), but p95 = 0.43 and max = 0.956.
High-rate cells are legitimate repeat-burn areas *and* sub-threshold persistent sources
(see F2 tail). The synthetic base rate is a *distribution*, not a constant — one more
reason outputs are ranks, not probabilities.

### Restated knowns (verified again, unchanged)
- Weather null on 76.6% of training rows (gridMET 317k-cell universe ⊂ 619k label cells)
  → the honest-vs-ceiling comparison remains not-apples-to-apples (VALIDATION §10 caveat).
- flep4/cfl dead (100% zero); cbd/cbh zero on 71%; whp/bp zero on 39% of *training* cells.
- Label semantics = fire **occurrence** (96% satellite), not strict ignition (HANDBOOK §19).

---

## 2. What PASSED (the clean bill items)

| Check | Result |
|---|---|
| **Label-shuffle negative control** | ✅ AUC-PR 0.226 ≈ base rate 0.234; AUROC 0.486 ≈ 0.5 → **no hidden pipeline leak** |
| **Spatial holdout (20% cells never trained on)** | ✅ unseen 0.446 AUC-PR / 1.91× vs seen 0.451 / 1.92× — memorization gap **0.005** → skill is transferable geography |
| Duplicate (cell,date) rows | ✅ 0 |
| Date continuity | ✅ labels: 4,593 consecutive positive days, no gaps; dynamic cache: 933/933 days, 3,624 cells every day |
| Physical bounds vs literature | ✅ all clean except tmmx artifact (F6) |
| **Train/serve VPD formula** | ✅ byte-identical Tetens (`05:68-72` vs `13:27-28`) |
| **Train/serve wind field** | ✅ same GRIB selector `:WIND:10 m above ground`, same rename chain, both sides |
| Train/serve spatial join | ✅ KDTree nearest-gridpoint both sides (`06:36-44` vs `13:69-71`) |
| Fire seasonality vs literature | ✅ March peak (102,743) + late-winter shoulder = documented TX cool-season wind-driven regime; Nov–Dec minimum |
| Smokehouse re-derivation | ✅ 4 of 5 documented numbers reproduce exactly (see F4 for the 5th) |
| **New 5th event: Windy Deuce** | ✅ validates (below) |

**One semantics note (consistent, not a bug):** the "daily" HRRR value is the **00Z UTC
snapshot** (≈ 6–7 pm CDT the prior evening), identically at train and serve — so no skew,
but the model's "tomorrow's weather" is start-of-day conditions, not the afternoon peak.
This is another face of the daily-resolution wind limitation (VALIDATION §6).

---

## 3. Train↔forecast workflow consistency (Phase 2 detail)

| Aspect | Training path (`05→06`) | Serving path (`13`) | Consistent? |
|---|---|---|---|
| Weather source | HRRR F24, herbie/AWS | HRRR F24/F48 (GFS retired) | ✅ (F48 = mild lead-time extrapolation, documented) |
| Fields pulled | `TMP/DPT 2m, WIND 10m, GUST sfc` (`05:112`) | same regex + MSTAV (`13:38`) | ✅ (MSTAV feeds FWI only, not the model) |
| VPD | Tetens from t2m/d2m (`05:68-72`) | identical fn (`13:27-28`) | ✅ |
| Wind | `max_10si/si10` → wind (`05:126`) | same rename (`13:54`) | ✅ |
| Valid time | 00Z snapshot of target day | 00Z snapshot | ✅ (semantics note above) |
| Cell join | KDTree cell→nearest gridpoint (`06:34-44`) | KDTree (`13:69-71`) | ✅ |
| Statics/temporal | from master + date (`03`) | same master + same encodings (`13:73-77`) | ✅ |
| Model file | trained `18/19` | `13` auto-prefers `*_hrrr_filtered.json` (`13:65-67`) | ✅ |

**Conclusion: no train/serve skew found.** (The sweep's pre-registered prediction that
this phase would surface a silent bug was wrong — recorded for honesty.)

---

## 4. Code walkthrough — where every number is born

Pipeline order, with the load-bearing lines:

1. **Labels** — `scripts/01_download_viirs.py` (FIRMS API → `data/labels_viirs/viirs_tx_h3.parquet`),
   `scripts/02_build_labels.py` (fuse FPA-FOD ∪ VIIRS → `data/labels_fused/ignitions_daily_tx.parquet`).
   ⚠️ F1 lives here (bbox, no TX-polygon filter).
2. **Training table** — `scripts/03_build_dataset.py`: matched negatives (same fire cells,
   other days; day pool at `03:73` — the phantom-date bug fixed downstream in 18), spatial
   controls, gridMET attach → `tdis_train_daily_tx.parquet`.
3. **Forecast weather attach** — `scripts/05_download_hrrr_forecast.py` (historical F24
   archive, ~14 GB, watchdog at `05:76-95`) + `scripts/06_attach_hrrr_forecast.py`
   (KDTree join `06:34-44`, per-date merge `06:52-64`) → `tdis_train_daily_hrrr.parquet`.
4. **Label cleaning + retrains** — `scripts/18_flare_filter_retrain.py` (flare cells
   `18:31-40`, phantom-date clip `18:47`, honest retrain `18:65-69` — 400 trees, no early
   stopping, note at `18:62-64`) and `scripts/19_retrain_hrrr_filtered.py` (operational
   retrain + dashboard ignition-layer refresh `19:75-105`).
   Ceiling: `scripts/22_retrain_ceiling_filtered.py`.
5. **Inference (the live forecast)** — `scripts/13_model_forecast_day.py`: HRRR pull
   (`13:30-42`), grid→cell join (`13:69-71`), temporal features (`13:73-77`),
   `predict_proba` on 1,708,940 × 20 (`13:79-80`), res-8→res-5 means (`13:82-83`).
6. **Fire-weather knob** — `scripts/fwi_config.py` (weights, FWI_MODE, HWP forms);
   arrays rebuilt by `scripts/15_reweight_fwi.py`; TX-HWP fit `scripts/16_fit_hwp_tx.py`.
7. **Serving** — `scripts/08/09` (dashboard data), `scripts/17_embed_standalone.py` (v1),
   `scripts/20_build_explorer_data.py` + `scripts/21_embed_explorer.py` (v2 zoom-adaptive).
8. **Validation** — `scripts/14_event_validation.py` (4-event battery),
   `scripts/sweep_phase6_credibility.py` (shuffle + spatial holdout).

---

## 5. Case studies (Phase 4)

**Smokehouse Creek (triple-check, independent code path):** VIIRS 3 → 184 → 3,529
(Feb 25/26/27) ✓ exact; box hazard 0.568 ✓; 2.9× state mean ✓; WHEN 75th (composite) ✓,
80th (NOAA HWP), 75th (TX HWP). Hazard percentile corrected to **94th** (F4).

**Windy Deuce (NEW, event #5)** — Moore County; ignited Feb 26 2024 ~18:20 CST near US-87
(downed power line, TAMFS investigation); 144,000+ acres; Fritch evacuated.
- **Captured ✅** — detections 46 → 47 → **255**: the spike lands Feb 27 *UTC*, exactly
  right for an 18:20 CST Feb 26 ignition (00:20 UTC Feb 27). The satellite timing physics
  self-consistently confirms the documented ignition hour. (Feb 25 baseline ≈ 46: raw
  VIIRS in this box includes Moore Co. industrial sources — the battery uses raw
  detections; label-side these are flare-filtered.)
- **WHERE ✅** — box hazard 0.490 = **86th percentile** (2.5× state mean).
- **WHEN 🟡** — 78th (composite), 78th (NOAA), 73rd (TX): same signature as every
  wind-driven cool-season event — daily-mean wind under-scores gust-driven days
  (VALIDATION §6's finding, now confirmed on a fifth independent event).

Updated event scoreboard: **capture 5/5 · WHERE 5/5 top-quintile (86th–100th) · WHEN:
heat/drought-driven 99th (2/2), wind-driven 73rd–94th (3/3 under-scored, mechanism known).**

---

## 6. How much does forecast weather actually help? (honest verdict)

All measurements, one place (clean labels, identical rows where stated):

| Evidence | Value |
|---|---|
| Controlled ablation (identical rows, HRRR on/off) | **+0.011 AUC-PR / +0.020 AUROC** |
| Clean retrains: honest → operational | 0.450 → 0.483 AUC-PR (+7% relative), lift 1.92× → 2.06× |
| Threshold sweep | operational beats honest at **every** threshold 0.2–0.8, on P and R and F1 simultaneously |
| F1 @ 0.5 | 0.463 → 0.488 (+0.026) |
| Feature importance | 3 HRRR features ≈ 15%, `hrrr_vpd` top-5 |
| Operationally | at threshold 0.5, ~2.7 more true fire-days per 100 flags; ~2 pp more recall |

**Plain statement:** forecast weather provides a **small, real, and fully robust**
improvement — it survives the ablation, dominates at all thresholds, and rests on a
physically sensible feature (forecast dryness). It is **not transformative**: geography
still carries ~85% of the signal, and the daily/00Z-snapshot resolution leaves most of the
wind signal unharvested (the sub-daily roadmap item is where the bigger weather payoff
lives, per both our TX-HWP fit and the NOAA HWP literature). Keeping it operational is
justified — the honest sales pitch is "a real edge plus true forward-forecast capability,"
not "weather-driven prediction."

---

## 7. File paths — data

| Path | Contents |
|---|---|
| `tdis_train_daily_tx.parquet` | training table, raw (3.60M rows) |
| `tdis_train_daily_tx_flarefiltered.parquet` | **canonical training table** (3.24M rows, flare+phantom filtered) |
| `tdis_train_daily_hrrr.parquet` | + hrrr_tmp/vpd/wind (F24 forecast attach) |
| `data/labels_fused/ignitions_daily_tx.parquet` | fused FPA-FOD ∪ VIIRS labels (960k; ⚠️ includes 6.1% out-of-state — F1) |
| `data/labels_fused/flare_cells.parquet` | 531 persistent-hotspot cells (the flare filter) |
| `data/labels_viirs/viirs_tx_h3.parquet` | raw VIIRS detections (1.62M, incl. FRP) |
| `data/static_features/tx_static_master.parquet` | 1,708,940 res-8 cells × 15 statics |
| `data/fwi_components_res5.parquet` | per-day weather components (erc/vpd/vs/fm100), 3.38M cell-days |
| `data/hwp_params.json` | TX-HWP fitted coefficients + normalization refs |
| `data/weather_hrrr_forecast/hrrr_24h/*.parquet` | historical HRRR F24 archive (~14 GB) |
| `../gridmet_tx/{var}_{year}_tx_cells.parquet` | observed daily weather source |

## 8. File paths — models, dashboards, docs, sweep artifacts

| Path | Contents |
|---|---|
| `models/tdis_forecast_baseline_honest_filtered.json` (+meta) | honest model (static+calendar) — **current** |
| `models/tdis_forecast_hrrr_filtered.json` (+meta) | operational model — **served by 13** |
| `models/tdis_forecast_baseline_ceiling_filtered.json` (+meta) | ceiling (observed-weather diagnostic) |
| `models/tdis_forecast_baseline_{honest,ceiling}.json`, `models/tdis_forecast_hrrr.json` | pre-flare-fix (provenance only) |
| `models/baseline_forecast_predictions_test_filtered.parquet` | clean test predictions |
| `dashboard/tdis_fire_dashboard_standalone.html` | v1 daily dashboard (61 MB, drag-drop) |
| `dashboard/tdis_fire_explorer_standalone.html` | v2 zoom-adaptive res 4→8 (102 MB) |
| `dashboard/tdis_dashboard_data.json`, `dashboard/forecast_*.json` | served data + live forecasts (24/48h) |
| `README.md` / `HANDBOOK.md` / `VALIDATION.md` / `DEMO_SMOKEHOUSE.md` / `PRESENTATION.md` | docs |
| `SANITY_CHECK.md` (this file), `sweep_phase1.log`, `sweep_phase6.log` | sweep artifacts |
| `scripts/sweep_phase1_forensics.py`, `scripts/sweep_phase6_credibility.py` | sweep code (rerunnable) |

## 9. Literature supporting the methodology

**Labels & fusion.** Short, K.C. (2014, updated) *Spatial wildfire occurrence data for the
US (FPA-FOD)*, USFS RDS-2013-0009 — the agency-record standard. Schroeder, W. et al.
(2014) *The New VIIRS 375 m active fire detection data product*, Remote Sens. Environ. —
the satellite label standard. **Fusco, E.J. et al. (2019)** *Detection rates and biases of
fire observations from MODIS and agency reports in the conterminous United States*, Remote
Sens. Environ. 220 — quantifies the complementary blind spots (<25%/<50% mutual coverage)
that motivate fusing the two; our 0.1% corroboration overlap is the expected behavior, not
a defect. Huot, F. et al. (2022) *Next Day Wildfire Spread*, IEEE TGRS — VIIRS-as-ML-label
precedent. Andela, N. et al. (2019) *The Global Fire Atlas*, Earth Syst. Sci. Data —
satellite-derived ignition products as published datasets.

**Fire-weather drivers.** Bradshaw, L. et al. (1983) *The 1978 NFDRS* (ERC); Fosberg &
Deeming (1971) (dead-fuel moisture); Srock, A. et al. (2018) *The Hot-Dry-Windy Index*,
Atmosphere (VPD+wind); Rothermel (1972) USDA INT-115 (wind/slope in spread); Higuera &
Abatzoglou (2021) PNAS (VPD dominance); **James, E.P. et al. (2025)** *An Hourly Wildfire
Potential Index…*, Wea. Forecasting, doi:10.1175/WAF-D-24-0068.1 (our NOAA HWP variant,
implemented exactly; also the evidence that wind's value needs hourly resolution — matching
our TX fit's dryness-dominance at daily resolution).

**Hazard layer.** Finney, M. et al. (2011) *FSim* (burn probability simulation);
Dillon, G. et al. (2015) *Wildfire Hazard Potential* (WHP); TxWRAP = the Texas
implementation. Our hazard composite is a weighted blend of these published products.

**Human-ignition geography & temporal encodings.** Balch, J. et al. (2017) PNAS (84%
human-started; weekly/holiday cycles — our Sunday-minimum matches their work-week
pattern); Syphard, A. et al. (2007) Ecol. Appl. (roads/WUI proximity); Parisien & Moritz
(2009) Ecol. Monogr. (environmental controls on fire regimes).

**Weather data.** Abatzoglou, J.T. (2013) Int. J. Climatol. (gridMET); Dowell, D. et al.
(2022) Wea. Forecasting (HRRR description); James et al. (2025) for HRRR fire-weather use.

**Evaluation discipline.** Temporal (past→future) splits and base-rate-aware metrics
(AUC-PR vs prevalence; lift as the base-rate-robust comparison) follow standard forecast-
verification practice; the controlled-ablation and negative-control (label-shuffle)
designs are standard ML-auditing tools. *(Citation years/venues above are from working
knowledge and the searches logged in this project; re-verify exact volumes before formal
submission.)*

## 10. Recommended actions from this sweep (priority order)

1. ✅ DONE (2026-08-06) — **Correct the WHERE percentiles** in VALIDATION §3 / DEMO with one declared method (F4) — quick.
2. ✅ DONE (code patched; applies on next label rebuild) — **TX-polygon filter in `02_build_labels.py`** + correct label counts in docs (F1) — quick.
3. ✅ DONE (code patched; applies on next dataset rebuild) — **Null the tmmx fill values** at build (F6) — quick.
4. **FIRMS `type`-flag cross-check** of the flare list + report 1%/3%/10% robustness band (F2) — needs re-download.
5. ✅ DONE — **Document the 2018 NOAA-20 label step-change** wherever per-year numbers appear (F3) — quick.
6. ✅ DONE — Treat WHP+burn-prob as one factor in importance narratives (F5) — writing change.
7. ✅ DONE (VALIDATION §11) — Add the label-shuffle + spatial-holdout results to VALIDATION as the standing credibility certificate — quick.
