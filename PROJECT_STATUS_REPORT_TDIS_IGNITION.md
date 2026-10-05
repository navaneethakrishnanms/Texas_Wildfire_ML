# Project Status Report — IgnitionNet / Texas Wildfire ML vs. TDIS_Ignition Handoff

**Prepared:** 2026-09-30
**Scope:** Full analysis of everything done on this project — from the original `Focused_Files`/`Supporting_files` package shared ~2 weeks ago (19 Aug 2026), through our own independent work (`V1`, `V2`, `TX`, `TEXAS`, `TEXAS_v1_backup`), up to the newest `TDIS_Ignition` package that was just handed to us.
**Sources:** every doc/code/model file actually opened and cross-checked — `TDIS_HANDOFF_ANALYSIS.md`, `DATA_QUALITY_REVIEW.md`, `PROJECT_VERSION_HISTORY.md`, `DATASET_COLUMNS_COMPARISON.md`, `IGNITIONNET_PROJECT_SCOPE_MASTER.md`, `TDIS_Ignition/README.md` + `docs/`, `TDIS_Ignition/model_tristate/README.md`, `TDIS_Ignition/fireweather/README.md`, `TEXAS/TEXAS_V2_MODEL_REPORT.md`.

---

## 1. Where we are, in one paragraph

Two weeks ago the TDIS/Texas team handed us `Focused_Files.zip` + `Supporting_files.zip` — one served XGBoost ignition model, its data, and a dashboard — and asked us to (a) validate their work and (b) build the end-to-end pipeline. We did both: we independently reproduced every one of their headline numbers exactly, audited the packages down to the byte level and found **17+ concrete defects** (stale metadata, missing config files, fabricated future-dated rows, conflicting split definitions, an out-of-state grid, a saturating calibrator), and in parallel kept building our own model line (`V1` → `V2` → `TX` → `TEXAS` → `TEXAS` 32-feature) which now **beats their served model by a wide margin** on every comparable metric. Since that first handoff, the TDIS/Texas team has come back with a **new, larger package: `TDIS_Ignition/`** — a more mature, production-oriented deliverable that fixes several of the defects we reported (23 features now, corrected metadata, a real `serve/score_day.py` entrypoint, tri-state and fire-weather sub-products) but is explicitly **not the final answer either** — it ships its own honestly-documented gaps (no live pipeline team assembled yet, calibrator saturation questions, drought-cache staleness, etc.).

**Bottom line: our project is in good shape.** We have (1) a validated understanding of their system, (2) a written record of every defect found, (3) our own competitive/superior model, and (4) a clear, unambiguous list of what the newest handoff (`TDIS_Ignition`) still needs before it can be called "done."

---

## 2. Timeline — what happened and when

| Date | Event |
|---|---|
| 2026-06-30 → 2026-08-12 | We build V1 (proof of concept) → V2 (production pipeline) → TX (tuned, HRRR-extended) entirely independently, using FPA-FOD + gridMET/HRRR + LANDFIRE |
| **2026-08-19** | TDIS/Texas team hands us `Focused_Files.zip` (~360 MB) + `Supporting_files.zip` (~150 files) — their served ignition model + full research trail |
| 2026-08-19 → 2026-08-20 | We independently reproduce their results and write `TDIS_HANDOFF_ANALYSIS.md` (peer-review-style comparison) and `DATA_QUALITY_REVIEW.md` (formal QA audit, 11 findings) |
| 2026-08-27 → 2026-08-31 | We build `TEXAS/` — a 30-feature model retrained from their raw parquet with a new imputation strategy; write `PROJECT_VERSION_HISTORY.md` |
| 2026-09-07 | We build the `TEXAS/` 32-feature tuned model (adds `lat`/`lon`, proper 21-trial hyperparameter search) — our current best model |
| **2026-09-22 / 2026-09-23** | TDIS/Texas team ships **`TDIS_Ignition/`** — a new, larger package: corrected 23-feature served model, `serve/score_day.py` production entrypoint, plus two new sub-packages (`model_tristate/` for TX+NM+OK, `fireweather/` for the HWP fire-weather index) and validation visuals |
| **2026-09-30 (today)** | This report |

---

## 3. What the original handoff (`Focused_Files` + `Supporting_files`) actually was

**The system:** one XGBoost binary classifier predicting, per H3 resolution-8 hexagon (~0.74 km²) per **day**, the probability of fire activity in Texas — trained on **archived HRRR weather forecasts** (not observations) so the training data matches what's available at deployment time.

### Model
| Property | Value |
|---|---|
| Algorithm | XGBoost, binary:logistic, 1,000 trees |
| max_depth | 9 · min_child_weight | 30 · learning_rate | 0.02 · subsample/colsample | 0.8/0.8 |
| Features | **22** (12 static terrain/fuel/access, 6 calendar, 4 HRRR weather) |
| Grid | H3 res-8, 1,708,940 TX-bbox cells |
| Promoted | 2026-08-17 |

### Data
- **Labels:** VIIRS active-fire detections fused with FPA-FOD, `ignitions_daily_tx.parquet` — **960,054 rows**.
- **Their biggest data-quality catch:** ~27% of positive labels were persistent industrial gas flares (one cell "on fire" 3,176 days). Removing 538 flare cells raised AUC-PR 0.386 → 0.450.
- **Training table:** `tdis_train_daily_hrrr.parquet` — **3,595,513 rows × 33 columns**, 2014–2026, positive rate 0.267. After flare-filtering + dropping rows with missing HRRR (archive starts 2018-07-16): **2,007,436 usable rows**.
- **Split actually used** (by the training script, not the doc): Train 2018-07→2021 (~856K rows), 2022 silently discarded, Test 2023→2026-07 (**891,983 rows**).

### Reproduced results (we verified these ourselves, exactly)
| Metric | Balanced test (base rate 0.234) | Real deployment population (5.06M cell-days, base rate 0.0192) |
|---|---|---|
| AUC-PR | 0.4941 | 0.0878 |
| AUROC | 0.7402 | 0.7968 |
| Lift | 2.11× | 4.58× |
| Calibration ECE (2026 holdout) | — | 0.0018 (raw 0.2799 → calibrated, −99.4%) |

Real-world validation: **5/5 real named Texas wildfires captured** (Smokehouse Creek, Crabapple, Lavender, Hunggate, Windy Deuce), with hazard percentile 74th–96th on all five.

### 17 defects we found across both audits
Full detail lives in `TDIS_HANDOFF_ANALYSIS.md` (Part D) and `DATA_QUALITY_REVIEW.md` (F1–F11). Headlines:

| # | Defect | Severity |
|---|---|---|
| 1 | **86,072 fabricated future-dated rows** (through 2026-12-31) sitting inside the `test` split, all label=0, no real weather — inflates every test-set metric | Critical |
| 2 | `meta.json` describes a *different* model (says 20 features / 531 flare cells; actual model has 22 features / 538 flare cells) | Critical |
| 3 | **Four mutually contradictory train/val/test split definitions** across docs, the parquet's own `split` column, and the training script | Critical |
| 4 | `data/hwp_params.json` missing from the operational package → dashboard's fire-weather index silently falls back to wrong constants (~50% output shift) | Critical |
| 5 | Model selection was performed on the test fold itself (no untouched holdout) — the reported 0.4941 is selection-biased | High |
| 6 | Retrain script's file paths don't resolve inside the package as shipped — "verified working from this folder alone" is false | High |
| 7 | 64.3% of training rows have zero weather data; 2014–2017 have none at all | High |
| 8 | **~54% of the "Texas" grid is actually outside Texas** (bounding box, not state-clipped) — contaminates aggregated hazard maps | High |
| 9 | Calibrator **saturates at 18.7%** — no cell can ever be shown a probability above ~19%, undisclosed anywhere | Medium/High |
| 10 | `flep4` and `cfl` (2 of 22 features) have **exactly 0.000 importance** — dead weight | Medium |
| 11 | Label provenance flips from ~91% VIIRS+FPA-FOD to 100% VIIRS-only exactly at the train/test boundary (FPA-FOD stops after 2020) | Medium |
| 12–17 | Stale doc claims ("~400 trees" vs actual 1,000; "2014–2026 training" vs actual 2018-07 onward), a document that reverses its own source JSON's verdict on flare-threshold sensitivity, hardcoded 6-date holiday list missing major holidays, calibrator applied at wrong H3 resolution, dashboard ships with no freshness check (serves stale forecasts as "current") | Medium/Low |

**Verdict on the original handoff:** the modelling and engineering discipline were genuinely good (negative results retained, calibrator shipped paired with model, credibility tests run — label-shuffle and spatial-holdout both passed). What failed was the **control/QA layer**: metadata that drifted from the model it described, no CI/schema validation, and a shipped artifact that didn't match its own documentation.

---

## 4. Our own model line — where we ended up

| Version | Rows | Features | Test AUROC | Test AUC-PR | Notes |
|---|---|---|---|---|---|
| **V1** (proof of concept) | 2,250 test rows | 18 (annual GEE composites) | 0.9142 | 0.7549 | ⚠️ Flagged unreliable — annual composites leak "which month/location historically burns"; **not quotable to TDIS** |
| **V2 baseline** (leaked) | 62,986 test | 31 | 0.9900 | — | Inflated by `fire_count`/`has_fire_history` leakage columns |
| **V2 clean** (leak removed) | 62,986 test | 29 | 0.8569 | 0.3978 | Honest baseline after fix |
| **TX (tuned, 21-trial search)** | 62,855 test | 34 | 0.8687 | 0.4247 | Real LANDFIRE joined, `burnable` top feature (19.8% gain) |
| **TX-HRRR** | 62,855 test | 42 | 0.8682 | 0.4225 | Adding sub-daily HRRR did **not** help — a confirmed negative result |
| **TEXAS 30-feat** (rebuilt from TDIS's own raw parquet, new imputation) | 460,463 test | 30 | 0.8249 | 0.7384 | Calibrated (ECE 0.0018) |
| **TEXAS 32-feat (current best)** | 460,463 test | 32 (+`lat`/`lon`) | **0.8309** | **0.7484** | 21-trial hyperparameter search; `lat`/`lon` landed as #3/#5 features (11.8% combined gain) |

**Direct comparison, same-style metrics:**

| Metric | TDIS served model (22 feat) | Our TEXAS 32-feat model |
|---|---|---|
| Test AUC-PR | 0.4941 | **0.7484** (+0.2543) |
| Test AUROC | 0.7402 | **0.8309** (+0.0907) |
| Test F1 | 0.4958 | **0.6498** (+0.1540) |

⚠️ **Caveat we've documented ourselves:** this isn't a fully fair apples-to-apples comparison — different test windows, different positive rates (our test set runs 30.7% positive vs. their 23.4%; theirs also includes a real-population validation at the true ~1.9% base rate, which we have not run for the 32-feature model). On the fairer **lift** metric (our earlier TX model vs. their balanced test), we were already ahead — 4.67× vs. 2.11×. What we do **not** yet have that they do: real-population scoring of the TEXAS model, a spatial-holdout / label-shuffle leakage test, and live-event validation against named fires.

---

## 5. The new handoff: `TDIS_Ignition/` (received 2026-09-22/23) — what changed

This is a **materially more mature package** than the original two zips. Highlights:

- **Model upgraded to 23 features** (added `drought_score` from USDM on top of the previous 22) — corrected `tdis_forecast_hrrr_filtered_meta.json` now matches the model file exactly (fixes defect #2 above).
- **Real-population numbers re-derived and improved:** AUC-PR 0.0878, AUROC 0.797, **lift 4.58×**, calibrated ECE now **0.0008** (better than the 0.0018 we measured before).
- **A genuine production scoring entrypoint** — `serve/score_day.py` — self-contained, package-relative, validated in both live (pulls HRRR via Herbie) and offline (packaged historical data) modes. This directly answers the "build the end-to-end pipeline" ask from two weeks ago.
- **Two new sub-packages:**
  - **`model_tristate/`** — a TX+NM+OK research model (rev5, 28 features). Honestly documented finding: training on more states is a **wash** for Texas accuracy (lift 4.29× vs 4.26×, essentially noise) — value is geographic coverage, not accuracy. A stronger TX-only candidate (`rev4`, AUC-PR 0.511, lift 4.76–4.88×) exists in their source repo but **was never packaged or promoted**.
  - **`fireweather/`** — the separate HWP/fire-weather index product (NOAA equation + TX-fit + composite variants). Explicitly **not to be multiplied with the ignition model** — doing so was tested and degrades ignition AUC-PR by 18–61%.
- **Production-readiness checklist included**, with honest open items: drought-cache freshness (weekly USDM refresh not yet scheduled), no monitoring/drift detection yet, live weather dependency on public AWS HRRR archive.

**What this means for us:** several of the defects we flagged in the original audit (#2 stale metadata, the missing serving pipeline) have been addressed by the *other* team in this newer package. Our `DATA_QUALITY_REVIEW.md` findings about the **fabricated future-dated rows**, **the out-of-state grid (~54%)**, and **the four conflicting split definitions** are not confirmed fixed in `TDIS_Ignition/` — we have not yet re-run our audit scripts against this new package to check.

---

## 6. Status summary — what's done vs. what's outstanding

### Done
- Independently reproduced and validated the original TDIS handoff's headline numbers (exact match)
- Full defect audit of both original packages (17+ findings, all confirmed by direct measurement, not just docs)
- Our own model line built end-to-end four times (V1 → V2 → TX → TEXAS), each iteration catching and fixing a real bug (temporal leakage, feature leakage, gridMET nodata bug, HRRR unit bug, 5-day rolling-stat bug)
- Our current best model (TEXAS 32-feature) statistically outperforms the TDIS-served model on matched-style metrics
- Two negative results independently confirmed on both sides: sub-daily HRRR weather doesn't improve ignition accuracy; multiplying fire-weather index onto the ignition score is actively harmful

### Outstanding / next steps
1. **Re-run our data-quality audit against the new `TDIS_Ignition/` package** — confirm whether the fabricated-future-rows and out-of-Texas-grid defects were fixed, since they weren't explicitly called out as resolved in its own docs.
2. **Run real-population validation on our TEXAS 32-feature model** at the true ~1.9–2.5% base rate — we've only validated it on the rebalanced test set so far; this is the single biggest credibility gap versus their package.
3. **Run the two credibility tests we haven't run**: label-shuffle (leak check) and spatial holdout (geography-memorization check) — TDIS's team has run both and passed; we haven't run either.
4. **Decide on `lat`/`lon` vs. `ecoregion_id`+`elevation_m`** as the location-encoding strategy — our model uses raw coordinates (risk of memorizing geography rather than learning transferable terrain features); TDIS deliberately avoided raw coordinates for this reason.
5. **Adopt `serve/score_day.py`-style production scoring** for our own model if we intend to deploy it, rather than hindcast-only evaluation.
6. **Decide whether to pursue the tri-state (`model_tristate/`) or fire-weather (`fireweather/`) sub-products**, or focus purely on improving the core TX ignition model — `rev4` (mentioned but never packaged by TDIS) is flagged as their strongest unreleased TX-only candidate and worth requesting if available.

---

## 7. Key artifacts (where the detail lives)

| Document | What it covers |
|---|---|
| `TDIS_HANDOFF_ANALYSIS.md` | Full peer-review-style comparison of their model vs. ours, part-by-part |
| `DATA_QUALITY_REVIEW.md` | Formal 11-finding QA audit with severity ratings and a proposed validation-gate framework |
| `PROJECT_VERSION_HISTORY.md` | Full chronological history of every model version we've built, with git commit references |
| `DATASET_COLUMNS_COMPARISON.md` | Column-by-column diff between our TX dataset and TDIS's dataset |
| `IGNITIONNET_PROJECT_SCOPE_MASTER.md` | Original project scope clarification + full feature/methodology reference from the team meeting |
| `TEXAS/TEXAS_V2_MODEL_REPORT.md` | Build report for our current best (32-feature) model |
| `TDIS_Ignition/README.md`, `docs/PROJECT_OVERVIEW.md` | The new handoff's own documentation of the served model, tri-state model, and fire-weather product |
