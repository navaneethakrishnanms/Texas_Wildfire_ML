# TDIS_Ignition — Full Folder Deep-Dive (file-by-file, line-by-line) + Complete Us-vs-Them Comparison

**Prepared:** 2026-09-30
**Method:** Every one of the 251 files in `TDIS_Ignition/` was inventoried. All 65 distinct Python scripts (~11,490 lines) were read in full. Every parquet/JSON/joblib/XGBoost-model file (104 data artifacts) was loaded programmatically (pandas / xgboost.Booster / joblib) to get real shapes, columns, null rates, and feature orderings — nothing below is taken from their documentation on faith; where their docs disagree with the loaded files, that's flagged explicitly.
**Companion document:** `PROJECT_STATUS_REPORT_TDIS_IGNITION.md` (doc-level comparison, written first). This document goes one level deeper — actual code and actual bytes.

---

## PART A — Complete feature lists (ground truth, from the model files themselves)

### A.1 Served TX model — `model/tdis_forecast_hrrr_filtered.json` — 23 features, in exact order

| # | Feature | Type | Group |
|---|---|---|---|
| 1 | `road_dist_km` | float | Static — access |
| 2 | `ecoregion_id` | float | Static — region |
| 3 | `elevation_m` | float | Static — terrain |
| 4 | `slope_deg` | float | Static — terrain |
| 5 | `aspect_deg` | float | Static — terrain |
| 6 | `avg_burn_prob` | float | Static — fuel hazard (FSim) |
| 7 | `whp` | float | Static — fuel hazard (USFS WHP) |
| 8 | `flep4` | float | Static — fuel hazard (LANDFIRE) — **0.0 importance** |
| 9 | `cfl` | float | Static — fuel hazard (LANDFIRE) |
| 10 | `cbd` | float | Static — canopy (LANDFIRE) |
| 11 | `cbh` | float | Static — canopy (LANDFIRE) |
| 12 | `powerline_dist_km` | float | Static — ignition-source proxy (HIFLD) |
| 13 | `sin_month` | float | Calendar |
| 14 | `cos_month` | float | Calendar |
| 15 | `sin_dow` | float | Calendar |
| 16 | `cos_dow` | float | Calendar |
| 17 | `is_weekend` | int | Calendar |
| 18 | `is_holiday` | int | Calendar |
| 19 | `hrrr_tmp` | float | Weather — HRRR forecast |
| 20 | `hrrr_vpd` | float | Weather — HRRR forecast |
| 21 | `hrrr_wind` | float | Weather — HRRR forecast |
| 22 | `hrrr_mstav` | float | Weather — HRRR soil moisture |
| 23 | `drought_score` | float | Weather — USDM (**new since our first audit — added after Aug 19**) |

`num_boosted_rounds = 1000`, `base_score = 0.5`. Confirmed by directly loading the booster — matches the corrected `meta.json` exactly now (our original audit's "stale metadata" defect is fixed for this file).

**A stale sibling still ships in the same folder:** `model/tdis_forecast_hrrr_tuned_d9.json` has only **20 features** (missing `powerline_dist_km`, `hrrr_mstav`, `drought_score`) — it's a pre-promotion artifact from an earlier step that was never removed. Anyone loading "the d9 model" by filename guess gets the wrong one.

### A.2 Tri-state model (`model_tristate/`) — rev5_v2 / control_v2 — 28 features

Same 12 static + 6 calendar + `hrrr_tmp/vpd/wind/mstav` + `drought_score` as the served model (23), **plus 5 more**:
`bp_x_vpd`, `whp_x_vpd`, `bp_x_drought` (hand-built interaction terms), `pop_density` (WorldPop), `fm100_5d_min` (gridMET 100-hr fuel moisture, 5-day trailing min, lag-2 to avoid publication-lag leakage).

### A.3 Research sub-daily model (`docs/DATASET_BREAKDOWN.md`, not in the served package) — 30 features

The 28 above minus `hrrr_mstav`/`drought_score` inconsistency aside, plus `sin_hour`/`cos_hour` for true 6-hour sub-daily windows — this is a separate research line (`TriState_subDaily`), not part of the deployed TX daily model.

### A.4 Dead features — confirmed three separate times

`flep4` and `cfl` show **exactly 0.000 gain importance** in the served model's own `feature_importance` dict inside `tdis_forecast_hrrr_filtered_meta.json`. Same finding independently reproduced in our own TX and TEXAS models. Three separate teams, three separate training runs, same two dead LANDFIRE features.

---

## PART B — What the code actually does (pipeline, stage by stage)

The `code/` tree is really **five interleaved lineages** sharing a flat numbered-script folder (`code/scripts/`), which is why numbers 22 and 23 each appear twice for unrelated scripts — a real organizational defect, not a typo.

### B.1 Label + static-data construction (01–06)
- **01_download_viirs.py** — pulls VIIRS FIRMS detections 2014–2026 over a **rectangular TX bounding box**, not the state polygon — this is the root cause of the ~6% NM/OK spillover their team later found and fixed downstream (script 02).
- **02_build_labels.py** — fuses VIIRS + FPA-FOD into `ignitions_daily_tx.parquet`, explicitly filters out the NM/OK spillover using the static-master polygon (a real, documented fix — "SANITY_CHECK F1").
- **03_build_dataset.py** — builds the actual training table: 8 temporal negatives + 0.4× spatial negatives per positive (seed 42). **Notable defect we hadn't seen before:** the train/val/test split *strategy itself* auto-switches between temporal and a spatial hash-based fallback depending on how much data currently exists at runtime — and the spatial fallback uses Python's built-in `hash()`, which is not stable across runs unless `PYTHONHASHSEED` is fixed (it isn't). That's a latent, silent reproducibility bug in a fallback code path.
- **04_build_full_tx_static.py** — builds the real 1,708,940-cell static master (elevation/slope/aspect from Copernicus DEM, roads from Census TIGER, ecoregion from EPA polygons, cbd/cbh from LANDFIRE). Incremental/cached, all paths hardcoded to a TAMU NAS share.
- **05/06** — pulls **archived HRRR forecasts** (not observations) specifically to avoid train/serve mismatch; documents that HRRR F24+ data didn't exist before 2018-07-15, explaining ~35% of label dates being weather-null by construction, not by bug.

### B.2 Model iteration lineage (07 → 34, three generations)
1. **Generation 1 (07, 08, 11, 12):** "honest" (static+calendar only) vs "ceiling" (+ same-day *observed* weather, never deployable — an upper bound). Fixed hyperparameters: 600 trees, depth 6, lr 0.05, early stopping 40 rounds.
2. **Flare/phantom-date cleanup (18, 19, 22b):** found ~27% of positives were persistent industrial flares (>3% of days on fire), and ~86K future-dated rows past the last real label. Retrained on cleaned data with a **fixed 400-tree budget and no early stopping** — their own comment explains they *removed* early stopping because the newly-cleaned validation year showed a spurious spike at iteration ~3, so early stopping was actually hurting, not helping, on the cleaned labels.
3. **Depth-9/1000-tree generation (step32, step34, step35, rev4, all of nm_ok):** promoted after `powerline_dist_km` + flare-list-v2 (step32), then `hrrr_mstav` soil moisture (step34), then contagion (`fire_nearby_3d`) + `pop_density` (step35). This depth-9 config becomes the standard for everything after.
4. **rev4 — the most important fix in the whole codebase.** SHAP analysis found the served model had a **non-monotone, physically-backwards wind response** (22 m/s → lower risk than 27 m/s → lower risk again) — root cause: training on a single 00Z HRRR snapshot rather than the true daily peak. Fix: switch every weather feature to the **daily peak** across the 27/30/33/36h leads off a 12Z-previous-day HRRR init, plus explicit XGBoost `monotone_constraints` forcing burn-prob/WHP/drought/gust/VPD to only increase risk. This was validated against a **named-fire battery** (Smokehouse Creek, Windy Deuce, Crabapple, Lavender, Hunggate) specifically because the old model badly under-ranked the 1.06M-acre Smokehouse Creek fire.
5. **rev4 was never promoted to production.** It beats the served model on every metric (AUC-PR 0.511 vs 0.494, real-pop lift 4.76–4.88× vs 4.62×) but sits unshipped in a research folder — this is their own documented fact, not our inference.

### B.3 Calibration (24, 26, 27, promote_d9.py)
Genuinely disciplined: calibrator fit strictly on 2024–2025 predictions, verified only on a 2026 holdout it never touched. This is the origin of the shipped `operational_isotonic_calibrator.joblib`.

### B.4 Tri-state extension (`code/nm_ok/`, `model_tristate/`)
Deliberately trains on TX+NM+OK to broaden the training distribution while testing **only on TX** (a hardcoded rule in `15_train_control.py`, not just documented — to prevent NM/OK ever leaking into a reported TX metric). **A real bug was caught mid-project:** TX's `ecoregion_id` used a locally re-indexed 0–19 scheme while NM/OK used the genuine national EPA code — meaning the same integer meant different biomes in different states, so the tri-state model could never have learned cross-state biome transfer. Fixed in `fix_tx_ecoregion.py`; re-validated (`_v2` scripts). **Conclusion after the fix: tri-state training is a statistical wash for TX accuracy** (lift 4.26× vs 4.29× control) — confirmed twice, pre- and post-bugfix, ruling out "the bug explains the null result."

### B.5 Production serving (`serve/`)
`serve/score_day.py` is the one script actually engineered for portability (package-relative paths, live + offline modes). `serve/validate_trust.py` is a genuinely strong self-audit — it re-derives every claimed headline number from the packaged data and **hard-fails (`sys.exit(1)`)** if anything drifts more than 2%. But `serve/make_validation_visuals.py` — despite living in the same "portable" folder — still reaches back to a TAMU-server-only absolute path to import `rev4_pipeline`, so it is **not** actually runnable outside the original server, contradicting the portability the rest of `serve/` was built for.

### B.6 Fire-weather product (`fireweather/`)
A deliberately **separate** product from ignition. Their own validation (`overall_validation_2026-09-23.json`) shows fire-weather indices have **essentially zero skill at predicting WHERE/IF a fire starts** (AUROC ≈ 0.49–0.51, random) — they only correlate with severity/timing on days a fire already happened. This is the evidence behind their hard rule: **never multiply the fire-weather index onto the ignition score** — tested, and it degrades ignition AUC-PR by 18–61%.

---

## PART C — Ground-truth data facts (loaded directly, not from docs)

| Artifact | Shape | Key fact |
|---|---|---|
| `ignitions_daily_tx.parquet` | 960,054 × 7 | `max_size_acres` null in 96.2% of rows; date range 2014-01-01 → 2026-07-29 |
| `data/flares/flare_cells_tx_v2.parquet` | **538 rows** | `meta.json` claims 531 — a 7-cell discrepancy between the shipped metadata and the actual shipped flare list |
| `data/static/tx_static_master.parquet` | 1,708,940 × 15 | 0 nulls — the clean, current full-TX static master |
| `data/nm_ok/static_master_tx.parquet` (older) | 824,052 × 16 | **7 columns 100% null** (`whp`,`cbd`,`cbh`,`powerline_dist_km`,`elevation_m`,`slope_deg`,`aspect_deg`) — a stale/broken static file still present in the package |
| `data/nm_ok/train_tristate.parquet` | 2,731,285 × 45 | Positive rate **28.19%** (training-time oversample); weather columns (`erc`,`vpd`, etc.) ~81% null (largely unused legacy gridMET columns carried along) |
| `data/scored/operational_historical_res5_v2.parquet` | **5,064,462 × 4** | The real-population panel — base rate 1.92%, matches `TRUST_VALIDATION.json` exactly |
| `model_tristate/data/realpop_res5_v2_panel.parquet` | 3,622,850 × 5 | Base rate **2.524%** |
| `model_tristate/data/static/tx_ecoregion_national.parquet` | 1,708,940 × 2 | **29.7% null** `ecoregion_id` — the file that was supposed to *fix* the ecoregion bug still has substantial missingness |
| `serve/output/forecast_2024-02-27.json` vs its sibling `.parquet` | JSON claims `n_cells_res8=1804` but its own `ign`/`ignCal` dicts contain only **585 keys** (~32% of claimed coverage) | The parquet output for the identical run does have the full 1,804 rows — the JSON serialization step silently drops ~69% of cells it claims to cover. **This is a newly-discovered defect our first two audits did not catch.** |

### C.1 Model sensitivity / monotonicity — a genuinely new finding

`model/TRUST_VALIDATION.json` (their own self-check, shipped with the package) reports the **served, currently-deployed model** fails physical-direction sanity checks on **5 of 9 tested features**:

| Feature | Verdict |
|---|---|
| `hrrr_wind` | ❌ WRONG DIRECTION |
| `hrrr_tmp` | ❌ WRONG DIRECTION (consistently, monotone=True but in the wrong direction) |
| `avg_burn_prob` | ❌ WRONG DIRECTION |
| `whp` | ❌ WRONG DIRECTION |
| `hrrr_mstav` | ❌ WRONG DIRECTION |
| `hrrr_vpd`, `drought_score` | dir-ok-not-monotone |
| `road_dist_km`, `elevation_m` | OK |

By contrast, the tri-state `rev5` model's own validation (`rev5_trust_validation.json`) passes **11/11** features with correct, monotone directions — because rev4/rev5 added the explicit `monotone_constraints` that the currently-served model still lacks. **In plain terms: the model actually running in production today has known, self-documented, backwards physical responses to wind, temperature, burn probability, and hazard potential — and the fix for this already exists in their own repo (rev4) but was never promoted.**

### C.2 Isotonic calibrator ceilings (confirmed by evaluating the actual `.joblib` files)

| Calibrator | Max output (f(1.0)) | Practical meaning |
|---|---|---|
| `operational_isotonic_calibrator.joblib` (served) | **0.3333** | No cell can ever be shown a probability above 33% |
| `tuned_d9_isotonic_calibrator.joblib` | 0.1942 | Ceiling of ~19% |
| `rev5_tristate_v2_isotonic_calibrator_REALPOP.joblib` | 0.2740 | Ceiling of ~27% |
| `rev5_tristate_v2_isotonic_calibrator_PRELIM.joblib` | 1.0 (matched-sample fit, not real-population — don't use) | N/A |

---

## PART D — Complete comparison table: What they did vs. what we did

Legend for the "Better" column: **Them** / **Us** / **Tie** / **N/A** (not directly comparable).

| Dimension | TDIS/Texas team (`TDIS_Ignition` + prior handoff) | Our team (V1→V2→TX→TEXAS) | Better |
|---|---|---|---|
| **Labels** | VIIRS (FIRMS, 2014–2026) fused with FPA-FOD (2014–2020); documented that only 0.12% of rows have true 2-source corroboration — effectively VIIRS-driven post-2020 | FPA-FOD only, 2014–2020, ≥1 acre | **Them** — longer horizon (extends to 2026), broader detection net via satellite |
| **Label cleaning** | Found and removed 538 "flare" cells (persistent industrial hotspots, ~27% of positives); cross-validated against VIIRS's own `type` flag (65.8% confirm rate on sampled cells) | Found and removed 2 leakage columns (`fire_count`, `has_fire_history`) computed across the full date range | **Them** — a harder, more subtle defect (label contamination vs. feature leakage), independently audited |
| **Spatial extent honesty** | Rectangular bbox, ~54% of grid falls outside actual Texas (confirmed by ecoregion names like Ouachita Mountains, Arizona/NM Mountains); documented as an open issue, partially addressed in tri-state via national ecoregion join (still 29.7% null) | H3 res-8 grid built directly over TX FPA-FOD records (naturally state-bound, no bbox contamination) | **Us** on this specific point — our grid doesn't carry ~54% out-of-state cells |
| **Training rows (core TX model)** | 3,595,513 raw → 2,007,436 usable (after flare + weather-availability filtering); real-population eval on 5,064,462 cell-days | 376,233 (V2/TX) → 3,595,513 raw (TEXAS, same TDIS-sourced parquet) → same table, imputed to 0 nulls | **Them** — larger native table and, critically, the only side with a true real-population evaluation |
| **Feature count (served model)** | **23** (ground-truth from the model file) | 32 (TEXAS 32-feat, our best) | N/A — more features isn't automatically better; see below |
| **Feature list overlap** | 12 static, 6 calendar, 5 weather (incl. `drought_score`, unique to them) | Same 12 static + 6 calendar + 4 HRRR + 8 gridMET + `lat`/`lon` (unique to us) | **Tie** — each has legitimate features the other lacks (`drought_score` vs `lat`/`lon`/gridMET) |
| **Dead features found** | `flep4`, `cfl` — 0.000 importance, confirmed in their own `meta.json` | Same two features, 0.000 importance, confirmed independently in TX and TEXAS models | **Tie** — same finding, reached independently; genuinely reinforcing cross-validation |
| **Negative sampling design** | 8 temporal + 0.4× spatial per positive (script 03); positive rate ~28% in training table vs ~1.9–2.5% real | Day-matched, ratio 1:10 (V2/TX); sensitivity-tested across ratios 1:5–1:50 | **Us** — explicit ratio sensitivity analysis documented; theirs is fixed without a stated sensitivity check in the code we read |
| **Model algorithm** | XGBoost, 1,000 trees, depth 9, min_child_weight 30, lr 0.02 (both served and tri-state) | XGBoost, tuned via genuine hyperparameter search (21 trials for TX; 21 trials again for TEXAS 32-feat) | **Tie on algorithm; Us on tuning discipline** — our depth-9/lr-0.02-ish config was *found* by search; theirs was carried forward from step32 as a fixed convention, not re-searched at each promotion |
| **Monotonicity constraints on served model** | ❌ **None** — and their own `TRUST_VALIDATION.json` shows the served model fails physical-direction checks on 5/9 features (wind, temp, burn-prob, WHP, soil moisture all "WRONG DIRECTION") | Not implemented on either side | **Them, conceptually** — they *diagnosed* the exact problem and *built the fix* (rev4's monotone_constraints + peak-weather), it's just unpromoted; we have not identified or attempted to fix this class of bug at all |
| **Real-population validation** | ✅ Yes — 5.06M cell-days scored at true ~1.9% base rate (AUC-PR 0.088, AUROC 0.797, lift 4.58×) | ❌ Not done for any of our models (TX, TEXAS, or TEXAS 32-feat) — all our numbers are on rebalanced/matched test sets | **Them — decisively.** This is the single biggest gap on our side |
| **Calibration** | ✅ Isotonic, fit 2024–25 / holdout 2026, ECE 0.2799→0.0018 (served) and even tighter (0.0008) in the newest package; but calibrator **saturates** — served model can never output above 33% | ✅ Isotonic, similar discipline, TEXAS 32-feat val→test ECE improvement documented but not saturation-checked | **Them** — same technique, but they found and documented their own ceiling defect; we haven't checked for one |
| **Leakage / credibility tests** | ✅ Label-shuffle test (AUC-PR collapses to base rate) + ✅ spatial holdout (20% of cells never trained on, gap only 0.005) | ❌ Neither test has been run on our models | **Them — decisively** |
| **Real named-event validation** | ✅ 5/5 Texas wildfires captured, hazard percentile 74th–96th on all five, explicit rev4 fix motivated by one event (Smokehouse Creek) being under-ranked | ❌ Not attempted | **Them — decisively** |
| **Test-set metrics (their own matched/balanced sets, NOT real-population — the two are not directly comparable across teams)** | Served: AUC-PR 0.494 / AUROC 0.740 (23-feat, 2023–26 test, ~23% pos rate). Unpromoted rev4: AUC-PR 0.511 / AUROC ~0.81, lift 4.76–4.88× real-pop | TEXAS 32-feat: AUC-PR 0.748 / AUROC 0.831 (2025–26 test, ~30.7% pos rate) | **N/A / caution** — different test windows and different (much higher) positive rates make raw AUC-PR not comparable; see lift discussion below |
| **Fair comparison metric (lift over base rate)** | Real-population lift **4.58×** (served) / 4.76–4.88× (rev4, unpromoted) | Lift on matched-sample tests only (not real-population) — TX model's fairest earlier figure was 4.67× vs their 2.11× *on that older comparison*, but this was against their pre-flare-fix model, not the current one | **Them, on current numbers** — because they measured lift at the true deployment base rate and we still have not |
| **Production serving pipeline** | ✅ `serve/score_day.py`, live + offline modes, tested both ways, self-contained relative paths, `verify_package.py` pre-flight checklist, `validate_trust.py` self-audit with hard failure gates | ❌ No production entrypoint; our models are trained/evaluated in notebooks/scripts, not served | **Them — decisively** |
| **Multi-state extension** | ✅ Tri-state (TX+NM+OK) model built, tested honestly (TX-only eval), found to be statistically neutral for TX accuracy — an honest negative result, not oversold | ❌ None (our V1 had a separate California pipeline mentioned in early docs, not carried forward into TEXAS line) | **Them** |
| **Fire-weather product as a separate layer** | ✅ Built, validated (AUROC ~0.5 for location — correctly concluded it has no ignition-location skill), hard "never multiply" rule enforced by evidence | ❌ Not built | **Them** |
| **Data quality self-audits** | Multiple internal scripts (`sweep_phase1_forensics.py`, `sweep_phase6_credibility.py`, `24_validate_ceiling_historical.py`, `25_rescore_all_models.py`) exist specifically to catch internal inconsistencies — and did catch several (label discrepancies, miscalibration, flare contamination) | Our two audit documents (`TDIS_HANDOFF_ANALYSIS.md`, `DATA_QUALITY_REVIEW.md`) served this role, but applied externally to *their* package, not to our own pipeline | **Them, for self-auditing their own work; Us, for catching things their audits missed** (e.g., we found the fabricated-future-dates issue independently) |
| **Reproducibility / portability** | Mostly **hardcoded to a specific TAMU server** (absolute NAS paths, specific conda env) across ~90% of scripts; only `serve/` was rebuilt portable, and even that has one script (`make_validation_visuals.py`) that breaks portability | Our scripts use local relative/absolute Windows paths tied to this machine, similarly not portable, but far fewer scripts overall so the surface area of the problem is smaller | **Tie** — neither codebase is cleanly portable; theirs is a bigger, more entangled version of the same issue |
| **Numbering/organizational hygiene** | Two different scripts both named `22_*.py`, two different scripts both named `23_*.py` — evidence of merged development branches without renumbering; duplicate data files under different names in multiple places (`powerline_dist_km.parquet` == `nm_ok_powerline_dist_km.parquet`; `fireweather/spec_validation...json` == `model_tristate/.../rev5_trust_validation.json`) | Our folder structure (`V1`/`V2`/`TX`/`TEXAS`/`TEXAS_v1_backup`) is versioned by renaming whole directories, which is cleaner at the folder level but has its own downside (old code paths inside scripts sometimes point at moved files) | **Us, slightly** — our versioning-by-folder convention is easier to reason about than a flat numbered-script tree with collisions |
| **Metadata/model consistency** | Was broken in the original handoff (meta said 20 features, model had 22); **now fixed** in the new package for the served model — but a stale 20-feature `tuned_d9.json` model still sits alongside it, and the flare-cell count still disagrees (531 in meta vs 538 in the actual parquet) | Our model reports (`training_results_*.json`) are generated at training time from the actual fitted model, so no drift observed | **Us**, on this specific point, for the current state of things — though to be fair, theirs *was* worse and has partially self-corrected |
| **Honesty / documentation candor** | Extremely candid in-code comments documenting their own past bugs (bbox spillover, gridMET fill-value artifact, non-monotone wind response, fm100 leakage, ecoregion incommensurability, OOM crash, missing drought/powerline joins) — arguably the strongest single trait of the codebase | Our two audit docs are candid about *their* package; our own version-history doc (`PROJECT_VERSION_HISTORY.md`) is similarly candid about *our own* four rounds of bug-fixing (temporal leakage in V1, feature leakage in V2, gridMET nodata bug and 5-day rolling-stat bug in TX, HRRR RH unit bug) | **Tie** — both sides show real self-correction discipline; this is a genuine strength on both teams |
| **New defect found in this deep-dive (not previously known to either side)** | `serve/output/forecast_2024-02-27.json` silently reports only 585 of the 1,804 cells its own `n_cells_res8` field claims to cover (~68% under-reporting) in the JSON output, even though the parquet sibling for the identical run has the full 1,804 rows | — | **N/A** — newly surfaced by this analysis, not yet acted on by either side |

---

## E — Net assessment

**On rigor of evidence, they are ahead, and it isn't close.** Real-population validation, calibration-ceiling awareness, label-shuffle and spatial-holdout leakage tests, named-fire validation, and a self-auditing production pipeline are all things they have and we do not. Their own code is also unusually honest about its own history of bugs, which is a real engineering strength, not just documentation polish.

**On raw predictive skill on a matched/balanced test set, we currently report higher numbers** (TEXAS 32-feat: AUC-PR 0.748 vs their served 0.494) — but this comparison is **not fair as stated**, because (a) our test window and positive rate differ substantially from theirs, and (b) they have already told us via `TRUST_VALIDATION.json` and `rev4` that their own *served* model is a known-suboptimal, un-promoted-fix state — the honest comparison point is their **rev4** model (AUC-PR 0.511, real-pop lift 4.76–4.88×), not the currently-deployed one, and even that comparison requires us to run real-population validation on our own model before it means anything.

**The single highest-leverage next step, unchanged from the first report but now sharper:** run real-population scoring, calibration, label-shuffle, and spatial-holdout tests on our TEXAS 32-feature model. Until that exists, every "we beat them" claim in this document is provisional.
