# New_Training817_moredata — tinker phase (started 2026-08-17)

Working folder for the model-improvement phase. The locked methodology
(`../METHODOLOGY.md`) does not change here — only candidate models and
new feature/label experiments live in this folder until one earns
promotion.

## What's here now

- **`tdis_forecast_hrrr_tuned_d9.json`** — the tuned candidate model
  (XGBoost, depth 9, min_child_weight 30, lr 0.02, 1000 trees; seed 42).
  Same 20 features, same training data, same split as the served model —
  only the tree settings differ.
- `step30_results.json` — its test-set validation (3-seed robustness) and
  the failed monotone-wind experiment.
- `step31_tuned_realpop.json` — its real-full-population validation.

## Candidate status: PROMOTED 2026-08-17 ✅

"Promoted" = made the served model (the one script 13 loads for the daily
dashboard forecast). This candidate has passed all three evidence gates:

| Gate | Served model | This candidate |
|---|---|---|
| Test set (balanced, 2023–26, never trained on) | AUC-PR 0.4825 | **0.4875** (+0.0050) |
| Seed robustness (3 seeds) | — | +0.0049…+0.0050, stable |
| Real full population (5.06M cell-days) | AUC-PR 0.0851 / AUROC 0.7838 / lift 4.43× | **0.0858 / 0.7892 / 4.47×** |

Promotion executed 2026-08-17 (`promote_d9.py`): calibrator refit with
2024-25 fit / 2026 holdout (holdout ECE 0.0019), served pair swapped
(backups: `models/*_pre_d9*.bak`), 24h/48h forecasts rerun, v3 dashboard
rebuilt, docs updated. This model IS now
`models/tdis_forecast_hrrr_filtered.json`.

## Queued experiments for this folder (from METHODOLOGY discussions)

1. **VIIRS `type`-flag check** — verify our 531 heuristically-removed
   flare cells against VIIRS's own fire-vs-static-source classification
   column. Label hygiene, no retrain unless discrepancies found.
2. **Power-line proximity feature** (HIFLD public data) — static feature,
   computed once; targets the powerline-in-wind ignition mechanism
   (Smokehouse Creek). Requires one retrain + recalibration.
3. Closed/rejected (do not redo): monotone wind constraints (-0.0003),
   gust/trailing/interaction features (flat), model×HWP multiplication
   (severely harmful). See `../rev2_improvements/`.

## Phase results (2026-08-17, all complete)

| Change | Test AUC-PR | Realpop lift | Status |
|---|---|---|---|
| baseline (served pre-phase) | 0.4825 | 4.43× | superseded |
| depth-9 tuning (step30/31) | 0.4875 | 4.47× | promoted, then superseded same day |
| + powerline_dist_km + flare v2 (step32) | 0.4925 | 4.56× | gate-passed |
| + hrrr_mstav soil moisture (step34) | **0.4941** | **4.58×** | **PROMOTED — the served model** |
| monotone wind constraint (step30) | 0.4822 | — | rejected |

Second promotion executed 2026-08-17: `step34_model.json` →
`models/tdis_forecast_hrrr_filtered.json`; calibrator refit (2026 holdout
ECE 0.0018) → `models/operational_isotonic_calibrator.joblib`; script 13
wired for both new features (powerline merge + MSTAV feature); forecasts +
v3 dashboard rebuilt. Backups: `models/*_pre_plms*.bak`.
Also produced: `aggregation_recall_experiment.json` — region-week
aggregation raises operating recall/precision dramatically (flag top 50%
of region-weeks → P 0.49 / R 0.78); the honest answer to small-fire
misses is coarser questions, not a different model.
