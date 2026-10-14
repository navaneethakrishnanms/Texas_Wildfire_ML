# Wildfire Ignition Model — Executive Summary (TDIS 23-feat vs. Our 38-feat vs. Our 36-feat)

**Date:** 2026-10-08 · **One-page-per-section, 3-page read**

---

## PAGE 1 — The journey, in one box each

```
┌─────────────────────────────┐   ┌─────────────────────────────┐   ┌─────────────────────────────┐
│   TDIS SERVED MODEL          │   │   OUR BIT v1 (38 features)   │   │   OUR BIT V2 (36 features)   │
│   23 features                │──▶│   +lat +lon +gridMET (8)     │──▶│   lat/lon removed,           │
│   Their data, their tuning   │   │   21-trial hyperparameter    │   │   better 3-stage tuning      │
│   (depth9/1000trees, lr0.02) │   │   search (2 stages)          │   │   (34 trials, regularized)   │
└─────────────────────────────┘   └─────────────────────────────┘   └─────────────────────────────┘
```

**What happened at each step:**
1. **TDIS gave us their newest package** (`TDIS_Ignition/`) — a 23-feature served XGBoost model, fixed hyperparameters, no fresh search.
2. **We rebuilt their own corrected dataset ourselves** (`train_tristate_v2.parquet`, TX rows), added `lat`/`lon` (derived from H3, since theirs were 100% null) + 8 gridMET weather columns → **38 features**, ran a real hyperparameter search → **TDIS_Ignition_BIT**.
3. **We tested removing `lat`/`lon`** (36 features) with a *wider, more regularized* search (3 stages, 34 trials) to check whether the model needed them, and to fight the overfitting gap v1 showed → **TDIS_Ignition_BIT_V2**.

---

## PAGE 2 — Full metrics comparison (box format)

### Headline metrics — side by side

| Metric | 📦 **TDIS (23 feat)**<br>matched test | 📦 **TDIS (23 feat)**<br>real population | 📦 **Our BIT v1 (38 feat)** | 📦 **Our BIT V2 (36 feat)** |
|---|:---:|:---:|:---:|:---:|
| **AUROC** | 0.7402 | 0.7968 | **0.8064** ⭐ | 0.8026 |
| **AUC-PR** | 0.4941 | 0.0878 | **0.7237** | 0.7164 |
| **F1** | 0.4958 | 0.1692 | **0.6284** ⭐ | 0.6216 |
| **Precision** | 0.441 | 0.149 | 0.6215 | 0.6105 |
| **Recall** | 0.566 | 0.185 | 0.6353 | 0.6332 |
| **Lift** (vs base rate) | 2.11× | **4.58×** ⭐ | 2.37× | 2.34× |
| Test rows | 891,983 | 5,064,462 | 1,261,502 | 1,261,502 |
| Base rate | 23.4% | 1.92% | 30.6% | 30.6% |

⭐ = best in that row. **Read with care:** AUC-PR/F1/Lift are not directly comparable across different base rates — our numbers are on a ~30% rebalanced sample, TDIS's real-pop column is on the true ~2% deployment rate. On **lift** (the fair metric), TDIS's real-population number (4.58×) is still the one to beat — we have not run that test on our models yet.

### Confusion matrices — side by side

```
TDIS (23 feat, matched test, n=891,983)      OUR BIT v1 (38 feat, n=1,261,502)         OUR BIT V2 (36 feat, n=1,261,502)
derived from published P/R                    EXACT, computed directly                  EXACT, computed directly
┌───────────┬──────────┬─────────┐            ┌───────────┬──────────┬─────────┐       ┌───────────┬──────────┬─────────┐
│           │ Pred No  │ Pred Fire│           │           │ Pred No  │ Pred Fire│       │           │ Pred No  │ Pred Fire│
├───────────┼──────────┼─────────┤            ├───────────┼──────────┼─────────┤       ├───────────┼──────────┼─────────┤
│ Actual No │ 533,414  │ 149,776 │            │ Actual No │ 727,000  │ 149,110 │       │ Actual No │ 720,388  │ 155,722 │
│ Actual Fire│ 90,616  │ 118,177 │            │ Actual Fire│ 140,535 │ 244,857 │       │ Actual Fire│ 141,348 │ 244,044 │
└───────────┴──────────┴─────────┘            └───────────┴──────────┴─────────┘       └───────────┴──────────┴─────────┘
Precision 0.441 · Recall 0.566                 Precision 0.622 · Recall 0.635            Precision 0.610 · Recall 0.633
```

### Top 5 important features — side by side

| Rank | TDIS (23 feat) | Our BIT v1 (38 feat) | Our BIT V2 (36 feat) |
|---|---|---|---|
| 1 | `ecoregion_id` (14.3%) | `ecoregion_id` (20.8%) | `ecoregion_id` (14.7%) |
| 2 | `elevation_m` (10.5%) | `elevation_m` (7.7%) | `elevation_m` (8.5%) |
| 3 | `sin_month` (7.3%) | **`lat`** (4.5%) | `hrrr_vpd` (7.4%) |
| 4 | `hrrr_vpd` (6.1%) | **`lon`** (3.4%) | `avg_burn_prob` (5.6%) |
| 5 | `avg_burn_prob` (6.0%) | `powerline_dist_km` (3.3%) | `road_dist_km` (4.4%) |
| Dead features | `flep4`, `cfl` (0.00%) | `flep4`, `cfl` (0.00%) | `flep4`, `cfl` (0.00%) |

`flep4`/`cfl` are confirmed dead in **all three** versions — the same two LANDFIRE features carry zero signal no matter who trains the model or which feature set is used.

---

## PAGE 3 — lat/lon removal result, deployment status, and verdict

### What removing lat/lon actually did

| | BIT v1 (with lat/lon) | BIT V2 (without lat/lon) | Change |
|---|---|---|---|
| Hyperparameter search | 21 trials, 2 stages | 34 trials, 3 stages (more regularized) | deeper search |
| `min_child_weight` chosen | 10 | 50 (much more regularized) | — |
| Test AUROC | 0.8064 | 0.8026 | **−0.0038** |
| Test AUC-PR | 0.7237 | 0.7164 | **−0.0073** |
| Train→Test AUROC gap (overfit check) | 0.143 | 0.128 | improved ~10%, not solved |

**Conclusion:** `lat`/`lon` were carrying real signal, not just memorization — removing them cost a small but consistent amount of accuracy, even with better tuning. The overfitting gap got a little smaller but is still driven mainly by tree depth, not by the features removed. **BIT v1 (38 features) remains the stronger model on every accuracy number.**

### Deployment status — honest, not assumed

```
┌──────────────────────────────────────────────────────────────┐
│  ⚠️  NOT YET DEPLOYED TO AZURE (or anywhere) IN THIS SESSION   │
│                                                                 │
│  What exists today:                                            │
│   ✅ Trained models + calibrators (BIT v1 and BIT V2)          │
│   ✅ Full metrics, confusion matrices, 9 evaluation plots      │
│   ✅ One real daily snapshot (2026-07-15.csv/.parquet)         │
│   ❌ No live scoring endpoint / API                             │
│   ❌ No Azure resource, container, or pipeline set up          │
│   ❌ No real-population validation (true ~2% base rate)        │
│   ❌ No leakage tests (label-shuffle, spatial holdout)          │
└──────────────────────────────────────────────────────────────┘
```

**To actually deploy this** (not done yet, scoped for next step):
1. Build the daily scoring entrypoint (static grid + live HRRR forecast pull + calendar features → `predict_proba` → calibrate).
2. Fix the gridMET-observed-vs-forecast gap flagged earlier (8 of our features aren't true forecasts — need a seasonal-fallback rule at serving time).
3. Package into a container / Azure Function / App Service, point it at a schedule (daily, after the HRRR run publishes).
4. Only then connect the frontend.

### Final verdict

- **Best model for accuracy right now:** our BIT v1 (38 features, AUROC 0.8064).
- **TDIS still wins on validation rigor:** real-population lift (4.58×), leakage tests, named-fire validation — none of which we've replicated yet.
- **Removing lat/lon (BIT V2) is a validated experiment, not an upgrade** — keep it as evidence the model doesn't *depend* on coordinates, but don't deploy it over BIT v1.
- **Deployment is a separate, not-yet-started track** — this document should not be read as confirming an Azure deployment exists.
