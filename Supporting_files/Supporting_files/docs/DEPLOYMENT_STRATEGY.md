# Deployment Strategy — TDIS Forecast

What model serves what, under which rules, and what remains before/after
go-live. Assembled 2026-08-13, after the deployment-population validation,
calibration fix, and flag-fix pass (see `CLASSIFICATION_METRICS.md` §7).

## 1. Model matrix — which model serves what

| Surface | Model | Output shown | Why |
|---|---|---|---|
| **Live forecast (24/48h)** — headline product | `tdis_forecast_hrrr_filtered.json` + `operational_isotonic_calibrator.joblib` | Color by rank (as now); `ignCal` in tooltips as the real probability | Best validated real-world ranking in the project (lift 4.43x, AUROC 0.784 on the true every-cell-every-day population); calibrated ECE 0.0018 on a 2026 temporal holdout |
| **Historical Ignition scrubber** | *Upgrade available*: calibrated operational replay | Replace the susceptibility x FWI heuristic | `models/operational_historical_res5.parquet` already holds all 5.06M cell-day scores 2024-26 — the real model replay the heuristic stood in for; one dashboard-JSON merge + calibrator application away |
| **Wildfire Risk / Static Hazard** | No ML (hazard x FWI / TxWRAP) | Unchanged | Works as designed; wind-forward FWI weights + HRRR gust already in the live path |
| **Fallback (HRRR outage)** | `tdis_forecast_baseline_honest_filtered.json` | Flag visibly as "reduced-skill mode" | Needs only statics + calendar — can never be blocked by a weather-data outage |
| **Never serve** | Ceiling, ceiling-historical (`ignC`), any non-`_filtered` variant | — | Diagnostic controls / superseded by the flare-label fix |

## 2. Operating rules

1. **Model + calibrator ship as a pair, always.** A retrained model with a
   stale calibrator is worse than no calibrator. Recalibration
   (`scripts/27_fit_isotonic_operational.py`) is a mandatory step of every
   retrain, not an optional follow-up.
2. **Rank for color, calibrated value for numbers.** Keep the "relative
   ranking" visual framing the dashboard already uses; only `ignCal` ever
   appears anywhere a user could read a probability. Raw scores are ranks —
   they overstate risk ~17x if read literally.
3. **Daily cadence**: cron pulls HRRR 24h + 48h → `scripts/13` (now emits
   raw + calibrated, res-5 JSON + res-8 parquet) → push to portal. A missed
   day must alert, not just log — HRRR gaps are routine, silence is not
   acceptable.
4. **Monitoring**: monthly, score the previous month's forecasts against
   incoming VIIRS labels — rolling lift and ECE. Alarm thresholds: lift
   trending toward ~2x, or ECE above ~0.05. That is the drift detector.
5. **Retrain annually** (adding one new label-year), **recalibrate
   quarterly** (cheap — script 27 rerun). Bump both files' versions
   together; metas must carry population context (they now do).

## 3. Hyperparameter tuning status

**Never performed on any TDIS model** — all use one inherited fixed config
(400 trees, depth 6, lr 0.05, mcw 20). Deferred during rev2 because
IgnitionNet's real 21-trial search bought only ~0.005 AUROC. Now being run
once, cleanly (`rev2_improvements/step4_hyperparameter_sweep.py`), for two
reasons: (a) due diligence before the paper; (b) IgnitionNet's tuned
optimum was depth 9 vs. our 6, and insufficient depth was the working
hypothesis for why the wind x dryness interaction never got learned.
Expectations: small. Selection on the 2022 validation year only; test
scored once for the winner.

## 4. Remaining tasks, ranked

1. **Prospective shadow validation** — run the daily forecast live for 4-8
   weeks and score against incoming VIIRS. All current numbers are replays;
   a prospective result is the strongest pre-deployment evidence available
   and costs only patience.
2. **Hyperparameter sweep** — running (see §3).
3. **Swap the historical Ignition layer** to the calibrated operational
   replay (data already computed).
4. **Deployment-population threshold sweep** — if any alerting use emerges,
   sweep on the calibrated scale (e.g. ignCal > 0.05/0.10/0.15); the
   0.2-0.8 raw-scale grid is meaningless at a 2% base rate.
5. **Cron + alerting + portal push** — blocked on portal ingestion details
   from the TDIS portal owner (`LIVE_PIPELINE_PLAN.md` §2).
6. **VIIRS `type`-flag flare cross-check** — definitive fix for the flare
   threshold; the robustness band (lift 3.54-4.69x across 1%/3%/10%)
   already de-risked it.
7. **Wind-driven WHEN gap** — mitigated in the fire-weather index (gust +
   wind-forward weights), unresolved in the ML model (three feature
   attempts all flat; documented in `rev2_improvements/`). Carry as a known
   limitation unless the depth sweep changes the picture.
