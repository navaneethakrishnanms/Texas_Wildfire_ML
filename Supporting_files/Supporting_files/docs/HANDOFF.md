# TDIS Forecast — Unattended Run Handoff

**Restarted:** 2026-07-29 12:27 CDT · tmux session **`tdis`** · **newest-year-first**

## What's running
VIIRS active fire for Texas, **downloaded newest → oldest (2026 → 2014)** so the most
recent, forecast-relevant data lands first. Sources: SP (archive) + **NRT** (recent ~2
months) so 2025–2026 are covered.

**Sequence (automatic):**
1. Download **2026** first → **auto-build STARTER dataset** (so you can train + map now)
2. Continue 2025 → 2014
3. At the end → **auto-build FULL dataset**

## STARTER dataset (ready ~30–40 min after start)
When you see `DATASET_READY_starter` (marker file) and `tdis_train_daily_tx.parquet`,
you can tinker with me on the model + dashboard **while the rest downloads**. The starter
already contains **all FPA-FOD 2014–2020 labels + 2026 VIIRS** (real-timestamp), so it's a
genuine trainable dataset, not a toy. It uses a spatial split at first and auto-switches to
a temporal (train≤2020 / val 2021 / test 2022+) split once 2021–2024 VIIRS is in.

## Check status
```bash
tmux attach -t tdis            # live (Ctrl-b d to detach)
tail -40 TDIS_Forecast/run_all.log
ls TDIS_Forecast/VIIRS_YEAR_*_DONE      # which years finished
ls TDIS_Forecast/DATASET_READY_*        # starter / full dataset markers
```

## Resume-safe
Re-run to continue from where it stopped:
```bash
cd .../alphaearth_nds/TDIS_Forecast
tmux new-session -d -s tdis "bash scripts/run_all.sh"
```
Rebuild the dataset from whatever's downloaded so far, anytime:
```bash
PY=/home/mte1224/mambaforge/envs/UAI2526/bin/python
$PY scripts/02_build_labels.py && $PY scripts/03_build_dataset.py
```

## Outputs
- `tdis_train_daily_tx.parquet` — daily training dataset (grows: starter → full)
- `data/labels_fused/ignitions_daily_tx.parquet` — fused ignition inventory
- `data/labels_viirs/viirs_tx_h3.parquet` — VIIRS detections, H3-gridded
- `data/labels_viirs/raw/` — raw FIRMS CSVs (resume cache)

## Path to FUTURE years (2026-ongoing, 2027+) — for the live forecast
1. **New labels:** the NRT sources (`VIIRS_*_NRT`) already in the driver pull the last ~2
   months; a **daily cron** re-running step 1 keeps labels current as new fires occur.
2. **Forecast inputs:** add **HRRR / NBM forecast weather** (next 24/48/72 h) as the runtime
   feature — this is what turns it from hindcast to forecast (separate phase).
3. **Refresh cadence:** daily job pulls latest VIIRS + latest HRRR forecast → re-scores the
   map for the next 24 h. (Operational host still TBD — offline build first.)

## Deferred (separate/larger jobs, not in this run)
GOES-16/18 sub-hourly timing · GOES-GLM lightning · NDVI (GEE) · HRRR forecast feed ·
5-day rolling / dryness-anomaly features (dataset v2) · model training.

## Knobs (scripts/03_build_dataset.py) — all easy to retune
`YEARS` (through 2026) · `WINDOW_HOURS` (24 now; 6 restores sub-daily) · `NEG_PER_POS` ·
`SPATIAL_FRAC` · `TRAIN_MAX/VAL_YR` split · cause/size filters
