#!/bin/bash
# HWP knob chain: waits for 09 (components cache w/ fm100), then fit -> arrays -> embed -> live forecasts.
# Run inside tmux so it survives disconnects. Log: TDIS_Forecast/hwp_chain.log
set -u
PY=/home/mte1224/mambaforge/envs/UAI2526/bin/python
TF=/net/cven-mosta-nas.engr.tamu.edu/volume2/NASdata2/miguel_shared/alphaearth_nds/TDIS_Forecast
cd "$TF/scripts"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for 09 components cache (fm100) ..."
until grep -q "Saved tdis_dashboard_data" "$TF/rebuild_components_fm100.log" 2>/dev/null; do
  # self-heal: if the external 09 run died, run it ourselves
  if ! pgrep -f "09_build_perday_dynamic" >/dev/null; then
    log "09 not running and not complete — (re)running it here"
    $PY 09_build_perday_dynamic.py 2>&1 | tee "$TF/rebuild_components_fm100.log" || { log "09 FAILED"; exit 1; }
    break
  fi
  sleep 30
done
log "09 done. Fitting TX HWP coefficients (16)..."
$PY 16_fit_hwp_tx.py || { log "FIT FAILED"; exit 1; }
log "Rebuilding 3-variant per-day arrays (15)..."
$PY 15_reweight_fwi.py || { log "15 FAILED"; exit 1; }
log "Regenerating live forecasts w/ soil moisture (13)..."
TOM=$(date -d tomorrow +%F); D2=$(date -d '+2 days' +%F)
$PY 13_model_forecast_day.py "$TOM" 24 || log "WARN: 24h forecast failed"
$PY 13_model_forecast_day.py "$D2" 48 || log "WARN: 48h forecast failed"
log "Embedding standalone dashboard (17)..."
$PY 17_embed_standalone.py || { log "EMBED FAILED"; exit 1; }
log "CHAIN COMPLETE"
