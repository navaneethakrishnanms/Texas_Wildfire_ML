#!/bin/bash
# TDIS Forecast — unattended pipeline (newest-year-first).
# The driver (01) downloads VIIRS newest->oldest, builds a STARTER dataset after year 1,
# and a FULL dataset at the end. Steps 02/03 are invoked by the driver; they can also be
# run standalone anytime to rebuild from whatever is downloaded so far. Resume-safe.
set -o pipefail
PY=/home/mte1224/mambaforge/envs/UAI2526/bin/python
cd /net/cven-mosta-nas.engr.tamu.edu/volume2/NASdata2/miguel_shared/alphaearth_nds/TDIS_Forecast
LOG=run_all.log
echo "=== TDIS pipeline started: $(date) ===" | tee -a $LOG
$PY -u scripts/01_download_viirs.py 2>&1 | tee -a $LOG
echo "=== TDIS pipeline finished: $(date) ===" | tee -a $LOG
echo "Dataset: TDIS_Forecast/tdis_train_daily_tx.parquet" | tee -a $LOG
