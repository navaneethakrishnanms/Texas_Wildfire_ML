"""
run_phase2c.py
--------------
Phase 2C — Label Generation

Converts raw FPA-FOD fire records into clean label=1 rows:
  (h3_cell, window_6h_utc, label=1, centroid_lat, centroid_lon, fire_year, date_utc)

HOW IT WORKS:
  1. Load Texas/California FPA-FOD parquet from Phase 1
  2. Filter: state=TX (or CA), FIRE_SIZE >= 1 acre, FIRE_YEAR 2014-2020
  3. For each fire:
       - Convert local discovery time → UTC with DST-aware timezone
       - Texas: America/Chicago (CDT/CST); El Paso 5 counties: America/Denver
       - California: America/Los_Angeles (PDT/PST)
       - Floor UTC hour to nearest 6-hour boundary → {0, 6, 12, 18}
       - Map lat/lon → H3-R8 cell
  4. Output: positives_labels.parquet (label=1 for every fire)

UTC WINDOW LOGIC:
  A fire at 14:30 CST on 2018-11-08:
    → 20:30 UTC → floor(20/6)*6 = 18 → window=18Z
  A fire at 03:00 PDT on 2019-07-04:
    → 10:00 UTC → floor(10/6)*6 = 6 → window=06Z

Usage:
    conda activate torch_gpu
    pip install h3 pytz
    python run_phase2c.py --state TX
    python run_phase2c.py --state CA
    python run_phase2c.py --state ALL

Output per state:
    phase2/outputs/<state>/positives_labels.parquet
    phase2/outputs/<state>/phase2c_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import pytz

# ── Path setup ────────────────────────────────────────────────────────────────
PHASE2_ROOT = Path(__file__).resolve().parent
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from config.phase2_config import STATE_CONFIG, LOGS_DIR

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# 5 El Paso-area Texas counties that use Mountain Time (not Central)
EL_PASO_FIPS = {48141, 48229, 48243, 48269, 48383}

_TZ = {
    "TX_MAIN":   pytz.timezone("America/Chicago"),     # CDT/CST
    "TX_ELPASO": pytz.timezone("America/Denver"),      # MDT/MST
    "CA":        pytz.timezone("America/Los_Angeles"), # PDT/PST
}


# ── Timezone conversion ───────────────────────────────────────────────────────
def local_to_utc_window(disc_date, disc_time, state: str, fips_code=None):
    """
    Convert FPA-FOD local discovery date+time → (utc_date, window_hour).

    Parameters
    ----------
    disc_date : str or datetime.date    e.g. '2018-11-08'
    disc_time : int/float or NaN        HHMM format, e.g. 1430 = 14:30
    state     : 'TX' or 'CA'
    fips_code : int or None             TX only — for El Paso detection

    Returns
    -------
    (utc_date, window_hour)  where window_hour ∈ {0, 6, 12, 18}
    or (None, None) if conversion fails
    """
    from datetime import datetime

    # Parse date
    if pd.isna(disc_date):
        return None, None
    if isinstance(disc_date, str):
        try:
            d = datetime.strptime(disc_date[:10], "%Y-%m-%d")
        except ValueError:
            return None, None
    else:
        try:
            d = pd.Timestamp(disc_date).to_pydatetime()
        except Exception:
            return None, None

    # Parse HHMM time
    if pd.isna(disc_time) or disc_time is None:
        hour, minute = 12, 0  # default noon if time missing
    else:
        try:
            t = int(float(disc_time))
            hour, minute = t // 100, t % 100
            hour   = max(0, min(hour, 23))
            minute = max(0, min(minute, 59))
        except (ValueError, TypeError):
            hour, minute = 12, 0

    # Select timezone
    if state == "TX":
        try:
            fips_int = int(fips_code) if fips_code and not pd.isna(fips_code) else 0
        except (ValueError, TypeError):
            fips_int = 0
        tz = _TZ["TX_ELPASO"] if fips_int in EL_PASO_FIPS else _TZ["TX_MAIN"]
    else:
        tz = _TZ["CA"]

    # Build and convert
    try:
        local_dt = datetime(d.year, d.month, d.day, hour, minute)
        try:
            local_aware = tz.localize(local_dt, is_dst=None)
        except pytz.exceptions.AmbiguousTimeError:
            local_aware = tz.localize(local_dt, is_dst=False)
        except pytz.exceptions.NonExistentTimeError:
            local_aware = tz.localize(
                datetime(d.year, d.month, d.day, hour + 1, minute), is_dst=True
            )
        utc_dt = local_aware.astimezone(pytz.utc)
        window_hour = (utc_dt.hour // 6) * 6
        return utc_dt.date(), window_hour
    except Exception:
        return None, None


# ── H3 helpers ────────────────────────────────────────────────────────────────
def _get_h3():
    try:
        import h3
        return h3
    except ImportError:
        raise ImportError("h3 not installed. Run: pip install h3")


def _latlng_to_cell(h3_lib, lat, lon, res):
    try:
        try:
            return h3_lib.geo_to_h3(lat, lon, res)       # v3
        except AttributeError:
            return h3_lib.latlng_to_cell(lat, lon, res)  # v4
    except Exception:
        return None


def _cell_to_latlng(h3_lib, cell):
    try:
        try:
            return h3_lib.h3_to_geo(cell)       # v3
        except AttributeError:
            return h3_lib.cell_to_latlng(cell)  # v4
    except Exception:
        return (None, None)


# ── Label generator ───────────────────────────────────────────────────────────
def generate_labels(state_key: str, cfg: dict) -> pd.DataFrame:
    h3_lib     = _get_h3()
    resolution = cfg["h3_level"]
    parquet    = cfg["parquet"]

    logger.info(f"{'=' * 60}")
    logger.info(f"PHASE 2C — {cfg['name'].upper()} — Label Generation")
    logger.info(f"  H3 Resolution : R{resolution}")
    logger.info(f"  Source        : {parquet}")
    logger.info(f"{'=' * 60}")

    if not parquet.exists():
        raise FileNotFoundError(f"Phase 1 parquet not found: {parquet}\n"
                                f"Run Phase 1 first: conda run -n torch_gpu python run_phase1.py")

    # Load
    logger.info("Loading FPA-FOD parquet...")
    df = pd.read_parquet(parquet)
    logger.info(f"  Loaded: {len(df):,} total records, {len(df.columns)} columns")

    # Detect column names (FPA-FOD uses all-caps)
    def find_col(candidates):
        return next((c for c in candidates if c in df.columns), None)

    state_col  = find_col(["STATE", "State", "state"])
    size_col   = find_col(["FIRE_SIZE", "Fire_Size"])
    year_col   = find_col(["FIRE_YEAR", "Fire_Year"])
    lat_col    = find_col(["LATITUDE",  "Latitude",  "lat"])
    lon_col    = find_col(["LONGITUDE", "Longitude", "lon"])
    date_col   = find_col(["DISCOVERY_DATE", "disc_date", "DISC_DATE"])
    time_col   = find_col(["DISCOVERY_TIME", "disc_time", "DISC_TIME"])
    fips_col   = find_col(["FIPS_CODE", "fips_code"])

    if not lat_col or not lon_col:
        raise ValueError(f"LATITUDE/LONGITUDE columns not found. Available: {list(df.columns[:20])}")
    if not date_col:
        raise ValueError(f"DISCOVERY_DATE column not found. Available: {list(df.columns[:20])}")

    # Filter by state
    if state_col:
        df = df[df[state_col] == state_key].copy()
        logger.info(f"  After state filter ({state_key}): {len(df):,}")

    # Filter by fire size >= 1 acre
    if size_col:
        df[size_col] = pd.to_numeric(df[size_col], errors="coerce")
        df = df[df[size_col] >= 1.0].copy()
        logger.info(f"  After FIRE_SIZE >= 1 acre: {len(df):,}")

    # Filter by year 2014–2020
    if year_col:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df = df[df[year_col].between(2014, 2020)].copy()
        logger.info(f"  After FIRE_YEAR 2014–2020: {len(df):,}")

    # Drop NaN lat/lon
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    df[lat_col] = df[lat_col].astype(float)
    df[lon_col] = df[lon_col].astype(float)
    logger.info(f"  After dropping NaN lat/lon: {len(df):,}")

    logger.info(f"\nConverting {len(df):,} fires → H3 cells + UTC windows...")

    # Process each fire event
    rows       = []
    n_ok       = 0
    n_fail_h3  = 0
    n_fail_utc = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        if i % 5000 == 0 and i > 0:
            logger.info(f"  Progress: {i:,}/{len(df):,} ({100*i/len(df):.0f}%)")

        lat  = row[lat_col]
        lon  = row[lon_col]
        date = row[date_col]
        time = row[time_col] if time_col else None
        fips = row[fips_col] if fips_col else None
        yr   = int(row[year_col]) if year_col else None

        # H3 cell
        h3_cell = _latlng_to_cell(h3_lib, lat, lon, resolution)
        if h3_cell is None:
            n_fail_h3 += 1
            continue

        # UTC window
        utc_date, window_hour = local_to_utc_window(date, time, state_key, fips)
        if utc_date is None:
            n_fail_utc += 1
            continue

        # H3 centroid (canonical location — not the exact fire point)
        clat, clon = _cell_to_latlng(h3_lib, h3_cell)

        window_ts = pd.Timestamp(
            year=utc_date.year, month=utc_date.month, day=utc_date.day,
            hour=window_hour
        )

        rows.append({
            "h3_cell":       h3_cell,
            "date_utc":      str(utc_date),
            "window_hour":   window_hour,
            "window_6h_utc": window_ts,
            "label":         1,
            "centroid_lat":  round(clat, 6),
            "centroid_lon":  round(clon, 6),
            "fire_year":     yr,
            "state":         state_key,
        })
        n_ok += 1

    logger.info(f"\n  Results:")
    logger.info(f"    ✔ Successful: {n_ok:,}")
    logger.info(f"    ✘ H3 failures: {n_fail_h3:,}")
    logger.info(f"    ✘ UTC failures: {n_fail_utc:,}")

    result = pd.DataFrame(rows)

    if len(result) > 0:
        # Window distribution
        wd = result["window_hour"].value_counts().sort_index()
        logger.info("\n  UTC Window Distribution:")
        for wh, cnt in wd.items():
            pct = 100 * cnt / len(result)
            logger.info(f"    {wh:02d}Z : {cnt:>6,} fires  ({pct:.1f}%)")

        # Year distribution
        yd = result["fire_year"].value_counts().sort_index()
        logger.info("\n  Year Distribution:")
        for yr, cnt in yd.items():
            logger.info(f"    {yr} : {cnt:,}")

    return result


# ── Runner ────────────────────────────────────────────────────────────────────
def run_state(state_key: str) -> bool:
    cfg        = STATE_CONFIG[state_key]
    output_dir = cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = generate_labels(state_key, cfg)
        if len(df) == 0:
            logger.error("No positive labels generated — check input parquet")
            return False

        out = output_dir / "positives_labels.parquet"
        df.to_parquet(out, index=False, compression="snappy")
        mb = out.stat().st_size / 1e6
        logger.info(f"\n  Saved: {out}  ({mb:.1f} MB)")

        # Summary CSV
        pd.DataFrame({
            "state": [state_key],
            "total_positives": [len(df)],
            "unique_h3_cells": [df["h3_cell"].nunique()],
            "unique_dates": [df["date_utc"].nunique()],
            "h3_resolution": [cfg["h3_level"]],
        }).to_csv(output_dir / "phase2c_summary.csv", index=False)

        logger.info(f"  Summary saved: {output_dir}/phase2c_summary.csv")
        return True

    except Exception as e:
        logger.error(f"Phase 2C FAILED [{state_key}]: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Phase 2C — Label Generation")
    parser.add_argument("--state", choices=["TX", "CA", "ALL"], required=True)
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "phase2c.log", encoding="utf-8"),
        ],
    )

    states = ["TX", "CA"] if args.state == "ALL" else [args.state]
    results = {s: run_state(s) for s in states}

    print("\n" + "═" * 60)
    for s, ok in results.items():
        icon = "✔ SUCCESS" if ok else "✘ FAILED"
        print(f"  {STATE_CONFIG[s]['name']:<15} {icon}")
    print("═" * 60)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
