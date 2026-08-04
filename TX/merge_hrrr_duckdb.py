"""
merge_hrrr_duckdb.py
---------------------
Join HRRR sub-daily features onto the clean TX training parquet using DuckDB
instead of pandas.

Why not the pandas merge_and_save() in fetch_hrrr_tx.py:
  The yearly hrrr_tx_YYYY.parquet files hold values for the FULL 317,142-cell
  grid at every (date, window), not just the 375,779 training rows. Loading
  and concatenating all ~933M rows into pandas (h3_cell as an object/string
  column) needs on the order of 150-250 GB RAM. This machine has 64 GB.

  DuckDB reads the parquet files lazily and hash-joins them against the small
  training table, spilling to disk as needed, so peak RAM stays in the
  single-digit GB range regardless of how many yearly files exist (2014-2020).

Usage:
  python merge_hrrr_duckdb.py
  python merge_hrrr_duckdb.py --years 2014 2015 2016 2017 2018 2019 2020
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
HRRR_DIR = DATA_DIR / "hrrr"
CLEAN_PQ = DATA_DIR / "full_tx_clean.parquet"
OUT_PQ = HRRR_DIR / "hrrr_tx_all.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=None,
                         help="Restrict join to these HRRR year files (default: all found)")
    parser.add_argument("--threads", type=int, default=8,
                         help="DuckDB worker threads (default: 8)")
    parser.add_argument("--mem-limit", type=str, default="12GB",
                         help="DuckDB memory soft-limit before spilling to disk (default: 12GB)")
    args = parser.parse_args()

    if not CLEAN_PQ.exists():
        raise SystemExit(f"Training parquet not found: {CLEAN_PQ}")

    if args.years:
        year_globs = [str(HRRR_DIR / f"hrrr_tx_{y}.parquet") for y in args.years]
        missing = [g for g in year_globs if not Path(g).exists()]
        if missing:
            raise SystemExit(f"Missing HRRR year file(s): {missing}")
        hrrr_glob = "[" + ", ".join(f"'{g}'" for g in year_globs) + "]"
    else:
        found = sorted(HRRR_DIR.glob("hrrr_tx_????.parquet"))
        if not found:
            raise SystemExit(f"No hrrr_tx_????.parquet files found in {HRRR_DIR}")
        print(f"Found {len(found)} year file(s): {[f.name for f in found]}")
        hrrr_glob = "[" + ", ".join(f"'{f.as_posix()}'" for f in found) + "]"

    con = duckdb.connect()
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = '{args.mem_limit}'")
    con.execute(f"SET temp_directory = '{HRRR_DIR / 'duckdb_tmp'}'")

    query = f"""
        COPY (
            SELECT
                t.*,
                h.temp_pw,
                h.rh_pw,
                h.wind_pw,
                h.vpd_pw   AS vpd_pw_hrrr,
                h.hpbl_pw,
                h.dswrf_pw,
                COALESCE(h.hrrr_pw, 0)::TINYINT AS hrrr_pw
            FROM read_parquet('{CLEAN_PQ.as_posix()}') t
            LEFT JOIN read_parquet({hrrr_glob}) h
              ON t.h3_cell     = h.h3_cell
             AND t.date_utc    = h.date_utc
             AND t.window_hour = h.window_hour
        ) TO '{OUT_PQ.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """
    print("Running DuckDB join...")
    con.execute(query)

    # Summary
    n_total, n_with = con.execute(f"""
        SELECT COUNT(*), SUM(hrrr_pw)
        FROM read_parquet('{OUT_PQ.as_posix()}')
    """).fetchone()
    n_train = con.execute(f"SELECT COUNT(*) FROM read_parquet('{CLEAN_PQ.as_posix()}')").fetchone()[0]

    print(f"\nTraining rows:        {n_train:,}")
    print(f"Output rows:           {n_total:,}")
    print(f"Rows with HRRR:        {n_with:,}  ({100*n_with/n_total:.1f}%)")
    print(f"Rows without HRRR:     {n_total-n_with:,}  ({100*(n_total-n_with)/n_total:.1f}%)")

    print("\nCoverage by year:")
    rows = con.execute(f"""
        SELECT year(date_utc) AS yr, COUNT(*) AS n, SUM(hrrr_pw) AS n_hrrr
        FROM read_parquet('{OUT_PQ.as_posix()}')
        GROUP BY 1 ORDER BY 1
    """).fetchall()
    for yr, n, n_hrrr in rows:
        print(f"  {yr}: {n_hrrr:,}/{n:,}  ({100*n_hrrr/n:.1f}%)")

    print(f"\nSaved: {OUT_PQ}  ({OUT_PQ.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
