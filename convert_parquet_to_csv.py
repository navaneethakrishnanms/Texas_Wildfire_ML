#!/usr/bin/env python3
"""
convert_parquet_to_csv.py — convert every .parquet found under the given roots
into a .csv sitting in the SAME directory, with the SAME basename.

  data/tx_static_master.parquet  ->  data/tx_static_master.csv

Streams row-group by row-group via pyarrow, so a 3.6M-row / 40-column table
converts in constant memory instead of loading the whole frame.

Usage
-----
  python convert_parquet_to_csv.py                       # both shared folders
  python convert_parquet_to_csv.py Focused_Files         # one root
  python convert_parquet_to_csv.py --gzip                # write .csv.gz instead
  python convert_parquet_to_csv.py --dry-run             # list, convert nothing
  python convert_parquet_to_csv.py --force               # re-convert existing
  python convert_parquet_to_csv.py --max-mb 100          # skip huge parquets

Notes
-----
* CSV is ~5-6x larger than parquet. The 22 files in Focused_Files +
  Supporting_files are ~862 MB of parquet -> expect ~4-6 GB of CSV.
  Use --gzip if disk space matters; pandas/Excel-via-import read .csv.gz fine.
* Existing .csv files are skipped unless --force is given.
* Timestamp columns are written ISO-8601 (e.g. 2026-07-29).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from pyarrow import csv as pacsv
except ImportError:
    sys.exit("pyarrow is required:  pip install pyarrow")

DEFAULT_ROOTS = ["Focused_Files", "Supporting_files"]
BATCH_ROWS = 200_000


def _timestamp_casts(pf: "pq.ParquetFile") -> dict[str, "pa.DataType"]:
    """Pick a narrower arrow type per timestamp column, to tidy the CSV rendering.

    Left alone, pyarrow writes nanosecond timestamps as `2018-02-04 00:00:00.000000000`.
    These tables are daily, so a column whose first batch is entirely midnight is cast to
    date32 and lands as a plain `YYYY-MM-DD`; anything with a real time-of-day is cast to
    second precision and lands as `YYYY-MM-DD HH:MM:SS`. Casting (rather than strftime)
    keeps this working on Windows boxes with no tzdata installed.
    """
    ts_cols = [f.name for f in pf.schema_arrow
               if pa.types.is_timestamp(f.type) and f.type.tz is None]
    if not ts_cols:
        return {}
    casts: dict[str, pa.DataType] = {c: pa.timestamp("s") for c in ts_cols}
    try:
        probe = next(pf.iter_batches(batch_size=BATCH_ROWS, columns=ts_cols))
    except StopIteration:
        return casts
    for c in ts_cols:
        col = probe.column(probe.schema.get_field_index(c))
        # round-trip through date32: equal everywhere => no time-of-day information
        same = pc.all(pc.equal(col, pc.cast(pc.cast(col, pa.date32()), col.type))).as_py()
        if same:
            casts[c] = pa.date32()
    return casts


def _apply_casts(batch: "pa.RecordBatch", casts: dict[str, "pa.DataType"]) -> "pa.RecordBatch":
    if not casts:
        return batch
    arrays, fields = [], []
    for i, field in enumerate(batch.schema):
        arr = batch.column(i)
        if field.name in casts:
            arr = pc.cast(arr, casts[field.name])
            field = pa.field(field.name, arr.type, field.nullable)
        arrays.append(arr)
        fields.append(field)
    return pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))


def _retry_fs(op, attempts: int = 6, delay: float = 0.5):
    """Windows hands out transient WinError 32 on a file an AV scanner has just opened.
    Retry the rename/unlink a few times before giving up."""
    for i in range(attempts):
        try:
            return op()
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay * (i + 1))


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def convert(src: Path, gzip: bool, force: bool) -> tuple[str, int]:
    """Returns (status, bytes_written)."""
    dst = src.with_suffix(".csv.gz" if gzip else ".csv")
    if dst.exists() and not force:
        return "skip (exists)", 0

    pf = pq.ParquetFile(src)
    n_rows, n_cols = pf.metadata.num_rows, pf.metadata.num_columns

    opts = pacsv.WriteOptions(include_header=True)
    ts_casts = _timestamp_casts(pf)
    t0 = time.time()
    written = 0
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        if gzip:
            import gzip as gz

            raw = gz.open(tmp, "wb", compresslevel=6)
        else:
            raw = open(tmp, "wb")
        with raw:
            writer = None
            for batch in pf.iter_batches(batch_size=BATCH_ROWS):
                batch = _apply_casts(batch, ts_casts)
                if writer is None:
                    writer = pacsv.CSVWriter(raw, batch.schema, write_options=opts)
                writer.write_batch(batch)
                written += batch.num_rows
            if writer is not None:
                writer.close()
            elif n_rows == 0:  # empty table: header only
                pacsv.CSVWriter(raw, pf.schema_arrow, write_options=opts).close()
        _retry_fs(lambda: tmp.replace(dst))
    except Exception:
        try:
            _retry_fs(lambda: tmp.unlink(missing_ok=True))
        except OSError:
            pass
        raise

    size = dst.stat().st_size
    print(
        f"    {written:>10,} rows x {n_cols:>2} cols -> {dst.name}"
        f"  ({human(size)}, {time.time() - t0:.1f}s)"
    )
    return "ok", size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=DEFAULT_ROOTS,
                    help="directories to scan recursively (default: %s)" % ", ".join(DEFAULT_ROOTS))
    ap.add_argument("--gzip", action="store_true", help="write .csv.gz instead of .csv")
    ap.add_argument("--force", action="store_true", help="overwrite existing output files")
    ap.add_argument("--dry-run", action="store_true", help="list what would be converted")
    ap.add_argument("--max-mb", type=float, default=None,
                    help="skip parquet files larger than this many MB")
    args = ap.parse_args()

    files: list[Path] = []
    for root in args.roots:
        p = Path(root)
        if not p.exists():
            print(f"!! root not found, skipping: {p}", file=sys.stderr)
            continue
        files.extend(sorted(p.rglob("*.parquet")))

    if not files:
        print("No .parquet files found.")
        return 1

    total_in = sum(f.stat().st_size for f in files)
    print(f"Found {len(files)} parquet file(s), {human(total_in)} total.")
    if args.dry_run:
        for f in files:
            print(f"  {human(f.stat().st_size):>10}  {f}")
        print("\n(dry run — nothing written)")
        return 0

    ok = skipped = failed = 0
    total_out = 0
    for i, f in enumerate(files, 1):
        mb = f.stat().st_size / 1e6
        print(f"[{i}/{len(files)}] {f}  ({mb:.1f} MB)")
        if args.max_mb is not None and mb > args.max_mb:
            print(f"    skip — larger than --max-mb {args.max_mb}")
            skipped += 1
            continue
        try:
            status, size = convert(f, args.gzip, args.force)
        except Exception as e:  # noqa: BLE001 — report and keep going
            print(f"    FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
            continue
        if status == "ok":
            ok += 1
            total_out += size
        else:
            print(f"    {status} — use --force to overwrite")
            skipped += 1

    print(f"\nDone. converted={ok}  skipped={skipped}  failed={failed}"
          f"  csv written={human(total_out)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
