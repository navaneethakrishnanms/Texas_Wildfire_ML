"""
run_all_eda.py -- Master EDA Runner (Texas + California)
=========================================================
Runs all 7 EDA scripts for both states in sequence.
Usage:
    conda run -n torch_gpu --no-capture-output python maps/run_all_eda.py

Flags:
    --state TX         run only Texas
    --state CA         run only California
    --script 01        run only script number 01 (for all states)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent    # -> V2/maps/
V2_DIR   = BASE_DIR.parent                   # -> V2/

SCRIPTS = [
    "01_data_overview.py",
    "02_distributions.py",
    "03_correlation_analysis.py",
    "04_geospatial_temporal.py",
    "05_advanced_eda.py",
    "06_summary_report.py",
    "07_interactive_hotspot_map.py",
]

STATES = {
    "TX": {"name": "Texas",      "scripts_dir": BASE_DIR / "texas"      / "scripts",
                                  "out_dir":     BASE_DIR / "texas"      / "eda_outputs"},
    "CA": {"name": "California", "scripts_dir": BASE_DIR / "california" / "scripts",
                                  "out_dir":     BASE_DIR / "california" / "eda_outputs"},
}


def run_script(script_path: Path, env: dict) -> tuple[bool, float]:
    """Run a single script. Returns (success, elapsed_seconds)."""
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env, cwd=str(V2_DIR),
            capture_output=False,
            timeout=600,
        )
        elapsed = time.perf_counter() - t0
        return result.returncode == 0, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"     [TIMEOUT] Script exceeded 600 seconds")
        return False, elapsed
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"     [ERROR] {exc}")
        return False, elapsed


def print_manifest(out_dir: Path, state_name: str) -> None:
    """Print all files generated in the eda_outputs directory."""
    if not out_dir.exists():
        print(f"  [MISSING] {out_dir}")
        return
    files = sorted(out_dir.iterdir())
    print(f"\n  {state_name} outputs ({out_dir}):")
    print(f"  {'File':<55}  {'Size':>10}")
    print(f"  {'-'*55}  {'-'*10}")
    total_bytes = 0
    for f in files:
        size = f.stat().st_size
        total_bytes += size
        size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"
        print(f"  {f.name:<55}  {size_str:>10}")
    print(f"  {'TOTAL':<55}  {total_bytes/1024/1024:.1f} MB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_all_eda.py",
        description="Master EDA runner for Texas and California wildfire datasets"
    )
    parser.add_argument("--state",  choices=["TX", "CA", "both"], default="both",
                        help="Which state to run (default: both)")
    parser.add_argument("--script", type=str, default=None,
                        help="Run only scripts matching this prefix, e.g. '01' or '07'")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    states = ["TX", "CA"] if args.state == "both" else [args.state]

    # Build env with UTF-8 encoding forced
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # Filter scripts if --script flag given
    script_list = SCRIPTS
    if args.script:
        script_list = [s for s in SCRIPTS if s.startswith(args.script)]
        if not script_list:
            print(f"[ERROR] No scripts matching prefix '{args.script}'")
            sys.exit(1)

    print("=" * 72)
    print("  WILDFIRE EDA MASTER RUNNER")
    print(f"  States  : {states}")
    print(f"  Scripts : {[s[:2] for s in script_list]}")
    print("=" * 72)

    global_t0  = time.perf_counter()
    all_results: list[dict] = []

    for state_code in states:
        state_cfg  = STATES[state_code]
        state_name = state_cfg["name"]
        scripts_dir = state_cfg["scripts_dir"]
        out_dir    = state_cfg["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*72}")
        print(f"  STATE: {state_name} ({state_code})")
        print(f"{'='*72}")

        for script_name in script_list:
            script_path = scripts_dir / script_name
            if not script_path.exists():
                print(f"\n  [SKIP] {script_name} -- not found at {script_path}")
                all_results.append({"state": state_code, "script": script_name,
                                    "status": "SKIP", "elapsed": 0.0})
                continue

            print(f"\n  Running {script_name} for {state_name} ...")
            print(f"  {'-'*68}")
            ok, elapsed = run_script(script_path, env)
            status = "[DONE]  " if ok else "[FAILED]"
            print(f"  {'-'*68}")
            print(f"  {status}  {script_name}  --  {elapsed:.1f}s")
            all_results.append({"state": state_code, "script": script_name,
                                 "status": "DONE" if ok else "FAILED", "elapsed": elapsed})

    # ─── FINAL SUMMARY ───────────────────────────────────────────
    total_elapsed = time.perf_counter() - global_t0
    print(f"\n{'='*72}")
    print("  EXECUTION SUMMARY")
    print("=" * 72)
    print(f"  {'State':<6}  {'Script':<35}  {'Status':<10}  {'Elapsed':>8}")
    print(f"  {'-'*6}  {'-'*35}  {'-'*10}  {'-'*8}")
    for r in all_results:
        print(f"  {r['state']:<6}  {r['script']:<35}  {r['status']:<10}  {r['elapsed']:>6.1f}s")
    print(f"\n  Total wall time : {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")
    done_ct   = sum(1 for r in all_results if r["status"] == "DONE")
    failed_ct = sum(1 for r in all_results if r["status"] == "FAILED")
    print(f"  Done    : {done_ct}")
    print(f"  Failed  : {failed_ct}")

    # ─── FILE MANIFEST ───────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  OUTPUT FILE MANIFEST")
    print("=" * 72)
    for state_code in states:
        print_manifest(STATES[state_code]["out_dir"], STATES[state_code]["name"])

    if failed_ct > 0:
        print(f"\n[WARNING] {failed_ct} script(s) failed. Check output above for errors.")
        sys.exit(1)
    else:
        print(f"\n[ALL DONE] EDA complete for {states}. Open eda_outputs/ to view results.")


if __name__ == "__main__":
    main()
