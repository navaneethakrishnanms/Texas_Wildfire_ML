"""
live_feed_daemon.py
---------------------
Simulates a live ingestion feed by revealing one pre-scored August day at a
time from live_august_staging.parquet (built by build_live_august_feed.py).

All 31 days were already scored by build_live_august_feed.py -- this daemon
does not compute anything new. It only advances a "published_through"
pointer in data/live_feed_state.json every INTERVAL_MINUTES. The FastAPI
backend reads that pointer and refuses to serve any August date beyond it,
so the portal genuinely appears to "gain" one more day of live data at each
tick, the same visible behavior a real Databricks-sync-driven feed would have.

Default: 1 day published every 2 minutes -> full August revealed in ~62 min.
Override with:  python live_feed_daemon.py --interval-minutes 5

Usage:
    python TDIS_REPLICA/ingestion/live_feed_daemon.py
"""

import argparse
import json
import time
from datetime import datetime, timezone

from common import DATA_OUT, LIVE_START, LIVE_END
import pandas as pd

STATE_PATH = DATA_OUT / "live_feed_state.json"


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "published_through": None,  # None = nothing live yet, only historical
        "started_at": datetime.now(timezone.utc).isoformat(),
        "last_update": None,
    }


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def run(interval_minutes: float):
    print("=" * 60)
    print("  Live feed daemon -- simulated August 2026 drip")
    print(f"  Publishing one day every {interval_minutes} minute(s)")
    print("=" * 60)

    dates = pd.date_range(LIVE_START, LIVE_END, freq="D")
    state = load_state()
    print(f"  Resuming from published_through = {state['published_through']}")

    for d in dates:
        d_str = d.strftime("%Y-%m-%d")
        if state["published_through"] is not None and d_str <= state["published_through"]:
            continue  # already published in a previous run of this daemon
        time.sleep(interval_minutes * 60)
        state["published_through"] = d_str
        state["last_update"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        print(f"  [{state['last_update']}] Published live risk data for {d_str}")

    print("\n  All 31 August days published. Feed is now fully caught up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-minutes", type=float, default=2.0)
    args = parser.parse_args()
    run(args.interval_minutes)
