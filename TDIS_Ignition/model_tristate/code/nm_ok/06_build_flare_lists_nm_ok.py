"""
NM_OK_TexRev — Step 6: episode-based industrial-hotspot (flare) exclusion
lists for NM + OK. Per QC_FINAL_REVIEW.md finding QC-A: a naive
>=N-fire-days/365d rule is WRONG -- it sweeps in 28% of real fires because
it catches multi-day wildfires, not just industrial flares. The fix is
EPISODE-based: consecutive detection-days within a gap<=2d collapse into
ONE episode (one wildfire); only cells with MANY DISTINCT, non-consecutive
episodes in a rolling window look industrial (a flare re-ignites constantly,
a wildfire burns once and stops).

No prior NM/OK flare-list script exists to copy -- this reimplements the
same concept documented for TX (flare_cells_v2.parquet) fresh. Threshold
(>=6 distinct episodes in any rolling 365d) is a FIRST-PASS choice, not
independently calibrated against known NM/OK industrial sites yet --
the explicit target case is Permian Basin flaring in southeast NM (see
PRD_DATA.md: "NM Permian + OK required"). Flag output for manual review
against a Permian Basin map before trusting it the way TX's list is trusted.

Output: NM_OK_TexRev/data/flare_cells_{nm,ok}.parquet [h3_cell]
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GAP_DAYS = 2          # consecutive detections within this gap = one episode
MIN_EPISODES = 6      # >=N distinct episodes in ANY rolling 365d window = flag as industrial
WINDOW_DAYS = 365


def log(m): print(m, flush=True)


def episodes_per_cell(v):
    v = v.sort_values(['h3_cell', 'date']).drop_duplicates(['h3_cell', 'date'])
    v['gap'] = v.groupby('h3_cell')['date'].diff().dt.days
    v['new_episode'] = (v['gap'].isna()) | (v['gap'] > GAP_DAYS)
    v['episode_id'] = v.groupby('h3_cell')['new_episode'].cumsum()
    return v


def rolling_max_episode_count(v):
    """For each cell, the max number of DISTINCT episodes whose start falls
    within any rolling WINDOW_DAYS-day window (checked at each episode start)."""
    starts = v.groupby(['h3_cell', 'episode_id'])['date'].min().reset_index()
    out = {}
    for cell, g in starts.groupby('h3_cell'):
        dates = g['date'].sort_values().values
        best = 0
        for i, d0 in enumerate(dates):
            cnt = ((dates >= d0) & (dates < d0 + pd.Timedelta(days=WINDOW_DAYS))).sum()
            best = max(best, cnt)
        out[cell] = best
    return out


def build_for_state(state):
    out = ROOT / "data" / f"flare_cells_{state}.parquet"
    if out.exists():
        log(f"[skip] {out.name} exists"); return

    vp = ROOT / "data" / "labels_viirs" / f"viirs_{state}_h3.parquet"
    if not vp.exists():
        log(f"[{state}] {vp.name} not found yet -- run 03_download_viirs_nm_ok.py first"); return

    v = pd.read_parquet(vp, columns=['h3_cell', 'date'])
    v['date'] = pd.to_datetime(v['date']).dt.normalize()
    log(f"[{state}] {len(v):,} raw VIIRS detection-days, {v.h3_cell.nunique():,} cells")

    v = episodes_per_cell(v)
    counts = rolling_max_episode_count(v)
    flare_cells = [c for c, n in counts.items() if n >= MIN_EPISODES]
    log(f"[{state}] {len(flare_cells):,} cells with >={MIN_EPISODES} distinct episodes "
        f"in any {WINDOW_DAYS}d window (flagged as industrial/flare)")

    pd.DataFrame({'h3_cell': flare_cells}).to_parquet(out, index=False)
    log(f"[{state}] saved {out.name} -- MANUAL REVIEW NEEDED before trusting "
        f"(check flagged NM cells against Permian Basin map)")


if __name__ == '__main__':
    for state in ['nm', 'ok']:
        build_for_state(state)
