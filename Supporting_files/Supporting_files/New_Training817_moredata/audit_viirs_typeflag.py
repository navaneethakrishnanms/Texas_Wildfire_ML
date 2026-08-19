"""
New_Training817_moredata — VIIRS type-flag audit of the flare filter.

Our training labels remove 531 "flare cells" found by a heuristic (any
res-8 cell with fire detections on >3% of all days = presumed industrial
flare/hotspot, not wildfires). VIIRS's standard-processing (SP) product
carries an official per-detection `type` classification our NRT downloads
lack:
    0 = presumed vegetation fire
    1 = active volcano
    2 = other static land source   (flares, industrial heat)
    3 = offshore

This script pulls SP data for Texas over sampled 10-day windows spanning
2023-2025 (FIRMS API limit: 10 days/call), assigns detections to res-8
cells, and checks BOTH directions:
  1. Of our 531 flare cells seen in the sample, what fraction of their
     detections are type=2?  (Did we remove real flares?)
  2. Are there cells NOT in our flare list whose detections are mostly
     type=2 with substantial counts?  (Did we MISS flares that are
     polluting the training labels?)

Output: New_Training817_moredata/viirs_typeflag_audit.json
"""
import json, time, io, urllib.request
import numpy as np, pandas as pd, h3
from pathlib import Path

HERE = Path(__file__).resolve().parent
TF = HERE.parent
KEY = (TF / "firms_map_key.txt").read_text().strip()
BBOX = "-106.7,25.75,-93.4,36.65"
PRODUCT = "VIIRS_SNPP_SP"

# 48 windows x 5 days, spread across 2023 - mid-2025 (~240 sampled days, all
# seasons). SP product caps at 5 days/call (10-day requests -> HTTP 400,
# verified empirically); SP also lags realtime by months, so don't sample
# recent dates.
STARTS = pd.date_range('2023-01-05', '2025-06-01', periods=48).normalize()


def log(m): print(m, flush=True)


def pull(start):
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/"
           f"{PRODUCT}/{BBOX}/5/{start.date()}")
    with urllib.request.urlopen(url, timeout=120) as r:
        txt = r.read().decode()
    if not txt.startswith('latitude'):
        raise RuntimeError(f"unexpected response for {start.date()}: {txt[:100]}")
    return pd.read_csv(io.StringIO(txt))


def run():
    flare = pd.read_parquet(TF / "data/labels_fused/flare_cells.parquet")
    flare_cells = set(flare['h3_cell'] if 'h3_cell' in flare.columns else flare.iloc[:, 0])
    log(f"flare cells (heuristic): {len(flare_cells)}")

    frames = []
    for i, s in enumerate(STARTS):
        for attempt in range(3):
            try:
                d = pull(s)
                break
            except Exception as e:
                log(f"  retry {attempt+1} for {s.date()}: {e}")
                time.sleep(10)
        else:
            log(f"  SKIPPED {s.date()}"); continue
        frames.append(d)
        log(f"  [{i+1}/{len(STARTS)}] {s.date()}: {len(d):,} detections")
        time.sleep(2)
    v = pd.concat(frames, ignore_index=True)
    log(f"total sampled detections: {len(v):,}")
    v['h3_cell'] = [h3.latlng_to_cell(la, lo, 8) for la, lo in zip(v['latitude'], v['longitude'])]
    v['is_static'] = v['type'].isin([2, 3])

    g = v.groupby('h3_cell').agg(n=('type', 'size'), n_static=('is_static', 'sum'),
                                  n_veg=('type', lambda s: int((s == 0).sum())))
    g['frac_static'] = g['n_static'] / g['n']
    g = g.reset_index()
    g['in_flare_list'] = g['h3_cell'].isin(flare_cells)

    # Direction 1: our flare cells -- are they really static sources?
    fl = g[g['in_flare_list']]
    fl_sub = fl[fl['n'] >= 5]     # enough sample to judge
    confirmed = fl_sub[fl_sub['frac_static'] >= 0.5]
    log(f"\nDirection 1 — our flare cells seen in sample: {len(fl)} "
        f"(with >=5 dets: {len(fl_sub)})")
    log(f"  confirmed static-source (>=50% type-2/3): {len(confirmed)}"
        f" ({len(confirmed)/max(len(fl_sub),1)*100:.0f}%)")
    log(f"  median frac_static among them: {fl_sub['frac_static'].median():.2f}")

    # Direction 2: cells NOT in our list that look like flares
    nf = g[~g['in_flare_list']]
    missed = nf[(nf['n'] >= 10) & (nf['frac_static'] >= 0.5)]
    log(f"\nDirection 2 — NON-flare-list cells with >=10 dets & >=50% static: {len(missed)}")
    log(f"  their detections total {int(missed['n'].sum()):,} "
        f"({int(missed['n_static'].sum()):,} static-typed)")
    tot_labels_at_risk = int(missed['n'].sum())

    out = dict(
        product=PRODUCT, windows=len(STARTS), sampled_detections=int(len(v)),
        flare_list_size=len(flare_cells),
        direction1=dict(flare_cells_seen=int(len(fl)), with_5plus_dets=int(len(fl_sub)),
                        confirmed_static=int(len(confirmed)),
                        confirm_rate=round(float(len(confirmed)/max(len(fl_sub),1)), 3),
                        median_frac_static=round(float(fl_sub['frac_static'].median()), 3)),
        direction2=dict(missed_flare_candidates=int(len(missed)),
                        their_detections=tot_labels_at_risk,
                        cells=missed.nlargest(20, 'n')[['h3_cell', 'n', 'frac_static']]
                              .assign(frac_static=lambda d: d['frac_static'].round(2))
                              .to_dict('records')))
    json.dump(out, open(HERE / "viirs_typeflag_audit.json", 'w'), indent=2)
    missed[['h3_cell', 'n', 'n_static', 'frac_static']].to_parquet(
        HERE / "missed_flare_candidates.parquet", index=False)
    log("\nSaved -> viirs_typeflag_audit.json, missed_flare_candidates.parquet")


if __name__ == '__main__':
    run()
