"""
NM_OK_TexRev — Step 7: fuse FPA-FOD (2014-2020) + VIIRS (2014-2026) into
daily ignition labels per state, same convention as TX's
scripts/02_build_labels.py: FPA-FOD gives complete imputed-time coverage
through 2020, VIIRS extends real-time coverage through 2026 and is the
only source for post-2020 positives. Flare cells excluded. Filtered to
each state's own static-master polygon (not the download bbox) so no
cross-state spillover, same fix as TX's SANITY_CHECK finding F1.

Output: NM_OK_TexRev/data/labels_daily_{nm,ok}.parquet
        [h3_cell, date, label, time_source, cause, max_size_acres, n_sources]
"""
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def log(m): print(m, flush=True)


def build_for_state(state):
    out = ROOT / "data" / f"labels_daily_{state}.parquet"
    if out.exists():
        log(f"[skip] {out.name} exists"); return

    fpa_p = ROOT / "data" / f"fpa_fod_{state}_h3.parquet"
    viirs_p = ROOT / "data" / "labels_viirs" / f"viirs_{state}_h3.parquet"
    if not (fpa_p.exists() and viirs_p.exists()):
        log(f"[{state}] missing inputs (need {fpa_p.name} + {viirs_p.name})"); return

    fpa = pd.read_parquet(fpa_p)
    fpa_daily = fpa.groupby(['h3_cell', 'date']).agg(
        cause=('cause_class', 'first'), max_size_acres=('max_size_acres', 'max')
    ).reset_index()
    fpa_daily['src_fpa'] = 1
    log(f"[{state}] FPA-FOD daily cell-days: {len(fpa_daily):,}")

    v = pd.read_parquet(viirs_p)
    v['date'] = pd.to_datetime(v['date']).dt.normalize()
    viirs_daily = v.groupby(['h3_cell', 'date']).agg(
        first_hour=('hour', 'min'), n_det=('hour', 'size'),
        frp_max=('frp', 'max') if 'frp' in v.columns else ('hour', 'size'),
    ).reset_index()
    viirs_daily['src_viirs'] = 1
    log(f"[{state}] VIIRS daily cell-days: {len(viirs_daily):,} "
        f"({viirs_daily.date.dt.year.min():.0f}-{viirs_daily.date.dt.year.max():.0f})")

    m = fpa_daily.merge(viirs_daily, on=['h3_cell', 'date'], how='outer')
    m['src_fpa'] = m['src_fpa'].fillna(0)
    m['src_viirs'] = m.get('src_viirs', 0)
    m['src_viirs'] = m['src_viirs'].fillna(0)
    m['n_sources'] = (m['src_fpa'] + m['src_viirs']).astype(int)
    m['time_source'] = np.where(m['src_viirs'] == 1, 'viirs', 'imputed')
    m['cause'] = m['cause'].fillna('satellite_only')
    m['label'] = 1
    out_df = m[['h3_cell', 'date', 'label', 'time_source', 'cause',
                'max_size_acres', 'n_sources']].copy()

    n0 = len(out_df)
    state_cells = set(pd.read_parquet(ROOT / "data" / f"static_master_{state}.parquet",
                                       columns=['h3_cell'])['h3_cell'])
    out_df = out_df[out_df.h3_cell.isin(state_cells)]
    log(f"[{state}] state-polygon filter: {n0:,} -> {len(out_df):,} "
        f"(dropped {n0-len(out_df):,} out-of-state spillover)")

    flare_p = ROOT / "data" / f"flare_cells_{state}.parquet"
    if flare_p.exists():
        flare = set(pd.read_parquet(flare_p)['h3_cell'])
        n1 = len(out_df)
        out_df = out_df[~out_df.h3_cell.isin(flare)]
        log(f"[{state}] flare filter: {n1:,} -> {len(out_df):,} "
            f"(dropped {n1-len(out_df):,} industrial-hotspot rows)")
    else:
        log(f"[{state}] WARNING: no flare list found, proceeding WITHOUT flare filter "
            f"(run 06_build_flare_lists_nm_ok.py first)")

    out_df.to_parquet(out, index=False)
    log(f"[{state}] fused ignition cell-days: {len(out_df):,}")
    log(f"[{state}] by year:\n{out_df.groupby(out_df.date.dt.year).size().to_string()}")
    log(f"[{state}] post-2020 (VIIRS-only) labels: {(out_df.date.dt.year>=2021).sum():,}")


if __name__ == '__main__':
    for state in ['nm', 'ok']:
        build_for_state(state)
