"""
NM_OK_TexRev — Step 8: tri-state training table. Same matched-sampling
scheme as TX's scripts/03_build_dataset.py (NEG_PER_POS=8 temporal + 0.4x
spatial never-fire negatives), same 12Z-day-before HRRR peak-weather
mechanism as rev4_pipeline.py (reused directly, not reimplemented), same
USDM drought join, PLUS the two 3-seed-confirmed candidates from the
2026-09-11 stakeholder-feedback round: pop_density (static, safe) and
fm100_5d_min (lag-2, 5-day trailing min -- NOT same-day fm100, which was
caught as leakage-equivalent to the teammate's raw-gridMET mistake; see
MODEL_UNDERSTANDING.md Training_rev4_dailymax/ Sec 11).

v1 LIMITATION (documented, not a bug): the 12Z HRRR archive only covers
a TX-centered bbox (lat 25.75-36.65, lon -106.7 to -93.4) -- ~60% of NM
(eastern plains) and ~90% of OK. Cells outside this bbox are DROPPED from
the tri-state table, not imputed. A full re-pull with a wider bbox would
fix this but is a separate, larger task.

Output: NM_OK_TexRev/data/train_tristate.parquet
        (h3_cell, date, label, state, year, + 28 features: rev4's 26 +
         pop_density + fm100_5d_min)
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / "Training_rev4_dailymax"))
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
import rev4_pipeline as R

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TF = ROOT.parent

NEG_PER_POS, SPATIAL_FRAC, SEED = 8, 0.4, 42
YEARS = list(range(2014, 2027))
HOL = {(1, 1), (7, 4), (6, 19), (11, 11), (12, 25), (10, 31)}
LAG_DAYS, WINDOW = 2, 5


def log(m): print(m, flush=True)


def sample_candidates(state, rng):
    lab_p = ROOT / "data" / f"labels_daily_{state}.parquet"
    st_p = ROOT / "data" / f"static_master_{state}.parquet"
    if not lab_p.exists():
        log(f"[{state}] {lab_p.name} missing -- run 07_build_labels_nm_ok.py first"); return None

    pos = pd.read_parquet(lab_p, columns=['h3_cell', 'date', 'label'])
    pos = pos[pos.label == 1].drop_duplicates(['h3_cell', 'date'])
    pos['date'] = pd.to_datetime(pos['date'])
    fire_cells = pos.h3_cell.unique()
    log(f"[{state}] {len(pos):,} positive cell-days, {len(fire_cells):,} fire-prone cells")

    static = pd.read_parquet(st_p, columns=['h3_cell'] + R.STATIC + ['lat', 'lon', 'fips'])
    # caught 2026-09-16 (post-run patch): TX's fillna(0) convention for FSim-derived
    # LAND_FEATS (genuine no-data at water/urban/barren cells) lives in scripts/
    # 03_build_dataset.py, never replicated here -- left cfl null for ~5-16% of NM/OK.
    for c in ['avg_burn_prob', 'whp', 'cfl', 'flep4', 'cbd', 'cbh']:
        if c in static.columns:
            static[c] = static[c].fillna(0)
    all_cells = static.h3_cell.unique()
    never_fire = np.setdiff1d(all_cells, fire_cells)

    day_pool = pd.to_datetime(pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31", freq='D'))
    pos_keys = set(zip(pos.h3_cell, pos.date))
    neg_rows = []
    for cell in fire_cells:
        picks = day_pool[rng.randint(0, len(day_pool), NEG_PER_POS * 2)]
        added = 0
        for d in picks:
            if (cell, d) not in pos_keys:
                neg_rows.append((cell, d)); added += 1
                if added >= NEG_PER_POS:
                    break
    n_spatial = int(len(pos) * SPATIAL_FRAC)
    sc = never_fire[rng.randint(0, len(never_fire), n_spatial)]
    sd = day_pool[rng.randint(0, len(day_pool), n_spatial)]
    neg_rows += list(zip(sc, sd))
    neg = pd.DataFrame(neg_rows, columns=['h3_cell', 'date']).drop_duplicates()
    neg = neg[~neg.set_index(['h3_cell', 'date']).index.isin(pos_keys)]
    neg['label'] = 0
    log(f"[{state}] {len(neg):,} negative cell-days")

    cand = pd.concat([pos, neg], ignore_index=True).drop_duplicates(['h3_cell', 'date'])
    cand['year'] = cand.date.dt.year
    cand = cand.merge(static, on='h3_cell', how='left')
    cand['state'] = state
    return cand.dropna(subset=['lat', 'lon'])


def attach_peak_weather(cand):
    """Same mechanism as rev4_pipeline.stage1_features -- cKDTree nearest to the
    HRRR archive grid. v1 LIMIT: cells outside the archive bbox get no match and
    are dropped (inner join below)."""
    sample = pd.read_parquet(sorted(R.DIRS[33].glob('*.parquet'))[0], columns=['lat', 'lon'])
    tree = cKDTree(sample[['lat', 'lon']].values)
    dist, gi = tree.query(cand[['lat', 'lon']].values)
    cand = cand.copy()
    cand['gidx'] = gi
    # HRRR grid spacing ~3km (~0.027deg); 0.05deg tolerance safely admits real matches,
    # excludes cells genuinely outside the archived bbox (the documented v1 limitation)
    in_bbox = dist < 0.05
    log(f"  peak-weather bbox coverage: {in_bbox.mean()*100:.1f}% of candidate rows "
        f"({in_bbox.sum():,}/{len(cand):,})")
    cand = cand[in_bbox].drop(columns=['gidx']).assign(gidx=gi[in_bbox])

    rows = []
    for d, grp in cand.groupby('date'):
        pk = R.read_day_peaks(d)
        if pk is None:
            continue
        idx = grp.gidx.values.astype(int)
        g = grp.copy()
        g['hrrr_tmp'] = pk['tmp_c'][idx]
        g['hrrr_vpd'] = pk['vpd_kpa'][idx]
        g['hrrr_wind'] = pk['gust_ms'][idx]
        g['hrrr_mstav'] = np.clip(pk['mstav_pct'][idx] / 100.0, 0, 1)
        rows.append(g)
    if not rows:
        return cand.iloc[0:0]
    return pd.concat(rows, ignore_index=True).drop(columns=['gidx'])


def attach_drought(cand, state):
    dwp = ROOT / "data" / f"drought_weekly_{state}.parquet"
    if not dwp.exists():
        log(f"[{state}] no drought file, skipping (run 05_build_drought_nm_ok.py)")
        cand['drought_score'] = np.nan; return cand
    dw = pd.read_parquet(dwp)
    dw['ValidStart'] = pd.to_datetime(dw.ValidStart); dw['ValidEnd'] = pd.to_datetime(dw.ValidEnd)
    rows = []
    for fips, g in cand.groupby('fips'):
        wk = dw[dw.fips == fips].sort_values('ValidStart')
        if len(wk) == 0:
            g = g.copy(); g['drought_score'] = np.nan; rows.append(g); continue
        idx = np.searchsorted(wk.ValidStart.values, g.date.values, side='right') - 1
        idx = np.clip(idx, 0, len(wk) - 1)
        g = g.copy(); g['drought_score'] = wk.drought_score.values[idx]
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def attach_pop_density(cand, state):
    p = (TF / "New_Training817_moredata" / "pop_density_feature.parquet" if state == 'tx'
         else ROOT / "data" / f"pop_density_{state}.parquet")
    pop = pd.read_parquet(p)
    return cand.merge(pop, on='h3_cell', how='left')


def attach_fm100_5d_min(cand, state):
    gdir = ROOT / "data" / "gridmet_nm_ok" if state in ('nm', 'ok') else TF / "data" / "gridmet_tx_full"
    frames = []
    for yr in sorted(cand.year.unique()):
        f = gdir / f"fm100_{yr}_{state}_cells.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        d = d[d.h3_cell.isin(cand.h3_cell.unique())]
        d = d.rename(columns={'date_utc': 'date'}); d['date'] = pd.to_datetime(d['date'])
        frames.append(d)
    if not frames:
        cand['fm100_5d_min'] = np.nan; return cand
    fm = pd.concat(frames, ignore_index=True).drop_duplicates(['h3_cell', 'date'])
    fm = fm.sort_values(['h3_cell', 'date'])
    fm['fm100_shift'] = fm.groupby('h3_cell')['fm100'].shift(LAG_DAYS)
    fm['fm100_5d_min'] = (fm.groupby('h3_cell')['fm100_shift']
                            .rolling(WINDOW, min_periods=3).min()
                            .reset_index(level=0, drop=True))
    return cand.merge(fm[['h3_cell', 'date', 'fm100_5d_min']], on=['h3_cell', 'date'], how='left')


def add_calendar(cand):
    dow = cand.date.dt.dayofweek
    cand['sin_dow'] = np.sin(2 * np.pi * dow / 7); cand['cos_dow'] = np.cos(2 * np.pi * dow / 7)
    cand['is_weekend'] = (dow >= 5).astype(int)
    mo = cand.date.dt.month
    cand['sin_month'] = np.sin(2 * np.pi * mo / 12); cand['cos_month'] = np.cos(2 * np.pi * mo / 12)
    cand['is_holiday'] = [(d.month, d.day) in HOL for d in cand.date]
    return cand


if __name__ == '__main__':
    rng = np.random.RandomState(SEED)
    parts = []
    for state in ['tx', 'nm', 'ok']:
        if state == 'tx':
            log("[tx] reusing tdis_train_daily_hrrr.parquet rows, but redoing the "
                "peak-weather merge -- the baked-in hrrr_tmp/vpd/wind are the STALE "
                "single-snapshot convention rev4_pipeline.stage2_train drops+recomputes; "
                "reusing them as-is would (a) use the wrong weather values and (b) skip "
                "the 2018-07-16 HRRR-archive floor NM/OK correctly get via attach_peak_weather "
                "-- caught 2026-09-16, see METHODOLOGY.md")
            tx = pd.read_parquet(TF / "tdis_train_daily_hrrr.parquet")
            tx = tx.drop(columns=['hrrr_tmp', 'hrrr_vpd', 'hrrr_wind'], errors='ignore')
            peaks = R.stage1_features()
            tx['date'] = pd.to_datetime(tx['date'])
            n0 = len(tx)
            tx = tx.merge(peaks, on=['h3_cell', 'date'], how='inner')
            log(f"[tx] peak-weather floor applied: {n0:,} -> {len(tx):,} "
                f"(dropped {n0-len(tx):,} pre-2018-07-16 rows, same floor as rev4)")
            tx['state'] = 'tx'
            # caught 2026-09-16 (post-run patch): rev4_pipeline.stage2_train also merges
            # drought_feature.parquet + powerline_dist_km.parquet + add_ix() for TX --
            # tdis_train_daily_hrrr.parquet has none of these baked in. Skipping them left
            # drought_score/powerline_dist_km/bp_x_vpd/whp_x_vpd/bp_x_drought 100% null for TX.
            drought = pd.read_parquet(R.DR / "drought_feature.parquet")
            tx = tx.merge(drought, on=['h3_cell', 'date'], how='left')
            pl = pd.read_parquet(R.NT / "powerline_dist_km.parquet")
            tx = tx.merge(pl, on='h3_cell', how='left')
            tx = R.add_ix(tx)
            log(f"[tx] drought_score attached: {tx['drought_score'].notna().mean()*100:.1f}% coverage, "
                f"powerline_dist_km: {tx['powerline_dist_km'].notna().mean()*100:.1f}%")
            tx = attach_pop_density(tx, 'tx')
            tx = attach_fm100_5d_min(tx, 'tx')
            parts.append(tx)
            continue
        cand = sample_candidates(state, rng)
        if cand is None:
            log(f"[{state}] SKIPPED -- prerequisite data not ready"); continue
        cand = attach_peak_weather(cand)
        if len(cand) == 0:
            log(f"[{state}] SKIPPED -- no rows survived the bbox weather join"); continue
        cand = attach_drought(cand, state)
        cand = attach_pop_density(cand, state)
        cand = attach_fm100_5d_min(cand, state)
        cand = add_calendar(cand)
        cand = R.add_ix(cand)
        parts.append(cand)
        log(f"[{state}] final rows: {len(cand):,}")

    if len(parts) < 3:
        log(f"\nINCOMPLETE: only {len(parts)}/3 states ready -- re-run once missing "
            f"prerequisites (VIIRS/flare/labels) finish. Not writing train_tristate.parquet yet.")
    else:
        tri = pd.concat(parts, ignore_index=True)
        tri.to_parquet(ROOT / "data" / "train_tristate.parquet", index=False)
        log(f"\nSaved train_tristate.parquet: {len(tri):,} rows, "
            f"by state:\n{tri.groupby('state').size().to_string()}")
