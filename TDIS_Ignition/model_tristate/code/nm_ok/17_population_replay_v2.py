"""
v2 (2026-09-23): scores with control_tx_only_v2 / rev5_tristate_v2 (trained on
train_tristate_v2.parquet, corrected national ecoregion_id for TX). Critically,
build_static() here ALSO overrides ecoregion_id with the corrected national
scale (NM_OK_TexRev/data/tx_ecoregion_national.parquet) -- the v2 models were
trained on national codes (21-40 for TX), so scoring MUST feed them the same
scale or predictions are silently out-of-distribution garbage. Cells with no
national ecoregion match (Mexico/Gulf-of-Mexico, ~30% of the full TX bbox --
see build_tx_ecoregion_national_full.py) are dropped, consistent with how
training dropped null-ecoregion rows.

NM_OK_TexRev — Step 17: population replay for control vs rev5. Scores
EVERY TX cell, every day 2024-01-01..2026-07-29, with both 28-feature models.
Per-ecoregion breakdown is the actual test this whole build was for: does
ecoregion 5 (Tablelands, rev4's one real weakness) improve with NM/OK data.

Output: results/population_replay_control_vs_rev5_v2.json + printed tables.
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / "Training_rev4_dailymax"))
import json, time
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from scipy.spatial import cKDTree
import rev4_pipeline as R

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TF = ROOT.parent

FEATS = R.FEATS + ['pop_density', 'fm100_5d_min']
REPLAY_START, REPLAY_END = pd.Timestamp('2024-01-01'), R.LAST
LAG_DAYS, WINDOW = 2, 5


def log(m): print(m, flush=True)


def build_static():
    flare = set(pd.read_parquet(R.NT / "flare_cells_v2.parquet")['h3_cell'])
    st = pd.read_parquet(TF / "data/static_features/tx_static_master.parquet")
    st = st[~st.h3_cell.isin(flare)]
    st = st.merge(pd.read_parquet(R.NT / "powerline_dist_km.parquet"), on='h3_cell', how='left')
    st = st.merge(pd.read_parquet(TF / "data/static_features/tx_county_fips.parquet"), on='h3_cell', how='left')
    st = st.merge(pd.read_parquet(TF / "New_Training817_moredata/pop_density_feature.parquet"),
                  on='h3_cell', how='left')
    # v2: override the legacy LOCAL 0-19 ecoregion_id with the corrected national
    # US_L3CODE the v2 models were actually trained on (see module docstring).
    nat = pd.read_parquet(ROOT / "data" / "tx_ecoregion_national.parquet")
    st = st.drop(columns=['ecoregion_id']).merge(nat, on='h3_cell', how='left')
    n0 = len(st)
    st = st.dropna(subset=['ecoregion_id'])  # non-US cells (Mexico/Gulf) -- same as training
    log(f"national ecoregion override: {n0:,} -> {len(st):,} "
        f"(dropped {n0-len(st):,} non-US cells, consistent with v2 training)")
    st = st.dropna(subset=R.STATIC)
    sample = pd.read_parquet(sorted(R.DIRS[33].glob('*.parquet'))[0], columns=['lat', 'lon'])
    _, gi = cKDTree(sample[['lat', 'lon']].values).query(st[['lat', 'lon']].values)
    st['gidx'] = gi
    log(f"static universe: {len(st):,} TX cells (flare-excluded, national ecoregion)")
    return st, flare


def build_fm100_lookup(cells):
    """WIDE-MATRIX rebuild (2026-09-17): v1 held ~1.6B long-format rows in RAM
    (plus concat/groupby copies) and was silently OOM-killed after ~85 min --
    no traceback, kernel SIGKILL. v2: one float32 matrix (n_cells x n_days,
    ~6.5GB), rolling-min computed on the day axis with a numpy sliding window,
    cached to disk so any rerun skips the whole build."""
    cache = ROOT / "results" / "fm100_5d_min_matrix.npz"
    cell_list = sorted(cells)
    cell_idx = {c: i for i, c in enumerate(cell_list)}
    all_days = pd.date_range(REPLAY_START - pd.Timedelta(days=10), REPLAY_END, freq='D')
    day_idx = {d: i for i, d in enumerate(all_days)}

    if cache.exists():
        log("fm100 matrix: cached")
        npz = np.load(cache, allow_pickle=True)
        return npz['matrix'], {str(c): i for i, c in enumerate(npz['cells'])}, \
               {pd.Timestamp(str(d)): i for i, d in enumerate(npz['days'])}

    mat = np.full((len(cell_list), len(all_days)), np.nan, dtype=np.float32)
    years = sorted(set(d.year for d in [all_days[0], all_days[-1]])) or []
    years = list(range(all_days[0].year, all_days[-1].year + 1))
    for yr in years:
        f = TF / "data" / "gridmet_tx_full" / f"fm100_{yr}_tx_cells.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        d['date'] = pd.to_datetime(d['date_utc'])
        d = d[(d.date >= all_days[0]) & (d.date <= all_days[-1])]
        d = d[d.h3_cell.isin(cell_idx)]
        ri = d.h3_cell.map(cell_idx).values
        ci = d.date.map(day_idx).values
        mat[ri, ci] = d.fm100.values.astype(np.float32)
        log(f"  fm100 {yr}: {len(d):,} rows placed")
        del d

    # lag-2 then 5-day trailing min along the day axis
    shifted = np.full_like(mat, np.nan)
    shifted[:, LAG_DAYS:] = mat[:, :-LAG_DAYS]
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full_like(mat, np.nan)
    win = sliding_window_view(shifted, WINDOW, axis=1)          # (cells, days-W+1, W)
    valid = (~np.isnan(win)).sum(axis=2) >= 3
    mins = np.nanmin(np.where(np.isnan(win), np.inf, win), axis=2)
    mins[~valid] = np.nan
    mins[np.isinf(mins)] = np.nan
    out[:, WINDOW - 1:] = mins
    np.savez_compressed(cache, matrix=out, cells=np.array(cell_list),
                         days=np.array([str(x.date()) for x in all_days]))
    log(f"fm100_5d_min matrix built+cached: {out.shape} "
        f"({np.isnan(out).mean()*100:.1f}% nan)")
    return out, cell_idx, day_idx


def score_day(model, st, weekly, fm_mat, fm_cell_rows, fm_day_idx, d):
    pk = R.read_day_peaks(d)
    if pk is None:
        return None
    day = st[['h3_cell', 'lat', 'lon', 'gidx', 'fips'] + R.STATIC + ['pop_density']].copy()
    idx = day.gidx.values.astype(int)
    day['hrrr_tmp'] = pk['tmp_c'][idx]; day['hrrr_vpd'] = pk['vpd_kpa'][idx]
    day['hrrr_wind'] = pk['gust_ms'][idx]
    day['hrrr_mstav'] = np.clip(pk['mstav_pct'][idx] / 100.0, 0, 1)
    mo, dow = d.month, d.dayofweek
    day['sin_month'] = np.sin(2*np.pi*mo/12); day['cos_month'] = np.cos(2*np.pi*mo/12)
    day['sin_dow'] = np.sin(2*np.pi*dow/7); day['cos_dow'] = np.cos(2*np.pi*dow/7)
    day['is_weekend'] = int(dow >= 5); day['is_holiday'] = int((mo, d.day) in R.HOL)
    # fm100 assigned BEFORE the drought merge: fm_cell_rows is positionally aligned
    # to st's row order, which `day` still shares here; a merge could break that.
    ci = fm_day_idx.get(d)
    day['fm100_5d_min'] = fm_mat[fm_cell_rows, ci] if ci is not None else np.nan
    day['fm100_5d_min'] = day['fm100_5d_min'].fillna(day['fm100_5d_min'].mean())
    wk = weekly[(weekly.ValidStart <= d) & (weekly.ValidEnd >= d)][['fips', 'drought_score']]
    day = day.merge(wk, on='fips', how='left')
    day['drought_score'] = day['drought_score'].fillna(day['drought_score'].mean())
    day = R.add_ix(day)
    day = day.dropna(subset=['hrrr_vpd'])
    day['p'] = model.predict_proba(day[FEATS])[:, 1]
    day['rank'] = day['p'].rank(pct=True)
    return day


def replay_model(model_path, label, st, weekly, fm_mat, fm_cell_rows, fm_day_idx, lab, eco_names):
    model = xgb.XGBClassifier(); model.load_model(str(model_path))
    fire_by_date = lab.groupby('date')['h3_cell'].apply(set).to_dict()

    rows, t0, nd = [], time.time(), 0
    for d in pd.date_range(str(REPLAY_START.date()), str(REPLAY_END.date()), freq='D'):
        fires = fire_by_date.get(d, set())
        if not fires:
            continue
        day = score_day(model, st, weekly, fm_mat, fm_cell_rows, fm_day_idx, d)
        if day is None:
            continue
        cr = dict(zip(day.h3_cell.values, day['rank'].values))
        for c in fires:
            r = cr.get(c)
            if r is not None:
                rows.append((d, c, r))
        nd += 1
        if nd % 200 == 0:
            log(f"  [{label}] {nd} fire-days replayed ({(time.time()-t0)/60:.1f} min)")

    fr = pd.DataFrame(rows, columns=['date', 'h3_cell', 'rank'])
    eco_map = dict(zip(st.h3_cell, st.ecoregion_id))
    fr['ecoregion_id'] = fr.h3_cell.map(eco_map)

    overall = dict(n=len(fr), median_rank=float(fr['rank'].median()),
                   top50=float((fr['rank'] >= 0.50).mean() * 100),
                   top25=float((fr['rank'] >= 0.75).mean() * 100),
                   bottom10=float((fr['rank'] < 0.10).mean() * 100))
    log(f"[{label}] OVERALL: n={overall['n']:,} median_rank={overall['median_rank']:.3f} "
        f"top50%={overall['top50']:.1f}% top25%={overall['top25']:.1f}% bottom10%={overall['bottom10']:.1f}%")

    eco = fr.groupby('ecoregion_id')['rank'].agg(
        n='count', median_rank='median',
        top50=lambda r: (r >= 0.50).mean() * 100,
    ).reset_index().sort_values('ecoregion_id')
    return overall, eco, fr


if __name__ == '__main__':
    st, flare = build_static()
    weekly = pd.read_parquet(TF / "data/static_features/drought_weekly.parquet")
    weekly['ValidStart'] = pd.to_datetime(weekly['ValidStart']); weekly['ValidEnd'] = pd.to_datetime(weekly['ValidEnd'])
    lab = pd.read_parquet(TF / "data/labels_fused/ignitions_daily_tx.parquet",
                           columns=['h3_cell', 'date', 'label'])
    lab['date'] = pd.to_datetime(lab['date']).dt.normalize()
    lab = lab[(~lab.h3_cell.isin(flare)) & (lab.label == 1) &
              (lab.date >= REPLAY_START) & (lab.date <= REPLAY_END)]
    log(f"replay fire-days in window: {len(lab):,}")

    fm_mat, fm_cell_idx, fm_day_idx = build_fm100_lookup(set(st.h3_cell))
    # row-index array aligned to st's cell order, so score_day can slice the matrix directly
    fm_cell_rows = st.h3_cell.map(fm_cell_idx).values.astype(int)

    results = {}
    for label, path in [('control', ROOT / "models" / "control_tx_only_v2_seed42.json"),
                         ('rev5', ROOT / "models" / "rev5_tristate_v2_seed42.json")]:
        log(f"\n=== Replaying {label} ===")
        overall, eco, fr = replay_model(path, label, st, weekly, fm_mat, fm_cell_rows, fm_day_idx, lab, None)
        results[label] = dict(overall=overall, eco=eco.to_dict('records'))
        fr.to_parquet(ROOT / "results" / f"fire_ranks_{label}_v2.parquet", index=False)

    log("\n=== CONTROL vs REV5: overall ===")
    for k in ['median_rank', 'top50', 'top25', 'bottom10']:
        c, r = results['control']['overall'][k], results['rev5']['overall'][k]
        log(f"  {k}: control={c:.3f}  rev5={r:.3f}  delta={r-c:+.3f}")

    log("\n=== CONTROL vs REV5: per-ecoregion (THE TABLELANDS TEST) ===")
    ce = {e['ecoregion_id']: e for e in results['control']['eco']}
    re = {e['ecoregion_id']: e for e in results['rev5']['eco']}
    for eid in sorted(set(ce) | set(re)):
        c, r = ce.get(eid), re.get(eid)
        cn = c['n'] if c else 0; rn = r['n'] if r else 0
        cm = c['median_rank'] if c else None; rm = r['median_rank'] if r else None
        ct = c['top50'] if c else None; rt = r['top50'] if r else None
        tag = " <<< TABLELANDS" if eid == 26 else ""  # national US_L3CODE 26 (was local code 5 pre-fix)
        log(f"  eco {eid}: control(n={cn}, median={cm}, top50={ct}) | "
            f"rev5(n={rn}, median={rm}, top50={rt}){tag}")

    with open(ROOT / "results" / "population_replay_control_vs_rev5_v2.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log("\nSaved results/population_replay_control_vs_rev5_v2.json")
