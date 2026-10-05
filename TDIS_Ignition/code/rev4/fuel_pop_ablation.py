"""
Fuel-moisture + population-density ablation, on top of rev4's exact config/split.

Stakeholder feedback (2026-09-11): add fuel moisture, revisit population density.
This isolates each addition (and both together) vs the served rev4 baseline,
same train/test rows, same hyperparams. Two single-seed (42) screening passes
already ran (partial fm100 coverage, then full coverage) and both showed a
real, strengthening signal (rank 3/27 importance both times) — promising
enough to now run the full 3-seed [42,7,2026] promotion-candidate check this
project requires before anything gets promoted (every past promotion in
METHODOLOGY.md cites "stable across 3 seeds").

fm100 source: TDIS_Forecast/data/gridmet_tx_full/fm100_{year}_tx_cells.parquet
(h3_cell, date_utc, fm100) -- FULL-COVERAGE rebuild (2026-09-11) against all
1,708,940 static-master cells, via scripts/30_build_fm100_tx_full.py. Replaces
the legacy alphaearth_nds/gridmet_tx/ archive, which only covered 23.2% of
the training-cell universe (see fuel_pop_ablation_results.json's first run
for that partial-coverage fair-comparison result: +0.0013 AUC-PR, rank 3/27).
Dead-fuel dryness: LOW fm100 = HIGH risk (monotone DOWN), opposite sign
convention from hrrr_mstav.

pop_density source: New_Training817_moredata/pop_density_feature.parquet
(h3_cell, pop_density) -- WorldPop 2020 1km, already tested standalone (no real
solo signal: 3.3% importance, +0.0067 AUC-PR). Re-testing here alongside fm100
and on rev4's current (TX-proper, flare-v2-filtered) row set.
"""
import glob
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
import rev4_pipeline as R

NT = R.NT
GRIDMET = R.TF / "data" / "gridmet_tx_full"  # full-coverage rebuild, replaces legacy alphaearth_nds/gridmet_tx


def load_fm100(cells, years):
    frames = []
    for yr in years:
        f = GRIDMET / f"fm100_{yr}_tx_cells.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        d = d[d.h3_cell.isin(cells)]
        d = d.rename(columns={'date_utc': 'date'})
        d['date'] = pd.to_datetime(d['date'])
        frames.append(d)
    return pd.concat(frames, ignore_index=True).drop_duplicates(['h3_cell', 'date'])


def build_base():
    peaks = R.stage1_features()
    flare = set(pd.read_parquet(NT / "flare_cells_v2.parquet")['h3_cell'])
    df = pd.read_parquet(R.TF / "tdis_train_daily_hrrr.parquet")
    df = df.drop(columns=['hrrr_tmp', 'hrrr_vpd', 'hrrr_wind'], errors='ignore')
    # drop stale gridMET-era fm100 (partial 2014-2020 coverage only) -- we join the
    # full 2014-2026 archive fresh below so train and test years use one consistent source
    df = df.drop(columns=['fm100', 'erc', 'vs', 'rmax', 'rmin', 'tmmx', 'pr'], errors='ignore')
    df['date'] = pd.to_datetime(df['date']); df['year'] = df.date.dt.year
    df = df[(~df.h3_cell.isin(flare)) & (df.date <= R.LAST)]
    df = df.merge(pd.read_parquet(NT / "powerline_dist_km.parquet"), on='h3_cell', how='left')
    df = df.merge(pd.read_parquet(R.DR / "drought_feature.parquet"), on=['h3_cell', 'date'], how='left')
    df = df.merge(peaks, on=['h3_cell', 'date'], how='inner')
    df = df.dropna(subset=R.STATIC).dropna(subset=['hrrr_vpd', 'hrrr_mstav', 'drought_score'])
    df = R.add_ix(df)
    return df


def run_variant(df, extra_feats, mono_down, label, seeds=(42, 7, 2026)):
    d = df.dropna(subset=extra_feats) if extra_feats else df
    feats = R.FEATS + extra_feats
    tr, te = d[d.year <= 2021], d[d.year >= 2023]
    spw = (tr.label == 0).sum() / max((tr.label == 1).sum(), 1)
    mono_map = {f: 1 for f in R.MONO_UP}
    for f in mono_down:
        mono_map[f] = -1
    mono = '(' + ','.join(str(mono_map.get(f, 0)) for f in feats) + ')'
    yt = te.label.astype(int).values

    seed_results = []
    print(f"\n== {label} == n_train={len(tr):,} n_test={len(te):,}")
    for seed in seeds:
        m = xgb.XGBClassifier(n_estimators=1000, max_depth=9, min_child_weight=30,
                               learning_rate=0.02, subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=spw, eval_metric='aucpr',
                               monotone_constraints=mono,
                               tree_method='hist', device='cuda', n_jobs=-1, random_state=seed)
        m.fit(tr[feats], tr.label.astype(int), verbose=False)
        p = m.predict_proba(te[feats])[:, 1]
        aucpr = float(average_precision_score(yt, p))
        auroc = float(roc_auc_score(yt, p))
        imp = dict(zip(feats, m.feature_importances_.round(4).tolist()))
        rank = {f: sorted(imp.values(), reverse=True).index(imp[f]) + 1 for f in extra_feats}
        print(f"  seed {seed}: AUC-PR={aucpr:.4f}  AUROC={auroc:.4f}"
              + ''.join(f"  {f} imp={imp[f]:.4f} rank={rank[f]}/{len(feats)}" for f in extra_feats))
        seed_results.append(dict(seed=seed, aucpr=aucpr, auroc=auroc, importances=imp, ranks=rank))

    aucprs = [r['aucpr'] for r in seed_results]
    aurocs = [r['auroc'] for r in seed_results]
    mean_aucpr, std_aucpr = float(np.mean(aucprs)), float(np.std(aucprs))
    mean_auroc = float(np.mean(aurocs))
    print(f"  MEAN across {len(seeds)} seeds: AUC-PR={mean_aucpr:.4f} (std {std_aucpr:.4f})  "
          f"AUROC={mean_auroc:.4f}")
    for f in extra_feats:
        mean_imp = float(np.mean([r['importances'][f] for r in seed_results]))
        mean_rank = float(np.mean([r['ranks'][f] for r in seed_results]))
        print(f"  {f} MEAN importance={mean_imp:.4f}  MEAN rank={mean_rank:.1f}/{len(feats)}")
    return dict(label=label, n_train=len(tr), n_test=len(te), seeds=seed_results,
                mean_aucpr=mean_aucpr, std_aucpr=std_aucpr, mean_auroc=mean_auroc)


if __name__ == '__main__':
    R.log("Building base training table (reuses cached peak_weather_features)...")
    base = build_base()
    cells = set(base.h3_cell)
    years = sorted(base.year.unique())

    R.log("Loading fm100 archive (filtered to training cells/years)...")
    fm = load_fm100(cells, years)
    R.log(f"  fm100 coverage: {len(fm):,} rows, {fm.h3_cell.nunique():,} cells")

    R.log("Loading pop_density_feature.parquet...")
    pop = pd.read_parquet(NT / "pop_density_feature.parquet")

    df_fm = base.merge(fm, on=['h3_cell', 'date'], how='left')
    df_both = df_fm.merge(pop, on='h3_cell', how='left')
    df_pop = base.merge(pop, on='h3_cell', how='left')

    results = []
    results.append(run_variant(base, [], [], "baseline (current rev4 FEATS)"))
    results.append(run_variant(df_pop, ['pop_density'], [], "+ pop_density (WorldPop, unconstrained)"))

    # with the full-coverage rebuild this should be ~100% of base -- keeping the
    # same fair-subset scaffolding as a coverage sanity check, not because it's
    # still needed to avoid a dataset-size confound.
    fair = df_fm[df_fm['fm100'].notna()].copy()
    R.log(f"\nfm100-covered subset: {len(fair):,} rows "
          f"({len(fair)/len(base)*100:.1f}% of full base -- should be ~100% now)")
    results.append(run_variant(fair, [], [], "baseline on fm100-covered subset (should == full baseline now)"))
    results.append(run_variant(fair, ['fm100'], ['fm100'], "+ fm100, FULL TX COVERAGE"))

    results.append(run_variant(df_both, ['fm100', 'pop_density'], ['fm100'], "+ fm100 + pop_density (combined, full coverage)"))

    import json
    with open(Path(__file__).parent / "fuel_pop_ablation_results_3seed.json", 'w') as f:
        json.dump(results, f, indent=2)
    R.log("\nSaved fuel_pop_ablation_results_3seed.json")
