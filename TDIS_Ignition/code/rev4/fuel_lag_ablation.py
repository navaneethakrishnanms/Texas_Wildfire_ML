"""
CORRECTED fuel-moisture test: fm100_5d_min, a forecast-safe trailing feature.

Caught 2026-09-11: the earlier fm100 ablation joined fm100 on the training
row's OWN target date -- but fm100 is gridMET (observational, not a forecast),
and the archive itself confirms ~2-day publication lag (2026 file ends
2026-09-13, two days behind actual). Same-day fm100 is NOT knowable at real
forecast time -- this is the exact ceiling-model leakage flagged in the
teammate's raw-gridMET dataset, just self-inflicted. Do not promote same-day
fm100. This script tests the corrected replacement: fm100_5d_min, the minimum
fm100 over a 5-day window ending 2 days before the target date (days D-6..D-2)
-- guaranteed available at forecast issuance time, same archive, no new pull.

pop_density has no such issue (static per cell, no date dependency) and
already passed 3-seed confirmation -- included here as the known-good
promotable feature, tested alongside the corrected fm100 candidate.

3 seeds [42,7,2026], TX only (tri-state training table for NM/OK does not
exist yet -- scripts 03-05 in NM_OK_TexRev pending).
"""
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
import rev4_pipeline as R
from fuel_pop_ablation import build_base, load_fm100, run_variant, NT, GRIDMET

LAG_DAYS = 2      # min days between gridMET publication and forecast issuance (confirmed >= 2)
WINDOW = 5        # trailing window length


def build_fm100_5d_min(cells, years):
    R.log("Loading raw fm100 archive for rolling-window computation...")
    fm = load_fm100(cells, years)
    fm = fm.sort_values(['h3_cell', 'date']).reset_index(drop=True)
    R.log(f"  {len(fm):,} rows, {fm.h3_cell.nunique():,} cells -- computing lag-{LAG_DAYS} "
          f"{WINDOW}-day trailing min per cell...")
    fm['fm100_shift'] = fm.groupby('h3_cell')['fm100'].shift(LAG_DAYS)
    fm['fm100_5d_min'] = (fm.groupby('h3_cell')['fm100_shift']
                            .rolling(WINDOW, min_periods=3).min()
                            .reset_index(level=0, drop=True))
    out = fm[['h3_cell', 'date', 'fm100_5d_min']].dropna(subset=['fm100_5d_min'])
    R.log(f"  done: {len(out):,} rows with a valid {WINDOW}-day lag-{LAG_DAYS} window")
    return out


if __name__ == '__main__':
    R.log("Building base training table (reuses cached peak_weather_features)...")
    base = build_base()
    cells = set(base.h3_cell)
    years = sorted(base.year.unique())

    fm5 = build_fm100_5d_min(cells, years)

    R.log("Loading pop_density_feature.parquet...")
    pop = pd.read_parquet(NT / "pop_density_feature.parquet")

    df_fm5 = base.merge(fm5, on=['h3_cell', 'date'], how='left')
    df_both = df_fm5.merge(pop, on='h3_cell', how='left')
    df_pop = base.merge(pop, on='h3_cell', how='left')

    fair = df_fm5[df_fm5['fm100_5d_min'].notna()].copy()
    R.log(f"\nfm100_5d_min coverage: {len(fair):,} rows "
          f"({len(fair)/len(base)*100:.1f}% of full base)")

    results = []
    results.append(run_variant(base, [], [], "baseline (current rev4 FEATS)"))
    results.append(run_variant(df_pop, ['pop_density'], [], "+ pop_density (WorldPop, unconstrained)"))
    results.append(run_variant(fair, [], [], "baseline on fm100_5d_min-covered subset"))
    results.append(run_variant(fair, ['fm100_5d_min'], ['fm100_5d_min'],
                                "+ fm100_5d_min (LAG-SAFE, replaces same-day fm100)"))
    results.append(run_variant(df_both[df_both['fm100_5d_min'].notna()].copy(),
                                ['fm100_5d_min', 'pop_density'], ['fm100_5d_min'],
                                "+ fm100_5d_min + pop_density (combined, lag-safe)"))

    import json
    with open(Path(__file__).parent / "fuel_lag_ablation_results_3seed.json", 'w') as f:
        json.dump(results, f, indent=2)
    R.log("\nSaved fuel_lag_ablation_results_3seed.json")
