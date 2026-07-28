import pandas as pd
import numpy as np

df = pd.read_parquet(r'V2\phase2\outputs\texas\final_training_dataset_tx.parquet')

print('=' * 65)
print('  FINAL TRAINING DATASET — Texas')
print('=' * 65)
print(f'\n  Shape: {df.shape[0]:,} rows  x  {df.shape[1]} columns')
print(f'  File:  V2/phase2/outputs/texas/final_training_dataset_tx.parquet')

print('\n=== ALL COLUMNS ===')
for i, c in enumerate(df.columns):
    dtype = str(df[c].dtype)
    n_nan = df[c].isna().sum()
    pct   = 100 * n_nan / len(df)
    note  = f'  NaN={n_nan:,} ({pct:.1f}%)' if n_nan > 0 else '  ✔ Complete'
    print(f'  {i+1:>2}. {c:<35} {dtype:<12}{note}')

print('\n=== LABEL DISTRIBUTION ===')
vc = df['label'].value_counts()
print(f'  label=1 (fire):     {vc.get(1,0):>8,}  ({100*vc.get(1,0)/len(df):.1f}%)')
print(f'  label=0 (non-fire): {vc.get(0,0):>8,}  ({100*vc.get(0,0)/len(df):.1f}%)')
print(f'  Total:              {len(df):>8,}')

print('\n=== FIRST 5 ROWS ===')
show = ['h3_cell','date_utc','window_hour','label','erc','fm100','vpd','tmmx',
        'erc_5D_mean','avg_burn_prob','centroid_lat','centroid_lon','fire_count']
show = [c for c in show if c in df.columns]
print(df[show].head().to_string(index=False))

print('\n=== WEATHER FEATURE STATS ===')
wcols = ['erc','fm100','vpd','tmmx','vs','rmax','pr','erc_5D_mean','fm100_5D_mean','tmmx_5D_mean']
wcols = [c for c in wcols if c in df.columns]
print(df[wcols].describe().round(3).to_string())

print('\n=== TRAIN / VAL / TEST SPLIT ===')
for yr_list, name in [([2014,2015,2016,2017],'TRAIN'), ([2018],'VAL'), ([2019,2020],'TEST')]:
    sub = df[df['fire_year'].isin(yr_list)]
    n1  = (sub['label']==1).sum()
    n0  = (sub['label']==0).sum()
    print(f'  {name:<6}: {len(sub):>8,} rows  fire={n1:,}  non-fire={n0:,}')

print('\n  Done.')
