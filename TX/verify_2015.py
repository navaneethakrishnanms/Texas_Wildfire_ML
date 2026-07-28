import pandas as pd
import numpy as np
from pathlib import Path

f = Path('data/hrrr/hrrr_tx_2015.parquet')
print(f'File exists: {f.exists()}')
print(f'File size:   {f.stat().st_size/1e6:.0f} MB')

df = pd.read_parquet(f)
print(f'Total rows:  {len(df):,}')
print(f'Pairs done:  {df.groupby(["date_utc","window_hour"]).ngroups:,} / 542 expected')
print(f'Date range:  {df.date_utc.min()} to {df.date_utc.max()}')
print(f'Columns:     {list(df.columns)}')
print()
print('NaN counts per column:')
for col in ['temp_pw','rh_pw','wind_pw','vpd_pw','hpbl_pw','dswrf_pw']:
    n = int(df[col].isna().sum())
    pct = 100*n/len(df)
    print(f'  {col:12s}: {n:,} NaN ({pct:.1f}%)')
print()
print('Value ranges:')
for col in ['temp_pw','wind_pw','hpbl_pw','dswrf_pw']:
    print(f'  {col:12s}: min={df[col].min():.1f}  max={df[col].max():.1f}  mean={df[col].mean():.1f}')
print()
print('VERDICT: 2015 data is GOOD' if df['temp_pw'].notna().sum() > 0 else 'VERDICT: PROBLEM')
