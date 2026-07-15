import pandas as pd

df = pd.read_parquet('outputs/texas/final_training_dataset_tx.parquet')
cell = df['h3_cell'].iloc[0]
print('Sample h3_cell :', cell)
print('First 4 chars  :', cell[:4])
print('Unique cells   :', df['h3_cell'].nunique())

# H3 v3: R7 cells start with '87', R8 cells start with '88'
prefix = cell[:2]
if prefix == '87':
    print('\n>>> RESOLUTION = 7 (R7) — cell starts with 87')
elif prefix == '88':
    print('\n>>> RESOLUTION = 8 (R8) — cell starts with 88')
else:
    print('\nUnknown prefix:', prefix)

try:
    import h3
    fn = h3.h3_get_resolution if hasattr(h3, 'h3_get_resolution') else h3.get_resolution
    print('h3 lib confirms resolution:', fn(cell))
except Exception as e:
    print('h3 lib error:', e)
