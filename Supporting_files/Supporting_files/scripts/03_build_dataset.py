"""
TDIS Forecast — Step 3: assemble the DAILY training dataset (builds on Texas v5).

- Positives: fused FPA-FOD + VIIRS daily ignitions (Step 2)
- Negatives: MATCHED sampling — same fire-prone cells on non-fire days (temporal controls)
             + a spatial-control sample of never-fire cell-days (keeps spatial range)
- Features (reuses v5's set): static geo + landscape + daily gridMET weather + temporal
- Target: label=1 if any ignition in (cell, DAY)  [6-h window collapsible via WINDOW_HOURS knob]
- Split: train <=2020 | val 2021 | test 2022-2024  (post-2020 = REAL VIIRS labels)

Output: TDIS_Forecast/tdis_train_daily_tx.parquet
All knobs at top — easy to retune (horizon, neg ratio, window, split years).
"""
import numpy as np, pandas as pd
from pathlib import Path
TF  = Path(__file__).resolve().parent.parent
PAR = Path("/net/cven-mosta-nas.engr.tamu.edu/volume2/NASdata2/miguel_shared/alphaearth_nds")  # external dep, see REPRODUCE.md
GRIDMET = PAR/"gridmet_tx"
def log(m): print(m, flush=True)

# ── KNOBS ──
NEG_PER_POS   = 8        # temporal controls per positive (same cell, other days)
SPATIAL_FRAC  = 0.4      # extra never-fire cell-days as fraction of n_pos
YEARS         = list(range(2014,2027))   # through 2026 (forecast-oriented)
TRAIN_MAX, VAL_YR = 2020, 2021         # temporal-split intent; adaptive fallback below
WEATHER_VARS  = ['erc','fm100','vpd','vs','rmax','rmin','tmmx','pr']
GEO_FEATS     = ['ecoregion_id','elevation_m','slope_deg','aspect_deg','road_dist_km']
LAND_FEATS    = ['avg_burn_prob','whp','cfl','flep4','cbd','cbh']
SEED = 42
rng = np.random.RandomState(SEED)

# ── positives ──
log("Loading fused ignition labels...")
pos = pd.read_parquet(TF/"data"/"labels_fused"/"ignitions_daily_tx.parquet")
pos['date']=pd.to_datetime(pos['date']).dt.normalize()
pos = pos[(pos.date.dt.year>=YEARS[0])&(pos.date.dt.year<=YEARS[-1])]
fire_cells = pos['h3_cell'].unique()
log(f"  {len(pos):,} positive cell-days across {len(fire_cells):,} fire-prone cells")

# ── static features ──
# MASTER file (full 1.7M-cell TX coverage, built by 04_build_full_tx_static.py) --
# replaces the old 317K-cell tx_geo_features.parquet + Excel-landscape merge, which
# left 55% of VIIRS-detected fire cells with no static features. Falls back to the
# old files only if the master hasn't been built yet (keeps this script runnable
# mid-pipeline, but log a loud warning so the gap isn't silently reintroduced).
MASTER_STATIC = TF/"data"/"static_features"/"tx_static_master.parquet"
log("Loading static features...")
if MASTER_STATIC.exists():
    static = pd.read_parquet(MASTER_STATIC, columns=['h3_cell']+GEO_FEATS+LAND_FEATS)
    log(f"  using FULL-COVERAGE master: {MASTER_STATIC.name}")
else:
    log("  WARNING: tx_static_master.parquet not found -- falling back to the OLD "
        "317K-cell file. This WILL reintroduce the coverage gap (55% of VIIRS fires "
        "missing static features). Run 04_build_full_tx_static.py first if possible.")
    geo = pd.read_parquet(PAR/"tx_geo_features.parquet", columns=['h3_cell']+GEO_FEATS)
    land_cache = TF/"data"/"static_features"/"landscape_by_cell.parquet"
    if land_cache.exists():
        land = pd.read_parquet(land_cache)
    else:
        land = pd.read_excel(PAR/"final_training_dataset_tx_22.07.2026_landfire.xlsx",
                             usecols=['h3_cell']+LAND_FEATS).drop_duplicates('h3_cell')
        for c in LAND_FEATS: land[c]=land[c].fillna(0)
        land.to_parquet(land_cache, index=False)
    static = geo.merge(land, on='h3_cell', how='outer')
for c in LAND_FEATS:
    if c in static.columns: static[c] = static[c].fillna(0)
all_cells = static['h3_cell'].unique()
never_fire = np.setdiff1d(all_cells, fire_cells)
log(f"  static ready · {len(all_cells):,} cells · {len(never_fire):,} never-fire")

# ── date pool per fire cell (for temporal negatives) ──
log("Sampling matched negatives...")
day_pool = pd.to_datetime(pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31", freq='D'))
pos_keys = set(zip(pos['h3_cell'], pos['date']))

neg_rows=[]
# temporal: same fire cells, random non-fire days
for cell in fire_cells:
    picks = day_pool[rng.randint(0, len(day_pool), NEG_PER_POS*2)]
    added=0
    for d in picks:
        if (cell,d) not in pos_keys:
            neg_rows.append((cell,d)); added+=1
            if added>=NEG_PER_POS: break
# spatial: never-fire cells on random days
n_spatial=int(len(pos)*SPATIAL_FRAC)
sc = never_fire[rng.randint(0,len(never_fire),n_spatial)]
sd = day_pool[rng.randint(0,len(day_pool),n_spatial)]
neg_rows += list(zip(sc,sd))
neg = pd.DataFrame(neg_rows, columns=['h3_cell','date']).drop_duplicates()
neg = neg[~neg.set_index(['h3_cell','date']).index.isin(pos_keys)]
neg['label']=0
log(f"  {len(neg):,} negative cell-days")

# ── combine ──
cand = pd.concat([pos[['h3_cell','date','label']], neg[['h3_cell','date','label']]], ignore_index=True)
cand = cand.drop_duplicates(['h3_cell','date'])
cand['year']=cand.date.dt.year
cand['date_only']=cand['date'].dt.date
log(f"  candidate rows: {len(cand):,}  (pos {int(cand.label.sum()):,} / neg {int((cand.label==0).sum()):,})")

# ── attach daily gridMET (filter to candidate cells first for speed) ──
log("Attaching gridMET daily weather...")
cand_cells = set(cand['h3_cell'].unique())
wparts=[]
for yr in YEARS:
    ck = cand.loc[cand.year==yr, ['h3_cell','date_only']].drop_duplicates()
    if len(ck)==0: continue
    base=ck.copy()
    for var in WEATHER_VARS:
        fp = GRIDMET/f"{var}_{yr}_tx_cells.parquet"
        if not fp.exists():
            base[var]=np.nan; continue
        dv=pd.read_parquet(fp)
        dv=dv[dv['h3_cell'].isin(cand_cells)]
        dv['date_only']=pd.to_datetime(dv['date_utc']).dt.date
        dv=dv[['h3_cell','date_only',var]]
        base=base.merge(dv, on=['h3_cell','date_only'], how='left')
    wparts.append(base)
    log(f"  {yr}: {len(base):,} cell-days weathered")
weather=pd.concat(wparts, ignore_index=True)
cand=cand.merge(weather, on=['h3_cell','date_only'], how='left')

# ── static + temporal features ──
log("Attaching static + temporal features...")
cand=cand.merge(static, on='h3_cell', how='left')
dow=cand.date.dt.dayofweek
cand['sin_dow']=np.sin(2*np.pi*dow/7); cand['cos_dow']=np.cos(2*np.pi*dow/7)
cand['is_weekend']=(dow>=5).astype(int)
mo=cand.date.dt.month
cand['sin_month']=np.sin(2*np.pi*mo/12); cand['cos_month']=np.cos(2*np.pi*mo/12)
HOLIDAYS={(1,1),(7,4),(6,19),(11,11),(12,25),(10,31)}  # fixed-date US holidays (v1)
cand['is_holiday']=[(d.month,d.day) in HOLIDAYS for d in cand.date]
# SANITY_CHECK F6 fix: gridMET tmmx carries a fill-value artifact (~-53C, impossible in
# TX; ~3% of non-null rows). Null anything below the physical floor.
if 'tmmx' in cand.columns:
    bad=(cand['tmmx']<-30).sum()
    cand.loc[cand['tmmx']<-30,'tmmx']=np.nan
    log(f"  tmmx fill-value artifact nulled: {bad:,} rows")
cand['is_holiday']=cand['is_holiday'].astype(int)

# ── split — temporal if viable, else spatial fallback (for the STARTER phase) ──
def choose_split(cand):
    temporal=np.where(cand.year<=TRAIN_MAX,'train',np.where(cand.year==VAL_YR,'val','test'))
    t=cand.assign(_s=temporal)
    val_pos =int(t[(t._s=='val') &(t.label==1)].shape[0])
    test_pos=int(t[(t._s=='test')&(t.label==1)].shape[0])
    if val_pos>=100 and test_pos>=100:
        return temporal, f"temporal (train<=2020 / val 2021 / test 2022+) [val+{val_pos},test+{test_pos}]"
    # fallback: stable spatial split by cell hash (70/15/15) so a model can train now
    h=cand['h3_cell'].apply(lambda c:(hash(c)%100))
    sp=np.where(h<70,'train',np.where(h<85,'val','test'))
    return sp, f"SPATIAL cell-hash 70/15/15 (temporal not yet viable: val+{val_pos},test+{test_pos}) — will switch to temporal once 2021-2024 VIIRS is in"
cand['split'], split_desc = choose_split(cand)
log(f"  split mode: {split_desc}")

cand=cand.drop(columns=['date_only'])
outp=TF/"tdis_train_daily_tx.parquet"
cand.to_parquet(outp, index=False)
log(f"\nSaved {outp.name}: {len(cand):,} rows, {len(cand.columns)} cols")
log("Split × label:\n"+cand.groupby(['split','label']).size().rename('n').reset_index().to_string(index=False))
log(f"Positive rate by split: "+str(cand.groupby('split')['label'].mean().round(4).to_dict()))
log("STEP 3 complete. Dataset ready to train a daily forecast model.")
