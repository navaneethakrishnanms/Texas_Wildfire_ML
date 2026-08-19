"""SANITY SWEEP Phase 1 — database forensics. Read-only. Output -> sweep_phase1.log"""
import json, warnings
import numpy as np, pandas as pd, h3
from pathlib import Path
warnings.filterwarnings('ignore')
TF = Path(__file__).resolve().parent.parent
def sec(t): print(f"\n{'='*72}\n{t}\n{'='*72}", flush=True)
def log(m): print(m, flush=True)

# ── 1. TRAINING TABLE (flare-filtered, the one that matters) ──
sec("1. TRAINING TABLE — tdis_train_daily_tx_flarefiltered.parquet")
df = pd.read_parquet(TF/"tdis_train_daily_tx_flarefiltered.parquet")
df['date'] = pd.to_datetime(df['date'])
log(f"rows {len(df):,}  cols {len(df.columns)}  cells {df.h3_cell.nunique():,}  "
    f"dates {df.date.min().date()}..{df.date.max().date()}")
log(f"duplicate (cell,date): {df.duplicated(subset=['h3_cell','date']).sum():,}")

# physical-bounds audit (literature ranges)
BOUNDS = {'erc':(0,113,'NFDRS max ~113 (fuel model G)'), 'vpd':(0,9,'TX summer extreme ~9 kPa'),
          'vs':(0,25,'daily-mean wind; TX daily mean >25 m/s implausible'),
          'fm100':(1,35,'NFDRS 100-h fuel moisture'), 'rmax':(0,100,'%'), 'rmin':(0,100,'%'),
          'tmmx':(-20,48,'TX record ~48C; <-20C impossible'), 'pr':(0,650,'daily rain; TX record ~640mm (Alvin 1979)')}
log("\nPHYSICAL-BOUNDS VIOLATIONS (non-null values outside literature range):")
for c,(lo,hi,note) in BOUNDS.items():
    if c not in df: continue
    v = df[c].dropna()
    bad = ((v<lo)|(v>hi)).sum()
    log(f"  {c:>6}: {bad:>8,} rows ({bad/len(v)*100:5.2f}% of non-null)  [{lo},{hi}] {note}"
        + (f"  min={v.min():.1f} max={v.max():.1f}" if bad else ""))

# nulls & zero-fractions
nn = df.isna().mean(); nn = nn[nn>0.001].sort_values(ascending=False)
log("\nNULL FRACTIONS >0.1%:"); log(nn.round(4).to_string())
num = df.select_dtypes(include=[np.number])
zf = (num==0).mean().sort_values(ascending=False)
log("\nZERO FRACTIONS >30%:"); log(zf[zf>0.3].round(3).to_string())

# imbalance: pos rate + neg:pos per year and per split
df['split'] = np.where(df.year<=2021,'train',np.where(df.year==2022,'val','test'))
g = df.groupby('year').agg(rows=('label','size'), pos=('label','sum'))
g['pos_rate']=g.pos/g.rows; g['neg_per_pos']=(g.rows-g.pos)/g.pos
log("\nCLASS BALANCE BY YEAR (pos_rate drift contaminates per-year AUC-PR comparisons):")
log(g.round(3).to_string())
log("\nBY SPLIT:")
log(df.groupby('split')['label'].agg(['size','mean']).round(4).to_string())

# matched-negative audit: per-cell positive-rate distribution
pc = df.groupby('h3_cell')['label'].agg(['size','mean'])
fire_cells = pc[pc['mean']>0]
log(f"\nMATCHED-NEGATIVE AUDIT: {len(pc):,} cells; {len(fire_cells):,} with >=1 positive")
log("per-fire-cell positive-rate distribution (should cluster near design ratio):")
log(fire_cells['mean'].describe(percentiles=[.05,.25,.5,.75,.95]).round(3).to_string())
nz = (pc['mean']==0).sum()
log(f"never-fire (spatial-control) cells: {nz:,} ({nz/len(pc)*100:.1f}% of cells, "
    f"{df[df.h3_cell.isin(pc[pc['mean']==0].index)].shape[0]:,} rows)")

# weather-null pattern: is it cell-structured (gridMET universe) or random?
cellnull = df.groupby('h3_cell')['erc'].apply(lambda s: s.isna().mean())
log(f"\nWEATHER-NULL STRUCTURE: cells with 100% null erc: {(cellnull==1).sum():,} | "
    f"0% null: {(cellnull==0).sum():,} | partial: {((cellnull>0)&(cellnull<1)).sum():,}")
log("  -> fully cell-structured = gridMET-universe artifact (expected), partial = suspicious")

# date continuity of labels within table
pos_days = pd.Series(sorted(df.loc[df.label==1,'date'].unique()))
gaps = pos_days.diff().dt.days
log(f"\nLABEL DATE CONTINUITY: {len(pos_days)} distinct positive days; "
    f"gaps>3d: {(gaps>3).sum()} (max gap {gaps.max():.0f}d ending {pos_days[gaps.idxmax()].date() if (gaps>3).any() else '—'})")

# ── 2. DYNAMIC COMPONENTS CACHE ──
sec("2. COMPONENTS CACHE — data/fwi_components_res5.parquet")
comp = pd.read_parquet(TF/"data/fwi_components_res5.parquet")
comp['date']=pd.to_datetime(comp['date'])
days = pd.Series(sorted(comp.date.unique()))
dg = days.diff().dt.days
log(f"days {len(days)} ({days.iloc[0].date()}..{days.iloc[-1].date()}); calendar gaps: {(dg>1).sum()}")
if (dg>1).any():
    for i in dg[dg>1].index: log(f"  GAP: {days[i-1].date()} -> {days[i].date()}")
log("nulls per var: " + json.dumps({c: round(float(comp[c].isna().mean()),4) for c in ['erc','vpd','vs','fm100']}))
log(f"cells per day min/max: {comp.groupby('date').size().min()} / {comp.groupby('date').size().max()}")

# ── 3. STATIC MASTER: multicollinearity ──
sec("3. STATIC MASTER — collinearity of hazard inputs")
st = pd.read_parquet(TF/"data/static_features/tx_static_master.parquet",
                     columns=['h3_cell','lat','lon','whp','avg_burn_prob','cbd','cbh','elevation_m','slope_deg','road_dist_km'])
cm = st[['whp','avg_burn_prob','cbd','cbh','elevation_m','slope_deg','road_dist_km']].corr()
log("correlation matrix:"); log(cm.round(2).to_string())

# ── 4. LABELS: seasonality + spatial coverage + flare sensitivity ──
sec("4. LABELS — seasonality, coverage, flare-threshold sensitivity")
lb = pd.read_parquet(TF/"data/labels_fused/ignitions_daily_tx.parquet")
lb['date']=pd.to_datetime(lb['date'])
flare = set(pd.read_parquet(TF/"data/labels_fused/flare_cells.parquet")['h3_cell'])
lbc = lb[~lb.h3_cell.isin(flare)]
log(f"labels: raw {len(lb):,} -> flare-filtered {len(lbc):,}")
mo = lbc.groupby(lbc.date.dt.month).size()
log("monthly distribution (clean): " + ', '.join(f"{m}:{v:,}" for m,v in mo.items()))
log(f"peak month: {mo.idxmax()} (lit: TX cool-season Feb-Apr wind-driven peak + summer secondary)")
# flare threshold sensitivity
ndays=(lb.date.max()-lb.date.min()).days
per=lb.groupby('h3_cell').size()
for f in [0.01,0.03,0.10]:
    s=per[per>ndays*f]
    log(f"flare threshold >{f*100:.0f}% of days: {len(s):,} cells, {s.sum():,} rows ({s.sum()/len(lb)*100:.1f}% of positives)")
# spatial coverage: label-universe by longitude band (east/west TX)
uni = df[['h3_cell']].drop_duplicates().merge(st[['h3_cell','lon']], on='h3_cell', how='left')
full_w = (st.lon<-100).mean(); uni_w=(uni.lon<-100).mean()
log(f"\nSPATIAL COVERAGE: cells west of -100 lon: full grid {full_w*100:.0f}% vs label-universe {uni_w*100:.0f}%")

log("\nPHASE 1 DONE")
