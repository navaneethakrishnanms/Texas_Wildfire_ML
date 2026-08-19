"""SANITY SWEEP Phase 6 — the two credibility retrains (GPU).
A. LABEL-SHUFFLE negative control: permute training labels -> skill must collapse
   (AUC-PR ~= base rate, AUROC ~= 0.5). Detects any hidden pipeline leak.
B. SPATIAL HOLDOUT: hold out 20% of CELLS entirely; evaluate test-years on
   (a) never-seen cells vs (b) seen cells -> the gap = cell-memorization component.
Uses the honest (static+temporal) config — the config where memorization risk is highest.
"""
import warnings, hashlib
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')
TF = Path(__file__).resolve().parent.parent
def log(m): print(m, flush=True)

STATIC=['road_dist_km','ecoregion_id','elevation_m','slope_deg','aspect_deg',
        'avg_burn_prob','whp','flep4','cfl','cbd','cbh']
TEMPORAL=['sin_month','cos_month','sin_dow','cos_dow','is_weekend','is_holiday']
FEATS=STATIC+TEMPORAL

df=pd.read_parquet(TF/"tdis_train_daily_tx_flarefiltered.parquet")
df=df.dropna(subset=STATIC).reset_index(drop=True)
df['split']=np.where(df.year<=2021,'train',np.where(df.year==2022,'val','test'))
tr,te=df[df.split=='train'],df[df.split=='test']
log(f"rows: train {len(tr):,} test {len(te):,}")

def train(X,y,spw):
    m=xgb.XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.8,
        colsample_bytree=0.8,min_child_weight=20,scale_pos_weight=spw,eval_metric='aucpr',
        tree_method='hist',device='cuda',n_jobs=-1,random_state=42)
    m.fit(X,y,verbose=False); return m

# ── A. label shuffle ──
log("\n[A] LABEL-SHUFFLE NEGATIVE CONTROL")
rng=np.random.default_rng(0)
ysh=rng.permutation(tr.label.values)
spw=(ysh==0).sum()/max(ysh.sum(),1)
m=train(tr[FEATS],ysh,spw)
p=m.predict_proba(te[FEATS])[:,1]; y=te.label.astype(int).values
log(f"  shuffled-label model on TRUE test: AUC-PR={average_precision_score(y,p):.4f} "
    f"(base rate {y.mean():.4f})  AUROC={roc_auc_score(y,p):.4f} (chance 0.5)")
log("  PASS if AUC-PR ~= base rate and AUROC ~= 0.5")

# ── B. spatial holdout ──
log("\n[B] SPATIAL HOLDOUT (20% of cells never seen in training)")
cells=df.h3_cell.unique()
hold=np.array([int(hashlib.md5(c.encode()).hexdigest(),16)%5==0 for c in cells])
hold_set=set(cells[hold])
log(f"  held-out cells: {hold.sum():,} / {len(cells):,}")
tr_sp=tr[~tr.h3_cell.isin(hold_set)]
spw=(tr_sp.label==0).sum()/max(tr_sp.label.sum(),1)
m=train(tr_sp[FEATS],tr_sp.label.astype(int),spw)
for tag,sub in [("UNSEEN cells",te[te.h3_cell.isin(hold_set)]),
                ("seen cells  ",te[~te.h3_cell.isin(hold_set)])]:
    p=m.predict_proba(sub[FEATS])[:,1]; y=sub.label.astype(int).values
    log(f"  test-years x {tag}: n={len(y):>9,} pos={y.mean():.4f} "
        f"AUC-PR={average_precision_score(y,p):.4f} lift={average_precision_score(y,p)/y.mean():.2f}x "
        f"AUROC={roc_auc_score(y,p):.4f}")
log("  gap between rows = cell-memorization component of the reported skill")
log("\nPHASE 6 DONE")
