"""Clean ablation: SAME rows, SAME split, only difference = HRRR features present or not.
Isolates the true contribution of HRRR forecast weather."""
import warnings, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from pathlib import Path
warnings.filterwarnings('ignore')
TF=Path(__file__).resolve().parent.parent
def log(m):print(m,flush=True)
STATIC=['road_dist_km','ecoregion_id','elevation_m','slope_deg','aspect_deg','avg_burn_prob','whp','flep4','cfl','cbd','cbh']
TEMPORAL=['sin_month','cos_month','sin_dow','cos_dow','is_weekend','is_holiday']
HRRR=['hrrr_tmp','hrrr_vpd','hrrr_wind']
df=pd.read_parquet(TF/"tdis_train_daily_hrrr.parquet")
df['date']=pd.to_datetime(df['date']);df['year']=df['date'].dt.year
df=df.dropna(subset=STATIC).dropna(subset=['hrrr_vpd'])   # identical row set for both models
df['split']=np.where(df.year<=2021,'train',np.where(df.year==2022,'val','test'))
tr,va,te=df[df.split=='train'],df[df.split=='val'],df[df.split=='test']
y_tr,y_va,y_te=tr.label.astype(int),va.label.astype(int),te.label.astype(int)
spw=(y_tr==0).sum()/max((y_tr==1).sum(),1); base=y_te.mean()
log(f"identical rows: train {len(tr):,} test {len(te):,}  test base rate {base:.3f}")
def run(feats,name):
    m=xgb.XGBClassifier(n_estimators=600,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
        min_child_weight=20,scale_pos_weight=spw,eval_metric='aucpr',early_stopping_rounds=40,
        tree_method='hist',device='cuda',random_state=42)
    m.fit(tr[feats],y_tr,eval_set=[(va[feats],y_va)],verbose=False)
    p=m.predict_proba(te[feats])[:,1]
    a,r=average_precision_score(y_te,p),roc_auc_score(y_te,p)
    log(f"  {name:28s} AUC-PR={a:.4f}  AUROC={r:.4f}  lift={a/base:.2f}x")
    return a
log("ABLATION (same rows/split, only HRRR differs):")
a0=run(STATIC+TEMPORAL, "no-weather (static+temporal)")
a1=run(STATIC+TEMPORAL+HRRR, "+ HRRR forecast weather")
log(f"\n  TRUE HRRR contribution: {a1-a0:+.4f} AUC-PR  ({(a1-a0)/a0*100:+.1f}%)")
log("DONE")
