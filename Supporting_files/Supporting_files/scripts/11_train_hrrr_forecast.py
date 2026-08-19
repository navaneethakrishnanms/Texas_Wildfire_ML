"""
TDIS Forecast — Step 11: train the OPERATIONAL forecast model with HRRR forecast weather.

Uses tdis_train_daily_hrrr.parquet (step 06). Features = static + temporal + HRRR FORECAST
weather (hrrr_tmp/hrrr_vpd/hrrr_wind). Because HRRR F24 exists only from 2018-07-16, we
restrict to HRRR-covered rows so every row has real forecast weather (apples-to-apples).

Split (temporal): train 2018-07-16..2021 | val 2022 | test 2023-2026 (same test years as
the baseline, so AUC-PR is directly comparable to the honest 0.6296 baseline).

GPU: device='cuda' (RTX A6000). Metric: AUC-PR (report lift).
Output: models/tdis_forecast_hrrr.json + _meta.json
"""
import json, warnings
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')
TF = Path(__file__).resolve().parent.parent
MODELS = TF/"models"
def log(m): print(m, flush=True)

STATIC   = ['road_dist_km','ecoregion_id','elevation_m','slope_deg','aspect_deg',
            'avg_burn_prob','whp','flep4','cfl','cbd','cbh']
TEMPORAL = ['sin_month','cos_month','sin_dow','cos_dow','is_weekend','is_holiday']
HRRR     = ['hrrr_tmp','hrrr_vpd','hrrr_wind']
FEATS = STATIC+TEMPORAL+HRRR

log("Loading HRRR-attached training table...")
df = pd.read_parquet(TF/"tdis_train_daily_hrrr.parquet")
df['date']=pd.to_datetime(df['date']); df['year']=df['date'].dt.year
df = df.dropna(subset=STATIC)
# HRRR-covered rows only (real forecast weather present)
before=len(df); df=df.dropna(subset=['hrrr_vpd']);
log(f"  {len(df):,} HRRR-covered rows (of {before:,}); {df.date.min().date()}..{df.date.max().date()}")

df['split']=np.where(df.year<=2021,'train',np.where(df.year==2022,'val','test'))
tr,va,te = df[df.split=='train'],df[df.split=='val'],df[df.split=='test']
log(f"  train {len(tr):,} / val {len(va):,} / test {len(te):,}  (test pos rate {te.label.mean():.3f})")

spw=(tr.label==0).sum()/max((tr.label==1).sum(),1)
m=xgb.XGBClassifier(n_estimators=600,max_depth=6,learning_rate=0.05,subsample=0.8,
   colsample_bytree=0.8,min_child_weight=20,scale_pos_weight=spw,eval_metric='aucpr',
   early_stopping_rounds=40,tree_method='hist',device='cuda',n_jobs=-1,random_state=42)
log("Training on GPU (cuda)...")
m.fit(tr[FEATS],tr.label.astype(int),eval_set=[(va[FEATS],va.label.astype(int))],verbose=False)

p=m.predict_proba(te[FEATS])[:,1]; y=te.label.astype(int)
aupr,auroc,base=average_precision_score(y,p),roc_auc_score(y,p),y.mean()
log(f"\nTEST 2023-2026 (HRRR forecast model): AUC-PR={aupr:.4f}  AUROC={auroc:.4f}  lift={aupr/base:.2f}x")
log(f"  vs honest baseline (no weather): AUC-PR=0.6296")
log(f"  vs ceiling (perfect observed wx): AUC-PR=0.6598")
log(f"  delta vs honest: {aupr-0.6296:+.4f}")
for yr,g in te.groupby('year'):
    gp=m.predict_proba(g[FEATS])[:,1]
    log(f"    {yr}: AUC-PR={average_precision_score(g.label,gp):.4f} AUROC={roc_auc_score(g.label,gp):.4f} n={len(g):,}")

imp=pd.Series(m.feature_importances_,index=FEATS).sort_values(ascending=False)
log("  top features: "+', '.join(f'{k}={v:.3f}' for k,v in imp.head(8).items()))
hrrr_imp=imp[HRRR].sum()
log(f"  HRRR forecast features combined importance: {hrrr_imp:.3f}")

m.get_booster().save_model(str(MODELS/"tdis_forecast_hrrr.json"))
json.dump({'model':'hrrr_forecast','features':FEATS,'n_features':len(FEATS),
   'split':{'train':'2018-07..2021','val':'2022','test':'2023-2026'},
   'test_aucpr':round(float(aupr),4),'test_auroc':round(float(auroc),4),
   'lift':round(float(aupr/base),2),'delta_vs_honest':round(float(aupr-0.6296),4),
   'hrrr_importance':round(float(hrrr_imp),4),
   'feature_importance':{k:round(float(v),4) for k,v in imp.items()},
   'note':'Operational forecast model: static+temporal+HRRR FORECAST weather. Trained on GPU.'},
   open(MODELS/"tdis_forecast_hrrr_meta.json",'w'),indent=2)
log("\nSaved models/tdis_forecast_hrrr.json + meta. DONE")
