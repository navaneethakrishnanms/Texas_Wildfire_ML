"""
TDIS Forecast — Step 22: retrain the CEILING model (same-day OBSERVED weather) on
flare-filtered labels, for a fair three-way honest/ceiling/operational comparison.

Same setup as 18 (400 trees, no early stopping) but with WEATHER = observed gridMET
(erc/fm100/vpd/vs/rmax/rmin/tmmx/pr), matching the original 07_train_baseline_forecast.py
'ceiling' config. This is the optimistic "if forecasts were perfect" bound — never served,
diagnostic only.
"""
import json, warnings
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')
TF = Path(__file__).resolve().parent.parent
def log(m): print(m, flush=True)

STATIC=['road_dist_km','ecoregion_id','elevation_m','slope_deg','aspect_deg',
        'avg_burn_prob','whp','flep4','cfl','cbd','cbh']
TEMPORAL=['sin_month','cos_month','sin_dow','cos_dow','is_weekend','is_holiday']
WEATHER=['erc','fm100','vpd','vs','rmax','rmin','tmmx','pr']
FEATS=STATIC+TEMPORAL+WEATHER

df=pd.read_parquet(TF/"tdis_train_daily_tx_flarefiltered.parquet")   # already flare+phantom filtered
df=df.dropna(subset=STATIC+WEATHER).reset_index(drop=True)
log(f"ceiling rows (static+weather present): {len(df):,}")
df['split']=np.where(df.year<=2021,'train',np.where(df.year==2022,'val','test'))
tr,va,te=df[df.split=='train'],df[df.split=='val'],df[df.split=='test']
spw=(tr.label==0).sum()/max((tr.label==1).sum(),1)
log(f"train {len(tr):,} / val {len(va):,} / test {len(te):,} | test pos rate {te.label.mean():.4f}")

m=xgb.XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.8,
   colsample_bytree=0.8,min_child_weight=20,scale_pos_weight=spw,eval_metric='aucpr',
   tree_method='hist',device='cuda',n_jobs=-1,random_state=42)
m.fit(tr[FEATS],tr.label.astype(int),eval_set=[(va[FEATS],va.label.astype(int))],verbose=False)

y=te.label.astype(int).values
p=m.predict_proba(te[FEATS])[:,1]
aucpr,auroc=average_precision_score(y,p),roc_auc_score(y,p)
log(f"CEILING filtered: AUC-PR={aucpr:.4f} AUROC={auroc:.4f} lift={aucpr/y.mean():.2f}x")

m.save_model(str(TF/"models/tdis_forecast_baseline_ceiling_filtered.json"))
imp=dict(sorted(zip(FEATS,[round(float(x),4) for x in m.feature_importances_]),key=lambda x:-x[1]))
json.dump({'tag':'ceiling_filtered','features':FEATS,'n_trees':400,
   'test_aucpr':round(float(aucpr),4),'test_auroc':round(float(auroc),4),
   'test_pos_rate':round(float(y.mean()),4),'lift':round(float(aucpr/y.mean()),2),
   'feature_importance':imp,
   'note':'Ceiling (same-day OBSERVED weather) retrained on flare-filtered labels. Diagnostic only, never served.'},
   open(TF/"models/tdis_forecast_baseline_ceiling_filtered_meta.json",'w'),indent=1)
log("saved tdis_forecast_baseline_ceiling_filtered.json (+meta)")
