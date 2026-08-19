"""
TDIS Forecast — Step 19: retrain the OPERATIONAL (HRRR) model on flare-filtered labels
and refresh the dashboard ignition layer.

Follows the flare finding (VALIDATION §9 / scripts/18): removes the 531 persistent-hotspot
cells + post-label-date phantom rows, retrains the HRRR forecast model (same setup as 11),
compares old vs new on the SAME clean test rows, then:
  - saves models/tdis_forecast_hrrr_filtered.json (+meta)
  - re-scores the filtered honest model on the filtered future-test rows and rewrites the
    dashboard's per-cell ignition susceptibility ('ign' + monthly 'ignm') in place.
Run 13 (forecasts) + 17 (embed) afterwards.
"""
import json, warnings
import numpy as np, pandas as pd, h3, xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')
TF = Path(__file__).resolve().parent.parent
def log(m): print(m, flush=True)

STATIC   = ['road_dist_km','ecoregion_id','elevation_m','slope_deg','aspect_deg',
            'avg_burn_prob','whp','flep4','cfl','cbd','cbh']
TEMPORAL = ['sin_month','cos_month','sin_dow','cos_dow','is_weekend','is_holiday']
HRRR     = ['hrrr_tmp','hrrr_vpd','hrrr_wind']
FEATS = STATIC+TEMPORAL+HRRR
LAST_LABEL = pd.Timestamp('2026-07-29')

flare_set = set(pd.read_parquet(TF/"data/labels_fused/flare_cells.parquet")['h3_cell'])
log(f"flare cells: {len(flare_set)}")

# ── 1. filtered HRRR training table ──
df = pd.read_parquet(TF/"tdis_train_daily_hrrr.parquet")
df['date']=pd.to_datetime(df['date']); df['year']=df['date'].dt.year
n0=len(df)
df = df[~df.h3_cell.isin(flare_set)]
df = df[df.date <= LAST_LABEL]
df = df.dropna(subset=STATIC).dropna(subset=['hrrr_vpd'])
log(f"HRRR rows: {n0:,} -> {len(df):,} after flare+phantom+coverage filters")
df['split']=np.where(df.year<=2021,'train',np.where(df.year==2022,'val','test'))
tr,va,te = df[df.split=='train'],df[df.split=='val'],df[df.split=='test']
log(f"train {len(tr):,} / val {len(va):,} / test {len(te):,} (test pos rate {te.label.mean():.4f})")

# ── 2. train ──
spw=(tr.label==0).sum()/max((tr.label==1).sum(),1)
# fixed 400-tree budget (no early stopping) — see note in 18: 2022 val spike is spurious
m=xgb.XGBClassifier(n_estimators=400,max_depth=6,learning_rate=0.05,subsample=0.8,
   colsample_bytree=0.8,min_child_weight=20,scale_pos_weight=spw,eval_metric='aucpr',
   tree_method='hist',device='cuda',n_jobs=-1,random_state=42)
log("Training on GPU...")
m.fit(tr[FEATS],tr.label.astype(int),eval_set=[(va[FEATS],va.label.astype(int))],verbose=False)

# ── 3. old vs new on the SAME clean rows ──
y=te.label.astype(int).values
old=xgb.XGBClassifier(); old.load_model(str(TF/"models/tdis_forecast_hrrr.json"))
res={}
for tag,mm in [('OLD hrrr (flare-trained)',old),('NEW hrrr (flare-filtered)',m)]:
    p=mm.predict_proba(te[mm.get_booster().feature_names])[:,1]
    ap,roc=average_precision_score(y,p),roc_auc_score(y,p)
    res[tag]=(round(float(ap),4),round(float(roc),4))
    log(f"{tag:>26}: AUC-PR={ap:.4f} AUROC={roc:.4f} lift={ap/y.mean():.2f}x [same clean rows]")

m.save_model(str(TF/"models/tdis_forecast_hrrr_filtered.json"))
imp=dict(sorted(zip(FEATS,[round(float(x),4) for x in m.feature_importances_]),key=lambda x:-x[1]))
json.dump({'model':'hrrr_forecast_filtered','features':FEATS,
   'filters':{'flare_cells':len(flare_set),'last_label':str(LAST_LABEL.date())},
   'split':{'train':'2018-07..2021','val':'2022','test':'2023-2026 clipped'},
   'test_pos_rate':round(float(y.mean()),4),
   'clean_test_comparison':{k:{'aucpr':v[0],'auroc':v[1]} for k,v in res.items()},
   'hrrr_importance':round(float(sum(imp[f] for f in HRRR)),4),
   'feature_importance':imp,
   'note':'Operational HRRR model retrained without flare cells + phantom dates. Use for live forecasts (13).'},
   open(TF/"models/tdis_forecast_hrrr_filtered_meta.json",'w'),indent=1)
log("saved tdis_forecast_hrrr_filtered.json (+meta)")

# ── 4. refresh dashboard ignition layer from the FILTERED HONEST model ──
log("Re-scoring filtered honest model for the dashboard ignition layer...")
hb=xgb.XGBClassifier(); hb.load_model(str(TF/"models/tdis_forecast_baseline_honest_filtered.json"))
bt=pd.read_parquet(TF/"tdis_train_daily_tx_flarefiltered.parquet")
bt['date']=pd.to_datetime(bt['date'])
bt=bt.dropna(subset=STATIC)
bt=bt[bt.year>=2023]                       # future test period
HF=hb.get_booster().feature_names
bt['p']=hb.predict_proba(bt[HF])[:,1]
bt['h3_5']=[h3.cell_to_parent(c,5) for c in bt['h3_cell'].values]
bt['month']=bt.date.dt.month
ign_overall=bt.groupby('h3_5')['p'].mean()
ign_month=bt.groupby(['h3_5','month'])['p'].mean().unstack().reindex(columns=range(1,13)).round(4)
pred_out=bt[['h3_cell','date','year','label','p']]
pred_out.to_parquet(TF/"models/baseline_forecast_predictions_test_filtered.parquet",index=False)

dash=json.load(open(TF/"dashboard/tdis_dashboard_data.json"))
n_upd=n_drop=0
for c in dash['cells']:
    cid=c['id']
    if cid in ign_overall.index:
        c['ign']=round(float(ign_overall[cid]),4)
        mrow=ign_month.loc[cid] if cid in ign_month.index else None
        c['ignm']=[None if (mrow is None or pd.isna(mrow[mo])) else float(mrow[mo]) for mo in range(1,13)]
        n_upd+=1
    elif 'ign' in c:
        del c['ign']; c.pop('ignm',None); n_drop+=1
dash['meta']['ignition_source']='honest FILTERED forecast model (flare cells removed), 2023-2026 clean future test, monthly mean'
dash['meta']['n_ignition_cells']=int(n_upd)
json.dump(dash,open(TF/"dashboard/tdis_dashboard_data.json",'w'),separators=(',',':'))
log(f"dashboard ignition layer refreshed: {n_upd:,} cells updated, {n_drop} flare-only cells dropped")
log("DONE — now run 13 (x3) + 17 to refresh live forecasts & standalone")
