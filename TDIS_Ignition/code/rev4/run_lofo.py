"""LOFO group ablation on the final rev4 feature set. Same split/config;
drop one feature GROUP at a time; matched test AUC-PR/AUROC (seed 42)."""
import sys; sys.path.insert(0,'.')
import json, pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from rev4_pipeline import *

GROUPS = {
 'hazard_fuels': ['avg_burn_prob','whp','flep4','cfl','cbd','cbh'],
 'terrain_region': ['elevation_m','slope_deg','aspect_deg','ecoregion_id'],
 'human_infra': ['road_dist_km','powerline_dist_km'],
 'calendar': ['sin_month','cos_month','sin_dow','cos_dow','is_weekend','is_holiday'],
 'weather_peaks': ['hrrr_tmp','hrrr_vpd','hrrr_wind','hrrr_mstav'],
 'drought': ['drought_score'],
 'interactions': ['bp_x_vpd','whp_x_vpd','bp_x_drought'],
}
peaks = stage1_features()
flare = set(pd.read_parquet(NT/"flare_cells_v2.parquet")['h3_cell'])
df = pd.read_parquet(TF/"tdis_train_daily_hrrr.parquet")
df = df.drop(columns=['hrrr_tmp','hrrr_vpd','hrrr_wind'], errors='ignore')
df['date']=pd.to_datetime(df['date']); df['year']=df.date.dt.year
df = df[(~df.h3_cell.isin(flare)) & (df.date <= LAST)]
df = df.merge(pd.read_parquet(NT/"powerline_dist_km.parquet"), on='h3_cell', how='left')
df = df.merge(pd.read_parquet(DR/"drought_feature.parquet"), on=['h3_cell','date'], how='left')
df = df.merge(peaks, on=['h3_cell','date'], how='inner')
df = df.dropna(subset=STATIC).dropna(subset=['hrrr_vpd','hrrr_mstav','drought_score'])
df = add_ix(df)
tr, te = df[df.year<=2021], df[df.year>=2023]
yt = te.label.astype(int).values
spw = (tr.label==0).sum()/(tr.label==1).sum()

def run(feats, tag):
    mono = '('+','.join('1' if f in MONO_UP else '0' for f in feats)+')'
    m = xgb.XGBClassifier(n_estimators=1000, max_depth=9, min_child_weight=30,
        learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric='aucpr', monotone_constraints=mono, tree_method='hist',
        device='cuda', n_jobs=-1, random_state=42)
    m.fit(tr[feats], tr.label.astype(int), verbose=False)
    p = m.predict_proba(te[feats])[:,1]
    a, r = average_precision_score(yt,p), roc_auc_score(yt,p)
    print(f"{tag:<28} n_feat={len(feats):<3} AUC-PR={a:.4f}  AUROC={r:.4f}", flush=True)
    return dict(tag=tag, n=len(feats), aupr=round(float(a),4), auroc=round(float(r),4))

out=[run(FEATS,'FULL (baseline)')]
for g, cols in GROUPS.items():
    # dropping an interaction ingredient means dropping the interaction too
    drop = set(cols)
    if g=='hazard_fuels': drop |= {'bp_x_vpd','whp_x_vpd','bp_x_drought'}
    if g=='weather_peaks': drop |= {'bp_x_vpd','whp_x_vpd'}
    if g=='drought': drop |= {'bp_x_drought'}
    out.append(run([f for f in FEATS if f not in drop], f'minus {g}'))
out.append(run(GROUPS['hazard_fuels']+GROUPS['terrain_region']+GROUPS['human_infra'], 'ONLY statics (WHERE)'))
out.append(run(GROUPS['weather_peaks']+GROUPS['drought']+GROUPS['calendar'], 'ONLY weather+season (WHEN)'))
json.dump(out, open('lofo_results.json','w'), indent=2)
print("DONE")
