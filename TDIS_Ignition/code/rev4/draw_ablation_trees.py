import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, xgboost as xgb, re
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rev4_pipeline import *

GROUPS = {
 'hazard_fuels': ['avg_burn_prob','whp','flep4','cfl','cbd','cbh'],
 'terrain_region': ['elevation_m','slope_deg','aspect_deg','ecoregion_id'],
 'human_infra': ['road_dist_km','powerline_dist_km'],
 'calendar': ['sin_month','cos_month','sin_dow','cos_dow','is_weekend','is_holiday'],
 'weather_peaks': ['hrrr_tmp','hrrr_vpd','hrrr_wind','hrrr_mstav'],
}
VARIANTS = {
 'only_static': GROUPS['hazard_fuels']+GROUPS['terrain_region']+GROUPS['human_infra'],
 'only_weather_season': GROUPS['weather_peaks']+['drought_score']+GROUPS['calendar'],
 'minus_hazard_fuels': [f for f in FEATS if f not in set(GROUPS['hazard_fuels'])|{'bp_x_vpd','whp_x_vpd','bp_x_drought'}],
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
tr = df[df.year<=2021]
spw = (tr.label==0).sum()/(tr.label==1).sum()

NICE={'hrrr_vpd':'VPD','hrrr_wind':'gust','hrrr_tmp':'temp','hrrr_mstav':'soil moisture',
      'ecoregion_id':'ecoregion','elevation_m':'elevation','slope_deg':'slope','aspect_deg':'aspect',
      'road_dist_km':'road dist','powerline_dist_km':'powerline dist','avg_burn_prob':'burn prob',
      'whp':'WHP','flep4':'FLEP4','cfl':'CFL','cbd':'CBD','cbh':'CBH','drought_score':'drought',
      'sin_month':'season(sin mo)','cos_month':'season(cos mo)','sin_dow':'day-of-wk(sin)',
      'cos_dow':'day-of-wk(cos)','is_weekend':'weekend','is_holiday':'holiday',
      'bp_x_vpd':'burnprob×VPD','whp_x_vpd':'WHP×VPD','bp_x_drought':'burnprob×drought'}
UNITS={'hrrr_vpd':' kPa','hrrr_wind':' m/s','hrrr_tmp':' °C','elevation_m':' m',
       'road_dist_km':' km','powerline_dist_km':' km'}

def draw_tree(bst, feats, title, outp):
    bst.feature_names = feats
    txt = bst.get_dump(with_stats=True)[0]
    nodes={}
    for line in txt.split('\n'):
        if not line.strip(): continue
        s=line.strip(); nodes[int(s.split(':')[0])]=s.split(':',1)[1]
    def children(nid):
        b=nodes[nid]
        if b.startswith('leaf'): return None
        return int(re.search(r'yes=(\d+)',b).group(1)), int(re.search(r'no=(\d+)',b).group(1))
    def split_of(nid):
        g=re.search(r'\[([^<]+)<([-\d.e+]+)\]', nodes[nid]); return g.group(1), float(g.group(2))
    def mean_leaf(nid):
        vals=[]; stack=[nid]
        while stack:
            n=stack.pop(); b=nodes[n]
            if b.startswith('leaf'): vals.append(float(re.search(r'leaf=([-\d.e]+)',b).group(1)))
            else: stack.extend(children(n))
        return np.mean(vals)
    fig,ax=plt.subplots(figsize=(14,7.2)); ax.axis('off')
    def draw(nid,x,y,dx,depth):
        ch=children(nid)
        if ch is None or depth==3:
            v=mean_leaf(nid); col='#d1495b' if v>0 else '#6da7ec'
            ax.scatter([x],[y],s=900,color=col,zorder=3)
            ax.annotate(f"{'RISK ↑' if v>0 else 'RISK ↓'}\n{v:+.3f}",(x,y-0.16),ha='center',fontsize=8.5)
            return
        f,thr=split_of(nid); name=NICE.get(f,f); u=UNITS.get(f,'')
        ax.scatter([x],[y],s=900,color='#2e8b57',zorder=3)
        ax.annotate(f"{name} < {thr:.3g}{u}?",(x,y+0.11),ha='center',fontsize=10,fontweight='bold')
        yl,nr=ch
        for cnid,cx,lab in [(yl,x-dx,'yes'),(nr,x+dx,'no')]:
            ax.plot([x,cx],[y-0.03,y-0.97+0.03],color='#2e8b57',lw=1.5,zorder=1)
            ax.annotate(lab,((x+cx)/2,y-0.5),ha='center',fontsize=9,style='italic',color='#444')
            draw(cnid,cx,y-1,dx/2,depth+1)
    draw(0,0,0,4.2,0)
    ax.set_xlim(-8.6,8.6); ax.set_ylim(-3.55,0.45); ax.set_title(title,fontsize=11)
    fig.tight_layout(); fig.savefig(outp,dpi=140); plt.close(fig)
    print("saved", outp, flush=True)

for name, feats in VARIANTS.items():
    mono = '('+','.join('1' if f in MONO_UP else '0' for f in feats)+')'
    m = xgb.XGBClassifier(n_estimators=1000, max_depth=9, min_child_weight=30,
        learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric='aucpr', monotone_constraints=mono, tree_method='hist',
        device='cuda', n_jobs=-1, random_state=42)
    m.fit(tr[feats], tr.label.astype(int), verbose=False)
    draw_tree(m.get_booster(), feats,
              f'Tree #1 — {name.replace("_"," ").upper()} model ({len(feats)} features)',
              f'fig_tree1_{name}.png')
