"""
TDIS Forecast — Step 18: FLARE-FILTERED labels + clean retrain (before/after).

Two label defects found in QC (see DEMO_SMOKEHOUSE §2B, label deep-dive 2026-08-05):
  1. INDUSTRIAL PERSISTENT HOTSPOTS: res-8 cells "on fire" >3% of all days are gas
     flares/plants, not wildfires — 531 cells carrying ~27% of all positive labels.
  2. PHANTOM FUTURE NEGATIVES: 86k test rows dated after the last real label
     (2026-07-29) — unknowables mislabeled as negatives (03 sampled the full calendar).

This script:
  a. identifies flare cells (persistence > FLARE_FRAC of the label-era days)
  b. writes data/labels_fused/flare_cells.parquet + a filtered training table
     (drops ALL rows — positives and matched negatives — at flare cells, and all
     rows after LAST_LABEL)
  c. retrains the honest baseline on the filtered table (same features/hyperparams
     as 07) -> models/tdis_forecast_baseline_honest_filtered.json
  d. before/after comparison ON THE SAME FILTERED TEST ROWS (apples-to-apples):
     old model vs new model.
"""
import json, warnings
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')
TF = Path(__file__).resolve().parent.parent
def log(m): print(m, flush=True)

FLARE_FRAC = 0.03          # 'fire' on >3% of all days = persistent industrial source
LAST_LABEL = pd.Timestamp('2026-07-29')

# ── a. flare cells from label persistence ──
lb = pd.read_parquet(TF/"data/labels_fused/ignitions_daily_tx.parquet")
lb['date'] = pd.to_datetime(lb['date'])
ndays = (lb.date.max() - lb.date.min()).days
per = lb.groupby('h3_cell').size()
flare = per[per > ndays*FLARE_FRAC]
flare.rename('fire_days').reset_index().to_parquet(TF/"data/labels_fused/flare_cells.parquet", index=False)
log(f"flare cells (>{FLARE_FRAC*100:.0f}% of {ndays} days): {len(flare):,} cells, "
    f"{flare.sum():,} label rows ({flare.sum()/len(lb)*100:.1f}% of all positives)")
flare_set = set(flare.index)

# ── b. filtered training table ──
df = pd.read_parquet(TF/"tdis_train_daily_tx.parquet")
df['date'] = pd.to_datetime(df['date'])
n0, p0 = len(df), int(df.label.sum())
df = df[~df.h3_cell.isin(flare_set)]
df = df[df.date <= LAST_LABEL]
log(f"training table: {n0:,} rows ({p0:,} pos) -> {len(df):,} rows ({int(df.label.sum()):,} pos) "
    f"after flare+phantom filter")
df.to_parquet(TF/"tdis_train_daily_tx_flarefiltered.parquet", index=False)

# ── c. retrain honest baseline (identical setup to 07) ──
STATIC   = ['road_dist_km','ecoregion_id','elevation_m','slope_deg','aspect_deg',
            'avg_burn_prob','whp','flep4','cfl','cbd','cbh']
TEMPORAL = ['sin_month','cos_month','sin_dow','cos_dow','is_weekend','is_holiday']
FEATS = STATIC + TEMPORAL
df = df.dropna(subset=STATIC).reset_index(drop=True)
df['split'] = np.where(df.year<=2021,'train',np.where(df.year==2022,'val','test'))
tr,va,te = df[df.split=='train'], df[df.split=='val'], df[df.split=='test']
spw = (tr.label==0).sum()/max((tr.label==1).sum(),1)
log(f"train {len(tr):,} / val {len(va):,} / test {len(te):,} | test pos rate {te.label.mean():.4f} | spw={spw:.2f}")
# NOTE: early stopping removed — the 2022 val year shows a spurious AUCPR spike at
# iteration ~3 under filtered labels (then climbs monotonically); fixed 400-tree budget
# (val still rising at 399) restores real spatial spread (pred-std 0.02 -> 0.18).
m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
    scale_pos_weight=spw, eval_metric='aucpr',
    tree_method='hist', device='cuda', n_jobs=-1, random_state=42)
m.fit(tr[FEATS], tr.label.astype(int), eval_set=[(va[FEATS], va.label.astype(int))], verbose=False)

# ── d. before/after on the SAME filtered test rows ──
old = xgb.XGBClassifier(); old.load_model(str(TF/"models/tdis_forecast_baseline_honest.json"))
yte = te.label.astype(int).values
p_new = m.predict_proba(te[FEATS])[:,1]
p_old = old.predict_proba(te[old.get_booster().feature_names])[:,1]
res = {}
for tag,p in [('OLD model (flare-trained)',p_old), ('NEW model (flare-filtered)',p_new)]:
    ap, roc = average_precision_score(yte,p), roc_auc_score(yte,p)
    res[tag] = (ap, roc)
    log(f"{tag:>28}: AUC-PR={ap:.4f}  AUROC={roc:.4f}  lift={ap/yte.mean():.2f}x   [same clean test rows]")

m.save_model(str(TF/"models/tdis_forecast_baseline_honest_filtered.json"))
imp = dict(zip(FEATS, [round(float(x),4) for x in m.feature_importances_]))
meta = {'tag':'honest_filtered', 'features':FEATS, 'n_features':len(FEATS),
        'filters':{'flare_frac':FLARE_FRAC, 'n_flare_cells':int(len(flare)),
                   'last_label':str(LAST_LABEL.date())},
        'split':{'train':'2014-2021','val':'2022','test':'2023-2026 (clipped to last label)'},
        'test_pos_rate':round(float(yte.mean()),4),
        'clean_test_comparison':{k:{'aucpr':round(v[0],4),'auroc':round(v[1],4)} for k,v in res.items()},
        'n_trees':400,
        'feature_importance':dict(sorted(imp.items(), key=lambda x:-x[1])),
        'note':'Honest baseline retrained WITHOUT flare cells (persistence>3%) and WITHOUT phantom post-label-date negatives. Comparison rows are identical for old/new.'}
json.dump(meta, open(TF/"models/tdis_forecast_baseline_honest_filtered_meta.json",'w'), indent=1)
log("Saved models/tdis_forecast_baseline_honest_filtered.json (+meta). DONE")
