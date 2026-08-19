"""
TDIS Forecast — Step 7: BASELINE 24-hour ignition FORECAST model.

Forecasting discipline (the whole point):
  1. TEMPORAL split — train on the PAST, test on the FUTURE. Never random.
       train = 2014-2021 | val = 2022 | test = 2023-2026 (real VIIRS labels, never seen)
  2. NO same-day weather leak in the deployable model. At forecast-issue time you do
       NOT know the target day's observed weather. The honest model uses only what's
       knowable in advance: static (roads/terrain/fuels/ecoregion) + temporal (season,
       day-of-week, holiday).

Two models on the identical split, to bracket reality:
  HONEST  = static + temporal            -> deployable-today floor (no leakage)
  CEILING = static + temporal + same-day weather -> optimistic bound IF forecasts were perfect
The real operational model (once HRRR forecast weather lands) swaps in FORECAST weather
and will land BETWEEN these two.

Primary metric: AUC-PR (rare-positive; consistent with all prior TX work).
Outputs (models/):
  tdis_forecast_baseline_honest.json + _meta.json
  tdis_forecast_baseline_ceiling.json + _meta.json
  baseline_forecast_predictions_test.parquet   (for the dashboard demo)
"""
import json, warnings
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')

TF = Path(__file__).resolve().parent.parent
MODELS = TF/"models"; MODELS.mkdir(exist_ok=True)
def log(m): print(m, flush=True)

# ── feature groups ──
STATIC   = ['road_dist_km','ecoregion_id','elevation_m','slope_deg','aspect_deg',
            'avg_burn_prob','whp','flep4','cfl','cbd','cbh']
TEMPORAL = ['sin_month','cos_month','sin_dow','cos_dow','is_weekend','is_holiday']
WEATHER  = ['erc','fm100','vpd','vs','rmax','rmin','tmmx','pr']   # SAME-DAY -> ceiling only

FEATURES = {'honest': STATIC+TEMPORAL, 'ceiling': STATIC+TEMPORAL+WEATHER}

# ── temporal split (train past / test future) ──
TRAIN_MAX, VAL_YR = 2021, 2022   # test = 2023-2026

log("Loading dataset...")
df = pd.read_parquet(TF/"tdis_train_daily_tx.parquet")
df = df.dropna(subset=STATIC).reset_index(drop=True)   # need static present for any model
df['split'] = np.where(df.year<=TRAIN_MAX,'train',np.where(df.year==VAL_YR,'val','test'))
log(f"  {len(df):,} rows (static-present). split counts: {df.groupby('split').size().to_dict()}")

def train_eval(tag):
    feats = FEATURES[tag]
    d = df.dropna(subset=feats) if tag=='ceiling' else df   # ceiling also needs weather present
    tr, va, te = d[d.split=='train'], d[d.split=='val'], d[d.split=='test']
    Xtr,ytr = tr[feats], tr.label.astype(int)
    Xva,yva = va[feats], va.label.astype(int)
    Xte,yte = te[feats], te.label.astype(int)
    spw = (ytr==0).sum()/max((ytr==1).sum(),1)
    log(f"\n[{tag}] {len(feats)} features | train {len(tr):,} / val {len(va):,} / test {len(te):,} | spw={spw:.2f}")
    m = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        scale_pos_weight=spw, eval_metric='aucpr', early_stopping_rounds=40,
        tree_method='hist', n_jobs=-1, random_state=42)
    m.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
    p_te = m.predict_proba(Xte)[:,1]
    aupr, auroc = average_precision_score(yte,p_te), roc_auc_score(yte,p_te)
    base = yte.mean()
    log(f"[{tag}] TEST (2023-2026, future holdout): AUC-PR={aupr:.4f}  AUROC={auroc:.4f}")
    log(f"[{tag}]   random baseline AUC-PR (=pos rate)={base:.4f}  ->  lift {aupr/base:.2f}x")
    # per-year future test breakdown
    te = te.copy(); te['p']=p_te
    log(f"[{tag}]   per future-year:")
    for yr,g in te.groupby('year'):
        log(f"      {yr}: AUC-PR={average_precision_score(g.label,g.p):.4f}  AUROC={roc_auc_score(g.label,g.p):.4f}  (n={len(g):,}, pos={int(g.label.sum()):,})")
    # feature importance
    imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
    log(f"[{tag}]   top features: {', '.join(f'{k}={v:.3f}' for k,v in imp.head(6).items())}")
    m.get_booster().save_model(str(MODELS/f"tdis_forecast_baseline_{tag}.json"))
    meta = {'tag':tag,'features':feats,'n_features':len(feats),
            'split':{'train':'2014-2021','val':'2022','test':'2023-2026'},
            'test_aucpr':round(float(aupr),4),'test_auroc':round(float(auroc),4),
            'test_pos_rate':round(float(base),4),'lift_over_random':round(float(aupr/base),2),
            'best_iteration':int(m.best_iteration),'scale_pos_weight':round(float(spw),2),
            'feature_importance':{k:round(float(v),4) for k,v in imp.items()},
            'note':'FORECAST baseline. Temporal split (past->future). honest=no same-day weather; ceiling=with same-day weather (optimistic). Operational model will use HRRR FORECAST weather.'}
    json.dump(meta, open(MODELS/f"tdis_forecast_baseline_{tag}_meta.json",'w'), indent=2)
    return m, te if tag=='honest' else None

log("="*70); log("TRAINING FORECAST BASELINES (temporal split, past->future)"); log("="*70)
m_honest, te_honest = train_eval('honest')
m_ceiling, _ = train_eval('ceiling')

# ── save honest-model predictions on the future test set for the dashboard ──
cols = ['h3_cell','date','year','label','p']
te_honest[cols].to_parquet(MODELS/"baseline_forecast_predictions_test.parquet", index=False)
log(f"\nSaved honest-model future-test predictions -> baseline_forecast_predictions_test.parquet ({len(te_honest):,} rows)")
log("\nDONE. honest = deployable floor; ceiling = perfect-weather bound; HRRR-forecast model will land between.")
