"""
rev2_improvements — model behavior diagnostics: train/val/test gap (overfitting
check) + SHAP (direction + interaction, fixing what gain-importance can't show).

Run against the current baseline (operational_hrrr_filtered) now, and again
against step5_model.json once that finishes, for a real before/after comparison
of not just "did the metric move" but "did the model's behavior change in an
understandable, physically-sensible way, or did it just get noisier."

Usage:
  $PY diagnose_model.py --variant baseline
  $PY diagnose_model.py --variant step5
"""
import argparse, json, tempfile, warnings
import numpy as np, pandas as pd, xgboost as xgb, shap
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')


def shap_safe_explainer(model_or_path):
    """SHAP's XGBTreeModelLoader calls xgb_model.save_raw(raw_format='ubj') and
    decodes it itself -- this XGBoost version embeds base_score as a bracketed
    string ('[5E-1]') even in that internal re-serialization, so patching the
    model *file* doesn't help (XGBoost re-emits the bracket form regardless).
    The only reliable interception point is decode_ubjson_buffer, which SHAP
    calls right after unpacking the raw bytes -- fix the value there."""
    import shap.explainers._tree as _t
    if not hasattr(_t, '_base_score_patch_applied'):
        _orig_decode = _t.decode_ubjson_buffer
        def _patched_decode(fd):
            jmodel = _orig_decode(fd)
            try:
                lmp = jmodel['learner']['learner_model_param']
                bs = lmp['base_score']
                if isinstance(bs, str) and bs.startswith('['):
                    lmp['base_score'] = bs.strip('[]')
            except Exception:
                pass
            return jmodel
        _t.decode_ubjson_buffer = _patched_decode
        _t._base_score_patch_applied = True

    if isinstance(model_or_path, xgb.XGBClassifier):
        m = model_or_path
    else:
        m = xgb.XGBClassifier(); m.load_model(str(model_or_path))
    return shap.TreeExplainer(m)

TF = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).parent
LAST_LABEL = pd.Timestamp('2026-07-29')

STATIC = ['road_dist_km', 'ecoregion_id', 'elevation_m', 'slope_deg', 'aspect_deg',
          'avg_burn_prob', 'whp', 'flep4', 'cfl', 'cbd', 'cbh']
TEMPORAL = ['sin_month', 'cos_month', 'sin_dow', 'cos_dow', 'is_weekend', 'is_holiday']

VARIANTS = {
    'baseline': dict(
        model=TF / "models" / "tdis_forecast_hrrr_filtered.json",
        table=TF / "tdis_train_daily_hrrr.parquet",
        extra_tables=[],
        feats=STATIC + TEMPORAL + ['hrrr_tmp', 'hrrr_vpd', 'hrrr_wind'],
    ),
    'step5': dict(
        model=OUT_DIR / "step5_model.json",
        table=TF / "tdis_train_daily_hrrr.parquet",
        extra_tables=[],  # step5's own script already merges gust+trailing; we reload its saved table if present
        feats=STATIC + TEMPORAL + ['hrrr_tmp', 'hrrr_vpd', 'hrrr_wind', 'hrrr_gust',
                                    'erc_5D_mean', 'erc_5D_max', 'vpd_5D_mean', 'vpd_5D_max',
                                    'vs_5D_mean', 'vs_5D_max'],
    ),
}


def log(m): print(m, flush=True)


def build_frame(variant, cfg):
    if variant == 'baseline':
        df = pd.read_parquet(cfg['table'])
        df['date'] = pd.to_datetime(df['date'])
        flare_set = set(pd.read_parquet(TF / "data/labels_fused/flare_cells.parquet")['h3_cell'])
        df = df[~df.h3_cell.isin(flare_set)]
        df = df[df.date <= LAST_LABEL]
        df = df.dropna(subset=STATIC).dropna(subset=['hrrr_vpd'])
        df['year'] = df['date'].dt.year
        df['split'] = np.where(df.year <= 2021, 'train', np.where(df.year == 2022, 'val', 'test'))
        return df
    else:
        # step5's own script rebuilds the merged table in-memory; re-derive here the same way
        # by re-running its build functions would duplicate a lot -- instead read back from
        # the saved model's own training run isn't persisted as a table, so rebuild minimally:
        import importlib.util
        spec = importlib.util.spec_from_file_location("step5mod", OUT_DIR / "step5_combined_gust_trailing.py")
        step5mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step5mod)
        gust = step5mod.build_gust()
        trailing = step5mod.build_trailing_stats()
        base = pd.read_parquet(TF / "tdis_train_daily_hrrr.parquet")
        base['date'] = pd.to_datetime(base['date'])
        df = base.merge(gust, on=['h3_cell', 'date'], how='left').merge(trailing, on=['h3_cell', 'date'], how='left')
        flare_set = set(pd.read_parquet(TF / "data/labels_fused/flare_cells.parquet")['h3_cell'])
        df['year'] = df['date'].dt.year
        df = df[~df.h3_cell.isin(flare_set)]
        df = df[df.date <= LAST_LABEL]
        df = df.dropna(subset=STATIC).dropna(subset=['hrrr_vpd'])
        df['split'] = np.where(df.year <= 2021, 'train', np.where(df.year == 2022, 'val', 'test'))
        return df


def run(variant):
    cfg = VARIANTS[variant]
    if not cfg['model'].exists():
        raise SystemExit(f"model not found yet: {cfg['model']}")
    log(f"=== Diagnosing variant: {variant} ===")
    df = build_frame(variant, cfg)
    feats = cfg['feats']
    m = xgb.XGBClassifier(); m.load_model(str(cfg['model']))

    log("\n--- Train/Val/Test gap (overfitting check) ---")
    gap_results = {}
    for split in ['train', 'val', 'test']:
        d = df[df.split == split]
        y = d.label.astype(int).values
        p = m.predict_proba(d[feats])[:, 1]
        auroc = roc_auc_score(y, p)
        aucpr = average_precision_score(y, p)
        gap_results[split] = dict(n=len(d), pos_rate=round(float(y.mean()), 4),
                                   auroc=round(float(auroc), 4), aucpr=round(float(aucpr), 4))
        log(f"  {split:5s}  n={len(d):>9,}  pos_rate={y.mean():.4f}  AUROC={auroc:.4f}  AUC-PR={aucpr:.4f}")
    train_val_gap = gap_results['train']['auroc'] - gap_results['val']['auroc']
    val_test_gap = gap_results['val']['auroc'] - gap_results['test']['auroc']
    log(f"  train-val gap (memorization risk): {train_val_gap:+.4f}")
    log(f"  val-test gap (forward-generalization risk): {val_test_gap:+.4f}")

    log("\n--- SHAP (direction + interaction, on a sample of test rows) ---")
    te = df[df.split == 'test']
    sample = te.sample(n=min(8000, len(te)), random_state=42)
    explainer = shap_safe_explainer(cfg['model'])
    sv = explainer.shap_values(sample[feats])
    mean_abs_shap = dict(sorted(zip(feats, np.abs(sv).mean(axis=0)), key=lambda x: -x[1]))
    log("  Mean |SHAP| (global importance, direction-aware method):")
    for k, v in list(mean_abs_shap.items())[:12]:
        # direction: correlation between feature value and its own SHAP contribution
        fv = sample[k].values
        valid = ~np.isnan(fv)
        direction = np.corrcoef(fv[valid], sv[valid, feats.index(k)])[0, 1] if valid.sum() > 10 else np.nan
        log(f"    {k:16s}  mean|SHAP|={v:.4f}  direction_corr={direction:+.3f}"
            f"  ({'higher value -> higher risk' if direction > 0 else 'higher value -> lower risk' if direction < 0 else 'n/a'})")

    interaction_note = {}
    wind_like = [f for f in ['hrrr_wind', 'hrrr_gust'] if f in feats]
    dryness_like = [f for f in ['hrrr_vpd', 'vpd_5D_mean', 'vpd_5D_max', 'erc_5D_mean', 'erc_5D_max'] if f in feats]
    if wind_like and dryness_like:
        log("\n  Checking wind x dryness interaction (does the model discover the HWP-style multiplicative pattern?)...")
        iv = explainer.shap_interaction_values(sample[feats].iloc[:min(2000, len(sample))])
        for wf in wind_like:
            for df_ in dryness_like:
                i, j = feats.index(wf), feats.index(df_)
                inter_strength = float(np.abs(iv[:, i, j]).mean())
                interaction_note[f'{wf}_x_{df_}'] = round(inter_strength, 5)
                log(f"    {wf} x {df_}: mean|interaction SHAP| = {inter_strength:.5f}")

    out = dict(variant=variant, gap=gap_results, train_val_gap=round(train_val_gap, 4),
               val_test_gap=round(val_test_gap, 4),
               mean_abs_shap={k: round(float(v), 5) for k, v in mean_abs_shap.items()},
               interactions=interaction_note)
    json.dump(out, open(OUT_DIR / f"diagnose_{variant}.json", 'w'), indent=2)
    log(f"\nSaved -> diagnose_{variant}.json")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=list(VARIANTS.keys()))
    args = ap.parse_args()
    run(args.variant)
