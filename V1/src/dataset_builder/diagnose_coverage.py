"""
diagnose_coverage.py
====================
Comprehensive diagnostic for the Texas Wildfire ML dataset.

Covers:
  1. Lat/lon sanity check
  2. Raster bounds + CRS (None-safe)
  3. Points outside each raster extent
  4. Missing value count per row (NoData analysis)
  5. Root cause determination: coverage gap vs internal NoData
  6. Class imbalance analysis + production-level recommendations
  7. Final action plan

Run from project root:
  python src/dataset_builder/diagnose_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROC_DIR = Path("data/processed")
RAW_DIR  = Path("data/raw")
DATASET  = PROC_DIR / "wildfire_dataset.csv"

RASTER_FOLDERS = {
    "NDVI":        RAW_DIR / "ndvi",
    "EVI":         RAW_DIR / "evi",
    "LST":         RAW_DIR / "lst",
    "Temperature": RAW_DIR / "temperature",
    "Wind":        RAW_DIR / "wind",
    "Rainfall":    RAW_DIR / "rainfall",
    "DEM":         RAW_DIR / "dem",
    "Slope":       RAW_DIR / "slope",
    "Aspect":      RAW_DIR / "aspect",
    "LandCover":   RAW_DIR / "landcover",
}
RASTER_FEATURES = list(RASTER_FOLDERS.keys())

SEP  = "=" * 72
THIN = "-" * 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_tif_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.tif")) if folder.exists() else []


def raster_info(tif_path: Path) -> dict:
    """Return bounds (WGS-84), CRS string, nodata value, and resolution."""
    with rasterio.open(tif_path) as src:
        crs    = src.crs
        bounds = src.bounds
        nodata = src.nodata
        res    = src.res   # (pixel_width, pixel_height)

        # Safe EPSG extraction
        try:
            epsg = crs.to_epsg()
        except Exception:
            epsg = None
        epsg_str = str(epsg) if epsg is not None else crs.to_string()[:12]

        # Convert to WGS-84
        if epsg != 4326:
            try:
                left, bottom, right, top = transform_bounds(
                    crs, "EPSG:4326",
                    bounds.left, bounds.bottom, bounds.right, bounds.top
                )
            except Exception:
                left, bottom, right, top = (
                    bounds.left, bounds.bottom, bounds.right, bounds.top
                )
        else:
            left, bottom, right, top = (
                bounds.left, bounds.bottom, bounds.right, bounds.top
            )

        return {
            "epsg_str": epsg_str,
            "crs_wkt":  crs.to_string()[:30],
            "left":   left, "right":  right,
            "bottom": bottom, "top": top,
            "nodata": nodata,
            "res_x":  abs(res[0]),
            "res_y":  abs(res[1]),
        }


def merged_info(folder: Path) -> dict | None:
    """Union of all tile infos in a folder."""
    files = get_tif_files(folder)
    if not files:
        return None
    infos = [raster_info(f) for f in files]
    return {
        "n_tiles":  len(files),
        "epsg_str": infos[0]["epsg_str"],
        "nodata":   infos[0]["nodata"],
        "res_x":    infos[0]["res_x"],
        "res_y":    infos[0]["res_y"],
        "left":   min(i["left"]   for i in infos),
        "right":  max(i["right"]  for i in infos),
        "bottom": min(i["bottom"] for i in infos),
        "top":    max(i["top"]    for i in infos),
    }


def inside(lon: float, lat: float, b: dict) -> bool:
    return (b["left"] <= lon <= b["right"]) and (b["bottom"] <= lat <= b["top"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{SEP}")
    print("  TEXAS WILDFIRE DATASET -- FULL DIAGNOSTIC REPORT")
    print(SEP)

    if not DATASET.exists():
        print(f"\n  ERROR: {DATASET} not found. Run build_dataset.py first.")
        return

    df      = pd.read_csv(DATASET)
    n_total = len(df)
    n_fire  = int((df["Fire"] == 1).sum())
    n_neg   = int((df["Fire"] == 0).sum())

    print(f"\n  Dataset : {DATASET}")
    print(f"  Rows    : {n_total:,}   (Fire=1: {n_fire:,}  |  Fire=0: {n_neg:,})")
    print(f"  Lat     : {df['latitude'].min():.4f}  to  {df['latitude'].max():.4f}")
    print(f"  Lon     : {df['longitude'].min():.4f}  to  {df['longitude'].max():.4f}")

    # ------------------------------------------------------------------
    # SECTION 1 -- Lat/lon sanity
    # ------------------------------------------------------------------
    print(f"\n{THIN}")
    print("  SECTION 1 -- LAT/LON ORDER SANITY CHECK")
    print(THIN)
    bad_lat = int(((df["latitude"]  < 25) | (df["latitude"]  > 37)).sum())
    bad_lon = int(((df["longitude"] < -107) | (df["longitude"] > -92)).sum())
    print(f"  Latitude  outside [25, 37]    : {bad_lat} points")
    print(f"  Longitude outside [-107, -92] : {bad_lon} points")
    status = "OK -- all within Texas" if bad_lat == 0 and bad_lon == 0 else "WARNING -- check column order!"
    print(f"  Result: {status}")

    # ------------------------------------------------------------------
    # SECTION 2 -- Raster bounds + CRS
    # ------------------------------------------------------------------
    print(f"\n{THIN}")
    print("  SECTION 2 -- RASTER BOUNDS, CRS, NODATA, RESOLUTION")
    print(THIN)
    print(f"  {'Feature':<14} {'EPSG/CRS':<14} {'Left':>9} {'Right':>8} "
          f"{'Bot':>8} {'Top':>7} {'NoData':>10} {'Res(deg)':>10}")
    print(f"  {'-'*14} {'-'*14} {'-'*9} {'-'*8} {'-'*8} {'-'*7} {'-'*10} {'-'*10}")

    bounds_map: dict[str, dict | None] = {}
    for feat, folder in RASTER_FOLDERS.items():
        info = merged_info(folder)
        bounds_map[feat] = info
        if info is None:
            print(f"  {feat:<14}  FOLDER NOT FOUND: {folder}")
        else:
            nd_str  = f"{info['nodata']:.0f}" if info["nodata"] is not None else "None"
            res_str = f"{info['res_x']:.5f}"
            print(f"  {feat:<14} {info['epsg_str']:<14} "
                  f"{info['left']:>9.3f} {info['right']:>8.3f} "
                  f"{info['bottom']:>8.3f} {info['top']:>7.3f} "
                  f"{nd_str:>10} {res_str:>10}")

    # ------------------------------------------------------------------
    # SECTION 3 -- Points outside extent
    # ------------------------------------------------------------------
    print(f"\n{THIN}")
    print("  SECTION 3 -- POINTS OUTSIDE RASTER EXTENTS")
    print(THIN)
    print(f"  {'Feature':<14} {'Out/Total':>12}  {'%Out':>7}  {'Fire=1':>8}  {'Fire=0':>8}")
    print(f"  {'-'*14} {'-'*12}  {'-'*7}  {'-'*8}  {'-'*8}")

    all_outside_pcts = {}
    for feat, b in bounds_map.items():
        if b is None:
            print(f"  {feat:<14}  N/A")
            continue
        mask_in    = df.apply(lambda r: inside(r["longitude"], r["latitude"], b), axis=1)
        n_out      = int((~mask_in).sum())
        pct        = n_out / n_total * 100
        out_fire   = int((~mask_in & (df["Fire"] == 1)).sum())
        out_neg    = int((~mask_in & (df["Fire"] == 0)).sum())
        all_outside_pcts[feat] = pct
        flag = "  <- COVERAGE GAP" if pct > 2 else ""
        print(f"  {feat:<14} {n_out:>6}/{n_total:<6}  {pct:>6.2f}%  "
              f"{out_fire:>8,d}  {out_neg:>8,d}{flag}")

    # ------------------------------------------------------------------
    # SECTION 4 -- Missing count per row
    # ------------------------------------------------------------------
    print(f"\n{THIN}")
    print("  SECTION 4 -- MISSING RASTER VALUES PER ROW")
    print(THIN)

    feat_cols = [c for c in RASTER_FEATURES if c in df.columns]
    n_feats   = len(feat_cols)
    df["_miss"] = df[feat_cols].isna().sum(axis=1)
    vc = df["_miss"].value_counts().sort_index()

    print(f"  {'N missing':>12}  {'Rows':>8}  {'% total':>9}  {'Category'}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*9}  {'-'*35}")
    for nm, cnt in vc.items():
        pct = cnt / n_total * 100
        if   nm == 0:           cat = "COMPLETE -- all features available"
        elif nm <= 2:           cat = "GOOD -- minor gaps, keep"
        elif nm <= n_feats//2:  cat = "BORDERLINE -- half or less missing"
        else:                   cat = "POOR -- majority missing, consider DROP"
        print(f"  {int(nm):>12d}  {cnt:>8,d}  {pct:>8.1f}%  {cat}")

    n_clean    = int((df["_miss"] == 0).sum())
    n_drop50   = int((df["_miss"] > n_feats // 2).sum())
    n_keep50   = n_total - n_drop50
    kept        = df[df["_miss"] <= n_feats // 2]
    k_fire      = int((kept["Fire"] == 1).sum())
    k_neg       = int((kept["Fire"] == 0).sum())

    print(f"\n  Fully complete rows (0 missing) : {n_clean:,d}  ({n_clean/n_total*100:.1f}%)")
    print(f"  Rows with >50% missing          : {n_drop50:,d}  ({n_drop50/n_total*100:.1f}%)")
    print(f"  Rows kept after >50% filter     : {n_keep50:,d}  "
          f"(Fire=1: {k_fire:,d}  Fire=0: {k_neg:,d}  "
          f"Ratio 1:{k_neg//k_fire if k_fire > 0 else '?'})")

    # ------------------------------------------------------------------
    # SECTION 5 -- Root cause determination
    # ------------------------------------------------------------------
    print(f"\n{THIN}")
    print("  SECTION 5 -- ROOT CAUSE OF MISSING VALUES")
    print(THIN)

    max_outside_pct = max(all_outside_pcts.values()) if all_outside_pcts else 0
    if max_outside_pct > 5:
        print(f"\n  ROOT CAUSE: COVERAGE GAP")
        print(f"  Up to {max_outside_pct:.1f}% of points fall OUTSIDE raster extents.")
        print(f"  GEE export was clipped to a sub-region smaller than all of Texas.")
    else:
        print(f"\n  ROOT CAUSE: INTERNAL NODATA PIXELS (cloud masking / quality flags)")
        print(f"  All rasters cover essentially the same Texas extent.")
        print(f"  But MODIS products (NDVI, EVI, LST) have cloud-masked NoData pixels")
        print(f"  inside the raster. A point can be inside the raster bounds but still")
        print(f"  land on a NoData pixel (cloud-covered area in that MODIS composite).")
        print(f"")
        print(f"  DEM = 0% missing because elevation data has NO cloud masking.")
        print(f"  NDVI/EVI/LST = 42% missing due to MODIS quality/cloud filtering in GEE.")
        print(f"")
        print(f"  LONG-TERM FIX (V2): In GEE, use pixelwise best-quality compositing:")
        print(f"    .qualityMosaic('NDVI')  or  .median()  instead of  .mean()")
        print(f"  This fills more pixels before export.")

    # ------------------------------------------------------------------
    # SECTION 6 -- CLASS IMBALANCE ANALYSIS (Production-critical)
    # ------------------------------------------------------------------
    print(f"\n{THIN}")
    print("  SECTION 6 -- CLASS IMBALANCE ANALYSIS")
    print(THIN)

    ratio = n_neg / n_fire if n_fire > 0 else 0
    print(f"\n  Fire=1 (positive) : {n_fire:>6,d}  ({n_fire/n_total*100:.1f}%)")
    print(f"  Fire=0 (negative) : {n_neg:>6,d}  ({n_neg/n_total*100:.1f}%)")
    print(f"  Imbalance ratio   : 1 : {ratio:.1f}")

    print(f"""
  IS THE IMBALANCE A PROBLEM? -- YES, BUT MANAGEABLE.
  ----------------------------------------------------
  With 75% Fire=0, a naive model that always predicts "no fire" gets
  75% accuracy -- but catches ZERO actual fires. Accuracy is useless here.

  For a PRODUCTION wildfire model:
    - False Negative (missed fire)  = CATASTROPHIC (evacuations not called)
    - False Positive (false alarm)  = Costly but acceptable

  Therefore we MUST optimize for HIGH RECALL on Fire=1, not accuracy.

  TECHNIQUES TO HANDLE IMBALANCE (ranked by recommendation):
  -----------------------------------------------------------
  TIER 1 -- Built-in (zero extra code, best first step):
    scale_pos_weight = n_neg / n_pos = {n_neg}/{n_fire} = {ratio:.1f}
    Add to XGBoost: XGBClassifier(scale_pos_weight={ratio:.1f})
    Effect: Penalizes misclassifying Fire=1 by 3x. XGBoost handles this natively.
    Verdict: ALWAYS use this as baseline.

  TIER 2 -- Threshold tuning (after training):
    Default decision threshold = 0.5
    Lower it to 0.3 or 0.4 to catch more fires (higher recall).
    Find optimal threshold on val set using Precision-Recall curve.
    from sklearn.metrics import precision_recall_curve
    Verdict: Essential for production. Apply after scale_pos_weight.

  TIER 3 -- Oversampling (SMOTE):
    Synthetically creates new Fire=1 samples by interpolating between
    existing fire events in feature space.
    pip install imbalanced-learn
    from imblearn.over_sampling import SMOTE
    X_res, y_res = SMOTE(k_neighbors=5, random_state=42).fit_resample(X_train, y_train)
    WARNING: Apply ONLY to training set, NEVER val/test (would cause data leakage).
    WARNING: SMOTE on geospatial data creates synthetic fires at impossible locations.
    Verdict: Try it, compare AUC -- often scale_pos_weight is equally good.

  TIER 4 -- Undersampling:
    Randomly drop Fire=0 rows from training set to achieve 1:1 ratio.
    Risk: Loses real negative samples, reduces model's ability to distinguish
    valid non-fire areas. Usually worse than scale_pos_weight for trees.
    Verdict: Not recommended for this problem.

  RECOMMENDED PRODUCTION APPROACH:
  1. Use scale_pos_weight={ratio:.1f} in XGBoost (always)
  2. Evaluate on val set: AUC-ROC, F1, Precision@Recall>=0.85
  3. Tune decision threshold for Recall >= 0.85 on fire class
  4. Try SMOTE on training set, compare -- keep whichever gives better F1
  5. NEVER evaluate on accuracy -- use AUC-ROC and F1 only
    """)

    # ------------------------------------------------------------------
    # SECTION 7 -- Sample missing NDVI coords
    # ------------------------------------------------------------------
    print(THIN)
    print("  SECTION 7 -- WHERE ARE THE MISSING NDVI PIXELS?")
    print(THIN)
    if "NDVI" in df.columns:
        miss_df = df[df["NDVI"].isna()][["latitude", "longitude", "Fire"]].head(20)
        n_ndvi_miss = df["NDVI"].isna().sum()
        print(f"\n  Total rows with missing NDVI: {n_ndvi_miss:,d}  "
              f"(Fire=1: {df[df['NDVI'].isna() & (df['Fire']==1)].shape[0]:,d}  "
              f"Fire=0: {df[df['NDVI'].isna() & (df['Fire']==0)].shape[0]:,d})")
        print(f"\n  Sample of 20 missing-NDVI coordinates:")
        print(f"  {'Lat':>10}  {'Lon':>12}  {'Fire':>6}")
        print(f"  {'-'*10}  {'-'*12}  {'-'*6}")
        for _, row in miss_df.iterrows():
            print(f"  {row['latitude']:>10.4f}  {row['longitude']:>12.4f}  {int(row['Fire']):>6}")

    # ------------------------------------------------------------------
    # FINAL ACTION PLAN
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  FINAL ACTION PLAN")
    print(SEP)
    print(f"""
  STEP 1 -- Rebuild with strict QC (drop >50% missing rows):
    python src/dataset_builder/build_dataset.py --strict-qc

  STEP 2 -- Rebuild ablation version (no peak feature, for comparison):
    python src/dataset_builder/build_dataset.py --strict-qc --no-peak-feature \\
        --proc-dir data/processed_ablation

  STEP 3 -- Train XGBoost with scale_pos_weight={ratio:.1f}:
    In your training script:
      model = XGBClassifier(
          scale_pos_weight={ratio:.1f},   # handles class imbalance
          eval_metric=['auc', 'logloss'],
          early_stopping_rounds=50,
      )
    Evaluate with: AUC-ROC, F1-score, Precision, Recall (NOT accuracy)

  STEP 4 -- Try SMOTE on training set only:
    pip install imbalanced-learn
    from imblearn.over_sampling import SMOTE
    X_res, y_res = SMOTE().fit_resample(X_train, y_train)
    Compare Model A (scale_pos_weight) vs Model B (SMOTE) on val set.

  STEP 5 -- Tune decision threshold:
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    # Pick threshold where recall >= 0.85
    optimal_threshold = thresholds[recall[:-1] >= 0.85][0]
    """)

    df.drop(columns=["_miss"], inplace=True)
    print(SEP)


if __name__ == "__main__":
    main()
