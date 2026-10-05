#!/usr/bin/env python
"""
TDIS 24-48h Ignition Forecast — production scoring entrypoint (self-contained).

Scores every TX H3 res-8 cell for a target day with the served, flare-filtered,
d9-promoted model + isotonic calibrator. All paths are resolved RELATIVE TO THIS
PACKAGE, so it runs standalone from wherever TDIS_Ignition/ is placed — no
dependency on the original source repo.

Two modes
---------
  LIVE (default): pull the 12Z-day-before HRRR forecast via Herbie (lead F24-F48),
    assemble the 23 features for all 1.7M cells, score + calibrate, write output.
      python serve/score_day.py 2026-08-01 24

  OFFLINE (--offline): use the packaged historical HRRR peaks
    (data/scored/peak_weather_features.parquet) instead of a live pull. Scores the
    cells that have weather on that date. Use for smoke tests / backfills — no
    network needed.
      python serve/score_day.py 2024-02-27 --offline

Output (written to serve/output/ unless --out given):
  forecast_<date>_lead<LEAD>h_res8.parquet   per-cell: h3_cell,lat,lon,ign,ign_cal,is_flare
  forecast_<date>.json                        res-5 aggregated {ign, ignCal}

Env: conda UAI2526 (xgboost, h3, herbie, xarray, cfgrib, scipy, joblib, pyarrow).
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, h3, xgboost as xgb, joblib
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent        # TDIS_Ignition/
MODEL_DIR, STATIC, SCORED, FLARES = ROOT/"model", ROOT/"data"/"static", ROOT/"data"/"scored", ROOT/"data"/"flares"
MODEL_F = MODEL_DIR/"tdis_forecast_hrrr_filtered.json"
CAL_F   = MODEL_DIR/"operational_isotonic_calibrator.joblib"
TX = dict(lat_min=25.75, lat_max=36.65, lon_min=-106.70, lon_max=-93.40)
HOLIDAYS = {(1,1),(7,4),(6,19),(11,11),(12,25),(10,31)}


def log(m): print(m, flush=True)
def vpd_from(t, td):
    es = 0.6108*np.exp(17.27*t/(t+237.3)); ea = 0.6108*np.exp(17.27*td/(td+237.3))
    return np.clip(es-ea, 0, None)


def load_static(feats):
    """Static master + powerline (joined separately) + is_flare flag."""
    st = pd.read_parquet(STATIC/"tx_static_master.parquet")
    if "powerline_dist_km" in feats:
        st = st.merge(pd.read_parquet(STATIC/"powerline_dist_km.parquet"), on="h3_cell", how="left")
    flare = set(pd.read_parquet(FLARES/"flare_cells_tx_v2.parquet")["h3_cell"])
    st["is_flare"] = st["h3_cell"].isin(flare)
    return st


def attach_drought(st, feats):
    if "drought_score" not in feats:
        return st
    dl = STATIC/"drought_latest.parquet"; cm = STATIC/"tx_county_fips.parquet"
    if dl.exists() and cm.exists():
        d = pd.read_parquet(dl); c = pd.read_parquet(cm)
        st = st.merge(c, on="h3_cell", how="left").merge(d, on="fips", how="left")
        st["drought_score"] = st["drought_score"].fillna(st["drought_score"].mean() if st["drought_score"].notna().any() else 1.0)
    else:
        log("  WARN: drought cache/county map missing -> neutral drought_score=1.0")
        st["drought_score"] = 1.0
    return st


def add_temporal(st, target):
    mo, dow = target.month, target.dayofweek
    st["sin_month"], st["cos_month"] = np.sin(2*np.pi*mo/12), np.cos(2*np.pi*mo/12)
    st["sin_dow"],   st["cos_dow"]   = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
    st["is_weekend"] = int(dow >= 5)
    st["is_holiday"] = int((mo, target.day) in HOLIDAYS)
    return st


def weather_live(st, target, lead):
    from herbie import Herbie
    from scipy.spatial import cKDTree
    init = target - pd.Timedelta(hours=lead)
    model, prod = ("hrrr", "sfc") if lead <= 48 else ("gfs", "pgrb2.0p25")
    log(f"Pulling {model.upper()} F{lead}: target {target.date()} init {init.date()}")
    h = Herbie(init.strftime("%Y-%m-%d %H:%M"), model=model, product=prod, fxx=lead, verbose=False)
    if h.grib is None:
        log("forecast not found on bucket"); sys.exit(2)
    subset = (":(TMP|DPT):2 m above ground|:WIND:10 m above ground|:GUST:surface|:MSTAV:"
              if model == "hrrr" else
              ":(TMP|DPT):2 m above ground|:(UGRD|VGRD):10 m above ground|:SOILW:0-0.1 m below ground")
    dsl = h.xarray(subset, remove_grib=True)
    if not isinstance(dsl, list): dsl = [dsl]
    mg = {}
    for ds in dsl:
        la = ds.latitude.values; lo = ds.longitude.values; lo = np.where(lo > 180, lo-360, lo)
        if la.ndim == 1 and lo.ndim == 1:
            LO, LA = np.meshgrid(lo, la); la, lo = LA.ravel(), LO.ravel()
        else:
            la, lo = la.ravel(), lo.ravel()
        m = (la >= TX["lat_min"]) & (la <= TX["lat_max"]) & (lo >= TX["lon_min"]) & (lo <= TX["lon_max"])
        mg.setdefault("lat", la[m]); mg.setdefault("lon", lo[m])
        for nm, da in ds.data_vars.items():
            mg[nm] = da.values.ravel()[m]
    g = pd.DataFrame(mg).rename(columns={"t2m":"tmp","d2m":"dpt","max_10si":"wind","si10":"wind",
                                         "mstav":"soilm","avail_smois":"soilm","soilw":"soilw"})
    g["tmp"] -= 273.15; g["dpt"] -= 273.15
    if "wind" not in g.columns and {"u10","v10"} <= set(g.columns):
        g["wind"] = np.sqrt(g["u10"]**2 + g["v10"]**2)
    g["vpd"] = vpd_from(g["tmp"].values, g["dpt"].values)
    if "soilm" in g.columns:   soilm = np.clip(g["soilm"].values/100.0, 0, 1)
    elif "soilw" in g.columns: soilm = np.clip(g["soilw"].values/0.45, 0, 1)
    else:                      soilm = None
    tree = cKDTree(np.c_[g["lat"].values, g["lon"].values])
    _, idx = tree.query(np.c_[st["lat"].values, st["lon"].values])
    st["hrrr_tmp"] = g["tmp"].values[idx]; st["hrrr_vpd"] = g["vpd"].values[idx]; st["hrrr_wind"] = g["wind"].values[idx]
    st["hrrr_mstav"] = (np.clip(soilm[idx], 0, 1) if soilm is not None else 0.5)
    log(f"  {len(g):,} grid points -> nearest-cell join for {len(st):,} cells")
    return st, model, lead, init


def weather_offline(st, target):
    pw = pd.read_parquet(SCORED/"peak_weather_features.parquet")
    day = pw[pw["date"] == pd.Timestamp(target)]
    if day.empty:
        log(f"OFFLINE: no packaged HRRR peaks for {target.date()} "
            f"(coverage {pw['date'].min().date()}..{pw['date'].max().date()})"); sys.exit(2)
    st = st.merge(day[["h3_cell","hrrr_tmp","hrrr_vpd","hrrr_wind","hrrr_mstav"]], on="h3_cell", how="inner")
    log(f"  OFFLINE: {len(st):,} cells with packaged weather on {target.date()}")
    return st, "HRRR-offline", 24, target - pd.Timedelta(hours=24)


def main():
    ap = argparse.ArgumentParser(description="TDIS 24-48h ignition forecast — score one day")
    ap.add_argument("date", help="target day YYYY-MM-DD")
    ap.add_argument("lead", nargs="?", type=int, default=24, help="forecast lead hours (default 24)")
    ap.add_argument("--offline", action="store_true", help="use packaged historical HRRR peaks (no live pull)")
    ap.add_argument("--out", type=Path, default=ROOT/"serve"/"output", help="output dir")
    ap.add_argument("--keep-flares", action="store_true", help="keep flare cells in output (default: flagged, kept)")
    args = ap.parse_args()
    target = pd.Timestamp(args.date)
    args.out.mkdir(parents=True, exist_ok=True)

    assert MODEL_F.exists() and CAL_F.exists(), f"model/calibrator missing under {MODEL_DIR}"
    mdl = xgb.XGBClassifier(); mdl.load_model(str(MODEL_F))
    feats = mdl.get_booster().feature_names
    iso = joblib.load(CAL_F)
    log(f"model: {MODEL_F.name} ({len(feats)} features) + {CAL_F.name}")

    st = load_static(feats)
    st = attach_drought(st, feats)
    st = add_temporal(st, target)
    st, model, lead, init = (weather_offline(st, target) if args.offline
                             else weather_live(st, target, args.lead))

    missing = [f for f in feats if f not in st.columns]
    assert not missing, f"features not assembled: {missing}"
    st["ign"] = mdl.predict_proba(st[feats])[:, 1]
    st["ign_cal"] = iso.predict(st["ign"].values)
    log(f"scored {len(st):,} cells | raw mean {st['ign'].mean():.3f} -> cal mean {st['ign_cal'].mean():.4f} "
        f"| flare cells {int(st['is_flare'].sum())}")

    out = st if args.keep_flares else st  # kept + flagged; consumers filter is_flare
    res8 = args.out/f"forecast_{target.date()}_lead{lead}h_res8.parquet"
    out[["h3_cell","lat","lon","ign","ign_cal","is_flare"]].to_parquet(res8, index=False)
    log(f"wrote {res8.name} ({res8.stat().st_size/1e6:.1f} MB)")

    st["h3_5"] = [h3.cell_to_parent(c, 5) for c in st["h3_cell"].values]
    keep = st[~st["is_flare"]]  # res-5 aggregate excludes flares (operational view)
    agg = keep.groupby("h3_5").agg(ign=("ign","mean"), ign_cal=("ign_cal","mean")).reset_index()
    payload = {"target": str(target.date()), "lead_h": lead, "init": str(init.date()),
               "source": f"{model} F{lead}", "n_cells_res8": int(len(st)),
               "ign":    {r.h3_5: round(float(r.ign), 4) for r in agg.itertuples()},
               "ignCal": {r.h3_5: round(float(r.ign_cal), 4) for r in agg.itertuples()}}
    pj = args.out/f"forecast_{target.date()}.json"; json.dump(payload, open(pj, "w"), separators=(",", ":"))
    log(f"wrote {pj.name} ({len(agg):,} res-5 cells, flares excluded from aggregate)")


if __name__ == "__main__":
    main()
