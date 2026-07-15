"""
Phase 3 — Texas Wildfire Risk Map (Fixed)
==========================================
Uses the pre-assembled test parquet (all features already computed).
No gridMET join needed — just predict on actual rows for a date.

Run:  python run_phase3_risk_map.py --state TX --date 2020-05-19 --window 18
      python run_phase3_risk_map.py --state TX --date 2019-04-15 --window 12

Tips:
  - Use dates from the TEST split (2019-2020) where fires are known
  - Use --topk 5000 to control how many cells appear on map
  - Open the .html file in Chrome/Edge browser
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

PHASE2_ROOT = Path(__file__).resolve().parent
if str(PHASE2_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE2_ROOT))

from config.phase2_config import STATE_CONFIG

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "avg_burn_prob","whp","flep4","cfl","burnable",
    "erc","fm100","vpd","vs","rmax","rmin","tmmx","pr",
    "erc_5D_mean","erc_5D_max","fm100_5D_mean","fm100_5D_min",
    "vpd_5D_mean","vpd_5D_max","vs_5D_mean","vs_5D_max",
    "rmax_5D_mean","rmax_5D_min","tmmx_5D_mean","tmmx_5D_max",
    "sin_month","cos_month","sin_hour","cos_hour",
    "centroid_lat","centroid_lon",
]
STATIC_COLS = ["avg_burn_prob","whp","flep4","cfl","burnable"]


def load_rows_for_date(out: Path, slug: str, target_date, window_hour: int) -> pd.DataFrame:
    """Load all rows from train/val/test splits for the given date+window."""
    rows = []
    for split in ["train","val","test"]:
        p = out / f"{split}_{slug}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["date_utc"] = pd.to_datetime(df["date_utc"])
        mask = (df["date_utc"].dt.date == target_date) & (df["window_hour"] == window_hour)
        day = df[mask]
        if len(day) > 0:
            day = day.copy()
            day["_split"] = split
            rows.append(day)
            logger.info(f"  {split}: {len(day):,} rows for {target_date} {window_hour}Z")

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def find_good_dates(out: Path, slug: str, min_fires: int = 3) -> list:
    """Find dates with actual fires in test set to suggest to the user."""
    test_path = out / f"test_{slug}.parquet"
    if not test_path.exists():
        return []
    df = pd.read_parquet(test_path, columns=["date_utc","window_hour","label"])
    df["date_utc"] = pd.to_datetime(df["date_utc"])
    grp = df[df["label"]==1].groupby(["date_utc","window_hour"]).size().reset_index(name="n_fires")
    grp = grp[grp["n_fires"] >= min_fires].sort_values("n_fires", ascending=False)
    results = []
    for _, row in grp.head(10).iterrows():
        results.append((str(row["date_utc"].date()), int(row["window_hour"]), int(row["n_fires"])))
    return results


def predict(model: xgb.Booster, df: pd.DataFrame) -> np.ndarray:
    present = [c for c in FEATURE_COLS if c in df.columns]
    X = df[present].copy()
    for col in STATIC_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(0)
    for col in X.select_dtypes(include="object").columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(float)
    dm = xgb.DMatrix(X, feature_names=present, missing=np.nan)
    return model.predict(dm)


def risk_color(score: float) -> str:
    if score >= 0.80:   return "#FF0000"   # Critical
    elif score >= 0.65: return "#FF4500"   # High
    elif score >= 0.50: return "#FF8C00"   # Medium-high
    elif score >= 0.35: return "#FFD700"   # Medium
    elif score >= 0.20: return "#ADFF2F"   # Low-medium
    else:               return "#2E8B57"   # Low


def risk_label(score: float) -> str:
    if score >= 0.80:   return "🔴 CRITICAL"
    elif score >= 0.65: return "🟠 HIGH"
    elif score >= 0.50: return "🟡 MEDIUM-HIGH"
    elif score >= 0.35: return "🟡 MEDIUM"
    elif score >= 0.20: return "🟢 LOW-MEDIUM"
    else:               return "🟢 LOW"


def build_map(pred_df: pd.DataFrame, threshold: float,
              target_date, window_hour: int, state: str, topk: int) -> object:
    try:
        import folium
        import h3
    except ImportError:
        logger.error("Install: pip install folium h3")
        return None

    # Show top-K by risk score (so map always shows something)
    display_df = pred_df.nlargest(topk, "risk_score").copy()
    fire_df    = pred_df[pred_df["label"] == 1].copy()  # actual fires

    logger.info(f"  Displaying top {len(display_df):,} cells | "
                f"Actual fires: {len(fire_df):,} | "
                f"Predicted fires (>={threshold:.3f}): {int((pred_df['risk_score']>=threshold).sum()):,}")

    m = folium.Map(location=[31.0, -100.0], zoom_start=7, tiles=None)

    # Dark base map
    folium.TileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="Dark (default)",
        max_zoom=19,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)

    # ── Risk hexagons ──────────────────────────────────────────────────────────
    hex_group = folium.FeatureGroup(name="🔥 Risk Hexagons", show=True)

    for _, row in display_df.iterrows():
        cell  = row["h3_cell"]
        score = row["risk_score"]
        lat   = row.get("centroid_lat", None)
        lon   = row.get("centroid_lon", None)

        if pd.isna(lat) or pd.isna(lon):
            try:
                lat, lon = h3.cell_to_latlng(cell)
            except Exception:
                continue

        # Get H3 boundary polygon
        try:
            boundary = h3.cell_to_boundary(cell)
            coords = [[p[0], p[1]] for p in boundary]
        except Exception:
            try:
                boundary = h3.h3_to_geo_boundary(cell, geo_json=False)
                coords = [[p[0], p[1]] for p in boundary]
            except Exception:
                continue

        color   = risk_color(score)
        opacity = 0.25 + score * 0.65

        popup_html = f"""
        <div style='font-family:monospace;font-size:11px;min-width:210px;background:#111;color:#eee;padding:8px;border-radius:6px'>
          <b style='color:{color};font-size:13px'>{risk_label(score)}</b><br>
          <b>Risk Score:</b> {score:.4f}  (threshold={threshold:.3f})<br>
          <b>Label:</b> {'🔥 ACTUAL FIRE' if row.get('label',0)==1 else '✅ No fire'}<br>
          <hr style='border-color:#444;margin:4px 0'>
          <b>ERC 5D max:</b> {row.get('erc_5D_max', float('nan')):.1f} BTU/ft²<br>
          <b>Wind 5D max:</b> {row.get('vs_5D_max', float('nan')):.2f} m/s<br>
          <b>VPD:</b> {row.get('vpd', float('nan')):.2f} kPa<br>
          <b>FM100:</b> {row.get('fm100', float('nan')):.1f}%<br>
          <b>Temp max:</b> {row.get('tmmx', float('nan')):.1f} °C<br>
          <hr style='border-color:#444;margin:4px 0'>
          <b>Cell:</b> {cell}
        </div>"""

        folium.Polygon(
            locations=coords,
            color=color,
            weight=0.8,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            popup=folium.Popup(popup_html, max_width=270),
            tooltip=f"{risk_label(score)} | {score:.3f}",
        ).add_to(hex_group)

    hex_group.add_to(m)

    # ── Actual fire markers ────────────────────────────────────────────────────
    if len(fire_df) > 0:
        fire_group = folium.FeatureGroup(name="★ Actual Fires (FPA-FOD)", show=True)
        for _, row in fire_df.iterrows():
            lat = row.get("centroid_lat", None)
            lon = row.get("centroid_lon", None)
            if pd.isna(lat) or pd.isna(lon):
                continue
            folium.CircleMarker(
                location=[lat, lon],
                radius=7,
                color="white",
                weight=2,
                fill=True,
                fill_color="#FF0000",
                fill_opacity=1.0,
                popup=f"<b>ACTUAL FIRE</b><br>Score: {row.get('risk_score', 0):.4f}<br>Cell: {row['h3_cell']}",
                tooltip=f"★ Actual Fire | Score={row.get('risk_score',0):.3f}",
            ).add_to(fire_group)
        fire_group.add_to(m)

    # ── Legend ─────────────────────────────────────────────────────────────────
    window_names = {0:"Midnight–6am UTC", 6:"6am–Noon UTC", 12:"Noon–6pm UTC", 18:"6pm–Midnight UTC"}
    n_predicted = int((pred_df["risk_score"] >= threshold).sum())
    n_fires     = int((pred_df["label"] == 1).sum())

    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:rgba(10,10,10,0.92);color:#eee;padding:14px 18px;
                border-radius:10px;font-family:monospace;font-size:12px;
                border:1px solid #FF4500;min-width:220px;line-height:1.7">
      <b style='font-size:14px;color:#FF6347'>🔥 Texas Fire Risk</b><br>
      <span style='color:#aaa'>Date:</span> <b>{target_date}</b><br>
      <span style='color:#aaa'>Window:</span> <b>{window_hour}Z</b> — {window_names.get(window_hour,'')}<br>
      <span style='color:#aaa'>Rows on map:</span> <b>Top {topk:,}</b><br>
      <span style='color:#aaa'>Predicted fires:</span> <b style='color:#FF4500'>{n_predicted:,}</b><br>
      <span style='color:#aaa'>Actual fires:</span> <b style='color:#FF0000'>{n_fires:,}</b><br>
      <hr style='border-color:#444;margin:5px 0'>
      <span style='color:#FF0000'>■</span> Critical (&gt;0.80)<br>
      <span style='color:#FF4500'>■</span> High (0.65–0.80)<br>
      <span style='color:#FF8C00'>■</span> Med-High (0.50–0.65)<br>
      <span style='color:#FFD700'>■</span> Medium (0.35–0.50)<br>
      <span style='color:#ADFF2F'>■</span> Low-Med (0.20–0.35)<br>
      <span style='color:#2E8B57'>■</span> Low (&lt;0.20)<br>
      <span style='color:white'>★</span> Actual Fire (FPA-FOD)<br>
      <hr style='border-color:#444;margin:5px 0'>
      <small style='color:#888'>Click hex for weather details<br>
      Model: XGBoost Baseline v1</small>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Title bar ──────────────────────────────────────────────────────────────
    title_html = f"""
    <div style="position:fixed;top:12px;left:50%;transform:translateX(-50%);
                z-index:9999;background:rgba(10,10,10,0.88);color:white;
                padding:9px 22px;border-radius:8px;font-family:sans-serif;
                border:1px solid #FF4500;text-align:center">
      <b style='font-size:14px'>🔥 IgnitionNet — {state} Wildfire Risk</b><br>
      <span style='font-size:11px;color:#aaa'>{target_date} | {window_hour}Z UTC | XGBoost Baseline (no LANDFIRE)</span>
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state",  default="TX")
    parser.add_argument("--date",   default=None,
                        help="Date to map (YYYY-MM-DD). If not given, auto-selects highest-fire date.")
    parser.add_argument("--window", default=18, type=int, choices=[0,6,12,18])
    parser.add_argument("--topk",   default=5000, type=int,
                        help="Top K cells to display on map (by risk score)")
    args = parser.parse_args()

    s    = args.state.upper()
    cfg  = STATE_CONFIG[s]
    out  = cfg["output_dir"]
    slug = s.lower()
    map_dir = out / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    model_path = out / "models" / f"xgb_baseline_{slug}.ubj"
    meta_path  = out / "models" / f"xgb_baseline_{slug}_meta.json"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}  — run Phase 3 first")
        return

    model = xgb.Booster()
    model.load_model(str(model_path))
    with open(meta_path) as f:
        meta = json.load(f)
    threshold = meta["threshold"]

    # ── Auto-select date if not given ───────────────────────────────────────
    if args.date is None:
        logger.info("  Auto-selecting best date (most fires in test split)...")
        good = find_good_dates(out, slug)
        if good:
            best_date, best_window, n_fires = good[0]
            logger.info(f"  Best date: {best_date} window={best_window}Z ({n_fires} fires)")
            target_date = datetime.strptime(best_date, "%Y-%m-%d").date()
            window_hour = best_window
        else:
            logger.error("No good dates found in test split")
            return
    else:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        window_hour = args.window

    logger.info(f"\nModel loaded | date={target_date} | window={window_hour}Z | threshold={threshold:.4f}")

    # ── Print available good dates for user ─────────────────────────────────
    logger.info("\n  Top dates with fires in TEST set (2019-2020):")
    for d, w, n in find_good_dates(out, slug):
        logger.info(f"    {d}  {w}Z  →  {n} fires")

    # ── Load rows ────────────────────────────────────────────────────────────
    logger.info(f"\n[1/3] Loading rows for {target_date} {window_hour}Z...")
    day_df = load_rows_for_date(out, slug, target_date, window_hour)

    if len(day_df) == 0:
        logger.error(f"No data found for {target_date} {window_hour}Z")
        logger.error("Try one of the dates listed above with --date YYYY-MM-DD --window W")
        return

    logger.info(f"  Total rows: {len(day_df):,}  "
                f"| Fires: {int((day_df['label']==1).sum()):,}  "
                f"| Non-fire: {int((day_df['label']==0).sum()):,}")

    # ── Predict ──────────────────────────────────────────────────────────────
    logger.info("\n[2/3] Generating predictions...")
    probs = predict(model, day_df)
    day_df = day_df.copy()
    day_df["risk_score"] = probs
    logger.info(f"  Scores: min={probs.min():.4f}  max={probs.max():.4f}  "
                f"mean={probs.mean():.4f}  above-threshold={int((probs>=threshold).sum()):,}")

    # ── Build map ─────────────────────────────────────────────────────────────
    logger.info(f"\n[3/3] Building map (top {args.topk:,} cells)...")
    m = build_map(day_df, threshold, target_date, window_hour, s, args.topk)
    if m is None:
        return

    # Save CSV
    csv_path = map_dir / f"predictions_{s}_{target_date}_{window_hour}Z.csv"
    (day_df[["h3_cell","centroid_lat","centroid_lon","risk_score","label"]]
     .sort_values("risk_score", ascending=False)
     .to_csv(csv_path, index=False))

    # Save HTML
    html_path = map_dir / f"risk_map_{s}_{target_date}_{window_hour}Z.html"
    m.save(str(html_path))

    logger.info("\n" + "=" * 65)
    logger.info(f"  ✔ MAP SAVED → Open in Chrome/Edge:")
    logger.info(f"  {html_path}")
    logger.info(f"  ✔ CSV SAVED: {csv_path}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
