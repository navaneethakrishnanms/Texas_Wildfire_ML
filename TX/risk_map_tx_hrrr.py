"""
risk_map_tx_hrrr.py
--------------------
Texas Wildfire Risk Map — H3-native, zoom-adaptive (HRRR model)

The TX/LANDFIRE+HRRR equivalent of V2/phase2/run_phase3_risk_map_v2.py.
Renders scored H3 cells as a true H3 layer in the browser (h3-js + Leaflet):
hexagon resolution follows the zoom level, and every hex / fire marker is
clickable for full detail — now including the HRRR sub-daily fields the
V2 map did not have.

What changed vs the V2 script:
  - reads data/hrrr/hrrr_tx_all.parquet (one file, has _split) instead of
    three per-split parquets
  - scores with the HRRR model, taking the feature list + threshold straight
    from the model's meta JSON so the map can never drift from training
  - detail popups show HRRR (temp/RH/wind/VPD/PBL/solar) alongside gridMET
    and the LANDFIRE landscape features, plus whether HRRR was actually
    available for that cell (hrrr_pw)

Coarser zoom levels aggregate the scored leaf (res 8) cells up to their H3
parents (res 4-7), same mechanism as the V2 map — only the sampled cells
present in the dataset for a given date/window are scored, not the full
~1.17M-cell TX grid.

Usage:
    python risk_map_tx_hrrr.py                                  # auto-pick biggest fire day
    python risk_map_tx_hrrr.py --date 2020-05-19 --window 18
    python risk_map_tx_hrrr.py --list-dates                     # show candidate dates
    python risk_map_tx_hrrr.py --top 5                          # build 5 maps at once
    python risk_map_tx_hrrr.py --model xgb_tx_hrrr
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT    = Path(__file__).resolve().parent
HRRR_PQ = ROOT / "data" / "hrrr" / "hrrr_tx_all.parquet"
OUT_DIR = ROOT / "outputs" / "texas_landfire"
MODELS  = OUT_DIR / "models"
MAP_DIR = OUT_DIR / "maps"
MAP_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RESOLUTIONS = [4, 5, 6, 7, 8]
WINDOW_NAMES = {0: "Midnight-6am UTC", 6: "6am-Noon UTC",
                12: "Noon-6pm UTC", 18: "6pm-Midnight UTC"}

# Fields carried into the leaf-cell popups
GRIDMET_COLS   = ["erc", "erc_5D_max", "vs_5D_max", "vpd", "fm100", "tmmx", "rmin"]
HRRR_COLS      = ["temp_pw", "rh_pw", "wind_pw", "vpd_pw_hrrr", "hpbl_pw", "dswrf_pw"]
LANDSCAPE_COLS = ["avg_burn_prob", "whp", "cfl", "cbd", "cbh", "burnable"]
FLAG_COLS      = ["hrrr_pw", "hrrr_rh_valid", "gridmet_missing"]
POPUP_COLS     = GRIDMET_COLS + HRRR_COLS + LANDSCAPE_COLS + FLAG_COLS


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return None
        return round(float(v), 3)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v


# ── Model + data ──────────────────────────────────────────────────────────────
def load_model(model_name: str):
    model_path = MODELS / f"{model_name}.ubj"
    meta_path  = MODELS / f"{model_name}_meta.json"
    if not model_path.exists():
        log.error(f"Model not found: {model_path}")
        log.error(f"Available: {[p.stem for p in MODELS.glob('*.ubj')]}")
        sys.exit(1)
    model = xgb.Booster()
    model.load_model(str(model_path))
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    features = meta.get("features") or model.feature_names
    log.info(f"Model: {model_name} | {len(features)} features | "
             f"threshold={meta['threshold']:.4f}")
    return model, meta, features


def predict(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = df[features].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(np.float32)
    dm = xgb.DMatrix(X, feature_names=features, missing=np.nan)
    return model.predict(dm)


def find_good_dates(df: pd.DataFrame, min_fires: int = 3, n: int = 10) -> list:
    """Dates/windows in the TEST split with the most real fires."""
    fires = df[(df["_split"] == "test") & (df["label"] == 1)].copy()
    fires["fire_date"] = fires["date_utc"].dt.date
    grp = (fires.groupby(["fire_date", "window_hour"])
           .size().reset_index(name="n_fires"))
    grp = grp[grp["n_fires"] >= min_fires].sort_values("n_fires", ascending=False)
    return [(str(r.fire_date), int(r.window_hour), int(r.n_fires))
            for r in grp.head(n).itertuples(index=False)]


def build_multires_cells(pred_df: pd.DataFrame) -> dict:
    """Roll scored leaf (res 8) cells up to every coarser resolution."""
    import h3
    out = {}
    for r in RESOLUTIONS:
        if r == 8:
            recs = []
            for row in pred_df.itertuples(index=False):
                rec = {"h": row.h3_cell, "s": _clean(row.risk_score),
                       "f": int(getattr(row, "label", 0) or 0), "n": 1}
                for c in POPUP_COLS:
                    rec[c] = _clean(getattr(row, c, None))
                recs.append(rec)
            out[r] = recs
            continue
        tmp = pred_df[["h3_cell", "risk_score", "label"]].copy()
        tmp["parent"] = tmp["h3_cell"].apply(lambda c: h3.cell_to_parent(c, r))
        grp = tmp.groupby("parent").agg(
            s=("risk_score", "mean"), f=("label", "sum"), n=("risk_score", "size"),
        ).reset_index()
        out[r] = [{"h": row.parent, "s": _clean(row.s), "f": int(row.f), "n": int(row.n)}
                  for row in grp.itertuples(index=False)]
    return out


def build_fire_events(pred_df: pd.DataFrame) -> list:
    fires = pred_df[pred_df["label"] == 1]
    events = []
    for row in fires.itertuples(index=False):
        rec = {"h": row.h3_cell, "s": _clean(row.risk_score)}
        for c in POPUP_COLS:
            rec[c] = _clean(getattr(row, c, None))
        events.append(rec)
    return events


# ── HTML template ─────────────────────────────────────────────────────────────
TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<script src="https://unpkg.com/h3-js@4.1.0/dist/h3-js.umd.js"></script>
<style>
  :root{
    --bg:#0a0a0a; --panel:#111214cc; --border:#ff4500; --text:#eee; --muted:#9aa0a6;
  }
  html,body{height:100%;margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#000;}
  #map{position:absolute;inset:0;}
  .panel{
    background:var(--panel); color:var(--text); backdrop-filter:blur(6px);
    border:1px solid var(--border); border-radius:10px; box-shadow:0 4px 18px rgba(0,0,0,.45);
  }
  .title-ctl{ padding:9px 22px; text-align:center; font-size:12px; }
  .title-ctl b{font-size:14px;}
  .title-ctl .sub{color:var(--muted); font-size:11px; display:block; margin-top:2px;}
  .res-badge{ padding:6px 12px; font-family:ui-monospace,Consolas,monospace; font-size:12px; font-weight:600; }
  .theme-toggle{ width:34px;height:34px;display:flex;align-items:center;justify-content:center;
                 font-size:17px;cursor:pointer;user-select:none; }
  .theme-toggle:hover{ filter:brightness(1.2); }
  .legend{ padding:12px 16px; font-family:ui-monospace,Consolas,monospace; font-size:11.5px; line-height:1.75; min-width:225px; }
  .legend b.h{font-size:13px;color:#ff6347;}
  .legend .row span.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:middle;}
  .legend hr{border:none;border-top:1px solid #444;margin:6px 0;}
  .legend .muted{color:var(--muted);}
  .detail-card{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; min-width:262px; max-height:420px; overflow-y:auto; color:#222; }
  .detail-card .hd{display:flex;align-items:center;gap:7px;font-size:15px;font-weight:700;margin-bottom:6px;}
  .detail-card .sec{margin-top:9px;margin-bottom:3px;font-size:10.5px;font-weight:700;
                    letter-spacing:.6px;text-transform:uppercase;color:#8a8a8a;
                    border-bottom:1px solid #e4e4e4;padding-bottom:2px;}
  .detail-card .row{display:flex;justify-content:space-between;gap:14px;padding:2.5px 0;font-size:12.5px;}
  .detail-card .row span:first-child{color:#666;white-space:nowrap;}
  .detail-card .row span:last-child{font-weight:600;text-align:right;}
  .detail-card .foot{color:#999;font-size:10.5px;margin-top:8px;line-height:1.4;}
  .detail-card .na{color:#bbb;font-weight:400;}
  .leaflet-popup-content-wrapper{border-radius:8px;}
  body.light-mode .panel{ background:#ffffffdd; color:#1a1a1a; border-color:#e0641f; }
  body.light-mode .legend .muted, body.light-mode .title-ctl .sub{ color:#555; }
  body.light-mode .legend hr{ border-top-color:#ccc; }
</style>
</head>
<body>
<div id="map"></div>
<script>
const HEX_DATA  = __DATA_JSON__;
const FIRE_DATA = __FIRES_JSON__;
const META      = __META_JSON__;

const map = L.map('map', { zoomControl: true, minZoom: 2, maxZoom: 14, worldCopyJump: true })
  .setView([31.0, -100.0], 7);

const dark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19,
}).addTo(map);
const light = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19,
});

let theme = 'dark';
function setTheme(next) {
  if (next === theme) return;
  theme = next;
  if (theme === 'light') { map.removeLayer(dark); map.addLayer(light); }
  else { map.removeLayer(light); map.addLayer(dark); }
  document.body.classList.toggle('light-mode', theme === 'light');
  themeToggle.setIcon(theme);
  renderHexes(true);
}

const EDGE_KM = { 0:1107.71, 1:418.68, 2:158.24, 3:59.81, 4:22.606, 5:8.544, 6:3.229, 7:1.221, 8:0.4614 };
const SUPPORTED_RES = [0,1,2,3,4,5,6,7,8];
const TARGET_HEX_PX = 60;

function metersPerPixel(lat, zoom) {
  return 156543.03392 * Math.cos(lat * Math.PI / 180) / Math.pow(2, zoom);
}
function resForView(zoom, lat) {
  const mpp = metersPerPixel(lat, zoom);
  const targetEdgeKm = (TARGET_HEX_PX * mpp) / 1000 / 2;
  let best = SUPPORTED_RES[0], bestDiff = Infinity;
  for (const r of SUPPORTED_RES) {
    const diff = Math.abs(EDGE_KM[r] - targetEdgeKm);
    if (diff < bestDiff) { bestDiff = diff; best = r; }
  }
  return best;
}

// Fire-danger ramp (NFDRS-style domain convention: green -> yellow -> red).
function riskColor(s) {
  if (s >= 0.80) return '#ff0000';
  if (s >= 0.65) return '#ff4500';
  if (s >= 0.50) return '#ff8c00';
  if (s >= 0.35) return '#ffd700';
  if (s >= 0.20) return '#adff2f';
  return '#2e8b57';
}
function riskLabel(s) {
  if (s >= 0.80) return 'CRITICAL';
  if (s >= 0.65) return 'HIGH';
  if (s >= 0.50) return 'MEDIUM-HIGH';
  if (s >= 0.35) return 'MEDIUM';
  if (s >= 0.20) return 'LOW-MEDIUM';
  return 'LOW';
}
function fmt(v, suffix) {
  if (v === null || v === undefined) return '<span class="na">N/A</span>';
  return v + (suffix || '');
}
function row(label, v, suffix) {
  return `<div class="row"><span>${label}</span><span>${fmt(v, suffix)}</span></div>`;
}

// Full detail card for a leaf (res 8) cell: risk, HRRR, gridMET, landscape.
function leafCard(rec, h, color) {
  const hrrrOk = rec.hrrr_pw === 1;
  const rhOk   = rec.hrrr_rh_valid === 1;
  return `<div class="detail-card">
    <div class="hd" style="color:${color}">&#128293; ${riskLabel(rec.s)}</div>
    ${row('Predicted risk', (rec.s*100).toFixed(1) + '%')}
    ${row('Actual fire recorded', rec.f ? 'Yes' : 'No')}

    <div class="sec">HRRR &mdash; sub-daily (${hrrrOk ? 'available' : 'not available'})</div>
    ${row('Temperature (2m)', rec.temp_pw, ' &deg;C')}
    ${row('Relative humidity', rhOk ? rec.rh_pw : null, ' %')}
    ${row('Wind speed (10m)', rec.wind_pw, ' m/s')}
    ${row('VPD (HRRR)', rhOk ? rec.vpd_pw_hrrr : null, ' kPa')}
    ${row('Boundary layer height', rec.hpbl_pw, ' m')}
    ${row('Solar radiation', rec.dswrf_pw, ' W/m&sup2;')}

    <div class="sec">gridMET &mdash; daily</div>
    ${row('ERC', rec.erc, ' BTU/ft&sup2;')}
    ${row('ERC (5-day max)', rec.erc_5D_max, ' BTU/ft&sup2;')}
    ${row('Wind (5-day max)', rec.vs_5D_max, ' m/s')}
    ${row('VPD', rec.vpd, ' kPa')}
    ${row('Fuel moisture (FM100)', rec.fm100, ' %')}
    ${row('Min relative humidity', rec.rmin, ' %')}
    ${row('Max temperature', rec.tmmx, ' &deg;C')}

    <div class="sec">Landscape &mdash; LANDFIRE / TxWRAP</div>
    ${row('Burn probability', rec.avg_burn_prob, ' / 11')}
    ${row('Wildfire hazard potential', rec.whp, ' / 9')}
    ${row('Canopy flame length', rec.cfl, ' ft')}
    ${row('Canopy bulk density', rec.cbd, ' kg/m&sup3;')}
    ${row('Canopy base height', rec.cbh, ' m')}
    ${row('Burnable', rec.burnable === 1 ? 'Yes' : (rec.burnable === 0 ? 'No' : null))}

    <div class="sec">Cell</div>
    <div class="row"><span>H3 cell (res 8)</span><span style="font-size:10px">${h}</span></div>
    <div class="foot">Model: ${META.model_name}. HRRR fields show N/A where the
    archive had no matching timestamp; the model handles those as missing rather
    than imputing them.</div>
  </div>`;
}

const hexLayer = L.layerGroup().addTo(map);
const MAX_CELLS = 6000;

const dataByRes = {};
for (const r of SUPPORTED_RES) {
  const m = {};
  for (const rec of (HEX_DATA[String(r)] || [])) m[rec.h] = rec;
  dataByRes[r] = m;
}

let lastKey = null;
function clampLat(v) { return Math.max(-89.9, Math.min(89.9, v)); }
function wrapLng(v) { return ((v + 180) % 360 + 360) % 360 - 180; }

function renderHexes(force) {
  const z = map.getZoom();
  const center = map.getCenter();
  const res = resForView(z, center.lat);
  const b = map.getBounds().pad(0.3);
  const south = clampLat(b.getSouth()), north = clampLat(b.getNorth());
  let west = wrapLng(b.getWest()), east = wrapLng(b.getEast());
  if (east <= west) east += 360;
  const key = `${res}|${south.toFixed(2)}|${west.toFixed(2)}|${north.toFixed(2)}|${east.toFixed(2)}`;
  if (!force && key === lastKey) return;
  lastKey = key;
  hexLayer.clearLayers();

  const ring = [[south,west],[north,west],[north,east],[south,east],[south,west]];
  let cells;
  try { cells = h3.polygonToCells(ring, res, false); } catch (e) { cells = []; }
  const truncated = cells.length > MAX_CELLS;
  if (truncated) cells = cells.slice(0, MAX_CELLS);

  const lookup = dataByRes[res] || {};
  const neutralFill   = theme === 'light' ? 'rgba(20,20,20,0.03)' : 'rgba(255,255,255,0.04)';
  const neutralBorder = theme === 'light' ? 'rgba(20,20,20,0.35)' : 'rgba(255,255,255,0.30)';
  const isLeaf = (res === 8);

  for (const h of cells) {
    let boundary;
    try { boundary = h3.cellToBoundary(h, false); } catch (e) { continue; }
    const rec = lookup[h];
    const hasData = !!rec;
    const color = hasData ? riskColor(rec.s) : neutralBorder;
    const poly = L.polygon(boundary, {
      color,
      weight: hasData ? (isLeaf ? 0.9 : 1.2) : 0.6,
      fillColor: hasData ? riskColor(rec.s) : neutralFill,
      fillOpacity: hasData ? (0.22 + rec.s * 0.62) : 1,
      opacity: hasData ? 1 : 0.9,
    });

    let html;
    if (hasData && isLeaf) {
      html = leafCard(rec, h, color);
    } else if (hasData) {
      html = `<div class="detail-card">
        <div class="hd" style="color:${color}">&#128293; ${riskLabel(rec.s)} (aggregated)</div>
        ${row('Mean predicted risk', (rec.s*100).toFixed(1) + '%')}
        ${row('Sampled cells here', rec.n)}
        ${row('Fire events among them', rec.f)}
        <div class="row"><span>H3 cell (res ${res})</span><span style="font-size:10px">${h}</span></div>
        <div class="foot">Zoom in for cell-level detail including HRRR fields.</div>
      </div>`;
    } else {
      html = `<div class="detail-card">
        <div class="hd" style="color:#888">No risk data</div>
        <div class="row"><span>H3 cell (res ${res})</span><span style="font-size:10px">${h}</span></div>
      </div>`;
    }
    poly.bindPopup(html, { maxWidth: 320 });
    if (hasData) poly.bindTooltip(`${riskLabel(rec.s)} | ${rec.s.toFixed(3)}`);
    poly.addTo(hexLayer);
  }
  resBadge.update(res, z, cells.length, truncated);
}

const fireLayer = L.layerGroup().addTo(map);
for (const f of FIRE_DATA) {
  let lat, lon;
  try { [lat, lon] = h3.cellToLatLng(f.h); } catch (e) { continue; }
  const marker = L.circleMarker([lat, lon], {
    radius: 7, color: '#fff', weight: 2, fillColor: '#ff0000', fillOpacity: 1,
  });
  const hrrrOk = f.hrrr_pw === 1, rhOk = f.hrrr_rh_valid === 1;
  marker.bindPopup(`<div class="detail-card">
    <div class="hd" style="color:#c0392b">&#128293; Fire Event</div>
    ${row('Date', META.date)}
    ${row('Window', META.window_label)}
    ${row('Model predicted risk', (f.s*100).toFixed(1) + '%')}
    ${row('Flagged at threshold?', f.s >= META.threshold ? 'Yes (true positive)' : 'No (missed)')}
    <div class="sec">HRRR (${hrrrOk ? 'available' : 'not available'})</div>
    ${row('Temperature', f.temp_pw, ' &deg;C')}
    ${row('Relative humidity', rhOk ? f.rh_pw : null, ' %')}
    ${row('Wind speed', f.wind_pw, ' m/s')}
    ${row('VPD (HRRR)', rhOk ? f.vpd_pw_hrrr : null, ' kPa')}
    <div class="sec">gridMET</div>
    ${row('ERC (5-day max)', f.erc_5D_max, ' BTU/ft&sup2;')}
    ${row('Wind (5-day max)', f.vs_5D_max, ' m/s')}
    ${row('VPD', f.vpd, ' kPa')}
    ${row('Fuel moisture', f.fm100, ' %')}
    <div class="sec">Landscape</div>
    ${row('Burn probability', f.avg_burn_prob, ' / 11')}
    ${row('Wildfire hazard potential', f.whp, ' / 9')}
    <div class="row"><span>H3 cell</span><span style="font-size:10px">${f.h}</span></div>
    <div class="foot">Source: FPA-FOD training label. Fire size, cause and county
    are not in this dataset.</div>
  </div>`, { maxWidth: 320 });
  marker.bindTooltip('Fire event');
  marker.addTo(fireLayer);
}

map.on('moveend', () => renderHexes(false));
renderHexes(true);

const Title = L.Control.extend({
  options: { position: 'topright' },
  onAdd: function () {
    const div = L.DomUtil.create('div', 'panel title-ctl');
    div.innerHTML = `<b>&#128293; IgnitionNet &mdash; ${META.state} Wildfire Risk (HRRR)</b>
      <span class="sub">${META.date} | ${META.window_label} | ${META.model_name}</span>`;
    return div;
  }
});
map.addControl(new Title());

const ThemeToggle = L.Control.extend({
  options: { position: 'topright' },
  onAdd: function () {
    const div = L.DomUtil.create('div', 'panel theme-toggle');
    this._div = div;
    this.setIcon('dark');
    L.DomEvent.disableClickPropagation(div);
    L.DomEvent.on(div, 'click', () => setTheme(theme === 'dark' ? 'light' : 'dark'));
    return div;
  },
  setIcon: function (t) {
    if (!this._div) return;
    this._div.innerHTML = t === 'dark' ? '&#9728;&#65039;' : '&#127769;';
    this._div.title = t === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
  }
});
const themeToggle = new ThemeToggle();
map.addControl(themeToggle);

const ResBadge = L.Control.extend({
  options: { position: 'topright' },
  onAdd: function () {
    this._div = L.DomUtil.create('div', 'panel res-badge');
    this.update(8, map.getZoom(), 0, false);
    return this._div;
  },
  update: function (res, zoom, count, truncated) {
    if (!this._div) return;
    this._div.innerHTML = `H3 Resolution: ${res} &nbsp;&middot;&nbsp; Zoom: ${zoom}
      &nbsp;&middot;&nbsp; Cells: ${truncated ? count + '+ (capped)' : count}`;
  }
});
const resBadge = new ResBadge();
map.addControl(resBadge);

const Legend = L.Control.extend({
  options: { position: 'bottomleft' },
  onAdd: function () {
    const div = L.DomUtil.create('div', 'panel legend');
    div.innerHTML = `
      <b class="h">&#128293; ${META.state} Fire Risk</b><br>
      <span class="muted">Date:</span> <b>${META.date}</b><br>
      <span class="muted">Window:</span> <b>${META.window_label}</b><br>
      <span class="muted">Sampled cells:</span> <b>${META.n_total}</b><br>
      <span class="muted">Actual fires:</span> <b style="color:#ff0000">${META.n_fires}</b><br>
      <span class="muted">HRRR coverage:</span> <b>${META.hrrr_coverage}</b><br>
      <span class="muted">Threshold:</span> <b>${META.threshold}</b><br>
      <hr>
      <div class="row"><span class="sw" style="background:#ff0000"></span>Critical (&gt;0.80)</div>
      <div class="row"><span class="sw" style="background:#ff4500"></span>High (0.65-0.80)</div>
      <div class="row"><span class="sw" style="background:#ff8c00"></span>Med-High (0.50-0.65)</div>
      <div class="row"><span class="sw" style="background:#ffd700"></span>Medium (0.35-0.50)</div>
      <div class="row"><span class="sw" style="background:#adff2f"></span>Low-Med (0.20-0.35)</div>
      <div class="row"><span class="sw" style="background:#2e8b57"></span>Low (&lt;0.20)</div>
      <div class="row"><span class="sw" style="background:transparent;border:1px solid #888"></span>No data (grid only)</div>
      <hr>
      <span class="muted">Pan/zoom changes H3 resolution.<br>Click any hex or marker for detail.</span>`;
    return div;
  }
});
map.addControl(new Legend());

L.control.layers(null,
  { 'Risk Hexagons (H3)': hexLayer, 'Fire Events': fireLayer },
  { collapsed: false, position: 'topleft' }
).addTo(map);
</script>
</body>
</html>
"""


def build_html(data_by_res: dict, fires: list, meta: dict) -> str:
    return (TEMPLATE
            .replace("__TITLE__", f"IgnitionNet {meta['state']} Wildfire Risk (HRRR) — {meta['date']}")
            .replace("__DATA_JSON__", json.dumps({str(k): v for k, v in data_by_res.items()}))
            .replace("__FIRES_JSON__", json.dumps(fires))
            .replace("__META_JSON__", json.dumps(meta)))


def build_one(df, model, features, threshold, model_name, target_date, window_hour):
    mask = (df["date_utc"].dt.date == target_date) & (df["window_hour"] == window_hour)
    day = df[mask].copy()
    if len(day) == 0:
        log.error(f"No rows for {target_date} {window_hour}Z")
        return None

    day["risk_score"] = predict(model, day, features)
    n_fires = int((day["label"] == 1).sum())
    cov = 100 * (day["hrrr_pw"] == 1).mean()
    log.info(f"  {target_date} {window_hour:>2}Z | cells={len(day):,} | "
             f"fires={n_fires} | HRRR={cov:.0f}% | "
             f"mean risk={day['risk_score'].mean():.3f}")

    meta = {
        "state": "TX",
        "date": str(target_date),
        "window_label": f"{window_hour}Z ({WINDOW_NAMES.get(window_hour, '')})",
        "model_name": model_name,
        "threshold": round(float(threshold), 4),
        "n_total": int(len(day)),
        "n_fires": n_fires,
        "hrrr_coverage": f"{cov:.0f}%",
    }
    html = build_html(build_multires_cells(day), build_fire_events(day), meta)

    stem = f"risk_map_h3_TX_hrrr_{target_date}_{window_hour}Z"
    html_path = MAP_DIR / f"{stem}.html"
    html_path.write_text(html, encoding="utf-8")

    csv_cols = (["h3_cell", "centroid_lat", "centroid_lon", "risk_score", "label"]
                + [c for c in POPUP_COLS if c in day.columns])
    (day[csv_cols].sort_values("risk_score", ascending=False)
     .to_csv(MAP_DIR / f"{stem}.csv", index=False))

    log.info(f"    -> {html_path.name}  +  {stem}.csv")
    return html_path


def main():
    ap = argparse.ArgumentParser(description="TX HRRR H3 risk map")
    ap.add_argument("--model", default="xgb_tx_hrrr_tuned",
                    help="Model stem (default: xgb_tx_hrrr_tuned)")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: auto-pick)")
    ap.add_argument("--window", type=int, default=18, choices=[0, 6, 12, 18])
    ap.add_argument("--top", type=int, default=1,
                    help="Build maps for the N highest-fire test dates (default 1)")
    ap.add_argument("--list-dates", action="store_true",
                    help="List candidate high-fire dates and exit")
    args = ap.parse_args()

    if not HRRR_PQ.exists():
        log.error(f"{HRRR_PQ} not found — run merge_hrrr_duckdb.py first.")
        sys.exit(1)

    try:
        import h3  # noqa: F401
    except ImportError:
        log.error("Python package 'h3' is required (pip install h3).")
        sys.exit(1)

    df = pd.read_parquet(HRRR_PQ)
    df["date_utc"] = pd.to_datetime(df["date_utc"])
    log.info(f"Data: {len(df):,} rows")

    if args.list_dates:
        log.info("Candidate dates (TEST split, most fires):")
        for d, w, n in find_good_dates(df, n=20):
            log.info(f"  {d}  {w:>2}Z   fires={n}")
        return

    model, meta, features = load_model(args.model)
    threshold = meta["threshold"]

    if args.date:
        targets = [(datetime.strptime(args.date, "%Y-%m-%d").date(), args.window)]
    else:
        good = find_good_dates(df, n=max(args.top, 1))
        if not good:
            log.error("No high-fire dates found in TEST split.")
            sys.exit(1)
        targets = [(datetime.strptime(d, "%Y-%m-%d").date(), w)
                   for d, w, _ in good[:args.top]]
        log.info(f"Auto-selected {len(targets)} date(s) with the most fires.")

    log.info("Building maps...")
    built = [p for t in targets
             if (p := build_one(df, model, features, threshold, args.model, *t))]

    log.info("=" * 65)
    log.info(f"  {len(built)} map(s) saved to: {MAP_DIR}")
    for p in built:
        log.info(f"  open  {p}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
