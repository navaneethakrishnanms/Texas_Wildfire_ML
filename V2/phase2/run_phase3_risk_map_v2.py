"""
Phase 3 — Texas Wildfire Risk Map (H3-native, zoom-adaptive)
==============================================================
Renders risk cells as a true H3 layer in the browser (h3-js + Leaflet),
matching the h3geo.org interaction model: hexagon resolution changes as
you zoom, and every hex/fire marker is clickable for full detail.

Unlike the original run_phase3_risk_map.py (which baked ~5,000 static
folium.Polygon objects into the HTML at a single fixed resolution), this
script exports the scored cells as compact JSON and lets the browser:
  - pick an H3 resolution from the current Leaflet zoom level
  - compute hex boundaries on the fly via h3-js (h3.cellToBoundary)
  - re-render the hex layer on every zoomend
  - show a detail popup on click for both hexes and fire markers

Data reality check: the training pipeline only scores the sampled cells
present in the dataset for a given date/window (a few hundred, not the
full ~1.17M-cell TX grid), so coarser zoom levels are built by
aggregating those sampled leaf (res 8) cells up to their H3 parents
(res 4-7) rather than by re-sampling a denser source. This is the same
mechanism a live/denser data feed would use later — only the input
density changes.

Run:  python run_phase3_risk_map_v2.py --state TX --date 2019-07-04 --window 12
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
from run_phase3_risk_map import load_rows_for_date, find_good_dates, predict

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

RESOLUTIONS = [4, 5, 6, 7, 8]           # coarsest -> finest (8 = leaf / dataset resolution)
WEATHER_COLS = ["erc_5D_max", "vs_5D_max", "vpd", "fm100", "tmmx"]
WINDOW_NAMES = {0: "Midnight-6am UTC", 6: "6am-Noon UTC", 12: "Noon-6pm UTC", 18: "6pm-Midnight UTC"}


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


def build_multires_cells(pred_df: pd.DataFrame) -> dict:
    """Aggregate scored leaf (res 8) cells up to every coarser resolution
    in RESOLUTIONS using H3 parent rollups, so the browser can pick the
    right tier for the current zoom level without recomputation."""
    import h3

    out = {}
    for r in RESOLUTIONS:
        if r == 8:
            recs = []
            for row in pred_df.itertuples(index=False):
                rec = {
                    "h": row.h3_cell,
                    "s": _clean(row.risk_score),
                    "f": int(getattr(row, "label", 0) or 0),
                    "n": 1,
                }
                for c in WEATHER_COLS:
                    rec[c] = _clean(getattr(row, c, None))
                recs.append(rec)
            out[r] = recs
            continue

        tmp = pred_df[["h3_cell", "risk_score", "label"]].copy()
        tmp["parent"] = tmp["h3_cell"].apply(lambda c: h3.cell_to_parent(c, r))
        grp = tmp.groupby("parent").agg(
            s=("risk_score", "mean"),
            f=("label", "sum"),
            n=("risk_score", "size"),
        ).reset_index()
        out[r] = [
            {"h": row.parent, "s": _clean(row.s), "f": int(row.f), "n": int(row.n)}
            for row in grp.itertuples(index=False)
        ]
    return out


def build_fire_events(pred_df: pd.DataFrame) -> list:
    fire_df = pred_df[pred_df["label"] == 1]
    events = []
    for row in fire_df.itertuples(index=False):
        rec = {"h": row.h3_cell, "s": _clean(row.risk_score)}
        for c in WEATHER_COLS:
            rec[c] = _clean(getattr(row, c, None))
        events.append(rec)
    return events


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
    --crit:#ff0000; --high:#ff4500; --medhigh:#ff8c00; --med:#ffd700; --lowmed:#adff2f; --low:#2e8b57;
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
  .theme-toggle{
    width:34px; height:34px; display:flex; align-items:center; justify-content:center;
    font-size:17px; cursor:pointer; user-select:none;
  }
  .theme-toggle:hover{ filter:brightness(1.2); }
  .legend{ padding:12px 16px; font-family:ui-monospace,Consolas,monospace; font-size:11.5px; line-height:1.75; min-width:210px; }
  .legend b.h{font-size:13px;color:#ff6347;}
  .legend .row span.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:middle;}
  .legend hr{border:none;border-top:1px solid #444;margin:6px 0;}
  .legend .muted{color:var(--muted);}
  /* Detail card — used for both hex and fire-event popups, matching a
     standard incident-tracker card: white bg, label/value rows. */
  .detail-card{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; min-width:235px; color:#222; }
  .detail-card .hd{display:flex;align-items:center;gap:7px;font-size:15px;font-weight:700;margin-bottom:6px;}
  .detail-card .row{display:flex;justify-content:space-between;gap:14px;padding:3px 0;font-size:12.5px;border-bottom:1px solid #eee;}
  .detail-card .row:last-of-type{border-bottom:none;}
  .detail-card .row span:first-child{color:#666;white-space:nowrap;}
  .detail-card .row span:last-child{font-weight:600;text-align:right;}
  .detail-card .foot{color:#999;font-size:10.5px;margin-top:6px;line-height:1.4;}
  .leaflet-popup-content-wrapper{border-radius:8px;}
  body.light-mode .panel{ background:#ffffffdd; color:#1a1a1a; border-color:#e0641f; }
  body.light-mode .legend .muted, body.light-mode .title-ctl .sub{ color:#555; }
  body.light-mode .legend hr{ border-top-color:#ccc; }
</style>
</head>
<body>
<div id="map"></div>
<script>
const HEX_DATA   = __DATA_JSON__;      // { "4":[...], "5":[...], ... "8":[...] }
const FIRE_DATA  = __FIRES_JSON__;     // [ {h,s,erc_5D_max,...}, ... ]
const META       = __META_JSON__;

// ── Map + base layers — full world pan/zoom, like h3geo.org (data is
//    Texas-only for now, but the grid and basemap are global) ──────────
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

// ── Zoom -> H3 resolution mapping, tuned so hexagons keep a roughly
//    constant on-screen size as you zoom (the mechanism h3geo.org uses) ──
//    Resolutions 0-3 have no risk data (Texas-only sample) but still
//    render as a neutral grid so the whole world stays tessellated.
const EDGE_KM = { 0: 1107.71, 1: 418.68, 2: 158.24, 3: 59.81, 4: 22.606, 5: 8.544, 6: 3.229, 7: 1.221, 8: 0.4614 };
const SUPPORTED_RES = [0, 1, 2, 3, 4, 5, 6, 7, 8];
const TARGET_HEX_PX = 60; // desired hex diameter on screen

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
  return (v === null || v === undefined) ? 'N/A' : (v + (suffix || ''));
}

// ── Hex layer: on every pan/zoom, compute the FULL H3 tessellation that
//    covers the current viewport at the resolution-for-this-zoom (via
//    h3.polygonToCells over the map bounds — this is what makes the grid
//    seamless like h3geo.org, instead of only drawing cells with data).
//    Cells that carry risk data are colored; the rest render as a plain
//    neutral grid, same as h3geo.org shading only landmass differently.
const hexLayer = L.layerGroup().addTo(map);
const MAX_CELLS = 6000;

// O(1) lookup: dataByRes[res][h3index] -> record
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
  if (east <= west) east += 360; // world-wrapped viewport (worldCopyJump)
  const key = `${res}|${south.toFixed(2)}|${west.toFixed(2)}|${north.toFixed(2)}|${east.toFixed(2)}`;
  if (!force && key === lastKey) return;
  lastKey = key;
  hexLayer.clearLayers();

  const ring = [
    [south, west],
    [north, west],
    [north, east],
    [south, east],
    [south, west],
  ];
  let cells;
  try { cells = h3.polygonToCells(ring, res, false); } catch (e) { cells = []; }
  const truncated = cells.length > MAX_CELLS;
  if (truncated) cells = cells.slice(0, MAX_CELLS);

  const lookup = dataByRes[res] || {};
  const neutralFill  = theme === 'light' ? 'rgba(20,20,20,0.03)' : 'rgba(255,255,255,0.04)';
  const neutralBorder = theme === 'light' ? 'rgba(20,20,20,0.35)' : 'rgba(255,255,255,0.30)';
  const isLeaf = (res === 8);

  for (const h of cells) {
    let boundary;
    try { boundary = h3.cellToBoundary(h, false); } catch (e) { continue; }

    const rec = lookup[h];
    const hasData = !!rec;
    const color = hasData ? riskColor(rec.s) : neutralBorder;
    const fillColor = hasData ? riskColor(rec.s) : neutralFill;
    const fillOpacity = hasData ? (0.22 + rec.s * 0.62) : 1;

    const poly = L.polygon(boundary, {
      color, weight: hasData ? (isLeaf ? 0.9 : 1.2) : 0.6,
      fillColor, fillOpacity, opacity: hasData ? 1 : 0.9,
    });

    let html;
    if (hasData && isLeaf) {
      html = `<div class="detail-card">
        <div class="hd" style="color:${color}">🔥 ${riskLabel(rec.s)}</div>
        <div class="row"><span>Predicted risk</span><span>${(rec.s*100).toFixed(1)}%</span></div>
        <div class="row"><span>Actual fire recorded</span><span>${rec.f ? 'Yes' : 'No'}</span></div>
        <div class="row"><span>ERC (5-day max)</span><span>${fmt(rec.erc_5D_max,' BTU/ft²')}</span></div>
        <div class="row"><span>Wind (5-day max)</span><span>${fmt(rec.vs_5D_max,' m/s')}</span></div>
        <div class="row"><span>VPD</span><span>${fmt(rec.vpd,' kPa')}</span></div>
        <div class="row"><span>Fuel moisture (FM100)</span><span>${fmt(rec.fm100,'%')}</span></div>
        <div class="row"><span>Max temperature</span><span>${fmt(rec.tmmx,' °C')}</span></div>
        <div class="row"><span>H3 cell (res 8)</span><span style="font-size:10px">${h}</span></div>
      </div>`;
    } else if (hasData) {
      html = `<div class="detail-card">
        <div class="hd" style="color:${color}">🔥 ${riskLabel(rec.s)} (aggregated)</div>
        <div class="row"><span>Mean predicted risk</span><span>${(rec.s*100).toFixed(1)}%</span></div>
        <div class="row"><span>Sampled cells here</span><span>${rec.n}</span></div>
        <div class="row"><span>Fire events among them</span><span>${rec.f}</span></div>
        <div class="row"><span>H3 cell (res ${res})</span><span style="font-size:10px">${h}</span></div>
        <div class="foot">Zoom in for cell-level detail</div>
      </div>`;
    } else {
      html = `<div class="detail-card">
        <div class="hd" style="color:#888">No risk data</div>
        <div class="row"><span>H3 cell (res ${res})</span><span style="font-size:10px">${h}</span></div>
      </div>`;
    }
    poly.bindPopup(html, { maxWidth: 280 });
    if (hasData) poly.bindTooltip(`${riskLabel(rec.s)} | ${rec.s.toFixed(3)}`);
    poly.addTo(hexLayer);
  }
  resBadge.update(res, z, cells.length, truncated);
}

// ── Fire event markers (always at leaf resolution) ──────────────────────
const fireLayer = L.layerGroup().addTo(map);
for (const f of FIRE_DATA) {
  let lat, lon;
  try { [lat, lon] = h3.cellToLatLng(f.h); } catch (e) { continue; }
  const marker = L.circleMarker([lat, lon], {
    radius: 7, color: '#fff', weight: 2, fillColor: '#ff0000', fillOpacity: 1,
  });
  const html = `<div class="detail-card">
    <div class="hd" style="color:#c0392b">🔥 Fire Event</div>
    <div class="row"><span>Date</span><span>${META.date}</span></div>
    <div class="row"><span>Window</span><span>${META.window_label}</span></div>
    <div class="row"><span>Model predicted risk</span><span>${(f.s*100).toFixed(1)}%</span></div>
    <div class="row"><span>ERC (5-day max)</span><span>${fmt(f.erc_5D_max,' BTU/ft²')}</span></div>
    <div class="row"><span>Wind (5-day max)</span><span>${fmt(f.vs_5D_max,' m/s')}</span></div>
    <div class="row"><span>VPD</span><span>${fmt(f.vpd,' kPa')}</span></div>
    <div class="row"><span>H3 cell</span><span style="font-size:10px">${f.h}</span></div>
    <div class="foot">Source: FPA-FOD training label (sample data). Size, cause, and county aren't in this dataset yet.</div>
  </div>`;
  marker.bindPopup(html, { maxWidth: 260 });
  marker.bindTooltip('★ Fire event');
  marker.addTo(fireLayer);
}

map.on('moveend', () => renderHexes(false));
renderHexes(true);

// ── Controls (title, resolution badge, legend, layers) ──────────────────
const Title = L.Control.extend({
  options: { position: 'topright' },
  onAdd: function () {
    const div = L.DomUtil.create('div', 'panel title-ctl');
    div.innerHTML = `<b>🔥 IgnitionNet — ${META.state} Wildfire Risk</b>
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
  setIcon: function (curTheme) {
    if (this._div) this._div.innerHTML = curTheme === 'dark' ? '☀️' : '🌙';
    if (this._div) this._div.title = curTheme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
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
    const countStr = truncated ? `${count}+ (capped)` : `${count}`;
    this._div.innerHTML = `H3 Resolution: ${res} &nbsp;·&nbsp; Zoom: ${zoom} &nbsp;·&nbsp; Cells: ${countStr}`;
  }
});
const resBadge = new ResBadge();
map.addControl(resBadge);

const Legend = L.Control.extend({
  options: { position: 'bottomleft' },
  onAdd: function () {
    const div = L.DomUtil.create('div', 'panel legend');
    div.innerHTML = `
      <b class="h">🔥 ${META.state} Fire Risk</b><br>
      <span class="muted">Date:</span> <b>${META.date}</b><br>
      <span class="muted">Window:</span> <b>${META.window_label}</b><br>
      <span class="muted">Sampled cells:</span> <b>${META.n_total}</b><br>
      <span class="muted">Actual fires:</span> <b style="color:#ff0000">${META.n_fires}</b><br>
      <hr>
      <div class="row"><span class="sw" style="background:#ff0000"></span>Critical (&gt;0.80)</div>
      <div class="row"><span class="sw" style="background:#ff4500"></span>High (0.65-0.80)</div>
      <div class="row"><span class="sw" style="background:#ff8c00"></span>Med-High (0.50-0.65)</div>
      <div class="row"><span class="sw" style="background:#ffd700"></span>Medium (0.35-0.50)</div>
      <div class="row"><span class="sw" style="background:#adff2f"></span>Low-Med (0.20-0.35)</div>
      <div class="row"><span class="sw" style="background:#2e8b57"></span>Low (&lt;0.20)</div>
      <div class="row"><span class="sw" style="background:transparent;border:1px solid #888"></span>No data (grid only)</div>
      <div class="row">★ Fire event (FPA-FOD)</div>
      <hr>
      <span class="muted">Pan/zoom to change H3 resolution.<br>Click any hex or marker for detail.</span>`;
    return div;
  }
});
map.addControl(new Legend());

L.control.layers(
  null,
  { '🔥 Risk Hexagons (H3)': hexLayer, '★ Fire Events': fireLayer },
  { collapsed: false, position: 'topleft' }
).addTo(map);
</script>
</body>
</html>
"""


def build_html(data_by_res: dict, fires: list, meta: dict) -> str:
    html = TEMPLATE
    html = html.replace("__TITLE__", f"IgnitionNet — {meta['state']} Wildfire Risk — {meta['date']}")
    html = html.replace("__DATA_JSON__", json.dumps({str(k): v for k, v in data_by_res.items()}))
    html = html.replace("__FIRES_JSON__", json.dumps(fires))
    html = html.replace("__META_JSON__", json.dumps(meta))
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="TX")
    parser.add_argument("--date", default=None,
                        help="Date to map (YYYY-MM-DD). If not given, auto-selects highest-fire date.")
    parser.add_argument("--window", default=18, type=int, choices=[0, 6, 12, 18])
    args = parser.parse_args()

    s = args.state.upper()
    cfg = STATE_CONFIG[s]
    out = cfg["output_dir"]
    slug = s.lower()
    map_dir = out / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    model_path = out / "models" / f"xgb_baseline_{slug}.ubj"
    meta_path = out / "models" / f"xgb_baseline_{slug}_meta.json"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path} — run Phase 3 training first")
        return

    model = xgb.Booster()
    model.load_model(str(model_path))
    with open(meta_path) as f:
        model_meta = json.load(f)
    threshold = model_meta["threshold"]

    if args.date is None:
        logger.info("Auto-selecting best date (most fires in test split)...")
        good = find_good_dates(out, slug)
        if not good:
            logger.error("No good dates found in test split")
            return
        best_date, best_window, n_fires = good[0]
        logger.info(f"Best date: {best_date} window={best_window}Z ({n_fires} fires)")
        target_date = datetime.strptime(best_date, "%Y-%m-%d").date()
        window_hour = best_window
    else:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        window_hour = args.window

    logger.info(f"\n[1/3] Loading rows for {target_date} {window_hour}Z...")
    day_df = load_rows_for_date(out, slug, target_date, window_hour)
    if len(day_df) == 0:
        logger.error(f"No data found for {target_date} {window_hour}Z")
        return
    logger.info(f"  Rows: {len(day_df):,} | Fires: {int((day_df['label']==1).sum())}")

    logger.info("[2/3] Generating predictions...")
    day_df = day_df.copy()
    day_df["risk_score"] = predict(model, day_df)

    logger.info("[3/3] Building H3-native map (multi-resolution rollup + client-side rendering)...")
    data_by_res = build_multires_cells(day_df)
    fires = build_fire_events(day_df)
    meta = {
        "state": s,
        "date": str(target_date),
        "window_label": f"{window_hour}Z ({WINDOW_NAMES.get(window_hour,'')})",
        "model_name": "XGBoost Baseline v1",
        "threshold": round(float(threshold), 4),
        "n_total": int(len(day_df)),
        "n_fires": int((day_df["label"] == 1).sum()),
    }
    html = build_html(data_by_res, fires, meta)

    html_path = map_dir / f"risk_map_h3_{s}_{target_date}_{window_hour}Z.html"
    html_path.write_text(html, encoding="utf-8")

    csv_path = map_dir / f"predictions_{s}_{target_date}_{window_hour}Z.csv"
    (day_df[["h3_cell", "centroid_lat", "centroid_lon", "risk_score", "label"]]
     .sort_values("risk_score", ascending=False)
     .to_csv(csv_path, index=False))

    logger.info("\n" + "=" * 65)
    logger.info("  MAP SAVED -> open in a browser:")
    logger.info(f"  {html_path}")
    logger.info(f"  CSV SAVED: {csv_path}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
