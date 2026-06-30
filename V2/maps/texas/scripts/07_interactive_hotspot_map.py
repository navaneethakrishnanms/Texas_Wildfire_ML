"""
07_interactive_hotspot_map.py -- Texas Wildfire Interactive Folium Map
=======================================================================
Matches the sample reference map design:
  - CartoDB dark matter base (default)
  - Tile layer switcher (4 options)
  - Layer 1: Fire Density HeatMap
  - Layer 2: MarkerCluster per cause category (4 causes)
  - Layer 3: High-Risk Hotspots by ERC (top 200, radius by ERC)
  - Layer 4: Lightning Fires Only
  - Fullscreen plugin, MiniMap plugin, LayerControl (collapsed=False)
  - Fixed HTML: title banner, stats panel, cause legend

Output: eda_outputs/wildfire_hotspot_map.html
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ─── CONFIG ──────────────────────────────────────────────────────
STATE_CODE = "TX"
STATE_NAME = "Texas"
MAP_CENTER = [31.0, -100.0]    # Texas center
BASE_DIR   = Path(__file__).resolve().parents[3]
DATA_PATH  = BASE_DIR / "data" / "processed" / "texas" / "texas_fire_2014_2020.parquet"
OUT_DIR    = BASE_DIR / "maps" / "texas" / "eda_outputs"
OUT_HTML   = OUT_DIR / "wildfire_hotspot_map.html"

# ─── IMPORTS (with install fallback) ─────────────────────────────
try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster, Fullscreen, MiniMap
except ImportError:
    print("Installing folium ...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "folium"], stdout=subprocess.DEVNULL)
    import folium
    from folium.plugins import HeatMap, MarkerCluster, Fullscreen, MiniMap

# ─── CAUSE COLORS ────────────────────────────────────────────────
CAUSE_CONFIG = {
    "Lightning":         {"color": "#00B4D8", "icon": "bolt"},
    "Human":             {"color": "#FF6B35", "icon": "user"},
    "Equipment Use":     {"color": "#F7C548", "icon": "wrench"},
    "Debris Burning":    {"color": "#2ECC71", "icon": "fire"},
    "Arson/Incendiary":  {"color": "#E74C3C", "icon": "warning-sign"},
    "Recreation":        {"color": "#9B59B6", "icon": "tree-conifer"},
    "Miscellaneous":     {"color": "#7F8C8D", "icon": "question-sign"},
    "Children":          {"color": "#F39C12", "icon": "star"},
}
DEFAULT_COLOR = "#888888"

# ─── LOAD DATA ───────────────────────────────────────────────────
print(f"[07] Loading {STATE_NAME} dataset ...")
OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(DATA_PATH)
df_geo = df.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()

cause_col = "NWCG_GENERAL_CAUSE"
erc_col   = "erc"

df_geo["disc_year"] = pd.to_datetime(df_geo.get("DISCOVERY_DATE", None), errors="coerce").dt.year.fillna(
    df_geo.get("FIRE_YEAR", np.nan)
).astype("Int64")

print(f"     Fire records with lat/lon: {len(df_geo):,}")

# Cause counts for legend
cause_counts = df_geo[cause_col].value_counts() if cause_col in df_geo.columns else pd.Series()
top_causes   = cause_counts.head(4).index.tolist()

# Stats for panel
total_recs = len(df)
total_geo  = len(df_geo)
years      = sorted(df_geo["FIRE_YEAR"].dropna().unique().astype(int).tolist()) if "FIRE_YEAR" in df_geo.columns else []
avg_erc    = round(df_geo[erc_col].mean(), 1) if erc_col in df_geo.columns else "N/A"
avg_fs     = round(df_geo["FIRE_SIZE"].mean(), 1) if "FIRE_SIZE" in df_geo.columns else "N/A"

disc_month = pd.to_datetime(df_geo.get("DISCOVERY_DATE", None), errors="coerce").dt.month
MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
peak_month = MONTH_NAMES.get(int(disc_month.value_counts().idxmax()), "N/A") if not disc_month.dropna().empty else "N/A"

# ─── BASE MAP ────────────────────────────────────────────────────
print("[07] Building folium map ...")
m = folium.Map(
    location=MAP_CENTER,
    zoom_start=6,
    tiles="CartoDB dark_matter",
    control_scale=True,
    prefer_canvas=True,
)

# ─── TILE LAYER OPTIONS ───────────────────────────────────────────
folium.TileLayer("CartoDB dark_matter", name="Dark Map (default)", show=True).add_to(m)
folium.TileLayer("OpenStreetMap",       name="Street Map",          show=False).add_to(m)
folium.TileLayer("CartoDB positron",    name="Light Map",           show=False).add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Satellite", show=False
).add_to(m)

# ─── LAYER 1: FIRE DENSITY HEATMAP ───────────────────────────────
print("[07] Adding HeatMap layer ...")
heat_data = df_geo[["LATITUDE", "LONGITUDE"]].values.tolist()
heat_fg   = folium.FeatureGroup(name="Fire Density Heatmap", show=True)
HeatMap(
    heat_data,
    gradient={"0.2": "blue", "0.4": "teal", "0.6": "yellow", "0.8": "orange", "1.0": "red"},
    radius=18, blur=25, min_opacity=0.3,
).add_to(heat_fg)
heat_fg.add_to(m)

# ─── LAYER 2: MARKER CLUSTERS PER CAUSE ──────────────────────────
print("[07] Adding MarkerCluster layers ...")
MAX_PER_CAUSE = 2000

for cause in top_causes:
    cfg     = CAUSE_CONFIG.get(cause, {"color": DEFAULT_COLOR, "icon": "info-sign"})
    c_color = cfg["color"]
    sub     = df_geo[df_geo[cause_col] == cause]
    if len(sub) > MAX_PER_CAUSE:
        sub = sub.sample(MAX_PER_CAUSE, random_state=42)

    fg = folium.FeatureGroup(name=f"Fire Points -- {cause} ({len(df_geo[df_geo[cause_col]==cause]):,})",
                             show=True)
    mc = MarkerCluster(options={"maxClusterRadius": 40}).add_to(fg)

    for _, row in sub.iterrows():
        lat  = row["LATITUDE"]
        lon  = row["LONGITUDE"]
        year = row.get("disc_year", row.get("FIRE_YEAR", "N/A"))
        size = round(row.get("FIRE_SIZE", 0), 2)
        erc  = round(row.get(erc_col, 0), 1) if erc_col in row.index else "N/A"
        county = row.get("COUNTY", "N/A")

        popup_html = f"""
        <div style='font-family:monospace;font-size:12px;min-width:200px'>
          <b style='color:{c_color}'>{cause}</b><br>
          <b>Year:</b> {year}<br>
          <b>County:</b> {county}<br>
          <b>Fire Size:</b> {size} acres<br>
          <b>ERC:</b> {erc}<br>
          <b>Lat/Lon:</b> {lat:.4f}, {lon:.4f}
        </div>"""
        tooltip_html = f"{cause} | {year}"

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=c_color, fill=True, fill_color=c_color, fill_opacity=0.7,
            weight=0.5,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=tooltip_html,
        ).add_to(mc)

    fg.add_to(m)

# ─── LAYER 3: HIGH-RISK HOTSPOTS (top 200 by ERC) ────────────────
print("[07] Adding High-Risk Hotspots layer ...")
if erc_col in df_geo.columns:
    hot_df  = df_geo.dropna(subset=[erc_col]).nlargest(200, erc_col)
    hot_fg  = folium.FeatureGroup(name="High-Risk Hotspots (Top ERC)", show=False)
    erc_max = hot_df[erc_col].max()
    erc_min = hot_df[erc_col].min()

    for _, row in hot_df.iterrows():
        erc_val = row[erc_col]
        norm    = (erc_val - erc_min) / (erc_max - erc_min + 1e-6)
        radius  = 6 + norm * 16
        color   = "#FF0000" if erc_val > 80 else ("#FF8C00" if erc_val > 60 else "#FFD700")
        popup_html = f"""
        <div style='font-family:monospace;font-size:12px'>
          <b style='color:{color}'>HIGH-RISK HOTSPOT</b><br>
          <b>ERC:</b> {round(erc_val,1)}<br>
          <b>Fire Size:</b> {round(row.get('FIRE_SIZE',0),1)} acres<br>
          <b>Year:</b> {row.get('FIRE_YEAR','N/A')}
        </div>"""
        folium.CircleMarker(
            location=[row["LATITUDE"], row["LONGITUDE"]],
            radius=radius, color=color, fill=True, fill_color=color,
            fill_opacity=0.6, weight=1.5,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"ERC: {round(erc_val,1)}",
        ).add_to(hot_fg)
    hot_fg.add_to(m)

# ─── LAYER 4: LIGHTNING FIRES ONLY ───────────────────────────────
print("[07] Adding Lightning-only layer ...")
if cause_col in df_geo.columns:
    light_df = df_geo[df_geo[cause_col].astype(str).str.contains("Lightning", case=False, na=False)]
    light_fg = folium.FeatureGroup(name=f"Lightning Fires Only ({len(light_df):,})", show=False)
    sample_l = light_df.sample(min(1500, len(light_df)), random_state=42) if len(light_df) > 0 else light_df
    for _, row in sample_l.iterrows():
        folium.CircleMarker(
            location=[row["LATITUDE"], row["LONGITUDE"]],
            radius=5, color="#00B4D8", fill=True, fill_color="#00B4D8",
            fill_opacity=0.65, weight=0.8,
            popup=f"Lightning | {row.get('FIRE_YEAR','N/A')} | {round(row.get('FIRE_SIZE',0),1)} ac",
            tooltip="Lightning Fire",
        ).add_to(light_fg)
    light_fg.add_to(m)

# ─── PLUGINS ──────────────────────────────────────────────────────
Fullscreen(position="topright", title="Fullscreen", title_cancel="Exit Fullscreen").add_to(m)
MiniMap(position="bottomright", toggle_display=True, width=150, height=120).add_to(m)
folium.LayerControl(position="topleft", collapsed=False).add_to(m)

# ─── HTML: TITLE BANNER (top-center) ──────────────────────────────
year_range = f"{min(years)}-{max(years)}" if years else "2014-2020"
title_html = f"""
<div style='
  position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
  z-index: 9999; background: rgba(26,26,46,0.92);
  border: 1px solid #FF6B35; border-radius: 8px;
  padding: 8px 20px; text-align: center; pointer-events: none;'>
  <span style='font-family: monospace; font-size: 16px; font-weight: bold; color: #F7C548;'>
    IgnitionNet &mdash; {STATE_NAME} Wildfire Hotspot Map
  </span><br>
  <span style='font-family: monospace; font-size: 11px; color: #e0e0e0;'>
    {total_geo:,} fire events &nbsp;|&nbsp; {year_range} &nbsp;|&nbsp; Toggle layers on the left
  </span>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

# ─── HTML: STATS PANEL (top-right) ────────────────────────────────
stats_html = f"""
<div style='
  position: fixed; top: 60px; right: 10px; z-index: 9998;
  background: rgba(26,26,46,0.93); border: 1px solid #333355;
  border-radius: 8px; padding: 12px 16px; min-width: 200px;
  font-family: monospace; font-size: 12px; color: #e0e0e0;'>
  <b style='color:#F7C548; font-size:13px;'>Dataset Statistics</b><br><br>
  Total records: <b style='color:#FF6B35'>{total_recs:,}</b><br>
  Fire events:   <b style='color:#FF6B35'>{total_geo:,}</b><br>
  Years covered: <b>{year_range}</b><br><br>
  Peak month:    <b>{peak_month}</b><br>
  Avg ERC (fire): <b>{avg_erc}</b><br>
  Avg Size (fire): <b>{avg_fs} ac</b>
</div>
"""
m.get_root().html.add_child(folium.Element(stats_html))

# ─── HTML: CAUSE LEGEND (bottom-right) ────────────────────────────
legend_rows = ""
for cause in top_causes:
    cfg     = CAUSE_CONFIG.get(cause, {"color": DEFAULT_COLOR})
    cnt     = int(cause_counts.get(cause, 0))
    legend_rows += (
        f"<span style='color:{cfg['color']};font-size:16px;'>&#9679;</span> "
        f"{cause} ({cnt:,})<br>"
    )

# ERC legend
legend_rows += "<br><b style='color:#e0e0e0'>HIGH-RISK ERC:</b><br>"
legend_rows += "<span style='color:#FF0000;'>&#9679;</span> ERC &gt; 80 (Extreme)<br>"
legend_rows += "<span style='color:#FF8C00;'>&#9679;</span> ERC 60-80 (High)<br>"
legend_rows += "<span style='color:#FFD700;'>&#9679;</span> ERC &lt; 60 (Moderate)<br>"
legend_rows += f"<br><b>Total fires: {total_geo:,}</b><br>"
legend_rows += f"{STATE_NAME} | 2014-2020"

legend_html = f"""
<div style='
  position: fixed; bottom: 30px; right: 10px; z-index: 9998;
  background: rgba(26,26,46,0.93); border: 1px solid #333355;
  border-radius: 8px; padding: 12px 16px; min-width: 200px;
  font-family: monospace; font-size: 12px; color: #e0e0e0;'>
  <b style='color:#FF6B35; font-size:13px;'>&#128293; Wildfire Hotspot Map</b><br><br>
  <b>FIRE CAUSE:</b><br>
  {legend_rows}
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ─── SAVE ────────────────────────────────────────────────────────
m.save(str(OUT_HTML))
print(f"\n[07] DONE -- Interactive map saved:")
print(f"     {OUT_HTML}")
print(f"\n     HOW TO USE:")
print(f"     1. Open the file in any web browser (Chrome/Firefox/Edge)")
print(f"     2. Use the layer panel (top-left) to toggle layers on/off")
print(f"     3. Click any marker to see fire details")
print(f"     4. Use the fullscreen button (top-right) for full view")
print(f"     5. Mini-map (bottom-right) shows your position")
