"""
app_gradio.py  –  Texas Wildfire Manual-Input Checker
=====================================================
Paste a CSV row OR type each value individually.
Columns match the dataset exactly:
  latitude, longitude, acq_date,
  NDVI, EVI, LST, Temperature, Wind, Rainfall,
  DEM, Slope, Aspect, LandCover,
  month, day_of_year, season_code,
  sin_month, cos_month, sin_doy, cos_doy,
  is_peak_fire_season, Fire  (actual label)

Usage:
    pip install gradio lightgbm
    python app_gradio.py
"""

import json
import pickle
import warnings
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# ── columns ────────────────────────────────────────────────────────────
# Exactly as they appear in the CSV header
CSV_COLS = [
    "latitude", "longitude", "acq_date",
    "NDVI", "EVI", "LST", "Temperature", "Wind", "Rainfall",
    "DEM", "Slope", "Aspect", "LandCover",
    "month", "day_of_year", "season_code",
    "sin_month", "cos_month", "sin_doy", "cos_doy",
    "is_peak_fire_season", "Fire",
]

# Columns the model actually uses (no lat/lon/date/Fire)
MODEL_FEATURES = [
    "NDVI", "EVI", "LST", "Temperature", "Wind", "Rainfall",
    "DEM", "Slope", "Aspect", "LandCover",
    "month", "day_of_year", "season_code",
    "sin_month", "cos_month", "sin_doy", "cos_doy",
    "is_peak_fire_season",
]

LC_MAP = {
    0:"Water", 1:"Evergreen Needleleaf Forest", 2:"Evergreen Broadleaf Forest",
    3:"Deciduous Needleleaf Forest", 4:"Deciduous Broadleaf Forest",
    5:"Mixed Forest", 6:"Closed Shrubland", 7:"Open Shrubland",
    8:"Woody Savanna", 9:"Savanna", 10:"Grassland",
    11:"Wetland", 12:"Cropland", 13:"Urban/Built-up",
    14:"Cropland/Natural Mosaic", 15:"Snow/Ice", 16:"Barren", 17:"Unclassified",
}
SEASON_MAP = {0:"Winter",1:"Spring",2:"Summer",3:"Fall"}

# ── model loading ───────────────────────────────────────────────────────
MODELS    = {}
RF_IMP    = None
OPT_THRESH = 0.46

def _load():
    global RF_IMP, OPT_THRESH
    for tag, fname in [
        ("Best (LightGBM — tuned)", "best_model.pkl"),
        ("LightGBM",                "lgbm.pkl"),
        ("XGBoost-A",               "xgb_a.pkl"),
        ("Random Forest",           "rf.pkl"),
    ]:
        p = MODELS_DIR / fname
        if p.exists():
            with open(p, "rb") as f:
                MODELS[tag] = pickle.load(f)

    p = MODELS_DIR / "rf_imputer.pkl"
    if p.exists():
        with open(p, "rb") as f:
            RF_IMP = pickle.load(f)

    p = MODELS_DIR / "optimal_threshold.json"
    if p.exists():
        with open(p) as f:
            OPT_THRESH = float(json.load(f)["threshold"])

_load()
MODEL_NAMES = list(MODELS.keys())

# ── helpers ─────────────────────────────────────────────────────────────
def safe_float(v, default=0.0):
    try:
        return float(str(v).strip())
    except Exception:
        return default

def safe_int(v, default=0):
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default

def parse_row(row_str: str):
    """Parse a comma-separated CSV row into a dict keyed by CSV_COLS."""
    parts = [p.strip() for p in row_str.strip().split(",")]
    if len(parts) < len(CSV_COLS):
        parts += [""] * (len(CSV_COLS) - len(parts))
    return {col: parts[i] for i, col in enumerate(CSV_COLS)}

# ── prediction ──────────────────────────────────────────────────────────
def predict(
    model_name, use_tuned, custom_thresh,
    # individual fields — same order as CSV_COLS
    lat, lon, acq_date,
    ndvi, evi, lst, temperature, wind, rainfall,
    dem, slope, aspect, landcover,
    month, day_of_year, season_code,
    sin_month, cos_month, sin_doy, cos_doy,
    is_peak, actual_fire,
):
    if model_name not in MODELS:
        return "<p style='color:red'>❌ Model not loaded.</p>"

    threshold = OPT_THRESH if use_tuned else float(custom_thresh)

    row = {
        "NDVI":               safe_float(ndvi),
        "EVI":                safe_float(evi),
        "LST":                safe_float(lst),
        "Temperature":        safe_float(temperature),
        "Wind":               safe_float(wind),
        "Rainfall":           safe_float(rainfall),
        "DEM":                safe_float(dem),
        "Slope":              safe_float(slope),
        "Aspect":             safe_float(aspect),
        "LandCover":          safe_int(landcover),
        "month":              safe_int(month),
        "day_of_year":        safe_int(day_of_year),
        "season_code":        safe_int(season_code),
        "sin_month":          safe_float(sin_month),
        "cos_month":          safe_float(cos_month),
        "sin_doy":            safe_float(sin_doy),
        "cos_doy":            safe_float(cos_doy),
        "is_peak_fire_season": safe_int(is_peak),
    }

    X = pd.DataFrame([row])[MODEL_FEATURES]

    model = MODELS[model_name]
    model_lower = model_name.lower()

    if "random forest" in model_lower and RF_IMP is not None:
        arr = RF_IMP.transform(X)
        X   = pd.DataFrame(arr, columns=MODEL_FEATURES)

    # LightGBM was trained with LandCover as category dtype — must match at inference
    if "lightgbm" in model_lower or "lgbm" in model_lower:
        X = X.copy()
        X["LandCover"] = X["LandCover"].astype("category")

    try:
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0, 1])
        else:
            import xgboost as xgb
            proba = float(model.predict(xgb.DMatrix(X))[0])
    except Exception as e:
        return f"<p style='color:red'>❌ Error: {e}</p>"

    fire_pred = int(proba >= threshold)
    actual    = safe_int(actual_fire, -1)
    lc_name   = LC_MAP.get(safe_int(landcover), "Unknown")
    seas_name = SEASON_MAP.get(safe_int(season_code), "?")

    # colours
    if proba < 0.25:    risk_lbl, risk_col = "🟢 LOW",      "#22c55e"
    elif proba < 0.45:  risk_lbl, risk_col = "🟡 MEDIUM",   "#f59e0b"
    elif proba < 0.65:  risk_lbl, risk_col = "🟠 HIGH",     "#f97316"
    else:               risk_lbl, risk_col = "🔴 CRITICAL",  "#ef4444"

    pred_txt = "🔥 FIRE" if fire_pred else "✅ NO FIRE"
    pred_col = "#ef4444" if fire_pred else "#22c55e"

    # actual label row
    if actual == -1:
        act_row = ""
    else:
        act_lbl = "🔥 FIRE (1)" if actual == 1 else "✅ NO FIRE (0)"
        act_col = "#ef4444" if actual == 1 else "#22c55e"
        match   = "✔ CORRECT" if fire_pred == actual else "✘ WRONG"
        match_c = "#22c55e"  if fire_pred == actual else "#ef4444"
        act_row = f"""
        <div style="display:flex; gap:12px; margin-top:12px; flex-wrap:wrap;">
          <div style="flex:1; min-width:160px; background:rgba(255,255,255,0.04); border-radius:10px; padding:14px 18px;">
            <div style="font-size:0.7rem; text-transform:uppercase; color:#64748b; letter-spacing:.08em; margin-bottom:6px;">
              Actual Label (CSV)
            </div>
            <div style="font-size:1.2rem; font-weight:700; color:{act_col};">{act_lbl}</div>
          </div>
          <div style="flex:1; min-width:160px; background:rgba(255,255,255,0.04); border-radius:10px; padding:14px 18px;">
            <div style="font-size:0.7rem; text-transform:uppercase; color:#64748b; letter-spacing:.08em; margin-bottom:6px;">
              Model vs Actual
            </div>
            <div style="font-size:1.2rem; font-weight:700; color:{match_c};">{match}</div>
          </div>
        </div>"""

    html = f"""
<div style="
  background:linear-gradient(135deg,#1e1e2e 0%,#12121a 100%);
  border:1px solid #2d2d44; border-radius:16px;
  padding:28px 28px 24px; font-family:'Segoe UI',system-ui,sans-serif; color:#e2e8f0;
">
  <!-- headline -->
  <h2 style="margin:0 0 18px; font-size:1.25rem; color:#94a3b8; letter-spacing:.05em;">
    🔥 WILDFIRE RISK RESULT
  </h2>

  <!-- probability + risk level -->
  <div style="display:flex; align-items:center; gap:20px;
              background:rgba(255,255,255,0.04); border-radius:12px; padding:18px 22px; margin-bottom:16px;">
    <div style="text-align:center; min-width:110px;">
      <div style="font-size:3rem; font-weight:800; color:{risk_col}; line-height:1;">
        {proba*100:.1f}%
      </div>
      <div style="font-size:0.72rem; color:#64748b; margin-top:4px; text-transform:uppercase; letter-spacing:.1em;">
        Fire Probability
      </div>
    </div>
    <div style="flex:1; padding-left:20px; border-left:1px solid #333;">
      <div style="font-size:1.5rem; font-weight:700; color:{risk_col}; margin-bottom:6px;">
        {risk_lbl}
      </div>
      <div style="font-size:1.05rem; font-weight:600; color:{pred_col}; margin-bottom:8px;">
        Prediction: {pred_txt}
      </div>
      <div style="background:#0f172a; border-radius:6px; height:9px; overflow:hidden;">
        <div style="width:{proba*100:.1f}%; height:100%;
                    background:linear-gradient(90deg,#22c55e 0%,#f59e0b 50%,#ef4444 100%);"></div>
      </div>
      <div style="font-size:0.8rem; color:#64748b; margin-top:5px;">
        Threshold: {threshold:.3f} ({model_name})
      </div>
    </div>
  </div>

  <!-- 2-col detail grid -->
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;">
    <div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:13px 16px;">
      <div style="font-size:.7rem; text-transform:uppercase; color:#64748b; letter-spacing:.08em; margin-bottom:7px;">📍 Location & Date</div>
      <div style="font-size:.85rem; color:#cbd5e1; line-height:1.9;">
        Latitude: <b>{safe_float(lat):.4f}°</b><br>
        Longitude: <b>{safe_float(lon):.4f}°</b><br>
        Date: <b>{str(acq_date).strip() or '—'}</b>
      </div>
    </div>
    <div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:13px 16px;">
      <div style="font-size:.7rem; text-transform:uppercase; color:#64748b; letter-spacing:.08em; margin-bottom:7px;">🌿 Vegetation</div>
      <div style="font-size:.85rem; color:#cbd5e1; line-height:1.9;">
        NDVI: <b>{safe_float(ndvi):.4f}</b><br>
        EVI: <b>{safe_float(evi):.4f}</b>
      </div>
    </div>
    <div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:13px 16px;">
      <div style="font-size:.7rem; text-transform:uppercase; color:#64748b; letter-spacing:.08em; margin-bottom:7px;">🌡️ Climate</div>
      <div style="font-size:.85rem; color:#cbd5e1; line-height:1.9;">
        LST: <b>{safe_float(lst):.2f} K</b><br>
        Temp: <b>{safe_float(temperature):.2f} °C</b><br>
        Wind: <b>{safe_float(wind):.3f} m/s</b>&nbsp; Rain: <b>{safe_float(rainfall):.3f} mm</b>
      </div>
    </div>
    <div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:13px 16px;">
      <div style="font-size:.7rem; text-transform:uppercase; color:#64748b; letter-spacing:.08em; margin-bottom:7px;">⛰️ Terrain & Land</div>
      <div style="font-size:.85rem; color:#cbd5e1; line-height:1.9;">
        DEM: <b>{safe_float(dem):.1f} m</b>&nbsp; Slope: <b>{safe_float(slope):.2f}°</b><br>
        Aspect: <b>{safe_float(aspect):.2f}°</b><br>
        LandCover: <b>{safe_int(landcover)} — {lc_name}</b>
      </div>
    </div>
    <div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:13px 16px;">
      <div style="font-size:.7rem; text-transform:uppercase; color:#64748b; letter-spacing:.08em; margin-bottom:7px;">📅 Temporal Features</div>
      <div style="font-size:.85rem; color:#cbd5e1; line-height:1.9;">
        Month: <b>{safe_int(month)}</b>&nbsp; DoY: <b>{safe_int(day_of_year)}</b><br>
        Season: <b>{seas_name}</b>&nbsp; Peak: <b>{'Yes 🔥' if safe_int(is_peak) else 'No'}</b>
      </div>
    </div>
    <div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:13px 16px;">
      <div style="font-size:.7rem; text-transform:uppercase; color:#64748b; letter-spacing:.08em; margin-bottom:7px;">〰️ Cyclical Encodings</div>
      <div style="font-size:.85rem; color:#cbd5e1; line-height:1.9;">
        sin_month: <b>{safe_float(sin_month):.5f}</b><br>
        cos_month: <b>{safe_float(cos_month):.5f}</b><br>
        sin_doy: <b>{safe_float(sin_doy):.5f}</b>&nbsp; cos_doy: <b>{safe_float(cos_doy):.5f}</b>
      </div>
    </div>
  </div>
  {act_row}
</div>
"""
    return html


# ── build UI ────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="🔥 Texas Wildfire Checker",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.orange,
            neutral_hue=gr.themes.colors.slate,
        ),
        css="""
        body,.gradio-container{background:#0f0f1a!important;color:#e2e8f0!important;
          font-family:'Segoe UI',system-ui,sans-serif!important}
        .gr-button-primary{background:linear-gradient(135deg,#f97316,#dc2626)!important;
          border:none!important;border-radius:10px!important;font-weight:700!important;
          font-size:1rem!important;padding:11px 28px!important;}
        .gr-button-primary:hover{transform:translateY(-2px)!important;
          box-shadow:0 8px 25px rgba(249,115,22,.4)!important;}
        label{color:#94a3b8!important;font-size:.8rem!important;
          text-transform:uppercase!important;letter-spacing:.06em!important;}
        .gr-group{background:#1a1a2e!important;border:1px solid #2d2d44!important;border-radius:14px!important;}
        """,
    ) as demo:

        gr.HTML("""
        <div style="text-align:center;padding:28px 20px 8px;
            background:linear-gradient(180deg,rgba(249,115,22,.12) 0%,transparent 100%);">
          <div style="font-size:2.5rem;">🔥</div>
          <h1 style="margin:6px 0;font-size:1.8rem;font-weight:800;
              background:linear-gradient(135deg,#f97316,#fbbf24);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
            Texas Wildfire Manual Checker</h1>
          <p style="color:#64748b;margin:8px 0 0;font-size:.92rem;">
            Paste a CSV row or fill each field individually — every column in the dataset is editable
          </p>
        </div>""")

        # ── paste row ───────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### 📋 Quick Fill — Paste a CSV Row")
            gr.Markdown(
                "_Paste a comma-separated row from the dataset "
                "(all 23 columns including the `Fire` label) and click **Auto-fill fields**._"
            )
            gr.Markdown(
                "**Column order:** `latitude, longitude, acq_date, NDVI, EVI, LST, "
                "Temperature, Wind, Rainfall, DEM, Slope, Aspect, LandCover, month, "
                "day_of_year, season_code, sin_month, cos_month, sin_doy, cos_doy, "
                "is_peak_fire_season, Fire`"
            )
            paste_box = gr.Textbox(
                label="Paste CSV row here",
                placeholder="31.8846,-97.854,2024-01-04,0.5207,0.2987,26.027,20.611,-0.234,2.415,330.0,2.371,66.996,10.0,1,4,0,0.5,0.866,0.0688,0.9976,0,1",
                lines=3,
            )
            fill_btn = gr.Button("⚡ Auto-fill fields from pasted row", variant="secondary")

        with gr.Row():
            # ── LEFT: all inputs ────────────────────────────────────
            with gr.Column(scale=1):

                with gr.Group():
                    gr.Markdown("### 🤖 Model & Threshold")
                    model_dd   = gr.Dropdown(choices=MODEL_NAMES, value=MODEL_NAMES[0],
                                             label="Model", interactive=True)
                    use_tuned  = gr.Checkbox(value=True,
                                             label=f"Use tuned threshold ({OPT_THRESH:.3f})")
                    cust_thr   = gr.Slider(0.0, 1.0, value=0.5, step=0.01,
                                           label="Custom threshold (when box unchecked)")

                with gr.Group():
                    gr.Markdown("### 📍 Location")
                    with gr.Row():
                        f_lat  = gr.Textbox(label="latitude",  placeholder="31.8846",  value="")
                        f_lon  = gr.Textbox(label="longitude", placeholder="-97.854",   value="")
                    f_date = gr.Textbox(label="acq_date", placeholder="2024-01-04", value="")

                with gr.Group():
                    gr.Markdown("### 🌿 Vegetation Indices")
                    with gr.Row():
                        f_ndvi = gr.Textbox(label="NDVI",  placeholder="0.5207", value="")
                        f_evi  = gr.Textbox(label="EVI",   placeholder="0.2987", value="")

                with gr.Group():
                    gr.Markdown("### 🌡️ Climate / Weather")
                    with gr.Row():
                        f_lst  = gr.Textbox(label="LST (K)",          placeholder="26.027",  value="")
                        f_temp = gr.Textbox(label="Temperature (°C)",  placeholder="20.611",  value="")
                    with gr.Row():
                        f_wind = gr.Textbox(label="Wind (m/s)",        placeholder="-0.234",  value="")
                        f_rain = gr.Textbox(label="Rainfall (mm)",     placeholder="2.415",   value="")

                with gr.Group():
                    gr.Markdown("### ⛰️ Terrain")
                    with gr.Row():
                        f_dem    = gr.Textbox(label="DEM / Elevation (m)", placeholder="330.0",  value="")
                        f_slope  = gr.Textbox(label="Slope (°)",           placeholder="2.371",  value="")
                    f_aspect = gr.Textbox(label="Aspect (°)", placeholder="66.996", value="")

                with gr.Group():
                    gr.Markdown("### 🗺️ Land Cover (MODIS 0–17)")
                    f_lc = gr.Textbox(label="LandCover code", placeholder="10", value="")
                    gr.Markdown(
                        "0=Water · 1=ENF · 2=EBF · 3=DNF · 4=DBF · 5=Mixed Forest · "
                        "6=Closed Shrub · 7=Open Shrub · 8=Woody Savanna · 9=Savanna · "
                        "**10=Grassland** · 11=Wetland · 12=Cropland · 13=Urban · "
                        "14=Mosaic · 15=Snow · 16=Barren · 17=Unclassified"
                    )

                with gr.Group():
                    gr.Markdown("### 📅 Temporal Features")
                    with gr.Row():
                        f_month  = gr.Textbox(label="month",      placeholder="1",  value="")
                        f_doy    = gr.Textbox(label="day_of_year",placeholder="4",  value="")
                        f_season = gr.Textbox(label="season_code (0=Win 1=Spr 2=Sum 3=Fall)",
                                              placeholder="0", value="")

                with gr.Group():
                    gr.Markdown("### 〰️ Cyclical Encodings")
                    with gr.Row():
                        f_sinm = gr.Textbox(label="sin_month", placeholder="0.5",      value="")
                        f_cosm = gr.Textbox(label="cos_month", placeholder="0.866",    value="")
                    with gr.Row():
                        f_sind = gr.Textbox(label="sin_doy",   placeholder="0.0688",   value="")
                        f_cosd = gr.Textbox(label="cos_doy",   placeholder="0.9976",   value="")

                with gr.Group():
                    gr.Markdown("### 🏷️ Labels")
                    f_peak = gr.Textbox(label="is_peak_fire_season (0 or 1)",
                                        placeholder="0", value="")
                    f_fire = gr.Textbox(label="Fire — ACTUAL label from CSV (0 or 1, optional)",
                                        placeholder="1", value="")

                predict_btn = gr.Button("🔍  Predict — Is it FIRE?",
                                        variant="primary", size="lg")

            # ── RIGHT: output ───────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Result")
                result_html = gr.HTML("""
                <div style="background:linear-gradient(135deg,#1e1e2e,#12121a);
                  border:1px solid #333;border-radius:16px;
                  padding:60px 32px;text-align:center;font-family:system-ui;color:#475569;">
                  <div style="font-size:2.5rem;opacity:.35;margin-bottom:14px;">🔥</div>
                  <p style="font-size:1rem;">Fill in the fields on the left<br>
                    (or paste a CSV row above) then click<br>
                    <b style="color:#f97316;">Predict — Is it FIRE?</b>
                  </p>
                </div>""")

        # ── all fields in one list for wiring ───────────────────────
        ALL_FIELDS = [
            f_lat, f_lon, f_date,
            f_ndvi, f_evi, f_lst, f_temp, f_wind, f_rain,
            f_dem, f_slope, f_aspect, f_lc,
            f_month, f_doy, f_season,
            f_sinm, f_cosm, f_sind, f_cosd,
            f_peak, f_fire,
        ]

        # ── auto-fill from pasted row ───────────────────────────────
        def autofill(row_str):
            d = parse_row(row_str)
            return [d.get(c, "") for c in CSV_COLS]

        fill_btn.click(fn=autofill, inputs=[paste_box], outputs=ALL_FIELDS)

        # ── predict ─────────────────────────────────────────────────
        predict_btn.click(
            fn=predict,
            inputs=[model_dd, use_tuned, cust_thr] + ALL_FIELDS,
            outputs=[result_html],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860,
                show_error=True, inbrowser=True)
