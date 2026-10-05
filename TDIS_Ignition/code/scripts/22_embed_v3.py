"""
TDIS Forecast — Step 22: build the LIVE-FORECAST-ONLY dashboard (v3).

Unlike v2 (21_embed_explorer.py), v3 drops the historical day-scrubber and the
static-hazard-only tab entirely -- it is always showing the current 24h/48h
live forecast, across three tabs: Wildfire Risk, Ignition, Fire Weather Index.
No tdis_dashboard_data.json (82MB historical blob) is needed at all.

Injects into dashboard/tdis_fire_dashboard_v3.html:
  /*__MRES__*/      <- explorer_static_multires.json  (multi-res STATIC HAZARD only,
                        used by the Wildfire Risk tab's zoom-refine)
  /*__FORECASTS__*/ <- newest TWO forecast_*.json      (24h + 48h; 72h retired)

Output: dashboard/tdis_fire_explorer_v3_standalone.html
"""
import json
from pathlib import Path
TF = Path(__file__).resolve().parent.parent
D = TF/"dashboard"
def log(m): print(m, flush=True)

tpl = (D/"tdis_fire_dashboard_v3.html").read_text()
for ph in ["/*__MRES__*/null", "/*__FORECASTS__*/[]"]:
    assert ph in tpl, f"placeholder missing: {ph}"

mres = (D/"explorer_static_multires.json").read_text()
fps = sorted(D.glob("forecast_*.json"), key=lambda p: p.name)[-2:]   # 24h + 48h only
fcs = [json.load(open(p)) for p in fps]
log(f"embedding MRES {len(mres)/1e6:.1f} MB + {len(fcs)} forecasts {[f['target']+' ('+str(f['lead_h'])+'h)' for f in fcs]}")

out = tpl.replace("/*__FORECASTS__*/[]", "/*__FORECASTS__*/"+json.dumps(fcs, separators=(',',':')))
out = out.replace("/*__MRES__*/null", "/*__MRES__*/"+mres)
p = D/"tdis_fire_explorer_v3_standalone.html"
p.write_text(out)
log(f"Saved {p.name} ({p.stat().st_size/1e6:.1f} MB) — drag-and-drop, live-forecast-only, 3 tabs")
