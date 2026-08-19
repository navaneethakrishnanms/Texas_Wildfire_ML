"""
TDIS Forecast — Step 21: build the ZOOM-ADAPTIVE standalone dashboard (v2).

Injects into dashboard/tdis_fire_dashboard_v2.html:
  /*__MRES__*/      <- explorer_static_multires.json  (multi-res statics, step 20)
  /*__DATA__*/      <- tdis_dashboard_data.json       (res-5 dynamics, steps 08/09/15)
  /*__FORECASTS__*/ <- newest TWO forecast_*.json     (24h + 48h; 72h retired)

Output: dashboard/tdis_fire_explorer_standalone.html (drag-and-drop)
"""
import json
from pathlib import Path
TF = Path(__file__).resolve().parent.parent
D = TF/"dashboard"
def log(m): print(m, flush=True)

tpl = (D/"tdis_fire_dashboard_v2.html").read_text()
for ph in ["/*__MRES__*/null","/*__DATA__*/null","/*__FORECASTS__*/[]"]:
    assert ph in tpl, f"placeholder missing: {ph}"

mres = (D/"explorer_static_multires.json").read_text()
data = (D/"tdis_dashboard_data.json").read_text()
fps = sorted(D.glob("forecast_*.json"), key=lambda p: p.name)[-2:]   # 24h + 48h only
fcs = [json.load(open(p)) for p in fps]
log(f"embedding MRES {len(mres)/1e6:.1f} MB + DATA {len(data)/1e6:.1f} MB + {len(fcs)} forecasts {[f['target'] for f in fcs]}")

out = tpl.replace("/*__FORECASTS__*/[]", "/*__FORECASTS__*/"+json.dumps(fcs, separators=(',',':')))
out = out.replace("/*__MRES__*/null", "/*__MRES__*/"+mres)
out = out.replace("/*__DATA__*/null", "/*__DATA__*/"+data)
p = D/"tdis_fire_explorer_standalone.html"
p.write_text(out)
log(f"Saved {p.name} ({p.stat().st_size/1e6:.1f} MB) — drag-and-drop, zoom-adaptive res 4-8")
