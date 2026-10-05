"""
TDIS Forecast — Step 24: build the v4 standalone dashboard.
Injects into dashboard/tdis_fire_dashboard_v4.html:
  /*__MRES__*/    <- explorer_static_multires.json (Wildfire Risk zoom)
  /*__V4__*/      <- newest TWO v4_forecast_*.json (24h + 48h, tiered)
  /*__DETAIL__*/  <- matching v4_detail8_*.json (res-8 High/VHigh zoom payload)
Output: dashboard/tdis_fire_explorer_v4_standalone.html
"""
import json
from pathlib import Path
TF = Path(__file__).resolve().parent.parent
D = TF / "dashboard"
def log(m): print(m, flush=True)

tpl = (D / "tdis_fire_dashboard_v4.html").read_text()
for ph in ["/*__MRES__*/null", "/*__V4__*/[]", "/*__DETAIL__*/[]"]:
    assert ph in tpl, f"placeholder missing: {ph}"

mres = (D / "explorer_static_multires.json").read_text()
fps = sorted(D.glob("v4_forecast_*.json"), key=lambda p: p.name)[-2:]
v4 = [json.load(open(p)) for p in fps]
det = []
for f in v4:
    dp = D / f"v4_detail8_{f['target']}_lead{f['lead_h']}h.json"
    det.append(json.load(open(dp)) if dp.exists() else None)
log(f"embedding MRES {len(mres)/1e6:.1f} MB + {len(v4)} tiered forecasts "
    f"{[f['target'] for f in v4]} + {sum(1 for d in det if d)} detail layers")

out = tpl.replace("/*__V4__*/[]", "/*__V4__*/" + json.dumps(v4, separators=(',', ':')))
out = out.replace("/*__DETAIL__*/[]", "/*__DETAIL__*/" + json.dumps(det, separators=(',', ':')))
out = out.replace("/*__MRES__*/null", "/*__MRES__*/" + mres)
p = D / "tdis_fire_explorer_v4_standalone.html"
p.write_text(out)
log(f"Saved {p.name} ({p.stat().st_size/1e6:.1f} MB)")
