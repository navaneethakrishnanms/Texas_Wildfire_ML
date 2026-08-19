"""
TDIS Forecast — Step 17: rebuild the drag-and-drop STANDALONE dashboard.

Injects dashboard/tdis_dashboard_data.json (per-day data, all FWI variants) and every
dashboard/forecast_*.json (live forecasts, sorted by target date) into the template's
/*__DATA__*/ and /*__FORECASTS__*/ placeholders.

Run after 15_reweight_fwi.py and/or 13_model_forecast_day.py.
"""
import json
from pathlib import Path
TF = Path(__file__).resolve().parent.parent
D = TF/"dashboard"
def log(m): print(m, flush=True)

tpl = (D/"tdis_fire_dashboard.html").read_text()
assert "/*__DATA__*/" in tpl and "/*__FORECASTS__*/" in tpl, "placeholders missing from template"

data = (D/"tdis_dashboard_data.json").read_text()
fps = sorted(D.glob("forecast_*.json"), key=lambda p: p.name)[-2:]   # newest 2 = 24/48h (72h retired: weakest skill + GFS soil gaps)
fcs = [json.load(open(p)) for p in fps]
log(f"embedding data ({len(data)/1e6:.1f} MB) + {len(fcs)} forecasts: {[f['target'] for f in fcs]}")

out = tpl.replace("/*__FORECASTS__*/[]", "/*__FORECASTS__*/" + json.dumps(fcs, separators=(',',':')))
out = out.replace("/*__DATA__*/null", "/*__DATA__*/" + data)
p = D/"tdis_fire_dashboard_standalone.html"
p.write_text(out)
log(f"Saved {p.name} ({p.stat().st_size/1e6:.1f} MB) — double-click / drag-and-drop ready")
