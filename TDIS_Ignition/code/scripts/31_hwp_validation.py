"""
TDIS Forecast — Step 31: fire-weather (HWP) validation battery, pre-handoff.
HWP is a FORMULA product, not a fitted classifier, so the battery differs
from the ignition model's:

A. Implementation correctness: reproduce DEMO_SMOKEHOUSE.md's documented
   worked example (G=11.0 m/s, VPD=7.8 hPa, M=0.300 -> normalized ~0.8)
   through the actual served code path (fwi_config.hwp_from_components).
B. Input QC: fwi_components_res5 ranges (wind>=0, vpd sane kPa, fm100 in
   [2,30]-ish, no fill-value artifacts), date coverage/gaps.
C. Event battery: NOAA-HWP percentile of each named-fire event's cells on
   its event day vs statewide that day + vs 20 random quiet days.
D. Climatology: monthly statewide mean/p95 of normalized NOAA-HWP --
   plausibility (TX fire weather peaks late-winter wind season + summer).
E. Normalizer behavior: share of cell-days >1.0 (p99.5 ref), saturation.

Output: results printed + data/hwp_validation.json
"""
import sys, json
import numpy as np, pandas as pd, h3
from pathlib import Path
TF = Path("/net/cven-mosta-nas.engr.tamu.edu/volume2/NASdata2/miguel_shared/alphaearth_nds/TDIS_Forecast")
sys.path.insert(0, str(TF / "scripts"))
import fwi_config as FC

EVENTS = [
    ("Smokehouse Creek", '2024-02-26', 35.5, 36.3, -101.6, -100.2),
    ("Windy Deuce", '2024-02-27', 35.55, 35.95, -102.0, -101.3),
    ("Crabapple", '2025-03-15', 30.15, 30.75, -99.15, -98.55),
    ("Lavender", '2026-02-17', 35.1, 35.8, -102.9, -101.4),
    ("Hunggate", '2026-05-14', 34.55, 35.25, -102.7, -101.4),
]
out = {}


def log(m): print(m, flush=True)


if __name__ == '__main__':
    # ---- A. worked-example reproduction ----
    log("=== A. IMPLEMENTATION: DEMO_SMOKEHOUSE worked example ===")
    # doc: G = 11.0 m/s, VPD = 7.8 hPa (=0.78 kPa), M = 0.300 -> "FWI ~ 0.8"
    val = float(FC.hwp_from_components(np.array([11.0]), np.array([0.78]),
                                        np.array([0.300]), variant='noaa')[0])
    ok_A = abs(val - 0.8) < 0.1
    log(f"  served code path: hwp_from_components(G=11.0, vpd=0.78kPa, M=0.30) = {val:.3f} "
        f"(doc says ~0.8) -> {'PASS' if ok_A else 'FAIL'}")
    out['worked_example'] = dict(value=round(val, 3), documented=0.8, passed=bool(ok_A))

    # ---- B. input QC ----
    log("\n=== B. INPUT QC (fwi_components_res5) ===")
    comp = pd.read_parquet(TF / "data/fwi_components_res5.parquet")
    comp['date'] = pd.to_datetime(comp.date)
    checks = {
        'vs_negative': int((comp.vs < 0).sum()),
        'vpd_negative': int((comp.vpd < 0).sum()),
        'vpd_over_10kpa': int((comp.vpd > 10).sum()),
        'fm100_below_1': int((comp.fm100 < 1).sum()),
        'fm100_over_40': int((comp.fm100 > 40).sum()),
        'nulls': int(comp[['vs', 'vpd', 'fm100']].isna().sum().sum()),
    }
    days = comp.date.dt.normalize().nunique()
    span = (comp.date.max() - comp.date.min()).days + 1
    checks['date_gaps'] = int(span - days)
    ok_B = all(v == 0 for k, v in checks.items() if k != 'nulls')
    log(f"  {checks}  span {comp.date.min().date()}..{comp.date.max().date()}")
    log(f"  -> {'PASS' if ok_B else 'CHECK'} (nulls are drop-not-impute rows, known)")
    out['input_qc'] = checks

    # ---- C. event battery ----
    log("\n=== C. EVENT BATTERY (NOAA-HWP percentile on event day) ===")
    comp = comp.dropna(subset=['vs', 'vpd', 'fm100'])
    comp['hwp'] = FC.hwp_gridmet(comp['vpd'].values, comp['vs'].values,
                                   comp['fm100'].values, variant='noaa')
    ll = {c: h3.cell_to_latlng(c) for c in comp.h3_5.unique()}
    comp['lat'] = comp.h3_5.map(lambda c: ll[c][0])
    comp['lon'] = comp.h3_5.map(lambda c: ll[c][1])
    ev_out = {}
    for name, ds, la0, la1, lo0, lo1 in EVENTS:
        d = pd.Timestamp(ds)
        day = comp[comp.date == d]
        if len(day) == 0:
            log(f"  {name}: no component data for {ds}"); continue
        box = day[(day.lat.between(la0, la1)) & (day.lon.between(lo0, lo1))]
        pct = float((day.hwp.values[None, :] <= box.hwp.values[:, None]).mean(axis=1).mean() * 100)
        ev_out[name] = round(pct, 1)
        log(f"  {name} ({ds}): event-box mean statewide HWP percentile = {pct:.1f}")
    rng = np.random.RandomState(42)
    quiet = rng.choice(comp.date.dt.normalize().unique(), 20, replace=False)
    qp = []
    for d in quiet:
        day = comp[comp.date == d]
        qp.append(float(day.hwp.mean()))
    log(f"  random-day statewide mean HWP: {np.mean(qp):.3f} (baseline)")
    out['events_percentile'] = ev_out

    # ---- D. climatology ----
    log("\n=== D. MONTHLY CLIMATOLOGY (normalized NOAA-HWP) ===")
    m = comp.groupby(comp.date.dt.month)['hwp'].agg(['mean', lambda s: s.quantile(0.95)])
    m.columns = ['mean', 'p95']
    log(m.round(3).to_string())
    out['monthly'] = {int(k): dict(mean=round(v['mean'], 3), p95=round(v['p95'], 3))
                      for k, v in m.iterrows()}

    # ---- E. normalizer ----
    over1 = float((comp.hwp > 1).mean() * 100)
    log(f"\n=== E. NORMALIZER: {over1:.2f}% of cell-days exceed 1.0 (p99.5 ref -> expect ~0.5%) ===")
    out['pct_over_1'] = round(over1, 2)

    with open(TF / "data/hwp_validation.json", 'w') as f:
        json.dump(out, f, indent=2)
    log("\nSaved data/hwp_validation.json")
