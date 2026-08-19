"""
TDIS Forecast — Step 16: FIT the Texas-calibrated HWP coefficients.

Replicates the methodology of James et al. 2025 (Wea. Forecasting,
doi:10.1175/WAF-D-24-0068.1) but on TEXAS data:
  - their target: normalized FRP of 9 large western-US wildfires (1,615 hourly pts)
  - our target:   daily VIIRS FRP per res-5 cell-day, 2024-2026 TX archive
                  (fire-prone cells only: cells with >=1 VIIRS detection ever)
  - same functional form:  HWP = C * G^a * VPD^b * dry^c
       G   = gust proxy (GUST_FACTOR * gridMET daily wind, floor 3 m/s)
       VPD = hPa
       dry = 1 - M,  M = fm100 / FM100_SAT
  - fitting: log-linear least squares on large-fire episode days (the paper's
    raw-space curve_fit degenerates on our many-small-fires daily data)

Outputs data/hwp_params.json:
  { "tx":   {C, a, b, c, ref, fit_r, n},        <- fitted + p99.5 normalizer
    "noaa": {ref} }                              <- p99.5 normalizer for exact Eq. 3

Then run 15_reweight_fwi.py to rebuild the dashboard's per-day variant arrays.
"""
import json, warnings
import numpy as np, pandas as pd, h3
from pathlib import Path
warnings.filterwarnings('ignore')
TF = Path(__file__).resolve().parent.parent
def log(m): print(m, flush=True)
import fwi_config as FC

# ── 1. daily VIIRS FRP per res-5 cell-day ──
log("Loading VIIRS detections...")
v = pd.read_parquet(TF/"data"/"labels_viirs"/"viirs_tx_h3.parquet",
                    columns=['h3_cell','date','frp'])
v['date'] = pd.to_datetime(v['date'])
v['h3_5'] = [h3.cell_to_parent(c,5) for c in v['h3_cell'].values]
frp = v.groupby(['h3_5','date']).agg(frp=('frp','sum')).reset_index()
fire_cells = set(frp['h3_5'].unique())
log(f"  {len(frp):,} fire cell-days across {len(fire_cells):,} res-5 cells")

# ── 2. weather components (2024-2026 archive) ──
comp = pd.read_parquet(TF/"data"/"fwi_components_res5.parquet")
comp['date'] = pd.to_datetime(comp['date'])
assert 'fm100' in comp.columns, "components cache lacks fm100 — re-run 09 first"
# EPISODE-WINDOWED fit table — mirrors the paper (they fit hours AT active fires, not
# a mostly-zero background): keep cell-days within +/- EPISODE_PAD days of any VIIRS
# detection in that cell. Zero-FRP days inside an episode ARE included (lulls carry
# signal); the 96%-zero background outside episodes is excluded.
EPISODE_PAD = 7
comp = comp.dropna(subset=['vs','vpd','fm100'])
pad = []
for k in range(-EPISODE_PAD, EPISODE_PAD+1):
    p = frp[['h3_5','date']].copy(); p['date'] = p['date'] + pd.Timedelta(days=k); pad.append(p)
episode = pd.concat(pad, ignore_index=True).drop_duplicates()
d = comp.merge(episode, on=['h3_5','date'], how='inner') \
        .merge(frp, on=['h3_5','date'], how='left')
d['frp'] = d['frp'].fillna(0.0)
log(f"  fit table (episode-windowed +/-{EPISODE_PAD}d): {len(d):,} cell-days "
    f"({(d.frp>0).mean()*100:.1f}% with fire)")

# ── 2b. LARGE-fire episodes only (paper trained on LARGE western wildfires) ──
# keep cells whose peak daily FRP reaches the top tier; small ag/trash burns carry
# no weather signal and swamp the fit otherwise.
LARGE_FRP = 500.0    # MW summed per res-5 cell-day at episode peak
peak = d.groupby('h3_5')['frp'].max()
large_cells = set(peak[peak >= LARGE_FRP].index)
d = d[d['h3_5'].isin(large_cells)]
log(f"  large-fire subset: {len(large_cells):,} cells with peak day >= {LARGE_FRP:.0f} MW "
    f"-> {len(d):,} cell-days ({(d.frp>0).mean()*100:.1f}% with fire)")

# ── 3. predictors in HWP units ──
G   = np.maximum(FC.GUST_FACTOR * d['vs'].values, FC.GUST_FLOOR)
VPD = np.clip(d['vpd'].values * 10.0, 0.01, None)            # kPa -> hPa
DRY = np.clip(1.0 - np.clip(d['fm100'].values / FC.FM100_SAT, 0, 1), 1e-3, 1)
y = np.minimum(d['frp'].values, np.quantile(d['frp'].values, 0.999))

def hwp_fn(X, C, a, b, c):
    g, vv, dd = X
    return C * g**a * vv**b * np.clip(dd, 1e-3, 1)**c

# The power law is LINEAR in log space -> fit exponents by least squares on
# fire days (frp>0), which is stable for heavy-tailed FRP (raw-space curve_fit
# degenerates on our data; documented deviation from the paper's raw-space fit).
log("Fitting log-linear least squares (power law linearized)...")
m = d['frp'].values > 0
Xl = np.c_[np.ones(m.sum()), np.log(G[m]), np.log(VPD[m]), np.log(DRY[m])]
yl = np.log(y[m])
coef, *_ = np.linalg.lstsq(Xl, yl, rcond=None)
C, a, b, c = float(np.exp(coef[0])), float(coef[1]), float(coef[2]), float(coef[3])
a = max(a, 0.05); b = max(b, 0.05); c = max(c, 0.05)          # physical-sign guard
pred = hwp_fn((G, VPD, DRY), C, a, b, c)
r  = float(np.corrcoef(pred, y)[0, 1])
rl = float(np.corrcoef(np.log(pred[m]), yl)[0, 1])
log(f"  TX fit:  HWP = {C:.4f} * G^{a:.2f} * VPD^{b:.2f} * dry^{c:.2f}   (r_raw={r:.3f}, r_log={rl:.3f}, n={len(y):,})")
log(f"  NOAA:    HWP = 0.213  * G^1.50 * VPD^0.73 * dry^5.10             (their r=0.44, n=1,615)")
popt = (C, a, b, c)

# ── 4. normalization refs (p99.5 of each raw variant over the full archive) ──
Ga  = np.maximum(FC.GUST_FACTOR * comp['vs'].values, FC.GUST_FLOOR)
Va  = np.clip(comp['vpd'].values * 10.0, 0.01, None)
Da  = 1.0 - np.clip(comp['fm100'].values / FC.FM100_SAT, 0, 1)
p_noaa = (FC.HWP_NOAA['C'], FC.HWP_NOAA['a'], FC.HWP_NOAA['b'], FC.HWP_NOAA['c'])
ref_tx   = float(np.quantile(hwp_fn((Ga, Va, Da), *popt), 0.995))
ref_noaa = float(np.quantile(hwp_fn((Ga, Va, Da), *p_noaa), 0.995))
log(f"  refs (p99.5 raw): tx={ref_tx:.2f}  noaa={ref_noaa:.2f}")

out = {'tx':   {'C': C, 'a': a, 'b': b, 'c': c, 'ref': ref_tx,
                'fit_r': round(r, 3), 'n': int(len(y)),
                'note': 'log-linear fit to daily VIIRS FRP, large-fire episodes (peak>=500MW), TX 2024-2026', 'fit_r_log': round(rl, 3)},
       'noaa': {'ref': ref_noaa,
                'note': 'James et al. 2025 Eq.3 coefficients; ref = p99.5 over TX archive'}}
json.dump(out, open(TF/"data"/"hwp_params.json", 'w'), indent=1)
log(f"Saved data/hwp_params.json — now run 15_reweight_fwi.py")
