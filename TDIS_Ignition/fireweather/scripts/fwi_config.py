"""
TDIS Forecast — Fire-Weather Index config: THE TUNABLE KNOBS.

Two knobs live here:

KNOB 1 — FWI_WEIGHTS (additive composite; the original wind knob)
  Old weights (pre-validation): erc 0.50, vpd 0.30, wind 0.20
  Current (wind-forward):       erc 0.30, vpd 0.30, wind 0.40

KNOB 2 — FWI_MODE (which fire-weather formulation drives risk)
  'composite'  additive weighted sum (original TDIS index, weights above)
  'hwp_noaa'   NOAA GSL Hourly Wildfire Potential, EXACT Eq. 3 of
               James et al. 2025, Wea. Forecasting, doi:10.1175/WAF-D-24-0068.1:
                   HWP = 0.213 * G^1.50 * VPD^0.73 * (1-M)^5.10 * S
               G = 10-m gust (m/s, floor 3), VPD in hPa, M = soil-moisture
               availability (0-1), S = snow term (=1 for the TX archive).
               On the gridMET (historical) path we must proxy:
                   G ≈ GUST_FACTOR * daily-mean wind ;  M ≈ fm100 / FM100_SAT
               On the live HRRR path it is exact: real GUST + real MSTAV.
  'hwp_tx'     Same multiplicative form, coefficients FIT TO TEXAS VIIRS
               fire activity (scripts/16_fit_hwp_tx.py). Loaded from
               data/hwp_params.json; falls back to NOAA coefficients if
               the fit hasn't been run yet.

All three variants are ALWAYS computed side-by-side by 15_reweight_fwi.py and
13_model_forecast_day.py so the dashboard can toggle between them; FWI_MODE
sets which one scripts treat as primary (e.g. 14_event_validation.py).

HWP is unbounded, so for dashboard use each HWP variant is divided by a fixed
reference (its p99.5 over the 2024-26 TX archive, stored in hwp_params.json)
and clipped to 0-1 — a relative scale, same spirit as the composite index.
"""
import json
import numpy as np
from pathlib import Path

_TF = Path(__file__).resolve().parent.parent

# ══ KNOB 1: composite weights (must sum to 1.0) ══
FWI_WEIGHTS = {'erc': 0.30, 'vpd': 0.30, 'wind': 0.40}
SCALE = {'erc': 100.0, 'vpd': 5.0, 'wind': 12.0, 'gust': 22.0}

# ══ KNOB 2: which formulation is primary ══
FWI_MODE = 'composite'          # 'composite' | 'hwp_noaa' | 'hwp_tx'

# HWP constants
HWP_NOAA = {'C': 0.213, 'a': 1.50, 'b': 0.73, 'c': 5.10}   # James et al. 2025 Eq. 3
GUST_FACTOR = 1.5               # daily-mean wind -> gust proxy (gridMET path)
GUST_FLOOR = 3.0                # m/s, per the paper
FM100_SAT = 30.0                # fm100 (%) treated as saturation -> M = fm100/30

def _load_hwp_params():
    p = _TF / "data" / "hwp_params.json"
    if p.exists():
        return json.load(open(p))
    # pre-fit fallback: TX = NOAA coefficients; refs make output land in ~0-1
    return {'tx': {**HWP_NOAA, 'ref': 60.0}, 'noaa': {'ref': 60.0}}

HWP_PARAMS = _load_hwp_params()

def _n(x, s): return np.clip(np.asarray(x, float) / s, 0, 1)

# ── core HWP form: C * G^a * VPD^b * dry^c   (VPD in hPa; dry = 1-M) ──
def _hwp_raw(G, vpd_hpa, dry, p):
    G = np.maximum(np.asarray(G, float), GUST_FLOOR)
    vpd_hpa = np.clip(np.asarray(vpd_hpa, float), 0.01, None)
    dry = np.clip(np.asarray(dry, float), 0.0, 1.0)
    return p['C'] * G**p['a'] * vpd_hpa**p['b'] * dry**p['c']

def hwp_from_components(G, vpd_kpa, M, variant='noaa'):
    """Normalized 0-1 HWP. G gust m/s, vpd in kPa, M = moisture availability 0-1."""
    p = HWP_PARAMS['tx'] if variant == 'tx' else {**HWP_NOAA, **HWP_PARAMS['noaa']}
    raw = _hwp_raw(G, np.asarray(vpd_kpa, float) * 10.0, 1.0 - np.asarray(M, float), p)
    return np.clip(raw / p.get('ref', 60.0), 0, 1)

# ── the three variants on the gridMET (historical) path ──
def fwi_composite_gridmet(erc, vpd, wind):
    w = FWI_WEIGHTS
    return np.clip(w['erc']*_n(erc, SCALE['erc']) + w['vpd']*_n(vpd, SCALE['vpd'])
                   + w['wind']*_n(wind, SCALE['wind']), 0, 1)

def hwp_gridmet(vpd, wind, fm100, variant='noaa'):
    """HWP on gridMET daily data. Proxies: G = GUST_FACTOR*wind, M = fm100/FM100_SAT."""
    G = GUST_FACTOR * np.asarray(wind, float)
    M = np.clip(np.asarray(fm100, float) / FM100_SAT, 0, 1)
    return hwp_from_components(G, vpd, M, variant=variant)

def fwi_gridmet(erc, vpd, wind, fm100=None):
    """Primary index per FWI_MODE (backward-compatible signature)."""
    if FWI_MODE == 'composite' or fm100 is None:
        return fwi_composite_gridmet(erc, vpd, wind)
    return hwp_gridmet(vpd, wind, fm100, variant='tx' if FWI_MODE == 'hwp_tx' else 'noaa')

# ── the three variants on the live HRRR path ──
def fwi_composite_hrrr(vpd, wind, gust=None):
    """Original composite on HRRR (no ERC -> weight folds into vpd+wind)."""
    w = FWI_WEIGHTS
    tot = w['vpd'] + w['wind']
    windterm = _n(gust, SCALE['gust']) if gust is not None else _n(wind, SCALE['wind'])
    return np.clip((w['vpd']/tot)*_n(vpd, SCALE['vpd']) + (w['wind']/tot)*windterm, 0, 1)

def hwp_hrrr(vpd, gust, soilm, variant='noaa'):
    """HWP on live HRRR: EXACT inputs — real gust + real soil-moisture availability (MSTAV)."""
    return hwp_from_components(gust, vpd, soilm, variant=variant)

def fwi_hrrr(vpd, wind, gust=None, soilm=None):
    """Primary index per FWI_MODE (backward-compatible signature)."""
    if FWI_MODE == 'composite' or soilm is None:
        return fwi_composite_hrrr(vpd, wind, gust)
    G = gust if gust is not None else GUST_FACTOR * np.asarray(wind, float)
    return hwp_hrrr(vpd, G, soilm, variant='tx' if FWI_MODE == 'hwp_tx' else 'noaa')
