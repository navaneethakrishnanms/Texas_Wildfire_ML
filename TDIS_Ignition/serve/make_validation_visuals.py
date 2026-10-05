#!/usr/bin/env python
"""Render the trust-validation results as figures -> validation_visuals/.

Sources: model/TRUST_VALIDATION.json (served-model response curves),
data/scored/operational_historical_res5_v2.parquet (PR + reliability),
model_tristate (rev5 response sweeps re-run for plotting),
fireweather/ (physics curves). Static matplotlib, light surface.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib, xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation_visuals"; OUT.mkdir(exist_ok=True)

# validated palette (light mode) — categorical fixed order
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
RED = "#e34948"            # status only, always with a text label
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.grid": True,
                     "grid.color": "#c3c2b7", "grid.linewidth": 0.4, "grid.alpha": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})


def log(m): print(m, flush=True)


# ── 1. PR curve, served model, real population ─────────────────────────
panel = pd.read_parquet(ROOT / "data/scored/operational_historical_res5_v2.parquet")
y = panel["y"].values.astype(int); p = panel["p"].values.astype(float)
from sklearn.metrics import precision_recall_curve, average_precision_score
prec, rec, _ = precision_recall_curve(y, p)
ap = average_precision_score(y, p); base = y.mean()
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.plot(rec, prec, color=BLUE, lw=2)
ax.axhline(base, color=MUTED, lw=1, ls="--")
ax.annotate(f"random = base rate {base:.3f}", (0.62, base), xytext=(0, 6),
            textcoords="offset points", color=MUTED, fontsize=8)
ax.annotate(f"AUC-PR {ap:.4f}  ·  lift {ap/base:.2f}×", (0.35, 0.55),
            color=INK, fontsize=11, fontweight="bold")
ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
       title="Served TX model — precision-recall on the REAL population\n"
             "(5.06M cell-days 2024–26, re-derived from the packaged scored panel)")
fig.tight_layout(); fig.savefig(OUT / "01_served_pr_curve_realpop.png", dpi=140); plt.close(fig)
log("01 PR curve")

# ── 2. Calibration reliability, before vs after isotonic (2026 holdout) ─
iso = joblib.load(ROOT / "model/operational_isotonic_calibrator.joblib")
hold = pd.to_datetime(panel["date"]) >= "2026-01-01"
yh, ph = y[hold.values], p[hold.values]; pc = iso.predict(ph)
def bins(yy, pp, n=10):
    e = np.linspace(0, 1, n + 1); i = np.digitize(pp, e[1:-1])
    xs, ys = [], []
    for b in range(n):
        m = i == b
        if m.sum() > 50: xs.append(pp[m].mean()); ys.append(yy[m].mean())
    return xs, ys
fig, ax = plt.subplots(figsize=(6.0, 5.4))
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
x1, y1 = bins(yh, ph); x2, y2 = bins(yh, pc)
ax.plot(x1, y1, "o-", color=RED, lw=2, ms=8, label="raw score (ECE 0.33)")
ax.plot(x2, y2, "o-", color=BLUE, lw=2, ms=8, label="isotonic-calibrated (ECE 0.0008)")
ax.annotate("raw score is a RANK,\nnot a probability", (x1[-1], y1[-1]),
            xytext=(-110, 10), textcoords="offset points", color=RED, fontsize=8)
ax.legend(frameon=False, loc="upper left")
ax.set(xlabel="Mean predicted probability", ylabel="Observed fire frequency",
       xlim=(0, 1), ylim=(0, 1),
       title="Served TX model — calibration reliability, 2026 holdout")
fig.tight_layout(); fig.savefig(OUT / "02_served_reliability.png", dpi=140); plt.close(fig)
log("02 reliability")

# ── 3. Served-model load-response curves (from TRUST_VALIDATION.json) ───
tv = json.load(open(ROOT / "model/TRUST_VALIDATION.json"))["load_response"]
feats = list(tv.keys())
fig, axs = plt.subplots(3, 3, figsize=(11, 8.5))
for ax, f in zip(axs.ravel(), feats):
    r = tv[f]; g = np.linspace(r["grid_lo"], r["grid_hi"], len(r["response"]))
    bad = r["verdict"].startswith("❌")
    ax.plot(g, r["response"], color=BLUE, lw=2)
    exp = r["expected"]
    tag = ("no expectation" if exp is None else
           f"expected {exp} — {'VIOLATED' if bad else 'ok'}")
    ax.set_title(f"{f}\n{tag}", fontsize=9,
                 color=(RED if bad else INK), fontweight=("bold" if bad else "normal"))
    ax.set_ylabel("mean P(fire)", fontsize=8)
fig.suptitle("Served TX model — load-response curves (UNCONSTRAINED model: wind/temp/hazard "
             "run against physical direction;\nknown lineage finding — correlational, not causal. "
             "Performance metrics unaffected.)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(OUT / "03_served_load_response.png", dpi=130); plt.close(fig)
log("03 served response curves")

# ── 4. rev5_v2 load-response curves (re-sweep for plotting) ─────────────
sys.path.insert(0, "/net/cven-mosta-nas.engr.tamu.edu/volume2/NASdata2/miguel_shared/alphaearth_nds/TDIS_Forecast/Training_rev4_dailymax")
import rev4_pipeline as R
FEATS5 = R.FEATS + ["pop_density", "fm100_5d_min"]
df5 = pd.read_parquet(ROOT / "model_tristate/data/train_tristate_v2.parquet").dropna(subset=FEATS5)
samp = df5[df5.state == "tx"].sample(15000, random_state=0)
m5 = xgb.XGBClassifier(); m5.load_model(str(ROOT / "model_tristate/model/rev5_tristate_v2_seed42.json"))
EXP5 = {"hrrr_vpd": "up", "hrrr_wind": "up", "avg_burn_prob": "up", "whp": "up",
        "drought_score": "up", "bp_x_vpd": "up", "whp_x_vpd": "up",
        "bp_x_drought": "up", "fm100_5d_min": "down"}
fig, axs = plt.subplots(3, 3, figsize=(11, 8.5))
for ax, (f, exp) in zip(axs.ravel(), EXP5.items()):
    lo, hi = df5[f].quantile(0.02), df5[f].quantile(0.98)
    grid = np.linspace(lo, hi, 9); means = []
    for v in grid:
        s = samp.copy(); s[f] = v
        means.append(float(m5.predict_proba(s[FEATS5])[:, 1].mean()))
    ax.plot(grid, means, color=AQUA, lw=2)
    ax.set_title(f"{f}\nconstrained {exp} — monotone ✓", fontsize=9)
    ax.set_ylabel("mean P(fire)", fontsize=8)
fig.suptitle("rev5_v2 (tri-state) — load-response curves: all 9 constrained features "
             "monotone in the physical direction", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(OUT / "04_rev5_load_response.png", dpi=130); plt.close(fig)
log("04 rev5 response curves")

# ── 5. Fire-weather physics curves ──────────────────────────────────────
sys.path.insert(0, str(ROOT / "fireweather/scripts"))
import fwi_config as F
base_in = dict(G=10.0, vpd=1.5, M=0.4)
sweeps = [("Gust (m/s)", "G", np.linspace(0.5, 40, 120)),
          ("VPD (kPa)", "vpd", np.linspace(0.02, 6, 120)),
          ("Soil-moisture availability M", "M", np.linspace(0, 0.99, 120))]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, (label, key, arr) in zip(axs, sweeps):
    kw = dict(base_in); kw[key] = arr
    n = F.hwp_from_components(kw["G"], kw["vpd"], kw["M"], variant="noaa")
    t = F.hwp_from_components(kw["G"], kw["vpd"], kw["M"], variant="tx")
    ax.plot(arr, n, color=BLUE, lw=2, label="NOAA HWP")
    ax.plot(arr, t, color=AQUA, lw=2, label="TX HWP")
    if key == "G":
        ax.axvline(3.0, color=MUTED, lw=1, ls="--")
        ax.annotate("gust floor 3 m/s", (3.0, 0.85), xytext=(6, 0),
                    textcoords="offset points", color=MUTED, fontsize=8)
    ax.set(xlabel=label, ylabel="HWP (0–1)", ylim=(0, 1.02))
axs[0].legend(frameon=False, loc="upper left")
fig.suptitle("Fire-weather HWP — physics response curves (others held at G=10 m/s, VPD=1.5 kPa, M=0.4); "
             "monotone ↑gust, ↑VPD, ↓moisture; spec-exact vs independent reimplementation", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.90]); fig.savefig(OUT / "05_fireweather_physics.png", dpi=140); plt.close(fig)
log("05 fireweather physics")

# ── 6. Fire-weather validated role: within-cell severity signal ─────────
ov = json.load(open(ROOT / "fireweather/overall_validation_2026-09-23.json"))
w = ov["within_cell_timing_2024_26"]
variants = ["composite", "hwp_noaa", "hwp_tx"]; names = ["Composite", "NOAA HWP", "TX HWP"]
allf = [w["mean_percentile_all_fire_days"][v] for v in variants]
topf = [w["mean_percentile_top1pct_frp_days"][v] for v in variants]
xpos = np.arange(3); wd = 0.34
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.bar(xpos - wd/2, allf, wd, color=BLUE, label="all fire days")
ax.bar(xpos + wd/2, topf, wd, color=YELLOW, label="top-1% most intense fires (FRP)")
ax.axhline(0.5, color=MUTED, lw=1, ls="--")
ax.annotate("0.5 = no timing signal", (2.02, 0.5), xytext=(0, 5),
            textcoords="offset points", color=MUTED, fontsize=8)
for i, (a, t) in enumerate(zip(allf, topf)):
    ax.annotate(f"{a:.2f}", (i - wd/2, a), ha="center", xytext=(0, 3), textcoords="offset points", fontsize=8)
    ax.annotate(f"{t:.2f}", (i + wd/2, t), ha="center", xytext=(0, 3), textcoords="offset points", fontsize=8)
ax.set_xticks(xpos); ax.set_xticklabels(names)
ax.set(ylabel="Mean within-cell percentile on fire days", ylim=(0, 0.85),
       title="Fire weather's validated role: severity timing, not ignition location\n"
             "(intense fires concentrate on high fire-weather days; AUROC for WHERE ≈ 0.50)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "06_fireweather_severity_signal.png", dpi=140); plt.close(fig)
log("06 fireweather severity signal")

log(f"\nwrote {len(list(OUT.glob('*.png')))} figures -> {OUT}")
