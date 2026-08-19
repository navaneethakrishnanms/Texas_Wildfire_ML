# Live Forecast Pipeline — Plan

Scoped for: model-focused ownership (this project) + collaboration with the
IgnitionNet team (the Azure-blob group), with infrastructure/portal
deployment as a separate, later-owned piece. This is a plan, not a build —
nothing below is implemented except where marked done.

## 1. Where things stand today

- **Forecasting capability exists and is validated, but is manual.**
  `scripts/13_model_forecast_day.py` pulls a real live HRRR/GFS forecast and
  scores the operational model on it — genuine forecasting, not a hindcast.
  It has to be run by hand for each target date/lead.
- **The model natively predicts at H3 res-8** (1.7M cells); the dashboards
  only ever see a res-5 aggregate (3,624 cells). ✅ *Done this session*: script
  13 now also saves the native res-8 output as parquet
  (`forecast_<date>_lead<N>h_res8.parquet`, ~48MB/snapshot) alongside the
  existing res-5 JSON, so the higher-resolution output exists without
  breaking anything currently wired up.
- **Skill is real but modest** (lift ~2×, AUROC ~0.73) — see the model
  improvement list below; that work is separate from, and should happen
  before, investing further in pipeline automation.

## 2. What "the pipeline" actually needs (not yet built)

1. **Scheduling** — a daily cron job running script 13 for both 24h and 48h
   leads. Needs retry/alerting for the days HRRR isn't available yet
   (script already exits cleanly on a miss — that failure needs to surface
   somewhere, not just log silently).
2. **Dashboard data-loading architecture change** — the *standalone*
   dashboards embed forecast JSON at build time (`17_embed_standalone.py`/
   `21_embed_explorer.py` bake in "the newest two" forecast files). A live
   dashboard needs the **served** version fetching fresh data over HTTP at
   load time, not a rebuilt static file. At res-8 scale (1.7M cells,
   ~48MB/snapshot as parquet, ~137MB if ever done as JSON) this can't be one
   browser-loaded blob — it needs the v2 explorer's existing tiling
   approach (renders ≤6,000 visible res-8 hexes at a time) fed by a real
   query path (a lightweight API or pre-tiled static files), not a single
   fetch.
3. **Portal deployment** (`portal.cloud.tdis.io`) — **blocked on information
   this project doesn't have**: the portal's actual ingestion mechanism
   (API? SFTP? object storage bucket? admin upload?), auth, and whether it's
   ready to receive automated pushes today. This needs to come from whoever
   owns that portal before any push step can be built — guessing at an
   integration here would just produce broken code.

**Recommended order**: fix model skill (§3) → build scheduling + res-8
serving locally, prove it end-to-end on the NAS → get portal ingestion
details → wire up the final push step. Don't build the portal connector
first against unknowns.

## 3. Model improvements to do before/alongside pipeline work

(Same list as discussed — repeated here so the plan is self-contained.)

| Priority | Change | Why | Cost |
|---|---|---|---|
| 1 | Add `hrrr_gust` as a real model feature (currently only feeds the FWI formula, not the ignition model) | Directly targets the diagnosed wind-driven-fire weakness (Smokehouse Creek, Lavender, Windy Deuce) | Retrain only, data already pulled |
| 2 | Add rolling multi-day weather trend features (e.g. 5-day trailing ERC/VPD/wind) | IgnitionNet's own feature importance shows this matters; TDIS has none of it | New feature engineering, forecastable from existing lead-time data |
| 3 | Check for a "burnable"/fuel-load equivalent feature | IgnitionNet's single best feature (19.8% of gain); TDIS may already have the raw LANDFIRE data to derive it | Data audit first, cheap if available |
| 4 | Calibrate operational model output (isotonic, proven fix) | Doesn't change AUROC/lift, but the raw number isn't currently trustworthy as a real probability | Post-hoc, no retrain |
| 5 | Dedicated hyperparameter search for operational-HRRR | IgnitionNet's own 21-trial search only gained ~0.005 AUROC — cheap but low expected payoff | Cheap, temper expectations |

**Do not** target the ceiling model's 3.88× lift as a forecast-skill
benchmark — it uses same-day observed weather, which is unknowable at
forecast time by definition. Its lesson (richer weather features help)
is real; the model itself is not deployable. Items 1-2 above are how that
lesson gets applied to something that actually can be forecast.

## 4. IgnitionNet collaboration — open threads worth raising with them

- Their own docs conflict on Texas's H3 resolution (res-7 vs res-8) — worth
  a quick check on their end (`check_resolution.py` exists for exactly this,
  just needs to be run against their current file).
- They don't appear to have a spatial-holdout / label-shuffle credibility
  check — more important for them than for us, since they use raw
  `centroid_lat`/`centroid_lon` as direct model features.
- Their entire pipeline is a hindcast/backtest system (archived HRRR,
  historical risk maps) — no live-forecast-pulling script found. If
  forecasting becomes a shared goal, TDIS's script 13 (pull live HRRR →
  score model) is the piece to share/adapt, not rebuild from scratch.
- Both teams independently found HRRR sub-daily data added little over
  daily gridMET+trailing-stats — worth confirming this is a genuinely
  shared, cross-validated finding before either team invests more there.
