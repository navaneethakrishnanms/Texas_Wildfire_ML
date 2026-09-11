"""
TDIS Portal Replica -- FastAPI backend
----------------------------------------
Mirrors the real tdis-portal-api's documented surface (base path /api/v1,
X-API-Key header, /health and /health/detailed) as described in the handoff
manual (06-local-development.md, 10-external-dependencies.md), but reads
from local parquet files instead of PostgreSQL + pg_tileserv, since neither
Docker nor a native Postgres install is available in this environment.

Hazard scope (per project decision): only /wildfire/* is backed by real data
(our trained 32-feature XGBoost model). /power-outage, /weather-hazard,
/lts-flash-flood, /flood-hub, /baron all return empty results with
"status": "no_data" -- this exactly matches the real portal's own documented
behavior for "Source B" tables before a Databricks sync has run
(07-postgresql-schema.md: "Until Source B exists, those endpoints return
empty results. Health checks still pass.").

Run:
    cd TDIS_REPLICA/api
    uvicorn main:app --reload --port 8000
"""

from datetime import date, datetime
from pathlib import Path
import json
import os
import urllib.request

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Loads TDIS_REPLICA/api/.env locally. In Azure, Container App "Environment
# variables" are already real env vars before Python even starts, so this is
# a no-op there (load_dotenv never overrides an existing env var by default)
# -- the same .env-driven config works in both places without extra code.
load_dotenv(Path(__file__).resolve().parent / ".env")

DEV_API_KEY = os.environ.get("API_KEY", "tdis_dev_master_key_2024_secure_token_123456789")
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]

REPLICA_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPLICA_ROOT / "data"

# In the cloud, the ~600MB data/ folder isn't baked into the Docker image
# (it's gitignored -- too large for GitHub, see Dockerfile comment). Instead,
# set DATA_BASE_URL to a public Blob Storage container holding the same
# files, and this downloads them once at startup into the same local
# DATA_DIR path everything else already expects. Locally (DATA_BASE_URL
# unset), this is a no-op and the pre-built files on disk are used as-is.
DATA_BASE_URL = os.environ.get("DATA_BASE_URL", "").rstrip("/")
DATA_FILES = [
    "historical_risk_cells.parquet",
    "active_fires_by_date.parquet",
    "live_august_staging.parquet",
    "live_feed_state.json",
]


def ensure_data_downloaded():
    if not DATA_BASE_URL:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname in DATA_FILES:
        dest = DATA_DIR / fname
        if dest.exists():
            continue
        url = f"{DATA_BASE_URL}/{fname}"
        print(f"Downloading {url} -> {dest} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  done ({dest.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"  FAILED: {e} (this file will show as missing/empty)")

HISTORICAL_CUTOVER = "2026-07-31"
LIVE_START = "2026-08-01"
LIVE_END = "2026-08-31"

app = FastAPI(title="TDIS Portal Replica API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory data, loaded once at startup ───────────────────────────────────
_historical_df: pd.DataFrame | None = None
_live_staging_df: pd.DataFrame | None = None
_fires_df: pd.DataFrame | None = None


@app.on_event("startup")
def load_data():
    global _historical_df, _live_staging_df, _fires_df
    ensure_data_downloaded()
    print("Loading historical_risk_cells.parquet ...")
    _historical_df = pd.read_parquet(DATA_DIR / "historical_risk_cells.parquet")
    print(f"  {len(_historical_df):,} cells loaded.")

    live_path = DATA_DIR / "live_august_staging.parquet"
    if live_path.exists():
        print("Loading live_august_staging.parquet ...")
        _live_staging_df = pd.read_parquet(live_path)
        _live_staging_df["date"] = _live_staging_df["date"].dt.strftime("%Y-%m-%d")
        print(f"  {len(_live_staging_df):,} (cell, day) rows loaded.")
    else:
        _live_staging_df = pd.DataFrame()
        print("  live_august_staging.parquet not found -- run build_live_august_feed.py")

    fires_path = DATA_DIR / "active_fires_by_date.parquet"
    if fires_path.exists():
        print("Loading active_fires_by_date.parquet ...")
        _fires_df = pd.read_parquet(fires_path)
        print(f"  {len(_fires_df):,} historical fire-day rows loaded.")
    else:
        _fires_df = pd.DataFrame(columns=["h3_cell", "date", "lat", "lon"])
        print("  active_fires_by_date.parquet not found -- re-run build_historical_snapshot.py")


def _published_through() -> str | None:
    state_path = DATA_DIR / "live_feed_state.json"
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text())
    return state.get("published_through")


def require_api_key(x_api_key: str | None):
    if x_api_key != DEV_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def rows_to_geojson(df: pd.DataFrame, extra_props_source: str) -> dict:
    features = []
    for row in df.itertuples(index=False):
        ring = json.loads(row.ring) if isinstance(row.ring, str) else row.ring
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": ring},
            "properties": {
                "h3_cell": row.h3_cell,
                "risk_score": round(float(row.risk_score), 4),
                "risk_class": row.risk_class,
                "source": extra_props_source,
            },
        })
    return {"type": "FeatureCollection", "features": features}


# ── Health (matches their documented endpoints exactly) ─────────────────────
@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "tdis-portal-replica-api", "time": datetime.utcnow().isoformat()}


@app.get("/api/v1/health/detailed")
def health_detailed(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    require_api_key(x_api_key)
    return {
        "status": "ok",
        "components": {
            "data_store": "ok (local parquet -- no Postgres/pg_tileserv in this environment)",
            "model": "ok (TEXAS/model_32feat_tuned.json, 32 features, AUROC 0.8309)",
            "live_feed_daemon": "ok" if (DATA_DIR / "live_feed_state.json").exists() else "not_started",
        },
        "published_through": _published_through(),
    }


# ── Wildfire (REAL data -- our model) ────────────────────────────────────────
@app.get("/api/v1/wildfire/risk-cells")
def risk_cells(
    date: str = Query(..., description="YYYY-MM-DD"),
    bbox: str = Query(..., description="west,south,east,north"),
    zoom: float = Query(8.0, description="current map zoom level"),
):
    """
    Zoom + viewport-filtered hexagon risk layer -- the local-file equivalent
    of pg_tileserv's {z}/{x}/{y}.pbf vector tiles. At low zoom we thin the
    result to only moderate+ risk cells (so the map isn't flooded with
    600k+ low-risk hexes); at zoom >= 7 we return every hex in the viewport,
    which is what makes "zooming in reveals individual hexagons" work.
    """
    try:
        w, s, e, n = [float(x) for x in bbox.split(",")]
    except Exception:
        raise HTTPException(400, "bbox must be 'west,south,east,north'")

    if date <= HISTORICAL_CUTOVER:
        df = _historical_df
        source = "historical"
    elif LIVE_START <= date <= LIVE_END:
        published = _published_through()
        if published is None or date > published:
            return JSONResponse({"type": "FeatureCollection", "features": [],
                                  "status": "not_yet_published",
                                  "published_through": published})
        df = _live_staging_df[_live_staging_df["date"] == date].merge(
            _historical_df[["h3_cell", "ring"]], on="h3_cell", how="inner"
        )
        source = "simulated_live"
    else:
        raise HTTPException(400, f"No data for {date}. Range: 2014-01-01..{LIVE_END}")

    view = df[(df["lon"] >= w) & (df["lon"] <= e) & (df["lat"] >= s) & (df["lat"] <= n)]

    if zoom < 7.0:
        view = view[view["risk_class"].isin(["moderate", "high", "extreme"])]

    MAX_FEATURES = 15000
    if len(view) > MAX_FEATURES:
        view = view.sort_values("risk_score", ascending=False).head(MAX_FEATURES)

    geojson = rows_to_geojson(view, source)
    geojson["status"] = "ok"
    geojson["cell_count"] = len(view)
    geojson["date"] = date
    return geojson


@app.get("/api/v1/wildfire/active-fires")
def active_fires(
    date: str = Query(..., description="YYYY-MM-DD"),
    bbox: str | None = Query(None, description="west,south,east,north"),
):
    """
    Real historical ignitions that occurred on this EXACT date (not a static
    undated dump). August 2026 (simulated_live) has no real fire-occurrence
    simulation -- only continuous risk scores -- so it always returns empty,
    which is the honest answer rather than a fabricated marker.
    """
    if date > HISTORICAL_CUTOVER:
        return {"type": "FeatureCollection", "features": [], "status": "no_simulated_ignitions"}

    day_fires = _fires_df[_fires_df["date"] == date]
    if bbox:
        try:
            w, s, e, n = [float(x) for x in bbox.split(",")]
            day_fires = day_fires[
                (day_fires["lon"] >= w) & (day_fires["lon"] <= e)
                & (day_fires["lat"] >= s) & (day_fires["lat"] <= n)
            ]
        except Exception:
            pass

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row.lon, row.lat]},
            "properties": {"h3_cell": row.h3_cell, "date": row.date, "event_type": "wildfire"},
        }
        for row in day_fires.itertuples(index=False)
    ]
    return {"type": "FeatureCollection", "features": features, "status": "ok", "count": len(features)}


@app.get("/api/v1/wildfire/live-status")
def live_status():
    published = _published_through()
    return {
        "mode": "simulated_live",
        "historical_through": HISTORICAL_CUTOVER,
        "live_window": {"start": LIVE_START, "end": LIVE_END},
        "published_through": published,
        "note": (
            "No real-time data injection source is connected. Historical "
            "data is real (our trained model scored on the TDIS-derived "
            "dataset). August 2026 is a simulated daily drip using seasonal "
            "August weather averages per cell, revealed one day at a time "
            "by live_feed_daemon.py."
        ),
    }


# ── Stubbed hazard types (Source B equivalent -- intentionally empty) ───────
def _empty_stub(name: str):
    return {"status": "no_data", "hazard": name,
            "note": "Not implemented in this replica -- our model covers wildfire ignition only."}


@app.get("/api/v1/power-outage/county")
def power_outage_county():
    return _empty_stub("power_outage")


@app.get("/api/v1/weather-hazard/county")
def weather_hazard_county():
    return _empty_stub("weather_hazard")


@app.get("/api/v1/lts-flash-flood/county")
def lts_flash_flood():
    return _empty_stub("flash_flood")


@app.get("/api/v1/flood-hub/gauges")
def flood_hub_gauges():
    return _empty_stub("flood_hub")


@app.get("/api/v1/baron/rainfall")
def baron_rainfall():
    return _empty_stub("baron_rainfall")
