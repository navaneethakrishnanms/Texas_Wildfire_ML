"""
Live weekly US Drought Monitor pull for TX counties, cached -- USDM only
publishes once/week (Thursdays), so script 13 shouldn't hit the API on
every daily forecast run. Cache is refreshed only if the newest cached
week is >7 days old.

Used by 13_model_forecast_day.py to attach drought_score (0-5 scale,
sum of D0..D4 area-percent /100) per county for the live forecast.

Output: data/static_features/drought_latest.parquet [fips, drought_score]
"""
import io, time, warnings
import pandas as pd
import urllib.request
from pathlib import Path
warnings.filterwarnings('ignore')

# package root = TDIS_Ignition/ (this file is at code/scripts/fetch_live_drought.py)
TF = Path(__file__).resolve().parents[2]
CACHE = TF / "data" / "static" / "drought_latest.parquet"
COUNTY_MAP = TF / "data" / "static" / "tx_county_fips.parquet"


def log(m): print(m, flush=True)


def get_drought_score(force=False):
    """Returns a DataFrame [fips, drought_score] for the current week, refreshing
    the cache only if stale (>7 days) or missing."""
    if CACHE.exists() and not force:
        age_days = (time.time() - CACHE.stat().st_mtime) / 86400
        if age_days < 7:
            log(f"drought cache fresh ({age_days:.1f} days old)")
            return pd.read_parquet(CACHE)
        log(f"drought cache stale ({age_days:.1f} days old) -- refreshing")

    fips_list = pd.read_parquet(COUNTY_MAP)['fips'].unique().tolist()
    today = pd.Timestamp.today()
    start = today - pd.Timedelta(days=14)  # small window, we only need the latest week
    rows = []
    for fips in fips_list:
        url = (f"https://usdmdataservices.unl.edu/api/CountyStatistics/"
               f"GetDroughtSeverityStatisticsByAreaPercent?aoi={fips}&"
               f"startdate={start.month}/{start.day}/{start.year}&"
               f"enddate={today.month}/{today.day}/{today.year}&statisticsType=1")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                txt = r.read().decode()
            if txt.startswith('MapDate'):
                d = pd.read_csv(io.StringIO(txt))
                latest = d.sort_values('ValidStart').iloc[-1]
                score = (float(latest.D0) + float(latest.D1) + float(latest.D2)
                         + float(latest.D3) + float(latest.D4)) / 100.0
                rows.append(dict(fips=fips, drought_score=score))
        except Exception as e:
            log(f"  {fips} failed: {str(e)[:60]}")
        time.sleep(0.2)
    out = pd.DataFrame(rows)
    out.to_parquet(CACHE, index=False)
    log(f"refreshed drought cache: {len(out)} counties, mean score {out.drought_score.mean():.2f}")
    return out


if __name__ == '__main__':
    d = get_drought_score(force=True)
    print(d.describe())
