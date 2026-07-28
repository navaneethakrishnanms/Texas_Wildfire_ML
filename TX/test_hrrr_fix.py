import pandas as pd, sys
sys.path.insert(0, '.')
from fetch_hrrr_tx import extract_one, load_centroids

centroids, _ = load_centroids()

print("\nTest 1: 2015-03-15 (rh expected NaN)")
r = extract_one(pd.Timestamp('2015-03-15'), 12, centroids)
if r is not None:
    print(f"  OK  rows={len(r):,}  temp={r.temp_pw.mean():.1f}C  rh_nan={r.rh_pw.isna().sum():,}  wind={r.wind_pw.mean():.1f}m/s")
else:
    print("  FAIL — still not working")

print("\nTest 2: 2017-04-10 (all variables expected)")
r2 = extract_one(pd.Timestamp('2017-04-10'), 12, centroids)
if r2 is not None:
    print(f"  OK  rows={len(r2):,}  temp={r2.temp_pw.mean():.1f}C  rh={r2.rh_pw.mean():.1f}%  wind={r2.wind_pw.mean():.1f}m/s")
else:
    print("  FAIL")

print("\nTest 3: 2018-07-04 (known good)")
r3 = extract_one(pd.Timestamp('2018-07-04'), 12, centroids)
if r3 is not None:
    print(f"  OK  rows={len(r3):,}  temp={r3.temp_pw.mean():.1f}C  rh={r3.rh_pw.mean():.1f}%  wind={r3.wind_pw.mean():.1f}m/s")
else:
    print("  FAIL")
