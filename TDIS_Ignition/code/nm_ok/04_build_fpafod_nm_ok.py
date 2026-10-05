"""
NM_OK_TexRev — Step 4: grid FPA-FOD points to H3-8 for NM + OK, 2014-2020
(the augmented national CSVs already on disk at 30%_Rev2/data/ignition/,
same source TX's fpa_fod_tx_h3.parquet came from). Filters by STATE,
converts lat/lon -> h3_cell, keeps the columns the label-fusion step needs.

Output: NM_OK_TexRev/data/fpa_fod_{nm,ok}_h3.parquet
        [h3_cell, date, cause_class, max_size_acres]
"""
import glob
import pandas as pd, h3
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TF = ROOT.parent
PAR = TF.parent
RAW_GLOB = str(PAR / "30%_Rev2" / "data" / "ignition" / "*_FPA_FOD_cons.csv")
H3_RES = 8


def log(m): print(m, flush=True)


def build_for_state(state, state_abbr):
    out = ROOT / "data" / f"fpa_fod_{state}_h3.parquet"
    if out.exists():
        log(f"[skip] {out.name} exists"); return

    frames = []
    for f in sorted(glob.glob(RAW_GLOB)):
        cols = ['STATE', 'LATITUDE', 'LONGITUDE', 'DISCOVERY_DATE',
                'NWCG_CAUSE_CLASSIFICATION', 'FIRE_SIZE']
        df = pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False)
        df = df[df.STATE == state_abbr]
        if len(df):
            frames.append(df)
    if not frames:
        log(f"[{state}] no rows found across {RAW_GLOB}"); return
    d = pd.concat(frames, ignore_index=True)
    d['date'] = pd.to_datetime(d['DISCOVERY_DATE'], errors='coerce').dt.normalize()
    d = d.dropna(subset=['date', 'LATITUDE', 'LONGITUDE'])
    d['h3_cell'] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(d.LATITUDE, d.LONGITUDE)]
    d = d.rename(columns={'NWCG_CAUSE_CLASSIFICATION': 'cause_class', 'FIRE_SIZE': 'max_size_acres'})
    out_df = d[['h3_cell', 'date', 'cause_class', 'max_size_acres']].copy()
    out_df.to_parquet(out, index=False)
    log(f"[{state}] {len(out_df):,} FPA-FOD fires, {out_df.date.dt.year.min():.0f}-"
        f"{out_df.date.dt.year.max():.0f} -> {out.name}")


if __name__ == '__main__':
    build_for_state('nm', 'NM')
    build_for_state('ok', 'OK')
