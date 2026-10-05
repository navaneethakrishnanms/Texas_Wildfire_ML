"""
v2 (2026-09-23): trains on train_tristate_v2.parquet (corrected national
ecoregion_id for TX -- see 15_train_control_v2.py header). Compares against
control_v2_results.json, not the original control_results.json, since only
the v2-vs-v2 comparison is apples-to-apples on the fixed feature.

NM_OK_TexRev — Step 16: rev5, the tri-state experiment. Identical
config/split/features to script 15's control, but trained on TX+NM+OK
rows instead of TX-only. Test set is STILL TX-only, 2024-2026 (NM/OK
never appear in any test set or metric -- per PRD split design). The
control vs rev5 delta is the honest measure of whether NM/OK data helps.

Output: models/rev5_tristate_v2_seed{42,7,2026}.json, results/rev5_v2_results.json
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import json
from pathlib import Path
import importlib
control = importlib.import_module('15_train_control_v2')

ROOT = Path(__file__).resolve().parent.parent

if __name__ == '__main__':
    control.log("=== REV5 v2: TX+NM+OK training, corrected national ecoregion, TX-only test ===")
    tr, te = control.load_split(['tx', 'nm', 'ok'])
    result = control.train_and_eval(tr, te, 'rev5_tristate_v2')
    with open(ROOT / "results" / "rev5_v2_results.json", 'w') as f:
        json.dump(result, f, indent=2)
    control.log(f"\nSaved results/rev5_v2_results.json -- MEAN AUC-PR={result['mean_aucpr']:.4f}")

    ctrl_path = ROOT / "results" / "control_v2_results.json"
    if ctrl_path.exists():
        ctrl = json.load(open(ctrl_path))
        delta = result['mean_aucpr'] - ctrl['mean_aucpr']
        control.log(f"\n=== CONTROL v2 vs REV5 v2 (corrected ecoregion) ===")
        control.log(f"control_v2 (TX-only):    AUC-PR={ctrl['mean_aucpr']:.4f} (std {ctrl['std_aucpr']:.4f}), n_train={ctrl['n_train']:,}")
        control.log(f"rev5_v2 (TX+NM+OK):      AUC-PR={result['mean_aucpr']:.4f} (std {result['std_aucpr']:.4f}), n_train={result['n_train']:,}")
        control.log(f"DELTA (does NM/OK data help, now that ecoregion is unified?): {delta:+.4f}")
    else:
        control.log("\n(control_v2_results.json not found yet -- run 15_train_control_v2.py first)")
