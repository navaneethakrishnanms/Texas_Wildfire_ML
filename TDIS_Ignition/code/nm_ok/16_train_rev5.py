"""
NM_OK_TexRev — Step 16: rev5, the actual tri-state experiment. Identical
config/split/features to script 15's control, but trained on TX+NM+OK
rows instead of TX-only. Test set is STILL TX-only, 2024-2026 (NM/OK
never appear in any test set or metric -- per PRD split design). The
control vs rev5 delta is the honest measure of whether NM/OK data helps,
isolated from every other change (new negative sample, new features,
new split all already baked into the control).

Output: models/rev5_tristate_seed{42,7,2026}.json, results/rev5_results.json
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import json
from pathlib import Path
import importlib
control = importlib.import_module('15_train_control')

ROOT = Path(__file__).resolve().parent.parent

if __name__ == '__main__':
    control.log("=== REV5: TX+NM+OK training, TX-only test, 28 features ===")
    tr, te = control.load_split(['tx', 'nm', 'ok'])
    result = control.train_and_eval(tr, te, 'rev5_tristate')
    with open(ROOT / "results" / "rev5_results.json", 'w') as f:
        json.dump(result, f, indent=2)
    control.log(f"\nSaved results/rev5_results.json -- MEAN AUC-PR={result['mean_aucpr']:.4f}")

    ctrl_path = ROOT / "results" / "control_results.json"
    if ctrl_path.exists():
        ctrl = json.load(open(ctrl_path))
        delta = result['mean_aucpr'] - ctrl['mean_aucpr']
        control.log(f"\n=== CONTROL vs REV5 ===")
        control.log(f"control (TX-only):    AUC-PR={ctrl['mean_aucpr']:.4f} (std {ctrl['std_aucpr']:.4f}), n_train={ctrl['n_train']:,}")
        control.log(f"rev5 (TX+NM+OK):       AUC-PR={result['mean_aucpr']:.4f} (std {result['std_aucpr']:.4f}), n_train={result['n_train']:,}")
        control.log(f"DELTA (does NM/OK data help?): {delta:+.4f}")
    else:
        control.log("\n(control_results.json not found yet -- run 15_train_control.py to get the comparison)")
